"""可学习的路由决策器（Learned Router）。

用一个小型 MLP 替代手写规则选择器 `_select_online`，从「诊断特征 + 各分支特征」
预测每个分支的「预期正确概率」（expected ground_truth_score），选最高的分支。

设计动机：
  - 规则选择器的阈值（delta_conf/tau_uncert/kappa/clip_max）是人工设定的，
    需要大量验证集调参，且是分段线性启发式，无法捕捉特征间复杂交互；
  - 可学习 router 直接以「分支是否真的分类正确」（ground_truth_score，0/1）
    作为训练信号，等价于学一个「验证器」，其决策目标就是最小化 Regret。

特征（固定 21 维，保证可训练、可部署）：
  - 诊断特征（6）：noise_score, lowlight_score, brightness_mean/255,
    log1p(noise_var), dark_pixel_ratio, clip_ratio
  - 每个分支（raw / denoise / lowlight，共 3 个，各 5 维）：
    present(0/1), downstream_confidence, margin, clip_ratio, fallback(0/1)

输出：3 个分支的预测正确概率（sigmoid），固定顺序 raw / denoise / lowlight。
推理时只在实际存在的候选分支中选择最高分（不存在的分支概率强制 -inf）。
"""
from __future__ import annotations

from typing import List, Optional, Tuple

import numpy as np

try:
    import torch
    import torch.nn as nn
    _HAS_TORCH = True
except ImportError:  # 无 torch 时降级为纯 numpy 前向（仅用于测试/占位）
    _HAS_TORCH = False

from .tool_registry import BRANCH_ORDER

# 分支顺序（与训练标签一致）：raw（""）+ 全部修复工具，从 tool_registry 导入
# （单一来源，避免与 orchestrator 的候选分组不一致）。
N_BRANCHES = len(BRANCH_ORDER)  # 分支数（raw + 12 工具 = 13）
N_DIAG_FEATURES = 6  # 诊断特征数（noise_score/lowlight_score/brightness_mean/log1p(noise_var)/dark_pixel_ratio/clip_ratio）
FEATURE_DIM = N_DIAG_FEATURES + N_BRANCHES * 5  # 6 + 13*5 = 71
NO_DIAG_FEATURE_DIM = N_BRANCHES * 5  # 13*5 = 65（诊断特征消融）

# clip_ratio 过曝判定的统一容差：clip_ratio 若被存成 float32，0.1 可能变成
# 0.10000000149，导致「clip_ratio == clip_max」边界在训练验证与线上判断不一致。
# 所有「clip_ratio 是否过曝」的判断都统一走此容差口径（含规则分支与 LLM 复核）。
CLIP_EPS = 1e-6


def is_overexposed(clip_ratio: float, clip_max: float) -> bool:
    """判断 clip_ratio 是否超过过曝阈值（统一容差口径）。

    全项目所有「clip_ratio 过曝」判断必须复用此函数，保证规则 Router、
    可学习 Router（safe_argmax）、LLM 复核三处口径一致。
    """
    return float(clip_ratio) > float(clip_max) + CLIP_EPS


def extract_features(diagnosis, outcomes: list,
                     visible_upstreams: Optional[set] = None,
                     include_diag: bool = True) -> np.ndarray:
    """从诊断结果 + 分支结果提取固定特征向量（默认 21 维）。

    diagnosis: Diagnosis 对象（含 noise_score/lowlight_score/brightness_mean/
               noise_var/dark_pixel_ratio/clip_ratio）
    outcomes:  PipelineOutcome 列表（可能只含部分分支，缺失分支用 0 填充）
    visible_upstreams: 推理时「实际可见」的分支 upstream 集合。训练时传入规则
               候选集合，使 present 掩码与真实推理一致（规则裁剪掉的分支即使
               被 global_oracle 补跑，也标记为不可见）。None 时用 outcomes 自身。
    include_diag: 是否把 6 个全局诊断值直接输入 learned selector。False 时仅返回
                  15 维分支特征。候选准入规则与 present 掩码仍保留诊断门控信息。
    """
    d = diagnosis.to_dict() if hasattr(diagnosis, "to_dict") else (diagnosis or {})

    if include_diag:
        diag_feats = np.array([
            float(d.get("noise_score", 0.0)),
            float(d.get("lowlight_score", 0.0)),
            float(d.get("brightness_mean", 0.0)) / 255.0,
            float(np.log1p(float(d.get("noise_var", 0.0)))),      # log 压缩动态范围
            float(d.get("dark_pixel_ratio", 0.0)),
            float(d.get("clip_ratio", 0.0)),
        ], dtype=np.float32)
    else:
        diag_feats = np.array([], dtype=np.float32)

    # 按分支名索引 outcome
    by_upstream = {o.upstream: o for o in outcomes}
    if visible_upstreams is None:
        visible_upstreams = {o.upstream for o in outcomes}

    branch_feats = []
    for up in BRANCH_ORDER:
        o = by_upstream.get(up)
        if o is None or up not in visible_upstreams:
            branch_feats.extend([0.0, 0.0, 0.0, 0.0, 0.0])
        else:
            branch_feats.extend([
                1.0,                                      # present
                float(getattr(o, "downstream_confidence", 0.0)),
                float(getattr(o, "margin", 0.0)),
                float(getattr(o, "clip_ratio", 0.0)),
                1.0 if getattr(o, "fallback", False) else 0.0,
            ])

    feats = np.concatenate([diag_feats, np.array(branch_feats, dtype=np.float32)])
    expected = FEATURE_DIM if include_diag else NO_DIAG_FEATURE_DIM
    assert feats.shape[0] == expected, f"特征维度错误 {feats.shape[0]} != {expected}"
    return feats


def safe_argmax(probs, present, conf, clip, fallback,
                clip_max: float = 0.10, conf_drop: float = 0.15,
                abstain_threshold: float = 0.0):
    """公共安全选择函数：从各分支概率中选最优，复用线上全部安全约束。

    供「训练验证指标」与「orchestrator 线上推理」共用，避免两处逻辑不一致。

    参数（numpy 数组，形状 (N, 3) 或 (3,)）：
      probs:   各分支的预测概率
      present: 分支是否存在（0/1）
      conf:    分支分类置信度（downstream_confidence）
      clip:    分支 clip_ratio（过曝伪影代理）
      fallback:分支是否降级（0/1）
    返回：安全选择的分支 index（int 或 ndarray）。
      约束依次为：排除不存在/fallback/过曝分支 → 置信度保护（选中分支置信度
      不得比 raw 低超过 conf_drop）→ abstain（最高概率低于 abstain_threshold）。
      触发置信度保护或 abstain 时回退 raw（index 0）。
    """
    probs = np.asarray(probs, dtype=np.float64)
    present = np.asarray(present, dtype=np.float64)
    conf = np.asarray(conf, dtype=np.float64)
    clip = np.asarray(clip, dtype=np.float64)
    fallback = np.asarray(fallback, dtype=np.float64)

    # 统一容差口径（CLIP_EPS）：避免 clip_ratio 存成 float32 时 0.1 变 0.10000000149，
    # 导致「clip_ratio == clip_max」的浮点边界在训练验证与线上判断不一致。
    valid = (present > 0.5) & (fallback < 0.5) & (clip <= clip_max + CLIP_EPS)
    masked = np.where(valid, probs, -np.inf)
    idx = masked.argmax(axis=-1)

    # 置信度保护：选中分支 conf < raw_conf - conf_drop → 回退 raw
    raw_conf = conf[..., 0]
    raw_visible = present[..., 0] > 0.5
    chosen_conf = np.take_along_axis(conf, idx[..., None], axis=-1)[..., 0]
    drop = raw_visible & (chosen_conf < raw_conf - conf_drop)
    idx = np.where(drop, 0, idx)

    # abstain：最高概率低于阈值 → 回退 raw
    if abstain_threshold > 0.0:
        max_prob = masked.max(axis=-1)
        abstain = max_prob < abstain_threshold
        idx = np.where(abstain, 0, idx)

    return idx


class LearnedRouter:
    """可学习 router 的封装：加载权重 + 前向打分 + 选择分支。"""

    def __init__(self, weight_path: Optional[str] = None, device: str = "cpu"):
        self.device = device
        self.model = None
        self.weight_path = weight_path
        self.feature_dim = FEATURE_DIM  # 默认 21 维；load() 后按 checkpoint 覆盖
        if weight_path:
            self.load(weight_path)

    def load(self, weight_path: str) -> None:
        if not _HAS_TORCH:
            raise RuntimeError("缺少 torch，无法加载 router 模型")
        from ._router_net import RouterNet
        ckpt = torch.load(weight_path, map_location=self.device, weights_only=False)
        if not isinstance(ckpt, dict):
            raise ValueError("router checkpoint 格式错误（应为含 state_dict 的 dict）")
        # 校验元数据：特征维度与分支顺序不一致时直接报错（避免旧权重输出含义错位）
        fd = ckpt.get("feature_dim", FEATURE_DIM)
        if fd is not None and int(fd) not in (FEATURE_DIM, NO_DIAG_FEATURE_DIM):
            raise ValueError(
                f"router feature_dim 不匹配：{fd} 不在 {[FEATURE_DIM, NO_DIAG_FEATURE_DIM]}，"
                f"请用当前特征重新训练"
            )
        self.feature_dim = int(fd) if fd is not None else FEATURE_DIM
        bo = ckpt.get("branch_order")
        if bo is not None and list(bo) != list(BRANCH_ORDER):
            raise ValueError(
                f"router branch_order 不匹配：{bo} != {BRANCH_ORDER}"
            )
        sd = ckpt.get("state_dict", ckpt)
        self.model = RouterNet(input_dim=self.feature_dim,
                               num_branches=len(BRANCH_ORDER)).to(self.device)
        self.model.load_state_dict(sd)
        self.model.eval()

    @property
    def available(self) -> bool:
        return self.model is not None

    def score(self, diagnosis, outcomes: list) -> Tuple[np.ndarray, np.ndarray]:
        """返回 (三分支概率, 三分支存在掩码)。"""
        feats = extract_features(diagnosis, outcomes,
                                 include_diag=(self.feature_dim == FEATURE_DIM))
        present = np.array([
            1.0 if any(o.upstream == up for o in outcomes) else 0.0
            for up in BRANCH_ORDER
        ], dtype=np.float32)

        if self.model is None or not _HAS_TORCH:
            # 未加载模型：退化为「置信度最高的分支」，实际不应走到这里
            probs = np.zeros(len(BRANCH_ORDER), dtype=np.float32)
            for i, up in enumerate(BRANCH_ORDER):
                o = next((x for x in outcomes if x.upstream == up), None)
                probs[i] = float(getattr(o, "downstream_confidence", 0.0)) if o else 0.0
            return probs, present

        with torch.no_grad():
            x = torch.from_numpy(feats).unsqueeze(0).to(self.device)
            logits = self.model(x)  # (1, 3)
            # BCE 训练：每个分支独立的「正确概率」，用 sigmoid（分支非互斥，
            # 多个分支可同时正确）；选概率最高者。
            probs = torch.sigmoid(logits).squeeze(0).cpu().numpy()
        return probs, present

    def select(self, diagnosis, outcomes: list, clip_max: float = 0.10,
               conf_drop: float = 0.15, abstain_threshold: float = 0.0):
        """选择最优分支，返回 (pipeline_id 或 None, 各分支概率)。

        复用公共安全选择函数 safe_argmax，统一训练验证与线上推理的约束：
        排除不存在/fallback/过曝分支 → 置信度保护 → abstain（回退 raw）。
        """
        probs, present = self.score(diagnosis, outcomes)
        by_up = {o.upstream: o for o in outcomes}
        conf = np.array(
            [getattr(by_up.get(up), "downstream_confidence", 0.0) for up in BRANCH_ORDER],
            dtype=np.float64)
        clip = np.array(
            [getattr(by_up.get(up), "clip_ratio", 0.0) for up in BRANCH_ORDER],
            dtype=np.float64)
        fb = np.array(
            [1.0 if getattr(by_up.get(up), "fallback", False) else 0.0 for up in BRANCH_ORDER],
            dtype=np.float64)
        idx = int(safe_argmax(probs, present, conf, clip, fb,
                              clip_max=clip_max, conf_drop=conf_drop,
                              abstain_threshold=abstain_threshold))
        up = BRANCH_ORDER[idx]
        o = by_up.get(up)
        return (o.pipeline_id if o is not None else None), probs
