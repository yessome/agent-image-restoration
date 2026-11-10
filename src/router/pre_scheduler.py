"""运行前调度器（Pre-execution Scheduler，v1：13 分支）。

与 `learned_router.py` 的本质区别
--------------------------------
`learned_router` 的每个分支特征包含 `downstream_confidence / margin / clip_ratio /
fallback`——这些只有「该分支已经修复、分类器已经前向」之后才存在，因此它结构上
只能是「先全部执行、再挑一个」的事后选择器。

本模块**只允许使用原图诊断得到的 6 维特征**，在调用任何修复器之前即可决策；
选定后只执行 1 个分支（选 raw 时一次修复都不调用）。

特征（6 维，全部来自 `ImageDiagnoser.diagnose(image)`，顺序固定）
  noise_score, lowlight_score, brightness_mean/255,
  log1p(noise_var), dark_pixel_ratio, clip_ratio

分支范围（v1 = 13 个）
  raw + 8 个去噪 + 4 个低光。
  不含去模糊（restormer_defocus / mprnet_deblur）与超分（realesrgan_x2 / x4）：
  现有训练数据没有对应退化类型，等补数据和诊断特征后再扩展。

标签语义（离线生成，允许跑全部工具）
  对训练图用冻结模型跑遍 13 个分支，得到每个分支「分类是否正确」。
  `raw_priority=True`（默认）时，若 raw 已判对，则其余分支标签置 0
  —— 即「多个分支都正确时优先 raw」，避免学到无谓的修复开销。
"""
from __future__ import annotations

from typing import Dict, List, Optional, Tuple

import numpy as np

try:
    import torch
    _HAS_TORCH = True
except ImportError:  # pragma: no cover - 无 torch 时降级为规则调度
    _HAS_TORCH = False

from .tool_registry import BRANCH_ORDER as ALL_BRANCH_ORDER
from .tool_registry import TOOL_GROUPS

# v1 纳入运行前调度的退化组
PRE_GROUPS: Tuple[str, ...] = ("noise", "lowlight")

# 分支顺序：raw（""）在前，其余按组展开
PRE_BRANCH_ORDER: List[str] = [""] + [up for g in PRE_GROUPS for up in TOOL_GROUPS[g]]

# 6 维诊断特征
N_DIAG_FEATURES = 6
PRE_FEATURE_DIM = N_DIAG_FEATURES
FEATURE_NAMES: List[str] = [
    "noise_score", "lowlight_score", "brightness_mean/255",
    "log1p(noise_var)", "dark_pixel_ratio", "clip_ratio",
]

# 在全部 17 分支的正确性向量里，本模块 13 个分支对应的下标（按名称解析，避免写死）
PRE_SOURCE_INDICES: List[int] = [ALL_BRANCH_ORDER.index(up) for up in PRE_BRANCH_ORDER]


def extract_pre_features(diagnosis) -> np.ndarray:
    """从诊断结果提取 6 维**运行前**特征（与 learned_router 的前 6 维口径一致）。"""
    d = diagnosis.to_dict() if hasattr(diagnosis, "to_dict") else (diagnosis or {})
    feats = np.array([
        float(d.get("noise_score", 0.0)),
        float(d.get("lowlight_score", 0.0)),
        float(d.get("brightness_mean", 0.0)) / 255.0,
        float(np.log1p(float(d.get("noise_var", 0.0)))),
        float(d.get("dark_pixel_ratio", 0.0)),
        float(d.get("clip_ratio", 0.0)),
    ], dtype=np.float32)
    assert feats.shape[0] == PRE_FEATURE_DIM
    return feats


def build_pre_labels(y_row, raw_priority: bool = True) -> np.ndarray:
    """把 17 分支的正确性向量转成 13 分支训练标签。

    y_row: 长度 = len(ALL_BRANCH_ORDER) 的 0/1 向量（按全部工具顺序）
    raw_priority: True 时，raw 正确就把其余分支标签清零（优先 raw）
    """
    y = np.asarray(y_row, dtype=np.float32).reshape(-1)
    if y.shape[0] != len(ALL_BRANCH_ORDER):
        raise ValueError(
            f"标签维度错误：{y.shape[0]} != {len(ALL_BRANCH_ORDER)}（应为全部工具的分支数）")
    lab = np.array([y[i] for i in PRE_SOURCE_INDICES], dtype=np.float32)
    if raw_priority and lab[0] > 0.5:
        lab[1:] = 0.0
    return lab


def build_pre_target(y_row) -> int:
    """决策式单标签目标（方案 A）——返回目标分支下标（0 = raw）。

    规则（与用户约定一致，不含任何阈值）：
      1. raw 判对            → raw（0）；此时不切换，避免无谓开销
      2. raw 判错、有工具判对 → 在「判对的工具」中取 PRE_BRANCH_ORDER 里最靠前的一个
                              （固定偏好，不引入额外超参；denoise 排在工具首位）
      3. 所有分支都判错       → raw（0）

    与 `build_pre_labels`（多标签 BCE）的区别：
      BCE 的每个工具标签语义是「工具对 且 raw 错」，其量级上界是 P(raw 错)，
      直接与 P(raw 对) 比大小会结构性塌缩为永远 raw；单标签 softmax 直接对
      「该选谁」建模，是决策一致的表述。
    """
    y = np.asarray(y_row, dtype=np.float32).reshape(-1)
    if y.shape[0] != len(ALL_BRANCH_ORDER):
        raise ValueError(
            f"标签维度错误：{y.shape[0]} != {len(ALL_BRANCH_ORDER)}（应为全部工具的分支数）")
    lab = np.array([y[i] for i in PRE_SOURCE_INDICES], dtype=np.float32)
    if lab[0] > 0.5:
        return 0
    for j in range(1, len(lab)):
        if lab[j] > 0.5:
            return j
    return 0


def rule_select(diagnosis, confidence_threshold: float = 0.6) -> Tuple[str, str]:
    """规则单分支基线：只选 1 个分支（运行前）。

    阈值复用 config/pipelines.yaml 的 `confidence_threshold`，本模块不另设阈值。

    明确低光且低光分高于噪声分 → lowlight（Zero-DCE）
    明确噪声且噪声分不低于低光分 → denoise（DnCNN）
    其余 → raw

    注意：这里只选 1 个，不做「多候选并行」——保证与运行前调度同样的调用次数，
    延迟口径才可比。
    """
    d = diagnosis.to_dict() if hasattr(diagnosis, "to_dict") else (diagnosis or {})
    noise = float(d.get("noise_score", 0.0))
    low = float(d.get("lowlight_score", 0.0))
    th = float(confidence_threshold)

    if low >= th and low > noise:
        return "lowlight", f"rule_pre:lowlight({low:.2f})≥{th}且>noise({noise:.2f})"
    if noise >= th and noise >= low:
        return "denoise", f"rule_pre:noise({noise:.2f})≥{th}且≥lowlight({low:.2f})"
    return "", f"rule_pre:max(noise={noise:.2f},lowlight={low:.2f})<{th}→raw"


def validate_checkpoint(ckpt: Dict) -> None:
    """校验 checkpoint 与本模块的分支/特征定义一致；不一致直接抛错。"""
    fd = ckpt.get("feature_dim")
    if fd is not None and int(fd) != PRE_FEATURE_DIM:
        raise ValueError(
            f"pre_scheduler feature_dim 不匹配：{fd} != {PRE_FEATURE_DIM}（本调度器只用 6 维诊断特征）")
    bo = ckpt.get("branch_order")
    if bo is None:
        raise ValueError("pre_scheduler checkpoint 缺少 branch_order，无法确认分支对齐")
    if list(bo) != list(PRE_BRANCH_ORDER):
        raise ValueError(
            f"pre_scheduler branch_order 不匹配：{bo} != {PRE_BRANCH_ORDER}")
    act = ckpt.get("output_activation", "sigmoid")
    if act not in ("sigmoid", "softmax"):
        raise ValueError(f"未知 output_activation：{act}（应为 sigmoid 或 softmax）")


class PreScheduler:
    """运行前调度器封装：加载权重 + 一次前向 + 选 1 个分支（含规则回退）。"""

    def __init__(self, weight_path: Optional[str] = None, device: str = "cpu"):
        self.device = device
        self.weight_path = weight_path
        self.model = None
        self.output_activation = "sigmoid"  # 由 checkpoint 覆盖：sigmoid(多标签) / softmax(单标签)
        self._load_error: Optional[str] = None
        if weight_path:
            self.load(weight_path)

    def load(self, weight_path: str) -> None:
        if not _HAS_TORCH:
            raise RuntimeError("缺少 torch，无法加载 pre_scheduler 模型")
        from ._router_net import RouterNet

        ckpt = torch.load(weight_path, map_location=self.device, weights_only=False)
        if not isinstance(ckpt, dict):
            raise ValueError("pre_scheduler checkpoint 格式错误（应为 dict）")
        validate_checkpoint(ckpt)
        sd = ckpt.get("state_dict", ckpt)
        self.model = RouterNet(input_dim=PRE_FEATURE_DIM,
                               num_branches=len(PRE_BRANCH_ORDER)).to(self.device)
        self.model.load_state_dict(sd)
        self.model.eval()
        self.output_activation = ckpt.get("output_activation", "sigmoid")
        self.weight_path = weight_path

    @property
    def available(self) -> bool:
        return self.model is not None

    def score(self, diagnosis) -> np.ndarray:
        """返回 13 个分支的「预测正确概率」（sigmoid 多标签 / softmax 单标签）。"""
        feats = extract_pre_features(diagnosis)
        if self.model is None or not _HAS_TORCH:
            raise RuntimeError("pre_scheduler 未加载，无法打分")
        with torch.no_grad():
            x = torch.from_numpy(feats).unsqueeze(0).to(self.device)
            logits = self.model(x)
            if self.output_activation == "softmax":
                probs = torch.softmax(logits, dim=-1).squeeze(0).cpu().numpy()
            else:
                probs = torch.sigmoid(logits).squeeze(0).cpu().numpy()
        return probs

    def select(self, diagnosis, confidence_threshold: float = 0.6
               ) -> Tuple[str, str, List[float]]:
        """选 1 个分支，返回 (upstream, reason, probs)。

        upstream 为 "" 表示 raw（不调用任何修复器）。
        平局时取 raw：np.argmax 返回第一个最大值，而 raw 位于 index 0。
        回退规则时使用的阈值由调用方传入（Agent 复用 confidence_threshold）。
        """
        if not self.available:
            upstream, reason = rule_select(diagnosis, confidence_threshold)
            err = f"（{self._load_error}）" if self._load_error else ""
            return upstream, f"fallback_rule{err}:" + reason.split(":", 1)[-1], []

        probs = self.score(diagnosis)
        idx = int(np.argmax(probs))
        upstream = PRE_BRANCH_ORDER[idx]
        name = upstream or "raw"
        reason = (f"pre_scheduler:{name} p={probs[idx]:.3f} "
                  f"(raw={probs[0]:.3f}, 2nd={np.sort(probs)[-2]:.3f})")
        return upstream, reason, [float(p) for p in probs]

    def topk(self, diagnosis, k: int = 3) -> List[Tuple[str, float]]:
        """调试用：返回概率最高的 k 个分支（不改变选择逻辑）。"""
        if not self.available:
            return []
        probs = self.score(diagnosis)
        order = np.argsort(probs)[::-1][:k]
        return [(PRE_BRANCH_ORDER[i] or "raw", float(probs[i])) for i in order]
