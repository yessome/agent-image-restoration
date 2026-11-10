"""诊断图像、调度修复分支并返回分类结果。"""
from __future__ import annotations

import importlib
import os
import sys
from dataclasses import dataclass, field
from typing import Any, Dict, List, Optional

import cv2
import numpy as np
import time

from ..router.diagnose import ImageDiagnoser, Diagnosis
from ..router.router import PipelineRoute
from ..router.llm_planner import LLMPlanner
from ..router.learned_router import LearnedRouter, is_overexposed
from ..router.tool_registry import TOOL_GROUPS as _TOOL_GROUPS, ALL_UPSTREAM_TOOLS
from ..router.pre_scheduler import PreScheduler, rule_select, PRE_BRANCH_ORDER
from ..router.iterative_router import IterativeRouter

_UPSTREAM_REGISTRY = {
    "denoise": ("src.upstream.dncnn", "DnCNNProcessor"),
    "lowlight": ("src.upstream.lowlight", "LowLightProcessor"),
    "zero_dce_plus_plus": ("src.upstream.lowlight", "ZeroDCEPlusPlusProcessor"),
    "sci_lowlight": ("src.upstream.lowlight", "SCIProcessor"),
    "retinexformer": ("src.upstream.retinexformer", "RetinexformerProcessor"),
    "ffdnet": ("src.upstream.deepinv_denoise", "FFDNetProcessor"),
    "scunet": ("src.upstream.deepinv_denoise", "SCUNetProcessor"),
    "restormer_real": ("src.upstream.restormer", "RestormerProcessor"),
    "restormer_blind": ("src.upstream.restormer", "RestormerProcessor"),
    "restormer_sigma15": ("src.upstream.restormer", "RestormerProcessor"),
    "restormer_sigma25": ("src.upstream.restormer", "RestormerProcessor"),
    "restormer_sigma50": ("src.upstream.restormer", "RestormerProcessor"),
    "restormer_defocus": ("src.upstream.restormer", "RestormerProcessor"),
    "mprnet_deblur": ("src.upstream.mprnet", "MPRNetDeblurProcessor"),
    "realesrgan_x2": ("src.upstream.realesrgan", "RealESRGANProcessor"),
    "realesrgan_x4": ("src.upstream.realesrgan", "RealESRGANProcessor"),
}
_DOWNSTREAM_REGISTRY = {
    "classify": ("src.downstream.classify", "ClassifyProcessor"),
}

EXPERIMENT_UPSTREAM_TOOLS = ["denoise", "lowlight"]


@dataclass
class PipelineOutcome:
    """单条流水线的执行结果。"""
    pipeline_id: str
    upstream: str                      # 上游名（"" 表示原图直通）
    downstream: str
    score: float = 0.0                 # 代理分数（在线验证用）
    downstream_confidence: float = 0.0 # 下游置信度
    margin: float = 0.0                # Top-1/Top-2 概率间隔（不确定度代理）
    pred_label: str = ""               # 预测类别（分类任务）
    task_result: Any = None
    restored_image: Optional[np.ndarray] = None
    fallback: bool = False             # 是否降级（无权重）
    clip_ratio: float = 0.0            # 修复后图像的高光裁剪比例（过曝伪影代理）
    ground_truth_score: float = 0.0    # 基于真实标签的任务得分（离线）
    is_correct: bool = False           # 分类是否正确（离线）
    labels: Optional[List[str]] = None   # 类别顺序（logits/probs 第 i 维对应 labels[i]）
    logits: Optional[List[float]] = None # 原始 logits（维度 = len(labels)）
    probs: Optional[List[float]] = None  # softmax 概率（维度 = len(labels)）
    gt_logit: Optional[float] = None     # 真实类别那一维的 logit（离线，需 ground_truth）
    gt_prob: Optional[float] = None      # 真实类别那一维的 softmax 概率（离线）
    upstream_latency_ms: float = 0.0    # 上游修复耗时（含 CUDA 同步；raw 分支为 0）
    downstream_latency_ms: float = 0.0  # 下游分类耗时（含 CUDA 同步）
    peak_memory_mb: float = 0.0         # 该分支执行后的 CUDA 峰值显存（MB，累计值）


@dataclass
class AgentResult:
    """Agent 的最终输出（含验证闭环结果）。"""
    chosen_pipeline: str
    diagnosis: Dict[str, Any] = field(default_factory=dict)
    outcomes: List[PipelineOutcome] = field(default_factory=list)      # 路由候选（诊断裁剪后）
    all_outcomes: List[PipelineOutcome] = field(default_factory=list)  # 含事后评估补跑的分支
    decision: str = ""                 # "修复采纳" / "回退原图" / "全部降级"
    explanation: str = ""
    oracle_pipeline: str = ""          # 路由后候选集内的真值最优分支（routed Oracle）
    hit: bool = False                  # Agent 选择是否命中 routed Oracle
    regret: float = 0.0                # routed Oracle 得分 - Agent 得分
    global_oracle_pipeline: str = ""   # 全局评估范围内的真值最优分支
    global_regret: float = 0.0         # global Oracle 得分 - Agent 在线选择得分
    vision_review: Optional[Dict[str, Any]] = None   # 模型复核原始意见（JSON）
    vision_applied: bool = False        # 复核建议通过本地校验并被采纳
    vision_triggered: bool = False      # 是否满足触发条件并尝试调用
    vision_succeeded: bool = False      # 视觉请求是否成功返回合法 JSON
    vision_error: str = ""              # 失败原因（空 = 无错误）
    vision_reject_reason: str = ""      # 建议被本地安全校验拒绝的原因（空 = 通过）
    vision_latency_ms: float = 0.0      # 视觉复核耗时（毫秒）
    # 训练时用同一候选集构建准入掩码，空字符串表示原图。
    routed_upstreams: List[str] = field(default_factory=list)
    routing_mode: str = "legacy"
    selected_upstream: str = ""          # 运行前选中的 upstream（"" = raw）
    select_reason: str = ""              # 选择原因（调度器概率 / 规则判据 / 回退原因）
    scheduler_probs: List[float] = field(default_factory=list)  # pre 模式的分支概率
    restore_calls: int = 0               # 本次实际调用的修复模型次数（pre 模式应为 0 或 1）
    routing_trace: List[Dict[str, Any]] = field(default_factory=list)
    oracle_scope: str = ""  # iterative: executed_only, or admitted_candidates with oracle audit

    def to_dict(self) -> Dict[str, Any]:
        def _serialize(o: PipelineOutcome) -> Dict[str, Any]:
            return {
                "pipeline_id": o.pipeline_id,
                "upstream": o.upstream,
                "score": o.score,
                "downstream_confidence": o.downstream_confidence,
                "margin": o.margin,
                "pred_label": o.pred_label,
                "ground_truth_score": o.ground_truth_score,
                "is_correct": o.is_correct,
                "fallback": o.fallback,
                "clip_ratio": o.clip_ratio,
                "labels": o.labels,
                "logits": o.logits,
                "probs": o.probs,
                "gt_logit": o.gt_logit,
                "gt_prob": o.gt_prob,
                "upstream_latency_ms": o.upstream_latency_ms,
                "downstream_latency_ms": o.downstream_latency_ms,
                "peak_memory_mb": o.peak_memory_mb,
            }

        return {
            "chosen_pipeline": self.chosen_pipeline,
            "diagnosis": self.diagnosis,
            "outcomes": [_serialize(o) for o in self.outcomes],
            "all_outcomes": [_serialize(o) for o in self.all_outcomes],
            "decision": self.decision,
            "oracle_pipeline": self.oracle_pipeline,
            "hit": self.hit,
            "regret": self.regret,
            "global_oracle_pipeline": self.global_oracle_pipeline,
            "global_regret": self.global_regret,
            "vision_review": self.vision_review,
            "vision_applied": self.vision_applied,
            "vision_triggered": self.vision_triggered,
            "vision_succeeded": self.vision_succeeded,
            "vision_error": self.vision_error,
            "vision_reject_reason": self.vision_reject_reason,
            "vision_latency_ms": self.vision_latency_ms,
            "explanation": self.explanation,
            "routing_mode": self.routing_mode,
            "routed_upstreams": self.routed_upstreams,
            "selected_upstream": self.selected_upstream,
            "select_reason": self.select_reason,
            "scheduler_probs": self.scheduler_probs,
            "restore_calls": self.restore_calls,
            "routing_trace": self.routing_trace,
            "oracle_scope": self.oracle_scope,
        }


class ImageRestorationAgent:
    """垃圾图像修复 Agent 编排器（v2：执行后验证闭环）。"""

    def __init__(self, config: Optional[Dict] = None, device: str = "cuda"):
        config = dict(config or {})
        config["upstreams"] = dict(config.get("upstreams") or {})
        config["router"] = dict(config.get("router") or {})
        # Load optional tool definitions for both CLI and direct Python callers.
        # The frozen main pipeline config remains the source of its two original tools.
        root = os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
        toolbox_path = os.path.join(root, "config", "pretrained_tools.yaml")
        if os.path.isfile(toolbox_path):
            try:
                import yaml
                with open(toolbox_path, encoding="utf-8") as handle:
                    toolbox = yaml.safe_load(handle) or {}
                for name, tool_cfg in (toolbox.get("upstreams") or {}).items():
                    merged = dict(tool_cfg)
                    weight_path = merged.get("weight_path")
                    if isinstance(weight_path, str) and weight_path and not os.path.isabs(weight_path):
                        merged["weight_path"] = os.path.join(root, weight_path)
                    config["upstreams"].setdefault(name, merged)
            except Exception as exc:
                print(f"[警告] 模型工具配置加载失败：{type(exc).__name__}: {exc}", file=sys.stderr)
        self.config = config
        self.device = device
        self._toolbox_enabled = bool((config.get("model_toolbox") or {}).get("enabled", False))
        # Exclusions apply to candidate generation and post-decision audits alike.
        excluded = (config.get("model_toolbox") or {}).get("exclude", [])
        if (not isinstance(excluded, (list, tuple))
                or any(not isinstance(up, str) for up in excluded)):
            raise ValueError("model_toolbox.exclude 必须是工具名称列表")
        self._excluded_tools = set(excluded)
        unknown_tools = self._excluded_tools - set(ALL_UPSTREAM_TOOLS)
        if unknown_tools:
            raise ValueError(f"未知排除工具：{sorted(unknown_tools)}")
        # Optional candidate availability, independent of the noise/lowlight
        # diagnosis. Admission does not execute a tool or select its output.
        admission_groups = (config.get("routing") or {}).get("always_admit_groups", [])
        if not isinstance(admission_groups, (list, tuple)) or any(
                not isinstance(group, str) for group in admission_groups):
            raise ValueError("routing.always_admit_groups 必须是工具组名称列表")
        unknown_groups = set(admission_groups) - set(_TOOL_GROUPS)
        if unknown_groups:
            raise ValueError(f"未知候选工具组：{sorted(unknown_groups)}；可用：{list(_TOOL_GROUPS)}")
        if admission_groups and not self._toolbox_enabled:
            raise ValueError("always_admit_groups 要求 model_toolbox.enabled=true")
        self._always_admit_groups = tuple(dict.fromkeys(admission_groups))
        self.diagnoser = ImageDiagnoser(config.get("diagnosis"))
        self.llm = LLMPlanner(config.get("llm"))
        self.confidence_threshold = config.get("confidence_threshold", 0.6)
        self._select_cfg = config.get("selector", {})  # 在线选择器阈值
        self._upstream_cache: Dict[str, Any] = {}
        self._downstream_cache: Dict[str, Any] = {}
        # 可学习 router（可选）：有权重时替代规则决策。
        # 加载失败必须可见（打印警告），避免「以为在测 learned、实际在测 rule」。
        self._learned_router: Optional[LearnedRouter] = None
        router_cfg = config["router"]
        # LearnedRouter.load 检查 checkpoint 的特征维度和分支顺序。
        self._router_cfg = router_cfg
        # 运行前调度不使用 learned router（它的特征依赖全分支执行结果），
        # 因此不加载、也不打印「权重不匹配」告警，避免误导为配置错误。
        _mode_cfg = str((config.get("routing") or {}).get("mode", "legacy"))
        learned_wp = router_cfg.get("learned_weight") if _mode_cfg == "legacy" else None
        if learned_wp:
            if not os.path.exists(learned_wp):
                print(f"[警告] learned router 权重不存在：{learned_wp}，回退规则决策",
                      file=sys.stderr)
            else:
                try:
                    self._learned_router = LearnedRouter(learned_wp, device)
                except Exception as e:
                    print(f"[警告] learned router 加载失败（{type(e).__name__}: {e}），"
                          f"回退规则决策", file=sys.stderr)
                    self._learned_router = None
        self._current_diagnosis: Optional[Diagnosis] = None

        # 缺少 routing.mode 的旧配置沿用 legacy。
        routing_cfg = dict(config.get("routing") or {})
        self.routing_mode = str(routing_cfg.get("mode", "legacy"))
        if self.routing_mode not in ("legacy", "pre", "rule_pre", "iterative"):
            raise ValueError(f"未知调度模式：{self.routing_mode}")
        self._iterative_router = None
        if self.routing_mode == "iterative":
            iterative_cfg = config.get("iterative_router") or {}
            wp = iterative_cfg.get("weight_path", "")
            if wp and not os.path.isabs(wp):
                wp = os.path.join(root, wp)
            if not wp or not os.path.isfile(wp):
                raise ValueError("循环 learned router 权重不存在；请先运行 train_iterative_router.py，不会回退规则 router")
            cls_wp = ((config.get("downstreams") or {}).get("classify") or {}).get("weight_path")
            if cls_wp and not os.path.isabs(cls_wp):
                cls_wp = os.path.join(root, cls_wp)
            if not cls_wp or not os.path.isfile(cls_wp):
                raise ValueError("循环 learned router 要求指定冻结分类器权重以校验训练配对")
            self._iterative_router = IterativeRouter(wp, device, classifier_weight=cls_wp)
            expected_order = [""] + [up for up in ALL_UPSTREAM_TOOLS
                                      if up not in self._excluded_tools]
            if list(self._iterative_router.branch_order) != expected_order:
                raise ValueError("循环 router 分支表与 model_toolbox.exclude 不一致；不会回退规则 router")
        pre_cfg = dict(config.get("pre_scheduler") or {})
        self._pre_scheduler: Optional[PreScheduler] = None
        if self.routing_mode in ("pre", "rule_pre"):
            wp = pre_cfg.get("weight_path") or ""
            if wp and not os.path.isabs(wp):
                wp = os.path.join(root, wp)
            if self.routing_mode == "pre":
                if not wp:
                    print("[警告] routing_mode=pre 但未配置 pre_scheduler.weight_path；回退规则调度",
                          file=sys.stderr)
                elif not os.path.exists(wp):
                    print(f"[警告] pre_scheduler 权重不存在：{wp}；回退规则调度", file=sys.stderr)
                else:
                    sched = PreScheduler(wp, device)
                    if sched.available:
                        self._pre_scheduler = sched
                    else:
                        print(f"[警告] pre_scheduler 加载失败（{sched._load_error}）；回退规则调度",
                              file=sys.stderr)

    def run(
        self,
        image: np.ndarray,
        intent: str = "classify",
        task_kwargs: Optional[Dict] = None,
        ground_truth: Optional[Any] = None,
        global_oracle: bool = False,
        force_all_candidates: bool = False,
    ) -> AgentResult:
        """主入口：诊断 → 候选方案 → 执行 → 验证 → 决策。

        参数:
            image: 输入图像
            intent: 下游意图（仅 classify，分割已移除）
            task_kwargs: 下游任务额外参数
            ground_truth: 真实标签（离线实验模式传入，用于计算
                          真值得分 + Oracle + 命中率 + Regret；
                          在线推理阶段为 None，用代理分数选择）
            global_oracle: 是否额外强制执行全部分支（raw + 全部修复器）来计算
                          真正的理论上界（global Oracle）。开启时不影响 Agent 的
                          实际选择（仍用诊断裁剪后的候选 + 在线规则），只额外计算
                          global_oracle_pipeline / global_regret。
            force_all_candidates: 候选裁剪消融（no_candidate_pruning）用。
                          默认固定为原始三分支（raw+denoise+lowlight）；仅显式
                          启用 model_toolbox 时扩展到全部修复工具。
        """
        task_kwargs = task_kwargs or {}

        # 先加载权重再检查后端。TrashNet 离线指标要求六类分类器。
        if ground_truth is not None and intent == "classify":
            downstream = self._get_downstream("classify")
            downstream._assert_loaded()  # 触发 load()（若未加载）
            backend = getattr(downstream, "_backend", "")
            if backend != "trashnet":
                raise RuntimeError(
                    f"离线六分类实验要求加载 TrashNet 六类权重，但当前分类器后端为 "
                    f"'{backend or '未加载'}'。请先微调 ResNet-50 分类头并将权重填入 "
                    f"config/pipelines.yaml 的 downstreams.classify.weight_path，"
                    f"否则 Oracle/命中率/Regret 不具备实验意义。"
                )

        diagnosis = self.diagnoser.diagnose(image)
        self._current_diagnosis = diagnosis  # 供 learned router 提取特征

        if self.routing_mode == "iterative":
            return self._run_iterative(image, diagnosis, task_kwargs, ground_truth,
                                       global_oracle, force_all_candidates)

        # 单步基线按原图诊断选一个分支，跳过全局 Oracle 补跑。
        if self.routing_mode in ("pre", "rule_pre"):
            return self._run_pre_scheduled(image, diagnosis, intent, task_kwargs, ground_truth)

        if force_all_candidates:
            candidates = self._gen_all_candidates(intent)
        else:
            candidates = self._gen_candidates(diagnosis, intent)
        routed_upstreams = [c.upstream for c in candidates]

        outcomes: List[PipelineOutcome] = []
        for cand in candidates:
            outcome = self._execute_pipeline(image, cand, task_kwargs, ground_truth)
            outcomes.append(outcome)

        # 全局 Oracle 补跑未准入的分支，仅用于离线评估。
        global_outcomes: List[PipelineOutcome] = []
        if global_oracle and ground_truth is not None and intent == "classify":
            seen = {o.upstream for o in outcomes}
            oracle_tools = ([u for u in ALL_UPSTREAM_TOOLS if u not in self._excluded_tools]
                            if self._toolbox_enabled else EXPERIMENT_UPSTREAM_TOOLS)
            for up in [""] + oracle_tools:
                if up in seen:
                    continue
                route = PipelineRoute(f"global_{up or 'raw'}_cls", up, "classify",
                                      reason="全局 Oracle 分支")
                global_outcomes.append(
                    self._execute_pipeline(image, route, task_kwargs, ground_truth)
                )

        # Oracle 使用已执行和补跑的全部结果。没有补跑时沿用 outcomes。
        all_outcomes = outcomes + global_outcomes

        chosen, oracle, hit, regret = self._decide(outcomes, ground_truth)

        # 视觉复核只用于在线不确定样本。复核建议仍需通过本地筛选。
        vision_review = None
        vision_applied = False
        vision_triggered = False
        vision_succeeded = False
        vision_error = ""
        vision_reject_reason = ""
        vision_latency_ms = 0.0
        if ground_truth is None and getattr(self.llm, "vision_available", False):
            vision_triggered = self._needs_vision_review(outcomes, diagnosis)
            if vision_triggered:
                raw_out = next((o for o in outcomes if o.upstream == ""), None)
                t0 = time.time()
                vision_review = self.llm.review_candidates(image, outcomes, diagnosis)
                vision_latency_ms = (time.time() - t0) * 1000
                vision_succeeded = vision_review is not None
                if not vision_succeeded:
                    vision_error = getattr(self.llm, "_last_error", "") or "视觉复核请求失败"
                else:
                    reviewed, vision_reject_reason = self.llm.validate_review(
                        vision_review, outcomes, chosen=chosen, raw=raw_out,
                        clip_max=self._cfg("clip_max", 0.10),
                        tau_uncert=self._cfg("tau_uncert", 0.10),
                        kappa=self._cfg("kappa", 1.5),
                    )
                    if reviewed is not None and reviewed.pipeline_id != chosen.pipeline_id:
                        chosen = reviewed
                        vision_applied = True

        # 全局 Oracle：在全部候选（outcomes + 补跑）中取理论上界
        global_oracle_pipeline = ""
        global_regret = 0.0
        if global_oracle and ground_truth is not None:
            g_oracle = self._accuracy_oracle(all_outcomes, ground_truth)
            if g_oracle is not None:
                global_oracle_pipeline = g_oracle.pipeline_id
                global_regret = g_oracle.ground_truth_score - chosen.ground_truth_score

        explanation = self.llm.explain(diagnosis, chosen, outcomes,
                                       vision_review=vision_review, force_local=True)

        return AgentResult(
            chosen_pipeline=chosen.pipeline_id,
            diagnosis=diagnosis.to_dict(),
            outcomes=outcomes,
            all_outcomes=all_outcomes,
            decision=self._decision_label(chosen),
            explanation=explanation,
            oracle_pipeline=oracle,
            hit=hit,
            regret=regret,
            global_oracle_pipeline=global_oracle_pipeline,
            global_regret=global_regret,
            vision_review=vision_review,
            vision_applied=vision_applied,
            vision_triggered=vision_triggered,
            vision_succeeded=vision_succeeded,
            vision_error=vision_error,
            vision_reject_reason=vision_reject_reason,
            vision_latency_ms=vision_latency_ms,
            routed_upstreams=routed_upstreams,
        )

    # ---------- Learned feedback loop ----------

    def _run_iterative(self, image, diagnosis, task_kwargs, ground_truth,
                       global_oracle=False, force_all_candidates=False):
        router = self._iterative_router
        if router is None:
            raise RuntimeError("循环模式缺少 learned router；禁止静默改用规则")
        routes = (self._gen_all_candidates() if force_all_candidates
                  else self._gen_candidates(diagnosis))
        by_up = {r.upstream: r for r in routes}
        outcomes = [self._execute_pipeline(image, by_up[""], task_kwargs, ground_truth)]
        trace = []
        # Each call consumes one distinct candidate; the budget guarantees termination.
        for step in range(router.budget + 1):
            action = router.decide(diagnosis, outcomes, set(by_up))
            event = {"step": step, "action": action.kind, "upstream": action.upstream,
                     "reason": action.reason, "probabilities": action.probabilities,
                     "observed": [o.upstream for o in outcomes]}
            trace.append(event)
            observed = {o.upstream: o for o in outcomes}
            if action.kind == "stop":
                if action.upstream not in observed:
                    raise RuntimeError("Router attempted to return an unexecuted branch")
                chosen = observed[action.upstream]
                break
            if (action.kind != "call" or not action.upstream
                    or action.upstream not in by_up or action.upstream in observed
                    or step >= router.budget):
                raise RuntimeError("Invalid/repeated/budget-exceeding learned action")
            # Restore the ORIGINAL image; never compose unrelated tool outputs.
            outcome = self._execute_pipeline(image, by_up[action.upstream], task_kwargs, ground_truth)
            outcomes.append(outcome)
            event["feedback"] = {"pred_label": outcome.pred_label,
                                 "confidence": outcome.downstream_confidence,
                                 "margin": outcome.margin, "clip_ratio": outcome.clip_ratio,
                                 "fallback": outcome.fallback}
        else:
            raise RuntimeError("Router did not stop within budget")

        # Oracle auditing occurs AFTER the final decision; never feeds the policy.
        all_outcomes = list(outcomes)
        oracle_outcomes = outcomes
        scope = "executed_only"
        global_pipeline, global_regret = "", 0.0
        if global_oracle and ground_truth is not None:
            seen = {o.upstream for o in outcomes}
            tools = ([u for u in ALL_UPSTREAM_TOOLS if u not in self._excluded_tools]
                            if self._toolbox_enabled else EXPERIMENT_UPSTREAM_TOOLS)
            for up in [""] + tools:
                if up not in seen:
                    route = PipelineRoute(f"global_{up or 'raw'}_cls", up, "classify",
                                          reason="post-decision oracle audit")
                    all_outcomes.append(self._execute_pipeline(image, route, task_kwargs, ground_truth))
            oracle_outcomes = [o for o in all_outcomes if o.upstream in by_up]
            scope = "admitted_candidates"
            oracle = self._accuracy_oracle(all_outcomes, ground_truth)
            if oracle:
                global_pipeline = oracle.pipeline_id
                global_regret = oracle.ground_truth_score - chosen.ground_truth_score
        oracle = self._accuracy_oracle(oracle_outcomes, ground_truth)
        return AgentResult(
            chosen_pipeline=chosen.pipeline_id, diagnosis=diagnosis.to_dict(),
            outcomes=outcomes, all_outcomes=all_outcomes,
            decision=self._decision_label(chosen),
            explanation=f"learned feedback loop: {len(outcomes)-1} tool calls; "
                        f"{trace[-1]['reason']}; return {chosen.upstream or 'raw'}",
            oracle_pipeline=oracle.pipeline_id if oracle else "",
            hit=bool(oracle and chosen.ground_truth_score >= oracle.ground_truth_score),
            regret=oracle.ground_truth_score-chosen.ground_truth_score if oracle else 0.0,
            global_oracle_pipeline=global_pipeline, global_regret=global_regret,
            routed_upstreams=list(by_up), routing_mode="iterative",
            selected_upstream=chosen.upstream, select_reason=trace[-1]["reason"],
            scheduler_probs=trace[-1]["probabilities"], restore_calls=len(outcomes)-1,
            routing_trace=trace, oracle_scope=scope,
        )

    # ---------- 运行前调度执行路径 ----------

    def _run_pre_scheduled(
        self,
        image: np.ndarray,
        diagnosis: Diagnosis,
        intent: str,
        task_kwargs: Dict,
        ground_truth: Optional[Any],
    ) -> "AgentResult":
        """运行前调度：诊断 → 选 1 个分支 → 只执行 1 次 → 返回。

        与 legacy 的关键差异（也是本模式的全部意义）：
          - 不生成候选列表，不并行/串行执行多个修复器；
          - 不做 global Oracle 补跑，因此 oracle_pipeline / global_regret 在该模式下为空，
            评测侧若要上界需在**离线**另行统计（不能混进本模式的延迟对比）；
          - restore_calls 记录实际修复调用次数（选 raw 时为 0），可直接验证
            「没有先跑多个模型再选」。
        """
        mode_used = "rule_pre"
        probs: List[float] = []

        if self.routing_mode == "pre" and self._pre_scheduler is not None:
            upstream, reason, probs = self._pre_scheduler.select(
                diagnosis, confidence_threshold=self.confidence_threshold)
            mode_used = "pre"
        else:
            upstream, reason = rule_select(diagnosis, self.confidence_threshold)
            if self.routing_mode == "pre":
                reason = "fallback→" + reason

        route = PipelineRoute(f"pre_{upstream or 'raw'}_cls", upstream, "classify",
                              reason=reason)
        outcome = self._execute_pipeline(image, route, task_kwargs, ground_truth)
        outcomes = [outcome]
        explanation = self.llm.explain(diagnosis, outcome, outcomes, force_local=True)

        return AgentResult(
            chosen_pipeline=outcome.pipeline_id,
            diagnosis=diagnosis.to_dict(),
            outcomes=outcomes,
            all_outcomes=list(outcomes),
            decision=self._decision_label(outcome),
            explanation=explanation,
            routed_upstreams=[upstream],
            routing_mode=mode_used,
            selected_upstream=upstream,
            select_reason=reason,
            scheduler_probs=probs,
            restore_calls=1 if upstream else 0,
        )

    # ---------- 视觉复核触发判断 ----------

    def _needs_vision_review(self, outcomes, diagnosis) -> bool:
        """判断是否属于「规则难以裁决」的不确定场景（触发视觉复核）。

        四个触发条件（满足其一即复核；普通图不调用 API，省成本且可复现）：
        1. raw 预测不确定（margin 低于 tau_uncert）；
        2. 修复分支与 raw 预测类别冲突（选择器面临切类抉择）；
        3. 诊断歧义（低光与噪声同时触发，路由方向不明）；
        4. 最高分修复候选 clip_ratio 接近伪影上限（0.08~0.10 灰带）。
        """
        raw = next((o for o in outcomes if o.upstream == ""), None)
        if raw is None:
            return False
        rest = [o for o in outcomes if o.upstream != "" and not o.fallback]
        # 无任何有效修复分支可比（raw-only），视觉复核无意义，直接不触发
        if not rest:
            return False

        if getattr(self.llm, "trigger_uncertain", True):
            if raw.margin < self._cfg("tau_uncert", 0.10):
                return True
        if getattr(self.llm, "trigger_conflict", True) and rest:
            if any(o.pred_label != raw.pred_label for o in rest):
                return True
        if getattr(self.llm, "trigger_ambiguous", True):
            d = diagnosis.to_dict()
            if d.get("noise_high") and d.get("lowlight"):
                return True
        if rest and max(o.clip_ratio for o in rest) > 0.08:
            return True
        return False

    # ---------- 候选方案生成 ----------

    def _gen_candidates(self, diagnosis: Diagnosis, intent: str = "classify") -> List[PipelineRoute]:
        """生成候选流水线列表。

        routing.always_admit_groups 显式列出的组在诊断裁剪前准入，
        使缺少对应诊断信号的工具也有被 learned router 考虑的机会。
        默认空列表，保留已冻结实验的候选集合与顺序。

        原图直通分支始终在列。修复分支按诊断分数独立判定（噪声、低光各自
        决定是否加入，互不遮蔽，以正确覆盖「噪声+低光」双退化图）：
        - 明确正常（噪声/低光分数都低）：只原图直通
        - 明确退化（某类分数 >= confidence_threshold）：原图 + 对应修复器
        - 真正不确定（分数在中间灰区）：原图 + 全部修复器（探索-验证）

        注意：噪声与低光是两个独立维度，高噪声图也可能同时低光，因此二者
        分别判定、可同时加入候选（此前「命中一类即 return」会漏掉另一退化分支）。
        """
        down = "classify"
        suffix = "cls"

        candidates: List[PipelineRoute] = []

        # 原图直通（始终参与竞争）
        candidates.append(PipelineRoute(f"raw_{suffix}", "", down, reason="原图直通基线"))

        admitted = {""}

        def admit(upstream, reason):
            if upstream not in admitted and upstream not in self._excluded_tools:
                candidates.append(PipelineRoute(
                    f"{upstream}_{suffix}", upstream, down, reason=reason))
                admitted.add(upstream)

        # These tools remain available even when diagnosis has no corresponding
        # signal (e.g. deblur). The iterative router still decides call/stop.
        for group in self._always_admit_groups:
            for up in _TOOL_GROUPS[group]:
                admit(up, f"配置准入工具组：{group}；执行与否由调度器决定")

        # 低分阈值（低于此值视为「明确无此退化」）
        low_th = 0.4
        noise_s = diagnosis.noise_score
        lowlight_s = diagnosis.lowlight_score

        # No noise/lowlight signal: retain raw and explicitly admitted groups.
        # This does not establish that the image is free of other degradations.
        if noise_s < low_th and lowlight_s < low_th:
            return candidates

        # 噪声、低光各自独立判定是否加入对应修复分支（支持双退化并存，
        # 避免此前「命中一类即 return」漏掉另一退化分支）。
        noise_high = noise_s >= self.confidence_threshold
        lowlight_high = lowlight_s >= self.confidence_threshold

        if self._toolbox_enabled:
            tool_groups = {k: [u for u in v if u not in self._excluded_tools]
                           for k, v in _TOOL_GROUPS.items()}
        else:
            tool_groups = {"noise": ["denoise"], "lowlight": ["lowlight"]}
        if noise_high:
            for up in tool_groups["noise"]:
                admit(up, "检测到高噪声")
        if lowlight_high:
            for up in tool_groups["lowlight"]:
                admit(up, "检测到低光照")

        # 真正不确定（灰区，两类都未达到明确退化阈值）：并行探索全部修复器
        # （含去模糊/超分等无诊断信号的工具，交由验证闭环判断是否有益）
        if not noise_high and not lowlight_high:
            active_tools = ([u for u in ALL_UPSTREAM_TOOLS if u not in self._excluded_tools]
                            if self._toolbox_enabled else EXPERIMENT_UPSTREAM_TOOLS)
            for up in active_tools:
                admit(up, "低置信度并行探索")
        return candidates

    def _gen_all_candidates(self, intent: str = "classify") -> List[PipelineRoute]:
        """候选裁剪消融（no_candidate_pruning）用：固定全部分支，跳过诊断裁剪。

        默认保持论文实验的三分支口径；完整模型工具箱只在显式启用时展开。
        注意：这**不是**「去掉诊断特征」的消融——diagnosis 仍会被计算并喂给
        learned router（router 的 6 维诊断特征仍在）。它只去掉「诊断裁剪候选」
        这一步，让候选固定为 raw + 当前模式中的修复工具，用于回答
        「诊断器的候选裁剪是否有价值」。
        """
        down = "classify"
        routes = [PipelineRoute("raw_cls", "", down, reason="候选裁剪消融")]
        active_tools = ([u for u in ALL_UPSTREAM_TOOLS if u not in self._excluded_tools]
                        if self._toolbox_enabled else EXPERIMENT_UPSTREAM_TOOLS)
        routes += [
            PipelineRoute(f"{up}_cls", up, down, reason="候选裁剪消融")
            for up in active_tools
        ]
        return routes

    # ---------- 执行单条流水线 ----------

    def _execute_pipeline(
        self, image: np.ndarray, route: PipelineRoute, task_kwargs: Dict,
        ground_truth: Optional[Any] = None,
    ) -> PipelineOutcome:
        """执行一条流水线：上游修复（可选）+ 下游任务。

        若有真实标签（离线模式），计算真值得分（分类是否正确 / 交叉熵）。
        记录真实成本：上游/下游各自耗时（含 CUDA 同步）+ 峰值显存。
        """
        import time as _time

        def _sync_cuda():
            try:
                import torch as _torch
                if _torch.cuda.is_available():
                    _torch.cuda.synchronize()
            except Exception:
                pass

        restored = None
        fallback = False
        upstream_latency = 0.0
        downstream_latency = 0.0

        # 上游修复
        if route.upstream:
            _sync_cuda()
            t0 = _time.perf_counter()
            upstream = self._get_upstream(route.upstream)
            result = upstream.restore(image)
            _sync_cuda()
            upstream_latency = (_time.perf_counter() - t0) * 1000.0
            restored = result.image
            fallback = fallback or result.meta.get("fallback", False)
            task_input = restored
        else:
            task_input = image

        # 下游任务
        _sync_cuda()
        t0 = _time.perf_counter()
        downstream = self._get_downstream(route.downstream)
        task_result = downstream.process(task_input, **task_kwargs)
        _sync_cuda()
        downstream_latency = (_time.perf_counter() - t0) * 1000.0
        fallback = fallback or task_result.meta.get("fallback", False)

        # 峰值显存（该分支执行后的累计峰值，MB）
        peak_memory = 0.0
        try:
            import torch as _torch
            if _torch.cuda.is_available():
                peak_memory = _torch.cuda.max_memory_allocated() / (1024.0 ** 2)
        except Exception:
            pass

        # 在线代理分数（下游置信度，仅作参考）
        confidence = self._extract_confidence(task_result)
        margin = self._extract_margin(task_result)
        pred_label = self._extract_pred_label(task_result)

        # 离线真值验证（若有标签）
        gt_score, is_correct = self._ground_truth_score(task_result, ground_truth)

        # 提取完整 logits/probs 与类别顺序，并定位「真实类别那一维」的 logit/概率
        logits, probs, labels, gt_logit, gt_prob = self._extract_logits_meta(
            task_result, ground_truth)

        # 修复后图像的重曝伪影代理（clip_ratio，V>clip_threshold 像素占比）
        clip_ratio = self._clip_ratio(task_input) if route.upstream else 0.0

        return PipelineOutcome(
            pipeline_id=route.pipeline_id,
            upstream=route.upstream,
            downstream=route.downstream,
            score=confidence,
            downstream_confidence=confidence,
            margin=margin,
            pred_label=pred_label,
            task_result=task_result,
            restored_image=restored,
            fallback=fallback,
            clip_ratio=clip_ratio,
            ground_truth_score=gt_score,
            is_correct=is_correct,
            labels=labels,
            logits=logits,
            probs=probs,
            gt_logit=gt_logit,
            gt_prob=gt_prob,
            upstream_latency_ms=upstream_latency,
            downstream_latency_ms=downstream_latency,
            peak_memory_mb=peak_memory,
        )

    @staticmethod
    def _clip_ratio(image: np.ndarray, threshold: float = 250.0) -> float:
        """计算图像高光裁剪比例（过曝伪影代理）。

        与 diagnoser 的 clip_ratio 定义一致：V 通道 > threshold 的像素占比。
        """
        if image is None:
            return 0.0
        hsv = cv2.cvtColor(image, cv2.COLOR_RGB2HSV)
        v = hsv[..., 2].astype(np.float32)
        return float(np.mean(v > threshold))

    def _ground_truth_score(self, task_result, ground_truth) -> tuple:
        """基于真实标签计算任务得分（离线实验用）。

        分类任务：预测是否正确（1.0）或错误（0.0）。
        返回 (得分, 是否正确)。
        """
        if ground_truth is None or task_result is None:
            return 0.0, False

        if task_result.task_type == "classify":
            result = task_result.result
            if isinstance(result, dict) and result:
                pred = max(result, key=result.get)  # 预测类别
                correct = (pred == ground_truth)
                return (1.0 if correct else 0.0), correct

        return 0.0, False

    @staticmethod
    def _extract_logits_meta(task_result, ground_truth):
        """从 TaskResult.meta 提取完整 logits/probs 与类别顺序，并定位真实类别那一维。

        返回 (logits, probs, labels, gt_logit, gt_prob)。无 logits 落盘（如 ImageNet 后端）
        时返回 (None, None, None, None, None)。gt 不在 labels 中时 gt_logit/gt_prob 为 None。
        """
        if task_result is None:
            return None, None, None, None, None
        meta = getattr(task_result, "meta", {}) or {}
        logits = meta.get("logits")
        probs = meta.get("probs")
        labels = meta.get("labels")
        if not logits or not labels:
            return None, None, None, None, None

        gt_logit = None
        gt_prob = None
        if ground_truth is not None and ground_truth in labels:
            idx = labels.index(ground_truth)
            gt_logit = float(logits[idx]) if idx < len(logits) else None
            gt_prob = float(probs[idx]) if (probs and idx < len(probs)) else None
        return list(logits), list(probs) if probs else None, list(labels), gt_logit, gt_prob

    @staticmethod
    def _extract_confidence(task_result) -> float:
        """提取下游置信度（Top-1 概率）作为在线代理分数之一。

        注意：置信度 ≠ 正确率，此分数仅用于在线启发式选择，
        其可靠性需在验证集上证明（与真值最优分支的一致性）。
        """
        if task_result is None:
            return 0.0
        meta = getattr(task_result, "meta", {}) or {}
        if meta.get("fallback"):
            return 0.0
        # 分类：取最高类别概率
        if task_result.task_type == "classify":
            result = task_result.result
            if isinstance(result, dict) and result:
                return float(max(result.values()))
        return 0.0

    @staticmethod
    def _extract_margin(task_result) -> float:
        """提取 Top-1 与 Top-2 概率间隔（分类任务的「不确定度」代理）。

        margin 小 → 预测不确定（Top-1 与 Top-2 接近）；
        margin 大 → 预测更确定。
        用于「是否允许类别切换」的判断。
        """
        if task_result is None:
            return 0.0
        if task_result.task_type != "classify":
            return 0.0
        result = task_result.result
        if not isinstance(result, dict) or len(result) < 2:
            return 0.0
        values = sorted(result.values(), reverse=True)
        top1, top2 = values[0], values[1]
        return float(top1 - top2)

    @staticmethod
    def _extract_pred_label(task_result) -> str:
        """提取预测类别标签（分类任务）。"""
        if task_result is None or task_result.task_type != "classify":
            return ""
        result = task_result.result
        if isinstance(result, dict) and result:
            return str(max(result, key=result.get))
        return ""

    # ---------- 验证 + 决策 ----------

    @staticmethod
    def _accuracy_oracle(outcomes: List[PipelineOutcome], ground_truth) -> Optional[PipelineOutcome]:
        """在给定候选集中计算 Accuracy Oracle（理论上界）。

        优先「预测正确」的分支；若都错，退化为真值得分最高（通常全 0）。
        这是「候选集内」的上界；全局上界需传入全部分支的结果。
        """
        if ground_truth is None or not outcomes:
            return None
        corrects = [o for o in outcomes if o.is_correct]
        pool = corrects if corrects else outcomes
        return max(pool, key=lambda o: o.ground_truth_score)

    def _decide(self, outcomes: List[PipelineOutcome], ground_truth=None):
        """比较候选方案，选最优，并计算 Oracle/命中率/Regret。

        两种模式：
        - 离线（有 ground_truth）：计算 Accuracy Oracle（路由后候选集内的理论上界：
          优先「预测正确」的分支，其次按真值得分最高）；在线选择器用代理规则
          （允许审慎类别切换）选，计算命中率（得分相等即命中，不依赖分支 ID）
          和 Regret。全局上界由 run() 的 global_oracle 模式另行计算。
        - 在线（无 ground_truth）：只用代理规则选。

        返回 (chosen, routed_oracle_id, hit, regret)。
        """
        if not outcomes:
            raise ValueError("无候选方案")

        raw = next((o for o in outcomes if o.upstream == ""), None)

        # 离线模式：routed Oracle = 诊断裁剪后候选集内的理论上界
        acc_oracle = self._accuracy_oracle(outcomes, ground_truth)

        # 在线选择器：代理规则（含审慎类别切换 + 回退）
        chosen = self._select_online(outcomes, raw)

        # 命中率 / Regret（用 routed Oracle 作为上界）
        # 命中判定：得分相等（而非分支 ID 相等），避免「两分支均正确却报未命中」。
        hit = False
        regret = 0.0
        if acc_oracle is not None:
            hit = (chosen.ground_truth_score >= acc_oracle.ground_truth_score)
            regret = acc_oracle.ground_truth_score - chosen.ground_truth_score

        acc_oracle_id = acc_oracle.pipeline_id if acc_oracle is not None else ""

        return chosen, acc_oracle_id, hit, regret

    def _select_online(
        self, outcomes: List[PipelineOutcome], raw: Optional[PipelineOutcome]
    ) -> PipelineOutcome:
        """在线代理选择规则（v5：允许审慎类别切换，避免「类别锁死」）。

        规则（阈值在验证集标定）：
        1. 若修复分支与 raw 预测类别一致，且校准置信度提升 > δ_conf，
           且无过曝伪影 → 采纳修复分支；
        2. 若类别不一致，仅在「raw 不确定（margin < τ_uncert）且修复分支
           校准后间隔显著更大（margin > κ·raw.margin）且无伪影」时才允许切类；
        3. 其余情况回退 raw。

        这样既能回退无效/有害修复，又不会让输出类别恒等于 raw，
        从而允许 Accuracy/Macro-F1 真正发生变化。
        """
        if raw is None:
            return max(outcomes, key=lambda o: o.score)

        # 可学习 router 优先：用网络从特征预测各分支正确概率，选最高分。
        # 保留两层安全约束：clip_ratio 过曝排除（select 内）+ 置信度保护
        # （learned 选的分支分类置信度不得比 raw 低超过 conf_drop_tolerance，
        #   防止网络选一个明显低置信度的分支）。
        if self._learned_router is not None and self._learned_router.available:
            diag = getattr(self, "_current_diagnosis", None)
            clip_max = self._cfg("clip_max", 0.10)
            conf_drop = float(self._router_cfg.get("conf_drop_tolerance", 0.15))
            abstain = float(self._router_cfg.get("abstain_threshold", 0.0))
            # select 内部已含排除/置信度保护/abstain（safe_argmax），此处只取结果
            best_pid, _probs = self._learned_router.select(
                diag, outcomes, clip_max=clip_max, conf_drop=conf_drop,
                abstain_threshold=abstain)
            if best_pid is not None:
                chosen = next((o for o in outcomes if o.pipeline_id == best_pid), None)
                if chosen is not None:
                    return chosen
            return raw

        # 候选修复分支（排除 raw）
        restorations = [o for o in outcomes if o.upstream != "" and not o.fallback]

        # 无有效修复分支 → 回退 raw
        if not restorations:
            return raw

        # 阈值（在验证集标定，此处给出默认值）
        delta_conf = self._cfg("delta_conf", 0.05)        # 置信度提升最小阈值
        tau_uncert = self._cfg("tau_uncert", 0.10)        # raw 不确定度阈值（margin 低于此视为不确定）
        kappa = self._cfg("kappa", 1.5)                   # 间隔提升倍数
        clip_max = self._cfg("clip_max", 0.10)            # 无过曝伪影阈值

        # 按代理分数降序遍历修复分支：若最高分分支过曝/不满足条件，
        # 继续尝试次好的分支，而非直接回退 raw。
        for best_rest in sorted(restorations, key=lambda o: o.score, reverse=True):
            same_label = (best_rest.pred_label == raw.pred_label and raw.pred_label != "")
            conf_gain = best_rest.downstream_confidence - raw.downstream_confidence
            # 无伪影约束：修复分支的 clip_ratio 需低于阈值，否则视为过曝伪影（不采纳）
            # 统一用 is_overexposed 容差口径，与 learned router / LLM 复核一致
            no_artifact = not is_overexposed(best_rest.clip_ratio, clip_max)

            # 情况 1：类别一致，且置信度提升足够，且无伪影 → 采纳
            if same_label and conf_gain > delta_conf and no_artifact:
                return best_rest

            # 情况 2：类别不一致，但 raw 不确定 + 修复分支间隔显著更大 + 无伪影 → 允许切类
            raw_uncertain = raw.margin < tau_uncert
            margin_gain = best_rest.margin > kappa * raw.margin + 1e-6
            if (not same_label) and raw_uncertain and margin_gain and no_artifact:
                return best_rest

        # 所有修复分支都不满足条件 → 回退 raw
        return raw

    def _cfg(self, key: str, default: float) -> float:
        """读取在线选择器阈值（config 传入，否则用默认值）。"""
        d = getattr(self, "_select_cfg", None)
        if not d:
            return default
        return float(d.get(key, default))

    @staticmethod
    def _decision_label(chosen: PipelineOutcome) -> str:
        if chosen.upstream == "":
            return "回退原图（修复未带来下游收益）"
        if chosen.fallback:
            return "修复采纳（模型降级占位）"
        return "修复采纳"

    # ---------- 处理器懒加载 ----------

    def _resolve_weight_path(self, name: str, registry_cfg: Optional[Dict], flat_map: Optional[Dict]) -> Optional[str]:
        """解析处理器权重路径（本地路径优先于 URL）。

        优先级：
        1. flat_map（upstream_weights / downstream_weights 扁平表）中的本地路径
        2. registry_cfg（upstreams / downstreams 子表）中的 weight_path 字段
        3. registry_cfg 中的 weight_url（仅当其为本地路径时才直接使用；
           纯 URL 需先 download_weights 落到本地，故返回 None 以触发降级）
        """
        if isinstance(flat_map, dict) and name in flat_map:
            p = flat_map[name]
            if isinstance(p, str) and p and not p.startswith(("http://", "https://")):
                return p
        if isinstance(registry_cfg, dict):
            wp = registry_cfg.get("weight_path")
            if isinstance(wp, str) and wp and not wp.startswith(("http://", "https://")):
                return wp
            wu = registry_cfg.get("weight_url")
            if isinstance(wu, str) and wu and not wu.startswith(("http://", "https://")):
                return wu
        return None

    def _get_upstream(self, name: str):
        if name not in self._upstream_cache:
            module_path, class_name = _UPSTREAM_REGISTRY[name]
            module = importlib.import_module(module_path)
            cls = getattr(module, class_name)
            weight_path = self._resolve_weight_path(
                name,
                self.config.get("upstreams", {}).get(name),
                self.config.get("upstream_weights", {}),
            )
            kwargs: Dict[str, Any] = {}
            if name == "ffdnet":
                cfg = self.config.get("upstreams", {}).get(name, {})
                kwargs["sigma"] = float(cfg.get("sigma", 25.0))
            if name.startswith("restormer_"):
                kwargs["variant"] = name
            if name.startswith("realesrgan_"):
                kwargs["scale"] = int(name.rsplit("x", 1)[1])
            self._upstream_cache[name] = cls(
                device=self.device, weight_path=weight_path, **kwargs
            )
        return self._upstream_cache[name]

    def _get_downstream(self, name: str):
        if name not in self._downstream_cache:
            module_path, class_name = _DOWNSTREAM_REGISTRY[name]
            module = importlib.import_module(module_path)
            cls = getattr(module, class_name)
            weight_path = self._resolve_weight_path(
                name,
                self.config.get("downstreams", {}).get(name),
                self.config.get("downstream_weights", {}),
            )
            kwargs: Dict[str, Any] = {}
            down_cfg = self.config.get("downstreams", {}).get(name)
            if isinstance(down_cfg, dict):
                if "num_classes" in down_cfg:
                    kwargs["num_classes"] = int(down_cfg["num_classes"])
                if "labels" in down_cfg:
                    kwargs["labels"] = list(down_cfg["labels"])
            self._downstream_cache[name] = cls(
                device=self.device, weight_path=weight_path, **kwargs
            )
        return self._downstream_cache[name]
