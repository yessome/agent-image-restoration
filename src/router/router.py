"""流水线路由的数据结构（候选方案描述）。

候选方案（PipelineRoute）描述「上游 + 下游」的流水线组合。
候选方案的生成逻辑在 orchestrator._gen_candidates（诊断裁剪 + 灰区探索），
决策逻辑在 orchestrator._select_online（规则）或 learned_router（可学习）。

注：v1 时代的规则表 Router.route() 已移除——它用二值 noise_high/lowlight
做判定，与 _gen_candidates 的连续 noise_score/lowlight_score 口径不一致，
且从未被调用（候选生成早已由 _gen_candidates 接管）。
"""
from __future__ import annotations

from dataclasses import dataclass


@dataclass
class PipelineRoute:
    """路由结果：选中的流水线。"""
    pipeline_id: str
    upstream: str
    downstream: str
    reason: str
