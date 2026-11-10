"""修复工具注册表（单一来源）。

工具分组（候选生成用）与分支顺序（learned router 用）都在此定义，
orchestrator 与 learned_router 从这里导入，避免两处维护不一致。

所有修复器均冻结、只做推理，不训练。
"""
from __future__ import annotations

from typing import Dict, List

# 工具按退化类型分组（候选生成用）。
# - noise/lowlight 有诊断信号，明确退化时触发本组全部工具参与竞争；
# - deblur/derain/sr 无诊断信号，仅在「灰区」时并行探索，由验证闭环判断是否有益。
TOOL_GROUPS: Dict[str, List[str]] = {
    "noise": ["denoise", "ffdnet", "scunet",
              "restormer_real", "restormer_blind",
              "restormer_sigma15", "restormer_sigma25", "restormer_sigma50"],
    "lowlight": ["lowlight", "zero_dce_plus_plus", "sci_lowlight", "retinexformer"],
    "deblur": ["restormer_defocus", "mprnet_deblur"],
    "sr": ["realesrgan_x2", "realesrgan_x4"],
}

# 全部修复工具（raw 直通用 "" 表示，不在此列）
ALL_UPSTREAM_TOOLS: List[str] = [
    up for group in TOOL_GROUPS.values() for up in group
]

# learned router 的分支顺序：raw（upstream=""）在前，其余按分组顺序展开。
# 注意：raw 分支的 upstream 是空字符串 ""，而非 "raw"。
BRANCH_ORDER: List[str] = [""] + ALL_UPSTREAM_TOOLS
