"""阶段三收口：六场景 × 3 seed 汇总（质量 + 成本）。

读取 results/ablation/{root}/seed{s}/ablation_{scene}_learned.json 和 cost_{scene}_seed{s}.json，
对每个方法输出多指标 mean/std/每 seed 原始值 + 相对 raw 的 paired difference，
并生成成本汇总。只做描述性统计，不写「显著/不显著」。

用法：
    python scripts/summarize_final.py --root results/ablation/final_v4
"""
from __future__ import annotations

import argparse
import json
import os

import numpy as np

SCENES = ["clean", "noise", "lowlight", "motion_blur", "jpeg", "mixed"]
SEEDS = [1, 2, 3]
METHODS = ["raw_only", "fixed_lowlight", "fixed_denoise", "no_raw_candidate", "no_fallback",
           "agent_learned", "uniform", "confidence", "soft_gating", "no_candidate_pruning",
           "global_oracle"]
METRICS = ["accuracy", "macro_f1", "negative_gain_rate", "net_correct_gain",
           "mean_global_regret", "coverage"]

# 多强度：目录名 -> (场景前缀, 强度取值列表)
MULTILEVEL = [
    ("noise_multilevel", "noise", ["15", "25", "50"]),
    ("lowlight_multilevel", "lowlight", ["2", "2.5", "3", "3.5"]),
]


def _mean_std(values):
    a = np.asarray([v for v in values if v is not None], dtype=np.float64)
    if len(a) == 0:
        return None, None
    return round(float(a.mean()), 4), round(float(a.std()), 4)


def _summarize_group(root, file_map):
    """对一组「场景名 -> {seed: 结果文件路径}」做质量汇总，返回 {scene: {method: {...}}}。"""
    summary = {}
    for scene, seed_paths in file_map.items():
        per_scene = {}
        for method in METHODS:
            per_metric = {}
            for metric in METRICS:
                vals = []
                for s in SEEDS:
                    p = seed_paths.get(s)
                    if not p or not os.path.exists(p):
                        continue
                    d = json.load(open(p, encoding="utf-8"))
                    st = d.get("strategies", {}).get(method)
                    if st is None:
                        continue
                    vals.append(st.get(metric))
                mean, std = _mean_std(vals)
                per_metric[metric] = {
                    "mean": mean,
                    "std": std,
                    "values": vals,
                    "n_seeds": len([v for v in vals if v is not None]),
                }
            per_scene[method] = per_metric

        # 相对 raw 的 paired difference
        raw_vals = {}
        for s in SEEDS:
            p = seed_paths.get(s)
            if not p or not os.path.exists(p):
                continue
            d = json.load(open(p, encoding="utf-8"))
            raw_vals[s] = d.get("strategies", {}).get("raw_only", {}).get("accuracy")
        for method in METHODS:
            if method == "raw_only" or method not in per_scene:
                continue
            diffs = []
            for s in SEEDS:
                p = seed_paths.get(s)
                if not p or not os.path.exists(p):
                    continue
                d = json.load(open(p, encoding="utf-8"))
                st = d.get("strategies", {}).get(method)
                if st is None or raw_vals.get(s) is None:
                    continue
                diffs.append(st.get("accuracy") - raw_vals[s])
            if diffs:
                dm, ds = _mean_std(diffs)
                per_scene[method]["vs_raw_diff"] = {
                    "mean": dm, "std": ds, "values": [round(x, 4) for x in diffs],
                }
        summary[scene] = per_scene
    return summary


def build_summary(root: str) -> dict:
    """纯函数：从 root 下的结果文件重算完整质量+成本汇总，无写盘副作用。

    返回 {"quality": {scene: {method: {metric: {mean,std,values,n_seeds},
                    vs_raw_diff: {mean,std,values}}}}, "cost": {scene: {seed: {...}}},
          "multilevel": {name: {...六场景同构 quality 汇总...}}}。
    供 main() 写盘，也供 validate_final.py 深度比较（校验汇总与结果一致）。
    """
    # 六场景质量汇总
    scene_map = {}
    for scene in SCENES:
        scene_map[scene] = {s: os.path.join(root, f"seed{s}", f"ablation_{scene}_learned.json")
                            for s in SEEDS}
    summary = _summarize_group(root, scene_map)

    # 多强度质量汇总（21 组 = 7 强度 × 3 seed）
    multilevel_map = {}
    for mdir, prefix, vals in MULTILEVEL:
        for v in vals:
            name = f"{prefix}_s{v}"
            multilevel_map[name] = {s: os.path.join(root, mdir, f"seed{s}", f"ablation_{name}_learned.json")
                                    for s in SEEDS}
    multilevel = _summarize_group(root, multilevel_map)

    # 成本汇总（仅六场景有 cost 文件）
    cost_summary = {}
    for scene in SCENES:
        for s in SEEDS:
            p = os.path.join(root, f"cost_{scene}_seed{s}.json")
            if not os.path.exists(p):
                continue
            d = json.load(open(p, encoding="utf-8"))
            cs = d.get("strategies", {}).get("candidate_search", {})
            sg = d.get("strategies", {}).get("soft_gating_fusion_overhead", {})
            cost_summary.setdefault(scene, {})[s] = {
                "candidate_search_e2e_ms": cs.get("candidate_search_e2e_ms", {}).get("mean_ms"),
                "executed_branches": cs.get("executed_branches"),
                "fallback_ratio": cs.get("fallback_ratio"),
                "router_trigger_rate": cs.get("router_trigger_rate"),
                "fusion_overhead_ms": sg.get("fusion_overhead_ms", {}).get("mean_ms"),
                "error_rate": cs.get("error_rate"),
            }

    return {"quality": summary, "cost": cost_summary, "multilevel": multilevel}


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--root", default="results/ablation/final_v4")
    args = ap.parse_args()

    summary = build_summary(args.root)
    cost_summary = summary["cost"]
    quality = summary["quality"]

    os.makedirs(args.root, exist_ok=True)
    out = os.path.join(args.root, "seed_scene_summary.json")
    with open(out, "w", encoding="utf-8") as f:
        json.dump(summary, f, ensure_ascii=False, indent=2)
    print(f"汇总已保存 -> {out}")

    # 打印关键对比表
    print("\n" + "=" * 78)
    print(f"{'场景':12s} {'raw':>16s} {'hard':>16s} {'soft':>16s} {'no_pruning':>16s}")
    print("-" * 78)
    for scene in SCENES:
        def acc(m):
            e = quality[scene].get(m, {}).get("accuracy", {})
            if not e or e.get("mean") is None:
                return "           -"
            return f"{e['mean']:.4f}±{e['std']:.4f}"
        print(f"{scene:12s} {acc('raw_only'):>16s} {acc('agent_learned'):>16s} "
              f"{acc('soft_gating'):>16s} {acc('no_candidate_pruning'):>16s}")
    print("=" * 78)
    print("\n相对 raw 的 paired difference（描述性）:")
    for scene in SCENES:
        hard = quality[scene].get("agent_learned", {}).get("vs_raw_diff", {})
        ncp = quality[scene].get("no_candidate_pruning", {}).get("vs_raw_diff", {})
        hd = hard.get("mean")
        nd = ncp.get("mean")
        print(f"  {scene:12s} hard: {hd:+.4f}   no_pruning: {nd:+.4f}")

    print("\n成本（candidate_search，mean over seeds）:")
    for scene in SCENES:
        vals = cost_summary.get(scene, {})
        if not vals:
            continue
        e2e = [v.get("candidate_search_e2e_ms") for v in vals.values() if v.get("candidate_search_e2e_ms") is not None]
        trig = [v.get("router_trigger_rate") for v in vals.values() if v.get("router_trigger_rate") is not None]
        em = round(float(np.mean(e2e)), 1) if e2e else None
        tm = round(float(np.mean(trig)), 3) if trig else None
        print(f"  {scene:12s} e2e={em}ms  trigger={tm}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
