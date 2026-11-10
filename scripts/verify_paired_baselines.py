"""复核稿件 Table 2 / Table 4 数值，并补算主流调度基线之间的配对差值区间。

数据来源（同一批 430 test sources、六个场景、三个 classifier-router 配对）：
  results/scene_matrix_13tool/seed{s}/per_image.csv   (clean, jpeg, mixed, motion_blur)
  results/test_matrix_13tool/seed{s}/per_image.csv    (lowlight, noise)

输出：
  results/verify_paired_baselines/summary.json
  results/verify_paired_baselines/report.md
"""
from __future__ import annotations

import csv
import json
import glob
import os
from collections import defaultdict

import numpy as np

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
OUT = os.path.join(ROOT, "results", "verify_paired_baselines")
SEEDS = (1, 2, 3)
SCENES = ("clean", "noise", "lowlight", "motion_blur", "jpeg", "mixed")
METHODS = ("raw", "fixed_denoise", "fixed_lowlight", "diag_rule", "legacy_rule", "iterative")
N_BOOT = 10000
RNG_SEED = 42


CLEAN_TSV = os.path.join(ROOT, "results", "table2_scenes", "clean.tsv")


def _base(path: str) -> str:
    return path.replace("\\", "/").split("/")[-1].lower()


def stem_to_source():
    """退化清单：退化图文件名 -> 原图文件名。两个结果目录的 group 是不同哈希体系
    （场景目录与原图内容相关，test 目录另一套），必须统一到原图名，
    否则同一张原图会被拆成两个聚类单元，区间偏窄。"""
    m = {}
    for f in glob.glob(os.path.join(ROOT, "data", "degraded", "*", "manifest_*.tsv")):
        with open(f, encoding="utf-8") as fh:
            rd = csv.DictReader(fh, delimiter="\t")
            rd.fieldnames = [x.lstrip("#").strip() for x in rd.fieldnames]
            for r in rd:
                m[_base(r["degraded_path"])] = _base(r["source_path"])
    return m


def load():
    s2src = stem_to_source()
    rows = []
    for seed in SEEDS:
        for tag, f in (("scene", os.path.join(ROOT, "results", "scene_matrix_13tool", f"seed{seed}", "per_image.csv")),
                       ("test", os.path.join(ROOT, "results", "test_matrix_13tool", f"seed{seed}", "per_image.csv"))):
            with open(f, encoding="utf-8") as fh:
                block = [dict(r) for r in csv.DictReader(fh)]
            # 每个 group 唯一对应一张原图；用清单核验后再映射
            g2s = {}
            for r in block:
                stem = _base(r["path"])
                g2s.setdefault(r["group"], set()).add(s2src.get(stem, stem))
            bad = [g for g, v in g2s.items() if len(v) != 1]
            if bad:
                raise SystemExit(f"{f}: {len(bad)} 个 group 对应多张原图，无法安全聚类")
            for r in block:
                r["uid"] = next(iter(g2s[r["group"]]))
                rows.append(r)
    return rows


def index(rows):
    """(seed, subset, uid) -> {method: (ok, calls)}"""
    d = defaultdict(dict)
    for r in rows:
        key = (int(r["seed"]), r["subset"], r["uid"])
        if r["method"] in METHODS:
            d[key][r["method"]] = (int(r["ok"]), int(r["restore_calls"]))
    return d


def paired_bootstrap(idx, a, b, scenes=None):
    """按 source(group) 聚类的配对 bootstrap。a、b 为方法名。"""
    groups = sorted({k[2] for k in idx})
    # 每个 group 内按 (seed, scene) 累计差值
    per_group_diff = {}
    per_group_n = {}
    for g in groups:
        ds, ns = [], []
        for k, v in idx.items():
            if k[2] != g:
                continue
            if scenes and k[1] not in scenes:
                continue
            if a not in v or b not in v:
                continue
            ds.append(v[a][0] - v[b][0])
            ns.append(1)
        per_group_diff[g] = np.array(ds, dtype=np.float64)
        per_group_n[g] = len(ds)

    def mean_diff(gs):
        tot = sum(per_group_diff[g].sum() for g in gs)
        n = sum(per_group_n[g] for g in gs)
        return tot / n if n else float("nan")

    point = mean_diff(groups)
    rng = np.random.default_rng(RNG_SEED)
    boots = np.empty(N_BOOT)
    for i in range(N_BOOT):
        sample = rng.choice(len(groups), size=len(groups), replace=True)
        boots[i] = mean_diff([groups[j] for j in sample])
    lo, hi = np.percentile(boots, [2.5, 97.5])
    return point, float(lo), float(hi), sum(per_group_n.values())


def main():
    rows = load()
    idx = index(rows)
    os.makedirs(OUT, exist_ok=True)

    # 完整性检查：每个 (seed, scene, group) 是否六个方法齐全
    incomplete = [(k, sorted(v)) for k, v in idx.items() if len(v) != len(METHODS)]
    n_records = len(idx)

    # Table 2 复核
    table2 = {}
    for scene in SCENES:
        table2[scene] = {}
        for m in METHODS:
            vals = [v[m][0] for k, v in idx.items() if k[1] == scene and m in v]
            table2[scene][m] = float(np.mean(vals)) if vals else None
    mean_acc = {}
    for m in METHODS:
        per_seed = []
        for seed in SEEDS:
            vals = [v[m][0] for k, v in idx.items() if k[0] == seed and m in v]
            per_seed.append(float(np.mean(vals)))
        mean_acc[m] = float(np.mean(per_seed))
        mean_acc[m + "_sd"] = float(np.std(per_seed, ddof=1))

    mean_calls = {}
    for m in METHODS:
        vals = [v[m][1] for k, v in idx.items() if m in v]
        mean_calls[m] = float(np.mean(vals))

    # 场景内调用数（诊断规则 vs 学习 agent）
    calls_by_scene = {scene: {m: float(np.mean([v[m][1] for k, v in idx.items()
                                                if k[1] == scene and m in v]))
                              for m in METHODS} for scene in SCENES}

    # 配对区间
    comparisons = [
        ("iterative", "raw"),
        ("iterative", "fixed_denoise"),
        ("iterative", "fixed_lowlight"),
        ("iterative", "diag_rule"),
        ("iterative", "legacy_rule"),
        ("diag_rule", "legacy_rule"),
        ("legacy_rule", "raw"),
    ]
    ci = {}
    for a, b in comparisons:
        p, lo, hi, n = paired_bootstrap(idx, a, b)
        ci[f"{a}_minus_{b}"] = {"diff_pp": 100 * p, "ci95_pp": [100 * lo, 100 * hi], "n": n}

    # 分场景：agent - diag_rule
    by_scene_ci = {}
    for scene in SCENES:
        p, lo, hi, n = paired_bootstrap(idx, "iterative", "diag_rule", scenes={scene})
        by_scene_ci[scene] = {"diff_pp": 100 * p, "ci95_pp": [100 * lo, 100 * hi], "n": n}

    # 时延字段健康检查
    lat = {}
    for field in ("decision_ms", "audit_ms"):
        vals = [float(r[field]) for r in rows if r.get(field) not in (None, "")]
        if vals:
            arr = np.array(vals)
            lat[field] = {"n": int(arr.size), "min": float(arr.min()),
                          "p50": float(np.percentile(arr, 50)),
                          "p99": float(np.percentile(arr, 99)), "max": float(arr.max()),
                          "frac_zero": float(np.mean(arr == 0))}

    summary = {
        "n_records": n_records,
        "incomplete_records": len(incomplete),
        "table2": table2,
        "mean_acc": mean_acc,
        "mean_calls": mean_calls,
        "calls_by_scene": calls_by_scene,
        "paired_ci": ci,
        "agent_minus_diag_by_scene": by_scene_ci,
        "latency_fields": lat,
    }
    with open(os.path.join(OUT, "summary.json"), "w", encoding="utf-8") as fh:
        json.dump(summary, fh, ensure_ascii=False, indent=2)

    # report.md
    L = []
    L.append("# 配对基线复核\n")
    L.append(f"- 记录数：(seed, scene, source) = {n_records}；方法不齐的记录 {len(incomplete)} 条")
    L.append("\n## Table 2 复核（六场景准确率）\n")
    L.append("| Scene | " + " | ".join(METHODS) + " |")
    L.append("|---|" + "---:|" * len(METHODS))
    for scene in SCENES:
        L.append(f"| {scene} | " + " | ".join(f"{table2[scene][m]:.4f}" for m in METHODS) + " |")
    L.append("| **mean** | " + " | ".join(f"**{mean_acc[m]:.4f}**" for m in METHODS) + " |")
    L.append("| **SD** | " + " | ".join(f"{mean_acc[m + '_sd']:.4f}" for m in METHODS) + " |")
    L.append("| **calls** | " + " | ".join(f"{mean_calls[m]:.3f}" for m in METHODS) + " |")
    L.append("\n## 配对差值（百分点，source 聚类 bootstrap）\n")
    L.append("| 比较 | 差值 | 95% 区间 | n |")
    L.append("|---|---:|---|---:|")
    for k, v in ci.items():
        L.append(f"| {k} | {v['diff_pp']:+.2f} | [{v['ci95_pp'][0]:+.2f}, {v['ci95_pp'][1]:+.2f}] | {v['n']} |")
    L.append("\n## agent − diag_rule 分场景\n")
    L.append("| Scene | 差值 | 95% 区间 |")
    L.append("|---|---:|---|")
    for s, v in by_scene_ci.items():
        L.append(f"| {s} | {v['diff_pp']:+.2f} | [{v['ci95_pp'][0]:+.2f}, {v['ci95_pp'][1]:+.2f}] |")
    L.append("\n## 时延字段健康检查\n")
    L.append("```json")
    L.append(json.dumps(lat, indent=2, ensure_ascii=False))
    L.append("```")
    with open(os.path.join(OUT, "report.md"), "w", encoding="utf-8") as fh:
        fh.write("\n".join(L) + "\n")

    print("\n".join(L))


if __name__ == "__main__":
    main()
