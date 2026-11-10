"""test 主对照报告：五方法 × 三 seed。

读取 results/test_matrix_seeds3/seed{s}/per_image.csv，输出
results/test_matrix_seeds3/{report.md, summary.json}

口径：
  * Oracle / hit / regret 在**完整候选池**（raw + 16 工具）上计算
  * decision_ms 只含该方法实际执行的修复+分类；audit_ms 是被补跑工具的耗时（单列）
"""
from __future__ import annotations

import json
import os
from collections import Counter, defaultdict
from math import comb

import numpy as np
from sklearn.metrics import f1_score

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
SRC = os.path.join(ROOT, "results", "test_matrix_seeds3")
OUT = SRC
SEEDS = (1, 2, 3)
LABELS = ["cardboard", "glass", "metal", "paper", "plastic", "trash"]
METHODS = ["raw", "fixed_type", "diag_rule", "legacy_rule", "iterative"]
NICE = {"raw": "Raw-only", "fixed_type": "Fixed restoration", "diag_rule": "诊断规则",
        "legacy_rule": "legacy 规则 Agent", "iterative": "learned 循环 Agent"}
N_BOOT = 2000


def load(seed):
    import csv
    p = os.path.join(SRC, f"seed{seed}", "per_image.csv")
    rows = list(csv.DictReader(open(p, encoding="utf-8")))
    for r in rows:
        r["ok"] = r["ok"] == "1"
        r["raw_ok"] = r["raw_ok"] == "1"
        r["hit_full"] = r["hit_full"] == "1"
        r["restore_calls"] = int(r["restore_calls"])
        r["decision_ms"] = float(r["decision_ms"])
        r["audit_ms"] = float(r["audit_ms"])
        r["regret_full"] = float(r["regret_full"])
    return rows


def group_ci(groups, diffs, n_boot=N_BOOT, seed=42):
    rng = np.random.default_rng(seed)
    idx = defaultdict(list)
    for i, g in enumerate(groups):
        idx[g].append(i)
    uniq = sorted(idx)
    arr = np.asarray(diffs, dtype=float)
    vals = []
    for _ in range(n_boot):
        pick = rng.choice(uniq, size=len(uniq), replace=True)
        ii = [i for g in pick for i in idx[g]]
        vals.append(arr[ii].mean())
    lo, hi = np.percentile(vals, [2.5, 97.5])
    return float(lo), float(hi)


def mcnemar(b, c):
    n = b + c
    if n == 0:
        return 1.0
    k = min(b, c)
    return float(min(1.0, 2 * sum(comb(n, i) for i in range(k + 1)) / (2 ** n)))


def summarize(rows):
    """rows: 某个 (seed, method) 的全部图。"""
    y = [r["label"] for r in rows]
    pred = [r["pred"] for r in rows]
    ok = np.array([r["ok"] for r in rows])
    raw_ok = np.array([r["raw_ok"] for r in rows])
    calls = np.array([r["restore_calls"] for r in rows])
    sel = Counter((r["chosen_upstream"] or "raw") for r in rows)
    return {
        "n": len(rows),
        "accuracy": float(ok.mean()),
        "macro_f1": float(f1_score(y, pred, labels=LABELS, average="macro", zero_division=0)),
        "raw_accuracy": float(raw_ok.mean()),
        "fixes": int(((~raw_ok) & ok).sum()),
        "harms": int((raw_ok & ~ok).sum()),
        "mean_calls": float(calls.mean()),
        "call_dist": {str(k): int((calls == k).sum()) for k in range(0, 18)},
        "raw_return_ratio": float(np.mean([(r["chosen_upstream"] or "raw") == "raw" for r in rows])),
        "mean_decision_ms": float(np.mean([r["decision_ms"] for r in rows])),
        "mean_audit_ms": float(np.mean([r["audit_ms"] for r in rows])),
        "oracle_full": float(np.mean([r["oracle_full_ok"] == "1" for r in rows])),
        "oracle_admitted": float(np.mean([r["oracle_admitted_ok"] == "1" for r in rows])),
        "hit_rate": float(np.mean([r["hit_full"] for r in rows])),
        "mean_regret": float(np.mean([r["regret_full"] for r in rows])),
        "selection_counts": dict(sel.most_common()),
        "by_subset": {
            sub: {
                "n": int(sum(1 for r in rows if r["subset"] == sub)),
                "accuracy": float(np.mean([r["ok"] for r in rows if r["subset"] == sub]) or 0.0),
                "raw_accuracy": float(np.mean([r["raw_ok"] for r in rows if r["subset"] == sub]) or 0.0),
            } for sub in ("lowlight", "noise")
        },
    }


def main() -> int:
    data = {}
    for s in SEEDS:
        rows = load(s)
        by_m = defaultdict(list)
        for r in rows:
            by_m[r["method"]].append(r)
        data[s] = {m: summarize(by_m[m]) for m in METHODS}
        # 配对差（vs raw）
        pivot = {(r["path"], r["method"]): r for r in rows}
        for m in METHODS:
            diffs, groups = [], []
            b = c = 0
            for path in {r["path"] for r in by_m[m]}:
                a = pivot[(path, m)]
                base = pivot[(path, "raw")]
                diffs.append(int(a["ok"]) - int(base["ok"]))
                groups.append(a["group"])
                if base["ok"] and not a["ok"]:
                    b += 1
                elif not base["ok"] and a["ok"]:
                    c += 1
            lo, hi = group_ci(groups, diffs)
            data[s][m]["vs_raw"] = {"diff": float(np.mean(diffs)), "ci95": [lo, hi],
                                    "only_method": c, "only_raw": b,
                                    "p": mcnemar(b, c)}

    summary = {"seeds": data,
               "note": "Oracle/hit/regret 在完整候选池（raw+16 工具）上；audit_ms 为补跑工具的耗时，单列"}
    json.dump(summary, open(os.path.join(OUT, "summary.json"), "w", encoding="utf-8"),
              ensure_ascii=False, indent=2)

    A = []
    A.append("# test 主对照：五种调度方法 × 三个 classifier seed\n")
    A.append("> 数据：`results/router_test_17branch/test_lowlight.tsv` + `test_noise.tsv`（430 + 430 = 860 张）。")
    A.append("> 修复结果跨 seed 共享：每张图的 16 个工具只修复一次，三个分类器各分类一次。")
    A.append("> 冻结清单见 `results/frozen_manifest/frozen_manifest.json`。\n")

    A.append("## 一、主表（三 seed 均值，括号内为 seed 间标准差）\n")
    A.append("| 方法 | Accuracy | Macro-F1 | 纠正 | 改错 | 平均调用 | 决策耗时(ms) | 审计耗时(ms) |")
    A.append("|---|---:|---:|---:|---:|---:|---:|---:|")
    for m in METHODS:
        acc = [data[s][m]["accuracy"] for s in SEEDS]
        f1 = [data[s][m]["macro_f1"] for s in SEEDS]
        fx = np.mean([data[s][m]["fixes"] for s in SEEDS])
        hm = np.mean([data[s][m]["harms"] for s in SEEDS])
        ca = [data[s][m]["mean_calls"] for s in SEEDS]
        dm = [data[s][m]["mean_decision_ms"] for s in SEEDS]
        am = [data[s][m]["mean_audit_ms"] for s in SEEDS]
        A.append(f"| **{NICE[m]}** | {np.mean(acc):.4f} ({np.std(acc, ddof=1):.4f}) | "
                 f"{np.mean(f1):.4f} ({np.std(f1, ddof=1):.4f}) | {fx:.1f} | {hm:.1f} | "
                 f"{np.mean(ca):.2f} ({np.std(ca, ddof=1):.2f}) | {np.mean(dm):.0f} | {np.mean(am):.0f} |")
    A.append("")
    A.append("> `决策耗时` = 该方法实际执行的修复+分类；`审计耗时` = 为算完整候选池而补跑的工具，"
             "两者严格分开，审计不回流决策。\n")

    A.append("## 二、逐 seed 准确率与相对 Raw 的配对差\n")
    A.append("| seed | 方法 | Raw | Accuracy | Δ vs Raw | 95% CI | McNemar p |")
    A.append("|---|---|---:|---:|---:|---|---:|")
    for s in SEEDS:
        for m in METHODS:
            d = data[s][m]
            if m == "raw":
                A.append(f"| {s} | {NICE[m]} | {d['raw_accuracy']:.4f} | {d['accuracy']:.4f} | — | — | — |")
            else:
                v = d["vs_raw"]
                A.append(f"| {s} | {NICE[m]} | {d['raw_accuracy']:.4f} | {d['accuracy']:.4f} | "
                         f"{v['diff']:+.4f} | [{v['ci95'][0]:+.4f}, {v['ci95'][1]:+.4f}] | {v['p']:.3g} |")
    A.append("")

    A.append("## 三、遗憾值（完整候选池，raw + 16 工具）\n")
    A.append("| 方法 | 候选池 Oracle | 准入池 Oracle | hit | 平均 regret |")
    A.append("|---|---:|---:|---:|---:|")
    for m in METHODS:
        of = np.mean([data[s][m]["oracle_full"] for s in SEEDS])
        oa = np.mean([data[s][m]["oracle_admitted"] for s in SEEDS])
        ht = np.mean([data[s][m]["hit_rate"] for s in SEEDS])
        rg = np.mean([data[s][m]["mean_regret"] for s in SEEDS])
        A.append(f"| **{NICE[m]}** | {of:.4f} | {oa:.4f} | {ht:.4f} | {rg:.4f} |")
    A.append("")
    A.append("> 候选池 Oracle 与 regret 对所有方法使用**同一池子**（每张图 raw+16 工具全跑），"
             "因此方法与 Oracle 的差距可比；这与 executed-only 口径不同，不可混用。\n")

    A.append("## 四、分退化类型\n")
    A.append("| 方法 | 低光 Acc | 低光 Raw | 噪声 Acc | 噪声 Raw |")
    A.append("|---|---:|---:|---:|---:|")
    for m in METHODS:
        lr = np.mean([data[s][m]["by_subset"]["lowlight"]["accuracy"] for s in SEEDS])
        lrr = np.mean([data[s][m]["by_subset"]["lowlight"]["raw_accuracy"] for s in SEEDS])
        ns = np.mean([data[s][m]["by_subset"]["noise"]["accuracy"] for s in SEEDS])
        nsr = np.mean([data[s][m]["by_subset"]["noise"]["raw_accuracy"] for s in SEEDS])
        A.append(f"| **{NICE[m]}** | {lr:.4f} | {lrr:.4f} | {ns:.4f} | {nsr:.4f} |")
    A.append("")

    A.append("## 五、循环 Agent 的调用分布与工具选择（三 seed）\n")
    A.append("| seed | 0 | 1 | 2 | 3 | 4 | 平均 | raw 返回 |")
    A.append("|---|---:|---:|---:|---:|---:|---:|---:|")
    for s in SEEDS:
        d = data[s]["iterative"]
        A.append(f"| {s} | " + " | ".join(str(d["call_dist"].get(str(k), 0)) for k in range(5)) +
                 f" | {d['mean_calls']:.2f} | {d['raw_return_ratio']:.3f} |")
    A.append("")
    for s in SEEDS:
        top = list(data[s]["iterative"]["selection_counts"].items())[:8]
        A.append(f"- seed{s}：" + "，".join(f"{k} {v}" for k, v in top))
    A.append("")

    A.append("## 六、口径与边界\n")
    A.append("- 冻结后未依据 valB/test 修改任何 router、分类器、工具权重或阈值")
    A.append("- 审计分支在决策前一次性执行以便共享修复，但**结果在决策结束后才可见**，等价于停止后补跑")
    A.append("- test 清单此前已有 82 张进入旧的三分支评测缓存，但本表数据与那次评测无关；"
             "本表为 860 张完整 test 退化集")
    A.append("- `Fixed restoration` 使用清单给定的退化类型（噪声→DnCNN、低光→Zero-DCE），是参考上界，不可部署")

    open(os.path.join(OUT, "report.md"), "w", encoding="utf-8").write("\n".join(A) + "\n")
    print(f"写出 {OUT}/report.md 与 summary.json")
    for m in METHODS:
        acc = np.mean([data[s][m]["accuracy"] for s in SEEDS])
        print(f"  {NICE[m]:22s} {acc:.4f}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
