"""13 工具 vs 17 工具 test 对照汇总（同数据、同分类器、同 router 训练协议）。

数据：
  13 工具 results/iterative_test_13tool/seed{s}/per_image.jsonl
  17 工具 results/test_matrix_seeds3/seed{s}/per_image.csv（method=iterative 行）
输出：results/iterative_test_13tool/{report.md, summary.json}
"""
from __future__ import annotations

import csv
import json
import os
from collections import defaultdict
from math import comb

import numpy as np
from sklearn.metrics import f1_score

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
OUT = os.path.join(ROOT, "results", "iterative_test_13tool")
SEEDS = (1, 2, 3)
LABELS = ["cardboard", "glass", "metal", "paper", "plastic", "trash"]


def subset_map():
    """用清单建立 path → 子集映射（图片目录名不含 test_ 前缀，不能靠字符串猜）。"""
    m = {}
    for sub, rel in (("lowlight", "results/router_test_17branch/test_lowlight.tsv"),
                     ("noise", "results/router_test_17branch/test_noise.tsv")):
        for line in open(os.path.join(ROOT, rel), encoding="utf-8"):
            line = line.strip()
            if line and not line.startswith("#"):
                m[line.split("\t")[0]] = sub
    return m


def load13(seed, smap):
    p = os.path.join(OUT, f"seed{seed}", "per_image.jsonl")
    rows = [json.loads(l) for l in open(p, encoding="utf-8")]
    out = {}
    for r in rows:
        out[r["path"]] = {
            "label": r["label"],
            "subset": smap.get(r["path"], "noise"),
            "raw_ok": bool(r["raw_ok"]),
            "ok": bool(r["chosen_ok"]),
            "pred": r["chosen_pred"],
            "raw_pred": r["raw_pred"],
            "calls": int(r["restore_calls"]),
            "selected": r["result"].get("selected_upstream") or "raw",
            "group": os.path.basename(r["path"]),
        }
    return out


def load17(seed):
    p = os.path.join(ROOT, "results", "test_matrix_seeds3", f"seed{seed}", "per_image.csv")
    out = {}
    for r in csv.DictReader(open(p, encoding="utf-8")):
        if r["method"] != "iterative":
            continue
        out[r["path"]] = {"ok": r["ok"] == "1", "calls": int(r["restore_calls"]),
                          "pred": r["pred"], "selected": r["chosen_upstream"]}
    return out


def mcnemar(b, c):
    n = b + c
    if n == 0:
        return 1.0
    k = min(b, c)
    return float(min(1.0, 2 * sum(comb(n, i) for i in range(k + 1)) / 2 ** n))


def boot(groups, diffs, n_boot=2000, seed=42):
    rng = np.random.default_rng(seed)
    idx = defaultdict(list)
    for i, g in enumerate(groups):
        idx[g].append(i)
    uniq = sorted(idx)
    arr = np.asarray(diffs, dtype=float)
    v = []
    for _ in range(n_boot):
        pick = rng.choice(uniq, size=len(uniq), replace=True)
        ii = [i for g in pick for i in idx[g]]
        v.append(arr[ii].mean())
    lo, hi = np.percentile(v, [2.5, 97.5])
    return float(lo), float(hi)


def main() -> int:
    data = {}
    smap = subset_map()
    for s in SEEDS:
        a, b = load13(s, smap), load17(s)
        paths = sorted(set(a) & set(b))
        acc13 = float(np.mean([a[p]["ok"] for p in paths]))
        acc17 = float(np.mean([b[p]["ok"] for p in paths]))
        raw = float(np.mean([a[p]["raw_ok"] for p in paths]))
        f1_13 = float(f1_score([a[p]["label"] for p in paths], [a[p]["pred"] for p in paths],
                               labels=LABELS, average="macro", zero_division=0))
        f1_17 = float(f1_score([a[p]["label"] for p in paths], [b[p]["pred"] for p in paths],
                               labels=LABELS, average="macro", zero_division=0))
        d17 = [int(a[p]["ok"]) - int(b[p]["ok"]) for p in paths]
        bb = sum(1 for p in paths if b[p]["ok"] and not a[p]["ok"])
        cc = sum(1 for p in paths if a[p]["ok"] and not b[p]["ok"])
        dr = [int(a[p]["ok"]) - int(a[p]["raw_ok"]) for p in paths]
        br = sum(1 for p in paths if a[p]["raw_ok"] and not a[p]["ok"])
        cr = sum(1 for p in paths if not a[p]["raw_ok"] and a[p]["ok"])
        data[s] = {
            "n": len(paths), "acc13": acc13, "acc17": acc17, "raw": raw,
            "f1_13": f1_13, "f1_17": f1_17,
            "calls13": float(np.mean([a[p]["calls"] for p in paths])),
            "calls17": float(np.mean([b[p]["calls"] for p in paths])),
            "vs17": {"diff": float(np.mean(d17)), "ci95": boot([a[p]["group"] for p in paths], d17),
                     "only13": cc, "only17": bb, "p": mcnemar(bb, cc)},
            "vs_raw": {"diff": float(np.mean(dr)), "ci95": boot([a[p]["group"] for p in paths], dr),
                       "fixes": cr, "harms": br, "p": mcnemar(br, cr)},
            "by_subset": {},
        }
        for sub in ("lowlight", "noise"):
            ps = [p for p in paths if a[p]["subset"] == sub]
            data[s]["by_subset"][sub] = {
                "n": len(ps),
                "raw": float(np.mean([a[p]["raw_ok"] for p in ps])),
                "acc13": float(np.mean([a[p]["ok"] for p in ps])),
                "acc17": float(np.mean([b[p]["ok"] for p in ps])),
            }
        sel = defaultdict(int)
        for p in paths:
            sel[a[p]["selected"]] += 1
        data[s]["selection13"] = dict(sorted(sel.items(), key=lambda x: -x[1]))

    def mean(key):
        v = [data[s][key] for s in SEEDS]
        return float(np.mean(v)), float(np.std(v, ddof=1))

    m13, s13 = mean("acc13")
    m17, s17 = mean("acc17")
    mraw, _ = mean("raw")
    mf13, _ = mean("f1_13")
    mf17, _ = mean("f1_17")
    mc13, _ = mean("calls13")
    mc17, _ = mean("calls17")

    summary = {"seeds": data,
               "mean": {"acc13": m13, "acc13_sd": s13, "acc17": m17, "acc17_sd": s17,
                        "raw": mraw, "f1_13": mf13, "f1_17": mf17,
                        "calls13": mc13, "calls17": mc17}}
    json.dump(summary, open(os.path.join(OUT, "summary.json"), "w", encoding="utf-8"),
              ensure_ascii=False, indent=2)

    A = ["# 13 工具 vs 17 工具（test 860 张，三个配对 seed）\n",
         "> 13 分支 router 由 17 分支缓存按列切片重训（未重跑任何修复）；"
         "准入改为 raw + 8 去噪 + 4 低光；工具与分类器权重冻结。\n",
         "## 一、总体\n",
         "| 配置 | Accuracy | Macro-F1 | 平均调用 |",
         "|---|---:|---:|---:|",
         f"| Raw-only | {mraw:.4f} | — | 0 |",
         f"| **13 工具** | **{m13:.4f}** ({s13:.4f}) | {mf13:.4f} | {mc13:.2f} |",
         f"| 17 工具 | {m17:.4f} ({s17:.4f}) | {mf17:.4f} | {mc17:.2f} |",
         "",
         "## 二、逐 seed\n",
         "| seed | 13 工具 | 17 工具 | 差 | 13 调用 | 17 调用 |",
         "|---|---:|---:|---:|---:|---:|"]
    for s in SEEDS:
        d = data[s]
        A.append(f"| {s} | {d['acc13']:.4f} | {d['acc17']:.4f} | {d['acc13'] - d['acc17']:+.4f} "
                 f"| {d['calls13']:.2f} | {d['calls17']:.2f} |")
    A.append("")
    A.append("## 三、配对检验（13 工具 vs 17 工具）\n")
    A.append("| seed | 只 13 对 | 只 17 对 | 95% CI | McNemar p |")
    A.append("|---|---:|---:|---|---:|")
    for s in SEEDS:
        v = data[s]["vs17"]
        A.append(f"| {s} | {v['only13']} | {v['only17']} | "
                 f"[{v['ci95'][0]:+.4f}, {v['ci95'][1]:+.4f}] | {v['p']:.3g} |")
    A.append("")
    A.append("## 四、13 工具相对 Raw\n")
    A.append("| seed | 纠正 | 改错 | 差值 | 95% CI | p |")
    A.append("|---|---:|---:|---:|---|---:|")
    for s in SEEDS:
        v = data[s]["vs_raw"]
        A.append(f"| {s} | {v['fixes']} | {v['harms']} | {v['diff']:+.4f} | "
                 f"[{v['ci95'][0]:+.4f}, {v['ci95'][1]:+.4f}] | {v['p']:.3g} |")
    A.append("")
    A.append("## 五、分退化类型\n")
    A.append("| seed | 低光 Raw | 低光 13 | 低光 17 | 噪声 Raw | 噪声 13 | 噪声 17 |")
    A.append("|---|---:|---:|---:|---:|---:|---:|")
    for s in SEEDS:
        b = data[s]["by_subset"]
        A.append(f"| {s} | {b['lowlight']['raw']:.4f} | {b['lowlight']['acc13']:.4f} | "
                 f"{b['lowlight']['acc17']:.4f} | {b['noise']['raw']:.4f} | "
                 f"{b['noise']['acc13']:.4f} | {b['noise']['acc17']:.4f} |")
    A.append("")
    A.append("## 六、13 工具的选择分布（逐 seed）\n")
    for s in SEEDS:
        A.append(f"- seed{s}：" + "，".join(f"{k} {v}" for k, v in
                                          list(data[s]["selection13"].items())[:8]))
    A.append("")
    A.append("## 七、结论口径\n")
    A.append(f"- 去掉 4 个工具后平均准确率 {m13:.4f}，17 工具 {m17:.4f}，差 {m13 - m17:+.4f}")
    A.append("- 该差异与「这 4 个工具在 17 工具版的 11610 次决策中被选中 0 次」一致；"
             "灰区候选由 17 降为 13，成本随之下降。")
    A.append("- 开发划分 valB、强度曲线与六场景主表尚未用 13 工具重算。")
    open(os.path.join(OUT, "report.md"), "w", encoding="utf-8").write("\n".join(A) + "\n")
    print(f"写出 {OUT}/report.md 与 summary.json")
    print(f"13 工具 {m13:.4f}±{s13:.4f} | 17 工具 {m17:.4f}±{s17:.4f} | raw {mraw:.4f}")
    for s in SEEDS:
        print(f"  seed{s}: 13 {data[s]['acc13']:.4f} vs 17 {data[s]['acc17']:.4f} "
              f"| p={data[s]['vs17']['p']:.3g}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
