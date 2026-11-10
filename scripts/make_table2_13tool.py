"""生成 13 工具版论文主表（六场景 × 7 方法，三 seed 均值 ± SD）。

与前版的区别：noise / lowlight 两个场景也来自**同一次 13 工具矩阵运行**
（results/test_matrix_13tool），因此 legacy 基线不再混用 16 工具候选池。
四个场景来自 results/scene_matrix_13tool，两者 CSV 结构完全一致。

输出：results/final_figures/table2_13tool.md 与 table2_13tool.json
"""
from __future__ import annotations

import argparse
import csv
import json
import os
from collections import defaultdict

import numpy as np

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
SEEDS = (1, 2, 3)
SCENES = [("clean", "Clean"), ("noise", "Noise, sigma 25"), ("lowlight", "Low light, gamma 3.0"),
          ("motion_blur", "Motion blur"), ("jpeg", "JPEG, quality 30"),
          ("mixed", "Mixed, gamma 3.0 then sigma 25")]
COLS = [("raw", "Raw"), ("fixed_denoise", "Fixed denoise"), ("fixed_lowlight", "Fixed low light"),
        ("diag_rule", "Diagnostic rule"), ("legacy_rule", "Legacy rule agent"),
        ("iterative", "Learned iterative agent")]
TEST_SCENES = {"noise", "lowlight"}


def load(dir_path, seed):
    p = os.path.join(dir_path, f"seed{seed}", "per_image.csv")
    out = defaultdict(dict)
    for r in csv.DictReader(open(p, encoding="utf-8")):
        out[(r["subset"], r["path"], r["method"])] = r
    return out


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--test-dir", default="results/test_matrix_13tool")
    ap.add_argument("--scene-dir", default="results/scene_matrix_13tool")
    ap.add_argument("--out", default="results/final_figures/table2_13tool")
    args = ap.parse_args()

    data = {s: {"test": load(os.path.join(ROOT, args.test_dir), s),
                "scene": load(os.path.join(ROOT, args.scene_dir), s)} for s in SEEDS}

    table = {}
    for key, nice in SCENES:
        rec = {"scene": nice, "cols": {}, "n": None}
        for s in SEEDS:
            src = data[s]["test" if key in TEST_SCENES else "scene"]
            paths = sorted({p for (k, p, m) in src if k == key})
            rec["n"] = len(paths)
            for meth, _ in COLS:
                vals = [src[(key, p, meth)]["ok"] == "1" for p in paths if (key, p, meth) in src]
                if vals:
                    rec["cols"].setdefault(meth, {})[s] = float(np.mean(vals))
        table[key] = rec

    lines = ["**Table 2. Mean accuracy over three paired classifier seeds (test split, 13-tool configuration).**",
             "Values are fractions; parenthesized values are population SDs across the three seeds.",
             "All columns come from the same 13-tool configuration and the same frozen classifier seeds.",
             "",
             "Scale: the test split holds 430 unique source images. Each of the six scenes degrades those same "
             "430 sources, giving 6 x 430 = 2,580 scene instances per seed; with seven methods per instance this "
             "yields 18,060 method records per seed (54,180 over three seeds).",
             "",
             "| Scene | " + " | ".join(n for _, n in COLS) + " | Agent − raw |",
             "|---|" + "---:|" * (len(COLS) + 1)]
    summary = {}
    for key, nice in SCENES:
        rec = table[key]
        cells, means = [], {}
        for meth, _ in COLS:
            d = rec["cols"].get(meth) or {}
            v = [d[s] for s in SEEDS if s in d]
            if not v:
                cells.append("—")
                continue
            means[meth] = float(np.mean(v))
            sd = float(np.std(v, ddof=1)) if len(v) > 1 else 0.0
            cells.append(f"{means[meth]:.4f} ({sd:.4f})")
            summary.setdefault(key, {})[meth] = {"mean": means[meth], "sd": sd, "vals": v}
        delta = means["iterative"] - means["raw"]
        summary[key]["delta_vs_raw"] = delta
        lines.append(f"| {nice} | " + " | ".join(cells) + f" | {delta:+.4f} |")
    lines.append("")
    lines.append("Agent − raw: " + "; ".join(
        f"{nice} {summary[k]['delta_vs_raw']:+.4f}" for k, nice in SCENES) + ".")

    mean_all = float(np.mean([summary[k]["iterative"]["mean"] for k, _ in SCENES]))
    raw_all = float(np.mean([summary[k]["raw"]["mean"] for k, _ in SCENES]))
    lines.append("")
    lines.append(f"Six-scene mean: raw {raw_all:.4f}, learned iterative agent {mean_all:.4f} "
                 f"({mean_all - raw_all:+.4f}).")
    open(args.out + ".md", "w", encoding="utf-8").write("\n".join(lines) + "\n")
    json.dump({"n_per_scene": table["clean"]["n"], "scenes": summary,
               "six_scene_mean": {"raw": raw_all, "iterative": mean_all}},
              open(args.out + ".json", "w", encoding="utf-8"), ensure_ascii=False, indent=2)
    print("\n".join(lines))
    print(f"\n写出 {args.out}.md / .json")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
