"""生成两条强度曲线（新主线：learned 循环 Agent）。

数据来源（同一冻结工具/分类器，三个 seed）：
  raw / fixed / legacy  ← results/legacy16_eval/seed{s}/{manifest}/per_image.csv
                          （raw_ok / audit_ok / chosen_ok；audit_ok 即该清单的固定工具）
  iterative             ← results/iterative_strength/seed{s}/{manifest}/summary.json
                          （生产脚本 eval_iterative_router.py 实跑，非 replay）

输出：results/final_figures/{noise_strength_curve.png, lowlight_strength_curve.png, strength_curves.json}
"""
from __future__ import annotations

import csv
import json
import os

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt  # noqa: E402
import numpy as np  # noqa: E402

plt.rcParams["font.family"] = "Arial"
plt.rcParams["ps.fonttype"] = 42
plt.rcParams["pdf.fonttype"] = 42

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
OUT = os.path.join(ROOT, "results", "final_figures")
SEEDS = (1, 2, 3)
NOISE = [("noise_s15", 15), ("noise_s25", 25), ("noise_s50", 50)]
LOWLIGHT = [("lowlight_s2", 2.0), ("lowlight_s2.5", 2.5),
            ("lowlight_s3", 3.0), ("lowlight_s3.5", 3.5)]

# 形状严格区分：圆=raw、方=固定工具、三角=legacy、菱形=循环
STYLE = {
    "raw": ("Raw only", "#8c8c8c", "o", "-"),
    "fixed": ("Fixed restoration", "#2e9e5b", "s", "-"),
    "legacy": ("Legacy rule agent (16-tool pool)", "#d9822b", "^", "-"),
    "iterative": ("Learned iterative agent (13 tools)", "#2f6fb5", "D", "-"),
}


def unit_metrics(seed, manifest):
    rows = list(csv.DictReader(open(
        os.path.join(ROOT, "results", "legacy16_eval", f"seed{seed}", manifest, "per_image.csv"),
        encoding="utf-8")))
    it = json.load(open(
        os.path.join(ROOT, "results", "iterative_strength_13tool", f"seed{seed}", manifest, "summary.json"),
        encoding="utf-8"))
    n = len(rows)
    return {
        "n": n,
        "raw": sum(int(r["raw_ok"]) for r in rows) / n,
        "fixed": sum(int(r["audit_ok"]) for r in rows) / n,
        "legacy": sum(int(r["chosen_ok"]) for r in rows) / n,
        "iterative": it["accuracy"],
        "iterative_calls": it["mean_restore_calls"],
        "legacy_calls": sum(int(r["n_restore_calls"]) for r in rows) / n,
    }


def collect(points):
    out = []
    for name, x in points:
        per = {s: unit_metrics(s, name) for s in SEEDS}
        rec = {"manifest": name, "x": x, "n": per[1]["n"]}
        for k in ("raw", "fixed", "legacy", "iterative", "iterative_calls", "legacy_calls"):
            vals = [per[s][k] for s in SEEDS]
            rec[k] = {"mean": float(np.mean(vals)), "std": float(np.std(vals, ddof=1)), "vals": vals}
        out.append(rec)
    return out


def save_fig(fig, stem):
    """Springer：矢量 EPS/PDF 优先；位图备 600 dpi RGB。"""
    from PIL import Image

    fig.savefig(stem + ".pdf", facecolor="white")
    fig.savefig(stem + ".eps", facecolor="white")
    png = stem + ".png"
    fig.savefig(png, dpi=600, facecolor="white")
    with Image.open(png) as im:
        im.convert("RGB").save(png, dpi=(600, 600))


def plot(series, xlabel, title, path, keys=("raw", "fixed", "iterative")):
    """单栏 84 mm；无图内标题（Springer 要求标题只出现在图注）。"""
    fig, ax = plt.subplots(figsize=(84 / 25.4, 78 / 25.4), dpi=200)
    xs = [r["x"] for r in series]
    short = {"raw": "Raw", "fixed": "Fixed", "legacy": "Legacy", "iterative": "Agent"}
    for key in keys:
        label, color, marker, ls = STYLE[key]
        label = short.get(key, label)
        mu = np.array([r[key]["mean"] for r in series])
        # 只画均值线（SD 另见伴随表格与图注）
        ax.plot(xs, mu, label=label, color=color, linestyle=ls,
                linewidth=1.0, marker=marker, markersize=4.5,
                markerfacecolor="white", markeredgecolor=color,
                markeredgewidth=1.0, zorder=3)
    ax.set_xlabel(xlabel, fontsize=8)
    ax.set_ylabel("Accuracy", fontsize=8)
    ax.set_xticks(xs)
    ax.tick_params(labelsize=8, width=0.7, length=2.8)
    ax.grid(True, color="#e6e6e6", linewidth=0.5, zorder=0)
    ax.set_axisbelow(True)
    for spine in ax.spines.values():
        spine.set_linewidth(0.6)
        spine.set_color("#333333")
    ax.legend(fontsize=8, loc="upper center", bbox_to_anchor=(0.5, -0.22),
              ncol=len(keys) if len(keys) <= 3 else 2, frameon=False,
              handlelength=1.5, columnspacing=0.9, handletextpad=0.35,
              labelspacing=0.3)
    fig.tight_layout(pad=0.3)
    stem = path
    for ext in (".png", ".pdf", ".eps"):
        if stem.lower().endswith(ext):
            stem = stem[: -len(ext)]
    save_fig(fig, stem)
    plt.close(fig)


def sd_range(series):
    vals = [r[k]["std"] for r in series for k in ("raw", "fixed", "legacy", "iterative")]
    return min(vals), max(vals)


def write_tables(noise, low):
    """伴随表格：均值 ± SD（图里不再画误差线）。"""
    lines = ["# Strength sweeps: mean ± SD over three classifier seeds", "",
             "Error bars are omitted from the figures; dispersion is reported here.",
             "",
             "† The legacy rule agent is reported with its original 16-tool candidate pool. "
             "It is excluded from the main figures and shown only in the supplementary panels, "
             "because its candidate pool differs from the 13-tool learned agent.", ""]
    for title, series, unit in (("Gaussian noise", noise, "sigma"),
                                ("Synthetic low light", low, "gamma")):
        lines += [f"## {title} (430 sources per point)", "",
                  f"| {unit} | Raw only | Fixed restoration | Legacy rule agent † | Learned iterative agent (13 tools) |",
                  "|---:|---:|---:|---:|---:|"]
        for r in series:
            cells = []
            for k in ("raw", "fixed", "legacy", "iterative"):
                cells.append(f"{r[k]['mean']:.4f} ± {r[k]['std']:.4f}")
            lines.append(f"| {r['x']:g} | " + " | ".join(cells) + " |")
        lo, hi = sd_range(series)
        lines.append("")
        lines.append(f"Seed SD range: {lo:.4f}–{hi:.4f}.")
        lines.append("")
    open(os.path.join(OUT, "strength_curves_table.md"), "w", encoding="utf-8").write("\n".join(lines) + "\n")


def main() -> int:
    os.makedirs(OUT, exist_ok=True)
    noise = collect(NOISE)
    low = collect(LOWLIGHT)

    # 主图：只放候选池一致的 Raw / Fixed / Learned(13 tools)
    plot(noise, r"Noise level $\sigma$", "",
         os.path.join(OUT, "noise_strength_curve.png"), keys=("raw", "fixed", "iterative"))
    plot(low, "Gamma", "",
         os.path.join(OUT, "lowlight_strength_curve.png"), keys=("raw", "fixed", "iterative"))
    # 补充材料：含 legacy（其候选池为 16 工具，与主图不可直接比较）
    plot(noise, r"Noise level $\sigma$", "",
         os.path.join(OUT, "supp_noise_with_legacy.png"),
         keys=("raw", "fixed", "legacy", "iterative"))
    plot(low, "Gamma", "",
         os.path.join(OUT, "supp_lowlight_with_legacy.png"),
         keys=("raw", "fixed", "legacy", "iterative"))

    write_tables(noise, low)
    json.dump({"noise": noise, "lowlight": low,
               "note": "raw/fixed/legacy 来自 legacy16_eval（同一审计分支）；iterative 来自 iterative_strength"},
              open(os.path.join(OUT, "strength_curves.json"), "w", encoding="utf-8"),
              ensure_ascii=False, indent=2)

    print(f"{'强度':<14}{'raw':>8}{'fixed':>8}{'legacy':>8}{'iterative':>11}{'循环调用':>9}")
    for grp in (noise, low):
        for r in grp:
            print(f"{r['manifest']:<14}{r['raw']['mean']:>8.4f}{r['fixed']['mean']:>8.4f}"
                  f"{r['legacy']['mean']:>8.4f}{r['iterative']['mean']:>11.4f}"
                  f"{r['iterative_calls']['mean']:>9.2f}")
    print(f"\n写出 {OUT}/*.png 与 strength_curves.json")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
