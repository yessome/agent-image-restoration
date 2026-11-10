"""六场景 test 的 13 工具消融报告（call_limit_1 / no_feedback）。

Full 复用既有记录；两个条件的复用部分与被重跑部分合并统计。
配对区间按 430 张源图聚类（源图的 6 个场景 × 3 个 seed 一起进出重采样）。
stop_reason 只对重跑子集报告，并注明分母；复用记录没有该字段，标为未知。
"""
from __future__ import annotations

import csv
import json
import os
import random
from collections import Counter, defaultdict
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
ABL = ROOT / "results" / "ablation_13tool_test"
OUT_MD = ROOT / "results" / "final_figures" / "table7_ablation_test_13tool.md"
OUT_JSON = ROOT / "results" / "final_figures" / "table7_ablation_test_13tool.json"
SEEDS = (1, 2, 3)
CONDS = ("call_limit_1", "no_feedback")
LABEL = {"full": "Full agent", "call_limit_1": "At most one call",
         "no_feedback": "No branch feedback"}
DIRS = ("scene_matrix_13tool", "test_matrix_13tool")
N_BOOT = 10000
RNG = 20261101
SCENES = ("clean", "noise", "lowlight", "motion_blur", "jpeg", "mixed")


def uid(p: str) -> str:
    return p.replace("\\", "/").split("/")[-1].lower()


def load_full(seed):
    out = {}
    for d in DIRS:
        with open(ROOT / "results" / d / f"seed{seed}" / "per_image.csv", encoding="utf-8") as fh:
            for r in csv.DictReader(fh):
                if r["method"] != "iterative":
                    continue
                out[(r["subset"], uid(r["path"]))] = {
                    "ok": int(r["ok"]), "raw_ok": int(r["raw_ok"]),
                    "calls": int(r["restore_calls"])}
    return out


def load_cond(seed, cond):
    p = ABL / f"seed{seed}" / f"{cond}.jsonl"
    rows = {}
    reasons, n_rerun = Counter(), 0
    for line in p.read_text(encoding="utf-8").splitlines():
        if not line.strip():
            continue
        r = json.loads(line)
        key = (r["subset"], uid(r["path"]))
        rows[key] = {"ok": r["ok"], "raw_ok": r["raw_ok"], "calls": r["restore_calls"],
                     "source": r["source"]}
        if r["source"] == "rerun":
            n_rerun += 1
            reasons[r["stop_reason"]] += 1
    return rows, reasons, n_rerun


def boot(diffs, n_boot=N_BOOT):
    rng = random.Random(RNG)
    m = len(diffs)
    point = sum(diffs) / m
    vals = []
    for _ in range(n_boot):
        s = 0.0
        for _ in range(m):
            s += diffs[rng.randrange(m)]
        vals.append(s / m)
    vals.sort()
    return point, vals[int(0.025 * n_boot)], vals[int(0.975 * n_boot) - 1]


def main() -> int:
    full = {s: load_full(s) for s in SEEDS}
    conds = {(s, c): load_cond(s, c) for s in SEEDS for c in CONDS}
    keys = sorted(full[1])
    assert all(len(full[s]) == len(keys) == 2580 for s in SEEDS), "Full 记录数不一致"

    # 源图分组：消融记录里 group 是原图文件名，Full 的 path 是退化图文件名，
    # 两者通过 (subset, 退化图名) 对应；这里以原图名为聚类单位。
    key2group = {}
    for line in (ABL / "seed1" / "call_limit_1.jsonl").read_text(encoding="utf-8").splitlines():
        if line.strip():
            r = json.loads(line)
            key2group[(r["subset"], uid(r["path"]))] = r["group"]
    missing = [k for k in keys if k not in key2group]
    if missing:
        raise RuntimeError(f"{len(missing)} 条 Full 记录没有对应的源图分组")
    groups = defaultdict(list)
    for k, gname in key2group.items():
        groups[gname].append(k)
    sizes = Counter(len(v) for v in groups.values())
    print(f"源图 {len(groups)} 个，每源记录数分布 {dict(sizes)}")

    per_seed = {}
    for s in SEEDS:
        per_seed[s] = {}
        for tag in ("full",) + CONDS:
            rec = full[s] if tag == "full" else conds[(s, tag)][0]
            n = len(rec)
            per_seed[s][tag] = {
                "acc": sum(v["ok"] for v in rec.values()) / n,
                "raw_acc": sum(v["raw_ok"] for v in rec.values()) / n,
                "calls": sum(v["calls"] for v in rec.values()) / n,
                "fixes": sum(1 for v in rec.values() if not v["raw_ok"] and v["ok"]),
                "harms": sum(1 for v in rec.values() if v["raw_ok"] and not v["ok"]),
            }

    # 配对区间：同一源图的全部记录一起重采样
    contrasts = {}
    for c in CONDS:
        diffs = []
        for name, members in groups.items():
            a = b = n = 0
            for key in members:
                for s in SEEDS:
                    b += full[s][key]["ok"]
                    a += conds[(s, c)][0][key]["ok"]
                    n += 1
            diffs.append((a - b) / n)
        point, lo, hi = boot(diffs)
        contrasts[c] = {"diff": point, "ci95": [lo, hi], "n_sources": len(diffs),
                        "records": sum(len(v) for v in groups.values()) * len(SEEDS)}

    # 多次调用子集：Full 实际调用 >= 2
    multi = defaultdict(dict)
    for s in SEEDS:
        for key, v in full[s].items():
            if v["calls"] >= 2:
                multi[s][key] = v
    multi_rows = []
    for c in CONDS:
        for s in SEEDS:
            items = multi[s]
            if not items:
                continue
            a_ok = sum(conds[(s, c)][0][k]["ok"] for k in items)
            f_ok = sum(v["ok"] for v in items.values())
            a_calls = sum(conds[(s, c)][0][k]["calls"] for k in items)
            f_calls = sum(v["calls"] for v in items.values())
            multi_rows.append({
                "condition": c, "seed": s, "n": len(items),
                "full_acc": f_ok / len(items), "cond_acc": a_ok / len(items),
                "diff": (a_ok - f_ok) / len(items),
                "full_calls": f_calls / len(items), "cond_calls": a_calls / len(items),
                "extra_calls": (f_calls - a_calls) / len(items)})
    by_scene = []
    for c in CONDS:
        for sc in SCENES:
            tot_n = tot_f = tot_c = 0
            for s in SEEDS:
                items = {k: v for k, v in multi[s].items() if k[0] == sc}
                if not items:
                    continue
                tot_n += len(items)
                tot_f += sum(v["ok"] for v in items.values())
                tot_c += sum(conds[(s, c)][0][k]["ok"] for k in items)
            if tot_n:
                by_scene.append({"condition": c, "scene": sc, "n": tot_n,
                                 "full_acc": tot_f / tot_n, "cond_acc": tot_c / tot_n,
                                 "diff": (tot_c - tot_f) / tot_n})

    write_md(per_seed, contrasts, conds, multi_rows, by_scene, groups)
    payload = {"per_seed": per_seed, "contrasts": contrasts,
               "multi_call": multi_rows, "multi_call_by_scene": by_scene,
               "stop_reasons": {f"seed{s}_{c}": {"n_rerun": conds[(s, c)][2],
                                                 "counts": dict(conds[(s, c)][1])}
                                for s in SEEDS for c in CONDS}}
    OUT_JSON.write_text(json.dumps(payload, ensure_ascii=False, indent=1), encoding="utf-8")
    print(f"写出 {OUT_MD}")
    return 0


def ms(vals):
    m = sum(vals) / len(vals)
    sd = (sum((x - m) ** 2 for x in vals) / len(vals)) ** 0.5
    return m, sd


def write_md(per_seed, contrasts, conds, multi_rows, by_scene, groups):
    L = ["**Table 7. Mechanism ablation on the six-scene test matrix (13 tools, 430 sources "
         "x 6 scenes x 3 seeds).**", ""]
    L.append("Full rows reuse the frozen main-run records; the two ablations change one factor "
             "each on the same checkpoint. The one-call condition keeps the original budget "
             "encoding and only truncates execution after the first call, so it is a mechanism "
             "ablation rather than an optimised single-step policy.")
    L.append("")
    L.append("### Per-seed")
    L.append("")
    L.append("| Seed | Condition | Accuracy | Fixes | Harms | Mean calls |")
    L.append("|---:|---|---:|---:|---:|---:|")
    for s in SEEDS:
        for tag in ("full",) + CONDS:
            d = per_seed[s][tag]
            L.append(f"| {s} | {LABEL[tag]} | {d['acc']:.4f} | {d['fixes']} | {d['harms']} | "
                     f"{d['calls']:.3f} |")
    L.append("")
    L.append("### Three-seed summary (mean ± population SD)")
    L.append("")
    L.append("| Condition | Accuracy | Delta vs full | Fixes | Harms | Mean calls |")
    L.append("|---|---:|---:|---:|---:|---:|")
    base = ms([per_seed[s]["full"]["acc"] for s in SEEDS])
    for tag in ("full",) + CONDS:
        acc = ms([per_seed[s][tag]["acc"] for s in SEEDS])
        fx = ms([per_seed[s][tag]["fixes"] for s in SEEDS])
        hm = ms([per_seed[s][tag]["harms"] for s in SEEDS])
        cl = ms([per_seed[s][tag]["calls"] for s in SEEDS])
        delta = f"{(acc[0]-base[0])*100:+.2f} pp" if tag != "full" else "—"
        L.append(f"| {LABEL[tag]} | {acc[0]:.4f} ± {acc[1]:.4f} | {delta} | "
                 f"{fx[0]:.1f} ± {fx[1]:.1f} | {hm[0]:.1f} ± {hm[1]:.1f} | {cl[0]:.3f} ± {cl[1]:.3f} |")
    L.append("")
    L.append("### Paired contrasts against Full (clustered by source image)")
    L.append("")
    L.append("Each source carries its six scenes and three seeds into every resample; "
             "10,000 percentile bootstrap draws over the 430 sources.")
    L.append("")
    L.append("| Contrast | Delta (pp) | 95% CI (pp) | Sources | Records |")
    L.append("|---|---:|---|---:|---:|")
    for c in CONDS:
        d = contrasts[c]
        L.append(f"| {LABEL[c]} − Full | {d['diff']*100:+.2f} | "
                 f"[{d['ci95'][0]*100:+.2f}, {d['ci95'][1]*100:+.2f}] | "
                 f"{d['n_sources']} | {d['records']} |")
    L.append("")
    L.append("### Stop reasons (rerun subset only)")
    L.append("")
    L.append("Reused records come from the frozen main run, which does not log a stop reason; "
             "they are excluded from this table rather than imputed. Denominators are the number "
             "of rerun records, not the full 2,580.")
    L.append("")
    L.append("| Seed | Condition | n rerun / 2580 | Distribution |")
    L.append("|---:|---|---:|---|")
    for s in SEEDS:
        for c in CONDS:
            rows, reasons, n_rerun = conds[(s, c)]
            dist = ", ".join(f"{k}: {v}" for k, v in sorted(reasons.items(), key=lambda x: -x[1]))
            L.append(f"| {s} | {LABEL[c]} | {n_rerun} / 2580 | {dist} |")
    L.append("")
    L.append("### Exploratory: images on which the full agent called at least twice")
    L.append("")
    L.append("Restricted to the 294 source-scene records (3.8%) where Full executed 2–4 tools. "
             "This is a post-hoc subgroup defined by the agent's own behaviour, not a "
             "pre-registered condition.")
    L.append("")
    L.append("| Condition | Seed | n | Full acc | Ablated acc | Delta | Extra calls removed |")
    L.append("|---|---:|---:|---:|---:|---:|---:|")
    for r in multi_rows:
        L.append(f"| {LABEL[r['condition']]} | {r['seed']} | {r['n']} | {r['full_acc']:.4f} | "
                 f"{r['cond_acc']:.4f} | {r['diff']*100:+.2f} pp | {r['extra_calls']:.2f} |")
    L.append("")
    L.append("| Condition | Scene | n | Full acc | Ablated acc | Delta |")
    L.append("|---|---|---:|---:|---:|---:|")
    for r in by_scene:
        L.append(f"| {LABEL[r['condition']]} | {r['scene']} | {r['n']} | {r['full_acc']:.4f} | "
                 f"{r['cond_acc']:.4f} | {r['diff']*100:+.2f} pp |")
    L.append("")
    OUT_MD.write_text("\n".join(L), encoding="utf-8")


if __name__ == "__main__":
    raise SystemExit(main())
