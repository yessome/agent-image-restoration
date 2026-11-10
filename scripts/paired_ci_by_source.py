"""按源图成组的配对区间（13 工具，test 六场景 × 三 seed）。

聚类单位 = 源图（430 张 TrashNet 原图）。每个源图把自己的 6 个退化版本 × 3 个 seed
一起带入或一起带出重采样，因此不把同一张图的不同退化当作独立样本。

两个结果目录用不同的 source_id 体系（scene_matrix 用原图哈希，test_matrix 用退化集
哈希），这里统一按 manifest 里的原图文件名对齐。

输出：差值点估计 + 95% 百分位 bootstrap 区间。不输出"非劣"结论——非劣效需要预先
设定的界值（margin）和对应的单侧检验，本脚本不提供。
"""
from __future__ import annotations

import csv
import json
import os
import random
from collections import defaultdict

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
SCENE_DIR = os.path.join(ROOT, "results", "scene_matrix_13tool")
PAIR_DIR = os.path.join(ROOT, "results", "test_matrix_13tool")
CLEAN_Tsv = os.path.join(ROOT, "results", "table2_scenes", "clean.tsv")
PAIR_Tsv = os.path.join(ROOT, "results", "router_test_17branch", "test_lowlight.tsv")
OUT_MD = os.path.join(ROOT, "results", "final_figures", "paired_ci_13tool.md")
OUT_JSON = os.path.join(ROOT, "results", "final_figures", "paired_ci_13tool.json")

SEEDS = (1, 2, 3)
COMPARISONS = (("iterative", "raw"), ("iterative", "fixed_denoise"),
               ("iterative", "fixed_lowlight"))
CLASSES = ("cardboard", "glass", "metal", "paper", "plastic", "trash")
N_BOOT = 10000
RNG_SEED = 20261101


def base(path: str) -> str:
    return path.replace("\\", "/").split("/")[-1].lower()


def uid_map() -> dict:
    """(dir_tag, group) -> 原图文件名"""
    m = {}
    with open(CLEAN_Tsv, encoding="utf-8") as fh:
        rd = csv.DictReader(fh, delimiter="\t")
        rd.fieldnames = [f.lstrip("#").strip() for f in rd.fieldnames]
        for r in rd:
            m[("scene", r["source_id"])] = base(r["path"])
    with open(PAIR_Tsv, encoding="utf-8") as fh:
        rd = csv.DictReader(fh, delimiter="\t")
        rd.fieldnames = [f.lstrip("#").strip() for f in rd.fieldnames]
        for r in rd:
            m[("test", r["source_id"])] = base(r["source_path"])
    return m


def load():
    """返回 rec[(method, uid, subset, seed)] -> ok，以及原始行（用于指标核对）"""
    umap = uid_map()
    rec, rows = {}, defaultdict(list)
    for seed in SEEDS:
        for tag, d in (("scene", SCENE_DIR), ("test", PAIR_DIR)):
            path = os.path.join(d, f"seed{seed}", "per_image.csv")
            with open(path, encoding="utf-8") as fh:
                for r in csv.DictReader(fh):
                    uid = umap[(tag, r["group"])]
                    rec[(r["method"], uid, r["subset"], seed)] = int(r["ok"])
                    rows[(tag, seed, r["method"])].append((r["label"], r["pred"],
                                                           int(r["ok"])))
    return rec, rows


def macro_f1(pairs) -> float:
    tp, fp, fn = defaultdict(int), defaultdict(int), defaultdict(int)
    for y, p in pairs:
        if y == p:
            tp[y] += 1
        else:
            fp[p] += 1
            fn[y] += 1
    out = []
    for c in CLASSES:
        d = 2 * tp[c] + fp[c] + fn[c]
        out.append(0.0 if d == 0 else 2 * tp[c] / d)
    return sum(out) / len(out)


def sd(xs):
    m = sum(xs) / len(xs)
    return (sum((x - m) ** 2 for x in xs) / len(xs)) ** 0.5


def metrics(rows):
    """两族(860) 与 六场景(2580) 两套口径"""
    out = {"two_family": {}, "six_scene": {}}
    for scope, tags in (("two_family", ("test",)), ("six_scene", ("scene", "test"))):
        for m in ("raw", "iterative"):
            accs, f1s = [], []
            for seed in SEEDS:
                pool = []
                for tag in tags:
                    pool.extend(rows[(tag, seed, m)])
                accs.append(sum(x[2] for x in pool) / len(pool))
                f1s.append(macro_f1([(x[0], x[1]) for x in pool]))
            out[scope][m] = {"acc_mean": sum(accs) / 3, "acc_sd": sd(accs),
                             "f1_mean": sum(f1s) / 3, "f1_sd": sd(f1s),
                             "per_seed_f1": f1s}
    return out


def main() -> int:
    rec, rows = load()
    keys = sorted({(u, s, sd_) for (_, u, s, sd_) in rec})
    uids = sorted({u for u, _, _ in keys})
    per_uid = defaultdict(int)
    for u, _, _ in keys:
        per_uid[u] += 1
    print(f"配对键 {len(keys)}；源图 {len(uids)}；每源图记录数 "
          f"{sorted(set(per_uid.values()))}")

    met = metrics(rows)
    print("两族 860：raw acc %.4f  F1 %.4f | iterative acc %.4f  F1 %.4f"
          % (met["two_family"]["raw"]["acc_mean"], met["two_family"]["raw"]["f1_mean"],
             met["two_family"]["iterative"]["acc_mean"],
             met["two_family"]["iterative"]["f1_mean"]))
    print("六场景 2580：raw acc %.4f  F1 %.4f | iterative acc %.4f  F1 %.4f"
          % (met["six_scene"]["raw"]["acc_mean"], met["six_scene"]["raw"]["f1_mean"],
             met["six_scene"]["iterative"]["acc_mean"],
             met["six_scene"]["iterative"]["f1_mean"]))

    gidx = {u: i for i, u in enumerate(uids)}
    rng = random.Random(RNG_SEED)

    def contrast(a, b, keep, n_boot=N_BOOT):
        sa, sb, n = [0.0] * len(uids), [0.0] * len(uids), [0] * len(uids)
        for (u, sub, sd_) in keys:
            if not keep(sub):
                continue
            i = gidx[u]
            sa[i] += rec[(a, u, sub, sd_)]
            sb[i] += rec[(b, u, sub, sd_)]
            n[i] += 1
        tot = sum(n)
        point = (sum(sa) - sum(sb)) / tot
        boot = []
        for _ in range(n_boot):
            da = db = dn = 0
            for _ in range(len(uids)):
                i = rng.randrange(len(uids))
                da += sa[i]
                db += sb[i]
                dn += n[i]
            boot.append((da - db) / dn)
        boot.sort()
        lo, hi = boot[int(0.025 * n_boot)], boot[int(0.975 * n_boot) - 1]
        return point, lo, hi, tot, ((lo > 0) or (hi < 0))

    contrasts = []
    for scope, keep in (("six_scene", lambda s: True),
                        ("two_family", lambda s: s in ("noise", "lowlight"))):
        for a, b in COMPARISONS:
            point, lo, hi, tot, sig = contrast(a, b, keep)
            contrasts.append({"scope": scope, "method_a": a, "method_b": b,
                              "diff": point, "ci95": [lo, hi], "n_records": tot,
                              "n_sources": len(uids), "boot": N_BOOT,
                              "ci_excludes_zero": sig})
            print(f"[{scope:10s}] {a} - {b}: {point*100:+.2f}pp  "
                  f"CI [{lo*100:+.2f}, {hi*100:+.2f}] "
                  f"{'CI 不含 0' if sig else 'CI 含 0'}")

    per_scene = []
    for a, b in COMPARISONS:
        for scene in sorted({s for (_, s, _) in keys}):
            point, lo, hi, _, _ = contrast(a, b, lambda s, sc=scene: s == sc,
                                           n_boot=2000)
            per_scene.append({"method_a": a, "method_b": b, "scene": scene,
                              "diff": point, "ci95": [lo, hi]})

    payload = {"contrasts": contrasts, "per_scene": per_scene, "metrics": met,
               "protocol": {"cluster": "source image (430 TrashNet sources)",
                            "records_per_source": "6 scenes x 3 seeds = 18",
                            "bootstrap": N_BOOT, "rng_seed": RNG_SEED,
                            "ci": "percentile 2.5-97.5"}}
    with open(OUT_JSON, "w", encoding="utf-8") as fh:
        json.dump(payload, fh, indent=1)
    write_md(payload)
    print(f"写出 {OUT_MD}")
    return 0


def write_md(p):
    L = ["**Paired contrasts, 13-tool configuration.**", ""]
    L.append("Each source image is one cluster: its degraded versions and three "
             "classifier seeds enter and leave the resampling together. Intervals are "
             "percentile bootstrap 95% CIs over 10,000 resamples of the 430 sources.")
    L.append("")
    for scope, title, nrec in (
            ("six_scene", "Six-scene matrix (430 sources x 6 scenes x 3 seeds, 7,740 records)", 7740),
            ("two_family", "Two-family matrix (430 sources x 2 scenes x 3 seeds, 2,580 records)", 2580)):
        L.append(f"**{title}**")
        L.append("")
        L.append("| Contrast | Delta (pp) | 95% CI (pp) |")
        L.append("|---|---:|---|")
        for r in p["contrasts"]:
            if r["scope"] != scope:
                continue
            L.append(f"| learned - {r['method_b'].replace('_', ' ')} | "
                     f"{r['diff']*100:+.2f} | [{r['ci95'][0]*100:+.2f}, "
                     f"{r['ci95'][1]*100:+.2f}] |")
        L.append("")
    L.append("| Scope | Metric | Raw | Learned iterative agent |")
    L.append("|---|---|---:|---:|")
    for scope, label in (("two_family", "Two-family matrix (860 instances)"),
                         ("six_scene", "Six-scene matrix (2,580 instances)")):
        m = p["metrics"][scope]
        L.append(f"| {label} | Accuracy | {m['raw']['acc_mean']:.4f} | "
                 f"{m['iterative']['acc_mean']:.4f} |")
        L.append(f"| {label} | Macro-F1 | {m['raw']['f1_mean']:.4f} | "
                 f"{m['iterative']['f1_mean']:.4f} |")
    L.append("")
    with open(OUT_MD, "w", encoding="utf-8") as fh:
        fh.write("\n".join(L))


if __name__ == "__main__":
    raise SystemExit(main())
