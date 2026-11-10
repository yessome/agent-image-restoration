"""退化增强分类器对照：直接用三视图训练的分类器分类，不经过修复与路由。

回答审稿人最可能问的问题：与其调度修复工具，为什么不直接把分类器在退化视图上训练。
输入与六场景 test 主表完全一致（430 源图 × 6 场景），只换分类器权重。
"""
from __future__ import annotations

import argparse
import csv
import json
import sys
from collections import Counter, defaultdict
from pathlib import Path

import cv2
import numpy as np

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
from src.downstream.classify import ClassifyProcessor  # noqa: E402

DEG = ROOT / "data" / "degraded"
DEG_TEST = ROOT / "data" / "degraded_test_13tool"
OUT = ROOT / "results" / "ablation" / "robust_classifier_baseline.json"


def uid(p: str) -> str:
    return p.replace("\\", "/").split("/")[-1].lower()


def items():
    out = []
    with open(ROOT / "results" / "table2_scenes" / "clean.tsv", encoding="utf-8") as fh:
        for r in csv.DictReader(fh, delimiter="\t"):
            k = [x.lstrip("#").strip() for x in r]
            out.append({"path": r.get("# path") or r.get("path"), "label": r["label"],
                        "subset": "clean"})
    for sub in ("jpeg", "motion_blur", "mixed"):
        with open(DEG / sub / "manifest_full.tsv", encoding="utf-8") as fh:
            rd = csv.reader(fh, delimiter="\t")
            next(rd)
            for row in rd:
                out.append({"path": row[0], "label": row[1], "subset": sub})
    for sub in ("lowlight", "noise"):
        with open(DEG_TEST / sub / "manifest_full.tsv", encoding="utf-8") as fh:
            rd = csv.reader(fh, delimiter="\t")
            next(rd)
            for row in rd:
                out.append({"path": row[0], "label": row[1], "subset": sub})
    return out


def main() -> int:
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--device", default="cuda")
    ap.add_argument("--seeds", default="1,2,3")
    args = ap.parse_args()

    rows = items()
    print(f"样本 {len(rows)}（六场景）")
    payload = {"n": len(rows), "per_seed": {}}
    for seed in [int(s) for s in args.seeds.split(",")]:
        w = ROOT / "weights" / f"robust_resnet50_seed{seed}.pth"
        if not w.exists():
            raise FileNotFoundError(w)
        clf = ClassifyProcessor(device=args.device, weight_path=str(w))
        clf.load()
        ok, by = 0, defaultdict(lambda: [0, 0])
        for i, it in enumerate(rows, 1):
            img = cv2.cvtColor(cv2.imread(it["path"], cv2.IMREAD_COLOR), cv2.COLOR_BGR2RGB)
            res = clf.process(img)
            pred = next(iter(res.result)) if isinstance(res.result, dict) else str(res.result)
            hit = int(pred == it["label"])
            ok += hit
            by[it["subset"]][0] += hit
            by[it["subset"]][1] += 1
            if i % 500 == 0:
                print(f"  seed{seed} {i}/{len(rows)}", flush=True)
        acc = ok / len(rows)
        payload["per_seed"][seed] = {
            "accuracy": acc,
            "by_scene": {k: v[0] / v[1] for k, v in by.items()},
        }
        print(f"seed{seed} acc={acc:.4f} " +
              " ".join(f"{k}={v[0]/v[1]:.4f}" for k, v in sorted(by.items())))

    accs = [v["accuracy"] for v in payload["per_seed"].values()]
    payload["mean_accuracy"] = float(np.mean(accs))
    payload["sd_accuracy"] = float(np.std(accs))
    OUT.parent.mkdir(parents=True, exist_ok=True)
    OUT.write_text(json.dumps(payload, ensure_ascii=False, indent=1), encoding="utf-8")
    print(f"mean {payload['mean_accuracy']:.4f} ± {payload['sd_accuracy']:.4f}")
    print(f"写出 {OUT}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
