"""从 test.tsv 生成多强度高斯噪声测试集（σ=15/25/50），泛化测试用。

关键：σ=25 与旧版 make_test_degraded.py 的 noise 生成规则一致——
  - 每个 σ 用独立的 rng = np.random.default_rng(seed=42)（从初始状态开始，
    与旧版 noise 相同，因为旧版在 noise 之前的 lowlight 是确定性 gamma 不消耗 rng）；
  - apply_gaussian（加性高斯噪声）+ clip[0,255] + uint8 量化。

输出（全部 E 盘）：
  data/degraded/noise_s{15,25,50}/ 目录 + 五列 manifest_eval.tsv
  （degraded_path / label / source_id / source_path / params_json）

用法：
    python scripts/make_noise_multilevel.py --manifest data/splits/test.tsv --sigmas 15,25,50
"""
from __future__ import annotations

import argparse
import hashlib
import json
import os
import sys

import cv2
import numpy as np

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from scripts.degrade import apply_gaussian


def _file_sha256(path: str) -> str:
    h = hashlib.sha256()
    with open(path, "rb") as f:
        for chunk in iter(lambda: f.read(1 << 20), b""):
            h.update(chunk)
    return h.hexdigest()


def _source_id(src_path: str) -> str:
    return _file_sha256(src_path)[:16]


def read_manifest(path):
    items = []
    with open(path, encoding="utf-8") as f:
        for line in f:
            line = line.rstrip("\n")
            if not line or line.startswith("#"):
                continue
            parts = line.split("\t")
            if len(parts) >= 2:
                items.append((parts[0], parts[1]))
    return items


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--manifest", default="data/splits/test.tsv")
    ap.add_argument("--sigmas", default="15,25,50")
    ap.add_argument("--seed", type=int, default=42)
    ap.add_argument("--out-base", default="data/degraded")
    args = ap.parse_args()

    items = read_manifest(args.manifest)
    print(f"test 清单 {len(items)} 张")
    sigmas = [float(x) for x in args.sigmas.split(",")]

    for sigma in sigmas:
        rng = np.random.default_rng(args.seed)  # 每个 σ 独立 seed（与旧版 noise 一致）
        out_dir = os.path.join(args.out_base, f"noise_s{int(sigma)}")
        os.makedirs(out_dir, exist_ok=True)
        rows = []
        n_fail = 0
        for i, (src, label) in enumerate(items, 1):
            img = cv2.imread(src, cv2.IMREAD_COLOR)
            if img is None:
                n_fail += 1
                continue
            deg = apply_gaussian(img, sigma, rng)
            out_fn = f"{label}_{i:05d}.png"
            out_path = os.path.join(out_dir, out_fn)
            cv2.imwrite(out_path, deg)
            rows.append((os.path.abspath(out_path), label,
                         _source_id(os.path.abspath(src)), os.path.abspath(src),
                         json.dumps({"gaussian_sigma": sigma}, sort_keys=True)))
        # 五列清单
        manifest = os.path.join(out_dir, "manifest_eval.tsv")
        with open(manifest, "w", encoding="utf-8") as f:
            f.write("# degraded_path\tlabel\tsource_id\tsource_path\tparams_json\n")
            for d, lb, sid, srcp, pj in rows:
                f.write(f"{d}\t{lb}\t{sid}\t{srcp}\t{pj}\n")
        print(f"σ={int(sigma):>2} -> {len(rows)} 张，清单 {manifest}（失败 {n_fail}）")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
