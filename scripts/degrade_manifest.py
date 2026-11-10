"""按 manifest 生成退化训练数据。

对给定的 (路径, 标签) 清单，对每张图施加退化（低光 / 噪声），
输出退化图 + 新的**五列**清单（degraded_path / label / source_id / source_path / params_json），
用于训练可学习 router 及跨退化对齐审计。

五列含义：
  - degraded_path：退化图路径（绝对）
  - label：类别标签
  - source_id：源图内容的稳定身份（内容 SHA-256 前 16 位），跨退化类型一致，供对齐
  - source_path：源图绝对路径，供溯源
  - params_json：退化参数（如 {"gamma": 3.0}）

用法：
    python scripts/degrade_manifest.py --manifest data/splits/train.tsv \
        --type lowlight --out-dir data/degraded/train_lowlight \
        --out-manifest data/degraded/train_lowlight.tsv
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

from scripts.degrade import apply_gamma, apply_gaussian


def _source_id(src_path: str) -> str:
    """源图内容的稳定身份（内容 SHA-256 前 16 位），跨退化类型一致。

    不用「路径 hash」：路径 hash 会因机器/目录移动而变，且无法反映源图内容；
    内容 hash 保证同一张源图在 lowlight/noise/motion_blur/jpeg/mixed 下 source_id 一致。
    """
    return _file_sha256(src_path)[:16]


def _file_sha256(path: str) -> str:
    h = hashlib.sha256()
    with open(path, "rb") as f:
        for chunk in iter(lambda: f.read(1 << 20), b""):
            h.update(chunk)
    return h.hexdigest()


def main() -> int:
    parser = argparse.ArgumentParser(description="按清单生成退化训练数据")
    parser.add_argument("--manifest", required=True, help="输入清单（路径\\t标签）")
    parser.add_argument("--type", required=True, choices=["lowlight", "noise"])
    parser.add_argument("--out-dir", required=True, help="退化图输出目录")
    parser.add_argument("--out-manifest", required=True, help="输出四列清单")
    parser.add_argument("--gamma", type=float, default=3.0)
    parser.add_argument("--sigma", type=float, default=25.0)
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--limit", type=int, default=0)
    args = parser.parse_args()

    rng = np.random.default_rng(args.seed)
    items = []
    with open(args.manifest, encoding="utf-8") as f:
        for line in f:
            line = line.rstrip("\n")
            if not line or line.startswith("#"):
                continue
            parts = line.split("\t")
            if len(parts) >= 2:
                items.append((parts[0], parts[1]))
    if args.limit > 0:
        items = items[: args.limit]

    os.makedirs(args.out_dir, exist_ok=True)
    rows = []
    n_fail = 0
    for i, (src, label) in enumerate(items, 1):
        img = cv2.imread(src, cv2.IMREAD_COLOR)
        if img is None:
            n_fail += 1
            continue
        if args.type == "lowlight":
            params = {"gamma": args.gamma}
            deg = apply_gamma(cv2.cvtColor(img, cv2.COLOR_BGR2RGB), args.gamma)
            deg = cv2.cvtColor(deg, cv2.COLOR_RGB2BGR)
        else:
            params = {"gaussian_sigma": args.sigma}
            deg = apply_gaussian(img, args.sigma, rng)
        out_fn = f"{label}_{i:05d}.png"
        out_path = os.path.join(args.out_dir, out_fn)
        cv2.imwrite(out_path, deg)
        rows.append((os.path.abspath(out_path), label,
                     _source_id(os.path.abspath(src)), os.path.abspath(src),
                     json.dumps(params, sort_keys=True)))

    with open(args.out_manifest, "w", encoding="utf-8") as f:
        f.write("# degraded_path\tlabel\tsource_id\tsource_path\tparams_json\n")
        for p, lb, sid, srcp, pj in rows:
            f.write(f"{p}\t{lb}\t{sid}\t{srcp}\t{pj}\n")

    # sidecar：记录 manifest 哈希与生成参数，供审计
    sidecar = args.out_manifest + ".json"
    with open(sidecar, "w", encoding="utf-8") as f:
        json.dump({
            "manifest": os.path.abspath(args.out_manifest),
            "manifest_sha256": _file_sha256(args.out_manifest),
            "source_manifest": os.path.abspath(args.manifest),
            "source_manifest_sha256": _file_sha256(args.manifest),
            "degradation_type": args.type,
            "gamma": args.gamma,
            "sigma": args.sigma,
            "seed": args.seed,
            "n_generated": len(rows),
            "n_fail": n_fail,
        }, f, ensure_ascii=False, indent=2)

    print(f"生成 {len(rows)} 张退化图 -> {args.out_dir}（失败 {n_fail}）")
    print(f"清单（五列，含 source_id/source_path/params_json）-> {args.out_manifest}")
    print(f"sidecar 溯源 -> {sidecar}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
