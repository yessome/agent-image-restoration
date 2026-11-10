"""从 test.tsv 生成 test 退化集（低光 + 噪声，各 N 张）。

按清单（path<TAB>label）对每张干净图分别施加两种退化，输出：
  - 退化图：data/degraded/<type>/<label>_<i:05d>.png
  - 四列 full 清单：degraded_path / source_path / label / params(JSON)
    （供 eval_restoration_quality.py 计算 PSNR/SSIM）
  - 两列 eval 清单：degraded_path / label
    （供 evaluate.py 计算分类 Accuracy/Macro-F1）

低光用 gamma=3.0（噪声用高斯 sigma=25），固定 seed 可复现。
注：gamma 曾从 2.2 调至 3.0（代码注释记载 2.2 时低光识别率 7.9%、3.0 时 21.2%），
但该识别率来自 val 还是 test 未记录，无法证明预注册；论文应如实标注"合成低光，选参来源未记录"。

用法：
    python scripts/make_test_degraded.py --manifest data/splits/test.tsv --seed 42
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

from scripts.degrade import (apply_gamma, apply_gaussian, apply_motion_blur,
                             apply_jpeg, apply_mixed)


def _file_sha256(path: str) -> str:
    h = hashlib.sha256()
    with open(path, "rb") as f:
        for chunk in iter(lambda: f.read(1 << 20), b""):
            h.update(chunk)
    return h.hexdigest()


def _source_id(src_path: str) -> str:
    """源图内容的稳定身份（内容 SHA-256 前 16 位），跨退化类型一致，供对齐审计。"""
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


def make_set(items, out_dir, deg_type, rng, gamma=3.0, sigma=25.0):
    """生成一组退化图，返回 (full_rows, eval_rows)，均为五元组：
    (degraded_path, label, source_id, source_path, params_json)。"""
    os.makedirs(out_dir, exist_ok=True)
    full_rows, eval_rows = [], []
    for i, (src, label) in enumerate(items, 1):
        img = cv2.imread(src, cv2.IMREAD_COLOR)
        if img is None:
            print(f"跳过（读失败）: {src}")
            continue
        if deg_type == "lowlight":
            params = {"gamma": gamma}
            deg = apply_gamma(cv2.cvtColor(img, cv2.COLOR_BGR2RGB), gamma)
            deg = cv2.cvtColor(deg, cv2.COLOR_RGB2BGR)
        elif deg_type == "noise":
            params = {"gaussian_sigma": sigma}
            deg = apply_gaussian(img, sigma, rng)
        elif deg_type == "motion_blur":
            params = {"kernel_size": 15, "angle_deg": 45.0}
            deg = apply_motion_blur(img, kernel_size=15, angle_deg=45.0)
        elif deg_type == "jpeg":
            params = {"jpeg_quality": 30}
            deg = apply_jpeg(img, quality=30)
        elif deg_type == "mixed":
            params = {"order": "lowlight_then_gaussian", "gamma": 3.0, "sigma": 25.0}
            deg = apply_mixed(cv2.cvtColor(img, cv2.COLOR_BGR2RGB), rng)
            deg = cv2.cvtColor(deg, cv2.COLOR_RGB2BGR)
        else:
            raise ValueError(f"未知退化类型: {deg_type}")
        out_fn = f"{label}_{i:05d}.png"
        out_path = os.path.join(out_dir, out_fn)
        cv2.imwrite(out_path, deg)
        sid = _source_id(os.path.abspath(src))
        srcp = os.path.abspath(src)
        pj = json.dumps(params, sort_keys=True)
        full_rows.append((os.path.abspath(out_path), label, sid, srcp, pj))
        eval_rows.append((os.path.abspath(out_path), label, sid, srcp, pj))
    return full_rows, eval_rows


def write_full(path, rows):
    os.makedirs(os.path.dirname(path) or ".", exist_ok=True)
    with open(path, "w", encoding="utf-8") as f:
        f.write("# degraded_path\tlabel\tsource_id\tsource_path\tparams_json\n")
        for d, lb, sid, srcp, pj in rows:
            f.write(f"{d}\t{lb}\t{sid}\t{srcp}\t{pj}\n")
    print(f"  五列 full 清单 -> {path} ({len(rows)} 条)")


def write_eval(path, rows):
    """写五列 eval 清单：degraded_path / label / source_id / source_path / params_json。"""
    os.makedirs(os.path.dirname(path) or ".", exist_ok=True)
    with open(path, "w", encoding="utf-8") as f:
        f.write("# degraded_path\tlabel\tsource_id\tsource_path\tparams_json\n")
        for d, lb, sid, srcp, pj in rows:
            f.write(f"{d}\t{lb}\t{sid}\t{srcp}\t{pj}\n")
    print(f"  五列 eval 清单 -> {path} ({len(rows)} 条)")


def main() -> int:
    parser = argparse.ArgumentParser(description="从 test.tsv 生成 test 退化集")
    parser.add_argument("--manifest", required=True, help="test 清单（path\\tlabel）")
    parser.add_argument("--out-base", default="data/degraded", help="退化图根目录")
    parser.add_argument("--gamma", type=float, default=3.0)
    parser.add_argument("--sigma", type=float, default=25.0)
    parser.add_argument("--seed", type=int, default=42)
    args = parser.parse_args()

    items = read_manifest(args.manifest)
    print(f"test 清单 {len(items)} 张，开始生成退化集...")

    rng = np.random.default_rng(args.seed)

    # 低光
    low_dir = os.path.join(args.out_base, "lowlight")
    low_full, low_eval = make_set(items, low_dir, "lowlight", rng, gamma=args.gamma)
    write_full(os.path.join(low_dir, "manifest_full.tsv"), low_full)
    write_eval(os.path.join(low_dir, "manifest_eval.tsv"), low_eval)

    # 噪声
    noise_dir = os.path.join(args.out_base, "noise")
    noise_full, noise_eval = make_set(items, noise_dir, "noise", rng, sigma=args.sigma)
    write_full(os.path.join(noise_dir, "manifest_full.tsv"), noise_full)
    write_eval(os.path.join(noise_dir, "manifest_eval.tsv"), noise_eval)

    # 未知退化 1：运动模糊（训练/标定不可见，参数固定）
    blur_dir = os.path.join(args.out_base, "motion_blur")
    blur_full, blur_eval = make_set(items, blur_dir, "motion_blur", rng)
    write_full(os.path.join(blur_dir, "manifest_full.tsv"), blur_full)
    write_eval(os.path.join(blur_dir, "manifest_eval.tsv"), blur_eval)

    # 未知退化 2：JPEG 压缩（训练/标定不可见，quality 固定 30）
    jpeg_dir = os.path.join(args.out_base, "jpeg")
    jpeg_full, jpeg_eval = make_set(items, jpeg_dir, "jpeg", rng)
    write_full(os.path.join(jpeg_dir, "manifest_full.tsv"), jpeg_full)
    write_eval(os.path.join(jpeg_dir, "manifest_eval.tsv"), jpeg_eval)

    # 未知退化 3：混合退化（低光 + 高斯噪声，顺序固定）
    mixed_dir = os.path.join(args.out_base, "mixed")
    mixed_full, mixed_eval = make_set(items, mixed_dir, "mixed", rng)
    write_full(os.path.join(mixed_dir, "manifest_full.tsv"), mixed_full)
    write_eval(os.path.join(mixed_dir, "manifest_eval.tsv"), mixed_eval)

    print(f"\n完成。低光 {len(low_eval)}、噪声 {len(noise_eval)}、"
          f"运动模糊 {len(blur_eval)}、JPEG {len(jpeg_eval)}、混合 {len(mixed_eval)} 张。")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
