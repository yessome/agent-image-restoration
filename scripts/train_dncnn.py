"""本地训练 DnCNN 彩色高斯去噪器。

动机：NAFNet-SIDD 面向真实传感器噪声，对项目实验用的合成高斯噪声（sigma=25）
去噪后反而过曝失真、损害下游分类。这里用项目自身 train 集干净图合成 sigma=25
高斯噪声训练对，训练一个匹配该噪声分布的 DnCNN，使去噪对下游分类真正有益。

训练数据：data/splits/train.tsv 指向的干净图（1767 张），随机裁剪 128x128 补丁，
          施加 sigma=25 高斯噪声（与 make_test_degraded.py / degrade_manifest.py 一致）。
损失：MSE（预测噪声 vs 真实噪声）。
输出：weights/dncnn_gaussian_color.pth（含 state_dict + 元数据）。

用法：
    python scripts/train_dncnn.py --epochs 40 --batch-size 16 --sigma 25 --seed 42
"""
from __future__ import annotations

import argparse
import hashlib
import os
import sys
import time

import cv2
import numpy as np

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

import torch
import torch.nn as nn

from src.upstream.dncnn import _build_dncnn


def read_manifest(path):
    items = []
    with open(path, encoding="utf-8") as f:
        for line in f:
            line = line.rstrip("\n")
            if not line or line.startswith("#"):
                continue
            parts = line.split("\t")
            if len(parts) >= 1 and parts[0].strip():
                items.append(parts[0].strip())
    return items


def make_patch_pair(img_bgr, sigma, rng, patch=128):
    """从干净图随机裁剪补丁并加高斯噪声，返回 (noisy, clean) 的 float32 [0,1] CHW。"""
    h, w = img_bgr.shape[:2]
    if h < patch or w < patch:
        # 图小于补丁：整图缩放到 patch
        img_bgr = cv2.resize(img_bgr, (patch, patch), interpolation=cv2.INTER_LINEAR)
        h, w = patch, patch
    y = rng.integers(0, h - patch + 1)
    x = rng.integers(0, w - patch + 1)
    crop = img_bgr[y:y + patch, x:x + patch].astype(np.float32)  # BGR
    # 加高斯噪声
    noise = rng.normal(0, sigma, crop.shape).astype(np.float32)
    noisy = np.clip(crop + noise, 0, 255)
    clean = crop
    # BGR -> RGB，转 [0,1] CHW
    noisy = noisy[:, :, ::-1].transpose(2, 0, 1) / 255.0
    clean = clean[:, :, ::-1].transpose(2, 0, 1) / 255.0
    return noisy, clean


def main() -> int:
    parser = argparse.ArgumentParser(description="训练 DnCNN 彩色高斯去噪器")
    parser.add_argument("--manifest", default="data/splits/train.tsv")
    parser.add_argument("--epochs", type=int, default=40)
    parser.add_argument("--batch-size", type=int, default=16)
    parser.add_argument("--lr", type=float, default=1e-3)
    parser.add_argument("--sigma", type=float, default=25.0)
    parser.add_argument("--patch", type=int, default=128)
    parser.add_argument("--steps-per-epoch", type=int, default=100)
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--device", default="cuda")
    parser.add_argument("--out", default="weights/dncnn_gaussian_color.pth")
    args = parser.parse_args()

    device = args.device
    if device == "cuda" and not torch.cuda.is_available():
        print("[警告] CUDA 不可用，回退 CPU", file=sys.stderr)
        device = "cpu"

    paths = read_manifest(args.manifest)
    print(f"训练图 {len(paths)} 张，sigma={args.sigma}，epochs={args.epochs}，"
          f"batch={args.batch_size}，patch={args.patch}，device={device}")

    # 固定随机种子（NumPy + PyTorch），保证可复现
    torch.manual_seed(args.seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(args.seed)
    rng = np.random.default_rng(args.seed)
    model = _build_dncnn(in_ch=3, out_ch=3).to(device)
    opt = torch.optim.Adam(model.parameters(), lr=args.lr)
    sched = torch.optim.lr_scheduler.CosineAnnealingLR(opt, T_max=args.epochs)
    criterion = nn.MSELoss()

    # 固定验证图（从 train 集取前 4 张，用于每个 epoch 算 PSNR，验证去噪是否真正有效）
    val_paths = [paths[i] for i in range(min(4, len(paths)))]
    val_pairs = []
    _vrng = np.random.default_rng(123)
    for p in val_paths:
        img = cv2.imread(p, cv2.IMREAD_COLOR)
        if img is None:
            continue
        n_, c_ = make_patch_pair(img, args.sigma, _vrng, args.patch)
        val_pairs.append((n_, c_))
    val_x = torch.from_numpy(np.stack([v[0] for v in val_pairs])).to(device)
    val_clean = torch.from_numpy(np.stack([v[1] for v in val_pairs])).to(device)

    for epoch in range(1, args.epochs + 1):
        model.train()
        total_loss = 0.0
        t0 = time.time()
        for _ in range(args.steps_per_epoch):
            noisy_b, clean_b = [], []
            for _ in range(args.batch_size):
                p = paths[rng.integers(0, len(paths))]
                img = cv2.imread(p, cv2.IMREAD_COLOR)
                if img is None:
                    continue
                noisy, clean = make_patch_pair(img, args.sigma, rng, args.patch)
                noisy_b.append(noisy)
                clean_b.append(clean)
            if not noisy_b:
                continue
            x = torch.from_numpy(np.stack(noisy_b)).to(device)
            y_clean = torch.from_numpy(np.stack(clean_b)).to(device)
            # 目标：预测噪声残差 = noisy - clean（DnCNN 学习预测加性噪声）
            target = x - y_clean
            pred = model(x)
            loss = criterion(pred, target)
            opt.zero_grad()
            loss.backward()
            opt.step()
            total_loss += loss.item()
        sched.step()
        avg = total_loss / args.steps_per_epoch

        # 每 5 个 epoch 在验证图上算 PSNR（重建 clean = noisy - pred）
        psnr_str = ""
        if epoch % 5 == 0 or epoch == args.epochs:
            model.eval()
            with torch.no_grad():
                rec = (val_x - model(val_x)).clamp(0, 1)
            mse = torch.mean((rec - val_clean) ** 2).item()
            psnr = 10 * np.log10(1.0 / mse) if mse > 0 else 99.0
            psnr_str = f"  val_PSNR={psnr:.2f}dB"
            model.train()

        print(f"epoch {epoch:3d}/{args.epochs}  loss={avg:.6f}  "
              f"lr={sched.get_last_lr()[0]:.2e}  ({time.time()-t0:.1f}s){psnr_str}")

    # 保存权重（含溯源元数据）
    def _sha256(p):
        h = hashlib.sha256()
        with open(p, "rb") as f:
            for c in iter(lambda: f.read(1 << 20), b""):
                h.update(c)
        return h.hexdigest()
    manifest_path = os.path.abspath(args.manifest)
    os.makedirs(os.path.dirname(args.out) or ".", exist_ok=True)
    torch.save({
        "state_dict": model.state_dict(),
        "sigma": args.sigma,
        "in_ch": 3,
        "out_ch": 3,
        "num_layers": 20,
        "desc": "DnCNN gaussian color denoiser, trained on TrashNet train set",
        "manifest": manifest_path,
        "manifest_sha256": _sha256(manifest_path) if os.path.exists(manifest_path) else None,
        "manifest_n_images": len(paths),
        "seed": args.seed,
        "epochs": args.epochs,
    }, args.out)
    print(f"\n权重已保存 -> {args.out}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
