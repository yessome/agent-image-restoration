"""下载 WEIGHTS 中列出的第三方检查点。DnCNN 权重由 train_dncnn.py 生成。"""
from __future__ import annotations

import argparse
import os
import sys
import urllib.request

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
WEIGHTS_DIR = os.path.join(ROOT, "weights")

WEIGHTS = {
    "lowlight": {
        "url": "https://raw.githubusercontent.com/Li-Chongyi/Zero-DCE/master/"
               "Zero-DCE_code/snapshots/Epoch99.pth",
        "filename": "Zero-DCE_Epoch99.pth",
    },
    "zero_dce_plus_plus": {
        # 直连: https://raw.githubusercontent.com/Li-Chongyi/Zero-DCE_extension/main/Zero-DCE++/snapshots_Zero_DCE++/Epoch99.pth
        "url": "https://gh-proxy.com/https://raw.githubusercontent.com/"
               "Li-Chongyi/Zero-DCE_extension/main/"
               "Zero-DCE++/snapshots_Zero_DCE++/Epoch99.pth",
        "filename": os.path.join("pretrained", "zero_dce_plus_plus.pth"),
    },
    "sci_lowlight": {
        # 直连: https://raw.githubusercontent.com/vis-opt-group/SCI/main/CVPR/weights/medium.pt
        "url": "https://gh-proxy.com/https://raw.githubusercontent.com/"
               "vis-opt-group/SCI/main/CVPR/weights/medium.pt",
        "filename": os.path.join("pretrained", "sci_medium.pt"),
    },
    "mprnet_deblur": {
        # 直连: https://huggingface.co/deepinv/MPRNet/resolve/main/model_deblurring.pth
        "url": "https://hf-mirror.com/deepinv/MPRNet/resolve/main/"
               "model_deblurring.pth",
        "filename": os.path.join("pretrained", "mprnet_deblurring.pth"),
    },
    "mprnet_derain": {
        # 直连: https://huggingface.co/deepinv/MPRNet/resolve/main/model_deraining.pth
        "url": "https://hf-mirror.com/deepinv/MPRNet/resolve/main/"
               "model_deraining.pth",
        "filename": os.path.join("pretrained", "mprnet_deraining.pth"),
    },
}


def download(name: str) -> str:
    info = WEIGHTS[name]
    os.makedirs(WEIGHTS_DIR, exist_ok=True)
    dest = os.path.join(WEIGHTS_DIR, info["filename"])
    os.makedirs(os.path.dirname(dest), exist_ok=True)
    if os.path.exists(dest) and os.path.getsize(dest) > 0:
        print(f"[跳过] {info['filename']} 已存在")
        return dest

    print(f"[下载] {name} -> {info['filename']}")
    print(f"       来源: {info['url']}")
    urllib.request.urlretrieve(info["url"], dest)
    size_mb = os.path.getsize(dest) / 1024 / 1024
    print(f"[完成] {info['filename']} ({size_mb:.1f} MB)")
    return dest


def main() -> int:
    parser = argparse.ArgumentParser(description="下载上游修复器权重")
    parser.add_argument("--all", action="store_true", help="下载全部")
    parser.add_argument("--lowlight", action="store_true", help="只下 Zero-DCE 低光")
    parser.add_argument("--tool", choices=WEIGHTS, action="append",
                        help="下载指定模型权重，可重复传入")
    args = parser.parse_args()

    targets = []
    if args.lowlight:
        targets.append("lowlight")
    if args.tool:
        targets.extend(args.tool)
    if args.all or not targets:
        targets = list(WEIGHTS)

    for name in targets:
        try:
            download(name)
        except Exception as e:
            print(f"[失败] {name}: {e}", file=sys.stderr)
            return 1

    print(f"\n权重已保存到: {WEIGHTS_DIR}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
