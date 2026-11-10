"""下载 TrashNet 六分类数据集并整理成 train_classifier.py 期望的目录结构。

数据来源（原始）：https://github.com/garythung/trashnet
  6 类：cardboard(403) / glass(501) / metal(410) / paper(594) / plastic(482) / trash(137)
  共 2527 张，512x384，白色背景自然光拍摄。

原仓库是 Lua 项目，图片需额外下载。本脚本按优先级尝试多个来源：
  1. 若本地已存在 data/trashnet/ 下的类别子目录，则跳过下载。
  2. 尝试从 HuggingFace 镜像下载整理好的 zip（若配置了 DATASET_URL）。
  3. 兜底：提示用户手动从 Kaggle/GitHub 获取并放到指定目录。

用法：
    python scripts/download_trashnet.py --out data/trashnet
"""
from __future__ import annotations

import argparse
import os
import sys
import urllib.request

# 留空时由 --url 提供镜像，或按提示手动准备数据。
DEFAULT_URLS = [""]

LABELS = ["cardboard", "glass", "metal", "paper", "plastic", "trash"]


def is_ready(root: str) -> bool:
    """检查目标目录是否已按类别整理好。"""
    for lb in LABELS:
        sub = os.path.join(root, lb)
        if not os.path.isdir(sub):
            return False
        imgs = [f for f in os.listdir(sub)
                if f.lower().endswith((".png", ".jpg", ".jpeg", ".bmp", ".webp"))]
        if len(imgs) < 1:
            return False
    return True


def download(url: str, dest: str) -> None:
    print(f"  下载: {url}")
    print(f"  到:   {dest}")
    os.makedirs(os.path.dirname(dest) or ".", exist_ok=True)
    req = urllib.request.Request(url, headers={"User-Agent": "Mozilla/5.0"})
    with urllib.request.urlopen(req, timeout=120) as r, open(dest, "wb") as f:
        total = 0
        while True:
            chunk = r.read(1024 * 256)
            if not chunk:
                break
            f.write(chunk)
            total += len(chunk)
            if total % (1024 * 1024 * 16) < 1024 * 256:
                print(f"    {total / 1024 / 1024:.1f} MB ...")
    print(f"  完成：{total / 1024 / 1024:.1f} MB")


def main() -> int:
    parser = argparse.ArgumentParser(description="下载 TrashNet 数据集")
    parser.add_argument("--out", default="data/trashnet", help="输出目录")
    parser.add_argument("--url", default="", help="手动指定整理好的 zip 下载地址")
    args = parser.parse_args()

    out = args.out
    if is_ready(out):
        print(f"检测到 {out} 已按类别整理好，跳过下载。")
        print("可直接运行：")
        print(f"  python scripts/train_classifier.py "
              f"--data {out} --split-four --device cuda")
        return 0

    urls = [u for u in ([args.url] if args.url else DEFAULT_URLS) if u]
    if urls:
        import zipfile
        import tempfile
        for u in urls:
            try:
                tmp = os.path.join(tempfile.gettempdir(), "trashnet.zip")
                download(u, tmp)
                os.makedirs(out, exist_ok=True)
                with zipfile.ZipFile(tmp) as z:
                    z.extractall(out)
                if is_ready(out):
                    print(f"\n成功：数据已就绪 -> {out}")
                    return 0
            except Exception as e:
                print(f"  该来源失败：{e}", file=sys.stderr)
                continue

    # 兜底：给出清晰的手动获取指引
    print("\n==============================================================", file=sys.stderr)
    print("未能自动下载。请手动获取 TrashNet 数据集并整理为如下结构：", file=sys.stderr)
    print(f"\n  {out}/\n    cardboard/   glass/   metal/   paper/   plastic/   trash/\n", file=sys.stderr)
    print("获取途径（任选其一）：", file=sys.stderr)
    print("  1. Kaggle: https://www.kaggle.com/datasets/feyzazkefe/trashnet", file=sys.stderr)
    print("     或搜索 'trashnet dataset'，下载解压后把 6 个类别文件夹放入上述目录。", file=sys.stderr)
    print("  2. GitHub 原仓库: https://github.com/garythung/trashnet", file=sys.stderr)
    print("     （需从 README 的 Google Drive 链接下载 dataset-resized.zip，", file=sys.stderr)
    print("      解压得到 dataset-resized/ 下的 6 个子目录，复制到上述目录。）", file=sys.stderr)
    print("  3. 若已有 zip，可用 --url 指定后重跑本脚本自动解压。", file=sys.stderr)
    print("==============================================================", file=sys.stderr)
    return 2


if __name__ == "__main__":
    raise SystemExit(main())
