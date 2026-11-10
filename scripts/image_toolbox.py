"""批量之外的单图工具入口：python scripts/image_toolbox.py --list"""
from __future__ import annotations

import argparse
import os
import sys

import cv2

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
from src.upstream.toolbox import TOOLS, apply_tool


def main() -> int:
    parser = argparse.ArgumentParser(description="单图图像处理工具箱")
    parser.add_argument("--list", action="store_true", help="列出工具")
    parser.add_argument("--tool", choices=TOOLS, help="工具名称")
    parser.add_argument("--image", help="输入图片路径")
    parser.add_argument("--output", help="输出图片路径")
    args = parser.parse_args()
    if args.list:
        for name, (description, _) in TOOLS.items():
            print(f"{name:20s} {description}")
        return 0
    if not (args.tool and args.image and args.output):
        parser.error("处理图片时必须提供 --tool、--image 和 --output")
    image_bgr = cv2.imread(args.image, cv2.IMREAD_UNCHANGED)
    if image_bgr is None:
        parser.error(f"无法读取图片：{args.image}")
    if image_bgr.ndim == 3:
        image = cv2.cvtColor(image_bgr, cv2.COLOR_BGRA2RGBA if image_bgr.shape[2] == 4 else cv2.COLOR_BGR2RGB)
    else:
        image = image_bgr
    result = apply_tool(args.tool, image)
    if result.ndim == 3:
        result = cv2.cvtColor(result, cv2.COLOR_RGB2BGR)
    os.makedirs(os.path.dirname(os.path.abspath(args.output)), exist_ok=True)
    if not cv2.imwrite(args.output, result):
        parser.error(f"无法写出结果：{args.output}")
    print(f"已应用 {args.tool}，输出：{args.output}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
