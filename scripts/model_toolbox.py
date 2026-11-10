"""兼容入口：GPU 模型修复已统一到 src.main。"""
from __future__ import annotations

import argparse
import os
import sys

import cv2
import yaml

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
from src.agents.orchestrator import ImageRestorationAgent


def main() -> int:
    parser = argparse.ArgumentParser(description="GPU 图像修复模型工具")
    root = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
    with open(os.path.join(root, "config", "pretrained_tools.yaml"), encoding="utf-8") as f:
        extra_config = yaml.safe_load(f)
    parser.add_argument("--model", choices=tuple(extra_config["upstreams"]), required=True)
    parser.add_argument("--image", required=True)
    parser.add_argument("--output", required=True)
    parser.add_argument("--sigma", type=float, default=25.0)
    parser.add_argument("--device", default="cuda")
    args = parser.parse_args()
    with open(os.path.join(root, "config", "pipelines.yaml"), encoding="utf-8") as f:
        config = yaml.safe_load(f)
    config.setdefault("upstreams", {})
    for name, cfg in extra_config["upstreams"].items():
        config["upstreams"][name] = dict(cfg)
        config["upstreams"][name]["weight_path"] = os.path.join(root, cfg["weight_path"])
    config["upstreams"]["ffdnet"]["sigma"] = args.sigma
    image_bgr = cv2.imread(args.image)
    if image_bgr is None:
        parser.error(f"无法读取图片：{args.image}")
    image = cv2.cvtColor(image_bgr, cv2.COLOR_BGR2RGB)
    result = ImageRestorationAgent(config=config, device=args.device)._get_upstream(args.model).restore(image)
    if result.meta.get("fallback"):
        parser.error(result.meta.get("note", "模型加载失败"))
    os.makedirs(os.path.dirname(os.path.abspath(args.output)), exist_ok=True)
    cv2.imwrite(args.output, cv2.cvtColor(result.image, cv2.COLOR_RGB2BGR))
    print(f"已完成 {args.model} 推理（device={args.device}），输出：{args.output}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
