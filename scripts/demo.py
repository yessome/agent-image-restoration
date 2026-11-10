"""运行单图分类并保存分支图像。"""
from __future__ import annotations

import argparse
import os
import sys

import cv2
import numpy as np

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from src.agents.orchestrator import ImageRestorationAgent
from src.main import load_config, _resolve_weight_paths


def main() -> int:
    parser = argparse.ArgumentParser(description="垃圾图像修复 Agent Demo")
    parser.add_argument("--image", required=True, help="输入图像路径")
    parser.add_argument("--intent", default="classify", choices=["classify"])
    parser.add_argument("--device", default="cuda", help="cuda / cpu")
    parser.add_argument("--ground-truth", default=None, help="离线真值标签（可选）")
    parser.add_argument("--outdir", default="demo", help="输出目录")
    parser.add_argument("--config", default="config/pipelines.yaml", help="配置文件路径")
    args = parser.parse_args()

    img_bgr = cv2.imread(args.image)
    if img_bgr is None:
        print(f"错误：无法读取 {args.image}", file=sys.stderr)
        return 1
    image = cv2.cvtColor(img_bgr, cv2.COLOR_BGR2RGB)

    root = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
    config_path = args.config if os.path.isabs(args.config) else os.path.join(root, args.config)
    config = load_config(config_path)
    config_dir = os.path.dirname(os.path.abspath(config_path))
    project_root = os.path.dirname(config_dir) if os.path.basename(config_dir) == "config" else config_dir
    config = _resolve_weight_paths(config, project_root)

    agent = ImageRestorationAgent(config=config, device=args.device)

    print("=" * 60)
    print(f"输入图像: {args.image}")
    print(f"下游意图: {args.intent}")
    print("=" * 60)

    gt = args.ground_truth
    result = agent.run(image, intent=args.intent, ground_truth=gt)

    print(f"\n[诊断] 噪声高={result.diagnosis.get('noise_high', False)} "
          f"低光={result.diagnosis.get('lowlight', False)} "
          f"(噪声分={result.diagnosis.get('noise_score', 0):.2f} "
          f"低光分={result.diagnosis.get('lowlight_score', 0):.2f})")
    print(f"[决策] 最终选择: {result.chosen_pipeline}")
    print(f"[说明] {result.explanation}")
    print(f"\n各候选方案:")
    for o in result.outcomes:
        mark = "  <- selected" if o.pipeline_id == result.chosen_pipeline else ""
        fb = " [降级]" if o.fallback else " [真实推理]"
        print(f"  {o.pipeline_id:16s} 置信度={o.score:.4f} 预测={o.pred_label}{fb}{mark}")

    if gt:
        print(f"\n[离线评价] Oracle={result.oracle_pipeline} "
              f"命中={result.hit} Regret={result.regret}")

    # 保存各候选分支的修复结果
    os.makedirs(args.outdir, exist_ok=True)
    for o in result.outcomes:
        if o.restored_image is not None and o.upstream:
            out_path = os.path.join(args.outdir, f"{o.pipeline_id}.png")
            cv2.imwrite(out_path, cv2.cvtColor(o.restored_image, cv2.COLOR_RGB2BGR))
            print(f"\n已保存修复结果: {out_path}")

    print("\n" + "=" * 60)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
