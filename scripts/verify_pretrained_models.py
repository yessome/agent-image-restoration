"""Verify every optional pretrained tool with a real forward pass."""
from __future__ import annotations

import argparse
import gc
import importlib
import os
import sys

import numpy as np
import yaml

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
from src.agents.orchestrator import _UPSTREAM_REGISTRY


def main() -> int:
    root = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
    with open(os.path.join(root, "config", "pretrained_tools.yaml"), encoding="utf-8") as handle:
        all_configs = yaml.safe_load(handle)["upstreams"]
    parser = argparse.ArgumentParser(description="验收额外预训练模型")
    parser.add_argument("--device", default="cuda")
    parser.add_argument("--model", choices=tuple(all_configs), action="append",
                        help="只验收指定模型，可重复传入；不指定时验收全部")
    args = parser.parse_args()
    configs = all_configs
    if args.model:
        configs = {name: config for name, config in configs.items() if name in args.model}
    image = np.full((65, 79, 3), 96, dtype=np.uint8)
    image[15:50, 20:60] = (170, 60, 30)
    failures = []
    for name, config in configs.items():
        processor = None
        try:
            module_name, class_name = _UPSTREAM_REGISTRY[name]
            cls = getattr(importlib.import_module(module_name), class_name)
            kwargs = {"device": args.device,
                      "weight_path": os.path.join(root, config["weight_path"])}
            if name == "ffdnet":
                kwargs["sigma"] = config.get("sigma", 25)
            if name.startswith("restormer_"):
                kwargs["variant"] = name
            if name.startswith("realesrgan_"):
                kwargs["scale"] = int(name.rsplit("x", 1)[1])
            processor = cls(**kwargs)
            result = processor.restore(image)
            scale = kwargs.get("scale", 1)
            expected_shape = (image.shape[0] * scale, image.shape[1] * scale, 3)
            if result.meta.get("fallback"):
                raise RuntimeError(result.meta.get("note", "fallback"))
            if result.image.dtype != np.uint8 or result.image.shape != expected_shape:
                raise RuntimeError(f"输出格式错误：{result.image.shape}, {result.image.dtype}")
            print(f"PASS {name}: {result.image.shape} {result.meta.get('device', args.device)}", flush=True)
        except Exception as exc:
            failures.append(name)
            print(f"FAIL {name}: {type(exc).__name__}: {exc}", flush=True)
        finally:
            if processor is not None:
                del processor
            gc.collect()
            try:
                import torch
                if torch.cuda.is_available():
                    torch.cuda.empty_cache()
            except ImportError:
                pass
    print(f"通过 {len(configs) - len(failures)}/{len(configs)} 个额外模型")
    return 1 if failures else 0


if __name__ == "__main__":
    raise SystemExit(main())
