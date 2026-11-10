"""生成最终 13 分支配置的冻结清单。"""
from __future__ import annotations

import hashlib
import json
import os
from datetime import datetime, timezone
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
OUT = ROOT / "results" / "frozen_manifest" / "frozen_manifest_13tool.json"

BRANCH_ORDER = ["", "denoise", "ffdnet", "scunet", "restormer_real", "restormer_blind",
                "restormer_sigma15", "restormer_sigma25", "restormer_sigma50",
                "lowlight", "zero_dce_plus_plus", "sci_lowlight", "retinexformer"]


def sha256_file(path: Path) -> str:
    digest = hashlib.sha256()
    with open(path, "rb") as stream:
        for block in iter(lambda: stream.read(1 << 20), b""):
            digest.update(block)
    return digest.hexdigest()


def entry(rel: str) -> dict:
    path = ROOT / rel
    if not path.is_file():
        raise FileNotFoundError(f"缺少文件：{path}")
    return {"path": rel, "sha256": sha256_file(path), "bytes": path.stat().st_size,
            "exists": True}


def main() -> int:
    tool_paths = {
        "denoise": "weights/dncnn_gaussian_color.pth",
        "ffdnet": "weights/pretrained/ffdnet_color.pth",
        "scunet": "weights/pretrained/scunet_color_real_psnr.pth",
        "restormer_real": "weights/pretrained/restormer_real_denoising.pth",
        "restormer_blind": "weights/pretrained/restormer_gaussian_blind.pth",
        "restormer_sigma15": "weights/pretrained/restormer_gaussian_sigma15.pth",
        "restormer_sigma25": "weights/pretrained/restormer_gaussian_sigma25.pth",
        "restormer_sigma50": "weights/pretrained/restormer_gaussian_sigma50.pth",
        "lowlight": "weights/Zero-DCE_Epoch99.pth",
        "zero_dce_plus_plus": "weights/pretrained/zero_dce_plus_plus.pth",
        "sci_lowlight": "weights/pretrained/sci_medium.pt",
        "retinexformer": "weights/pretrained/retinexformer_lol_v2_real.pth",
    }

    payload = {
        "schema": "frozen-13tool-v1",
        "note": "最终 13 工具配置的冻结凭据。旧版 17 工具资产不属于此发布包。",
        "created_at": datetime.now(timezone.utc).strftime("%Y-%m-%dT%H:%M:%SZ"),
        "supersedes_note": "本清单是最终论文口径；三组 router 与 classifier 一一配对。",
        "routers": {
            f"seed{s}": entry(f"weights/router_iterative_feedback_13tool_seed{s}.pth")
            for s in (1, 2, 3)
        },
        "classifiers": {
            f"seed{s}": entry(f"weights/trashnet_resnet50_seed{s}.pth")
            for s in (1, 2, 3)
        },
        "configs": {
            "config/iterative_agent_13tool.yaml": entry("config/iterative_agent_13tool.yaml"),
            "config/pipelines.yaml": entry("config/pipelines.yaml"),
            "config/pretrained_tools.yaml": entry("config/pretrained_tools.yaml"),
        },
        "tools": {k: entry(v) for k, v in tool_paths.items()},
        "branch_order": BRANCH_ORDER,
        "n_branches": len(BRANCH_ORDER),
        "excluded_branches": ["restormer_defocus", "mprnet_deblur",
                              "realesrgan_x2", "realesrgan_x4"],
        "iterative_hparams": {"max_calls": 4, "call_cost": 0.01, "clip_max": 0.05},
        "thresholds": {
            "confidence_threshold": 0.6,
            "selector": {"delta_conf": 0.02, "tau_uncert": 0.25,
                          "kappa": 1.2, "clip_max": 0.05},
            "routing_mode": "iterative",
            "iterative_weight": "weights/router_iterative_feedback_13tool_seed1.pth",
            "llm_enabled": False,
        },
        "pairing": {
            f"seed{s}": {"router": f"seed{s}", "classifier": f"seed{s}"}
            for s in (1, 2, 3)
        },
        "routing_mode": "iterative",
        "llm_enabled": False,
    }
    OUT.write_text(json.dumps(payload, ensure_ascii=False, indent=2), encoding="utf-8")

    print(f"写出 {OUT}")
    for k, v in payload["routers"].items():
        print(f"  router {k}: {v['sha256'][:16]}… {v['bytes']} bytes")
    missing = [k for k, v in payload["tools"].items() if not v["exists"]]
    print(f"  工具 {len(payload['tools'])} 个，缺失 {len(missing)}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
