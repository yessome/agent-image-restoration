"""Restormer pretrained variants via DeepInverse, with bounded GPU tiling."""
from __future__ import annotations

import os

import numpy as np

from .base import RestoreResult, UpstreamProcessor
from .deepinv_denoise import _tiled_inference


class RestormerProcessor(UpstreamProcessor):
    name = "restormer"

    def __init__(self, device: str = "cuda", weight_path: str | None = None,
                 variant: str = "restormer_real"):
        super().__init__(device=device, weight_path=weight_path)
        self.variant = variant

    def load(self) -> None:
        if not self.weight_path or not os.path.exists(self.weight_path):
            self.model = None
            return
        try:
            from deepinv.models import Restormer
            norm_type = "WithBias" if self.variant in ("restormer_defocus", "restormer_derain") else "BiasFree"
            self.model = Restormer(pretrained=self.weight_path, device=self.device,
                                   LayerNorm_type=norm_type).eval()
        except Exception as exc:
            self.model = None
            self._load_error = str(exc)

    def restore(self, image: np.ndarray) -> RestoreResult:
        self._assert_loaded()
        image = self._ensure_uint8(image)
        if self.model is None:
            return RestoreResult(image=image, meta={
                "model": self.variant, "fallback": True,
                "note": getattr(self, "_load_error", "Restormer 权重不存在"),
            })
        import torch
        tensor = torch.from_numpy(image.copy()).permute(2, 0, 1).float().div(255).unsqueeze(0)
        output, height, width = _tiled_inference(
            self.model, tensor, self.device, tile_size=256, overlap=32,
        )
        restored = (output[0, :, :height, :width].clamp(0, 1)
                    .permute(1, 2, 0).numpy() * 255).round().astype(np.uint8)
        return RestoreResult(image=restored, meta={
            "model": self.variant, "device": str(self.device), "tile_size": 256,
        })
