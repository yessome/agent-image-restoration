"""MPRNet motion deblurring with tiled inference for 8 GB GPUs."""
from __future__ import annotations

import os

import numpy as np

from .base import RestoreResult, UpstreamProcessor
from .deepinv_denoise import _tiled_inference


class MPRNetDeblurProcessor(UpstreamProcessor):
    name = "mprnet_deblur"

    def load(self) -> None:
        if not self.weight_path or not os.path.isfile(self.weight_path):
            self.model = None
            return
        try:
            from deepinv.models import MPRNet
            # deepinv 的 deblurring 官方权重要求 n_feat=96；
            # 默认 n_feat=80 是去噪任务架构，直接 load 会 shape mismatch。
            self.model = MPRNet(
                n_feat=96, pretrained=self.weight_path, device=self.device
            ).eval()
        except Exception as exc:
            self.model = None
            self._load_error = str(exc)

    def restore(self, image: np.ndarray) -> RestoreResult:
        self._assert_loaded()
        image = self._ensure_uint8(image)
        if self.model is None:
            return RestoreResult(image=image, meta={
                "model": "MPRNet-GoPro", "fallback": True,
                "note": getattr(self, "_load_error", "MPRNet 权重无法加载"),
            })
        import torch
        tensor = torch.from_numpy(image.copy()).permute(2, 0, 1).float().div(255).unsqueeze(0)
        output, height, width = _tiled_inference(
            self.model, tensor, self.device, tile_size=256, overlap=32,
        )
        restored = (output[0, :, :height, :width].clamp(0, 1)
                    .permute(1, 2, 0).numpy() * 255).round().astype(np.uint8)
        return RestoreResult(image=restored, meta={
            "model": "MPRNet-GoPro", "device": str(self.device), "tile_size": 256,
        })


class MPRNetDerainProcessor(MPRNetDeblurProcessor):
    """MPRNet deraining checkpoint with the same bounded tiled inference."""

    name = "mprnet_derain"

    def load(self) -> None:
        if not self.weight_path or not os.path.isfile(self.weight_path):
            self.model = None
            return
        try:
            from deepinv.models import MPRNet
            # deepinv 的 deraining 官方权重用更小架构：n_feat=40、
            # scale_unetfeats=20、scale_orsnetfeats=16（deblurring 为 96/48/32，
            # denoising 为 80/48/32）。不传会 shape mismatch。
            self.model = MPRNet(
                n_feat=40, scale_unetfeats=20, scale_orsnetfeats=16,
                pretrained=self.weight_path, device=self.device,
            ).eval()
        except Exception as exc:
            self.model = None
            self._load_error = str(exc)

    def restore(self, image: np.ndarray) -> RestoreResult:
        result = super().restore(image)
        result.meta["model"] = "MPRNet-Deraining"
        return result
