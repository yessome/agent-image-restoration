"""FFDNet 与 SCUNet 预训练去噪器，使用 DeepInverse 权重实现。"""
from __future__ import annotations

import os

import numpy as np

from .base import RestoreResult, UpstreamProcessor


def _tile_starts(length: int, tile: int, overlap: int) -> list[int]:
    if length <= tile:
        return [0]
    step = tile - overlap
    starts = list(range(0, length - tile + 1, step))
    if starts[-1] != length - tile:
        starts.append(length - tile)
    return starts


def _tiled_inference(model, tensor, device, tile_size: int, overlap: int, sigma=None):
    """加权拼接分块结果，限制峰值显存并减轻块边界接缝。"""
    import torch
    import torch.nn.functional as F

    _, _, height, width = tensor.shape
    minimum = tile_size
    pad_h, pad_w = max(0, minimum - height), max(0, minimum - width)
    if pad_h or pad_w:
        mode = "reflect" if pad_h < height and pad_w < width else "replicate"
        tensor = F.pad(tensor, (0, pad_w, 0, pad_h), mode=mode)
    padded_h, padded_w = tensor.shape[-2:]
    ys = _tile_starts(padded_h, tile_size, overlap)
    xs = _tile_starts(padded_w, tile_size, overlap)
    result = torch.zeros_like(tensor, device="cpu")
    weights = torch.zeros((1, 1, padded_h, padded_w), dtype=torch.float32)

    for y in ys:
        for x in xs:
            patch = tensor[:, :, y:y + tile_size, x:x + tile_size].to(device)
            with torch.inference_mode():
                restored = model(patch, sigma) if sigma is not None else model(patch)
            restored = restored.detach().float().cpu()
            wy = torch.ones(tile_size)
            wx = torch.ones(tile_size)
            ramp = torch.linspace(0.05, 1.0, overlap)
            if y > 0:
                wy[:overlap] = ramp
            if y + tile_size < padded_h:
                wy[-overlap:] = ramp.flip(0)
            if x > 0:
                wx[:overlap] = ramp
            if x + tile_size < padded_w:
                wx[-overlap:] = ramp.flip(0)
            weight = (wy[:, None] * wx[None, :])[None, None]
            result[:, :, y:y + tile_size, x:x + tile_size] += restored * weight
            weights[:, :, y:y + tile_size, x:x + tile_size] += weight

    return result / weights.clamp_min(1e-6), height, width


class FFDNetProcessor(UpstreamProcessor):
    """FFDNet 彩色高斯去噪；sigma 使用 0-255 像素尺度。"""

    name = "ffdnet"

    def __init__(self, device: str = "cuda", weight_path: str | None = None,
                 sigma: float = 25.0):
        super().__init__(device=device, weight_path=weight_path)
        self.sigma = float(sigma)
        if not 0.0 <= self.sigma <= 75.0:
            raise ValueError("FFDNet sigma must be in [0, 75]")

    def load(self) -> None:
        if not self.weight_path or not os.path.exists(self.weight_path):
            self.model = None
            return
        try:
            from deepinv.models import FFDNet
            self.model = FFDNet(
                n_conv_layers=12, nf=96, img_channels=3,
                pretrained=self.weight_path, device=self.device,
            ).eval()
        except Exception as exc:
            self.model = None
            self._load_error = str(exc)

    def restore(self, image: np.ndarray) -> RestoreResult:
        self._assert_loaded()
        image = self._ensure_uint8(image)
        if self.model is None:
            return RestoreResult(image=image, meta={
                "model": "FFDNet-color", "fallback": True,
                "note": getattr(self, "_load_error", "未找到 FFDNet 权重"),
            })
        import torch
        tensor = torch.from_numpy(image.copy()).permute(2, 0, 1).float().div(255).unsqueeze(0)
        sigma = torch.tensor([self.sigma / 255.0], device=self.device)
        output, height, width = _tiled_inference(
            self.model, tensor, self.device, tile_size=512, overlap=32, sigma=sigma,
        )
        restored = (output[0, :, :height, :width].clamp(0, 1)
                    .permute(1, 2, 0).numpy() * 255).round().astype(np.uint8)
        return RestoreResult(image=restored, meta={
            "model": "FFDNet-color", "sigma": self.sigma, "device": str(self.device),
        })


class SCUNetProcessor(UpstreamProcessor):
    """SCUNet 彩色真实噪声去除，256 像素分块适配 8GB 显存。"""

    name = "scunet"

    def load(self) -> None:
        if not self.weight_path or not os.path.exists(self.weight_path):
            self.model = None
            return
        try:
            from deepinv.models import SCUNet
            self.model = SCUNet(pretrained=self.weight_path, device=self.device).eval()
        except Exception as exc:
            self.model = None
            self._load_error = str(exc)

    def restore(self, image: np.ndarray) -> RestoreResult:
        self._assert_loaded()
        image = self._ensure_uint8(image)
        if self.model is None:
            return RestoreResult(image=image, meta={
                "model": "SCUNet-color-real", "fallback": True,
                "note": getattr(self, "_load_error", "未找到 SCUNet 权重"),
            })
        import torch
        tensor = torch.from_numpy(image.copy()).permute(2, 0, 1).float().div(255).unsqueeze(0)
        output, height, width = _tiled_inference(
            self.model, tensor, self.device, tile_size=256, overlap=32,
        )
        restored = (output[0, :, :height, :width].clamp(0, 1)
                    .permute(1, 2, 0).numpy() * 255).round().astype(np.uint8)
        return RestoreResult(image=restored, meta={
            "model": "SCUNet-color-real", "device": str(self.device),
            "tile_size": 256,
        })
