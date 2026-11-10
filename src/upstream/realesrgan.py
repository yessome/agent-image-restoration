"""Real-ESRGAN x2/x4 super-resolution, tiled for an 8 GB GPU."""
from __future__ import annotations

import os
import sys
import types

import numpy as np

from .base import RestoreResult, UpstreamProcessor


class RealESRGANProcessor(UpstreamProcessor):
    name = "realesrgan"

    def __init__(self, device: str = "cuda", weight_path: str | None = None,
                 scale: int = 4):
        super().__init__(device=device, weight_path=weight_path)
        if scale not in (2, 4):
            raise ValueError("Real-ESRGAN scale must be 2 or 4")
        self.scale = scale

    def load(self) -> None:
        if not self.weight_path or not os.path.exists(self.weight_path):
            self.model = None
            return
        try:
            # BasicSR 1.4.2 still imports this removed torchvision module.
            if "torchvision.transforms.functional_tensor" not in sys.modules:
                from torchvision.transforms import functional
                shim = types.ModuleType("torchvision.transforms.functional_tensor")
                shim.rgb_to_grayscale = functional.rgb_to_grayscale
                sys.modules[shim.__name__] = shim
            import torch
            from basicsr.archs.rrdbnet_arch import RRDBNet
            net = RRDBNet(num_in_ch=3, num_out_ch=3, num_feat=64,
                          num_block=23, num_grow_ch=32, scale=self.scale)
            state = torch.load(self.weight_path, map_location="cpu", weights_only=True)
            net.load_state_dict(state, strict=True)
            self.model = net.to(self.device).eval()
        except Exception as exc:
            self.model = None
            self._load_error = str(exc)

    def restore(self, image: np.ndarray) -> RestoreResult:
        self._assert_loaded()
        image = self._ensure_uint8(image)
        if self.model is None:
            return RestoreResult(image=image, meta={
                "model": f"Real-ESRGAN-x{self.scale}", "fallback": True,
                "note": getattr(self, "_load_error", "Real-ESRGAN 权重不存在"),
            })
        import torch
        import torch.nn.functional as F
        tensor = torch.from_numpy(image.copy()).permute(2, 0, 1).float().div(255).unsqueeze(0)
        height, width = image.shape[:2]
        tile, context = 128, 10
        output = np.empty((height * self.scale, width * self.scale, 3), dtype=np.uint8)
        for y in range(0, height, tile):
            for x in range(0, width, tile):
                y1, x1 = min(y + tile, height), min(x + tile, width)
                py0, px0 = max(0, y - context), max(0, x - context)
                py1, px1 = min(height, y1 + context), min(width, x1 + context)
                patch = tensor[:, :, py0:py1, px0:px1]
                pad_h, pad_w = patch.shape[-2] % 2, patch.shape[-1] % 2
                if self.scale == 2 and (pad_h or pad_w):
                    patch = F.pad(patch, (0, pad_w, 0, pad_h), mode="replicate")
                with torch.inference_mode():
                    enlarged = self.model(patch.to(self.device)).float().cpu()
                crop = enlarged[:, :, (y - py0) * self.scale:(y1 - py0) * self.scale,
                                (x - px0) * self.scale:(x1 - px0) * self.scale]
                pixels = (crop[0].clamp(0, 1).permute(1, 2, 0).numpy() * 255).round().astype(np.uint8)
                output[y * self.scale:y1 * self.scale,
                       x * self.scale:x1 * self.scale] = pixels
        return RestoreResult(image=output, meta={
            "model": f"Real-ESRGAN-x{self.scale}", "device": str(self.device),
            "tile_size": 128,
        })
