"""DnCNN 彩色图像去噪处理器（针对合成高斯噪声）。

与 NAFNet-SIDD（真实传感器噪声）不同，DnCNN 采用残差学习直接预测
加性高斯噪声，输入减噪声残差得到干净图。这里针对 sigma=25 高斯噪声
在项目自身数据上训练，保证与实验退化集（apply_gaussian, sigma=25）
分布一致，从而去噪对下游垃圾分类真正有益。

架构（经典 20 层，彩色版）：
  第一层 Conv(3->64, 3x3) + ReLU
  中间 18 层 Conv(64->64, 3x3) + BN + ReLU
  末层   Conv(64->3, 3x3)   （输出预测噪声，无激活）
  重建：clean = noisy - predicted_noise

训练脚本见 scripts/train_dncnn.py，产出 weights/dncnn_gaussian_color.pth。
"""
from __future__ import annotations

import os

import numpy as np

from .base import UpstreamProcessor, RestoreResult


def _build_dncnn(in_ch: int = 3, out_ch: int = 3, num_layers: int = 20,
                 mid_ch: int = 64):
    """构建 DnCNN 网络（残差学习，预测噪声）。"""
    import torch
    import torch.nn as nn

    layers = []
    # 第一层：无 BN（遵循原论文，BN 仅用于中间层）
    layers.append(nn.Conv2d(in_ch, mid_ch, 3, 1, 1, bias=True))
    layers.append(nn.ReLU(inplace=True))
    # 中间 18 层：Conv + BN + ReLU
    for _ in range(num_layers - 2):
        layers.append(nn.Conv2d(mid_ch, mid_ch, 3, 1, 1, bias=False))
        layers.append(nn.BatchNorm2d(mid_ch))
        layers.append(nn.ReLU(inplace=True))
    # 末层：输出残差（无激活）
    layers.append(nn.Conv2d(mid_ch, out_ch, 3, 1, 1, bias=True))

    model = nn.Sequential(*layers)
    return model


class DnCNNProcessor(UpstreamProcessor):
    name = "dncnn"

    def load(self) -> None:
        if self.weight_path is None or not os.path.exists(self.weight_path):
            self.model = None
            return
        import torch
        try:
            model = _build_dncnn()
            sd = torch.load(self.weight_path, map_location="cpu", weights_only=False)
            if isinstance(sd, dict) and "state_dict" in sd:
                sd = sd["state_dict"]
            # 去除可能的 "module." 前缀
            sd = {k.replace("module.", ""): v for k, v in sd.items()}
            model.load_state_dict(sd, strict=True)
            self.model = model.to(self.device).eval()
        except Exception as e:
            self.model = None
            self._load_error = str(e)

    def restore(self, image: np.ndarray) -> RestoreResult:
        self._assert_loaded()
        image = self._ensure_uint8(image)

        if self.model is None:
            return RestoreResult(
                image=image,
                meta={"model": "DnCNN-gaussian-color", "fallback": True,
                      "note": f"权重未加载或加载失败：{getattr(self, '_load_error', '未找到权重')}"},
            )

        import torch

        # DnCNN 输入范围 [0,1]，预测噪声残差
        t = torch.from_numpy(image).permute(2, 0, 1).float() / 255.0
        t = t.unsqueeze(0).to(self.device)

        with torch.no_grad():
            noise = self.model(t)

        # 残差重建：clean = noisy - noise，再 clamp 到 [0,1]
        clean = (t - noise).clamp(0, 1)
        restored = (clean.squeeze(0).permute(1, 2, 0).cpu().numpy() * 255).astype(np.uint8)

        return RestoreResult(image=restored, meta={"model": "DnCNN-gaussian-color"})
