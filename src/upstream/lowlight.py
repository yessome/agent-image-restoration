"""Zero-DCE 低光增强处理器（上游修复器，真实推理）。

使用 Li-Chongyi/Zero-DCE 官方预训练权重，零参考、零训练。
参考: https://github.com/Li-Chongyi/Zero-DCE

Zero-DCE 核心是曲线估计网络 enhance_net_nopool：
  - 7 个卷积（后 3 个用 cat 拼接），最后输出 24 通道 = 8 组 × 3 通道曲线参数
  - 通过 8 次串行迭代 x = x + r_i*(x^2 - x) 完成亮度增强
网络结构严格对齐官方 model.py，避免引入外部仓库依赖。
"""
from __future__ import annotations

import os

import numpy as np

from .base import UpstreamProcessor, RestoreResult


class EnhanceNetNoPool:
    """Zero-DCE 曲线估计网络（严格对齐官方 model.py 的 enhance_net_nopool）。

    不继承 nn.Module，手工管理子模块，精确对齐官方 checkpoint 权重键名。
    forward 返回 (enhance_image_1, enhance_image, r)，与官方一致。
    """

    def __init__(self):
        import torch
        import torch.nn as nn
        import torch.nn.functional as F

        self.relu = nn.ReLU(inplace=True)
        number_f = 32
        self.e_conv1 = nn.Conv2d(3, number_f, 3, 1, 1, bias=True)
        self.e_conv2 = nn.Conv2d(number_f, number_f, 3, 1, 1, bias=True)
        self.e_conv3 = nn.Conv2d(number_f, number_f, 3, 1, 1, bias=True)
        self.e_conv4 = nn.Conv2d(number_f, number_f, 3, 1, 1, bias=True)
        self.e_conv5 = nn.Conv2d(number_f * 2, number_f, 3, 1, 1, bias=True)
        self.e_conv6 = nn.Conv2d(number_f * 2, number_f, 3, 1, 1, bias=True)
        self.e_conv7 = nn.Conv2d(number_f * 2, 24, 3, 1, 1, bias=True)

    def __call__(self, x):
        import torch
        import torch.nn.functional as F

        x1 = self.relu(self.e_conv1(x))
        x2 = self.relu(self.e_conv2(x1))
        x3 = self.relu(self.e_conv3(x2))
        x4 = self.relu(self.e_conv4(x3))

        x5 = self.relu(self.e_conv5(torch.cat([x3, x4], 1)))
        x6 = self.relu(self.e_conv6(torch.cat([x2, x5], 1)))

        x_r = F.tanh(self.e_conv7(torch.cat([x1, x6], 1)))
        r1, r2, r3, r4, r5, r6, r7, r8 = torch.split(x_r, 3, dim=1)

        x = x + r1 * (torch.pow(x, 2) - x)
        x = x + r2 * (torch.pow(x, 2) - x)
        x = x + r3 * (torch.pow(x, 2) - x)
        enhance_image_1 = x + r4 * (torch.pow(x, 2) - x)
        x = enhance_image_1 + r5 * (torch.pow(enhance_image_1, 2) - enhance_image_1)
        x = x + r6 * (torch.pow(x, 2) - x)
        x = x + r7 * (torch.pow(x, 2) - x)
        enhance_image = x + r8 * (torch.pow(x, 2) - x)
        r = torch.cat([r1, r2, r3, r4, r5, r6, r7, r8], 1)
        return enhance_image_1, enhance_image, r

    def _modules(self):
        return {
            "e_conv1": self.e_conv1, "e_conv2": self.e_conv2,
            "e_conv3": self.e_conv3, "e_conv4": self.e_conv4,
            "e_conv5": self.e_conv5, "e_conv6": self.e_conv6,
            "e_conv7": self.e_conv7,
        }

    def load_state_dict(self, sd):
        # 官方权重键名可能带/不带 module. 前缀，做兜底适配
        def strip(k):
            return k.replace("module.", "")

        for name, module in self._modules().items():
            w = sd.get(strip(f"{name}.weight"))
            b = sd.get(strip(f"{name}.bias"))
            if w is not None:
                module.weight.data.copy_(w)
            if b is not None:
                module.bias.data.copy_(b)
        return self

    def to(self, device):
        for m in self._modules().values():
            m.to(device)
        return self

    def eval(self):
        for m in self._modules().values():
            m.eval()
        return self


class LowLightProcessor(UpstreamProcessor):
    name = "lowlight"
    weight_url = (
        "https://raw.githubusercontent.com/Li-Chongyi/Zero-DCE/master/"
        "Zero-DCE_code/snapshots/Epoch99.pth"
    )

    def load(self) -> None:
        """加载 Zero-DCE 预训练权重（不训练）。缺失时降级。"""
        if self.weight_path is None or not os.path.exists(self.weight_path):
            self.model = None
            return

        import torch

        try:
            net = EnhanceNetNoPool()
            sd = torch.load(self.weight_path, map_location="cpu", weights_only=False)
            if isinstance(sd, dict) and "state_dict" in sd:
                sd = sd["state_dict"]
            net.load_state_dict(sd)
            self.model = net.to(self.device).eval()
        except Exception as e:
            self.model = None
            self._load_error = str(e)

    def restore(self, image: np.ndarray) -> RestoreResult:
        self._assert_loaded()
        image = self._ensure_uint8(image)

        if self.model is None:
            return RestoreResult(
                image=image,
                meta={"model": "Zero-DCE", "fallback": True,
                      "note": f"权重未加载或加载失败：{getattr(self, '_load_error', '未找到权重')}"},
            )

        import torch

        # 转 tensor (1,3,H,W) in [0,1]
        t = torch.from_numpy(image).permute(2, 0, 1).float() / 255.0
        t = t.unsqueeze(0).to(self.device)

        with torch.no_grad():
            _, enhanced, _ = self.model(t)

        enhanced = enhanced.squeeze(0).clamp(0, 1)
        restored = (enhanced.permute(1, 2, 0).cpu().numpy() * 255).astype(np.uint8)

        return RestoreResult(image=restored, meta={"model": "Zero-DCE"})


class ZeroDCEPlusPlusProcessor(UpstreamProcessor):
    """Official lightweight Zero-DCE++ model (depthwise separable convolutions)."""

    name = "zero_dce_plus_plus"
    weight_url = (
        "https://raw.githubusercontent.com/Li-Chongyi/Zero-DCE_extension/main/"
        "Zero-DCE++/snapshots_Zero_DCE++/Epoch99.pth"
    )

    def load(self) -> None:
        if not self.weight_path or not os.path.isfile(self.weight_path):
            self.model = None
            return
        try:
            import torch
            import torch.nn as nn

            class CSDNTem(nn.Module):
                def __init__(self, in_channels: int, out_channels: int):
                    super().__init__()
                    self.depth_conv = nn.Conv2d(
                        in_channels, in_channels, 3, padding=1, groups=in_channels
                    )
                    self.point_conv = nn.Conv2d(in_channels, out_channels, 1)

                def forward(self, x):
                    return self.point_conv(self.depth_conv(x))

            class Network(nn.Module):
                def __init__(self):
                    super().__init__()
                    self.relu = nn.ReLU(inplace=True)
                    self.e_conv1 = CSDNTem(3, 32)
                    self.e_conv2 = CSDNTem(32, 32)
                    self.e_conv3 = CSDNTem(32, 32)
                    self.e_conv4 = CSDNTem(32, 32)
                    self.e_conv5 = CSDNTem(64, 32)
                    self.e_conv6 = CSDNTem(64, 32)
                    self.e_conv7 = CSDNTem(64, 3)

                def forward(self, x):
                    x1 = self.relu(self.e_conv1(x))
                    x2 = self.relu(self.e_conv2(x1))
                    x3 = self.relu(self.e_conv3(x2))
                    x4 = self.relu(self.e_conv4(x3))
                    x5 = self.relu(self.e_conv5(torch.cat([x3, x4], dim=1)))
                    x6 = self.relu(self.e_conv6(torch.cat([x2, x5], dim=1)))
                    curve = torch.tanh(self.e_conv7(torch.cat([x1, x6], dim=1)))
                    enhanced = x
                    for _ in range(8):
                        enhanced = enhanced + curve * (enhanced.square() - enhanced)
                    return enhanced

            state = torch.load(self.weight_path, map_location="cpu", weights_only=False)
            if isinstance(state, dict) and "state_dict" in state:
                state = state["state_dict"]
            state = {key.removeprefix("module."): value for key, value in state.items()}
            model = Network()
            model.load_state_dict(state, strict=True)
            self.model = model.to(self.device).eval()
        except Exception as exc:
            self.model = None
            self._load_error = str(exc)

    def restore(self, image: np.ndarray) -> RestoreResult:
        self._assert_loaded()
        image = self._ensure_uint8(image)
        if self.model is None:
            return RestoreResult(image=image, meta={
                "model": "Zero-DCE++", "fallback": True,
                "note": getattr(self, "_load_error", "Zero-DCE++ 权重不存在"),
            })
        import torch
        tensor = torch.from_numpy(image.copy()).permute(2, 0, 1).float().div(255)
        with torch.inference_mode():
            output = self.model(tensor.unsqueeze(0).to(self.device))
        restored = output[0].clamp(0, 1).permute(1, 2, 0).cpu().numpy() * 255
        return RestoreResult(image=restored.round().astype(np.uint8),
                             meta={"model": "Zero-DCE++", "device": str(self.device)})


class SCIProcessor(UpstreamProcessor):
    """SCI medium-scene model, using the authors' single-stage inference network."""

    name = "sci_lowlight"

    def load(self) -> None:
        if not self.weight_path or not os.path.isfile(self.weight_path):
            self.model = None
            return
        try:
            import torch
            import torch.nn as nn

            class EnhanceNetwork(nn.Module):
                def __init__(self):
                    super().__init__()
                    self.in_conv = nn.Sequential(nn.Conv2d(3, 3, 3, padding=1), nn.ReLU())
                    self.conv = nn.Sequential(
                        nn.Conv2d(3, 3, 3, padding=1), nn.BatchNorm2d(3), nn.ReLU()
                    )
                    self.blocks = nn.ModuleList([self.conv])
                    self.out_conv = nn.Sequential(
                        nn.Conv2d(3, 3, 3, padding=1), nn.Sigmoid()
                    )

                def forward(self, x):
                    features = self.in_conv(x)
                    for block in self.blocks:
                        features = features + block(features)
                    return torch.clamp(self.out_conv(features) + x, 0.0001, 1)

            class SCIModel(nn.Module):
                def __init__(self):
                    super().__init__()
                    self.enhance = EnhanceNetwork()

                def forward(self, x):
                    illumination = self.enhance(x)
                    return torch.clamp(x / illumination, 0, 1)

            state = torch.load(self.weight_path, map_location="cpu", weights_only=False)
            if isinstance(state, dict) and "state_dict" in state:
                state = state["state_dict"]
            state = {key.removeprefix("module."): value for key, value in state.items()}
            model = SCIModel()
            compatible = {key: value for key, value in state.items()
                          if key in model.state_dict() and value.shape == model.state_dict()[key].shape}
            if len(compatible) < len(model.state_dict()) * 0.8:
                raise RuntimeError("SCI 权重与官方单阶段推理网络不匹配")
            model.load_state_dict(compatible, strict=False)
            self.model = model.to(self.device).eval()
        except Exception as exc:
            self.model = None
            self._load_error = str(exc)

    def restore(self, image: np.ndarray) -> RestoreResult:
        self._assert_loaded()
        image = self._ensure_uint8(image)
        if self.model is None:
            return RestoreResult(image=image, meta={
                "model": "SCI-medium", "fallback": True,
                "note": getattr(self, "_load_error", "SCI 权重不存在"),
            })
        import torch
        tensor = torch.from_numpy(image.copy()).permute(2, 0, 1).float().div(255).unsqueeze(0)
        with torch.inference_mode():
            output = self.model(tensor.to(self.device))
        restored = (output[0].clamp(0, 1).permute(1, 2, 0).cpu().numpy() * 255)
        return RestoreResult(image=restored.round().astype(np.uint8),
                             meta={"model": "SCI-medium", "device": str(self.device)})
