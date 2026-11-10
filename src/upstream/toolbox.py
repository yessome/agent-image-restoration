"""依赖 OpenCV 的独立图像处理工具箱，不参与实验 Agent 的默认路由。"""
from __future__ import annotations

from typing import Callable

import cv2
import numpy as np


def _rgb(image: np.ndarray) -> np.ndarray:
    if image.ndim == 2:
        return cv2.cvtColor(image, cv2.COLOR_GRAY2RGB)
    if image.shape[2] == 4:
        return cv2.cvtColor(image, cv2.COLOR_RGBA2RGB)
    return image


def clahe(image: np.ndarray) -> np.ndarray:
    lab = cv2.cvtColor(_rgb(image), cv2.COLOR_RGB2LAB)
    l, a, b = cv2.split(lab)
    l = cv2.createCLAHE(clipLimit=2.0, tileGridSize=(8, 8)).apply(l)
    return cv2.cvtColor(cv2.merge((l, a, b)), cv2.COLOR_LAB2RGB)


def auto_contrast(image: np.ndarray) -> np.ndarray:
    image = _rgb(image)
    out = np.empty_like(image)
    for channel in range(3):
        plane = image[:, :, channel]
        lo, hi = np.percentile(plane, (1, 99))
        if hi <= lo:
            out[:, :, channel] = plane
        else:
            out[:, :, channel] = np.clip((plane.astype(np.float32) - lo) * (255.0 / (hi - lo)), 0, 255)
    return out.astype(np.uint8)


def gamma_lift(image: np.ndarray) -> np.ndarray:
    lut = np.clip(np.power(np.arange(256, dtype=np.float32) / 255.0, 0.7) * 255, 0, 255).astype(np.uint8)
    return cv2.LUT(_rgb(image), lut)


def white_balance(image: np.ndarray) -> np.ndarray:
    image = _rgb(image).astype(np.float32)
    means = image.mean(axis=(0, 1))
    target = float(means.mean())
    gains = np.clip(target / np.maximum(means, 1.0), 0.75, 1.35)
    return np.clip(image * gains, 0, 255).astype(np.uint8)


def unsharp_mask(image: np.ndarray) -> np.ndarray:
    image = _rgb(image)
    soft = cv2.GaussianBlur(image, (0, 0), 1.2)
    return cv2.addWeighted(image, 1.6, soft, -0.6, 0)


def edge_detail(image: np.ndarray) -> np.ndarray:
    image = _rgb(image)
    smooth = cv2.bilateralFilter(image, 7, 45, 45)
    return cv2.addWeighted(image, 1.35, smooth, -0.35, 0)


def median_denoise(image: np.ndarray) -> np.ndarray:
    return cv2.medianBlur(_rgb(image), 3)


def bilateral_denoise(image: np.ndarray) -> np.ndarray:
    return cv2.bilateralFilter(_rgb(image), 7, 45, 45)


def nlmeans_denoise(image: np.ndarray) -> np.ndarray:
    return cv2.fastNlMeansDenoisingColored(_rgb(image), None, 5, 5, 7, 21)


def jpeg_cleanup(image: np.ndarray) -> np.ndarray:
    image = _rgb(image)
    ycrcb = cv2.cvtColor(image, cv2.COLOR_RGB2YCrCb)
    y, cr, cb = cv2.split(ycrcb)
    cr = cv2.bilateralFilter(cr, 5, 25, 25)
    cb = cv2.bilateralFilter(cb, 5, 25, 25)
    softened = cv2.cvtColor(cv2.merge((y, cr, cb)), cv2.COLOR_YCrCb2RGB)
    return cv2.addWeighted(image, 0.65, softened, 0.35, 0)


def dehaze(image: np.ndarray) -> np.ndarray:
    image = _rgb(image).astype(np.float32) / 255.0
    dark = cv2.erode(np.min(image, axis=2), np.ones((15, 15), np.uint8))
    atmospheric = np.percentile(image.reshape(-1, 3), 99, axis=0)
    transmission = np.clip(1.0 - 0.75 * dark, 0.25, 1.0)
    restored = (image - atmospheric) / transmission[:, :, None] + atmospheric
    return np.clip(restored * 255.0, 0, 255).astype(np.uint8)


def upscale_lanczos(image: np.ndarray) -> np.ndarray:
    image = _rgb(image)
    h, w = image.shape[:2]
    return cv2.resize(image, (w * 2, h * 2), interpolation=cv2.INTER_LANCZOS4)


def upscale_cubic(image: np.ndarray) -> np.ndarray:
    image = _rgb(image)
    h, w = image.shape[:2]
    return cv2.resize(image, (w * 2, h * 2), interpolation=cv2.INTER_CUBIC)


def saturation_boost(image: np.ndarray) -> np.ndarray:
    hsv = cv2.cvtColor(_rgb(image), cv2.COLOR_RGB2HSV).astype(np.float32)
    hsv[:, :, 1] = np.clip(hsv[:, :, 1] * 1.2, 0, 255)
    return cv2.cvtColor(hsv.astype(np.uint8), cv2.COLOR_HSV2RGB)


TOOLS: dict[str, tuple[str, Callable[[np.ndarray], np.ndarray]]] = {
    "clahe": ("局部对比度增强", clahe),
    "auto-contrast": ("自动色阶", auto_contrast),
    "gamma-lift": ("暗部提亮", gamma_lift),
    "white-balance": ("灰世界白平衡", white_balance),
    "unsharp": ("反锐化清晰化", unsharp_mask),
    "edge-detail": ("边缘细节增强", edge_detail),
    "median-denoise": ("中值滤波去噪", median_denoise),
    "bilateral-denoise": ("双边滤波去噪", bilateral_denoise),
    "nlmeans-denoise": ("非局部均值去噪", nlmeans_denoise),
    "jpeg-cleanup": ("JPEG 色块与色噪清理", jpeg_cleanup),
    "dehaze": ("暗通道去雾", dehaze),
    "upscale-lanczos": ("Lanczos 二倍放大", upscale_lanczos),
    "upscale-cubic": ("三次插值二倍放大", upscale_cubic),
    "saturation": ("饱和度增强", saturation_boost),
}


def apply_tool(name: str, image: np.ndarray) -> np.ndarray:
    """按工具名处理 RGB 或灰度 uint8 图像。"""
    try:
        operation = TOOLS[name][1]
    except KeyError as exc:
        raise ValueError(f"未知工具 {name!r}；可用工具：{', '.join(TOOLS)}") from exc
    result = operation(image)
    if result.dtype != np.uint8:
        result = np.clip(result, 0, 255).astype(np.uint8)
    return result
