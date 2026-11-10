"""可复现的低质量退化生成底层函数（仅供 import）。

本文件只提供退化底层函数，被以下实验脚本 import：
  - `make_test_degraded.py`（test 退化集生成）
  - `degrade_manifest.py`（train 退化集生成）

实验退化参数统一为：低光 gamma=3.0、噪声高斯 sigma=25、固定 seed=42，
由上述两个脚本直接传参（不再通过本文件的分级入口）。

支持的退化类型：
  - gamma：Gamma 亮度下降（gamma > 1 变暗），模拟低光/欠曝光
  - gaussian：加性高斯噪声
  - poisson：泊松噪声（光子散粒噪声，低光下更显著）
"""
from __future__ import annotations

import cv2
import numpy as np


def apply_gamma(img_uint8: np.ndarray, gamma: float) -> np.ndarray:
    """Gamma 亮度下降（gamma > 1 变暗）。输入输出均为 uint8 RGB。"""
    f = img_uint8.astype(np.float32) / 255.0
    out = np.power(f, gamma)
    return (np.clip(out, 0, 1) * 255).astype(np.uint8)


def apply_gaussian(img_uint8: np.ndarray, sigma: float, rng: np.random.Generator) -> np.ndarray:
    """加性高斯噪声。"""
    noise = rng.normal(0, sigma, img_uint8.shape).astype(np.float32)
    out = img_uint8.astype(np.float32) + noise
    return np.clip(out, 0, 255).astype(np.uint8)


def apply_poisson(img_uint8: np.ndarray, rng: np.random.Generator) -> np.ndarray:
    """泊松噪声（光子散粒噪声）。"""
    img = img_uint8.astype(np.float32)
    noisy = rng.poisson(img)
    return np.clip(noisy, 0, 255).astype(np.uint8)


def apply_motion_blur(img_uint8: np.ndarray, kernel_size: int = 15,
                      angle_deg: float = 45.0) -> np.ndarray:
    """运动模糊：沿指定方向的方向性平均核做卷积（模拟相机/物体运动）。

    核大小 kernel_size、方向 angle_deg（度数，0=水平、90=垂直）。参数固定、可复现。
    """
    import cv2 as _cv2
    k = max(3, int(kernel_size) | 1)  # 保证奇数
    c = k // 2
    kernel = np.zeros((k, k), dtype=np.float32)
    rad = np.deg2rad(angle_deg)
    dx, dy = np.cos(rad), np.sin(rad)
    for t in np.linspace(-(k // 2), k // 2, k):
        x = int(round(c + t * dx))
        y = int(round(c + t * dy))
        if 0 <= x < k and 0 <= y < k:
            kernel[y, x] = 1.0
    kernel /= kernel.sum() if kernel.sum() > 0 else 1.0
    return _cv2.filter2D(img_uint8, -1, kernel)


def apply_jpeg(img_uint8: np.ndarray, quality: int = 30) -> np.ndarray:
    """JPEG 压缩伪影：按 quality 编码再解码。"""
    import cv2 as _cv2
    ok, buf = _cv2.imencode(".jpg", img_uint8,
                            [int(_cv2.IMWRITE_JPEG_QUALITY), int(quality)])
    if not ok:
        return img_uint8
    return _cv2.imdecode(buf, _cv2.IMREAD_COLOR)


def apply_mixed(img_uint8: np.ndarray, rng: np.random.Generator) -> np.ndarray:
    """混合退化（预注册固定顺序）：低光 gamma=3.0 → 高斯噪声 sigma=25。

    模拟真实场景中「低光 + 传感器噪声」同时出现，参数与单退化实验一致。
    """
    img = apply_gamma(img_uint8, 3.0)
    return apply_gaussian(img, 25.0, rng)
