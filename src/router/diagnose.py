"""用亮度统计和中值滤波残差生成工具准入所需的诊断分数。"""
from __future__ import annotations

from dataclasses import dataclass, asdict
from typing import Dict

import cv2
import numpy as np


@dataclass
class Diagnosis:
    """诊断结果。"""
    noise_high: bool = False        # 是否高噪声（二值）
    lowlight: bool = False          # 是否低光
    noise_var: float = 0.0          # 噪声方差估计（中值滤波残差）
    brightness_mean: float = 0.0    # V 通道亮度均值
    brightness_median: float = 0.0  # V 通道亮度中位数（抗极端值）
    dark_pixel_ratio: float = 0.0   # 暗像素占比（V < 40）
    clip_ratio: float = 0.0         # 高光裁剪比例（V > 250）
    noise_score: float = 0.0        # 噪声置信度 [0,1]
    lowlight_score: float = 0.0     # 低光置信度 [0,1]

    def to_dict(self) -> Dict:
        return asdict(self)

    def summary(self) -> str:
        tags = []
        if self.noise_high:
            tags.append("噪声")
        if self.lowlight:
            tags.append("低光")
        return " + ".join(tags) if tags else "质量正常"

    def confident(self, threshold: float = 0.6) -> bool:
        return max(self.noise_score, self.lowlight_score) >= threshold


class ImageDiagnoser:
    """基于 OpenCV 统计量的质量诊断器（v3 多指标）。"""

    # 嵌套 YAML 键 -> 扁平阈值键 的映射（适配 config/pipelines.yaml 的嵌套结构）
    _NESTED_KEY_MAP = {
        "noise_high_threshold": [("noise", "high_threshold")],
        "lowlight_score_threshold": [("lowlight", "score_threshold")],
        "dark_pixel_threshold": [("lowlight", "dark_pixel_threshold")],
        "clip_threshold": [("lowlight", "clip_threshold")],
        "clip_penalty_lambda": [("lowlight", "clip_penalty_lambda")],
    }

    def __init__(self, thresholds: Dict[str, float] | None = None):
        self.thresholds = {
            "noise_high_threshold": 80.0,      # 噪声方差阈值（中值滤波残差，标定见 config）
            "lowlight_score_threshold": 0.6,   # 低光置信度阈值
            "dark_pixel_threshold": 40.0,      # 暗像素判定阈值（V 通道）
            "clip_threshold": 250.0,           # 高光裁剪判定阈值
        }
        if thresholds:
            self.thresholds.update(self._flatten(thresholds))

    @classmethod
    def _flatten(cls, raw: Dict) -> Dict:
        """把嵌套配置扁平化为内部阈值表。

        支持两种形态：
        1. 扁平：{"noise_high_threshold": 80.0, "lowlight_score_threshold": 0.6, ...}
        2. 嵌套：{"noise": {"high_threshold": 80.0}, "lowlight": {"score_threshold": 0.6, ...}}

        嵌套键通过 _NESTED_KEY_MAP 映射到扁平键，使 config/pipelines.yaml 的
        嵌套结构能正确生效（此前直接 dict.update 导致嵌套值被忽略）。
        """
        flat: Dict[str, float] = {}
        for flat_key, nested_paths in cls._NESTED_KEY_MAP.items():
            for path in nested_paths:
                node = raw
                ok = True
                for part in path:
                    if isinstance(node, dict) and part in node:
                        node = node[part]
                    else:
                        ok = False
                        break
                if ok and isinstance(node, (int, float)):
                    flat[flat_key] = float(node)
                    break
        # 也接纳直接以扁平键传入的配置（兼容旧用法）
        for k, v in raw.items():
            if k in cls._NESTED_KEY_MAP and isinstance(v, (int, float)):
                flat[k] = float(v)
        return flat

    def diagnose(self, image: np.ndarray) -> Diagnosis:
        """对输入图像做质量诊断（RGB 输入）。

        低光诊断（v3 多指标，抗黑色物体/阴影误判）：
        - 亮度均值、中位数（HSV V 通道）
        - 暗像素占比、高光裁剪比例
        """
        # 转 HSV，取 V 通道（亮度）
        hsv = cv2.cvtColor(image, cv2.COLOR_RGB2HSV)
        v = hsv[..., 2].astype(np.float32)
        gray = v

        # 残差对噪声和纹理都有响应。先转浮点数再相减，保留负残差。
        median = cv2.medianBlur(gray.astype(np.uint8), 3)
        noise = gray - median.astype(np.float32)
        noise_var = float(np.var(noise))

        # 2. 亮度统计（V 通道）
        brightness_mean = float(np.mean(v))
        brightness_median = float(np.median(v))

        # 3. 暗像素占比（V < dark_threshold）
        dark_th = self.thresholds["dark_pixel_threshold"]
        dark_pixel_ratio = float(np.mean(v < dark_th))

        # 4. 高光裁剪比例（V > clip_threshold）
        clip_th = self.thresholds["clip_threshold"]
        clip_ratio = float(np.mean(v > clip_th))

        # 5. 低光置信度（多指标加权 + 高光裁剪惩罚）
        lowlight_score = self._lowlight_score(
            brightness_mean, brightness_median, dark_pixel_ratio, clip_ratio
        )
        lowlight = lowlight_score >= self.thresholds["lowlight_score_threshold"]

        # 6. 噪声置信度
        noise_score = self._noise_score(noise_var)
        noise_high = noise_var > self.thresholds["noise_high_threshold"]

        return Diagnosis(
            noise_high=noise_high,
            lowlight=lowlight,
            noise_var=noise_var,
            brightness_mean=brightness_mean,
            brightness_median=brightness_median,
            dark_pixel_ratio=dark_pixel_ratio,
            clip_ratio=clip_ratio,
            noise_score=noise_score,
            lowlight_score=lowlight_score,
        )

    # ---------- 置信度公式（可复现） ----------

    def _noise_score(self, noise_var: float) -> float:
        """噪声方差 -> [0,1]（sigmoid 映射）。

        公式：score = 1 / (1 + exp(-(var - th) / (0.5*th)))
        阈值 th 在验证集标定。
        """
        import math
        th = self.thresholds["noise_high_threshold"]
        return 1.0 / (1.0 + math.exp(-(noise_var - th) / (th * 0.5 + 1e-6)))

    def _lowlight_score(
        self, brightness_mean, brightness_median, dark_ratio, clip_ratio
    ) -> float:
        """低光置信度（多指标加权 + 过曝惩罚，抗黑色物体/阴影误判）。

        公式：
            brightness_norm(x) = (1 - x/255)            # 越暗 → 越接近 1
            base = 0.4 * min(1, 2*dark_ratio)           # 暗像素占比，越暗越大
                 + 0.3 * brightness_norm(median)
                 + 0.3 * brightness_norm(mean)
            lowlight_score = clamp(base - λ * clip_ratio, 0, 1)

        高光裁剪比例作为过曝惩罚项：画面有明显高光（clip_ratio 高）说明不是
        整体低光，需压低低光分数，避免把过曝图误送入 Zero-DCE。
        权重与 λ 为经验值，在验证集标定。
        """
        # 暗像素占比（本身已是 [0,1]，放大使 >50% 即接近满）
        dark_term = min(1.0, dark_ratio * 2.0)

        # 亮度中位数/均值归一化（0~255 -> [0,1]，越暗越大）
        median_term = max(0.0, min(1.0, 1.0 - brightness_median / 255.0))
        mean_term = max(0.0, min(1.0, 1.0 - brightness_mean / 255.0))

        base = 0.4 * dark_term + 0.3 * median_term + 0.3 * mean_term

        # 高光裁剪惩罚（λ 为经验系数，在验证集标定）
        lam = self.thresholds.get("clip_penalty_lambda", 1.0)
        score = base - lam * clip_ratio
        return float(max(0.0, min(1.0, score)))
