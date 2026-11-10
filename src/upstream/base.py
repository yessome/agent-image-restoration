"""上游修复器抽象基类。

铁律：Agent 只做调度，不训练修复器。所有上游处理器只负责
加载官方预训练权重 + 前向推理，不包含任何训练逻辑。
"""
from __future__ import annotations

import abc
from dataclasses import dataclass, field
from typing import Any, Dict, Optional

import numpy as np


@dataclass
class RestoreResult:
    """上游修复器的输出结果。"""
    image: np.ndarray                 # 修复后的图像 (H, W, 3) uint8
    meta: Dict[str, Any] = field(default_factory=dict)  # 附加信息（耗时、模型名等）

    @property
    def shape(self) -> tuple:
        return self.image.shape


class UpstreamProcessor(abc.ABC):
    """所有上游修复器的统一抽象接口。

    子类只需实现 load() 和 restore() 两个方法，
    保证路由层能以统一方式调度任意修复器。
    """

    name: str = "upstream"

    def __init__(self, device: str = "cuda", weight_path: Optional[str] = None):
        self.device = device
        self.weight_path = weight_path
        self.model: Any = None

    @abc.abstractmethod
    def load(self) -> None:
        """加载预训练权重。不包含训练逻辑。"""
        raise NotImplementedError

    @abc.abstractmethod
    def restore(self, image: np.ndarray) -> RestoreResult:
        """对输入图像执行修复/增强，返回结果。

        参数:
            image: 输入图像 (H, W, 3) uint8，RGB 顺序
        返回:
            RestoreResult
        """
        raise NotImplementedError

    def _assert_loaded(self) -> None:
        if self.model is None:
            self.load()

    @staticmethod
    def _ensure_uint8(image: np.ndarray) -> np.ndarray:
        """将图像规整为 uint8 RGB。"""
        if image.dtype != np.uint8:
            image = np.clip(image, 0, 255).astype(np.uint8)
        if image.ndim == 2:  # 灰度 -> RGB
            image = np.stack([image] * 3, axis=-1)
        return image
