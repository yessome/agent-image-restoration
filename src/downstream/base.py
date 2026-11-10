"""下游任务抽象基类。

铁律：下游任务同样不训练，使用官方预训练权重。
统一接口保证上游修复结果可无缝喂给任意下游任务。
"""
from __future__ import annotations

import abc
from dataclasses import dataclass, field
from typing import Any, Dict, Optional

import numpy as np


@dataclass
class TaskResult:
    """下游任务的输出结果。"""
    task_type: str                      # 任务类型（"classify"）
    result: Any                         # 任务特定输出（mask / label 等）
    meta: Dict[str, Any] = field(default_factory=dict)


class DownstreamProcessor(abc.ABC):
    """所有下游任务的统一抽象接口。

    子类实现 load() 和 process()，
    路由层以统一方式调度，实现 上游+下游 自由组合。
    """

    name: str = "downstream"

    def __init__(self, device: str = "cuda", weight_path: Optional[str] = None):
        self.device = device
        self.weight_path = weight_path
        self.model: Any = None

    @abc.abstractmethod
    def load(self) -> None:
        """加载预训练权重。"""
        raise NotImplementedError

    @abc.abstractmethod
    def process(self, image: np.ndarray, **kwargs) -> TaskResult:
        """对（修复后的）图像执行下游任务。

        参数:
            image: 输入图像 (H, W, 3) uint8
            **kwargs: 任务特定参数（如分类候选标签）
        返回:
            TaskResult
        """
        raise NotImplementedError

    def _assert_loaded(self) -> None:
        if self.model is None:
            self.load()
