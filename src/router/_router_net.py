"""LearnedRouter 的网络结构（轻量 MLP）。"""
from __future__ import annotations

import torch.nn as nn


class RouterNet(nn.Module):
    """特征向量 → N 个分支的预测正确概率（logits）。"""

    def __init__(self, input_dim: int = 71, hidden: int = 64, num_branches: int = 13):
        super().__init__()
        self.net = nn.Sequential(
            nn.Linear(input_dim, hidden),
            nn.ReLU(),
            nn.Dropout(0.1),
            nn.Linear(hidden, hidden // 2),
            nn.ReLU(),
            nn.Linear(hidden // 2, num_branches),
        )

    def forward(self, x):
        return self.net(x)
