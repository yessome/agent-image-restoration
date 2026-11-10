"""垃圾分类处理器（下游任务，真实推理）。

使用 ResNet-50（ImageNet 预训练）做垃圾分类。
注意：ImageNet 预训练输出 1000 类，无法直接识别 TrashNet 6 类，
需替换分类头并在 TrashNet 上微调。

本实现分两种模式：
- 未提供垃圾分类权重时：加载 ImageNet 预训练 ResNet-50，做通用 1000 类推理，
  可验证「诊断→修复→分类」整条链路真实可用（但类别是 ImageNet 语义）。
- 提供垃圾分类权重（weight_path）时：加载 6 类分类头权重，做真实垃圾 6 类推理。
"""
from __future__ import annotations

import os
from typing import Any, Dict, List

import numpy as np

from .base import DownstreamProcessor, TaskResult

# TrashNet 6 类默认标签
DEFAULT_LABELS = ["cardboard", "glass", "metal", "paper", "plastic", "trash"]

# ImageNet 归一化参数
IMAGENET_MEAN = [0.485, 0.456, 0.406]
IMAGENET_STD = [0.229, 0.224, 0.225]


class ClassifyProcessor(DownstreamProcessor):
    name = "classify"

    def __init__(self, device: str = "cuda", weight_path: str | None = None,
                 num_classes: int = 6, labels: List[str] | None = None):
        super().__init__(device=device, weight_path=weight_path)
        self.num_classes = num_classes
        self.labels = labels or DEFAULT_LABELS[:num_classes]
        self._backend = "imagenet"  # "imagenet" | "trashnet"

    def load(self) -> None:
        """加载分类器。

        优先加载垃圾 6 类权重（weight_path 提供时）；否则加载 ImageNet 预训练
        ResNet-50 做通用分类（验证链路用）。
        加载顺序：torchvision → timm（HF 镜像，规避 download.pytorch.org 被墙）。
        """
        import torch
        import torch.nn as nn

        # 1. 若有垃圾 6 类权重，加载它（strict 校验：缺参数即报错，不误判成功）
        if self.weight_path and os.path.exists(self.weight_path):
            try:
                from torchvision import models
                self.backbone = models.resnet50(weights=None)
                in_features = self.backbone.fc.in_features
                self.backbone.fc = nn.Linear(in_features, self.num_classes)
                sd = torch.load(self.weight_path, map_location="cpu", weights_only=False)
                sd = sd.get("state_dict", sd) if isinstance(sd, dict) else sd
                sd = {k.replace("module.", ""): v for k, v in sd.items()}
                # strict=True：六类 checkpoint 必须完整覆盖模型参数，
                # 否则视为加载失败（避免把缺参数的权重误认为成功）
                missing, unexpected = self.backbone.load_state_dict(sd, strict=False)
                if missing:
                    raise RuntimeError(
                        f"六类权重缺 {len(missing)} 个参数（如 {missing[:3]}...），"
                        f"拒绝加载以保持实验有效性"
                    )
                self._backend = "trashnet"
                self.model = self.backbone.to(self.device).eval()
                self._preprocess = self._make_preprocess()
                self._load_error = None
                return
            except Exception as e:
                self._load_error = str(e)
                raise RuntimeError(
                    f"垃圾六类权重加载失败（{self.weight_path}）：{e}"
                )

        # 2. 未提供垃圾权重：ImageNet 预训练仅用于验证「诊断→修复→分类」链路
        #    不具备 TrashNet 六分类语义；离线实验必须提供六类权重。
        try:
            from torchvision import models
            self.backbone = models.resnet50(weights=models.ResNet50_Weights.IMAGENET1K_V1)
            self._backend = "imagenet"
        except Exception:
            # torchvision 权重下载失败（download.pytorch.org 被墙）→ 回退 timm
            try:
                import timm
                self.backbone = timm.create_model("resnet50", pretrained=True)
                self._backend = "imagenet"
            except Exception as e:
                self.model = None
                self._load_error = str(e)
                return

        self.model = self.backbone.to(self.device).eval()
        self._preprocess = self._make_preprocess()

    @staticmethod
    def _make_preprocess():
        from torchvision import transforms
        return transforms.Compose([
            transforms.ToPILImage(),
            transforms.Resize(256),
            transforms.CenterCrop(224),
            transforms.ToTensor(),
            transforms.Normalize(mean=IMAGENET_MEAN, std=IMAGENET_STD),
        ])

    @staticmethod
    def _imagenet_classes() -> List[str]:
        """返回 ImageNet 1000 类类别名（torchvision 或 timm 来源）。"""
        try:
            from torchvision import models
            weights = models.ResNet50_Weights.IMAGENET1K_V1
            return list(weights.meta["categories"])
        except Exception:
            import timm
            m = timm.create_model("resnet50", pretrained=False)
            cfg = timm.data.resolve_data_config({}, model=m)
            return list(m.default_cfg.get("labels", [])) or \
                   [str(i) for i in range(1000)]

    def process(self, image: np.ndarray, labels: List[str] | None = None, **kwargs) -> TaskResult:
        """对（修复后的）图像执行分类。

        返回 TaskResult(result 为 {label: score} 按分数降序)。
        """
        self._assert_loaded()

        if self.model is None:
            raise RuntimeError(
                "分类器未加载（" + str(getattr(self, "_load_error", "未知错误")) + "）。"
                "离线六分类实验必须提供有效 TrashNet 权重，不得以均匀占位概率继续评价。"
            )

        import torch

        inp = self._preprocess(image).unsqueeze(0).to(self.device)
        with torch.no_grad():
            logits = self.model(inp)
            probs = logits.softmax(-1)[0]

        # 落盘完整 logits / softmax 概率与「类别顺序」，供下游机制分析
        # （真实类别 logit / 概率 gt_logit / gt_prob）与融合方法使用。
        # 只在 trashnet 后端落盘（六类，维度可控）；ImageNet 1000 类不落全量 logits。
        meta_extra = {"model": "resnet50", "backend": self._backend}

        if self._backend == "trashnet":
            # 垃圾 6 类
            use_labels = labels or self.labels
            scores = {lb: float(probs[i]) for i, lb in enumerate(use_labels)}
            logits_np = logits.detach().cpu().numpy().reshape(-1)
            probs_np = probs.detach().cpu().numpy().reshape(-1)
            meta_extra.update({
                # labels 顺序与 logits/probs 的第 i 维一一对应，用于定位 gt_logit/gt_prob
                "labels": list(use_labels),
                "logits": [float(x) for x in logits_np],
                "probs": [float(x) for x in probs_np],
            })
        else:
            # ImageNet 1000 类：取 top-5 类别
            class_names = self._imagenet_classes()
            topk = torch.topk(probs, k=5)
            scores = {class_names[idx]: float(p)
                      for p, idx in zip(topk.values, topk.indices)}

        scores = dict(sorted(scores.items(), key=lambda x: -x[1]))
        return TaskResult(
            task_type="classify", result=scores,
            meta=meta_extra,
        )
