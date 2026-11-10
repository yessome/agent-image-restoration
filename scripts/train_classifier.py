"""下游垃圾分类头微调脚本（TrashNet 6 类，ResNet-50）。

这是唯一允许训练的部分（下游分类头）。
上游 DnCNN/Zero-DCE 严格不训练。

用法：
    python scripts/train_classifier.py --data <TrashNet 根目录> \
        --out weights/trashnet_resnet50.pth --epochs 10 --device cuda

数据组织：<data> 下每个子目录名为类别（cardboard/glass/metal/paper/plastic/trash），
子目录内为图像。脚本自动按比例划分 train/val（默认 80/20，按类别分层）。

产出：单一 checkpoint（state_dict，含完整 resnet50 参数 + 6 类 fc），
可直接填入 config/pipelines.yaml 的 downstreams.classify.weight_path。
"""
from __future__ import annotations

import argparse
import os
import random
import sys

import numpy as np

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))


LABELS = ["cardboard", "glass", "metal", "paper", "plastic", "trash"]


def collect(data_dir: str):
    """按类别收集图像路径，返回 {label: [paths]}。"""
    from collections import defaultdict
    d = defaultdict(list)
    for label in sorted(os.listdir(data_dir)):
        sub = os.path.join(data_dir, label)
        if not os.path.isdir(sub):
            continue
        for fn in os.listdir(sub):
            if fn.lower().endswith((".png", ".jpg", ".jpeg", ".bmp", ".webp")):
                d[label].append(os.path.join(sub, fn))
    return d


def split(d: dict, val_ratio: float, seed: int):
    """按类别分层划分 train/val（快速微调用）。"""
    rng = random.Random(seed)
    train, val = [], []
    for label, paths in d.items():
        paths = list(paths)  # 拷贝，避免就地 shuffle 污染输入 dict（保证同 seed 可复现）
        rng.shuffle(paths)
        n_val = max(1, int(len(paths) * val_ratio))
        for p in paths[n_val:]:
            train.append((p, label))
        for p in paths[:n_val]:
            val.append((p, label))
    return train, val


def split_four(d: dict, seed: int, ratios=(0.6, 0.2, 0.1, 0.1)):
    """按类别分层划分四份：train / 验证集A / 验证集B / test。

    方案要求（避免选择器过拟合）：
    - 验证集 A：分类器早停
    - 验证集 B：路由阈值（δ_conf/τ_uncert/κ/clip_max）标定
    - test：最终只测一次，报告 Accuracy/Macro-F1/Regret

    返回 (train, valA, valB, test)，每份为 [(path, label)]。
    若某类样本不足以覆盖四份，抛错提示（避免极小类别训练集为空）。
    """
    rng = random.Random(seed)
    r_train, r_a, r_b, r_test = ratios
    train, valA, valB, test = [], [], [], []
    for label, paths in d.items():
        paths = list(paths)  # 拷贝，避免就地 shuffle 污染输入 dict（保证同 seed 可复现）
        rng.shuffle(paths)
        n = len(paths)
        # 每份至少 1 张，四份共需至少 4 张；不足则报错
        if n < 4:
            raise ValueError(
                f"类别 '{label}' 仅 {n} 张图，不足以划分 train/valA/valB/test 四份"
                f"（每份至少 1 张）。请补充数据或合并类别。"
            )
        n_test = max(1, int(n * r_test))
        n_b = max(1, int(n * r_b))
        n_a = max(1, int(n * r_a))
        # 依次切分（从尾部切 test，再 valB，再 valA，剩余 train）
        rest = paths
        test_part = rest[-n_test:]
        rest = rest[:-n_test]
        valB_part = rest[-n_b:]
        rest = rest[:-n_b]
        valA_part = rest[-n_a:]
        rest = rest[:-n_a]
        train_part = rest
        if not train_part:
            raise ValueError(
                f"类别 '{label}' 的 train 子集为空（共 {n} 张）。"
                f"请增大该类别数据量或调整划分比例。"
            )
        test += [(p, label) for p in test_part]
        valB += [(p, label) for p in valB_part]
        valA += [(p, label) for p in valA_part]
        train += [(p, label) for p in train_part]
    return train, valA, valB, test


def split_paper(d: dict, seed: int, ratios=(0.70, 0.13, 0.17)):
    """按原论文 70/13/17 对齐的三份划分，再扩展为项目所需的四份。

    原论文（Thung & Yang 2016, TrashNet）使用 70% train / 13% val / 17% test。
    本项目调度研究额外需要一个「路由阈值标定」验证集（valB），因此把原论文的
    13% val 拆成 valA（分类器早停）与 valB（路由阈值标定）两半：
      - train = 70%
      - valA  = 6.5%  } 合计 13% = 原论文 val
      - valB  = 6.5%  }
      - test  = 17%

    这样 valA+valB 之和与 test 严格对齐原论文比例，train 亦对齐；同时保留四份
    结构以支撑「分类器早停/阈值标定/test 只测一次」的调度实验协议。

    返回 (train, valA, valB, test)，每份为 [(path, label)]，按类别分层 + 固定 seed。
    """
    r_train, r_val, r_test = ratios
    r_a = r_b = r_val / 2.0
    rng = random.Random(seed)
    train, valA, valB, test = [], [], [], []
    for label, paths in d.items():
        paths = list(paths)  # 拷贝，避免就地 shuffle 污染输入 dict（保证同 seed 可复现）
        rng.shuffle(paths)
        n = len(paths)
        if n < 4:
            raise ValueError(
                f"类别 '{label}' 仅 {n} 张图，不足以划分 train/valA/valB/test 四份"
                f"（每份至少 1 张）。请补充数据或合并类别。"
            )
        n_test = max(1, round(n * r_test))
        n_b = max(1, round(n * r_b))
        n_a = max(1, round(n * r_a))
        # 依次切分（从尾部切 test，再 valB，再 valA，剩余 train）
        rest = paths
        test_part = rest[-n_test:]
        rest = rest[:-n_test]
        valB_part = rest[-n_b:]
        rest = rest[:-n_b]
        valA_part = rest[-n_a:]
        rest = rest[:-n_a]
        train_part = rest
        if not train_part:
            raise ValueError(
                f"类别 '{label}' 的 train 子集为空（共 {n} 张）。"
                f"请增大该类别数据量或调整划分比例。"
            )
        test += [(p, label) for p in test_part]
        valB += [(p, label) for p in valB_part]
        valA += [(p, label) for p in valA_part]
        train += [(p, label) for p in train_part]
    return train, valA, valB, test


def save_manifest(items, path):
    """保存 (path, label) 清单为 TSV。"""
    os.makedirs(os.path.dirname(path) or ".", exist_ok=True)
    with open(path, "w", encoding="utf-8") as f:
        for p, lb in items:
            f.write(f"{p}\t{lb}\n")
    print(f"  保存清单 -> {path} ({len(items)} 条)")


def main() -> int:
    parser = argparse.ArgumentParser(description="微调 ResNet-50 垃圾分类头")
    parser.add_argument("--data", required=True, help="TrashNet 根目录")
    parser.add_argument("--out", default="weights/trashnet_resnet50.pth", help="输出权重路径")
    parser.add_argument("--epochs", type=int, default=10)
    parser.add_argument("--batch-size", type=int, default=32)
    parser.add_argument("--lr", type=float, default=1e-3)
    parser.add_argument("--val-ratio", type=float, default=0.2)
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--split-seed", type=int, default=None,
                        help="数据划分种子（与 --seed 分离；None 时回退 --seed）。"
                             "多 seed 实验应固定 --split-seed、只改 --seed，"
                             "保证 test split 不变、只变训练随机性（模型初始化/数据顺序）。")
    parser.add_argument("--device", default="cuda")
    parser.add_argument("--freeze-backbone", action="store_true", default=True,
                        help="冻结 backbone，只训 fc（默认开启，符合方案「先训练分类头」；"
                             "加 --no-freeze-backbone 可全量微调）")
    parser.add_argument("--no-freeze-backbone", action="store_false", dest="freeze_backbone",
                        help="全量微调整个 ResNet-50（分类头精度不足时再启用）")
    parser.add_argument("--split-four", action="store_true",
                        help="按四份划分（train/valA/valB/test）并保存清单，"
                             "符合方案要求（避免选择器过拟合）")
    parser.add_argument("--split-paper", action="store_true",
                        help="按原论文 70/13/17 对齐划分（train=70%, valA=valB=6.5% 合计 13%, "
                             "test=17%）并保存清单，test 与论文 75% accuracy 可直接对比")
    parser.add_argument("--manifest-dir", default="data/splits",
                        help="四份清单输出目录（--split-four / --split-paper 时生效）")
    parser.add_argument("--splits-dir", default=None,
                        help="从已有清单读取 train.tsv/valA.tsv（跳过 collect+split），"
                             "用于数据去重后重训、保持 test 集不变")
    args = parser.parse_args()

    # 完整随机种子（保证可复现）：Python / NumPy / PyTorch / CUDA
    import random
    random.seed(args.seed)
    np.random.seed(args.seed)
    import torch
    import torch.nn as nn
    torch.manual_seed(args.seed)
    torch.cuda.manual_seed_all(args.seed)
    # 确定性（可选，牺牲速度换取可复现）
    try:
        torch.backends.cudnn.deterministic = True
        torch.backends.cudnn.benchmark = False
    except Exception:
        pass

    from torchvision import models, transforms
    from torch.utils.data import DataLoader, Dataset

    # 数据划分用独立的 split_seed（与训练随机性 args.seed 分离），
    # 保证多 seed 实验时 test split 不变（只改训练随机性）。
    split_seed = args.split_seed if args.split_seed is not None else args.seed

    if args.splits_dir:
        # 从已有清单读取 train/valA（数据去重后重训用），跳过 collect+split，
        # 保持 test 集不变（否则 split_paper 会重新切分并覆盖清单、且目录里仍含重复图）。
        def _read_tsv(p):
            items = []
            with open(p, encoding="utf-8") as f:
                for line in f:
                    line = line.rstrip("\n")
                    if not line or line.startswith("#"):
                        continue
                    parts = line.split("\t")
                    if len(parts) >= 2:
                        items.append((parts[0], parts[1]))
            return items
        train_list = _read_tsv(os.path.join(args.splits_dir, "train.tsv"))
        val_list = _read_tsv(os.path.join(args.splits_dir, "valA.tsv"))
        print(f"从已有清单读取：train={len(train_list)}  valA={len(val_list)}"
              f"（不重新 split，保持 test 集不变）")
    else:
        d = collect(args.data)
        print("各类样本数:", {k: len(v) for k, v in d.items()})

        if args.split_paper:
            train_list, valA_list, valB_list, test_list = split_paper(d, split_seed)
            print(f"train={len(train_list)}  valA={len(valA_list)}  valB={len(valB_list)}  test={len(test_list)}")
            save_manifest(train_list, os.path.join(args.manifest_dir, "train.tsv"))
            save_manifest(valA_list, os.path.join(args.manifest_dir, "valA.tsv"))
            save_manifest(valB_list, os.path.join(args.manifest_dir, "valB.tsv"))
            save_manifest(test_list, os.path.join(args.manifest_dir, "test.tsv"))
            print("\n提示：对齐原论文 70/13/17 划分。valA(6.5%)+valB(6.5%)=13%=原论文 val；"
                  "test=17% 可与论文 75% accuracy 直接对比。")
            val_list = valA_list  # 训练期间用 valA 做早停
        elif args.split_four:
            train_list, valA_list, valB_list, test_list = split_four(d, split_seed)
            print(f"train={len(train_list)}  valA={len(valA_list)}  valB={len(valB_list)}  test={len(test_list)}")
            save_manifest(train_list, os.path.join(args.manifest_dir, "train.tsv"))
            save_manifest(valA_list, os.path.join(args.manifest_dir, "valA.tsv"))
            save_manifest(valB_list, os.path.join(args.manifest_dir, "valB.tsv"))
            save_manifest(test_list, os.path.join(args.manifest_dir, "test.tsv"))
            print("\n提示：valA 用于分类器早停；valB 用于路由阈值标定；test 最终只测一次。")
            val_list = valA_list  # 训练期间用 valA 做早停
        else:
            train_list, val_list = split(d, args.val_ratio, split_seed)
            print(f"train={len(train_list)}  val={len(val_list)}")

    IMG = transforms.Compose([
        transforms.ToPILImage(),
        transforms.Resize(256),
        transforms.CenterCrop(224),
        transforms.ToTensor(),
        transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225]),
    ])

    class ClsDataset(Dataset):
        def __init__(self, items):
            self.items = items
        def __len__(self):
            return len(self.items)
        def __getitem__(self, idx):
            import cv2
            p, lb = self.items[idx]
            img = cv2.cvtColor(cv2.imread(p), cv2.COLOR_BGR2RGB)
            return IMG(img), LABELS.index(lb)

    train_ds = ClsDataset(train_list)
    val_ds = ClsDataset(val_list)
    # 完整随机种子（保证可复现）：DataLoader 的 generator（若 num_workers>0 也用固定 seed）
    g = torch.Generator()
    g.manual_seed(args.seed)
    train_loader = DataLoader(train_ds, batch_size=args.batch_size, shuffle=True,
                              num_workers=0, generator=g)
    val_loader = DataLoader(val_ds, batch_size=args.batch_size, shuffle=False, num_workers=0)

    model = models.resnet50(weights=models.ResNet50_Weights.IMAGENET1K_V1)
    in_features = model.fc.in_features
    model.fc = nn.Linear(in_features, len(LABELS))

    if args.freeze_backbone:
        for p in model.parameters():
            p.requires_grad = False
        for p in model.fc.parameters():
            p.requires_grad = True

    model = model.to(args.device)
    optimizer = torch.optim.AdamW(filter(lambda p: p.requires_grad, model.parameters()), lr=args.lr)
    criterion = nn.CrossEntropyLoss()

    best_acc = 0.0
    for epoch in range(args.epochs):
        model.train()
        total = correct = 0
        for x, y in train_loader:
            x, y = x.to(args.device), y.to(args.device)
            optimizer.zero_grad()
            logits = model(x)
            loss = criterion(logits, y)
            loss.backward()
            optimizer.step()
            total += y.size(0)
            correct += (logits.argmax(1) == y).sum().item()
        train_acc = correct / total

        model.eval()
        vtotal = vcorrect = 0
        with torch.no_grad():
            for x, y in val_loader:
                x, y = x.to(args.device), y.to(args.device)
                logits = model(x)
                vtotal += y.size(0)
                vcorrect += (logits.argmax(1) == y).sum().item()
        val_acc = vcorrect / vtotal
        print(f"epoch {epoch+1}/{args.epochs}  train_acc={train_acc:.4f}  val_acc={val_acc:.4f}")

        if val_acc > best_acc:
            best_acc = val_acc
            os.makedirs(os.path.dirname(args.out) or ".", exist_ok=True)
            torch.save(model.state_dict(), args.out)
            print(f"  保存最佳权重 -> {args.out}")

    print(f"\n完成。最佳 val_acc={best_acc:.4f}，权重已存 {args.out}")
    print("将该路径填入 config/pipelines.yaml 的 downstreams.classify.weight_path。")

    # 保存实验配置（复现用）：记录划分方式、seed、超参等
    exp_meta = {
        "seed": args.seed,
        "split_seed": split_seed,  # 数据划分种子（与训练 seed 分离，证明「固定划分、只变训练随机性」）
        "epochs": args.epochs,
        "batch_size": args.batch_size,
        "lr": args.lr,
        "freeze_backbone": args.freeze_backbone,
        "split_four": args.split_four,
        "split_paper": args.split_paper,
        "splits_dir": args.splits_dir,
        "val_ratio": args.val_ratio,
        "best_val_acc": best_acc,
        "num_classes": len(LABELS),
        "labels": LABELS,
    }
    import json
    meta_path = os.path.splitext(args.out)[0] + "_meta.json"
    with open(meta_path, "w", encoding="utf-8") as f:
        json.dump(exp_meta, f, ensure_ascii=False, indent=2)
    print(f"实验参数已保存 -> {meta_path}")

    return 0


if __name__ == "__main__":
    raise SystemExit(main())
