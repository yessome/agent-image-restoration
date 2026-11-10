"""分层抽样 router 训练清单（类别平衡 + 按原图分组防泄漏）。

为什么需要抽样：
  全量 train_mixed_grouped.tsv（约 5300 样本）在 16 工具下要跑 17 个分支，
  单次特征收集约 12 小时以上。抽样在保持类别平衡的前提下把成本降到可接受区间。

抽样口径（与 train_router.py 的切分配合，三层防泄漏）：
  1. 按 group_id 抽：同一原图的 clean/lowlight/noise 三变体同进同出，
     退化类型天然 1:1:1 平衡（每 group 固定 3 变体），无需额外分层；
  2. 类别平衡：每类抽 --per-class 个 group（trash 组数最少，是瓶颈）；
  3. train_router.py 内部再按 group_id 切 train/val（--val-ratio），
     最终评估仍在独立的 valB/test 上完成。

用法：
    python scripts/sample_router_training.py \
        --per-class 60 --seed 42 \
        --out data/degraded/train_mixed_grouped_sampled_1080.tsv
"""
from __future__ import annotations

import argparse
import os
import random
import sys
from collections import defaultdict


def read_groups(manifest: str):
    """读清单，返回 (groups, order)。groups: gid -> [(path,label)]，order: gid 首次出现顺序。"""
    groups: dict = defaultdict(list)
    order: list = []
    with open(manifest, encoding="utf-8") as f:
        for line in f:
            line = line.rstrip("\n")
            if not line or line.startswith("#"):
                continue
            parts = line.split("\t")
            if len(parts) < 3:
                continue
            path, label, gid = parts[0], parts[1], parts[2].strip()
            if not gid:
                continue
            if gid not in groups:
                order.append(gid)
            groups[gid].append((path, label))
    return groups, order


def main() -> int:
    ap = argparse.ArgumentParser(description="分层抽样 router 训练清单")
    ap.add_argument("--manifest", default="data/degraded/train_mixed_grouped.tsv")
    ap.add_argument("--out", default="data/degraded/train_mixed_grouped_sampled_1080.tsv")
    ap.add_argument("--per-class", type=int, default=60,
                    help="每类抽取的 group 数（×3 变体 = 样本数；60 → 1080 样本）")
    ap.add_argument("--seed", type=int, default=42)
    args = ap.parse_args()

    if not os.path.exists(args.manifest):
        print(f"错误：清单不存在 {args.manifest}", file=sys.stderr)
        return 2

    groups, order = read_groups(args.manifest)
    if not groups:
        print("错误：清单为空或缺少 group_id（需 路径\\t标签\\tgroup_id 三列）", file=sys.stderr)
        return 2

    by_label: dict = defaultdict(list)
    inconsistent = 0
    for gid, items in groups.items():
        labels = {lb for _, lb in items}
        if len(labels) != 1:
            inconsistent += 1
            continue
        by_label[items[0][1]].append(gid)

    total = sum(len(v) for v in groups.values())
    print(f"清单：{len(groups)} 组 / {total} 样本" + (f"（{inconsistent} 组类别不一致已跳过）" if inconsistent else ""))

    rng = random.Random(args.seed)
    picked = set()
    print(f"{'类别':12s} {'组总数':>8s} {'抽取':>6s}")
    for label in sorted(by_label):
        gids = sorted(by_label[label])  # 排序后再抽，保证不受文件行序影响、可复现
        n = min(args.per_class, len(gids))
        picked.update(rng.sample(gids, n))
        print(f"{label:12s} {len(gids):8d} {n:6d}")

    # 按原清单顺序输出，保持可读性
    n_rows = 0
    os.makedirs(os.path.dirname(args.out) or ".", exist_ok=True)
    with open(args.out, "w", encoding="utf-8") as f:
        f.write("# 分层抽样清单（类别平衡 + group 防泄漏）\n")
        f.write(f"# 源清单: {args.manifest}\n")
        f.write(f"# per_class={args.per_class} seed={args.seed}\n")
        f.write("# 路径\t标签\tgroup_id\n")
        for gid in order:
            if gid not in picked:
                continue
            for path, label in groups[gid]:
                f.write(f"{path}\t{label}\t{gid}\n")
                n_rows += 1

    n_clean = sum(1 for gid in picked for _ in groups[gid])
    print(f"\n已写出 {args.out}")
    print(f"共 {len(picked)} 组 / {n_rows} 样本")
    print(f"平均每组变体数：{n_clean / max(1, len(picked)):.2f}（应为 3.00）")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
