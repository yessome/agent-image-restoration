"""从 17 分支缓存切片出 13 分支（去掉 2 个去模糊 + 2 个超分）训练缓存。

因为 13 分支正好是 17 分支的**前缀**，特征列与标签列可以直接切：
  X[:, :6+13*5] = X[:, :71]，Y[:, :13]
不需要重跑任何修复模型或分类器。
"""
from __future__ import annotations

import json
import os
import shutil

import numpy as np

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
SRC_ROOT = os.path.join(ROOT, "results", "router_train_cache")
EXCLUDE = ["restormer_defocus", "mprnet_deblur", "realesrgan_x2", "realesrgan_x4"]
SEEDS = (1, 2, 3)


def main() -> int:
    for s in SEEDS:
        src = os.path.join(SRC_ROOT, f"iter_seed{s}_1080")
        dst = os.path.join(SRC_ROOT, f"iter13_seed{s}_1080")
        X = np.load(os.path.join(src, "X.npy"))
        Y = np.load(os.path.join(src, "Y.npy"))
        meta = json.load(open(os.path.join(src, "meta.json"), encoding="utf-8"))
        full_order = list(meta["branch_order"])

        order13 = [u for u in full_order if u not in set(EXCLUDE)]
        assert order13 == full_order[: len(order13)], "子集不是前缀，无法切片"
        n13 = len(order13)
        feat13 = 6 + 5 * n13

        os.makedirs(dst, exist_ok=True)
        np.save(os.path.join(dst, "X.npy"), X[:, :feat13].astype(np.float32))
        np.save(os.path.join(dst, "Y.npy"), Y[:, :n13].astype(np.float32))
        shutil.copy(os.path.join(src, "G.json"), os.path.join(dst, "G.json"))

        meta13 = dict(meta)
        meta13.update({
            "branch_order": order13,
            "feature_dim": feat13,
            "state_dim": feat13 + n13 + 1,
            "excluded_branches": EXCLUDE,
            "note": "由 iter_seed%d_1080 切片（17→13 分支），修复结果与标签未重新计算" % s,
            "sliced_from": os.path.relpath(src, ROOT),
        })
        json.dump(meta13, open(os.path.join(dst, "meta.json"), "w", encoding="utf-8"),
                  ensure_ascii=False, indent=2)
        print(f"seed{s}: X {X.shape}→{X[:, :feat13].shape} | Y {Y.shape}→{Y[:, :n13].shape} "
              f"| state_dim {feat13 + n13 + 1}")
    print(f"写出 {SRC_ROOT}/iter13_seed{{1,2,3}}_1080")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
