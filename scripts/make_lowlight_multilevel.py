"""从 test.tsv 生成多强度合成低光测试集（gamma=2.0/2.5/3.0/3.5），泛化测试用。

关键：gamma=3.0 与旧版 make_test_degraded.py 的 lowlight 生成规则一致——
  BGR 读入 → 转 RGB → apply_gamma（逐通道幂运算）→ 转回 BGR → imwrite。
  apply_gamma 是确定性函数（无随机性），故 gamma=3.0 与旧 lowlight 逐字节一致。

输出（全部 E 盘）：
  data/degraded/lowlight_s{2,2.5,3,3.5}/ 目录 + 五列 manifest_eval.tsv

用法：
    python scripts/make_lowlight_multilevel.py --manifest data/splits/test.tsv
"""
from __future__ import annotations

import argparse
import hashlib
import json
import os
import sys

import cv2

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from scripts.degrade import apply_gamma


def _file_sha256(path: str) -> str:
    h = hashlib.sha256()
    with open(path, "rb") as f:
        for chunk in iter(lambda: f.read(1 << 20), b""):
            h.update(chunk)
    return h.hexdigest()


def _source_id(src_path: str) -> str:
    return _file_sha256(src_path)[:16]


def read_manifest(path):
    items = []
    with open(path, encoding="utf-8") as f:
        for line in f:
            line = line.rstrip("\n")
            if not line or line.startswith("#"):
                continue
            parts = line.split("\t")
            if len(parts) >= 2:
                items.append((parts[0], parts[1]))
    return items


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--manifest", default="data/splits/test.tsv")
    ap.add_argument("--gammas", default="2.0,2.5,3.0,3.5")
    ap.add_argument("--out-base", default="data/degraded")
    args = ap.parse_args()

    items = read_manifest(args.manifest)
    print(f"test 清单 {len(items)} 张")
    gammas = [float(x) for x in args.gammas.split(",")]

    for gamma in gammas:
        key = f"{gamma:g}"  # 2.0 -> "2", 2.5 -> "2.5", 3.0 -> "3", 3.5 -> "3.5"
        out_dir = os.path.join(args.out_base, f"lowlight_s{key}")
        os.makedirs(out_dir, exist_ok=True)
        rows = []
        n_fail = 0
        for i, (src, label) in enumerate(items, 1):
            img = cv2.imread(src, cv2.IMREAD_COLOR)
            if img is None:
                n_fail += 1
                continue
            # 复刻旧 lowlight 逻辑：BGR→RGB→gamma→BGR
            deg = apply_gamma(cv2.cvtColor(img, cv2.COLOR_BGR2RGB), gamma)
            deg = cv2.cvtColor(deg, cv2.COLOR_RGB2BGR)
            out_fn = f"{label}_{i:05d}.png"
            out_path = os.path.join(out_dir, out_fn)
            cv2.imwrite(out_path, deg)
            rows.append((os.path.abspath(out_path), label,
                         _source_id(os.path.abspath(src)), os.path.abspath(src),
                         json.dumps({"gamma": gamma}, sort_keys=True)))
        manifest = os.path.join(out_dir, "manifest_eval.tsv")
        with open(manifest, "w", encoding="utf-8") as f:
            f.write("# degraded_path\tlabel\tsource_id\tsource_path\tparams_json\n")
            for d, lb, sid, srcp, pj in rows:
                f.write(f"{d}\t{lb}\t{sid}\t{srcp}\t{pj}\n")
        print(f"gamma={key} -> {len(rows)} 张，清单 {manifest}（失败 {n_fail}）")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
