"""为 clean / motion_blur / jpeg / mixed 四个场景构建评测清单。

命名约定已验证：`data/degraded/<scene>/<label>_<序号:05d>.png` 与 `data/splits/test.tsv`
的行序一一对应（430/430 命中）。group 统一取源图路径的哈希，保证跨场景同源一致。
"""
from __future__ import annotations

import hashlib
import json
import os

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
OUT = os.path.join(ROOT, "results", "table2_scenes")
SCENES = {
    "clean": None,                                     # 直接用 test 源图
    "motion_blur": os.path.join(ROOT, "data", "degraded", "motion_blur"),
    "jpeg": os.path.join(ROOT, "data", "degraded", "jpeg"),
    "mixed": os.path.join(ROOT, "data", "degraded", "mixed"),
}


def main() -> int:
    os.makedirs(OUT, exist_ok=True)
    rows = [l.rstrip("\n").split("\t") for l in open(os.path.join(ROOT, "data", "splits", "test.tsv"),
                                                     encoding="utf-8")
            if l.strip() and not l.startswith("#")]
    print(f"test.tsv {len(rows)} 行")
    summary = {}
    for scene, d in SCENES.items():
        out_rows, missing = [], 0
        for i, r in enumerate(rows, 1):
            src, label = r[0], r[1]
            g = hashlib.sha256(src.replace("\\", "/").encode()).hexdigest()[:16]
            if scene == "clean":
                path = os.path.join(ROOT, src.replace("\\", os.sep))
            else:
                path = os.path.join(d, f"{label}_{i:05d}.png")
            if not os.path.isfile(path):
                missing += 1
                continue
            out_rows.append((path, label, g))
        p = os.path.join(OUT, f"{scene}.tsv")
        with open(p, "w", encoding="utf-8") as f:
            f.write("# path\tlabel\tsource_id\n")
            for a, b, c in out_rows:
                f.write(f"{a}\t{b}\t{c}\n")
        summary[scene] = {"n": len(out_rows), "missing": missing, "manifest": os.path.relpath(p, ROOT)}
        print(f"  {scene:12s} {len(out_rows):3d} 行 | 缺失 {missing}")
    json.dump(summary, open(os.path.join(OUT, "scenes.json"), "w", encoding="utf-8"),
              ensure_ascii=False, indent=2)
    print(f"写出 {OUT}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
