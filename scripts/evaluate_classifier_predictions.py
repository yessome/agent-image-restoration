"""六场景逐图预测评估（实验 A 用）：输出逐图 source_id/label/pred + 汇总。

与 eval_classifier_on_scenes.py 的区别：额外输出**逐图预测**（source_id 用于同源图配对分析）
和权重/manifest 哈希溯源。输出到独立目录，不覆盖任何已有结果。

用法：
    python scripts/evaluate_classifier_predictions.py \
        --weight "weights/robust_clean_budget_seed{seed}.pth" --seeds 1,2,3 \
        --out results/ablation/robust_classifier_baseline/clean_budget
"""
from __future__ import annotations

import argparse
import hashlib
import json
import os
import sys

import numpy as np

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
LABELS = ["cardboard", "glass", "metal", "paper", "plastic", "trash"]
SCENE_MANIFESTS = {
    "clean": "data/splits/test.tsv",
    "noise": "data/degraded/noise/manifest_eval.tsv",
    "lowlight": "data/degraded/lowlight/manifest_eval.tsv",
    "motion_blur": "data/degraded/motion_blur/manifest_eval.tsv",
    "jpeg": "data/degraded/jpeg/manifest_eval.tsv",
    "mixed": "data/degraded/mixed/manifest_eval.tsv",
}


def _sha256_file(path):
    if not path or not os.path.exists(path):
        return None
    h = hashlib.sha256()
    with open(path, "rb") as f:
        for chunk in iter(lambda: f.read(1 << 20), b""):
            h.update(chunk)
    return h.hexdigest()


def _read_manifest(path):
    """返回 [(abs_path, label, source_id)]。source_id 五列取第三列，两列算内容哈希前16位。"""
    items = []
    with open(path, encoding="utf-8") as f:
        for line in f:
            line = line.rstrip("\n")
            if not line or line.startswith("#"):
                continue
            parts = line.split("\t")
            if len(parts) < 2:
                continue
            p = parts[0]
            if not os.path.isabs(p):
                p = os.path.join(ROOT, p)
            label = parts[1]
            if len(parts) >= 3 and parts[2].strip():
                sid = parts[2].strip()
            else:
                sid = (_sha256_file(p) or "")[:16]
            items.append((p, label, sid))
    return items


def _macro_f1(y_true, y_pred, n_classes):
    f1s = []
    for c in range(n_classes):
        tp = sum(1 for t, p in zip(y_true, y_pred) if p == c and t == c)
        fp = sum(1 for t, p in zip(y_true, y_pred) if p == c and t != c)
        fn = sum(1 for t, p in zip(y_true, y_pred) if p != c and t == c)
        if tp + fp == 0 or tp + fn == 0:
            continue
        prec, rec = tp / (tp + fp), tp / (tp + fn)
        if prec + rec > 0:
            f1s.append(2 * prec * rec / (prec + rec))
    return float(np.mean(f1s)) if f1s else 0.0


def eval_one(weight_path, scene):
    import torch
    from torchvision import models, transforms
    from torch.utils.data import DataLoader, Dataset

    items = _read_manifest(os.path.join(ROOT, SCENE_MANIFESTS[scene]))
    IMG = transforms.Compose([
        transforms.ToPILImage(), transforms.Resize(256), transforms.CenterCrop(224),
        transforms.ToTensor(),
        transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225]),
    ])

    class D(Dataset):
        def __init__(self, items):
            self.items = items
        def __len__(self):
            return len(self.items)
        def __getitem__(self, idx):
            import cv2
            p, lb, sid = self.items[idx]
            img = cv2.cvtColor(cv2.imread(p), cv2.COLOR_BGR2RGB)
            return IMG(img), LABELS.index(lb), sid

    model = models.resnet50(weights=None)
    model.fc = torch.nn.Linear(model.fc.in_features, len(LABELS))
    model.load_state_dict(torch.load(weight_path, map_location="cpu"))
    device = "cuda" if torch.cuda.is_available() else "cpu"
    model = model.to(device)
    model.eval()

    records = []
    y_true, y_pred = [], []
    with torch.no_grad():
        for x, y, sids in DataLoader(D(items), batch_size=64, shuffle=False, num_workers=0):
            x = x.to(device)
            logits = model(x)
            pred = logits.argmax(1).cpu().tolist()
            for sid, lb, pr in zip(sids, y.tolist(), pred):
                records.append({"source_id": sid, "label": LABELS[lb],
                                "pred_label": LABELS[pr], "is_correct": lb == pr})
            y_pred.extend(pred)
            y_true.extend(y.tolist())

    acc = sum(1 for r in records if r["is_correct"]) / len(records)
    f1 = _macro_f1(y_true, y_pred, len(LABELS))
    return records, {"accuracy": round(acc, 4), "macro_f1": round(f1, 4), "n": len(records),
                     "weight_path": weight_path, "weight_sha256": _sha256_file(weight_path),
                     "manifest": SCENE_MANIFESTS[scene],
                     "manifest_sha256": _sha256_file(os.path.join(ROOT, SCENE_MANIFESTS[scene]))}


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--weight", required=True, help="权重路径，支持 {seed}")
    ap.add_argument("--seeds", default="1,2,3")
    ap.add_argument("--out", required=True)
    args = ap.parse_args()
    seeds = [int(s) for s in args.seeds.split(",") if s.strip()]

    os.makedirs(args.out, exist_ok=True)
    summary = {}
    for scene in SCENE_MANIFESTS:
        scene_accs, scene_f1s = [], []
        for s in seeds:
            wp = args.weight.replace("{seed}", str(s))
            records, meta = eval_one(wp, scene)
            scene_accs.append(meta["accuracy"])
            scene_f1s.append(meta["macro_f1"])
            jl = os.path.join(args.out, f"seed{s}", f"preds_{scene}.jsonl")
            os.makedirs(os.path.dirname(jl), exist_ok=True)
            with open(jl, "w", encoding="utf-8") as f:
                for r in records:
                    f.write(json.dumps(r, ensure_ascii=False) + "\n")
            print(f"[{scene}] seed{s}: acc={meta['accuracy']} f1={meta['macro_f1']} -> {jl}")
        summary[scene] = {
            "accuracy": {"mean": round(float(np.mean(scene_accs)), 4),
                         "std": round(float(np.std(scene_accs)), 4), "values": scene_accs},
            "macro_f1": {"mean": round(float(np.mean(scene_f1s)), 4),
                         "std": round(float(np.std(scene_f1s)), 4), "values": scene_f1s},
        }
    with open(os.path.join(args.out, "scene_summary.json"), "w", encoding="utf-8") as f:
        json.dump(summary, f, ensure_ascii=False, indent=2)
    print(f"\n汇总 -> {args.out}/scene_summary.json")
    print("\n=== 六场景 accuracy（mean±std）===")
    for scene, v in summary.items():
        print(f"{scene:12s} {v['accuracy']['mean']:.4f}±{v['accuracy']['std']:.4f}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
