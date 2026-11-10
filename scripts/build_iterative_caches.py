"""为三个 classifier seed 构建循环 router 的训练特征缓存。

关键点：
  - 修复结果**跨 seed 共享**：同一张图的每个候选工具只真正跑一次；
  - 每个分类器分别产生自己的反馈（conf/margin）与正确性标签（Y）；
  - 三个缓存独立落盘，互不覆盖，且记录各自的 classifier 校验和。

产物（每个 seed 一份，格式与 train_iterative_router.py 的 load_train_cache 对齐）：
  results/router_train_cache/iter_seed{N}_1080/{X.npy, Y.npy, G.json, meta.json}

用法：
  python scripts/build_iterative_caches.py --verify 4     # 等价性校验（不落盘全量）
  python scripts/build_iterative_caches.py                # 全量构建
"""
from __future__ import annotations

import argparse
import hashlib
import json
import os
import sys
import time

import cv2
import numpy as np

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, ROOT)

from src.agents.orchestrator import ImageRestorationAgent, PipelineOutcome  # noqa: E402
from src.downstream.classify import ClassifyProcessor  # noqa: E402
from src.main import _resolve_weight_paths, load_config  # noqa: E402
from src.router.learned_router import FEATURE_DIM, extract_features  # noqa: E402
from src.router.tool_registry import BRANCH_ORDER  # noqa: E402

MANIFEST = os.path.join(ROOT, "data", "degraded", "train_mixed_grouped_sampled_1080.tsv")
CACHE_ROOT = os.path.join(ROOT, "results", "router_train_cache")


def fsha(path):
    h = hashlib.sha256()
    with open(path, "rb") as f:
        for c in iter(lambda: f.read(1 << 20), b""):
            h.update(c)
    return h.hexdigest()


def read_manifest():
    rows = []
    for line in open(MANIFEST, encoding="utf-8"):
        line = line.strip()
        if line and not line.startswith("#"):
            p = line.split("\t")
            rows.append({"path": p[0], "label": p[1], "group": p[2]})
    return rows


def build_agent(classifier_weight, device):
    cfg = _resolve_weight_paths(load_config(os.path.join(ROOT, "config", "pipelines.yaml")), ROOT)
    cfg.setdefault("model_toolbox", {})["enabled"] = True
    cfg.setdefault("router", {})["learned_weight"] = ""
    cfg.setdefault("routing", {})["mode"] = "legacy"
    cfg.setdefault("llm", {})["enabled"] = False
    cfg.setdefault("downstreams", {}).setdefault("classify", {})["weight_path"] = \
        os.path.abspath(classifier_weight)
    return ImageRestorationAgent(config=cfg, device=device)


def classify(clf, image):
    res = clf.process(image)
    probs = res.result if isinstance(res.result, dict) else {}
    if not probs:
        return "", 0.0, 0.0
    vals = sorted(probs.values(), reverse=True)
    conf = float(vals[0])
    margin = float(vals[0] - vals[1]) if len(vals) > 1 else 0.0
    pred = max(probs, key=probs.get)
    return pred, conf, margin


def collect_image(agent, clfs, seeds, path, label):
    """返回 {seed: (features(91), y(17))}；修复只跑一次。"""
    img = cv2.cvtColor(cv2.imread(path, cv2.IMREAD_COLOR), cv2.COLOR_BGR2RGB)
    diag = agent.diagnoser.diagnose(img)
    cands = agent._gen_candidates(diag, "classify")
    ups = [c.upstream for c in cands]

    restored = {"": None}
    clips = {"": 0.0}
    fallbacks = {"": False}
    for up in ups:
        if up == "":
            continue
        out = agent._get_upstream(up).restore(img)     # 只跑一次，跨 seed 共享
        restored[up] = out.image
        clips[up] = float(agent._clip_ratio(out.image))
        fallbacks[up] = bool((out.meta or {}).get("fallback", False))

    per_seed = {}
    for seed in seeds:
        outcomes = []
        for up in ups:
            task_input = restored[up] if up else img
            pred, conf, margin = classify(clfs[seed], task_input)
            outcomes.append(PipelineOutcome(
                pipeline_id=f"{up or 'raw'}_cls", upstream=up, downstream="classify",
                score=conf, downstream_confidence=conf, margin=margin, pred_label=pred,
                clip_ratio=clips[up], fallback=fallbacks[up],
                is_correct=(pred == label)))
        feats = extract_features(diag, outcomes)
        y = np.zeros(len(BRANCH_ORDER), dtype=np.float32)
        for o in outcomes:
            y[BRANCH_ORDER.index(o.upstream)] = 1.0 if o.is_correct else 0.0
        per_seed[seed] = (feats.astype(np.float32), y)
    return per_seed


def verify(agent, clfs, seeds, rows, n=4):
    """与 _execute_pipeline 逐分支比对：conf/margin/pred/clip/fallback/is_correct。"""
    rng = np.random.default_rng(0)
    picks = [rows[i] for i in rng.permutation(len(rows))[:n]]
    bad = 0
    for r in picks:
        img = cv2.cvtColor(cv2.imread(r["path"], cv2.IMREAD_COLOR), cv2.COLOR_BGR2RGB)
        diag = agent.diagnoser.diagnose(img)
        ups = [c.upstream for c in agent._gen_candidates(diag, "classify")]
        built = collect_image(agent, clfs, seeds, r["path"], r["label"])
        for seed in seeds:
            agent._downstream_cache["classify"] = clfs[seed]
            ref_out = []
            for up in ups:
                from src.router.router import PipelineRoute
                o = agent._execute_pipeline(img, PipelineRoute(f"{up or 'raw'}_cls", up, "classify",
                                                               reason="verify"), {}, r["label"])
                ref_out.append(o)
            ref_feats = extract_features(diag, ref_out)
            got_feats = built[seed][0]
            d = float(np.max(np.abs(ref_feats - got_feats)))
            ref_y = np.zeros(len(BRANCH_ORDER)); got_y = built[seed][1]
            for o in ref_out:
                ref_y[BRANCH_ORDER.index(o.upstream)] = 1.0 if o.is_correct else 0.0
            ok = d < 1e-6 and np.array_equal(ref_y, got_y)
            if not ok:
                bad += 1
            print(f"  {os.path.basename(r['path'])} seed{seed}: max|Δfeat|={d:.2e} "
                  f"Y一致={np.array_equal(ref_y, got_y)} → {'OK' if ok else '差异'}")
    print(f"等价性校验：{len(picks)} 图 × {len(seeds)} seed，异常 {bad} 项")
    return bad == 0


def main():
    ap = argparse.ArgumentParser(description="构建三 seed 的循环 router 训练缓存")
    ap.add_argument("--seeds", default="1,2,3")
    ap.add_argument("--device", default="cuda")
    ap.add_argument("--verify", type=int, default=0, help="只做等价性校验（抽 N 图）")
    ap.add_argument("--limit", type=int, default=0)
    args = ap.parse_args()

    seeds = [int(s) for s in args.seeds.split(",")]
    rows = read_manifest()
    if args.limit:
        rows = rows[: args.limit]
    print(f"清单 {len(rows)} 张 / {len({r['group'] for r in rows})} 组")

    wpaths = {s: os.path.join(ROOT, "weights", f"trashnet_resnet50_seed{s}.pth") for s in seeds}
    agent = build_agent(wpaths[seeds[0]], args.device)
    dcfg = agent.config["downstreams"]["classify"]
    clfs = {s: ClassifyProcessor(device=args.device, weight_path=wpaths[s],
                                 num_classes=int(dcfg.get("num_classes", 6)),
                                 labels=list(dcfg.get("labels") or [])) for s in seeds}
    print("classifiers:", {s: os.path.basename(c.weight_path) for s, c in clfs.items()})
    print(f"toolbox={agent.config['model_toolbox']['enabled']} | learned={agent._learned_router}")

    if args.verify:
        ok = verify(agent, clfs, seeds, rows, args.verify)
        return 0 if ok else 1

    # ---- 全量构建 ----
    buckets = {s: {"X": [], "Y": []} for s in seeds}
    t0 = time.time()
    for i, r in enumerate(rows, 1):
        per_seed = collect_image(agent, clfs, seeds, r["path"], r["label"])
        for s in seeds:
            f, y = per_seed[s]
            buckets[s]["X"].append(f)
            buckets[s]["Y"].append(y)
        if i % 50 == 0:
            el = time.time() - t0
            print(f"  {i}/{len(rows)} | {el/60:.1f} 分钟 | 预计剩余 {el/i*(len(rows)-i)/60:.0f} 分钟",
                  flush=True)

    groups = [r["group"] for r in rows]
    msha = fsha(MANIFEST)
    for s in seeds:
        d = os.path.join(CACHE_ROOT, f"iter_seed{s}_1080")
        os.makedirs(d, exist_ok=True)
        np.save(os.path.join(d, "X.npy"), np.stack(buckets[s]["X"]).astype(np.float32))
        np.save(os.path.join(d, "Y.npy"), np.stack(buckets[s]["Y"]).astype(np.float32))
        json.dump(groups, open(os.path.join(d, "G.json"), "w", encoding="utf-8"))
        json.dump({
            "manifest": MANIFEST, "manifest_sha256": msha,
            "branch_order": BRANCH_ORDER, "feature_dim": FEATURE_DIM, "toolbox": True,
            "n_done": len(rows),
            "classifier_weight_path": os.path.abspath(wpaths[s]),
            "classifier_sha256": fsha(wpaths[s]),
            "collection": "shared-restoration/iterative-v1",
            "note": "restorations computed once per image and shared across classifier seeds",
        }, open(os.path.join(d, "meta.json"), "w", encoding="utf-8"), ensure_ascii=False, indent=2)
        print(f"写出 {d}")
    print(f"总用时 {(time.time()-t0)/60:.1f} 分钟")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
