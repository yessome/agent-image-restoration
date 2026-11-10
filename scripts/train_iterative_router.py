"""Train only the feedback router from frozen train-set branch caches.

No image restoration, classifier training, valB/test reads, or GPU is required.
Random partial observations teach the network to score both executed and unseen
eligible branches. Entire source groups are split before creating any states.
"""
from __future__ import annotations

import argparse
from collections import Counter
import json
import os
from pathlib import Path
import sys

import numpy as np

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

from src.router import iterative_router as ir
from src.router.iterative_router import (SCHEMA, IterativeRouter,
                                         make_state, choose_action, file_hash)


def load_train_cache(cache_dir):
    path = Path(cache_dir)
    meta = json.loads((path / "meta.json").read_text(encoding="utf-8"))
    manifest = Path(meta["manifest"])
    # Check both recorded provenance and actual source membership, not only names.
    train_root = (ROOT / "data" / "degraded").resolve()
    if (manifest.resolve().parent != train_root or not manifest.name.startswith("train_")
            or meta.get("manifest_sha256") != file_hash(manifest)
            or meta.get("branch_order") != ir.BRANCH_ORDER or meta.get("feature_dim") != ir.FEATURE_DIM
            or not meta.get("toolbox")):
        raise ValueError("Cache provenance/manifest/branch order mismatch")
    source_ids = set()
    for line in (ROOT / "data" / "splits" / "train.tsv").read_text(encoding="utf-8").splitlines():
        if line and not line.startswith("#"):
            src = Path(line.split("\t")[0])
            if not src.is_absolute():
                src = ROOT / src
            source_ids.add(file_hash(src)[:16])
    manifest_rows = [line.split("\t") for line in manifest.read_text(encoding="utf-8").splitlines()
                     if line and not line.startswith("#")]
    manifest_groups = [row[2] for row in manifest_rows]
    x, y = np.load(path / "X.npy"), np.load(path / "Y.npy")
    groups = json.loads((path / "G.json").read_text(encoding="utf-8"))
    if (len(x) != len(y) or len(x) != len(groups) or len(x) != meta["n_done"]
            or list(groups) != manifest_groups[:len(x)]
            or len(set(groups)) < 4 or x.shape != (len(x), ir.FEATURE_DIM)
            or y.shape != (len(x), ir.N_BRANCHES) or not np.isfinite(x).all()
            or not np.isin(y, [0, 1]).all()):
        raise ValueError("Invalid cache arrays, group alignment, or train source membership")
    # Legacy caches use numeric group IDs, not source hashes. Resolve each group's
    # clean source and split by its content hash, so aliases cannot cross folds.
    by_group = {}
    for row in manifest_rows[:len(x)]:
        by_group.setdefault(row[2], []).append(row)
    canonical = {}
    for gid, rows in by_group.items():
        clean = []
        for row in rows:
            image = Path(row[0])
            if not image.is_absolute():
                image = ROOT / image
            image = image.resolve()
            if image.is_relative_to((ROOT / "data" / "trashnet").resolve()):
                clean.append(image)
            elif image.parent not in {(train_root / "train_lowlight"), (train_root / "train_noise")}:
                raise ValueError("Cached variant is not a train degradation")
        if len(clean) != 1 or len({row[1] for row in rows}) != 1:
            raise ValueError("Each cache group needs one traceable clean train source")
        sid = file_hash(clean[0])[:16]
        if sid not in source_ids:
            raise ValueError("Cached source not in clean train split")
        canonical[gid] = sid
    available = x[:, 6::5] > 0.5
    if not available[:, 0].all() or not np.isin(x[:, 6::5], [0, 1]).all():
        raise ValueError("Invalid branch presence mask")
    cls_path = meta["classifier_weight_path"]
    if file_hash(cls_path) != meta["classifier_sha256"]:
        raise ValueError("Classifier checksum differs from feature collection")
    return x.astype(np.float32), y.astype(np.float32), np.array([canonical[g] for g in groups]), meta


def split_groups(groups, ratio, seed):
    rng = np.random.default_rng(seed)
    keys = np.unique(groups)
    rng.shuffle(keys)
    val = np.isin(groups, keys[:max(1, int(len(keys) * ratio))])
    return np.flatnonzero(~val), np.flatnonzero(val)


def partial_states(x, y, indices, budget, count, seed):
    rng = np.random.default_rng(seed)
    states, targets, masks = [], [], []
    for i in indices:
        available = x[i, 6::5] > 0.5
        tools = np.flatnonzero(available)[1:]
        for repetition in range(count):
            k = 0 if repetition == 0 else int(rng.integers(min(budget, len(tools)) + 1))
            observed = np.zeros(ir.N_BRANCHES, dtype=bool)
            observed[0] = True
            if k:
                observed[rng.choice(tools, k, replace=False)] = True
            states.append(make_state(x[i], available, observed, budget-k, budget))
            targets.append(y[i])  # ALL branch correctness labels keep identical semantics.
            masks.append(available)
    return np.stack(states), np.stack(targets), np.stack(masks)


def replay(router, x, y, indices, feedback=True):
    """Offline held-out rollout. Future branch features are revealed only on CALL."""
    chosen, calls, records = [], [], []
    for i in indices:
        available = x[i, 6::5] > 0.5
        observed = np.zeros(ir.N_BRANCHES, dtype=bool)
        observed[0] = True
        trace = []
        for _ in range(router.budget + 1):
            state = make_state(x[i], available, observed,
                               router.budget - int(observed[1:].sum()), router.budget)
            model_state = state.copy()
            if not feedback:
                # Control: retain execution masks and raw feedback; remove tool evidence.
                model_state[6:ir.FEATURE_DIM].reshape(ir.N_BRANCHES, 5)[1:, 1:] = 0
            action = choose_action(router.probabilities(model_state), state,
                                   router.call_cost, router.clip_max)
            idx = ir.BRANCH_ORDER.index(action.upstream)
            trace.append({"action": action.kind, "tool": action.upstream or "raw"})
            if action.kind == "stop":
                break
            observed[idx] = True
        chosen.append(idx)
        calls.append(int(observed[1:].sum()))
        records.append({"row": int(i), "chosen": action.upstream or "raw", "trace": trace})
    selected = y[indices, np.array(chosen)]
    raw = y[indices, 0]
    oracle = np.array([y[i, x[i, 6::5] > 0.5].max() for i in indices])
    return {"n": len(indices), "raw_accuracy": float(raw.mean()),
            "accuracy": float(selected.mean()), "candidate_oracle": float(oracle.mean()),
            "candidate_regret": float((oracle-selected).mean()),
            "fixes": int(((raw == 0) & (selected == 1)).sum()),
            "harms": int(((raw == 1) & (selected == 0)).sum()),
            "mean_restore_calls": float(np.mean(calls)),
            "selection_counts": dict(Counter(ir.BRANCH_ORDER[i] or "raw" for i in chosen)),
            "call_counts": dict(Counter(calls))}, records


def main():
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--cache-dir", required=True)
    ap.add_argument("--out", required=True)
    ap.add_argument("--epochs", type=int, default=60)
    ap.add_argument("--patience", type=int, default=10)
    ap.add_argument("--seed", type=int, default=42)
    ap.add_argument("--max-calls", type=int, default=4)
    ap.add_argument("--exclude", default="",
                    help="排除的分支（逗号分隔，必须是完整表的后缀）；用于训练 13 工具 router")
    ap.add_argument("--call-cost", type=float, default=0.01,
                    help="Explicit per-call penalty in correctness-probability units")
    ap.add_argument("--check-only", action="store_true")
    args = ap.parse_args()

    if args.exclude:
        order = ir.configure_branches(exclude=[x.strip() for x in args.exclude.split(",") if x.strip()])
        print(f"分支表切换为 {len(order)} 个：{order}", flush=True)
    if not 0 <= args.max_calls < ir.N_BRANCHES or not 0 <= args.call_cost < 1:
        ap.error("Invalid budget/call cost")
    x, y, groups, meta = load_train_cache(args.cache_dir)
    tr, va = split_groups(groups, 0.15, args.seed)
    print(f"Validated train cache: {len(x)} rows / {len(set(groups))} sources; "
          f"train={len(tr)}, validation={len(va)}, state_dim={ir.STATE_DIM}", flush=True)
    if args.check_only:
        return 0
    out = Path(args.out)
    if out.exists():
        raise FileExistsError(f"Refusing to overwrite existing checkpoint: {out}")
    import torch
    from torch import nn
    from src.router._router_net import RouterNet

    torch.set_num_threads(2)  # CPU training does not compete with the running GPU eval.
    torch.manual_seed(args.seed)
    mean = x[tr, :6].mean(0)
    std = np.maximum(x[tr, :6].std(0), 1e-4)
    model = RouterNet(ir.STATE_DIM, hidden=128, num_branches=ir.N_BRANCHES)
    optimizer = torch.optim.AdamW(model.parameters(), lr=1e-3, weight_decay=1e-4)
    criterion = nn.BCEWithLogitsLoss(reduction="none")

    def tensors(indices, seed):
        s, t, m = partial_states(x, y, indices, args.max_calls, 8, seed)
        s[:, :6] = (s[:, :6] - mean) / std
        return tuple(torch.from_numpy(v.astype(np.float32)) for v in (s, t, m))

    vs, vt, vm = tensors(va, args.seed + 10000)

    def loss_on(logits, target, mask):
        # Equal weight per image-state; images with many tools do not dominate.
        return ((criterion(logits, target) * mask).sum(1) / mask.sum(1)).mean()

    best_loss, best_state, stale, history = float("inf"), None, 0, []
    for epoch in range(args.epochs):
        ts, tt, tm = tensors(tr, args.seed + epoch)
        model.train()
        perm = torch.randperm(len(ts))
        total = 0.0
        for batch in perm.split(256):
            optimizer.zero_grad()
            loss = loss_on(model(ts[batch]), tt[batch], tm[batch])
            loss.backward()
            optimizer.step()
            total += loss.item() * len(batch)
        model.eval()
        with torch.inference_mode():
            vl = loss_on(model(vs), vt, vm).item()
        history.append({"epoch": epoch+1, "train_loss": total/len(ts), "val_loss": vl})
        if vl < best_loss:
            best_loss = vl
            best_state = {k: v.detach().clone() for k, v in model.state_dict().items()}
            stale = 0
        else:
            stale += 1
        if epoch % 5 == 0 or stale >= args.patience:
            print(f"epoch={epoch+1} train={total/len(ts):.4f} val={vl:.4f}", flush=True)
        if stale >= args.patience:
            break
    checkpoint = {"schema": SCHEMA, "feature_dim": ir.STATE_DIM, "branch_order": ir.BRANCH_ORDER,
                  "output_activation": "sigmoid", "hidden": 128, "state_dict": best_state,
                  "diag_mean": mean.tolist(), "diag_std": std.tolist(), "clip_max": 0.05,
                  "max_calls": args.max_calls, "call_cost": args.call_cost,
                  "classifier_sha256": meta["classifier_sha256"],
                  "classifier_weight_path": meta["classifier_weight_path"],
                  "cache_dir": str(Path(args.cache_dir).resolve()),
                  "cache_sha256": {name: file_hash(Path(args.cache_dir) / name)
                                   for name in ("X.npy", "Y.npy", "G.json", "meta.json")},
                  "manifest_sha256": meta["manifest_sha256"], "train_seed": args.seed,
                  "train_groups": sorted(set(groups[tr])), "val_groups": sorted(set(groups[va])),
                  "best_val_loss": best_loss, "history": history,
                  "training_kind": "masked_branch_correctness_myopic_acquisition",
                  "validation_scope": "internal grouped train-cache replay; not valB/test",
                  "tool_provenance": "inherited cache does not record tool weight hashes"}
    out.parent.mkdir(parents=True, exist_ok=True)
    tmp = str(out) + ".tmp"
    with open(tmp, "wb") as handle:
        torch.save(checkpoint, handle)
        handle.flush()
        os.fsync(handle.fileno())
    os.replace(tmp, out)
    router = IterativeRouter(str(out), classifier_weight=meta["classifier_weight_path"])
    metrics, records = replay(router, x, y, va)
    control, _ = replay(router, x, y, va, feedback=False)
    report = {"scope": checkpoint["validation_scope"], "checkpoint": str(out.resolve()),
              "full_feedback": metrics, "without_tool_feedback": control,
              "validation_rows": records,
              "note": "Prototype; accuracy gain and feedback benefit must be measured, not assumed."}
    out.with_suffix(".json").write_text(json.dumps(report, ensure_ascii=False, indent=2), encoding="utf-8")
    print(json.dumps({"full_feedback": metrics, "without_tool_feedback": control}, indent=2), flush=True)
    print(f"Saved {out}; no restoration/classifier weights changed.", flush=True)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
