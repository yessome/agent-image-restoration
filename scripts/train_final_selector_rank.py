"""Bounded final-selection experiment; frozen acquisition, train caches only.

Fit three paired residual rankers on the original source-group training split.
Select one common residual strength on internal validation, then evaluate that
locked choice once on saved valB trajectories. No test/RealWaste inputs.
Original weights, predictions, configurations and manuscript are never edited.
"""
from __future__ import annotations

import argparse
import csv
import json
from pathlib import Path
import sys
from types import SimpleNamespace

import numpy as np
import torch
from torch import nn
from torch.nn import functional as F

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
sys.path.insert(0, str(ROOT / "scripts"))
from src.router import iterative_router as ir
from train_iterative_router import load_train_cache, partial_states

EXCLUDE = ["restormer_defocus", "mprnet_deblur", "realesrgan_x2", "realesrgan_x4"]
SEEDS = (1, 2, 3)
STRENGTHS = (0.0, 0.25, 0.5, 1.0)
DEFAULT_OUT = ROOT / "results" / "final_selector_rank_valb_20261103"


def dump(path, obj):
    Path(path).write_text(json.dumps(obj, indent=2, ensure_ascii=True), encoding="utf-8")


def return_mask(states, clip_max=0.05):
    blocks = states[:, 6:ir.FEATURE_DIM].reshape(-1, ir.N_BRANCHES, 5)
    valid = ((blocks[:, :, 0] > .5) & (blocks[:, :, 4] < .5)
             & (blocks[:, :, 3] <= clip_max + 1e-6))
    valid[:, 0] = True
    return valid


def terminal_states(router, x):
    """Replay the frozen router, revealing features only after each CALL."""
    states, probabilities, choices = [], [], []
    for full in x:
        available = full[6::5] > .5
        observed = np.zeros(ir.N_BRANCHES, dtype=bool)
        observed[0] = True
        for _ in range(router.budget + 1):
            state = ir.make_state(full, available, observed,
                                  router.budget - int(observed[1:].sum()), router.budget)
            probs = router.probabilities(state)
            action = ir.choose_action(probs, state, router.call_cost, router.clip_max)
            index = ir.BRANCH_ORDER.index(action.upstream)
            if action.kind == "stop":
                break
            observed[index] = True
        if action.kind != "stop":
            raise RuntimeError("Frozen rollout did not terminate")
        states.append(state)
        probabilities.append(probs)
        choices.append(index)
    return np.stack(states), np.stack(probabilities), np.array(choices)


def branch_inputs(states, probabilities, router):
    """35 features/branch, all available at the recorded stopping state.

    Diagnosis, observed branch/raw feedback, feedback differences, original
    scores, branch identity and execution/admission counts. No labels, source
    identity, unseen feedback, logits/entropy or ground-truth-derived features.
    """
    n, b = len(states), ir.N_BRANCHES
    blocks = states[:, 6:ir.FEATURE_DIM].reshape(n, b, 5)
    p = np.clip(probabilities.astype(np.float64), 1e-7, 1 - 1e-7)
    logits = (np.log(p) - np.log1p(-p)).astype(np.float32)
    repeat = lambda a: np.broadcast_to(a[:, None, :], (n, b, a.shape[-1]))
    diag = (states[:, :6] - router.diag_mean) / router.diag_std
    raw = repeat(blocks[:, 0, 1:])
    raw_logit = np.broadcast_to(logits[:, :1], (n, b))
    counts = np.column_stack(((blocks[:, :, 0].sum(1) - 1) / router.budget,
                              (states[:, ir.FEATURE_DIM:-1].sum(1) - 1) / (b - 1),
                              states[:, -1]))
    feats = np.concatenate((repeat(diag), blocks[:, :, 1:], raw,
                            blocks[:, :, 1:3] - raw[:, :, :2],
                            np.stack((logits, raw_logit, logits - raw_logit), axis=2) / 5,
                            np.broadcast_to(np.eye(b, dtype=np.float32), (n, b, b)),
                            repeat(counts)), axis=2).astype(np.float32)
    if not np.isfinite(feats).all():
        raise ValueError("Nonfinite selector features")
    return feats, logits


class ResidualRanker(nn.Module):
    def __init__(self, feature_dim):
        super().__init__()
        self.net = nn.Sequential(nn.Linear(feature_dim, 32), nn.ReLU(), nn.Linear(32, 1))
        nn.init.zeros_(self.net[-1].weight)
        nn.init.zeros_(self.net[-1].bias)

    def forward(self, x):
        return self.net(x).squeeze(-1)


def rank_loss(offset, correction, target, valid):
    scores = offset + correction
    # Rows with only raw cannot benefit from learning a final choice.
    useful = valid.sum(1) > 1
    if not useful.any():
        return correction.sum() * 0
    scores, target, valid, correction = (a[useful] for a in (scores, target, valid, correction))
    pointwise = ((F.binary_cross_entropy_with_logits(scores, target, reduction="none")
                  * valid).sum(1) / valid.sum(1)).mean()
    pairs = (valid[:, :, None] & valid[:, None, :]
             & (target[:, :, None] > .5) & (target[:, None, :] < .5))
    count = pairs.sum((1, 2))
    informative = count > 0
    if informative.any():
        losses = F.softplus(scores[:, None, :] - scores[:, :, None])
        pairwise = ((losses * pairs).sum((1, 2))[informative] / count[informative]).mean()
    else:
        pairwise = correction.sum() * 0
    restraint = ((correction.square() * valid).sum(1) / valid.sum(1)).mean()
    return pointwise + pairwise + .01 * restraint


def predictions(probabilities, offsets, residuals, valid, strength):
    scores = probabilities if strength == 0 else offsets + strength * residuals
    return np.where(valid, scores, -np.inf).argmax(1)


def fit(out):
    out.mkdir(parents=True, exist_ok=False)
    protocol = {
        "method": "shared-per-branch residual MLP 35->32->1; original logit offset",
        "loss": "eligible pointwise BCE + within-state positive/negative logistic ranking + 0.01 residual-square",
        "training": "original 306 train-source groups; 1 terminal + 7 random partial states per image",
        "validation": "original 54 held-out train-source groups; frozen terminal states only",
        "max_epochs": 60, "patience": 10, "adamw_lr": .001, "weight_decay": .0001,
        "batch_size": 256, "strength_grid": STRENGTHS, "seeds": SEEDS,
        "selection": "minimum internal loss per paired model; maximum combined internal accuracy for common strength; ties smaller strength",
        "valB": "evaluate only locked common strength; previously used development set, not an untouched test",
        "fixed": "admission, calls, stopping, eligible-return mask, classifier and restoration weights",
        "test": "not read or evaluated; previous test-guided toolbox exploration remains disclosed",
        "script_sha256": ir.file_hash(__file__), "torch_version": str(torch.__version__),
        "numpy_version": np.__version__,
    }
    dump(out / "protocol.json", protocol)
    candidates = {a: {"correct": 0, "per_seed": {}} for a in STRENGTHS}
    train_groups, val_groups, inputs, summaries = None, None, {}, {}
    for seed in SEEDS:
        cache = ROOT / "results" / "router_train_cache" / f"iter13_seed{seed}_1080"
        x, y, groups, meta = load_train_cache(cache)
        weight = ROOT / "weights" / f"router_iterative_feedback_13tool_seed{seed}.pth"
        base_ck = torch.load(weight, map_location="cpu", weights_only=False)
        tr_groups, va_groups = set(base_ck["train_groups"]), set(base_ck["val_groups"])
        if tr_groups & va_groups or tr_groups | va_groups != set(groups):
            raise ValueError("Original source groups do not partition training cache")
        if train_groups is not None and (train_groups != tr_groups or val_groups != va_groups):
            raise ValueError("Paired models use inconsistent source splits")
        train_groups, val_groups = tr_groups, va_groups
        tr, va = np.flatnonzero(np.isin(groups, list(tr_groups))), np.flatnonzero(np.isin(groups, list(va_groups)))
        router = ir.IterativeRouter(str(weight), classifier_weight=meta["classifier_weight_path"])
        states, probs, old = terminal_states(router, x)
        valid = return_mask(states, router.clip_max)
        if not np.array_equal(np.where(valid, probs, -np.inf).argmax(1), old):
            raise ValueError("Terminal argmax differs from frozen router")
        inputs[str(seed)] = {"router": ir.file_hash(weight), "classifier": meta["classifier_sha256"],
                             "cache": {f: ir.file_hash(cache / f) for f in ("X.npy", "Y.npy", "G.json", "meta.json")}}
        rs, rt, _ = partial_states(x, y, tr, router.budget, 7, 20261103 + seed)
        ts = np.concatenate((states[tr], rs))
        tt = np.concatenate((y[tr], rt))
        tp = np.concatenate((probs[tr], router.probabilities(rs)))
        tx, to = branch_inputs(ts, tp, router)
        vx, vo = branch_inputs(states[va], probs[va], router)
        tensors = [torch.from_numpy(a) for a in (tx, to, tt, return_mask(ts), vx, vo, y[va], valid[va])]
        tx, to, tt, tm, vx, vo, vy, vm = tensors
        torch.manual_seed(20261103 + seed)
        model = ResidualRanker(tx.shape[-1])
        optimizer = torch.optim.AdamW(model.parameters(), lr=.001, weight_decay=.0001)
        best_loss, stale, best_epoch, best_state, history = float("inf"), 0, None, None, []
        for epoch in range(60):
            model.train()
            total = 0.0
            for idx in torch.randperm(len(tx)).split(256):
                optimizer.zero_grad()
                loss = rank_loss(to[idx], model(tx[idx]), tt[idx], tm[idx])
                loss.backward()
                optimizer.step()
                total += float(loss.detach()) * len(idx)
            model.eval()
            with torch.inference_mode():
                vl = float(rank_loss(vo, model(vx), vy, vm))
            history.append({"epoch": epoch + 1, "train_loss": total / len(tx), "val_loss": vl})
            if vl < best_loss:
                best_loss, best_epoch, stale = vl, epoch + 1, 0
                best_state = {k: v.detach().clone() for k, v in model.state_dict().items()}
            else:
                stale += 1
            if epoch % 10 == 0 or stale >= 10:
                print(f"seed={seed} epoch={epoch+1} internal_loss={vl:.5f}", flush=True)
            if stale >= 10:
                break
        model.load_state_dict(best_state)
        with torch.inference_mode():
            residual = model(vx).numpy()
        for strength in STRENGTHS:
            chosen = predictions(probs[va], vo.numpy(), residual, valid[va], strength)
            correct = int(y[va, chosen].sum())
            candidates[strength]["correct"] += correct
            candidates[strength]["per_seed"][str(seed)] = correct
        ck = {"schema": "terminal-residual-ranking-v1", "state_dict": best_state,
              "input_dim": tx.shape[-1], "branch_order": ir.BRANCH_ORDER,
              "original_router": str(weight), "original_router_sha256": inputs[str(seed)]["router"],
              "classifier_sha256": meta["classifier_sha256"], "best_epoch": best_epoch,
              "train_groups": sorted(train_groups), "val_groups": sorted(val_groups),
              "clip_max": router.clip_max, "train_seed": 20261103 + seed}
        torch.save(ck, out / f"selector_seed{seed}.pth")
        summaries[str(seed)] = {"best_epoch": best_epoch, "history": history,
                                "internal_n": len(va), "original_correct": int(y[va, old[va]].sum()),
                                "executed_eligible_oracle_correct": int(np.where(valid[va], y[va], 0).max(1).sum())}
    strength = max(STRENGTHS, key=lambda a: (candidates[a]["correct"], -a))
    selection = {"locked_strength": strength, "internal_n": sum(s["internal_n"] for s in summaries.values()),
                 "candidates": candidates, "seed_summaries": summaries,
                 "train_groups": sorted(train_groups), "val_groups": sorted(val_groups), "inputs": inputs,
                 "selector_sha256": {str(s): ir.file_hash(out / f"selector_seed{s}.pth") for s in SEEDS},
                 "decision": "candidate_for_valB" if strength else "internal_validation_did_not_improve"}
    dump(out / "selection.json", selection)
    print(json.dumps({"locked_strength": strength, "internal_candidates": candidates}), flush=True)
    return selection


def valb_source_map():
    paths, hashes = {}, {}
    for scene in ("lowlight", "noise"):
        manifest = ROOT / "data" / "degraded" / f"valB_{scene}.tsv"
        hashes[scene] = ir.file_hash(manifest)
        for line in manifest.read_text(encoding="utf-8").splitlines():
            if not line or line.startswith("#"):
                continue
            path, label, source, original, _ = line.split("\t", 4)
            if ir.file_hash(original)[:16] != source:
                raise ValueError("valB source hash mismatch")
            paths[str(Path(path).resolve()).lower()] = (source, scene, label)
    return paths, hashes


def evaluate(out):
    selection_path = out / "selection.json"
    selection = json.loads(selection_path.read_text(encoding="utf-8"))
    strength = selection["locked_strength"]
    if not strength:
        (out / "REPORT.md").write_text(
            "# Final selector experiment\n\nThe fixed ranking experiment did not improve internal grouped validation. "
            "The selected strength was zero (original policy). valB and test were not evaluated.\n",
            encoding="utf-8")
        print("No internal improvement; original policy retained, no valB/test evaluation.", flush=True)
        return
    if (out / "metrics.json").exists():
        raise FileExistsError("valB evaluation already exists; refusing overwrite")
    sources, manifest_hashes = valb_source_map()
    forbidden = set(selection["train_groups"]) | set(selection["val_groups"])
    if forbidden & {v[0] for v in sources.values()}:
        raise ValueError("valB sources overlap training cache")
    output, input_hashes, max_error = [], {}, 0.0
    for seed in SEEDS:
        weight = ROOT / "weights" / f"router_iterative_feedback_13tool_seed{seed}.pth"
        folder = ROOT / "results" / "iterative_valb_13tool" / f"seed{seed}"
        meta = json.loads((folder / "run_meta.json").read_text(encoding="utf-8"))
        if (ir.file_hash(weight) != meta["router_sha256"]
                or meta["router_sha256"] != selection["inputs"][str(seed)]["router"]
                or meta["classifier_sha256"] != selection["inputs"][str(seed)]["classifier"]):
            raise ValueError("Frozen valB model pairing changed")
        selector_file = out / f"selector_seed{seed}.pth"
        if ir.file_hash(selector_file) != selection["selector_sha256"][str(seed)]:
            raise ValueError("Selector changed after model selection")
        ck = torch.load(selector_file, map_location="cpu", weights_only=False)
        model = ResidualRanker(ck["input_dim"])
        model.load_state_dict(ck["state_dict"])
        model.eval()
        router = ir.IterativeRouter(str(weight))
        raw_records = [json.loads(l) for l in (folder / "per_image.jsonl").read_text(encoding="utf-8").splitlines() if l]
        input_hashes[str(seed)] = {f: ir.file_hash(folder / f) for f in ("per_image.jsonl", "run_meta.json")}
        for record in raw_records:
            res = record["result"]
            outcomes = res["outcomes"]  # Executed outcomes only; never all/oracle outputs.
            names = [o["upstream"] for o in outcomes]
            admitted = res["routed_upstreams"]
            if len(names) != len(set(names)) or not set(names) <= set(admitted):
                raise ValueError("Invalid execution/admission masks")
            visible = [SimpleNamespace(**{k: o[k] for k in
                       ("upstream", "downstream_confidence", "margin", "clip_ratio", "fallback")}) for o in outcomes]
            features = ir.extract_features_for(res["diagnosis"], visible, ir.BRANCH_ORDER)
            available = np.array([b in admitted for b in ir.BRANCH_ORDER])
            observed = np.array([b in names for b in ir.BRANCH_ORDER])
            calls = int(observed[1:].sum())
            if calls != record["restore_calls"]:
                raise ValueError("Executed call count differs from logged count")
            state = ir.make_state(features, available, observed, router.budget - calls, router.budget)
            probs = np.array(res["scheduler_probs"], dtype=np.float32)
            error = float(np.abs(router.probabilities(state) - probs).max())
            max_error = max(error, max_error)
            if error > 2e-5:
                raise ValueError("Stopping-state scores cannot be reproduced")
            mask = return_mask(state[None], router.clip_max)
            old = ir.BRANCH_ORDER.index(res["selected_upstream"])
            if int(np.where(mask[0], probs, -np.inf).argmax()) != old:
                raise ValueError("Original terminal selection mismatch")
            feats, offset = branch_inputs(state[None], probs[None], router)
            with torch.inference_mode():
                correction = model(torch.from_numpy(feats)).numpy()
            new = int(predictions(probs[None], offset, correction, mask, strength)[0])
            if not mask[0, new]:
                raise ValueError("Returned unavailable/ineligible output")
            by_branch = {o["upstream"]: o for o in outcomes}
            source, scene, label = sources[str(Path(record["path"]).resolve()).lower()]
            if label != record["label"]:
                raise ValueError("Manifest/record label mismatch")
            # Ground truth enters only here, after inference has fixed the choice.
            correctness = {b: int(o["pred_label"] == label) for b, o in by_branch.items()}
            if correctness[""] != int(record["raw_ok"]) or correctness[ir.BRANCH_ORDER[old]] != int(record["chosen_ok"]):
                raise ValueError("Baseline correctness mismatch")
            output.append({"seed": seed, "source": source, "scene": scene, "path": record["path"],
                           "old_branch": ir.BRANCH_ORDER[old], "new_branch": ir.BRANCH_ORDER[new],
                           "raw_ok": correctness[""], "old_ok": correctness[ir.BRANCH_ORDER[old]],
                           "new_ok": correctness[ir.BRANCH_ORDER[new]], "calls": calls,
                           "eligible_oracle_ok": max(correctness[b] for j, b in enumerate(ir.BRANCH_ORDER) if mask[0, j])})
    def summarize(rows):
        old, new, raw = (np.array([r[k] for r in rows]) for k in ("old_ok", "new_ok", "raw_ok"))
        return {"n": len(rows), "original_correct": int(old.sum()), "new_correct": int(new.sum()),
                "original_accuracy": float(old.mean()), "new_accuracy": float(new.mean()),
                "delta_pp": float(100 * (new - old).mean()),
                "recovered_errors": int(((old == 0) & (new == 1)).sum()),
                "introduced_errors": int(((old == 1) & (new == 0)).sum()),
                "original_fixes": int(((raw == 0) & (old == 1)).sum()),
                "original_harms": int(((raw == 1) & (old == 0)).sum()),
                "new_fixes": int(((raw == 0) & (new == 1)).sum()),
                "new_harms": int(((raw == 1) & (new == 0)).sum()),
                "mean_calls": float(np.mean([r["calls"] for r in rows])),
                "eligible_oracle_correct": sum(r["eligible_oracle_ok"] for r in rows)}
    overall = summarize(output)
    if (overall["n"], overall["original_correct"], overall["original_fixes"], overall["original_harms"]) != (984, 779, 113, 20):
        raise ValueError("Previously verified valB baseline differs")
    ids = sorted({r["source"] for r in output})
    clusters = [[r for r in output if r["source"] == s] for s in ids]
    if len(ids) != 164 or any(len(c) != 6 or len({(r["seed"], r["scene"]) for r in c}) != 6 for c in clusters):
        raise ValueError("valB source clustering differs from 164 x 2 x 3")
    differences = np.array([sum(r["new_ok"] - r["old_ok"] for r in c) for c in clusters])
    rng = np.random.default_rng(20261103)
    boot = 100 * differences[rng.integers(0, len(ids), size=(10000, len(ids)))].sum(1) / len(output)
    overall["source_cluster_95ci_pp"] = np.percentile(boot, [2.5, 97.5]).tolist()
    metrics = {"locked_strength": strength, "overall": overall,
               "per_seed": {str(s): summarize([r for r in output if r["seed"] == s]) for s in SEEDS},
               "per_scene": {s: summarize([r for r in output if r["scene"] == s]) for s in ("lowlight", "noise")},
               "interpretation": "development result conditional on fixed models; source clusters jointly retain both scenes and all three paired models; not test-set evidence"}
    with (out / "per_record.csv").open("w", newline="", encoding="utf-8") as handle:
        writer = csv.DictWriter(handle, fieldnames=list(output[0]))
        writer.writeheader()
        writer.writerows(output)
    dump(out / "metrics.json", metrics)
    dump(out / "run_meta.json", {"selection_sha256": ir.file_hash(selection_path), "input_sha256": input_hashes,
                                "manifest_sha256": manifest_hashes, "script_sha256": ir.file_hash(__file__),
                                "max_reconstructed_score_error": max_error, "test_or_realwaste_read": False})
    rows = ["# Frozen-trajectory final selector", "", f"Locked residual strength: {strength}.", "",
            "| Paired model | Original correct | New correct | Delta (pp) |", "|---|---:|---:|---:|"]
    for s, m in metrics["per_seed"].items():
        rows.append(f"| {s} | {m['original_correct']} / {m['n']} | {m['new_correct']} / {m['n']} | {m['delta_pp']:+.4f} |")
    lo, hi = overall["source_cluster_95ci_pp"]
    rows += ["", f"Overall: {overall['original_correct']} -> {overall['new_correct']} / 984; "
             f"delta {overall['delta_pp']:+.4f} pp, source-cluster bootstrap 95% CI [{lo:+.4f}, {hi:+.4f}].",
             f"Recovered {overall['recovered_errors']} original errors; introduced {overall['introduced_errors']} new errors.",
             f"Fixes {overall['original_fixes']} -> {overall['new_fixes']}; harms {overall['original_harms']} -> {overall['new_harms']}.",
             f"Mean calls remain {overall['mean_calls']:.6f}; eligible executed oracle {overall['eligible_oracle_correct']} / 984.",
             "", "Training: 306 source groups; internal selection: 54 groups. valB: 164 sources, two scenes, three paired models.",
             "valB is a previously inspected development set. No test/RealWaste inference or parameter selection used here.",
             "The original policy and all official manuscript results remain unchanged.",
             "No additional threshold or model search was performed after seeing this valB result."]
    (out / "REPORT.md").write_text("\n".join(rows) + "\n", encoding="utf-8")
    print(json.dumps(metrics), flush=True)


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("phase", choices=("fit", "valb"))
    parser.add_argument("--out", type=Path, default=DEFAULT_OUT)
    args = parser.parse_args()
    torch.set_num_threads(2)
    ir.configure_branches(exclude=EXCLUDE)
    if args.phase == "fit":
        fit(args.out)
    else:
        evaluate(args.out)


if __name__ == "__main__":
    main()
