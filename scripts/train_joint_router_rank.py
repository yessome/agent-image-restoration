"""One fixed BCE + ranking candidate for tool acquisition and final selection.

Fine-tune copies of the original router on its original training-source split.
Choose epochs on internal grouped rollouts, including the unmodified model.
The separate valB runner must execute newly chosen tools; old logs cannot supply
unobserved branch feedback. No test or RealWaste inputs are read by this script.
"""
from __future__ import annotations

import argparse
import csv
import json
from pathlib import Path
import sys

import numpy as np
import torch
from torch.nn import functional as F

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
sys.path.insert(0, str(ROOT / "scripts"))
from src.router import iterative_router as ir
from train_iterative_router import load_train_cache, partial_states, replay
from train_final_selector_rank import EXCLUDE, SEEDS, dump, valb_source_map

OUT = ROOT / "results" / "joint_router_rank_valb_20261103"


def objective(logits, target, admitted, state):
    bce = ((F.binary_cross_entropy_with_logits(logits, target, reduction="none")
            * admitted).sum(1) / admitted.sum(1)).mean()
    branches = state[:, 6:ir.FEATURE_DIM].reshape(-1, ir.N_BRANCHES, 5)
    observed = branches[:, :, 0] > .5
    usable = admitted & (~observed | ((branches[:, :, 4] < .5) & (branches[:, :, 3] <= .05 + 1e-6)))
    usable[:, 0] = True
    pairs = (usable[:, :, None] & usable[:, None, :]
             & (target[:, :, None] > .5) & (target[:, None, :] < .5))
    count = pairs.sum((1, 2))
    rows = count > 0
    ranking = logits.sum() * 0
    if rows.any():
        ranking = ((F.softplus(logits[:, None, :] - logits[:, :, None]) * pairs).sum((1, 2))[rows]
                   / count[rows]).mean()
    return bce + .25 * ranking, bce


def fit(out):
    out.mkdir(parents=True, exist_ok=False)
    dump(out / "protocol.json", {
        "method": "fine-tune original MLP with masked BCE + 0.25 pairwise logistic ranking",
        "scope": "one fixed candidate; acquisition, stopping and return may change",
        "preserved": "admission rules, 13 branches, budget 4, call cost 0.01, clip 0.05, classifier/tool weights",
        "train": "original 306 source groups, 8 random partial states per image per epoch",
        "internal_validation": "original 54 source groups, exact rollout on complete training caches",
        "epoch_selection": "maximum internal correct count, then fewer calls, then lower BCE; epoch 0 included",
        "adamw_lr": .0001, "weight_decay": .0001, "batch_size": 256,
        "max_epochs": 30, "validation_interval": 2, "patience_validations": 6,
        "proceed_rule": "strictly positive combined internal accuracy gain over original across all 3 paired models",
        "evaluation": "lock before fresh valB execution; no old-trajectory replay for changed calls; no test or RealWaste",
        "script_sha256": ir.file_hash(__file__), "torch_version": str(torch.__version__),
    })
    summary, common_groups = {}, None
    for seed in SEEDS:
        cache = ROOT / "results" / "router_train_cache" / f"iter13_seed{seed}_1080"
        x, y, groups, meta = load_train_cache(cache)
        weight = ROOT / "weights" / f"router_iterative_feedback_13tool_seed{seed}.pth"
        base = torch.load(weight, map_location="cpu", weights_only=False)
        tr_groups, va_groups = set(base["train_groups"]), set(base["val_groups"])
        if tr_groups & va_groups or tr_groups | va_groups != set(groups):
            raise ValueError("Invalid original group partition")
        if common_groups and common_groups != (tr_groups, va_groups):
            raise ValueError("Paired source splits differ")
        common_groups = (tr_groups, va_groups)
        tr, va = np.flatnonzero(np.isin(groups, list(tr_groups))), np.flatnonzero(np.isin(groups, list(va_groups)))
        router = ir.IterativeRouter(str(weight), classifier_weight=meta["classifier_weight_path"])
        model = router.model
        torch.manual_seed(20261103 + seed)
        optimizer = torch.optim.AdamW(model.parameters(), lr=.0001, weight_decay=.0001)
        vs, vt, vm = partial_states(x, y, va, router.budget, 8, 20261103)
        vs[:, :6] = (vs[:, :6] - router.diag_mean) / router.diag_std
        vs, vt, vm = torch.from_numpy(vs), torch.from_numpy(vt), torch.from_numpy(vm)

        def validate():
            model.eval()
            metrics, _ = replay(router, x, y, va)
            with torch.inference_mode():
                _, bce = objective(model(vs), vt, vm, vs)
            correct = int(round(metrics["accuracy"] * len(va)))
            calls = int(round(metrics["mean_restore_calls"] * len(va)))
            return metrics, (correct, -calls, -float(bce))

        original, best_key = validate()
        best_metrics, best_epoch = original, 0
        best = {k: v.detach().clone() for k, v in model.state_dict().items()}
        history, stale = [], 0
        for epoch in range(1, 31):
            ts, tt, tm = partial_states(x, y, tr, router.budget, 8, 20261103 + seed * 100 + epoch)
            ts[:, :6] = (ts[:, :6] - router.diag_mean) / router.diag_std
            ts, tt, tm = torch.from_numpy(ts), torch.from_numpy(tt), torch.from_numpy(tm)
            model.train()
            total = 0.0
            for ix in torch.randperm(len(ts)).split(256):
                optimizer.zero_grad()
                loss, _ = objective(model(ts[ix]), tt[ix], tm[ix], ts[ix])
                loss.backward()
                optimizer.step()
                total += float(loss.detach()) * len(ix)
            if epoch % 2:
                continue
            metrics, key = validate()
            history.append({"epoch": epoch, "train_loss": total / len(ts), "correct": key[0],
                            "calls": -key[1], "bce": -key[2]})
            if key > best_key:
                best_key, best_metrics, best_epoch, stale = key, metrics, epoch, 0
                best = {k: v.detach().clone() for k, v in model.state_dict().items()}
            else:
                stale += 1
            print(f"seed={seed} epoch={epoch} internal_correct={key[0]}/{len(va)} "
                  f"calls={-key[1]} selected_epoch={best_epoch}", flush=True)
            if stale >= 6:
                break
        checkpoint = dict(base)
        checkpoint.update({"state_dict": best, "training_kind": "masked_BCE_plus_0.25_pairwise_ranking_finetune",
                           "history": history, "best_epoch": best_epoch, "best_val_loss": -best_key[2],
                           "epoch_selection": "internal rollout correct, then fewer calls, then BCE; baseline included",
                           "base_router_sha256": ir.file_hash(weight), "finetune_seed": 20261103 + seed,
                           "validation_scope": "original grouped training cache; no valB/test used for selection"})
        output = out / f"router_seed{seed}.pth"
        torch.save(checkpoint, output)
        summary[str(seed)] = {"original": original, "selected": best_metrics, "selected_epoch": best_epoch,
                              "history": history, "new_router_sha256": ir.file_hash(output),
                              "base_router_sha256": ir.file_hash(weight), "classifier_sha256": meta["classifier_sha256"]}
    delta = sum(round(v["selected"]["accuracy"] * v["selected"]["n"])
                - round(v["original"]["accuracy"] * v["original"]["n"]) for v in summary.values())
    result = {"internal_delta_correct": delta, "proceed_to_valb": delta > 0, "per_seed": summary,
              "train_groups": sorted(common_groups[0]), "val_groups": sorted(common_groups[1]),
              "selection_completed_before_valb": True}
    dump(out / "selection.json", result)
    print(json.dumps({"internal_delta_correct": delta, "proceed_to_valb": delta > 0}), flush=True)
    if delta <= 0:
        (out / "REPORT.md").write_text("# Joint ranking candidate\n\nNo combined internal accuracy improvement. "
                                       "valB/test not evaluated. See selection.json for all three paired results.\n",
                                       encoding="utf-8")


def report(out):
    selection = json.loads((out / "selection.json").read_text(encoding="utf-8"))
    if not selection["proceed_to_valb"]:
        raise ValueError("Candidate did not pass the internal proceed rule")
    if (out / "metrics.json").exists():
        raise FileExistsError("Report already exists")
    sources, _ = valb_source_map()
    if set(selection["train_groups"] + selection["val_groups"]) & {s[0] for s in sources.values()}:
        raise ValueError("Source split overlap")
    rows, evaluation_modes, input_hashes = [], {}, {}
    for seed in SEEDS:
        old_dir = ROOT / "results" / "iterative_valb_13tool" / f"seed{seed}"
        selected = selection["per_seed"][str(seed)]
        checkpoint = out / f"router_seed{seed}.pth"
        if ir.file_hash(checkpoint) != selected["new_router_sha256"]:
            raise ValueError("Locked candidate checkpoint changed")
        unchanged = selected["selected_epoch"] == 0
        if unchanged:
            base = torch.load(ROOT / "weights" / f"router_iterative_feedback_13tool_seed{seed}.pth", map_location="cpu", weights_only=False)
            candidate = torch.load(checkpoint, map_location="cpu", weights_only=False)
            if set(base["state_dict"]) != set(candidate["state_dict"]) or any(
                    not torch.equal(v, candidate["state_dict"][k]) for k, v in base["state_dict"].items()):
                raise ValueError("Epoch-zero checkpoint differs from original")
            for key in ("branch_order", "diag_mean", "diag_std", "max_calls", "call_cost", "clip_max", "classifier_sha256"):
                if base[key] != candidate[key]:
                    raise ValueError("Epoch-zero policy metadata differs")
        new_dir = old_dir if unchanged else out / f"valb_seed{seed}"
        evaluation_modes[str(seed)] = "reused verified identical epoch-zero policy" if unchanged else "fresh valB executions"
        old_meta, new_meta = [json.loads((d / "run_meta.json").read_text()) for d in (old_dir, new_dir)]
        expected_router = selected["base_router_sha256"] if unchanged else selected["new_router_sha256"]
        # Older metadata also listed the four explicitly excluded tools. Their
        # absence in the current config cannot change this 13-branch policy.
        active_tools = set(ir.BRANCH_ORDER[1:])
        if not active_tools <= set(old_meta["tool_sha256"]) & set(new_meta["tool_sha256"]):
            raise ValueError("Missing active-tool provenance")
        tool_differences = {k for k in set(old_meta["tool_sha256"]) | set(new_meta["tool_sha256"])
                            if old_meta["tool_sha256"].get(k) != new_meta["tool_sha256"].get(k)}
        if (old_meta["classifier_sha256"] != new_meta["classifier_sha256"]
                or not tool_differences <= set(EXCLUDE)
                or old_meta["manifests"] != new_meta["manifests"]
                or new_meta["router_sha256"] != expected_router):
            raise ValueError("Evaluation models/manifests changed")
        input_hashes[str(seed)] = {"original": ir.file_hash(old_dir / "per_image.jsonl"),
                                   "candidate": ir.file_hash(new_dir / "per_image.jsonl")}
        old, new = [{r["path"]: r for r in (json.loads(l) for l in (d / "per_image.jsonl").read_text(encoding="utf-8").splitlines() if l)}
                    for d in (old_dir, new_dir)]
        if set(old) != set(new) or len(new) != 328:
            raise ValueError("Paired image sets differ or evaluation incomplete")
        for path, a in old.items():
            b = new[path]
            if (a["input_sha256"] != b["input_sha256"] or a["label"] != b["label"]
                    or a["raw_pred"] != b["raw_pred"]
                    or set(a["result"]["routed_upstreams"]) != set(b["result"]["routed_upstreams"])):
                raise ValueError("Input, raw output or admission changed")
            source, scene, label = sources[str(Path(path).resolve()).lower()]
            rows.append({"seed": seed, "source": source, "scene": scene, "path": path,
                         "raw_ok": int(a["raw_ok"]), "old_ok": int(a["chosen_ok"]), "new_ok": int(b["chosen_ok"]),
                         "old_calls": a["restore_calls"], "new_calls": b["restore_calls"],
                         "old_branch": a["result"]["selected_upstream"], "new_branch": b["result"]["selected_upstream"]})
    def stats(records):
        a, b, raw = [np.array([r[k] for r in records]) for k in ("old_ok", "new_ok", "raw_ok")]
        return {"n": len(records), "original_correct": int(a.sum()), "new_correct": int(b.sum()),
                "original_accuracy": float(a.mean()), "new_accuracy": float(b.mean()), "delta_pp": float(100 * (b-a).mean()),
                "recovered_errors": int(((a == 0) & (b == 1)).sum()), "introduced_errors": int(((a == 1) & (b == 0)).sum()),
                "old_fixes": int(((raw == 0) & (a == 1)).sum()), "new_fixes": int(((raw == 0) & (b == 1)).sum()),
                "old_harms": int(((raw == 1) & (a == 0)).sum()), "new_harms": int(((raw == 1) & (b == 0)).sum()),
                "old_mean_calls": float(np.mean([r["old_calls"] for r in records])),
                "new_mean_calls": float(np.mean([r["new_calls"] for r in records]))}
    overall = stats(rows)
    clusters = [[r for r in rows if r["source"] == s] for s in sorted({r["source"] for r in rows})]
    if len(clusters) != 164 or any(len(c) != 6 for c in clusters) or overall["original_correct"] != 779:
        raise ValueError("Source clusters/baseline differ")
    diff = np.array([sum(r["new_ok"] - r["old_ok"] for r in c) for c in clusters])
    draw = np.random.default_rng(20261103).integers(0, 164, (10000, 164))
    overall["source_cluster_95ci_pp"] = np.percentile(100 * diff[draw].sum(1) / len(rows), [2.5, 97.5]).tolist()
    metrics = {"overall": overall, "evaluation_modes": evaluation_modes,
               "per_seed": {str(s): stats([r for r in rows if r["seed"] == s]) for s in SEEDS},
               "per_scene": {s: stats([r for r in rows if r["scene"] == s]) for s in ("lowlight", "noise")},
               "scope": "previously inspected valB development set, all three paired models; no test or RealWaste evaluation"}
    dump(out / "metrics.json", metrics)
    dump(out / "run_meta.json", {"selection_sha256": ir.file_hash(out / "selection.json"),
                                "input_prediction_sha256": input_hashes, "evaluation_modes": evaluation_modes,
                                "tool_hash_scope": "all 12 active tools match; four explicitly excluded tools are ignored",
                                "report_script_sha256": ir.file_hash(__file__), "test_or_realwaste_read": False})
    with (out / "per_record.csv").open("w", newline="", encoding="utf-8") as handle:
        writer = csv.DictWriter(handle, fieldnames=list(rows[0]))
        writer.writeheader(); writer.writerows(rows)
    lo, hi = overall["source_cluster_95ci_pp"]
    lines = ["# Joint router ranking candidate", "", "One fixed BCE + 0.25 pairwise-ranking candidate; epochs locked on internal train-source validation.",
             "Changed policies receive fresh valB executions. Identical epoch-zero policies reuse verified original records.", "",
             "| Paired model | Original correct | New correct | Delta (pp) |", "|---|---:|---:|---:|"]
    for s, m in metrics["per_seed"].items():
        lines.append(f"| {s} | {m['original_correct']} / 328 | {m['new_correct']} / 328 | {m['delta_pp']:+.4f} |")
    lines += ["", f"Overall {overall['original_correct']} -> {overall['new_correct']} / 984; {overall['delta_pp']:+.4f} pp.",
              f"Source-cluster 95% CI [{lo:+.4f}, {hi:+.4f}] pp (164 source clusters, 10,000 bootstrap draws).",
              f"Recovered {overall['recovered_errors']} errors, introduced {overall['introduced_errors']} errors.",
              f"Fixes {overall['old_fixes']} -> {overall['new_fixes']}; harms {overall['old_harms']} -> {overall['new_harms']}.",
              f"Mean calls {overall['old_mean_calls']:.6f} -> {overall['new_mean_calls']:.6f}.", "",
              "This is a development result; original weights, manuscript, test and RealWaste results remain unchanged.",
              "No additional model or threshold search followed this valB evaluation."]
    (out / "REPORT.md").write_text("\n".join(lines) + "\n", encoding="utf-8")
    print(json.dumps(metrics), flush=True)


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("phase", choices=("fit", "report"))
    parser.add_argument("--out", type=Path, default=OUT)
    args = parser.parse_args()
    torch.set_num_threads(2)
    ir.configure_branches(exclude=EXCLUDE)
    (fit if args.phase == "fit" else report)(args.out)
