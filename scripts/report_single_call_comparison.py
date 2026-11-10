"""Pair the separately trained one-call control with existing six-scene results.

Source images, not scene/run rows, are the bootstrap units. This script never
trains a model or selects a threshold. The existing test-informed tool-pool
selection remains part of the interpretation of this supplemental comparison.
"""
from __future__ import annotations

import argparse
import csv
import hashlib
import json
from collections import Counter, defaultdict
from pathlib import Path

import numpy as np

ROOT = Path(__file__).resolve().parents[1]
OUT = ROOT / "results/single_call_13tool_20261103"
MANIFESTS = {
    "clean": "results/table2_scenes/clean.tsv",
    "jpeg": "data/degraded/jpeg/manifest_full.tsv",
    "motion_blur": "data/degraded/motion_blur/manifest_full.tsv",
    "mixed": "data/degraded/mixed/manifest_full.tsv",
    "lowlight": "data/degraded_test_13tool/lowlight/manifest_full.tsv",
    "noise": "data/degraded_test_13tool/noise/manifest_full.tsv",
}
LABELS = ("cardboard", "glass", "metal", "paper", "plastic", "trash")


def path_key(value):
    return str(Path(value).resolve()).lower()


def image_key(scene, value):
    return scene, value.replace("\\", "/").rsplit("/", 1)[-1].lower()


def read_samples():
    samples, by_path, hashes = {}, {}, {}
    for scene, name in MANIFESTS.items():
        with (ROOT / name).open(encoding="utf-8") as handle:
            reader = csv.reader(handle, delimiter="\t")
            header = [h.lstrip("#").strip() for h in next(reader)]
            for values in reader:
                if not values:
                    continue
                row = dict(zip(header, values))
                path = row.get("path", row.get("degraded_path"))
                source = row.get("source_path", path)
                if not Path(path).is_file() or not Path(source).is_file():
                    raise ValueError(f"Missing current image or source: {path}, {source}")
                if source not in hashes:
                    hashes[source] = hashlib.sha256(Path(source).read_bytes()).hexdigest()
                key = image_key(scene, path)
                item = {"path": path, "label": row["label"], "subset": scene,
                        "group": hashes[source]}
                if key in samples or path_key(path) in by_path:
                    raise ValueError(f"Duplicate sample: {key}")
                samples[key] = item
                by_path[path_key(path)] = key
    counts = Counter(row["subset"] for row in samples.values())
    groups = Counter(row["group"] for row in samples.values())
    if set(counts.values()) != {430} or len(groups) != 430 or set(groups.values()) != {6}:
        raise ValueError(f"Expected 430 unique sources in each of six scenes: {counts}")
    return samples, by_path


def read_old(samples):
    methods = {name: {} for name in ("raw", "diag_rule", "legacy_rule", "full", "truncated")}
    rename = {"iterative": "full"}
    for seed in (1, 2, 3):
        for folder in ("scene_matrix_13tool", "test_matrix_13tool"):
            with (ROOT / "results" / folder / f"seed{seed}/per_image.csv").open(encoding="utf-8") as h:
                for row in csv.DictReader(h):
                    method = rename.get(row["method"], row["method"])
                    if method not in methods:
                        continue
                    key = image_key(row["subset"], row["path"])
                    item = samples[key]
                    idx = (seed, *key)
                    if idx in methods[method] or row["label"] != item["label"]:
                        raise ValueError(f"Duplicate or mislabelled old row: {method}, {idx}")
                    methods[method][idx] = {**item, "seed": seed, "pred": row["pred"],
                        "ok": int(row["ok"]), "raw_ok": int(row["raw_ok"]),
                        "calls": int(row["restore_calls"])}
        path = ROOT / f"results/ablation_13tool_test/seed{seed}/call_limit_1.jsonl"
        with path.open(encoding="utf-8") as h:
            for line in h:
                row = json.loads(line)
                key = image_key(row["subset"], row["path"])
                idx = (seed, *key)
                item = samples[key]
                if idx in methods["truncated"] or row["label"] != item["label"]:
                    raise ValueError(f"Duplicate or mislabelled truncated row: {idx}")
                methods["truncated"][idx] = {**item, "seed": seed, "pred": row["pred"],
                    "ok": int(row["ok"]), "raw_ok": int(row["raw_ok"]),
                    "calls": int(row["restore_calls"])}
    expected = {(seed, *key) for seed in (1, 2, 3) for key in samples}
    for method, rows in methods.items():
        if set(rows) != expected:
            raise ValueError(f"Incomplete old comparator: {method}, {len(rows)}")
        for idx, row in rows.items():
            if row["ok"] != int(row["pred"] == row["label"]):
                raise ValueError(f"Prediction/correctness mismatch: {method}, {idx}")
            if row["raw_ok"] != methods["raw"][idx]["ok"]:
                raise ValueError(f"Raw correctness mismatch: {method}, {idx}")
    return methods


def read_new(samples, by_path, old):
    rows = {}
    for seed in (1, 2, 3):
        summary = json.loads((OUT / f"seed{seed}/summary.json").read_text(encoding="utf-8"))
        if summary["n"] != len(samples):
            raise ValueError(f"Incomplete new evaluation: seed {seed}")
        with (OUT / f"seed{seed}/per_image.jsonl").open(encoding="utf-8") as h:
            for line in h:
                record = json.loads(line)
                key = by_path[path_key(record["path"])]
                item = samples[key]
                idx = (seed, *key)
                if idx in rows or record["label"] != item["label"]:
                    raise ValueError(f"Duplicate or mislabelled new row: {idx}")
                raw = old["raw"][idx]
                if record["raw_pred"] != raw["pred"] or int(record["raw_ok"]) != raw["ok"]:
                    raise ValueError(f"Historical/current raw prediction differs: {idx}")
                calls = int(record["restore_calls"])
                trace = record["result"]["routing_trace"]
                if calls not in (0, 1) or sum(t["action"] == "call" for t in trace) != calls:
                    raise ValueError(f"One-call budget/trace mismatch: {idx}")
                if trace[0]["observed"] != [""] or trace[-1]["action"] != "stop":
                    raise ValueError(f"Invalid observation/termination trace: {idx}")
                ok = int(record["chosen_ok"])
                if ok != int(record["chosen_pred"] == item["label"]):
                    raise ValueError(f"New prediction/correctness mismatch: {idx}")
                rows[idx] = {**item, "seed": seed, "pred": record["chosen_pred"],
                             "ok": ok, "raw_ok": raw["ok"], "calls": calls}
    if set(rows) != set(old["full"]):
        raise ValueError("New/old sample keys differ")
    return rows


def macro_f1(rows):
    scores = []
    for label in LABELS:
        tp = sum(r["label"] == label and r["pred"] == label for r in rows)
        fp = sum(r["label"] != label and r["pred"] == label for r in rows)
        fn = sum(r["label"] == label and r["pred"] != label for r in rows)
        scores.append(2 * tp / (2 * tp + fp + fn) if 2 * tp + fp + fn else 0.0)
    return float(np.mean(scores))


def summarize(records):
    rows = list(records.values())
    per_seed = {}
    for seed in (1, 2, 3):
        subset = [r for r in rows if r["seed"] == seed]
        per_seed[seed] = {"accuracy": float(np.mean([r["ok"] for r in subset])),
                          "macro_f1": macro_f1(subset),
                          "mean_calls": float(np.mean([r["calls"] for r in subset]))}
    return {"n": len(rows), "accuracy": float(np.mean([r["ok"] for r in rows])),
            "sample_sd": float(np.std([r["accuracy"] for r in per_seed.values()], ddof=1)),
            "macro_f1": float(np.mean([r["macro_f1"] for r in per_seed.values()])),
            "mean_calls": float(np.mean([r["calls"] for r in rows])),
            "fixes": sum(r["ok"] and not r["raw_ok"] for r in rows),
            "harms": sum(r["raw_ok"] and not r["ok"] for r in rows),
            "call_distribution": dict(sorted(Counter(r["calls"] for r in rows).items())),
            "per_seed": per_seed,
            "per_scene": {s: float(np.mean([r["ok"] for r in rows if r["subset"] == s]))
                          for s in MANIFESTS},
            "per_scene_sample_sd": {s: float(np.std([
                np.mean([r["ok"] for r in rows if r["subset"] == s and r["seed"] == seed])
                for seed in (1, 2, 3)], ddof=1)) for s in MANIFESTS}}


def paired_difference(a, b, field, seed=20261103):
    grouped = defaultdict(list)
    for idx, row in a.items():
        grouped[row["group"]].append(row[field] - b[idx][field])
    if len(grouped) != 430 or any(len(v) != 18 for v in grouped.values()):
        raise ValueError("Each bootstrap source must contain six scenes and three model pairs")
    values = np.array([np.mean(grouped[g]) for g in sorted(grouped)])
    rng = np.random.default_rng(seed)
    boot = np.concatenate([values[rng.integers(0, len(values), (500, len(values)))].mean(axis=1)
                           for _ in range(20)])
    return {"difference": float(values.mean()),
            "ci95": np.quantile(boot, [0.025, 0.975]).tolist(),
            "per_seed": {s: float(np.mean([r[field] - b[i][field] for i, r in a.items()
                                           if r["seed"] == s])) for s in (1, 2, 3)}}


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--check-inputs", action="store_true")
    args = parser.parse_args()
    samples, by_path = read_samples()
    methods = read_old(samples)
    if args.check_inputs:
        print(json.dumps({"sources": 430, "scenes": 6, "samples_per_seed": len(samples),
                          "existing_comparators": {k: summarize(v) for k, v in methods.items()}}, indent=2))
        return
    methods["trained_one_call"] = read_new(samples, by_path, methods)
    summaries = {name: summarize(rows) for name, rows in methods.items()}
    comparisons = {}
    for ref in ("full", "truncated", "diag_rule", "legacy_rule", "raw"):
        comparisons[f"trained_one_call_minus_{ref}"] = {
            "accuracy": paired_difference(methods["trained_one_call"], methods[ref], "ok"),
            "calls": paired_difference(methods["trained_one_call"], methods[ref], "calls")}
    report = {"protocol": json.loads((OUT / "protocol.json").read_text(encoding="utf-8")),
              "checks": {"sources": 430, "scenes": 6, "model_pairs": 3,
                         "records_per_method": 7740, "raw_prediction_matches": 7740,
                         "one_call_budget_and_trace_checks": 7740},
              "metrics": summaries, "comparisons": comparisons}
    (OUT / "comparison.json").write_text(json.dumps(report, indent=2), encoding="utf-8")
    names = {"raw": "Raw", "diag_rule": "Diagnostic rule", "legacy_rule": "Legacy rule agent",
             "full": "Full four-call agent", "truncated": "Four-call agent truncated after one call",
             "trained_one_call": "Separately trained one-call agent"}
    lines = ["# Separately trained one-call control (2026-11-03)", "",
             "430 sources × 6 scenes × 3 paired classifiers. SD is the sample SD across fitted model pairs.",
             "Corrections and harms below are totals across 7,740 records.", "",
             "| Method | Accuracy | SD | Macro-F1 | Mean calls | Corrections | Harms |",
             "|---|---:|---:|---:|---:|---:|---:|"]
    for method, m in summaries.items():
        lines.append(f"| {names[method]} | {m['accuracy']:.4f} | {m['sample_sd']:.4f} | "
                     f"{m['macro_f1']:.4f} | {m['mean_calls']:.4f} | {m['fixes']} | {m['harms']} |")
    lines += ["", "Differences are separately trained one-call minus comparator. Accuracy differences are percentage points.",
              "Intervals use 10,000 source-cluster bootstrap resamples, retaining all scenes and model pairs.", "",
              "| Comparator | Accuracy difference (pp) | 95% CI (pp) | Call difference | 95% CI |",
              "|---|---:|---|---:|---|"]
    for key, cmp in comparisons.items():
        ref = key.removeprefix("trained_one_call_minus_")
        acc, calls = cmp["accuracy"], cmp["calls"]
        lines.append(f"| {names[ref]} | {acc['difference']*100:+.3f} | "
                     f"[{acc['ci95'][0]*100:+.3f}, {acc['ci95'][1]*100:+.3f}] | "
                     f"{calls['difference']:+.4f} | [{calls['ci95'][0]:+.4f}, {calls['ci95'][1]:+.4f}] |")
    lines += ["", "## Interpretation boundary", "",
              "The one-call policy is trained separately with the same sources, source-group split, architecture, classifier pairing and fixed call penalty. Its partial states and normalized remaining budget use a one-call limit. It can use the result of its single call for terminal output selection, but cannot call a second tool.",
              "", "This comparison changes the training observation distribution as well as the permitted horizon. The earlier frozen-policy truncation remains the direct test of additional calls within the original trained agent.",
              "", report["protocol"]["test_selection_note"], "",
              "Runtime fields are not used to claim latency superiority. No threshold or checkpoint was chosen using these new test results."]
    (OUT / "REPORT.md").write_text("\n".join(lines) + "\n", encoding="utf-8")
    print(json.dumps({"checks": report["checks"], "metrics": summaries,
                      "comparisons": comparisons}, indent=2))


if __name__ == "__main__":
    main()
