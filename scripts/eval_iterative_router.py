"""Evaluate the learned feedback loop; legacy rule results are a separate baseline."""
from __future__ import annotations

import argparse
from collections import Counter
import hashlib
import json
import os
from pathlib import Path
import sys
import time

import cv2
import numpy as np

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
from src.agents.orchestrator import ImageRestorationAgent
from src.main import load_config, _resolve_weight_paths
from src.router.iterative_router import file_hash


def main():
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--manifest", action="append", required=True)
    ap.add_argument("--config", default="config/iterative_agent_13tool.yaml")
    ap.add_argument("--exclude", default="",
                    help="排除的分支（逗号分隔），必须与 router checkpoint 的分支表一致")
    ap.add_argument("--weight", help="Matched iterative router checkpoint")
    ap.add_argument("--classifier-weight", help="Must match checkpoint collection classifier")
    ap.add_argument("--out-dir", required=True)
    ap.add_argument("--device", default="cuda")
    ap.add_argument("--limit", type=int, default=0)
    ap.add_argument("--resume", action="store_true")
    ap.add_argument("--oracle-audit", action="store_true",
                    help="Execute all unseen tools AFTER stopping, solely for oracle/regret evaluation")
    args = ap.parse_args()

    cfg = _resolve_weight_paths(load_config(args.config), str(ROOT))
    if args.exclude:
        cfg.setdefault("model_toolbox", {})["exclude"] = [
            up.strip() for up in args.exclude.split(",") if up.strip()]
    cfg.setdefault("routing", {})["mode"] = "iterative"
    cfg.setdefault("router", {})["learned_weight"] = ""
    cfg.setdefault("llm", {})["enabled"] = False
    if args.weight:
        cfg.setdefault("iterative_router", {})["weight_path"] = str(Path(args.weight).resolve())
    if args.classifier_weight:
        cfg["downstreams"]["classify"]["weight_path"] = str(Path(args.classifier_weight).resolve())
    agent = ImageRestorationAgent(cfg, args.device)
    samples = []
    for manifest in args.manifest:
        for line in Path(manifest).read_text(encoding="utf-8").splitlines():
            if line and not line.startswith("#"):
                row = line.split("\t")
                image = Path(row[0])
                if not image.is_absolute():
                    image = ROOT / image
                samples.append((str(image.resolve()), row[1]))
    if len({p for p, _ in samples}) != len(samples):
        raise ValueError("Duplicate image paths in evaluation manifests")
    if args.limit:
        samples = samples[:args.limit]
    router_path = Path(agent.config["iterative_router"]["weight_path"])
    if not router_path.is_absolute():
        router_path = ROOT / router_path
    provenance = {"schema": "iterative-eval-v1", "config": agent.config,
                  "router_sha256": file_hash(router_path),
                  "classifier_sha256": file_hash(cfg["downstreams"]["classify"]["weight_path"]),
                  "manifests": {str(Path(p).resolve()): file_hash(p) for p in args.manifest},
                  "tool_sha256": {name: file_hash(c["weight_path"])
                                    for name, c in agent.config["upstreams"].items()},
                  "oracle_audit": args.oracle_audit, "device": args.device}
    signature = hashlib.sha256(json.dumps(provenance, sort_keys=True).encode()).hexdigest()
    out = Path(args.out_dir)
    out.mkdir(parents=True, exist_ok=True)
    meta_file = out / "run_meta.json"
    if meta_file.exists():
        if not args.resume:
            raise FileExistsError("Existing evaluation: use --resume or another output directory")
        if json.loads(meta_file.read_text(encoding="utf-8"))["signature"] != signature:
            raise ValueError("Resume provenance differs; refusing to mix results")
    else:
        meta_file.write_text(json.dumps({"signature": signature, **provenance}, indent=2), encoding="utf-8")
    results_file = out / "per_image.jsonl"
    done = {}
    if results_file.exists():
        for line in results_file.read_text(encoding="utf-8").splitlines():
            r = json.loads(line)
            if r["path"] in done:
                raise ValueError("Duplicate evaluation record")
            done[r["path"]] = r
    with results_file.open("a", encoding="utf-8") as handle:
        for i, (path, label) in enumerate(samples, 1):
            sha = file_hash(path)
            if path in done:
                if done[path]["input_sha256"] != sha or done[path]["label"] != label:
                    raise ValueError("Cached image/label changed")
                continue
            bgr = cv2.imread(path)
            if bgr is None:
                raise ValueError(f"Unreadable image: {path}")
            t0 = time.perf_counter()
            res = agent.run(cv2.cvtColor(bgr, cv2.COLOR_BGR2RGB), ground_truth=label,
                            global_oracle=args.oracle_audit)
            wall_ms = (time.perf_counter()-t0)*1000
            raw = res.outcomes[0]
            chosen = next(o for o in res.outcomes if o.pipeline_id == res.chosen_pipeline)
            record = {"path": path, "label": label, "input_sha256": sha,
                      "raw_pred": raw.pred_label, "chosen_pred": chosen.pred_label,
                      "raw_ok": raw.is_correct, "chosen_ok": chosen.is_correct,
                      "restore_calls": res.restore_calls,
                      "executed_model_ms": sum(o.upstream_latency_ms+o.downstream_latency_ms for o in res.outcomes),
                      "wall_ms_including_audit": wall_ms, "result": res.to_dict()}
            handle.write(json.dumps(record, ensure_ascii=False)+"\n")
            handle.flush()
            os.fsync(handle.fileno())
            done[path] = record
            print(f"{i}/{len(samples)} calls={res.restore_calls} selected={chosen.upstream or 'raw'}", flush=True)
    rows = [done[p] for p, _ in samples]
    if not rows:
        raise ValueError("Empty evaluation")
    labels = cfg["downstreams"]["classify"]["labels"]

    def macro_f1(key):
        scores = []
        for label in labels:
            tp = sum(r[key] == label and r["label"] == label for r in rows)
            fp = sum(r[key] == label and r["label"] != label for r in rows)
            fn = sum(r[key] != label and r["label"] == label for r in rows)
            scores.append(2*tp/(2*tp+fp+fn) if 2*tp+fp+fn else 0)
        return float(np.mean(scores))
    summary = {"n": len(rows), "raw_accuracy": float(np.mean([r["raw_ok"] for r in rows])),
               "accuracy": float(np.mean([r["chosen_ok"] for r in rows])),
               "raw_macro_f1": macro_f1("raw_pred"), "macro_f1": macro_f1("chosen_pred"),
               "fixes": sum(not r["raw_ok"] and r["chosen_ok"] for r in rows),
               "harms": sum(r["raw_ok"] and not r["chosen_ok"] for r in rows),
               "mean_restore_calls": float(np.mean([r["restore_calls"] for r in rows])),
               "selection_counts": dict(Counter(r["result"]["selected_upstream"] or "raw" for r in rows)),
               "mean_executed_model_ms": float(np.mean([r["executed_model_ms"] for r in rows])),
               "timing_note": "executed_model_ms excludes diagnosis/router; wall includes optional audit",
               "oracle_scope": "admitted_candidates" if args.oracle_audit else "executed_only"}
    if args.oracle_audit:
        summary["candidate_regret"] = float(np.mean([r["result"]["regret"] for r in rows]))
        summary["global_regret"] = float(np.mean([r["result"]["global_regret"] for r in rows]))
    (out / "summary.json").write_text(json.dumps(summary, indent=2), encoding="utf-8")
    print(json.dumps(summary, indent=2))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
