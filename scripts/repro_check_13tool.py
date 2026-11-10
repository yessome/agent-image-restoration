"""复现核查：用同一 checkpoint 重跑 Full，与既有六场景记录逐字段比对。

样本覆盖三个 seed 与三条路径（raw 停止 / 恰好一次调用 / 多次调用），
比对 pred、所执行分支、调用次数。不一致时先查输入、配置、权重、预处理，
不直接开始消融。
"""
from __future__ import annotations

import argparse
import csv
import json
import os
from pathlib import Path

import cv2
import numpy as np

ROOT = Path(__file__).resolve().parents[1]
import sys  # noqa: E402
sys.path.insert(0, str(ROOT))
from src.agents.orchestrator import ImageRestorationAgent  # noqa: E402
from src.main import _resolve_weight_paths, load_config  # noqa: E402

DIRS = ("scene_matrix_13tool", "test_matrix_13tool")
OUT = ROOT / "results" / "ablation_13tool_test"


def load_csv(seed):
    rows = []
    for d in DIRS:
        with open(ROOT / "results" / d / f"seed{seed}" / "per_image.csv", encoding="utf-8") as fh:
            for r in csv.DictReader(fh):
                if r["method"] == "iterative":
                    r["_dir"] = d
                    rows.append(r)
    return rows


def pick(rows, calls, seed):
    """每个调用档挑一张，优先不同 subset。"""
    cand = [r for r in rows if int(r["restore_calls"]) == calls]
    if not cand:
        return None
    seen, out = set(), []
    for r in cand:
        if r["subset"] not in seen:
            seen.add(r["subset"])
            out.append(r)
        if len(out) >= 2:
            break
    return out[0]


def main() -> int:
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--seeds", default="1,2,3")
    ap.add_argument("--device", default="cuda")
    args = ap.parse_args()

    cfg = _resolve_weight_paths(load_config(str(ROOT / "config" / "iterative_agent_13tool.yaml")),
                                str(ROOT))
    cfg.setdefault("routing", {})["mode"] = "iterative"
    cfg.setdefault("llm", {})["enabled"] = False
    cfg["router"]["learned_weight"] = ""
    agent = ImageRestorationAgent(cfg, args.device)
    router = agent._iterative_router
    print(f"router={cfg['iterative_router']['weight_path']}")
    print(f"budget={router.budget} cost={router.call_cost} branches={router.n_branches}")

    report, all_ok = [], True
    for seed in [int(s) for s in args.seeds.split(",")]:
        # 每个 seed 用配对的 router/classifier
        cfg2 = _resolve_weight_paths(load_config(str(ROOT / "config" / "iterative_agent_13tool.yaml")),
                                     str(ROOT))
        cfg2.setdefault("routing", {})["mode"] = "iterative"
        cfg2.setdefault("llm", {})["enabled"] = False
        cfg2["router"]["learned_weight"] = ""
        cfg2["iterative_router"]["weight_path"] = str(
            ROOT / f"weights/router_iterative_feedback_13tool_seed{seed}.pth")
        cfg2["downstreams"]["classify"]["weight_path"] = str(
            ROOT / f"weights/trashnet_resnet50_seed{seed}.pth")
        ag = ImageRestorationAgent(cfg2, args.device)

        rows = load_csv(seed)
        for calls in (0, 1, 2):
            rec = pick(rows, calls, seed)
            if rec is None:
                print(f"seed{seed} calls={calls}: 无样本")
                continue
            img = cv2.cvtColor(cv2.imread(rec["path"], cv2.IMREAD_COLOR), cv2.COLOR_BGR2RGB)
            res = ag.run(img, ground_truth=rec["label"], global_oracle=False)
            chosen = next(o for o in res.outcomes if o.pipeline_id == res.chosen_pipeline)
            got = {"pred": chosen.pred_label,
                   "chosen_upstream": res.selected_upstream or "raw",
                   "restore_calls": int(res.restore_calls),
                   "ok": int(bool(chosen.is_correct))}
            exp = {"pred": rec["pred"],
                   "chosen_upstream": rec["chosen_upstream"] or "raw",
                   "restore_calls": int(rec["restore_calls"]),
                   "ok": int(rec["ok"])}
            same = got == exp
            all_ok &= same
            report.append({"seed": seed, "subset": rec["subset"], "path": rec["path"],
                           "expected": exp, "reproduced": got, "match": same})
            print(f"  seed{seed} {rec['subset']:12s} calls={exp['restore_calls']} "
                  f"{'OK ' if same else 'MISMATCH'} exp={exp} got={got}", flush=True)

    OUT.mkdir(parents=True, exist_ok=True)
    (OUT / "repro_check.json").write_text(
        json.dumps({"all_match": all_ok, "n": len(report), "cases": report},
                   ensure_ascii=False, indent=1), encoding="utf-8")
    print(f"\n全部一致: {all_ok}  样本 {len(report)}")
    return 0 if all_ok else 1


if __name__ == "__main__":
    raise SystemExit(main())
