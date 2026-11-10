"""六场景 test 上的 13 工具机制消融（call_limit_1 / no_feedback）。

Full 条件复用既有记录（已通过 9 张复现核查：pred / chosen branch / calls 全一致）。

复用规则（不是启发式，是逐位等价）：
- 初始 state 的预算位是归一化后的 1.0（remaining=4 / budget=4），不是原始计数；
  且 raw 已执行，raw 的反馈（pred/conf/margin/clip/fallback）参与初始决策。
  这两个事实在两个条件下完全相同，因此复用成立。
- call_limit_1：Full 调用 ≤1 次的样本，第 1 步 state 与强制停时完全相同，
  probs 与 best 一致，结果必然相同；只有 Full 调用 ≥2 次的样本需要重跑。
- no_feedback：Full 调用 0 次的样本，第 1 步只执行了 raw，而遮蔽只作用于
  i>=1 的分支（raw 的反馈保留），state 与 Full 逐位相同；其余需要重跑。

两个条件都不重训、不改阈值、不改 checkpoint。
"""
from __future__ import annotations

import argparse
import csv
import json
import os
import sys
import time
from pathlib import Path

import cv2
import numpy as np

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

from src.agents.orchestrator import ImageRestorationAgent  # noqa: E402
from src.main import _resolve_weight_paths, load_config  # noqa: E402
from src.router.iterative_router import (RouterAction, choose_action,  # noqa: E402
                                        extract_features_for, make_state)

OUT = ROOT / "results" / "ablation_13tool_test"
SCENE_TSV = ROOT / "results" / "table2_scenes"
DEG = ROOT / "data" / "degraded"
DEG_TEST = ROOT / "data" / "degraded_test_13tool"
DIRS = ("scene_matrix_13tool", "test_matrix_13tool")


def uid_of(path: str) -> str:
    return path.replace("\\", "/").split("/")[-1].lower()


def read_manifest(tsv: Path, subset: str):
    out = []
    with open(tsv, encoding="utf-8") as fh:
        rd = csv.reader(fh, delimiter="\t")
        header = next(rd)
        header = [h.lstrip("#").strip() for h in header]
        idx = {h: i for i, h in enumerate(header)}
        for row in rd:
            if not row:
                continue
            path = row[idx["path"]] if "path" in idx else row[idx["degraded_path"]]
            src = row[idx["source_path"]] if "source_path" in idx else path
            out.append({"path": path, "label": row[idx["label"]], "subset": subset,
                        "group": uid_of(src)})
    return out


def load_full(seed: int):
    rec = {}
    for d in DIRS:
        with open(ROOT / "results" / d / f"seed{seed}" / "per_image.csv", encoding="utf-8") as fh:
            for r in csv.DictReader(fh):
                if r["method"] != "iterative":
                    continue
                rec[(r["subset"], uid_of(r["path"]))] = {
                    "pred": r["pred"], "ok": int(r["ok"]), "raw_ok": int(r["raw_ok"]),
                    "selected": r["chosen_upstream"] or "raw",
                    "restore_calls": int(r["restore_calls"]),
                }
    return rec


def custom_decide(router, diagnosis, outcomes, candidates, *,
                  mask_feedback=False, call_limit=None):
    order = router.branch_order
    n = len(order)
    names = {o.upstream for o in outcomes}
    available = np.array([up in candidates for up in order])
    observed = np.array([up in names for up in order])

    feats_true = extract_features_for(diagnosis, outcomes, order)
    feats_net = feats_true.copy()
    if mask_feedback:
        for i in range(1, n):                     # raw 的反馈不遮蔽
            base = 6 + i * 5
            feats_net[base + 1:base + 5] = 0.0

    calls = sum(1 for o in outcomes if o.upstream)
    remaining = max(0, router.budget - calls)      # 预算编码保持原样
    state_net = make_state(feats_net, available, observed, remaining, router.budget,
                           branch_order=order)
    probs = router.probabilities(state_net)

    state_true = make_state(feats_true, available, observed, remaining, router.budget,
                            branch_order=order)
    hybrid = state_net.copy()
    for i in range(n):                             # 安全判断用真实 clip / fallback
        base = 6 + i * 5
        hybrid[base + 3] = state_true[base + 3]
        hybrid[base + 4] = state_true[base + 4]

    act = choose_action(probs, hybrid, router.call_cost, router.clip_max, branch_order=order)
    if call_limit is not None and act.kind == "call" and calls >= call_limit:
        forced = hybrid.copy()
        forced[-1] = 0.0
        stop = choose_action(probs, forced, router.call_cost, router.clip_max, branch_order=order)
        return RouterAction("stop", stop.upstream, "call_limit_ablation",
                            [float(v) for v in probs])
    return act


def run_one(agent, item):
    img = cv2.cvtColor(cv2.imread(item["path"], cv2.IMREAD_COLOR), cv2.COLOR_BGR2RGB)
    t0 = time.perf_counter()
    res = agent.run(img, ground_truth=item["label"], global_oracle=False)
    ms = (time.perf_counter() - t0) * 1000
    chosen = next(o for o in res.outcomes if o.pipeline_id == res.chosen_pipeline)
    raw = res.outcomes[0]
    return {"pred": chosen.pred_label, "ok": int(bool(chosen.is_correct)),
            "raw_ok": int(bool(raw.is_correct)),
            "selected": res.selected_upstream or "raw",
            "restore_calls": int(res.restore_calls),
            "stop_reason": res.select_reason, "wall_ms": ms}


def build_agent(seed):
    cfg = _resolve_weight_paths(load_config(str(ROOT / "config" / "iterative_agent_13tool.yaml")),
                                str(ROOT))
    cfg.setdefault("routing", {})["mode"] = "iterative"
    cfg.setdefault("llm", {})["enabled"] = False
    cfg["router"]["learned_weight"] = ""
    cfg["iterative_router"]["weight_path"] = str(
        ROOT / f"weights/router_iterative_feedback_13tool_seed{seed}.pth")
    cfg["downstreams"]["classify"]["weight_path"] = str(
        ROOT / f"weights/trashnet_resnet50_seed{seed}.pth")
    return ImageRestorationAgent(cfg, "cuda")


def main() -> int:
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--seeds", default="1,2,3")
    ap.add_argument("--conditions", default="call_limit_1,no_feedback")
    ap.add_argument("--limit", type=int, default=0)
    ap.add_argument("--resume", action="store_true")
    args = ap.parse_args()

    items = []
    items += read_manifest(SCENE_TSV / "clean.tsv", "clean")
    for name in ("jpeg", "motion_blur", "mixed"):
        items += read_manifest(DEG / name / "manifest_full.tsv", name)
    for name in ("lowlight", "noise"):
        items += read_manifest(DEG_TEST / name / "manifest_full.tsv", name)
    if args.limit:
        items = items[: args.limit]
    print(f"样本 {len(items)}（六场景），条件 {args.conditions}")

    OUT.mkdir(parents=True, exist_ok=True)
    want = [c.strip() for c in args.conditions.split(",") if c.strip()]

    for seed in [int(s) for s in args.seeds.split(",")]:
        full = load_full(seed)
        missing = [i for i in items if (i["subset"], uid_of(i["path"])) not in full]
        if missing:
            raise RuntimeError(f"seed{seed}: {len(missing)} 张缺 Full 记录")
        agent = build_agent(seed)
        router = agent._iterative_router
        orig = router.decide

        for cond in want:
            path = OUT / f"seed{seed}" / f"{cond}.jsonl"
            path.parent.mkdir(parents=True, exist_ok=True)
            done = {}
            if path.exists() and args.resume:
                for line in path.read_text(encoding="utf-8").splitlines():
                    if line.strip():
                        r = json.loads(line)
                        done[r["group"] + "|" + r["subset"]] = r
            elif path.exists() and not args.resume:
                raise FileExistsError(f"{path} 已存在，用 --resume")

            need, reuse = [], 0
            for it in items:
                key = it["group"] + "|" + it["subset"]
                if key in done:
                    continue
                f = full[(it["subset"], uid_of(it["path"]))]
                if cond == "call_limit_1":
                    keep = f["restore_calls"] <= 1
                else:                              # no_feedback
                    keep = f["restore_calls"] == 0
                if keep:
                    reuse += 1
                    rec = {"group": it["group"], "subset": it["subset"], "path": it["path"],
                           "label": it["label"], "condition": cond, "source": "reused_full",
                           **f, "stop_reason": "unknown", "stop_reason_source": "stop_reason_not_recorded_in_full_csv"}
                else:
                    need.append(it)
                    rec = None
                if rec:
                    done[key] = rec
            print(f"[seed{seed} {cond}] 复用 {reuse}，需重跑 {len(need)}", flush=True)

            if cond == "call_limit_1":
                router.decide = lambda d, o, c: custom_decide(router, d, o, c, call_limit=1)
            else:
                router.decide = lambda d, o, c: custom_decide(router, d, o, c, mask_feedback=True)
            t0 = time.perf_counter()
            with path.open("a", encoding="utf-8") as fh:
                for k, it in enumerate(need, 1):
                    r = run_one(agent, it)
                    rec = {"group": it["group"], "subset": it["subset"], "path": it["path"],
                           "label": it["label"], "condition": cond, "source": "rerun", **r}
                    fh.write(json.dumps(rec, ensure_ascii=False) + "\n")
                    fh.flush()
                    os.fsync(fh.fileno())
                    done[it["group"] + "|" + it["subset"]] = rec
                    if k % 100 == 0:
                        el = time.perf_counter() - t0
                        print(f"    {k}/{len(need)}  {el/k:.2f}s/张", flush=True)
            router.decide = orig
            # 复用记录补写，保证文件完整
            if reuse:
                with path.open("a", encoding="utf-8") as fh:
                    for key, rec in done.items():
                        if rec["source"] == "reused_full":
                            fh.write(json.dumps(rec, ensure_ascii=False) + "\n")
                            rec["source"] = "reused_written"
                    fh.flush()
                    os.fsync(fh.fileno())
            print(f"[seed{seed} {cond}] 完成，共 {len(done)} 条", flush=True)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
