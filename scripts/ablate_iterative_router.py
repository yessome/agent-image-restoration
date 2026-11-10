"""循环 router 的反馈消融 / 调用上限 / 固定策略基线（同 checkpoint，不重训）。

条件：
  full             : 原样（复用 per_image.jsonl）
  no_feedback      : **只**遮蔽网络看到的已执行分支反馈（conf/margin/clip/fallback 置零成网络输入），
                     `choose_action` 的安全判断（真实 clip / fallback）仍然生效
  call_limit_1     : 保持原预算编码（budget=4 进 state），仅在已调用 1 次后强制 stop
  fixed_lowlight   : 所有图强制 Zero-DCE（对照，不参与选择）
  fixed_denoise    : 所有图强制 DnCNN
  fixed_type_oracle: 按**真实退化类型**匹配工具（低光→Zero-DCE，噪声→DnCNN）——不可部署的上界参考
  fixed_diag_rule  : 按**现有诊断器**匹配工具（低光分≥阈值且>噪声→Zero-DCE；噪声分≥阈值→DnCNN；否则 raw）
                     ——可部署的规则基线

所有差值与 CI 都从逐图结果计算；固定策略的 raw 返回比例恒为 0（它们总会施加工具）。
"""
from __future__ import annotations

import argparse
import json
import os
import sys
from pathlib import Path
from dataclasses import replace
from math import comb

import cv2
import numpy as np

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, ROOT)

from src.agents.orchestrator import ImageRestorationAgent  # noqa: E402
from src.main import _resolve_weight_paths, load_config  # noqa: E402
from src.router.iterative_router import (  # noqa: E402
    RouterAction, choose_action, make_state, extract_features_for,
)
from src.router.router import PipelineRoute  # noqa: E402

OUT = os.path.join(ROOT, "results", "iterative_valb_v1")
ABL = os.path.join(OUT, "ablation")


def read_samples(manifests=None):
    out = []
    pairs = manifests or [("lowlight", "data/degraded/valB_lowlight.tsv"),
                          ("noise", "data/degraded/valB_noise.tsv")]
    for sub, path in pairs:
        for line in open(os.path.join(ROOT, path), encoding="utf-8"):
            line = line.strip()
            if line and not line.startswith("#"):
                p = line.split("\t")
                out.append({"path": p[0], "label": p[1], "subset": sub,
                            "group": p[2] if len(p) > 2 else p[0]})
    return out


def custom_decide(router, diagnosis, outcomes, candidates,
                  mask_feedback=False, call_limit=None):
    """复刻 IterativeRouter.decide，但允许「只遮蔽网络输入」与「限制实际调用次数」。"""
    order = router.branch_order
    n_branches = len(order)
    names = {o.upstream for o in outcomes}
    available = np.array([up in candidates for up in order])
    observed = np.array([up in names for up in order])

    feats_true = extract_features_for(diagnosis, outcomes, order)
    feats_net = feats_true.copy()
    if mask_feedback:
        # 只把「网络看到的」反馈清零；安全判断稍后用 feats_true 覆盖回去
        for i in range(1, n_branches):
            base = 6 + i * 5
            feats_net[base + 1:base + 5] = 0.0

    calls = sum(1 for o in outcomes if o.upstream)
    remaining = max(0, router.budget - calls)          # 预算编码保持原样
    state_net = make_state(feats_net, available, observed, remaining, router.budget,
                           branch_order=order)
    probs = router.probabilities(state_net)

    state_safe = make_state(feats_true, available, observed, remaining, router.budget,
                            branch_order=order)
    hybrid = state_net.copy()
    for i in range(n_branches):                        # 安全列用真实值
        base = 6 + i * 5
        hybrid[base + 3] = state_safe[base + 3]        # clip
        hybrid[base + 4] = state_safe[base + 4]        # fallback

    act = choose_action(probs, hybrid, router.call_cost, router.clip_max, branch_order=order)
    if call_limit is not None and act.kind == "call" and calls >= call_limit:
        forced = hybrid.copy()
        forced[-1] = 0.0                               # 只让「本次」返回 stop
        stop = choose_action(probs, forced, router.call_cost, router.clip_max, branch_order=order)
        return RouterAction("stop", stop.upstream, "call_limit_ablation",
                            [float(v) for v in probs])
    return act


def run_agent(agent, samples, condition):
    rows = {}
    for i, s in enumerate(samples, 1):
        img = cv2.cvtColor(cv2.imread(s["path"], cv2.IMREAD_COLOR), cv2.COLOR_BGR2RGB)
        res = agent.run(img, ground_truth=s["label"], global_oracle=False)
        chosen = next(o for o in res.outcomes if o.pipeline_id == res.chosen_pipeline)
        raw = res.outcomes[0]
        rows[s["path"]] = {
            "path": s["path"], "subset": s["subset"], "group": s["group"], "label": s["label"],
            "raw_ok": bool(raw.is_correct), "chosen_ok": bool(chosen.is_correct),
            "selected": res.selected_upstream or "raw",
            "restore_calls": int(res.restore_calls),
            "reason": res.select_reason,
            "executed_ms": float(sum(o.upstream_latency_ms + o.downstream_latency_ms
                                     for o in res.outcomes)),
        }
        if i % 50 == 0:
            print(f"    {condition}: {i}/{len(samples)}", flush=True)
    return rows


def run_policy(agent, samples, condition, policy):
    """policy(sample, diagnosis) -> upstream（"" 表示 raw）。"""
    rows = {}
    for i, s in enumerate(samples, 1):
        img = cv2.cvtColor(cv2.imread(s["path"], cv2.IMREAD_COLOR), cv2.COLOR_BGR2RGB)
        diag = agent.diagnoser.diagnose(img)
        up = policy(s, diag)
        raw_o = agent._execute_pipeline(img, PipelineRoute("raw_cls", "", "classify", reason="raw"),
                                        {}, s["label"])
        if up:
            o = agent._execute_pipeline(
                img, PipelineRoute(f"{up}_cls", up, "classify", reason="fixed policy"),
                {}, s["label"])
            calls, sel, ms = 1, up, o.upstream_latency_ms + o.downstream_latency_ms
            ok = bool(o.is_correct)
        else:
            calls, sel, ms, ok = 0, "raw", raw_o.upstream_latency_ms + raw_o.downstream_latency_ms, bool(raw_o.is_correct)
        rows[s["path"]] = {
            "path": s["path"], "subset": s["subset"], "group": s["group"], "label": s["label"],
            "raw_ok": bool(raw_o.is_correct), "chosen_ok": ok, "selected": sel,
            "restore_calls": calls, "reason": "policy", "executed_ms": float(ms),
        }
        if i % 50 == 0:
            print(f"    {condition}: {i}/{len(samples)}", flush=True)
    return rows


def main():
    ap = argparse.ArgumentParser(description="循环 router 消融与基线（修正版）")
    ap.add_argument("--config", default=os.path.join(ROOT, "config", "iterative_agent_13tool.yaml"),
                    help="最终 13 分支配置")
    ap.add_argument("--device", default="cuda")
    ap.add_argument("--limit", type=int, default=0)
    ap.add_argument("--exclude", default="", help="排除的分支（逗号分隔）")
    ap.add_argument("--weight", help="循环 router checkpoint（默认取 config）")
    ap.add_argument("--classifier-weight", help="配对分类器权重（默认取 config）")
    ap.add_argument("--out-dir", default=ABL, help="结果目录")
    ap.add_argument("--manifests", default="",
                    help="如 lowlight=results/router_test_17branch/test_lowlight.tsv,"
                         "noise=results/router_test_17branch/test_noise.tsv")
    ap.add_argument("--conditions", default="no_feedback,call_limit_1,fixed_diag_rule",
                    help="要跑的条件，逗号分隔；可选 no_feedback,call_limit_1,fixed_lowlight,"
                         "fixed_denoise,fixed_type_oracle,fixed_diag_rule")
    args = ap.parse_args()
    ABL_OUT = args.out_dir
    os.makedirs(ABL_OUT, exist_ok=True)
    want = [c.strip() for c in args.conditions.split(",") if c.strip()]

    manifests = None
    if args.manifests:
        manifests = [tuple(x.split("=", 1)) for x in args.manifests.split(",")]
    samples = read_samples(manifests)
    if args.limit:
        samples = samples[: args.limit]
    print(f"样本 {len(samples)}")

    cfg = _resolve_weight_paths(load_config(args.config), ROOT)
    if args.weight:
        cfg.setdefault("iterative_router", {})["weight_path"] = str(Path(args.weight).resolve())
    if args.classifier_weight:
        cfg["downstreams"]["classify"]["weight_path"] = str(Path(args.classifier_weight).resolve())
    cfg.setdefault("routing", {})["mode"] = "iterative"
    cfg.setdefault("llm", {})["enabled"] = False
    agent = ImageRestorationAgent(cfg, args.device)
    router = agent._iterative_router
    print(f"router budget={router.budget} call_cost={router.call_cost} clip_max={router.clip_max}")

    orig_decide = router.decide
    res = {}

    if "no_feedback" in want:
        print("\n[no_feedback] 只遮蔽网络输入，安全判断用真实 clip/fallback")
        router.decide = lambda d, o, c: custom_decide(router, d, o, c, mask_feedback=True)
        res["no_feedback"] = run_agent(agent, samples, "no_feedback")
        router.decide = orig_decide

    if "call_limit_1" in want:
        print("\n[call_limit_1] 预算编码不变，仅限实际调用 1 次")
        router.decide = lambda d, o, c: custom_decide(router, d, o, c, call_limit=1)
        res["call_limit_1"] = run_agent(agent, samples, "call_limit_1")
        router.decide = orig_decide

    if "fixed_lowlight" in want:
        print("\n[fixed_lowlight] 全部强制 Zero-DCE")
        res["fixed_lowlight"] = run_policy(agent, samples, "fixed_lowlight", lambda s, d: "lowlight")

    if "fixed_denoise" in want:
        print("\n[fixed_denoise] 全部强制 DnCNN")
        res["fixed_denoise"] = run_policy(agent, samples, "fixed_denoise", lambda s, d: "denoise")

    if "fixed_type_oracle" in want:
        print("\n[fixed_type_oracle] 按真实退化类型匹配（不可部署参考）")
        res["fixed_type_oracle"] = run_policy(
            agent, samples, "fixed_type_oracle",
            lambda s, d: "lowlight" if s["subset"] == "lowlight" else "denoise")

    print("\n[fixed_diag_rule] 按现有诊断器匹配（可部署基线）")
    th = agent.confidence_threshold

    def diag_policy(s, d):
        n, l = d.noise_score, d.lowlight_score
        if l >= th and l > n:
            return "lowlight"
        if n >= th and n >= l:
            return "denoise"
        return ""

    res["fixed_diag_rule"] = run_policy(agent, samples, "fixed_diag_rule", diag_policy)

    json.dump({"conditions": res, "threshold": th, "budget": router.budget,
               "router_weight": str(agent.config["iterative_router"]["weight_path"]),
               "classifier_weight": str(agent.config["downstreams"]["classify"]["weight_path"])},
              open(os.path.join(ABL_OUT, "ablation.json"), "w", encoding="utf-8"),
              ensure_ascii=False, indent=2)
    print(f"\n写出 {os.path.join(ABL_OUT, 'ablation.json')}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
