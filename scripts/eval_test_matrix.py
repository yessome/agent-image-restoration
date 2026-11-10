"""test 主对照：五种方法 × 三 seed，修复共享、分类与决策独立计算。

方法：
  raw          原图直通
  fixed_type   固定工具（按清单退化类型：噪声→DnCNN，低光→Zero-DCE；参考线，非可部署）
  diag_rule    诊断规则（现有诊断器 + confidence_threshold，可部署）
  legacy_rule  legacy 规则 Agent（诊断候选裁剪 + 规则验收，可部署）
  iterative    learned 循环 Agent（router 决策，最多 max_calls 次调用）

关键口径：
  * 每张图的 16 个工具只修复一次，三 seed 共享；分类按 seed 各算一次。
  * 决策只使用「被准入的候选」与「已执行分支」的反馈；审计分支结果不回流。
  * 审计分支（未被该方法执行的工具）耗时单列在 audit_ms，不计入 decision_ms。
  * Oracle/regret 在完整候选池（raw + 16 工具）上计算。

产物：results/test_matrix_seeds3/seed{s}/per_image.csv（每图每方法一行，flush+fsync）
"""
from __future__ import annotations

import argparse
import csv
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
from src.router.iterative_router import IterativeRouter, file_hash  # noqa: E402
from src.router.tool_registry import ALL_UPSTREAM_TOOLS  # noqa: E402

MANIFESTS = {
    "lowlight": os.path.join(ROOT, "results", "router_test_17branch", "test_lowlight.tsv"),
    "noise": os.path.join(ROOT, "results", "router_test_17branch", "test_noise.tsv"),
}
METHODS = ["raw", "fixed_type", "fixed_denoise", "fixed_lowlight",
           "diag_rule", "legacy_rule", "iterative"]
HEADER = ["path", "label", "subset", "group", "seed", "method", "chosen_upstream",
          "pred", "ok", "raw_ok", "restore_calls", "decision_ms", "audit_ms",
          "oracle_full_ok", "oracle_admitted_ok", "hit_full", "regret_full",
          "n_admitted", "n_executed", "noise_score", "lowlight_score", "diag_bucket"]


def sync_cuda():
    try:
        import torch
        if torch.cuda.is_available():
            torch.cuda.synchronize()
    except Exception:
        pass


def read_manifest(subset, path):
    rows = []
    for line in open(path, encoding="utf-8"):
        line = line.strip()
        if not line or line.startswith("#"):
            continue
        p = line.split("\t")
        rows.append({"path": p[0], "label": p[1], "group": p[2], "subset": subset})
    return rows


def diag_bucket(diag, th=0.6, low=0.4):
    n, l = diag.noise_score, diag.lowlight_score
    if n < low and l < low:
        return "clean"
    if n >= th and l >= th:
        return "both"
    if n >= th:
        return "noise"
    if l >= th:
        return "lowlight"
    return "gray"


def rule_pick(diag, th):
    """与 pre_scheduler.rule_select 一致：只看诊断分数与既有阈值。"""
    n, l = diag.noise_score, diag.lowlight_score
    if l >= th and l > n:
        return "lowlight"
    if n >= th and n >= l:
        return "denoise"
    return ""


def main() -> int:
    ap = argparse.ArgumentParser(description="test 主对照（5 方法 × 3 seed）")
    ap.add_argument("--seeds", default="1,2,3")
    ap.add_argument("--device", default="cuda")
    ap.add_argument("--out-dir", default=os.path.join(ROOT, "results", "test_matrix_seeds3"))
    ap.add_argument("--limit", type=int, default=0)
    ap.add_argument("--resume", action="store_true")
    ap.add_argument("--config", default="config/iterative_agent_13tool.yaml")
    ap.add_argument("--exclude", default="",
                    help="排除的分支（逗号分隔），与 router checkpoint 的分支表一致")
    ap.add_argument("--no-audit", action="store_true",
                    help="不跑完整 16 工具审计（只修复准入候选+两种固定工具），"
                         "oracle_full/hit/regret 记为 NA")
    ap.add_argument("--router-template", default="",
                    help="router 权重文件名模板，如 router_iterative_feedback_13tool_seed{seed}.pth")
    ap.add_argument("--scene-manifests", default="",
                    help="自定义场景清单，如 clean=results/table2_scenes/clean.tsv,jpeg=...")
    ap.add_argument("--verify-replay", type=int, default=0,
                    help="用 valB 前 N 张校验循环 replay 与真实评测逐图一致，不写 test 结果")
    args = ap.parse_args()

    seeds = [int(s) for s in args.seeds.split(",")]
    if args.exclude:
        from src.router import iterative_router as _ir
        _order = _ir.configure_branches(exclude=[x.strip() for x in args.exclude.split(",") if x.strip()])
        print(f"分支表切换为 {len(_order)} 个", flush=True)
    cfg = _resolve_weight_paths(load_config(os.path.join(ROOT, args.config)), ROOT)
    cfg.setdefault("routing", {})["mode"] = "legacy"       # 只用内部函数，不走 run()
    cfg.setdefault("llm", {})["enabled"] = False
    cfg["router"]["learned_weight"] = ""                   # 确保规则路径不用 learned
    cfg.setdefault("model_toolbox", {})["enabled"] = True
    agent = ImageRestorationAgent(cfg, args.device)
    th = float(agent.confidence_threshold)
    dcfg = cfg["downstreams"]["classify"]

    clf_paths = {s: os.path.join(ROOT, "weights", f"trashnet_resnet50_seed{s}.pth") for s in seeds}
    clfs = {s: ClassifyProcessor(device=args.device, weight_path=clf_paths[s],
                                num_classes=int(dcfg.get("num_classes", 6)),
                                labels=list(dcfg.get("labels") or [])) for s in seeds}
    routers = {}
    for s in seeds:
        rp = os.path.join(ROOT, "weights", args.router_template.format(seed=s) if args.router_template
                          else f"router_iterative_feedback_seed{s}.pth")
        routers[s] = IterativeRouter(rp, args.device, classifier_weight=clf_paths[s])
        print(f"seed{s}: router={os.path.basename(rp)} sha={file_hash(rp)[:12]} "
              f"classifier sha={file_hash(clf_paths[s])[:12]} budget={routers[s].budget}")

    # ---------- 校验模式：replay 是否与真实评测一致 ----------
    if args.verify_replay:
        return verify_replay(agent, clfs, routers, seeds, args.verify_replay, th)

    os.makedirs(args.out_dir, exist_ok=True)
    if args.scene_manifests:
        samples = []
        for item in args.scene_manifests.split(","):
            name, path = item.split("=", 1)
            samples += read_manifest(name, path)
    else:
        samples = (read_manifest("lowlight", MANIFESTS["lowlight"])
                   + read_manifest("noise", MANIFESTS["noise"]))
    if args.limit:
        samples = samples[: args.limit]

    done = set()
    if args.resume:
        for s in seeds:
            p = os.path.join(args.out_dir, f"seed{s}", "per_image.csv")
            if os.path.isfile(p):
                with open(p, encoding="utf-8") as f:
                    for r in csv.DictReader(f):
                        if r["method"] == "raw":
                            done.add(r["path"])
        samples = [x for x in samples if x["path"] not in done]

    print(f"待处理 {len(samples)} 张（已完成 {len(done)}）")
    writers, files = {}, {}
    for s in seeds:
        d = os.path.join(args.out_dir, f"seed{s}")
        os.makedirs(d, exist_ok=True)
        p = os.path.join(d, "per_image.csv")
        new = not os.path.isfile(p) or os.path.getsize(p) == 0
        f = open(p, "a", newline="", encoding="utf-8")
        w = csv.writer(f)
        if new:
            w.writerow(HEADER)
        writers[s], files[s] = w, f

    t0 = time.time()
    for i, smp in enumerate(samples, 1):
        img = cv2.cvtColor(cv2.imread(smp["path"], cv2.IMREAD_COLOR), cv2.COLOR_BGR2RGB)
        label = smp["label"]
        diag = agent.diagnoser.diagnose(img)
        routes = agent._gen_candidates(diag, "classify")
        admitted = {r.upstream for r in routes}

        # ---- 16 工具各修复一次（跨 seed 共享） ----
        restored, fb, clip, t_rest = {}, {}, {}, {}
        restored[""], fb[""], clip[""], t_rest[""] = img, False, 0.0, 0.0
        need = set(ALL_UPSTREAM_TOOLS) if not args.no_audit else (
            (admitted - {""}) | {"denoise", "lowlight"})
        for up in [u for u in ALL_UPSTREAM_TOOLS if u in need]:
            sync_cuda()
            a = time.perf_counter()
            out = agent._get_upstream(up).restore(img)
            sync_cuda()
            t_rest[up] = (time.perf_counter() - a) * 1000.0
            restored[up] = out.image
            fb[up] = bool((out.meta or {}).get("fallback", False))
            clip[up] = float(agent._clip_ratio(out.image))

        for s in seeds:
            outcomes, t_cls = {}, {}
            for up in [""] + [u for u in ALL_UPSTREAM_TOOLS if u in restored]:
                src = restored[up]
                sync_cuda()
                a = time.perf_counter()
                pred, conf, margin = classify(clfs[s], src)
                sync_cuda()
                t_cls[up] = (time.perf_counter() - a) * 1000.0
                outcomes[up] = PipelineOutcome(
                    pipeline_id=f"{up or 'raw'}_cls", upstream=up, downstream="classify",
                    score=conf, downstream_confidence=conf, margin=margin, pred_label=pred,
                    clip_ratio=clip[up], fallback=fb[up],
                    ground_truth_score=1.0 if pred == label else 0.0,
                    is_correct=(pred == label))

            have_full = not args.no_audit
            oracle_full = any(o.is_correct for o in outcomes.values()) if have_full else False
            oracle_adm = any(outcomes[u].is_correct for u in admitted)

            def cost(ups):
                return float(sum(t_rest[u] + t_cls[u] for u in ups))

            decisions = {}
            # raw
            decisions["raw"] = ("", set())
            # fixed_type（按清单退化类型；参考线）
            ft = "denoise" if smp["subset"] == "noise" else "lowlight"
            decisions["fixed_type"] = (ft, {ft})
            decisions["fixed_denoise"] = ("denoise", {"denoise"})
            decisions["fixed_lowlight"] = ("lowlight", {"lowlight"})
            # diag_rule
            dr = rule_pick(diag, th)
            decisions["diag_rule"] = (dr, {dr} if dr else set())
            # legacy_rule：候选池内规则验收
            adm_list = [outcomes[u] for u in sorted(admitted, key=lambda x: (x != "", x))]
            raw_o = outcomes[""]
            chosen_rule = agent._select_online(adm_list, raw_o)
            decisions["legacy_rule"] = (chosen_rule.upstream, admitted - {""})
            # iterative：只观测已执行分支的 replay
            chosen_it, exec_it = replay(routers[s], diag, outcomes, admitted)
            decisions["iterative"] = (chosen_it, exec_it)

            for m in METHODS:
                up, executed = decisions[m]
                o = outcomes[up]
                audit = (set(restored) - {""}) - executed   # 只统计真正跑过的分支
                writers[s].writerow([
                    smp["path"], label, smp["subset"], smp["group"], s, m,
                    up or "raw", o.pred_label, int(o.is_correct), int(outcomes[""].is_correct),
                    len(executed), round(cost(executed), 2), round(cost(audit), 2),
                    int(oracle_full) if have_full else "",
                    int(oracle_adm),
                    int(o.ground_truth_score >= (1.0 if oracle_full else 0.0)) if have_full else "",
                    round((1.0 if oracle_full else 0.0) - o.ground_truth_score, 4) if have_full else "",
                    len(admitted), len(executed),
                    round(float(diag.noise_score), 4), round(float(diag.lowlight_score), 4),
                    diag_bucket(diag),
                ])
        for s in seeds:
            files[s].flush()
            os.fsync(files[s].fileno())
        if i % 20 == 0:
            el = time.time() - t0
            print(f"  {i}/{len(samples)} | {el/60:.1f} 分钟 | 剩余约 {el/i*(len(samples)-i)/60:.0f} 分钟",
                  flush=True)

    for f in files.values():
        f.close()
    print(f"完成 {len(samples)} 张，用时 {(time.time()-t0)/60:.1f} 分钟")
    return 0


def classify(clf, image):
    """ClassifyProcessor 单图分类，返回 (pred, conf, margin)。"""
    res = clf.process(image)
    result = res.result or {}
    pred = max(result, key=result.get)
    vals = sorted(result.values(), reverse=True)
    conf = float(vals[0]) if vals else 0.0
    margin = float(vals[0] - vals[1]) if len(vals) > 1 else 0.0
    return pred, conf, margin


def replay(router, diag, outcomes, admitted):
    """复现 _run_iterative 的决策序列：只用已执行分支，禁止提前看审计结果。"""
    observed = [outcomes[""]]
    executed = set()
    for step in range(router.budget + 1):
        act = router.decide(diag, observed, set(admitted))
        if act.kind == "stop":
            return act.upstream, executed
        if act.kind != "call" or act.upstream not in admitted or act.upstream in executed:
            raise RuntimeError(f"非法动作 {act.kind}:{act.upstream}")
        observed.append(outcomes[act.upstream])
        executed.add(act.upstream)
    raise RuntimeError("router 未在预算内停止")


def verify_replay(agent, clfs, routers, seeds, n, th):
    """在 valB 前 n 张上比对 replay 与真实评测的 chosen/calls。"""
    half = max(1, n // 2)
    samples = read_manifest("lowlight", os.path.join(ROOT, "data", "degraded", "valB_lowlight.tsv"))[:half]
    samples += read_manifest("noise", os.path.join(ROOT, "data", "degraded", "valB_noise.tsv"))[:half]
    gt = {}
    for s in seeds:
        p = os.path.join(ROOT, "results", f"iterative_valb_seed{s}", "per_image.jsonl")
        d = {}
        for line in open(p, encoding="utf-8"):
            r = json.loads(line)
            d[r["path"]] = (r["result"]["selected_upstream"] or "", int(r["restore_calls"]))
        gt[s] = d

    bad = 0
    tot = 0
    for smp in samples:
        img = cv2.cvtColor(cv2.imread(smp["path"], cv2.IMREAD_COLOR), cv2.COLOR_BGR2RGB)
        diag = agent.diagnoser.diagnose(img)
        admitted = {r.upstream for r in agent._gen_candidates(diag, "classify")}
        restored, fb, clip = {}, {}, {}
        restored[""], fb[""], clip[""] = img, False, 0.0
        for up in ALL_UPSTREAM_TOOLS:
            out = agent._get_upstream(up).restore(img)
            restored[up] = out.image
            fb[up] = bool((out.meta or {}).get("fallback", False))
            clip[up] = float(agent._clip_ratio(out.image))
        for s in seeds:
            outcomes = {}
            for up in [""] + ALL_UPSTREAM_TOOLS:
                pred, conf, margin = classify(clfs[s], restored[up])
                outcomes[up] = PipelineOutcome(
                    pipeline_id=f"{up or 'raw'}_cls", upstream=up, downstream="classify",
                    score=conf, downstream_confidence=conf, margin=margin, pred_label=pred,
                    clip_ratio=clip[up], fallback=fb[up],
                    ground_truth_score=1.0 if pred == smp["label"] else 0.0,
                    is_correct=(pred == smp["label"]))
            up, ex = replay(routers[s], diag, outcomes, admitted)
            want = gt[s].get(smp["path"])
            tot += 1
            if want != (up, len(ex)):
                bad += 1
                print(f"  不一致 seed{s} {os.path.basename(smp['path'])}: replay={up}/{len(ex)} real={want}")
    print(f"replay 校验：{tot} 条比较，{bad} 处不一致")
    return 0 if bad == 0 else 1


if __name__ == "__main__":
    raise SystemExit(main())
