"""Pure offline checks for final_v4 sample evidence; never loads inference models.

The ten reconstructible strategies follow evaluate_ablation's recorded protocol.
no_candidate_pruning uses a separate, unsaved inference record and is deliberately
not claimed to be reconstructible from the ordinary samples JSONL.
"""
from __future__ import annotations

import math

BRANCHES = ("", "denoise", "lowlight")
FIXED = {"raw_only": "", "fixed_denoise": "denoise", "fixed_lowlight": "lowlight"}
FUSION = {"uniform", "confidence", "soft_gating"}
RECONSTRUCTIBLE = (*FIXED, "no_raw_candidate", "no_fallback", "agent_learned",
                   "uniform", "confidence", "soft_gating", "global_oracle")


def number(value):
    return isinstance(value, (int, float)) and not isinstance(value, bool) and math.isfinite(value)


def close(left, right, tolerance=1e-6):
    return number(left) and number(right) and abs(left - right) <= tolerance


def vector(value, size):
    return isinstance(value, list) and len(value) == size and all(number(v) for v in value)


def validate_sample_evidence(row, labels):
    """Validate schema and relationships before any metrics use the sample."""
    errors = []
    if not isinstance(row, dict):
        return ["JSONL 行非对象"]
    if (not isinstance(labels, list) or len(labels) != 6
            or any(not isinstance(x, str) or not x for x in labels) or len(set(labels)) != 6):
        return ["JSONL 缺有效 config labels 上下文"]
    gt = row.get("gt")
    if not isinstance(gt, str) or gt not in labels:
        errors.append("JSONL gt 不属于 config labels")
    routed = row.get("routed_upstreams")
    if (not isinstance(routed, list) or not routed
            or any(not isinstance(up, str) or up not in BRANCHES for up in routed)
            or len(set(routed)) != len(routed) or "" not in routed):
        errors.append("JSONL routed_upstreams 非法（须为含 raw 的唯一三分支候选列表）")
        routed = []
    if not isinstance(row.get("diagnosis"), dict) or not row["diagnosis"]:
        errors.append("JSONL diagnosis 非法（须为非空对象）")
    if not vector(row.get("router_probs"), 3):
        errors.append("JSONL router_probs 非法（须为长度 3 的有限数值列表）")
    branches = row.get("branches")
    if not isinstance(branches, dict) or set(branches) != set(BRANCHES):
        return errors + ["JSONL branches 须恰好包含 raw/denoise/lowlight 三分支，禁止额外分支"]
    pids = []
    for upstream in BRANCHES:
        branch = branches[upstream]
        prefix = f"JSONL 分支 {upstream!r}"
        if not isinstance(branch, dict):
            errors.append(f"{prefix} 非对象")
            continue
        pid = branch.get("pipeline_id")
        expected_pid = "raw_cls" if upstream == "" else (
            f"{upstream}_cls" if upstream in routed else f"global_{upstream}_cls")
        if pid != expected_pid:
            errors.append(f"{prefix} pipeline_id={pid!r} 与实际路由候选不一致（期望 {expected_pid}）")
        pids.append(pid)
        pred = branch.get("pred_label")
        if not isinstance(pred, str) or pred not in labels:
            errors.append(f"{prefix} pred_label 非法")
        correct = pred == gt
        if not isinstance(branch.get("is_correct"), bool) or branch["is_correct"] != correct:
            errors.append(f"{prefix} is_correct 非 bool 或 != (pred_label==gt)")
        if not number(branch.get("gt_score")) or branch["gt_score"] != float(correct):
            errors.append(f"{prefix} gt_score 非法或与 pred_label==gt 不一致")
        if not isinstance(branch.get("fallback"), bool):
            errors.append(f"{prefix} fallback 非 bool")
        if branch.get("labels") != labels:
            errors.append(f"{prefix} labels 与 config 类别顺序不一致")
        logits, probs = branch.get("logits"), branch.get("probs")
        valid_logits, valid_probs = vector(logits, len(labels)), vector(probs, len(labels))
        if not valid_logits:
            errors.append(f"{prefix} logits 非法（须为六维有限数值列表）")
        if not valid_probs or any(p < 0 or p > 1 for p in probs) or abs(sum(probs) - 1) > 1e-5:
            errors.append(f"{prefix} probs 非法（须为六维归一化概率）")
            valid_probs = False
        for field in ("confidence", "margin", "clip_ratio"):
            value = branch.get(field)
            if not number(value) or not 0 <= value <= 1:
                errors.append(f"{prefix} {field} 非法（须为 [0,1] 有限数值）")
        if not close(branch.get("score"), branch.get("confidence")):
            errors.append(f"{prefix} score 与 confidence 不一致")
        if valid_probs:
            ordered = sorted(probs, reverse=True)
            if pred != labels[max(range(len(labels)), key=probs.__getitem__)]:
                errors.append(f"{prefix} pred_label 与 probs.argmax 不一致")
            if not close(branch.get("confidence"), ordered[0]):
                errors.append(f"{prefix} confidence 与 probs 不一致")
            if not close(branch.get("margin"), ordered[0] - ordered[1]):
                errors.append(f"{prefix} margin 与 probs 不一致")
            if isinstance(gt, str) and gt in labels and not close(branch.get("gt_prob"), probs[labels.index(gt)]):
                errors.append(f"{prefix} gt_prob 与真实类别概率不一致")
        if valid_logits:
            high = max(logits)
            exps = [math.exp(x - high) for x in logits]
            total = sum(exps)
            if valid_probs and any(abs(p - e / total) > 1e-5 for p, e in zip(probs, exps)):
                errors.append(f"{prefix} probs 与 logits.softmax 不一致")
            if isinstance(gt, str) and gt in labels and not close(branch.get("gt_logit"), logits[labels.index(gt)]):
                errors.append(f"{prefix} gt_logit 与真实类别 logit 不一致")
    pid = row.get("agent_chosen")
    matches = [up for up, branch in branches.items()
               if isinstance(branch, dict) and branch.get("pipeline_id") == pid]
    if len(matches) != 1:
        errors.append(f"JSONL agent_chosen={pid!r} 未唯一匹配分支")
    elif matches[0] not in routed:
        errors.append("JSONL agent_chosen 不在实际候选集内（禁止选补跑 global 分支）")
    if any(not isinstance(pid, str) or not pid for pid in pids) or len(set(pids)) != 3:
        errors.append("JSONL pipeline_id 必须为三个唯一的非空字符串")
    return errors


def _overexposed(branch, clip_max):
    return branch["clip_ratio"] > clip_max + 1e-6


def _choose(row, method, selector):
    branches = row["branches"]
    candidates = {up: branches[up] for up in row["routed_upstreams"]}
    raw = branches[""]
    if method in FIXED:
        return branches[FIXED[method]], False
    if method == "agent_learned":
        return next(b for b in candidates.values() if b["pipeline_id"] == row["agent_chosen"]), False
    if method == "global_oracle":
        corrects = [b for b in branches.values() if b["pred_label"] == row["gt"]]
        return max(corrects or list(branches.values()), key=lambda b: b["gt_score"]), False
    if method in ("no_raw_candidate", "no_fallback"):
        rest = [b for up, b in candidates.items() if up in ("denoise", "lowlight")]
        if not rest:
            return raw, False
        if method == "no_raw_candidate":
            safe = [b for b in rest if not _overexposed(b, selector["clip_max"])]
            return max(safe or rest, key=lambda b: b["confidence"]), False
        for branch in sorted(rest, key=lambda b: b["confidence"], reverse=True):
            safe = not _overexposed(branch, selector["clip_max"])
            if branch["pred_label"] == raw["pred_label"]:
                accept = branch["confidence"] - raw["confidence"] > selector["delta_conf"]
            else:
                accept = (raw["margin"] < selector["tau_uncert"]
                          and branch["margin"] > selector["kappa"] * raw["margin"] + 1e-6)
            if accept and safe:
                return branch, False
        return max(rest, key=lambda b: b["confidence"]), False
    raise ValueError(f"Unknown strategy: {method}")


def _fuse(row, method, selector, tau, labels):
    valid = [(up, row["branches"][up]) for up in row["routed_upstreams"]
             if not row["branches"][up]["fallback"]
             and not _overexposed(row["branches"][up], selector["clip_max"])]
    if not valid:
        return row["branches"][""]["pred_label"], False
    if method == "uniform":
        weights = [1.0] * len(valid)
    elif method == "confidence":
        weights = [b["confidence"] for _, b in valid]
    else:
        # Subtracting max preserves normalized weights and avoids overflow.
        scores = [row["router_probs"][BRANCHES.index(up)] / tau for up, _ in valid]
        high = max(scores)
        weights = [math.exp(value - high) for value in scores]
    if sum(weights) <= 0:
        return row["branches"][""]["pred_label"], False
    field = "logits" if method == "soft_gating" else "probs"
    fused = [sum(w * b[field][i] for (_, b), w in zip(valid, weights)) / sum(weights)
             for i in range(len(labels))]
    return labels[max(range(len(labels)), key=fused.__getitem__)], len(valid) > 1


def recompute_strategy_metrics(rows, *, labels, selector, tau):
    """Reconstruct every numeric evidence metric for ten strategies from validated rows."""
    n = len(rows)
    if not n:
        raise ValueError("JSONL 为空")
    output = {}
    for method in RECONSTRUCTIBLE:
        true, predictions = [], []
        right, wrong, changed, coverage = 0, 0, 0, 0
        restoration, degraded, routed_total, global_regret, routed_regret = 0, 0, 0, 0.0, 0.0
        for row in rows:
            branches, gt = row["branches"], row["gt"]
            raw = branches[""]
            routed = row["routed_upstreams"]
            routed_total += len(routed)
            has_restoration = any(up in ("denoise", "lowlight") for up in routed)
            restoration += has_restoration
            degraded += method in ("no_raw_candidate", "no_fallback") and not has_restoration
            if method in FUSION:
                pred, covered = _fuse(row, method, selector, tau, labels)
                chosen_gt = float(pred == gt)
            else:
                branch, _ = _choose(row, method, selector)
                pred, chosen_gt = branch["pred_label"], branch["gt_score"]
                covered = branch["pipeline_id"] != raw["pipeline_id"]
            true.append(gt)
            predictions.append(pred)
            right += raw["pred_label"] != gt and pred == gt
            wrong += raw["pred_label"] == gt and pred != gt
            changed += raw["pred_label"] != gt and pred != gt and raw["pred_label"] != pred
            coverage += covered
            global_regret += max(b["gt_score"] for b in branches.values()) - chosen_gt
            if method == "agent_learned":
                routed_regret += max(branches[up]["gt_score"] for up in routed) - chosen_gt
        f1s = []
        for label in labels:
            tp = sum(t == label and p == label for t, p in zip(true, predictions))
            fp = sum(t != label and p == label for t, p in zip(true, predictions))
            fn = sum(t == label and p != label for t, p in zip(true, predictions))
            precision = tp / (tp + fp) if tp + fp else 0.0
            recall = tp / (tp + fn) if tp + fn else 0.0
            f1s.append(2 * precision * recall / (precision + recall) if precision + recall else 0.0)
        output[method] = {
            "n": n, "n_has_restoration": restoration, "n_degraded_to_raw": degraded,
            "degraded_to_raw_ratio": round(degraded / n, 4),
            "accuracy": round(sum(t == p for t, p in zip(true, predictions)) / n, 4),
            "macro_f1": round(sum(f1s) / len(labels), 4),
            "raw_to_right": right, "raw_to_wrong": wrong,
            "raw_wrong_still_wrong_changed": changed,
            "negative_gain_rate": round(wrong / n, 4), "net_correct_gain": right - wrong,
            "mean_sample_net_gain": round((right - wrong) / n, 4),
            "coverage": round(coverage / n, 4),
            "avg_routed_candidates": round(routed_total / n, 3),
            "avg_executed_branches": 1.0 if method in FIXED else round(routed_total / n, 3),
            "avg_padded_branches": round(3 - routed_total / n, 3),
            "mean_global_regret": round(global_regret / n, 4),
            "tau": tau if method == "soft_gating" else None,
        }
        if method == "agent_learned":
            output[method]["mean_routed_regret"] = round(routed_regret / n, 4)
    return output
