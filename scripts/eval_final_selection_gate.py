"""受控验证集实验：最终输出门槛（final-selection gate），仅回退 raw。

范围（严格遵守）：
  * 只读 results/iterative_valb_13tool/seed{1,2,3}/per_image.jsonl 及其 run_meta.json；
  * 不重训 router、不调用修复模型、不重训分类器；
  * 不读 test / RealWaste 的预测或标签来选参数；
  * 不修改原始记录、权重、router、正式配置、论文或正式结果表。

机制：在原策略停止时刻，用同一状态下的 learned router 分支正确性评分做终止后处理。
  设 b_old=原最终分支、p_old=其评分、p_raw=raw 评分、τ=门槛。
  若 b_old 是 raw → 保持 raw；
  若 b_old 是修复分支且 p_old - p_raw > τ → 保留 b_old；否则返回 raw。
  不重选其他分支、不增删重排调用、不改变停止时刻、不改变准入/资格规则、不把 τ 加进 call_cost。
  真实标签只用于事后评价。

分支顺序（权威来源：src/agents/orchestrator.py 的 expected_order 公式
  [""] + [up for up in ALL_UPSTREAM_TOOLS if up not in excluded]，13 分支与稿件 B13 一致）。

产物：results/final_selection_gate_valb_20261103/{REPORT.md, metrics.csv, per_record.csv, run_meta.json}
"""
from __future__ import annotations

import argparse
import csv
import hashlib
import json
import os
import sys
from collections import defaultdict
from pathlib import Path

import numpy as np

ROOT = Path(__file__).resolve().parents[1]
VALB_DIR = ROOT / "results" / "iterative_valb_13tool"
OUT_DIR = ROOT / "results" / "final_selection_gate_valb_20261103"
SEEDS = (1, 2, 3)
TAUS = (0.0, 0.005, 0.01, 0.02, 0.03, 0.05)
N_BOOT = 10000
RNG_SEED = 20261103

# 13 分支顺序：raw（""）在前，其余按 TOOL_GROUPS 顺序去掉 4 个 excluded 工具。
BRANCH_ORDER = [
    "", "denoise", "ffdnet", "scunet", "restormer_real", "restormer_blind",
    "restormer_sigma15", "restormer_sigma25", "restormer_sigma50",
    "lowlight", "zero_dce_plus_plus", "sci_lowlight", "retinexformer",
]
BRANCH_IDX = {up: i for i, up in enumerate(BRANCH_ORDER)}


def sha256(path: Path) -> str:
    h = hashlib.sha256()
    with open(path, "rb") as fh:
        for chunk in iter(lambda: fh.read(1 << 20), b""):
            h.update(chunk)
    return h.hexdigest()


def scene_of(path: str) -> str:
    p = path.replace("\\", "/")
    if "valB_lowlight" in p:
        return "lowlight"
    if "valB_noise" in p:
        return "noise"
    raise ValueError(f"未知场景：{path}")


def load_records():
    records = []
    meta_sha = {}
    for s in SEEDS:
        jp = VALB_DIR / f"seed{s}" / "per_image.jsonl"
        mp = VALB_DIR / f"seed{s}" / "run_meta.json"
        meta_sha[s] = {"jsonl": sha256(jp), "run_meta": sha256(mp)}
        for ln in jp.read_text(encoding="utf-8").splitlines():
            if not ln.strip():
                continue
            r = json.loads(ln)
            res = r["result"]
            b_old = res.get("selected_upstream") or ""
            sp = res.get("scheduler_probs")
            if not isinstance(sp, list) or len(sp) != len(BRANCH_ORDER):
                raise RuntimeError(f"评分缺失/维度异常：{r['path']} len={len(sp or [])}")
            if b_old != "" and b_old not in BRANCH_IDX:
                raise RuntimeError(f"selected_upstream 不在分支表：{b_old}")
            records.append({
                "seed": s,
                "scene": scene_of(r["path"]),
                "source": Path(r["path"]).name.rsplit(".", 1)[0],
                "path": r["path"],
                "label": r["label"],
                "b_old": b_old,
                "p_old": float(sp[BRANCH_IDX[b_old]]) if b_old != "" else float(sp[0]),
                "p_raw": float(sp[0]),
                "raw_ok": int(bool(r["raw_ok"])),
                "orig_ok": int(bool(r["chosen_ok"])),
                "restore_calls": int(r["restore_calls"]),
            })
    return records, meta_sha


def verify_baseline(records):
    """逐条复算基线；不匹配即抛错，不凑数。"""
    n = len(records)
    correct = sum(r["orig_ok"] for r in records)
    fix = sum((1 - r["raw_ok"]) * r["orig_ok"] for r in records)
    harm = sum(r["raw_ok"] * (1 - r["orig_ok"]) for r in records)
    per_seed = {}
    for s in SEEDS:
        rs = [r for r in records if r["seed"] == s]
        per_seed[s] = (len(rs), sum(r["orig_ok"] for r in rs),
                       sum((1 - r["raw_ok"]) * r["orig_ok"] for r in rs),
                       sum(r["raw_ok"] * (1 - r["orig_ok"]) for r in rs))
    # 源图 = 164，每个源图低光+噪声各 1，三组模型共 6 条
    sources = {r["source"] for r in records}
    ok = (n == 984 and correct == 779 and fix == 113 and harm == 20
          and per_seed[1] == (328, 257, 37, 11)
          and per_seed[2] == (328, 257, 38, 5)
          and per_seed[3] == (328, 265, 38, 4)
          and len(sources) == 164)
    info = {"n": n, "correct": correct, "fix": fix, "harm": harm,
            "per_seed": per_seed, "n_sources": len(sources)}
    if not ok:
        raise RuntimeError(f"基线复算不一致：{json.dumps(info, ensure_ascii=False)}")
    return info


def gated_output(r, tau):
    """返回 (new_output, new_ok)。new_output 只能是 b_old 或 raw（raw 用空串表示，与 b_old 约定一致）。"""
    if r["b_old"] == "":
        return "", r["orig_ok"]             # 本来就是 raw，不变
    if r["p_old"] - r["p_raw"] > tau:
        return r["b_old"], r["orig_ok"]     # 保留原输出
    return "", r["raw_ok"]                  # 回退 raw


def accuracy(records):
    return sum(r["orig_ok"] for r in records) / len(records)


def paired_bootstrap(records, tau):
    """源图聚类 bootstrap：单位=源图(164)，每个源图带 2 场景×3 模型=6 条记录。"""
    per_src = defaultdict(list)
    for r in records:
        per_src[r["source"]].append(r)
    srcs = sorted(per_src)
    # 每源图在 τ 下相对原策略的正确数变化（配对）
    def src_delta(rs):
        return sum(gated_output(r, tau)[1] - r["orig_ok"] for r in rs)
    point = sum(src_delta(per_src[s]) for s in srcs) / len(records)
    rng = np.random.default_rng(RNG_SEED)
    boots = np.empty(N_BOOT)
    for i in range(N_BOOT):
        idx = rng.choice(len(srcs), size=len(srcs), replace=True)
        tot = sum(src_delta(per_src[srcs[j]]) for j in idx)
        boots[i] = tot / len(records)
    lo, hi = np.percentile(boots, [2.5, 97.5])
    return 100 * point, 100 * lo, 100 * hi


def main() -> int:
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--out", default=str(OUT_DIR))
    args = ap.parse_args()
    out = Path(args.out)
    if out.exists():
        raise SystemExit(f"目标目录已存在，拒绝覆盖：{out}")

    records, meta_sha = load_records()
    base = verify_baseline(records)

    # τ=0 必须逐条复现原始选择
    mismatch0 = [r for r in records if gated_output(r, 0.0)[0] != r["b_old"]]
    if mismatch0:
        raise RuntimeError(f"τ=0 未复现原策略，{len(mismatch0)} 条不一致")

    # 各 τ 指标
    metrics_rows = []
    for tau in TAUS:
        new_ok = [gated_output(r, tau)[1] for r in records]
        acc = sum(new_ok) / len(records)
        per_seed_acc = {s: (sum(gated_output(r, tau)[1] for r in records if r["seed"] == s)
                            / sum(1 for r in records if r["seed"] == s)) for s in SEEDS}
        flipped = [r for r in records if r["b_old"] != "" and gated_output(r, tau)[0] == ""]
        # 挽回有害：原有害( raw_ok=1, orig_ok=0 )且现在正确
        recovered = [r for r in flipped if r["raw_ok"] == 1 and r["orig_ok"] == 0]
        # 损失有效：原有效纠正( raw_ok=0, orig_ok=1 )且现在变错
        lost = [r for r in flipped if r["raw_ok"] == 0 and r["orig_ok"] == 1]
        fixes = sum((1 - r["raw_ok"]) * gated_output(r, tau)[1] for r in records)
        harms = sum(r["raw_ok"] * (1 - gated_output(r, tau)[1]) for r in records)
        metrics_rows.append({
            "tau": tau,
            "acc": acc,
            "acc_seed1": per_seed_acc[1],
            "acc_seed2": per_seed_acc[2],
            "acc_seed3": per_seed_acc[3],
            "diff_pp": 100 * (acc - accuracy(records)),
            "fixes": fixes,
            "harms": harms,
            "flipped": len(flipped),
            "recovered_harms": len(recovered),
            "lost_fixes": len(lost),
            "mean_calls": sum(r["restore_calls"] for r in records) / len(records),
        })

    # 一致性检查 3：新增正确 - 新增错误 = 挽回有害 - 损失有效
    for m in metrics_rows:
        delta_correct = m["acc"] * len(records) - base["correct"]
        rhs = m["recovered_harms"] - m["lost_fixes"]
        if abs(delta_correct - rhs) > 1e-9:
            raise RuntimeError(f"τ={m['tau']} 一致性检查失败：{delta_correct} vs {rhs}")
    # 检查 4：仅回退 raw，最多修复 20 次有害；无损上限 20/984≈2.03pp
    max_gain = max(m["diff_pp"] for m in metrics_rows)
    if max_gain > 100 * (20 / 984) + 1e-9:
        raise RuntimeError(f"增益超过无损上限：{max_gain}")

    # 选择规则：总体准确率最高；并列取更小 τ；无改善则 τ=0
    best = metrics_rows[0]
    improved = False
    for m in metrics_rows:
        if m["acc"] > best["acc"] + 1e-12:
            best = m
            improved = True
        elif abs(m["acc"] - best["acc"]) <= 1e-12 and m["tau"] < best["tau"]:
            best = m
    chosen_tau = best["tau"]
    outcome = "验证集改善" if improved else "未发现验证集改善，保留 τ=0"

    # 固定 τ 的源图聚类 bootstrap（探索性）
    pt, lo, hi = paired_bootstrap(records, chosen_tau)

    out.mkdir(parents=True, exist_ok=False)

    # per_record.csv
    pc_path = out / "per_record.csv"
    with open(pc_path, "w", newline="", encoding="utf-8") as fh:
        w = csv.writer(fh)
        head = ["seed", "scene", "source", "label", "b_old", "p_old", "p_raw",
                "score_diff", "raw_ok", "orig_ok", "restore_calls"]
        for tau in TAUS:
            head += [f"new_output_t{tau}", f"new_ok_t{tau}"]
        w.writerow(head)
        for r in records:
            row = [r["seed"], r["scene"], r["source"], r["label"], r["b_old"],
                   round(r["p_old"], 8), round(r["p_raw"], 8),
                   round(r["p_old"] - r["p_raw"], 8), r["raw_ok"], r["orig_ok"],
                   r["restore_calls"]]
            for tau in TAUS:
                no, nok = gated_output(r, tau)
                row += [no, nok]
            w.writerow(row)

    # metrics.csv
    mc_path = out / "metrics.csv"
    with open(mc_path, "w", newline="", encoding="utf-8") as fh:
        w = csv.writer(fh)
        w.writerow(["tau", "accuracy", "acc_seed1", "acc_seed2", "acc_seed3",
                    "diff_pp_vs_orig", "fixes", "harms", "flipped_to_raw",
                    "recovered_harms", "lost_fixes", "mean_calls"])
        for m in metrics_rows:
            w.writerow([m["tau"], f"{m['acc']:.8f}", f"{m['acc_seed1']:.8f}",
                        f"{m['acc_seed2']:.8f}", f"{m['acc_seed3']:.8f}",
                        f"{m['diff_pp']:.6f}", m["fixes"], m["harms"], m["flipped"],
                        m["recovered_harms"], m["lost_fixes"], f"{m['mean_calls']:.6f}"])

    # run_meta.json
    script_path = Path(__file__)
    run_meta = {
        "experiment": "final_selection_gate_valb",
        "schema": "final-selection-gate-v1",
        "input_files": {str(s): meta_sha[s] for s in SEEDS},
        "branch_order": BRANCH_ORDER,
        "branch_order_source": "src/agents/orchestrator.py expected_order = [''] + [up for up in ALL_UPSTREAM_TOOLS if up not in excluded]; 13-branch (exclude restormer_defocus/mprnet_deblur/realesrgan_x2/realesrgan_x4)",
        "gate_definition": "若 b_old 非 raw 且 p_old - p_raw > tau 保留 b_old，否则回退 raw；不重选分支、不改调用轨迹",
        "score_source": "result.scheduler_probs（停止时刻 learned router 分支正确性评分，非分类器 softmax confidence）",
        "threshold_grid": list(TAUS),
        "selection_rule": "按三组等量记录的总体准确率选最高者；并列取更小 tau；无优于 tau=0 则保留 tau=0",
        "chosen_tau": chosen_tau,
        "selection_outcome": outcome,
        "chosen_tau_bootstrap_diff_pp": {"point": pt, "ci95": [lo, hi]},
        "baseline": base,
        "tau0_reproduced_exact": True,
        "notes": "阈值在同一 valB 上选择；区间与增益是探索性开发结果，不是独立测试证据。调用轨迹全程不变，新输出只能是原输出或 raw。",
        "script_sha256": sha256(script_path),
        "command": "python " + " ".join(sys.argv),
    }
    with open(out / "run_meta.json", "w", encoding="utf-8") as fh:
        json.dump(run_meta, fh, ensure_ascii=False, indent=2)

    # REPORT.md
    L = [
        "# 最终输出门槛验证（valB，探索性）\n",
        f"## 基线复算\nn={base['n']}，正确 {base['correct']}，有效纠正 {base['fix']}，"
        f"有害覆盖 {base['harm']}；分模型正确数 "
        + " / ".join(str(base["per_seed"][s][1]) for s in SEEDS)
        + f"；源图 {base['n_sources']} 张。τ=0 逐条复现原策略（0 条不一致）。\n",
        "## 全部阈值结果\n",
        "| τ | 总体准确率 | seed1 | seed2 | seed3 | Δ vs 原 (pp) | 有效纠正 | 有害覆盖 | 回退数 | 挽回有害 | 损失有效 | 平均调用 |",
        "|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|",
    ]
    for m in metrics_rows:
        L.append(f"| {m['tau']} | {m['acc']:.4f} | {m['acc_seed1']:.4f} | {m['acc_seed2']:.4f} | "
                 f"{m['acc_seed3']:.4f} | {m['diff_pp']:+.3f} | {m['fixes']} | {m['harms']} | "
                 f"{m['flipped']} | {m['recovered_harms']} | {m['lost_fixes']} | {m['mean_calls']:.3f} |")
    L += [
        "",
        f"## 选择结果\n选中 τ = {chosen_tau}（{outcome}）。固定该阈值下相对原策略的配对差值 "
        f"{pt:+.3f} pp（95% 源图聚类 bootstrap CI [{lo:+.3f}, {hi:+.3f}]）。\n",
        "## 结论与局限",
        "- 是否存在净改善：" + ("是（准确率提升，且 bootstrap 区间不跨零）" if improved and lo > 0
                              else ("是（点估计为正，但区间跨零，改善不稳健）" if improved
                                    else "否，未发现验证集净改善")),
        f"- 改善集中度：seed 分模型准确率见上表；",
        f"- 挽回有害覆盖 / 损失有效纠正：见上表各列；",
        "- 门槛是否值得进入后续独立评估：以「选中阈值 + 是否稳健改善」综合判断，见结论段。",
        "- 局限：阈值在同一 valB 上选择，所有增益与区间均为探索性开发结果，非独立测试证据；"
        "调用轨迹全程不变，本实验只改变最终输出在「原输出 vs raw」之间的二选一。",
    ]
    (out / "REPORT.md").write_text("\n".join(L) + "\n", encoding="utf-8")

    print("\n".join(L))
    print(f"\n产出目录：{out}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
