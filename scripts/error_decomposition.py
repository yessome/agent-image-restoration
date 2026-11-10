"""把 agent 的错误按「是否有可用正确分支」分解，并对比 diag_rule。

口径（只使用已保存的逐图记录，不重跑推理）：
  E1 覆盖/库受限：准入集合内不存在正确分支（oracle_admitted_ok=0）
  E2 决策错误：准入集合内存在正确分支，但返回的预测错误（oracle_admitted_ok=1）
     E2a 零调用终止：agent 一次修复都没调，直接返回了错误的 raw，而准入集合里有正确分支
     E2b 已调用但未返回正确结果：至少执行过一次修复，仍返回错误（无法区分「没调用到」与「调用了但没选中」）

同时给出各方法 (mean_calls, accuracy) 的准确率—成本点，以及 agent 相对准入 oracle 的差距。
"""
from __future__ import annotations

import csv
import json
import os
from collections import defaultdict

import numpy as np

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
OUT = os.path.join(ROOT, "results", "verify_paired_baselines")
SEEDS = (1, 2, 3)
SCENES = ("clean", "noise", "lowlight", "motion_blur", "jpeg", "mixed")


def load():
    rows = []
    for seed in SEEDS:
        for f in (os.path.join(ROOT, "results", "scene_matrix_13tool", f"seed{seed}", "per_image.csv"),
                  os.path.join(ROOT, "results", "test_matrix_13tool", f"seed{seed}", "per_image.csv")):
            with open(f, encoding="utf-8") as fh:
                rows.extend(csv.DictReader(fh))
    return rows


def main():
    rows = [r for r in load() if r["method"] in ("iterative", "diag_rule", "legacy_rule", "raw")]
    by_method = defaultdict(list)
    for r in rows:
        by_method[r["method"]].append(r)

    report = {"n": {m: len(v) for m, v in by_method.items()}, "decomposition": {}, "pareto": {},
              "oracle_gap_pp": {}}

    for m, rs in by_method.items():
        n = len(rs)
        wrong = [r for r in rs if int(r["ok"]) == 0]
        e1 = [r for r in wrong if int(r["oracle_admitted_ok"] or 0) == 0]
        e2 = [r for r in wrong if int(r["oracle_admitted_ok"] or 0) == 1]
        e2a = [r for r in e2 if int(r["n_executed"]) == 0]
        e2b = [r for r in e2 if int(r["n_executed"]) > 0]
        adm_oracle = np.mean([int(r["oracle_admitted_ok"] or 0) for r in rs])
        full_oracle = None
        fo = [r["oracle_full_ok"] for r in rs]
        if all(x not in ("", None) for x in fo):
            full_oracle = float(np.mean([int(x) for x in fo]))
        report["decomposition"][m] = {
            "accuracy": float(np.mean([int(r["ok"]) for r in rs])),
            "wrong": len(wrong),
            "E1_no_admitted_correct": len(e1),
            "E2_admitted_correct_missed": len(e2),
            "E2a_zero_call_stop": len(e2a),
            "E2b_called_but_wrong": len(e2b),
            "share_E1_of_wrong": len(e1) / len(wrong) if wrong else None,
            "share_E2_of_wrong": len(e2) / len(wrong) if wrong else None,
            "share_E2a_of_wrong": len(e2a) / len(wrong) if wrong else None,
            "admitted_oracle_acc": float(adm_oracle),
            "full_oracle_acc": full_oracle,
        }
        report["oracle_gap_pp"][m] = 100 * (adm_oracle - float(np.mean([int(r["ok"]) for r in rs])))
        report["pareto"][m] = {
            "accuracy": float(np.mean([int(r["ok"]) for r in rs])),
            "mean_calls": float(np.mean([int(r["restore_calls"]) for r in rs])),
        }

    # 分场景的 E2a 占比（只用 iterative）
    it = by_method["iterative"]
    by_scene = {}
    for s in SCENES:
        rs = [r for r in it if r["subset"] == s]
        wrong = [r for r in rs if int(r["ok"]) == 0]
        e2a = [r for r in wrong if int(r["oracle_admitted_ok"] or 0) == 1 and int(r["n_executed"]) == 0]
        by_scene[s] = {"n": len(rs), "wrong": len(wrong), "E2a": len(e2a),
                       "acc": float(np.mean([int(r["ok"]) for r in rs]))}
    report["iterative_by_scene"] = by_scene

    os.makedirs(OUT, exist_ok=True)
    with open(os.path.join(OUT, "error_decomposition.json"), "w", encoding="utf-8") as fh:
        json.dump(report, fh, ensure_ascii=False, indent=2)

    L = ["# 错误分解与准确率—成本点\n",
         "| 方法 | 准确率 | 平均调用 | 错例 | E1 无可用正确分支 | E2 有正确分支未返回 | E2a 其中零调用终止 | E2b 已调用仍错 | E2 占错例 |",
         "|---|---:|---:|---:|---:|---:|---:|---:|---:|"]
    for m in ("raw", "diag_rule", "legacy_rule", "iterative"):
        d = report["decomposition"][m]
        L.append(f"| {m} | {d['accuracy']:.4f} | {report['pareto'][m]['mean_calls']:.3f} | {d['wrong']} | "
                 f"{d['E1_no_admitted_correct']} | {d['E2_admitted_correct_missed']} | {d['E2a_zero_call_stop']} | "
                 f"{d['E2b_called_but_wrong']} | {100*d['share_E2_of_wrong']:.1f}% |")
    L.append("\n| 方法 | 准入 oracle 准确率 | 全库 oracle 准确率 | agent 距准入 oracle |")
    L.append("|---|---:|---:|---:|")
    for m in ("diag_rule", "legacy_rule", "iterative"):
        d = report["decomposition"][m]
        fo = f"{d['full_oracle_acc']:.4f}" if d["full_oracle_acc"] else "n/a"
        L.append(f"| {m} | {d['admitted_oracle_acc']:.4f} | {fo} | {report['oracle_gap_pp'][m]:.2f} pp |")
    L.append("\n## iterative 分场景\n")
    L.append("| Scene | n | 准确率 | 错例 | E2a 零调用终止 |")
    L.append("|---|---:|---:|---:|---:|")
    for s, v in by_scene.items():
        L.append(f"| {s} | {v['n']} | {v['acc']:.4f} | {v['wrong']} | {v['E2a']} |")
    with open(os.path.join(OUT, "error_decomposition.md"), "w", encoding="utf-8") as fh:
        fh.write("\n".join(L) + "\n")
    print("\n".join(L))


if __name__ == "__main__":
    main()
