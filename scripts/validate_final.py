"""阶段三收口自动校验（final_v4）：17 项全面验证，全程 fail-closed。

验证失败返回非零退出码。用法：
    python scripts/validate_final.py --root results/ablation/final_v4 --require-multilevel

fail-closed 原则：任何"缺失 / 空值 / 哈希为空 / 哈希不匹配 / 路径错误 / 数量不对"都必须失败，
不允许 `None == None` 或静默 continue 导致误通过。

17 项：
  1. 六场景 × 3 seed 结果统一校验（18 组）
  2. 噪声多强度 × 3 seed 结果统一校验（9 组）
  3. 低光多强度 × 3 seed 结果统一校验（12 组）
  4. 六场景 samples JSONL 精确校验（18 份）
  5. 多强度 samples JSONL 精确校验（21 份）
  6. split 泄漏：train/valB/test 缺失或解析为空即失败 + 两两无交集
  7. 六场景 source_id 全量校验（存在/唯一/内容哈希/跨场景一致）
  8. 五列 manifest source_id == 源图内容哈希
  9. cost 文件 fail-closed（18 组）
  10. memory 文件 fail-closed（12 组）
  11. no_candidate_pruning 口径（routed=3/executed=3/padded=0）
  12. 汇总深度比较（build_summary 重算 vs seed_scene_summary.json）
  13. τ 校准文件 fail-closed（best_tau / config_sha256 非空）
  14. 冻结快照资产哈希 + path 一致性（12 项资产）
  15. 冻结快照 eval_manifests 13 项 fail-closed
  16. 快照 best_tau + selector 一致性
  17. 结果 → 快照 → 磁盘 eval_manifest 闭环（39 组）
"""
from __future__ import annotations

import argparse
import hashlib
import json
import math
import os
import sys

# 保证直接运行 `python scripts/validate_final.py` 时，`scripts` 包可被导入
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

SCENES = ["clean", "noise", "lowlight", "motion_blur", "jpeg", "mixed"]
SEEDS = [1, 2, 3]
DEGRADED = ["lowlight", "noise", "motion_blur", "jpeg", "mixed"]
METHODS = ["raw_only", "fixed_lowlight", "fixed_denoise", "no_raw_candidate", "no_fallback",
           "agent_learned", "uniform", "confidence", "soft_gating", "no_candidate_pruning",
           "global_oracle"]
EXPECTED_N = 430

SCENE_MANIFEST = {
    "clean": "data/splits/test.tsv",
    "noise": "data/degraded/noise/manifest_eval.tsv",
    "lowlight": "data/degraded/lowlight/manifest_eval.tsv",
    "motion_blur": "data/degraded/motion_blur/manifest_eval.tsv",
    "jpeg": "data/degraded/jpeg/manifest_eval.tsv",
    "mixed": "data/degraded/mixed/manifest_eval.tsv",
}

# 多强度：目录名 -> (场景前缀, 强度取值列表)
MULTILEVEL = [
    ("noise_multilevel", "noise", ["15", "25", "50"]),
    ("lowlight_multilevel", "lowlight", ["2", "2.5", "3", "3.5"]),
]

# 全部 13 个评估 manifest 名 -> 路径（与 make_snapshot.py 的 EVAL_MANIFESTS 严格一致）
ALL_EVAL_MANIFESTS = {
    "clean": "data/splits/test.tsv",
    "noise": "data/degraded/noise/manifest_eval.tsv",
    "lowlight": "data/degraded/lowlight/manifest_eval.tsv",
    "motion_blur": "data/degraded/motion_blur/manifest_eval.tsv",
    "jpeg": "data/degraded/jpeg/manifest_eval.tsv",
    "mixed": "data/degraded/mixed/manifest_eval.tsv",
    "noise_s15": "data/degraded/noise_s15/manifest_eval.tsv",
    "noise_s25": "data/degraded/noise_s25/manifest_eval.tsv",
    "noise_s50": "data/degraded/noise_s50/manifest_eval.tsv",
    "lowlight_s2": "data/degraded/lowlight_s2/manifest_eval.tsv",
    "lowlight_s2.5": "data/degraded/lowlight_s2.5/manifest_eval.tsv",
    "lowlight_s3": "data/degraded/lowlight_s3/manifest_eval.tsv",
    "lowlight_s3.5": "data/degraded/lowlight_s3.5/manifest_eval.tsv",
}

# 预期资产路径（相对，用 abspath 比较）
EXPECTED_CLASSIFIER = {s: f"weights/trashnet_resnet50_seed{s}.pth" for s in SEEDS}
EXPECTED_CLASSIFIER_META = {s: f"weights/trashnet_resnet50_seed{s}_meta.json" for s in SEEDS}
EXPECTED_ROUTER = {s: f"weights/router_mixed_seed{s}.pth" for s in SEEDS}
EXPECTED_DENOISE = "weights/dncnn_gaussian_color.pth"
EXPECTED_LOWLIGHT = "weights/Zero-DCE_Epoch99.pth"
EXPECTED_CONFIG = "config/pipelines.yaml"
EXPECTED_TEST_MANIFEST = "data/splits/test.tsv"
EXPECTED_ROUTER_MANIFEST = "data/degraded/train_mixed_grouped.tsv"

failures = []


def check(cond: bool, msg: str):
    if cond:
        print(f"  [OK] {msg}")
    else:
        failures.append(msg)
        print(f"  [FAIL] {msg}")


def sha256_file(path):
    if not path or not os.path.exists(path):
        return None
    h = hashlib.sha256()
    with open(path, "rb") as f:
        for chunk in iter(lambda: f.read(1 << 20), b""):
            h.update(chunk)
    return h.hexdigest()


def asset_hash_valid(snap_sha, disk_path):
    """fail-closed 校验：快照 sha 非空、磁盘文件存在、两者一致。任一不满足返回 False。"""
    if not snap_sha:
        return False
    disk_sha = sha256_file(disk_path)
    if not disk_sha:
        return False
    return snap_sha == disk_sha


def eval_manifest_valid(em, expected_manifest):
    """fail-closed 校验：eval_manifest path 存在且与期望一致、sha256 非空且与磁盘一致。"""
    if not isinstance(em, dict):
        return False
    path = em.get("path")
    sha = em.get("sha256")
    if not path or not sha:
        return False
    if os.path.abspath(path) != os.path.abspath(expected_manifest):
        return False
    return sha == sha256_file(path)


def tau_equal(a, b):
    return (_is_number(a) and _is_number(b)
            and math.isfinite(a) and math.isfinite(b)
            and a > 0 and b > 0 and a == b)


def source_id_of(path):
    h = sha256_file(path)
    return h[:16] if h else None


def _normalized_path(path):
    """比较 manifest/元数据路径时统一相对路径、分隔符和 Windows 大小写。"""
    if not isinstance(path, str) or not path.strip():
        return None
    return os.path.normcase(os.path.abspath(path))


def _fmt_label(kind, name, seed):
    return f"{kind}/{name}/seed{seed}"


# 策略对象必填字段
STRATEGY_REQUIRED_FIELDS = [
    "strategy", "n", "n_has_restoration", "n_degraded_to_raw", "degraded_to_raw_ratio",
    "accuracy", "macro_f1", "raw_to_right", "raw_to_wrong", "raw_wrong_still_wrong_changed",
    "negative_gain_rate", "net_correct_gain", "mean_sample_net_gain", "coverage",
    "avg_routed_candidates", "avg_executed_branches", "avg_padded_branches", "mean_global_regret",
]
# [0,1] 范围的浮点指标
STRATEGY_UNIT_FIELDS = ["accuracy", "macro_f1", "negative_gain_rate", "degraded_to_raw_ratio",
                        "coverage", "mean_global_regret"]
# [0,430] 的整数计数指标
STRATEGY_COUNT_FIELDS = ["n", "n_has_restoration", "n_degraded_to_raw", "raw_to_right",
                         "raw_to_wrong", "raw_wrong_still_wrong_changed"]


def _is_number(v):
    """v 是否为合法数值（int/float，排除 bool）。"""
    return isinstance(v, (int, float)) and not isinstance(v, bool)


def _is_int(v):
    """v 是否为合法整数（int，排除 bool）。"""
    return isinstance(v, int) and not isinstance(v, bool)


def validate_strategy(st, method, lab):
    """校验单个策略对象：结构、必填指标类型、有限数值、合法范围、计数与算术一致性。

    任何字段缺失或类型错误都报错（不 continue 跳过）；n 必须严格为整数 430；
    strategy 必须与方法名一致。
    """
    errors = []
    if not isinstance(st, dict) or not st:
        return [f"{lab} 缺 strategy {method}"]
    # strategy 字段必须与方法名一致
    if st.get("strategy") != method:
        errors.append(f"{lab} {method}.strategy={st.get('strategy')!r} != {method}")
    # 必填字段存在
    for f in STRATEGY_REQUIRED_FIELDS:
        if f not in st:
            errors.append(f"{lab} {method} 缺字段 {f}")
    # n 必须严格为整数 430
    n = st.get("n")
    if not _is_int(n) or n != EXPECTED_N:
        errors.append(f"{lab} {method}.n={n!r} != {EXPECTED_N}（须为整数 430）")
    # [0,1] 浮点指标：类型 + 有限 + 范围
    for f in STRATEGY_UNIT_FIELDS:
        if f not in st:
            continue
        v = st[f]
        if not _is_number(v):
            errors.append(f"{lab} {method}.{f}={v!r} 类型非法（须为数值）")
        elif not math.isfinite(float(v)):
            errors.append(f"{lab} {method}.{f}={v!r} 非有限值")
        elif not (0.0 <= float(v) <= 1.0):
            errors.append(f"{lab} {method}.{f}={v!r} 超出 [0,1]")
    # 整数计数指标：必须为整数（拒绝 bool/None/字符串/小数）
    for f in STRATEGY_COUNT_FIELDS:
        if f == "n":
            continue  # n 已单独严格校验
        if f not in st:
            continue
        v = st[f]
        if not _is_int(v):
            errors.append(f"{lab} {method}.{f}={v!r} 类型非法（须为整数）")
        elif not (0 <= v <= EXPECTED_N):
            errors.append(f"{lab} {method}.{f}={v!r} 超出 [0,{EXPECTED_N}]")
    # net_correct_gain：整数（可负，[-430,430]）
    if "net_correct_gain" in st:
        ncg = st["net_correct_gain"]
        if not _is_int(ncg):
            errors.append(f"{lab} {method}.net_correct_gain={ncg!r} 类型非法（须为整数）")
        elif not (-EXPECTED_N <= ncg <= EXPECTED_N):
            errors.append(f"{lab} {method}.net_correct_gain={ncg!r} 超出 [-{EXPECTED_N},{EXPECTED_N}]")
    # mean_sample_net_gain：数值，[-1,1]
    if "mean_sample_net_gain" in st:
        v = st["mean_sample_net_gain"]
        if not _is_number(v):
            errors.append(f"{lab} {method}.mean_sample_net_gain={v!r} 类型非法（须为数值）")
        elif not math.isfinite(float(v)):
            errors.append(f"{lab} {method}.mean_sample_net_gain={v!r} 非有限值")
        elif not (-1.0 <= float(v) <= 1.0):
            errors.append(f"{lab} {method}.mean_sample_net_gain={v!r} 超出 [-1,1]")
    # 分支数与真实三分支执行口径绑定，不能接受任意非负数。
    for f in ["avg_routed_candidates", "avg_executed_branches", "avg_padded_branches"]:
        if f not in st:
            continue
        v = st[f]
        if not _is_number(v):
            errors.append(f"{lab} {method}.{f}={v!r} 类型非法（须为数值）")
        elif not math.isfinite(float(v)):
            errors.append(f"{lab} {method}.{f}={v!r} 非有限值")
        elif not ((0 <= v <= 2) if f == "avg_padded_branches" else (1 <= v <= 3)):
            errors.append(f"{lab} {method}.{f}={v!r} 超出三分支执行范围")
    routed, executed, padded = (st.get(k) for k in
                               ("avg_routed_candidates", "avg_executed_branches", "avg_padded_branches"))
    if all(_is_number(v) and math.isfinite(v) for v in (routed, executed, padded)):
        expected_executed = 1 if method in ("raw_only", "fixed_lowlight", "fixed_denoise") else routed
        if abs(executed - expected_executed) > 1e-6 or abs(routed + padded - 3) > 0.0011:
            errors.append(f"{lab} {method} avg_executed_branches/avg_padded_branches 执行口径不一致")
        if method == "no_candidate_pruning" and (routed, executed, padded) != (3, 3, 0):
            errors.append(f"{lab} no_candidate_pruning 必须为 routed=3/executed=3/padded=0")
    # 算术一致性（仅当所有相关字段类型合法时计算，避免异常）
    if all(_is_int(st.get(k)) for k in ("raw_to_right", "raw_to_wrong", "net_correct_gain")):
        rt = st["raw_to_right"]; rw = st["raw_to_wrong"]; ncg = st["net_correct_gain"]
        if rt - rw != ncg:
            errors.append(f"{lab} {method} net_correct_gain={ncg} != raw_to_right-raw_to_wrong={rt - rw}")
    if _is_int(n) and n > 0:
        if _is_int(st.get("raw_to_wrong")) and _is_number(st.get("negative_gain_rate")):
            rw = st["raw_to_wrong"]
            if abs(round(float(st["negative_gain_rate"]), 4) - round(float(rw) / float(n), 4)) > 1e-6:
                errors.append(f"{lab} {method} negative_gain_rate={st['negative_gain_rate']} != raw_to_wrong/n={round(float(rw)/float(n),4)}")
        if _is_int(st.get("net_correct_gain")) and _is_number(st.get("mean_sample_net_gain")):
            ncg = st["net_correct_gain"]
            if abs(round(float(st["mean_sample_net_gain"]), 4) - round(float(ncg) / float(n), 4)) > 1e-6:
                errors.append(f"{lab} {method} mean_sample_net_gain={st['mean_sample_net_gain']} != net_correct_gain/n={round(float(ncg)/float(n),4)}")
    return errors


def recompute_accuracies_from_jsonl(jl_path):
    """兼容旧 API；统一使用生产 JSONL 证据校验，不再有宽松旁路。"""
    from scripts.validation_evidence import validate_sample_evidence, recompute_strategy_metrics
    ctx = make_context("results/ablation/final_v4")
    rows = []
    errors = []
    try:
        with open(jl_path, encoding="utf-8") as f:
            for line in f:
                if line.strip():
                    row = json.loads(line)
                    rows.append(row)
                    errors.extend(validate_sample_evidence(row, ctx["labels"]))
    except (OSError, UnicodeError, json.JSONDecodeError) as e:
        return None, [f"读 JSONL 失败 {jl_path}：{e}"]
    if not rows:
        return None, [f"JSONL 为空：{jl_path}"]
    if errors:
        return None, errors
    metrics = recompute_strategy_metrics(rows, labels=ctx["labels"], selector=ctx["selector"], tau=ctx["best_tau"])
    return {m: metrics[m]["accuracy"] for m in ("raw_only", "fixed_denoise", "fixed_lowlight", "agent_learned")}, []


def validate_result_record(d, *, scene, seed, expected_manifest, cal_dir, ctx):
    """统一校验单个评估结果 dict，返回错误消息列表（空=通过）。

    逐字段 fail-closed：n/completeness/scene/router/strategies/manifest/eval_manifest/
    config/tau/classifier/router/upstreams/calibration 任一缺失或不匹配都报错。
    """
    errors = []
    rm = d.get("run_meta", {}) if isinstance(d.get("run_meta"), dict) else {}
    lab = _fmt_label("result", scene, seed)

    # 1. n
    if d.get("n") != EXPECTED_N:
        errors.append(f"{lab} n={d.get('n')}（期望 {EXPECTED_N}）")

    # 2. completeness
    comp = d.get("completeness", {})
    if not isinstance(comp, dict):
        errors.append(f"{lab} 缺 completeness")
    else:
        if comp.get("total") != EXPECTED_N:
            errors.append(f"{lab} completeness.total={comp.get('total')}（期望 {EXPECTED_N}）")
        if comp.get("valid") != EXPECTED_N:
            errors.append(f"{lab} completeness.valid={comp.get('valid')}（期望 {EXPECTED_N}）")
        if comp.get("n_missing", -1) != 0:
            errors.append(f"{lab} n_missing={comp.get('n_missing')}（期望 0）")
        if comp.get("n_read_fail", -1) != 0:
            errors.append(f"{lab} n_read_fail={comp.get('n_read_fail')}（期望 0）")
        if comp.get("n_exception", -1) != 0:
            errors.append(f"{lab} n_exception={comp.get('n_exception')}（期望 0）")

    # 3. scene / router 字段
    if d.get("scene") != scene:
        errors.append(f"{lab} scene={d.get('scene')}（期望 {scene}）")
    if d.get("router") != "learned":
        errors.append(f"{lab} router={d.get('router')}（期望 learned）")

    # 4. 所有正式 strategies 存在 + 内容校验（结构/必填指标/有限值/范围/计数关系）
    strats = d.get("strategies", {})
    if not isinstance(strats, dict) or not strats:
        errors.append(f"{lab} 缺 strategies")
    else:
        for m in METHODS:
            if m not in strats:
                errors.append(f"{lab} 缺 strategy {m}")
            else:
                errors.extend(validate_strategy(strats[m], m, lab))

    # 5. 顶层 manifest / manifest_sha256（必须与场景预期一致，且 sha 与磁盘一致）
    mf = d.get("manifest")
    if not mf:
        errors.append(f"{lab} 缺顶层 manifest")
    elif os.path.abspath(mf) != os.path.abspath(expected_manifest):
        errors.append(f"{lab} 顶层 manifest={mf} 非场景预期 {expected_manifest}")
    mf_sha = d.get("manifest_sha256")
    if not mf_sha:
        errors.append(f"{lab} 缺顶层 manifest_sha256")
    elif mf and sha256_file(mf) != mf_sha:
        errors.append(f"{lab} 顶层 manifest_sha256 与磁盘不一致")

    # 6. eval_manifest 溯源（fail-closed）
    em = rm.get("eval_manifest", {})
    if not eval_manifest_valid(em, expected_manifest):
        errors.append(f"{lab} eval_manifest 无效（path={em.get('path') if isinstance(em, dict) else em}）")

    # 7. config
    cfg_path = rm.get("config_path")
    if not cfg_path:
        errors.append(f"{lab} 缺 config_path")
    elif os.path.abspath(cfg_path) != os.path.abspath(ctx["cfg_path"]):
        errors.append(f"{lab} config_path={cfg_path} 非当前 {ctx['cfg_path']}")
    cfg_sha = rm.get("config_sha256")
    if not cfg_sha:
        errors.append(f"{lab} 缺 config_sha256")
    elif cfg_sha != ctx["cfg_sha"]:
        errors.append(f"{lab} config_sha256 与当前 config 不一致")

    # 8. tau == best_tau
    tau = rm.get("tau")
    if tau is None:
        errors.append(f"{lab} 缺 tau")
    elif not tau_equal(tau, ctx["best_tau"]):
        errors.append(f"{lab} tau={tau} != best_tau={ctx['best_tau']}")
    soft_tau = strats.get("soft_gating", {}).get("tau") if isinstance(strats, dict) and isinstance(strats.get("soft_gating"), dict) else None
    if not tau_equal(soft_tau, ctx["best_tau"]):
        errors.append(f"{lab} soft_gating.tau={soft_tau!r} != best_tau={ctx['best_tau']}")
    if not isinstance(ctx.get("selector"), dict) or not ctx["selector"]:
        errors.append(f"{lab} 缺有效 config selector 上下文")
    elif not deep_equal(d.get("selector"), ctx["selector"]):
        errors.append(f"{lab} selector 与 config 不一致")
    if not _is_number(d.get("clip_max")) or not deep_equal(d.get("clip_max"), ctx.get("clip_max")):
        errors.append(f"{lab} clip_max 与 config 不一致")

    # 9. classifier（全路径绑定预期资产 + 哈希匹配预期磁盘资产）
    cls = rm.get("classifier", {})
    if not isinstance(cls, dict) or not cls:
        errors.append(f"{lab} 缺 classifier")
    else:
        cwp = cls.get("weight_path")
        if not cwp:
            errors.append(f"{lab} classifier 缺 weight_path")
        elif os.path.abspath(cwp) != os.path.abspath(EXPECTED_CLASSIFIER[seed]):
            errors.append(f"{lab} classifier weight_path={cwp} 非预期 {EXPECTED_CLASSIFIER[seed]}")
        if not cls.get("sha256"):
            errors.append(f"{lab} classifier 缺 sha256")
        elif cls.get("sha256") != sha256_file(EXPECTED_CLASSIFIER[seed]):
            errors.append(f"{lab} classifier sha256 与预期资产 {EXPECTED_CLASSIFIER[seed]} 不一致")
        cmp = cls.get("meta_path")
        if not cmp:
            errors.append(f"{lab} classifier 缺 meta_path")
        elif os.path.abspath(cmp) != os.path.abspath(EXPECTED_CLASSIFIER_META[seed]):
            errors.append(f"{lab} classifier meta_path={cmp} 非预期 {EXPECTED_CLASSIFIER_META[seed]}")
        if not cls.get("meta_sha256"):
            errors.append(f"{lab} classifier 缺 meta_sha256")
        elif cls.get("meta_sha256") != sha256_file(EXPECTED_CLASSIFIER_META[seed]):
            errors.append(f"{lab} classifier meta_sha256 与预期资产 {EXPECTED_CLASSIFIER_META[seed]} 不一致")
        if cls.get("train_seed") != seed:
            errors.append(f"{lab} classifier train_seed={cls.get('train_seed')}（期望 {seed}）")
        if cls.get("split_seed") != 42:
            errors.append(f"{lab} classifier split_seed={cls.get('split_seed')}（期望 42）")

    # 10. router（全路径绑定预期资产 + 哈希匹配预期磁盘资产 + 训练 manifest 必须存在）
    rtr = rm.get("router", {})
    if not isinstance(rtr, dict) or not rtr:
        errors.append(f"{lab} 缺 router")
    else:
        rw = rtr.get("learned_weight")
        if not rw:
            errors.append(f"{lab} router 缺 learned_weight")
        elif os.path.abspath(rw) != os.path.abspath(EXPECTED_ROUTER[seed]):
            errors.append(f"{lab} router learned_weight={rw} 非预期 {EXPECTED_ROUTER[seed]}")
        if not rtr.get("learned_sha256"):
            errors.append(f"{lab} router 缺 learned_sha256")
        elif rtr.get("learned_sha256") != sha256_file(EXPECTED_ROUTER[seed]):
            errors.append(f"{lab} router learned_sha256 与预期资产 {EXPECTED_ROUTER[seed]} 不一致")
        if rtr.get("train_seed") != seed:
            errors.append(f"{lab} router train_seed={rtr.get('train_seed')}（期望 {seed}）")
        if rtr.get("classifier_sha256") != cls.get("sha256"):
            errors.append(f"{lab} router classifier_sha256 != classifier.sha256")
        # 训练 manifest 溯源（必须存在且哈希正确）
        rmf = rtr.get("manifest")
        if not rmf:
            errors.append(f"{lab} router 缺训练 manifest 路径")
        elif not os.path.exists(rmf):
            errors.append(f"{lab} router 训练 manifest 不存在：{rmf}")
        if _normalized_path(rmf) != _normalized_path(EXPECTED_ROUTER_MANIFEST):
            errors.append(f"{lab} router 训练 manifest 非预期 {EXPECTED_ROUTER_MANIFEST}")
        rmf_sha = rtr.get("manifest_sha256")
        if not rmf_sha:
            errors.append(f"{lab} router 缺 manifest_sha256")
        elif rmf and os.path.exists(rmf) and sha256_file(rmf) != rmf_sha:
            errors.append(f"{lab} router manifest_sha256 与磁盘不一致")
        if not rmf_sha or rmf_sha != sha256_file(EXPECTED_ROUTER_MANIFEST):
            errors.append(f"{lab} router manifest_sha256 与预期训练 manifest 不一致")

    # 11. upstreams（denoise/lowlight，全路径绑定 + 哈希匹配预期磁盘资产）
    ups = rm.get("upstreams", {})
    if not isinstance(ups, dict):
        errors.append(f"{lab} 缺 upstreams")
    else:
        for key, expected in [("denoise", EXPECTED_DENOISE), ("lowlight", EXPECTED_LOWLIGHT)]:
            u = ups.get(key, {})
            if not isinstance(u, dict) or not u:
                errors.append(f"{lab} upstreams 缺 {key}")
                continue
            up = u.get("weight_path")
            if not up:
                errors.append(f"{lab} {key} 缺 weight_path")
            elif os.path.abspath(up) != os.path.abspath(expected):
                errors.append(f"{lab} {key} weight_path={up} 非预期 {expected}")
            if not u.get("sha256"):
                errors.append(f"{lab} {key} 缺 sha256")
            elif u.get("sha256") != sha256_file(expected):
                errors.append(f"{lab} {key} sha256 与预期资产 {expected} 不一致")

    # 12. calibration 文件路径 + 哈希
    for key, fname in [("tau_calibration_file", "tau_calibration.json"),
                       ("threshold_calibration_file", "threshold_calibration.json")]:
        cf = rm.get(key)
        if not cf:
            errors.append(f"{lab} 缺 {key}")
        elif os.path.abspath(cf) != os.path.abspath(os.path.join(cal_dir, fname)):
            errors.append(f"{lab} {key}={cf} 非 {os.path.join(cal_dir, fname)}")
        sha_key = key.replace("_file", "_sha256")
        cs = rm.get(sha_key)
        if not cs:
            errors.append(f"{lab} 缺 {sha_key}")
        elif cf and cs != sha256_file(cf):
            errors.append(f"{lab} {sha_key} 与磁盘不一致")

    return errors


JSONL_TOP_FIELDS = ["path", "gt", "source_id", "source_path", "routed_upstreams",
                    "agent_chosen", "diagnosis", "branches", "router_probs"]
BRANCH_FIELDS = ["pipeline_id", "pred_label", "confidence", "margin", "clip_ratio",
                 "fallback", "score", "gt_score", "is_correct", "labels", "logits", "probs"]


def read_eval_manifest_records(manifest):
    """严格读取两列/五列评估清单，保留每个 source_id 的路径和标签证据。"""
    records, errors = {}, []
    if not os.path.isfile(manifest):
        return records, [f"评估 manifest 缺失：{manifest}"]
    try:
        with open(manifest, encoding="utf-8") as f:
            for lineno, line in enumerate(f, 1):
                line = line.rstrip("\r\n")
                if not line.strip() or line.lstrip().startswith("#"):
                    continue
                parts = line.split("\t")
                lab = f"{manifest}:{lineno}"
                if len(parts) not in (2, 5):
                    errors.append(f"{lab} 评估 manifest 须为两列或五列，实际 {len(parts)} 列")
                    continue
                path, gt = parts[0].strip(), parts[1].strip()
                if not path or not gt or not os.path.isfile(path):
                    errors.append(f"{lab} 评估图片路径无效或标签为空：{path!r}/{gt!r}")
                    continue
                srcp = parts[3].strip() if len(parts) == 5 else os.path.abspath(path)
                if not srcp or not os.path.isfile(srcp):
                    errors.append(f"{lab} source_path 无效：{srcp!r}")
                    continue
                computed = source_id_of(srcp)
                sid = parts[2].strip() if len(parts) == 5 else computed
                if not sid or not computed or sid != computed:
                    errors.append(f"{lab} source_id={sid!r} != 源图内容哈希 {computed!r}")
                    continue
                if sid in records:
                    errors.append(f"{lab} source_id 重复：{sid}")
                    continue
                records[sid] = {"path": path, "gt": gt, "source_path": srcp}
    except (OSError, UnicodeError) as e:
        errors.append(f"读评估 manifest 失败 {manifest}：{e}")
    if not records:
        errors.append(f"评估 manifest 解析为空：{manifest}")
    return records, errors


def validate_jsonl(jl_path, expected_records, result_strategies=None, *, ctx=None):
    """校验 samples JSONL：
    1. 恰好 430 条、source_id 非空唯一、集合与 manifest 精确相等；
    2. 按 source_id 逐条绑定 manifest 的 gt/path/source_path（路径归一化后比较）；
    3. 每条样本必填字段结构完整（path/gt/source_id/source_path/routed_upstreams/agent_chosen/
       diagnosis/branches/router_probs + branches 三分支的预测证据字段）；
    4. 统一校验样本证据语义；从 JSONL 重算十策略全部可重建数值指标。
    """
    from scripts.validation_evidence import validate_sample_evidence, recompute_strategy_metrics
    errors = []
    ctx = ctx if ctx is not None else make_context("results/ablation/final_v4")
    if not isinstance(expected_records, dict) or not expected_records:
        return [f"JSONL 缺有效的 manifest 样本预期：{jl_path}"]
    if len(expected_records) != EXPECTED_N:
        errors.append(f"JSONL manifest 样本预期数 {len(expected_records)} != {EXPECTED_N}：{jl_path}")
    for sid, expected in expected_records.items():
        if (not isinstance(sid, str) or not sid
                or not isinstance(expected, dict)
                or not isinstance(expected.get("gt"), str) or not expected.get("gt")
                or _normalized_path(expected.get("path")) is None
                or _normalized_path(expected.get("source_path")) is None):
            errors.append(f"JSONL manifest 样本预期无效，须含 gt/path/source_path：{sid!r}")
    if errors:
        return errors
    if not os.path.exists(jl_path):
        return [f"缺 JSONL：{jl_path}"]
    sids = set()
    n = 0
    rows = []
    try:
        with open(jl_path, encoding="utf-8") as f:
            for line in f:
                line = line.strip()
                if not line:
                    continue
                try:
                    rec = json.loads(line)
                except json.JSONDecodeError:
                    errors.append(f"JSONL 行非法：{jl_path} -> {line[:60]}")
                    continue
                if not isinstance(rec, dict):
                    errors.append(f"JSONL 行非对象：{jl_path}")
                    continue
                rows.append(rec)
                errors.extend(f"{error}：{jl_path} 行 {len(rows)}" for error in
                              validate_sample_evidence(rec, ctx.get("labels")))
                # 样本结构完整性
                for f in JSONL_TOP_FIELDS:
                    if f not in rec or rec[f] in (None, "", []):
                        errors.append(f"JSONL 行缺字段 {f}：{jl_path}")
                sid = rec.get("source_id")
                if not isinstance(sid, str) or not sid:
                    errors.append(f"JSONL 行缺 source_id 或类型非法：{jl_path}")
                    continue
                if sid in sids:
                    errors.append(f"JSONL source_id 重复：{jl_path} -> {sid}")
                sids.add(sid)
                n += 1
                expected = expected_records.get(sid)
                if expected is not None:
                    if rec.get("gt") != expected["gt"]:
                        errors.append(f"JSONL gt 与 manifest 不一致：{jl_path} source_id={sid}")
                    for field in ("path", "source_path"):
                        if _normalized_path(rec.get(field)) != _normalized_path(expected[field]):
                            errors.append(f"JSONL {field} 与 manifest 不一致：{jl_path} source_id={sid}")
                # branches 三分支证据
                b = rec.get("branches")
                if isinstance(b, dict):
                    for bk in ("", "denoise", "lowlight"):
                        br = b.get(bk)
                        if not isinstance(br, dict):
                            errors.append(f"JSONL 行缺分支 {bk!r}：{jl_path}")
                            continue
                        for bf in BRANCH_FIELDS:
                            if bf not in br:
                                errors.append(f"JSONL 行分支 {bk!r} 缺字段 {bf}：{jl_path}")
                # router_probs 长度
                rp = rec.get("router_probs")
                if isinstance(rp, list) and len(rp) != 3:
                    errors.append(f"JSONL 行 router_probs 长度 {len(rp)} != 3：{jl_path}")
    except OSError as e:
        return [f"读 JSONL 失败 {jl_path}：{e}"]
    if n != EXPECTED_N:
        errors.append(f"JSONL 行数 {n} != {EXPECTED_N}：{jl_path}")
    if sids != set(expected_records.keys()):
        errors.append(f"JSONL source_id 集合与 manifest 不一致：{jl_path}")
    # 先保证全部样本有效，避免用坏证据重算或异常退出；不再只比较 accuracy。
    if result_strategies is not None and not errors:
        recomputed = recompute_strategy_metrics(rows, labels=ctx["labels"],
                                               selector=ctx["selector"], tau=ctx["best_tau"])
        for method, metrics in recomputed.items():
            stored = result_strategies.get(method) if isinstance(result_strategies, dict) else None
            if not isinstance(stored, dict):
                errors.append(f"JSONL 缺策略对象 {method}：{jl_path}")
                continue
            for field, value in metrics.items():
                if field not in stored or not deep_equal(stored[field], value):
                    errors.append(f"JSONL 重算 {method}.{field}={value!r} != 策略对象 {stored.get(field)!r}：{jl_path}")
    return errors


def read_manifest_sids_strict(manifest, *, clean_split=False):
    """fail-closed 版：返回 (sids, errors)。

    拒绝：manifest 缺失、解析为空、列数不足、source_id 为 None/空、源图路径无效
    （内容哈希为 None）、重复 source_id 被静默覆盖，以及**显式 source_id 与源图内容
    哈希不一致**（防止向训练清单追加真实测试图并伪造 ID 绕过 split 防泄漏）。
    """
    errors = []
    if not os.path.exists(manifest):
        return {}, [f"manifest 缺失：{manifest}"]
    sids = {}
    with open(manifest, encoding="utf-8") as f:
        for lineno, line in enumerate(f, 1):
            line = line.rstrip("\n")
            if not line or line.startswith("#"):
                continue
            parts = line.split("\t")
            if len(parts) < 2:
                errors.append(f"{manifest}:{lineno} 列数 {len(parts)} < 2（非法行）")
                continue
            path = parts[0].strip()
            if not path:
                errors.append(f"{manifest}:{lineno} 空路径")
                continue
            actual_sid = source_id_of(path)
            if not actual_sid:
                errors.append(f"{manifest}:{lineno} source_id 为 None（实际输入图片不存在：{path}）")
                continue
            explicit_sid = len(parts) >= 3 and bool(parts[2].strip())
            if explicit_sid:
                sid = parts[2].strip()
                srcp = parts[3].strip() if len(parts) > 3 and parts[3].strip() else path
            else:
                sid = source_id_of(path)
                srcp = os.path.abspath(path)
            if not sid:
                errors.append(f"{manifest}:{lineno} source_id 为 None（源图无效或不存在：{path}）")
                continue
            # 关键防泄漏：显式 source_id 必须等于源图内容哈希（srcp 内容 SHA-256 前 16 位）
            computed = source_id_of(srcp)
            if computed is None:
                errors.append(f"{manifest}:{lineno} 源图不存在或无法计算内容哈希：{srcp}")
                continue
            if sid != computed:
                errors.append(f"{manifest}:{lineno} source_id={sid} != 源图内容哈希 {computed}（{srcp}）")
                continue
            if clean_split and actual_sid != sid:
                errors.append(f"{manifest}:{lineno} clean split 实际输入图片内容哈希 {actual_sid} != source_id {sid}")
                continue
            if sid in sids:
                errors.append(f"{manifest}:{lineno} source_id 重复 {sid}（禁止静默覆盖）")
                continue
            sids[sid] = srcp
    if not sids:
        errors.append(f"manifest 解析为空：{manifest}")
    return sids, errors


def deep_equal(a, b):
    """深度比较，dict 键统一 str 化（JSON 整数键序列化后变字符串）。"""
    if isinstance(a, bool) != isinstance(b, bool):
        return False
    if isinstance(a, (int, float)) and isinstance(b, (int, float)):
        return float(a) == float(b)
    if isinstance(a, dict) and isinstance(b, dict):
        ka = {str(k): v for k, v in a.items()}
        kb = {str(k): v for k, v in b.items()}
        if set(ka) != set(kb):
            return False
        return all(deep_equal(ka[k], kb[k]) for k in ka)
    if isinstance(a, list) and isinstance(b, list):
        return len(a) == len(b) and all(deep_equal(x, y) for x, y in zip(a, b))
    return a == b


def compare_summary(recomputed, stored):
    """深度比较 build_summary 重算结果与磁盘 seed_scene_summary.json，返回错误列表。"""
    errors = []
    for part in ("quality", "cost", "multilevel"):
        if not deep_equal(recomputed.get(part, {}), stored.get(part, {})):
            diffs = _diff_detail(recomputed.get(part, {}), stored.get(part, {}), part)
            errors.extend(diffs[:20] or [f"{part} 汇总不一致"])
            if len(diffs) > 20:
                errors.append(f"... 共 {len(diffs)} 处差异")
    return errors


def _diff_detail(a, b, path=""):
    """返回 a 与 b 的差异描述列表（key 统一 str 化）。"""
    out = []
    if isinstance(a, dict) and isinstance(b, dict):
        ka = {str(k): v for k, v in a.items()}
        kb = {str(k): v for k, v in b.items()}
        for k in sorted(set(ka) | set(kb)):
            if k not in ka:
                out.append(f"{path}/{k} 仅存于磁盘汇总")
            elif k not in kb:
                out.append(f"{path}/{k} 仅存于重算汇总")
            else:
                out.extend(_diff_detail(ka[k], kb[k], f"{path}/{k}"))
    elif isinstance(a, list) and isinstance(b, list):
        if len(a) != len(b):
            out.append(f"{path} 长度 {len(a)} vs {len(b)}")
        else:
            for i, (x, y) in enumerate(zip(a, b)):
                out.extend(_diff_detail(x, y, f"{path}[{i}]"))
    else:
        if not deep_equal(a, b):
            out.append(f"{path} 值不同：{a!r} vs {b!r}")
    return out


def check_multilevel(root, ctx, mdir, prefix, vals, require):
    """对某个多强度目录的 seed×强度 结果做统一校验，直接调用 check()。"""
    multi_ok = True
    n_files = 0
    if not os.path.isdir(os.path.join(root, mdir)):
        if require:
            check(False, f"缺 {mdir} 目录（--require-multilevel 已开启）")
        return
    for v in vals:
        name = f"{prefix}_s{v}"
        exp_manifest = ALL_EVAL_MANIFESTS[name]
        cal_dir = os.path.join(root, mdir)
        for s in SEEDS:
            p = os.path.join(root, mdir, f"seed{s}", f"ablation_{name}_learned.json")
            if not os.path.exists(p):
                multi_ok = False
                check(False, f"缺多强度结果 {p}")
                continue
            n_files += 1
            try:
                d = json.load(open(p, encoding="utf-8"))
            except (OSError, json.JSONDecodeError) as e:
                multi_ok = False
                check(False, f"多强度结果无法解析：{p} -> {e}")
                continue
            errs = validate_result_record(d, scene=name, seed=s,
                                          expected_manifest=exp_manifest, cal_dir=cal_dir, ctx=ctx)
            if errs:
                multi_ok = False
                for e in errs:
                    check(False, e)
            else:
                check(True, f"{name}/seed{s} 结果全字段一致")
    check(multi_ok and n_files == len(vals) * len(SEEDS),
          f"{mdir} 结果共 {n_files}（期望 {len(vals) * len(SEEDS)}）")


def validate_snapshot_eval_manifests(snap):
    """校验快照 eval_manifests 精确 13 项且每项 path/sha 与磁盘一致，返回错误列表。"""
    errors = []
    ems = snap.get("eval_manifests") if isinstance(snap, dict) else None
    if not isinstance(ems, dict):
        return ["快照缺 eval_manifests"]
    expected_names = set(ALL_EVAL_MANIFESTS.keys())
    got_names = set(ems.keys())
    if got_names != expected_names:
        missing = expected_names - got_names
        extra = got_names - expected_names
        if missing:
            errors.append(f"快照 eval_manifests 缺项：{sorted(missing)}")
        if extra:
            errors.append(f"快照 eval_manifests 多项：{sorted(extra)}")
    for name, exp_mp in ALL_EVAL_MANIFESTS.items():
        entry = ems.get(name)
        if not isinstance(entry, dict):
            errors.append(f"快照 eval_manifests[{name}] 非 dict")
            continue
        p = entry.get("path")
        sha = entry.get("sha256")
        if not p:
            errors.append(f"快照 eval_manifests[{name}] 缺 path")
        elif os.path.abspath(p) != os.path.abspath(exp_mp):
            errors.append(f"快照 eval_manifests[{name}] path={p} 非预期 {exp_mp}")
        if not sha:
            errors.append(f"快照 eval_manifests[{name}] 缺 sha256")
        elif sha != sha256_file(exp_mp):
            errors.append(f"快照 eval_manifests[{name}] sha256 与磁盘不一致")
    return errors


def validate_cost_file(cd, scene, seed, *, ctx):
    """校验 cost 文件：顶层 scene/router、run_meta.seed/config/manifest/tau、上游修复模型哈希、
    classifier/router_learned 绑定预期 seed 资产，以及 5 个测量对象的字段类型/范围/算术关系。"""
    errors = []
    lab = f"cost {scene}/seed{seed}"
    if not isinstance(cd, dict):
        return [f"{lab} 文件内容须为对象"]
    crm = cd.get("run_meta")
    if not isinstance(crm, dict):
        return [f"{lab} 缺 run_meta 对象"]
    if not isinstance(ctx, dict):
        return [f"{lab} 缺配置/校准上下文"]
    # 顶层 scene / router
    if cd.get("scene") != scene:
        errors.append(f"{lab} scene={cd.get('scene')!r} != {scene}")
    if cd.get("router") != "learned":
        errors.append(f"{lab} router={cd.get('router')!r} != learned")
    if crm.get("router") != "learned":
        errors.append(f"{lab} run_meta.router={crm.get('router')!r} != learned")
    # run_meta.seed 必须 == 预期 seed
    if not _is_int(crm.get("seed")) or crm.get("seed") != seed:
        errors.append(f"{lab} run_meta.seed={crm.get('seed')!r} != {seed}")
    # 配置和温度绑定当前 --root 的校准上下文，不能只信任 cost 自报的路径/哈希。
    cfg_path, cfg_sha = ctx.get("cfg_path"), ctx.get("cfg_sha")
    if (_normalized_path(cfg_path) != _normalized_path(EXPECTED_CONFIG)
            or not asset_hash_valid(cfg_sha, EXPECTED_CONFIG)):
        errors.append(f"{lab} 配置上下文无效或与当前 config 不一致")
    if (_normalized_path(crm.get("config_path")) is None
            or _normalized_path(crm.get("config_path")) != _normalized_path(cfg_path)):
        errors.append(f"{lab} config_path={crm.get('config_path')!r} 非当前 {cfg_path}")
    if not crm.get("config_sha256") or crm.get("config_sha256") != cfg_sha:
        errors.append(f"{lab} config_sha256 与当前 config 不一致")
    expected_manifest = SCENE_MANIFEST.get(scene)
    if (not expected_manifest
            or _normalized_path(crm.get("manifest")) != _normalized_path(expected_manifest)):
        errors.append(f"{lab} manifest={crm.get('manifest')!r} 非场景预期 {expected_manifest}")
    if not asset_hash_valid(crm.get("manifest_sha256"), expected_manifest):
        errors.append(f"{lab} manifest_sha256 与场景预期 manifest 不一致")
    best_tau, tau = ctx.get("best_tau"), crm.get("tau")
    if (not _is_number(best_tau) or not math.isfinite(float(best_tau)) or best_tau <= 0
            or not ctx.get("tau_cfg_sha") or ctx.get("tau_cfg_sha") != cfg_sha):
        errors.append(f"{lab} tau 校准上下文无效")
    if (not _is_number(tau) or not math.isfinite(float(tau)) or tau <= 0
            or not tau_equal(tau, best_tau)):
        errors.append(f"{lab} tau={tau!r} != best_tau={best_tau!r}（须为有限正数）")
    # 上游修复模型哈希必须匹配当前预期资产（拒绝旧 DnCNN 哈希）
    if not asset_hash_valid(crm.get("upstream_denoise_sha256"), EXPECTED_DENOISE):
        errors.append(f"{lab} upstream_denoise_sha256={crm.get('upstream_denoise_sha256')!r} 非当前 {EXPECTED_DENOISE} 哈希")
    if not asset_hash_valid(crm.get("upstream_lowlight_sha256"), EXPECTED_LOWLIGHT):
        errors.append(f"{lab} upstream_lowlight_sha256={crm.get('upstream_lowlight_sha256')!r} 非当前 {EXPECTED_LOWLIGHT} 哈希")
    cls = crm.get("classifier", {})
    rtr = crm.get("router_learned", {})
    if not isinstance(cls, dict) or not cls:
        errors.append(f"{lab} 缺 classifier")
        cls = {}
    else:
        cwp = cls.get("weight_path")
        if not cwp:
            errors.append(f"{lab} classifier 缺 weight_path")
        elif os.path.abspath(cwp) != os.path.abspath(EXPECTED_CLASSIFIER[seed]):
            errors.append(f"{lab} classifier weight_path={cwp} 非预期 {EXPECTED_CLASSIFIER[seed]}")
        if not cls.get("sha256"):
            errors.append(f"{lab} classifier 缺 sha256")
        elif cls.get("sha256") != sha256_file(EXPECTED_CLASSIFIER[seed]):
            errors.append(f"{lab} classifier sha256 与预期资产 {EXPECTED_CLASSIFIER[seed]} 不一致")
        if cls.get("train_seed") is not None and cls.get("train_seed") != seed:
            errors.append(f"{lab} classifier train_seed={cls.get('train_seed')} != {seed}")
    if not isinstance(rtr, dict) or not rtr:
        errors.append(f"{lab} 缺 router_learned")
    else:
        rw = rtr.get("weight_path")
        if not rw:
            errors.append(f"{lab} router 缺 weight_path")
        elif os.path.abspath(rw) != os.path.abspath(EXPECTED_ROUTER[seed]):
            errors.append(f"{lab} router weight_path={rw} 非预期 {EXPECTED_ROUTER[seed]}")
        if not rtr.get("sha256"):
            errors.append(f"{lab} router 缺 sha256")
        elif rtr.get("sha256") != sha256_file(EXPECTED_ROUTER[seed]):
            errors.append(f"{lab} router sha256 与预期资产 {EXPECTED_ROUTER[seed]} 不一致")
        if rtr.get("classifier_sha256") != cls.get("sha256"):
            errors.append(f"{lab} router.classifier_sha256 != classifier.sha256")
    # 5 个测量对象必须齐全
    strats = cd.get("strategies")
    if not isinstance(strats, dict):
        return errors + [f"{lab} strategies 缺测量对象（须为对象）"]
    timing_fields = {
        "raw_only": ("fixed_branch_e2e_ms", "branch_latency_ms"),
        "fixed_denoise": ("fixed_branch_e2e_ms", "branch_latency_ms"),
        "fixed_lowlight": ("fixed_branch_e2e_ms", "branch_latency_ms"),
        "candidate_search": ("candidate_search_e2e_ms", "branch_latency_ms"),
        "soft_gating_fusion_overhead": ("soft_gating_e2e_ms", "fusion_overhead_ms"),
    }
    # 每个测量对象的字段类型/范围/算术关系（按策略类型区分字段集）
    COUNT_STRATS = ("raw_only", "fixed_denoise", "fixed_lowlight", "candidate_search")
    for sk, required_timings in timing_fields.items():
        st = strats.get(sk)
        if not isinstance(st, dict):
            errors.append(f"{lab} strategies 缺测量对象 {sk}")
            continue
        if sk in COUNT_STRATS:
            # 样本数/尝试数：整数 == 430
            for f in ("n_samples", "n_attempted"):
                v = st.get(f)
                if not _is_int(v):
                    errors.append(f"{lab} {sk}.{f}={v!r} 类型非法（须为整数）")
                elif v != EXPECTED_N:
                    errors.append(f"{lab} {sk}.{f}={v!r} != {EXPECTED_N}")
            # 错误数/回退数：整数 [0,430]
            for f in ("n_errors", "n_fallback"):
                v = st.get(f)
                if not _is_int(v):
                    errors.append(f"{lab} {sk}.{f}={v!r} 类型非法（须为整数）")
                elif not (0 <= v <= EXPECTED_N):
                    errors.append(f"{lab} {sk}.{f}={v!r} 超出 [0,{EXPECTED_N}]")
            # 比率字段：[0,1] 有限
            for f in ("error_rate", "fallback_ratio"):
                v = st.get(f)
                if not _is_number(v):
                    errors.append(f"{lab} {sk}.{f}={v!r} 类型非法（须为数值）")
                elif not math.isfinite(float(v)) or not (0.0 <= float(v) <= 1.0):
                    errors.append(f"{lab} {sk}.{f}={v!r} 超出 [0,1]")
            if sk == "candidate_search":
                v = st.get("router_trigger_rate")
                if not _is_number(v) or not math.isfinite(float(v)) or not (0.0 <= float(v) <= 1.0):
                    errors.append(f"{lab} candidate_search.router_trigger_rate={v!r} 非法")
            # 算术关系
            if _is_int(st.get("n_attempted")) and st.get("n_attempted") > 0 and _is_int(st.get("n_errors")) and _is_number(st.get("error_rate")):
                if abs(round(float(st["error_rate"]), 4) - round(st["n_errors"] / st["n_attempted"], 4)) > 1e-6:
                    errors.append(f"{lab} {sk}.error_rate={st['error_rate']} != n_errors/n_attempted={round(st['n_errors']/st['n_attempted'],4)}")
            if _is_int(st.get("n_samples")) and st.get("n_samples") > 0 and _is_int(st.get("n_fallback")) and _is_number(st.get("fallback_ratio")):
                if abs(round(float(st["fallback_ratio"]), 4) - round(st["n_fallback"] / st["n_samples"], 4)) > 1e-6:
                    errors.append(f"{lab} {sk}.fallback_ratio={st['fallback_ratio']} != n_fallback/n_samples={round(st['n_fallback']/st['n_samples'],4)}")
            if all(_is_int(st.get(f)) for f in ("n_samples", "n_errors", "n_attempted")):
                if st["n_samples"] + st["n_errors"] != st["n_attempted"]:
                    errors.append(f"{lab} {sk}.n_samples + n_errors != n_attempted")
        elif sk == "soft_gating_fusion_overhead":
            for f in ("multi_candidate_rate", "fusion_error_rate", "router_score_missing_rate"):
                v = st.get(f)
                if not _is_number(v) or not math.isfinite(float(v)) or not (0.0 <= float(v) <= 1.0):
                    errors.append(f"{lab} soft_gating_fusion_overhead.{f}={v!r} 非法")
        # 耗时字段（dict 含 mean_ms/median_ms/p95_ms，有限非负）
        for f in required_timings:
            timing = st.get(f)
            if not isinstance(timing, dict):
                errors.append(f"{lab} 缺计时测量对象 {sk}.{f}（须为对象）")
                continue
            for sub in ("mean_ms", "median_ms", "p95_ms"):
                v = timing.get(sub)
                if not _is_number(v) or not math.isfinite(float(v)) or float(v) < 0:
                    errors.append(f"{lab} {sk}.{f}.{sub}={v!r} 非法")
        # executed_branches / peak_memory_mb：有限非负
        for f in ("executed_branches", "peak_memory_mb"):
            if sk in COUNT_STRATS:
                v = st.get(f)
                if not _is_number(v) or not math.isfinite(float(v)) or float(v) < 0:
                    errors.append(f"{lab} {sk}.{f}={v!r} 非法")
    return errors


def validate_memory_file(md, strat, seed):
    """校验 memory 文件 models_loaded 绑定预期 seed 资产 + 测量字段（memory_allocated_mb/peak_memory_mb）。"""
    errors = []
    lab = f"memory {strat}/seed{seed}"
    # 各策略加载的模型集合（精确匹配，不多不少）
    expected_models = {
        "raw_only": ["classifier"],
        "fixed_denoise": ["classifier", "dncnn"],
        "fixed_lowlight": ["classifier", "lowlight"],
        "candidate_search": ["classifier", "dncnn", "lowlight", "router"],
    }.get(strat, [])
    # 顶层策略/seed 绑定
    if md.get("strategy") != strat:
        errors.append(f"{lab} strategy={md.get('strategy')} != {strat}")
    if md.get("seed") != seed:
        errors.append(f"{lab} seed={md.get('seed')} != {seed}")
    # 测量字段：必须存在、有限、非负
    for f in ["memory_allocated_mb", "peak_memory_mb"]:
        v = md.get(f)
        if isinstance(v, bool) or not isinstance(v, (int, float)):
            errors.append(f"{lab} 缺测量字段 {f}")
        elif not math.isfinite(float(v)) or float(v) < 0:
            errors.append(f"{lab} {f}={v} 非法（须为有限非负数）")
    ml = md.get("models_loaded", {}) if isinstance(md, dict) else {}
    # 模型集合精确匹配
    if expected_models:
        got = set(ml.keys())
        exp = set(expected_models)
        if got != exp:
            if exp - got:
                errors.append(f"{lab} models_loaded 缺模型：{sorted(exp - got)}")
            if got - exp:
                errors.append(f"{lab} models_loaded 多模型：{sorted(got - exp)}")
    # classifier 绑定
    cls = ml.get("classifier", {})
    if not isinstance(cls, dict) or not cls:
        errors.append(f"{lab} 缺 classifier")
    else:
        cwp = cls.get("weight_path")
        if not cwp:
            errors.append(f"{lab} classifier 缺 weight_path")
        elif os.path.abspath(cwp) != os.path.abspath(EXPECTED_CLASSIFIER[seed]):
            errors.append(f"{lab} classifier weight_path={cwp} 非预期 {EXPECTED_CLASSIFIER[seed]}")
        if not cls.get("sha256"):
            errors.append(f"{lab} classifier 缺 sha256")
        elif cls.get("sha256") != sha256_file(EXPECTED_CLASSIFIER[seed]):
            errors.append(f"{lab} classifier sha256 与预期资产 {EXPECTED_CLASSIFIER[seed]} 不一致")
    # dncnn / lowlight（按策略决定是否应加载）
    for key, expected in [("dncnn", EXPECTED_DENOISE), ("lowlight", EXPECTED_LOWLIGHT)]:
        should_load = key in expected_models
        u = ml.get(key, {})
        if not should_load:
            continue
        if not isinstance(u, dict) or not u:
            errors.append(f"{lab} models_loaded 缺 {key}")
            continue
        up = u.get("weight_path")
        if not up:
            errors.append(f"{lab} {key} 缺 weight_path")
        elif os.path.abspath(up) != os.path.abspath(expected):
            errors.append(f"{lab} {key} weight_path={up} 非预期 {expected}")
        if not u.get("sha256"):
            errors.append(f"{lab} {key} 缺 sha256")
        elif u.get("sha256") != sha256_file(expected):
            errors.append(f"{lab} {key} sha256 与预期资产 {expected} 不一致")
    # candidate_search 额外加载 router
    if "router" in expected_models:
        rtr = ml.get("router", {})
        if not isinstance(rtr, dict) or not rtr:
            errors.append(f"{lab} 缺 router")
        else:
            rw = rtr.get("weight_path")
            if not rw:
                errors.append(f"{lab} router 缺 weight_path")
            elif os.path.abspath(rw) != os.path.abspath(EXPECTED_ROUTER[seed]):
                errors.append(f"{lab} router weight_path={rw} 非预期 {EXPECTED_ROUTER[seed]}")
            if not rtr.get("sha256"):
                errors.append(f"{lab} router 缺 sha256")
            elif rtr.get("sha256") != sha256_file(EXPECTED_ROUTER[seed]):
                errors.append(f"{lab} router sha256 与预期资产 {EXPECTED_ROUTER[seed]} 不一致")
            if rtr.get("classifier_sha256") != cls.get("sha256"):
                errors.append(f"{lab} router.classifier_sha256 != classifier.sha256")
    return errors


def make_context(root):
    """构建校验上下文：加载校准文件 best_tau/config_sha256 与当前 config 哈希。"""
    ctx = {
        "root": root,
        "cfg_path": EXPECTED_CONFIG,
        "cfg_sha": sha256_file(EXPECTED_CONFIG),
        "best_tau": None,
        "tau_cfg_sha": None,
    }
    import yaml
    with open(EXPECTED_CONFIG, encoding="utf-8") as f:
        cfg = yaml.safe_load(f)
    ctx["selector"] = cfg.get("selector")
    ctx["clip_max"] = cfg.get("selector", {}).get("clip_max")
    ctx["labels"] = cfg.get("downstreams", {}).get("classify", {}).get("labels")
    tau_json = os.path.join(root, "tau_calibration.json")
    if os.path.exists(tau_json):
        td = json.load(open(tau_json, encoding="utf-8"))
        ctx["best_tau"] = td.get("best_tau")
        ctx["tau_cfg_sha"] = td.get("config_sha256")
    return ctx


def _iter_result_files(root):
    """迭代 (kind, name, seed, 文件路径, expected_manifest, cal_dir)。

    kind ∈ {"scene", "noise", "lowlight"}；name 为场景/强度名。
    """
    for scene in SCENES:
        for s in SEEDS:
            p = os.path.join(root, f"seed{s}", f"ablation_{scene}_learned.json")
            yield ("scene", scene, s, p, SCENE_MANIFEST[scene], root)
    for mdir, prefix, vals in MULTILEVEL:
        for v in vals:
            name = f"{prefix}_s{v}"
            for s in SEEDS:
                p = os.path.join(root, mdir, f"seed{s}", f"ablation_{name}_learned.json")
                yield (prefix, name, s, p, ALL_EVAL_MANIFESTS[name], os.path.join(root, mdir))


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--root", default="results/ablation/final_v4")
    ap.add_argument("--require-multilevel", action="store_true",
                    help="要求 noise_multilevel / lowlight_multilevel 多强度结果存在（final_v4 等正式版应显式传入）")
    args = ap.parse_args()
    root = args.root
    ctx = make_context(root)

    # 预收集：多强度目录是否存在
    multi_present = {}
    for mdir, prefix, vals in MULTILEVEL:
        multi_present[mdir] = os.path.isdir(os.path.join(root, mdir))

    # ===== 1. 六场景 × 3 seed 结果统一校验 =====
    print("=== 1. 六场景 × 3 seed 结果统一校验（18 组） ===")
    n_scene = 0
    for kind, name, seed, p, exp_manifest, cal_dir in _iter_result_files(root):
        if kind != "scene":
            continue
        if not os.path.exists(p):
            check(False, f"缺结果：{p}")
            continue
        n_scene += 1
        try:
            d = json.load(open(p, encoding="utf-8"))
        except (OSError, json.JSONDecodeError) as e:
            check(False, f"结果无法解析：{p} -> {e}")
            continue
        errs = validate_result_record(d, scene=name, seed=seed,
                                      expected_manifest=exp_manifest, cal_dir=cal_dir, ctx=ctx)
        if errs:
            for e in errs:
                check(False, e)
        else:
            check(True, f"{name}/seed{seed} 结果全字段一致")
    check(n_scene == len(SCENES) * len(SEEDS), f"六场景结果共 {n_scene}（期望 18）")

    # ===== 2. 噪声多强度 × 3 seed =====
    print("\n=== 2. 噪声多强度 × 3 seed 结果统一校验（9 组） ===")
    check_multilevel(root, ctx, "noise_multilevel", "noise", ["15", "25", "50"], args.require_multilevel)

    # ===== 3. 低光多强度 × 3 seed =====
    print("\n=== 3. 低光多强度 × 3 seed 结果统一校验（12 组） ===")
    check_multilevel(root, ctx, "lowlight_multilevel", "lowlight", ["2", "2.5", "3", "3.5"], args.require_multilevel)

    # ===== 4. 六场景 samples JSONL =====
    print("\n=== 4. 六场景 samples JSONL 精确校验（18 份） ===")
    scene_records = {}
    for scene in SCENES:
        records, errs = read_eval_manifest_records(SCENE_MANIFEST[scene])
        scene_records[scene] = records
        for e in errs:
            check(False, e)
    for scene in SCENES:
        for s in SEEDS:
            jl = os.path.join(root, f"seed{s}", f"samples_{scene}_learned.jsonl")
            res_p = os.path.join(root, f"seed{s}", f"ablation_{scene}_learned.json")
            strategies = None
            if os.path.exists(res_p):
                strategies = json.load(open(res_p, encoding="utf-8")).get("strategies")
            errs = validate_jsonl(jl, scene_records[scene], strategies, ctx=ctx)
            if errs:
                for e in errs:
                    check(False, e)
            else:
                check(True, f"{scene}/seed{s} JSONL 430 条且 source_id/gt/path/source_path 与 manifest 一致 + 证据重算一致")

    # ===== 5. 多强度 samples JSONL =====
    print("\n=== 5. 多强度 samples JSONL 精确校验（21 份） ===")
    for mdir, prefix, vals in MULTILEVEL:
        if not multi_present[mdir]:
            if args.require_multilevel:
                check(False, f"缺 {mdir} 目录（--require-multilevel 已开启）")
            continue
        for v in vals:
            name = f"{prefix}_s{v}"
            records, errs = read_eval_manifest_records(ALL_EVAL_MANIFESTS[name])
            for e in errs:
                check(False, e)
            for s in SEEDS:
                jl = os.path.join(root, mdir, f"seed{s}", f"samples_{name}_learned.jsonl")
                res_p = os.path.join(root, mdir, f"seed{s}", f"ablation_{name}_learned.json")
                strategies = None
                if os.path.exists(res_p):
                    strategies = json.load(open(res_p, encoding="utf-8")).get("strategies")
                errs = validate_jsonl(jl, records, strategies, ctx=ctx)
                if errs:
                    for e in errs:
                        check(False, e)
                else:
                    check(True, f"{name}/seed{s} JSONL 430 条且 source_id/gt/path/source_path 与 manifest 一致 + 证据重算一致")

    # ===== 6. split 泄漏（缺失/空即失败 + 两两无交集） =====
    print("\n=== 6. split 泄漏：train/valB/test 缺失或空即失败 + 两两无交集 ===")
    train_sids, terr = read_manifest_sids_strict("data/splits/train.tsv", clean_split=True)
    valb_sids, verr = read_manifest_sids_strict("data/splits/valB.tsv", clean_split=True)
    test_sids, xerr = read_manifest_sids_strict("data/splits/test.tsv", clean_split=True)
    for e in terr + verr + xerr:
        check(False, e)
    # 结构化失败：sids 始终为 dict（缺失时为空 dict），直接 keys() 取集合
    train_set = set(train_sids.keys())
    valb_set = set(valb_sids.keys())
    test_set = set(test_sids.keys())
    check(len(train_set) > 0, f"train 解析出 {len(train_set)} 个 source_id（>0）")
    check(len(valb_set) > 0, f"valB 解析出 {len(valb_set)} 个 source_id（>0）")
    check(len(test_set) > 0, f"test 解析出 {len(test_set)} 个 source_id（>0）")
    check(not (train_set & test_set), f"train 与 test source_id 无交集（交集 {len(train_set & test_set)}）")
    check(not (valb_set & test_set), f"valB 与 test source_id 无交集（交集 {len(valb_set & test_set)}）")
    check(not (train_set & valb_set), f"train 与 valB source_id 无交集（交集 {len(train_set & valb_set)}）")

    # ===== 7. 六场景 source_id 全量校验 =====
    print("\n=== 7. 六场景 source_id 全量校验 ===")
    for scene in SCENES:
        sids = scene_records[scene]
        check(len(sids) == EXPECTED_N, f"{scene} source_id 数={len(sids)}（期望 {EXPECTED_N}）")
        check(len(set(sids.keys())) == EXPECTED_N, f"{scene} source_id 唯一")
    base = set(scene_records["lowlight"].keys())
    for scene in ["noise", "motion_blur", "jpeg", "mixed", "clean"]:
        check(set(scene_records[scene].keys()) == base, f"{scene} 与 lowlight source_id 集合一致")

    # ===== 8. 五列 manifest source_id == 源图内容哈希 =====
    print("\n=== 8. 五列 manifest source_id == 源图内容哈希 ===")
    five_col = []
    for scene in DEGRADED:
        five_col.append(f"data/degraded/{scene}/manifest_eval.tsv")
        full = f"data/degraded/{scene}/manifest_full.tsv"
        if os.path.exists(full):
            five_col.append(full)
    five_col += ["data/degraded/valB_degraded.tsv",
                 "data/degraded/valB_lowlight.tsv",
                 "data/degraded/valB_noise.tsv"]
    sid_ok = True
    n_sid = 0
    for mf in five_col:
        if not os.path.exists(mf):
            sid_ok = False
            check(False, f"缺五列清单 {mf}")
            continue
        with open(mf, encoding="utf-8") as f:
            for line in f:
                line = line.rstrip("\n")
                if not line or line.startswith("#"):
                    continue
                parts = line.split("\t")
                if len(parts) < 5:
                    sid_ok = False
                    check(False, f"{mf}: 列数 {len(parts)} < 5（期望五列），行：{line[:60]}")
                    continue
                sid, srcp = parts[2], parts[3]
                n_sid += 1
                if sid != source_id_of(srcp):
                    sid_ok = False
                    check(False, f"{mf}: source_id {sid} != 内容哈希 {source_id_of(srcp)}（{srcp}）")
                    break
        if not sid_ok:
            break
    if sid_ok:
        check(True, f"五列 manifest source_id 全部等于源图内容哈希（{n_sid} 行）")

    # ===== 9. cost 文件 fail-closed =====
    print("\n=== 9. cost 文件 fail-closed（18 组） ===")
    cost_ok = True
    for scene in SCENES:
        for s in SEEDS:
            cp = os.path.join(root, f"cost_{scene}_seed{s}.json")
            if not os.path.exists(cp):
                cost_ok = False
                check(False, f"缺成本文件 {cp}")
                continue
            cd = json.load(open(cp, encoding="utf-8"))
            errs = validate_cost_file(cd, scene, s, ctx=ctx)
            if errs:
                cost_ok = False
                for e in errs:
                    check(False, e)
            else:
                check(True, f"{scene}/seed{s} cost fail-closed 通过")
    if cost_ok:
        check(True, "六场景×3seed 成本文件齐全且 fail-closed 通过")

    # ===== 10. memory 文件 fail-closed =====
    print("\n=== 10. memory 文件 fail-closed（12 组） ===")
    mem_ok = True
    for strat in ["raw_only", "fixed_denoise", "fixed_lowlight", "candidate_search"]:
        for s in SEEDS:
            mp = os.path.join(root, f"memory_{strat}_seed{s}.json")
            if not os.path.exists(mp):
                mem_ok = False
                check(False, f"缺显存文件 {mp}")
                continue
            md = json.load(open(mp, encoding="utf-8"))
            errs = validate_memory_file(md, strat, s)
            if errs:
                mem_ok = False
                for e in errs:
                    check(False, e)
            else:
                check(True, f"{strat}/seed{s} memory fail-closed 通过")
    if mem_ok:
        check(True, "四策略×3seed 显存文件齐全且 fail-closed 通过")

    # ===== 11. no_candidate_pruning 口径 =====
    print("\n=== 11. no_candidate_pruning 口径（routed=3/executed=3/padded=0） ===")
    ncp_ok = True
    for kind, name, seed, p, exp_manifest, cal_dir in _iter_result_files(root):
        if not os.path.exists(p):
            continue
        d = json.load(open(p, encoding="utf-8"))
        ncp = d.get("strategies", {}).get("no_candidate_pruning", {})
        if ncp is None:
            ncp_ok = False
            check(False, f"{name}/seed{seed} 缺 no_candidate_pruning")
            continue
        if abs(ncp.get("avg_routed_candidates", -1) - 3) > 0.01:
            ncp_ok = False
            check(False, f"{name}/seed{seed} no_candidate_pruning avg_routed={ncp.get('avg_routed_candidates')}（期望3）")
        if abs(ncp.get("avg_executed_branches", -1) - 3) > 0.01:
            ncp_ok = False
            check(False, f"{name}/seed{seed} no_candidate_pruning avg_executed={ncp.get('avg_executed_branches')}（期望3）")
        if abs(ncp.get("avg_padded_branches", -1) - 0) > 0.01:
            ncp_ok = False
            check(False, f"{name}/seed{seed} no_candidate_pruning avg_padded={ncp.get('avg_padded_branches')}（期望0）")
    if ncp_ok:
        check(True, "no_candidate_pruning 口径正确（routed=3/executed=3/padded=0）")

    # ===== 12. 汇总深度比较 =====
    print("\n=== 12. 汇总深度比较（build_summary 重算 vs seed_scene_summary.json） ===")
    sp = os.path.join(root, "seed_scene_summary.json")
    if not os.path.exists(sp):
        check(False, "缺汇总文件 seed_scene_summary.json")
    else:
        try:
            from scripts.summarize_final import build_summary
            stored = json.load(open(sp, encoding="utf-8"))
            recomputed = build_summary(root)
            errs = compare_summary(recomputed, stored)
            if errs:
                for e in errs:
                    check(False, e)
            else:
                check(True, "汇总与结果深度一致（全场景全方法全指标 + cost）")
        except Exception as e:
            check(False, f"汇总比较异常：{e}")

    # ===== 13. τ 校准文件 fail-closed =====
    print("\n=== 13. τ 校准文件 fail-closed ===")
    tau_json = os.path.join(root, "tau_calibration.json")
    thr_json = os.path.join(root, "threshold_calibration.json")
    if not os.path.exists(tau_json):
        check(False, "缺 tau_calibration.json")
    else:
        td = json.load(open(tau_json, encoding="utf-8"))
        if td.get("best_tau") is None:
            check(False, "tau_calibration.json 缺 best_tau")
        else:
            check(True, f"tau_calibration.json best_tau={td.get('best_tau')}")
        if td.get("config_sha256") is None:
            check(False, "tau_calibration.json 缺 config_sha256")
        else:
            check(True, "tau_calibration.json config_sha256 非空")
    if not os.path.exists(thr_json):
        check(False, "缺 threshold_calibration.json")
    else:
        check(True, "threshold_calibration.json 存在")

    # ===== 14. 冻结快照资产哈希 + path 一致性 =====
    print("\n=== 14. 冻结快照资产哈希 + path 一致性 ===")
    snap_path = os.path.join(root, "FROZEN_SNAPSHOT.json")
    snap = None
    if not os.path.exists(snap_path):
        check(False, "缺冻结快照 FROZEN_SNAPSHOT.json")
    else:
        try:
            snap = json.load(open(snap_path, encoding="utf-8"))
        except json.JSONDecodeError as e:
            check(False, f"快照无法解析：{e}")
            snap = None

    if snap is not None:
        snap_ok = True
        # 资产列表：(label, 快照 entry dict, 预期磁盘相对路径)
        assets = []
        for s in SEEDS:
            assets.append((f"classifier seed{s}", snap.get("classifiers", {}).get(f"seed{s}"), EXPECTED_CLASSIFIER[s]))
            assets.append((f"router seed{s}", snap.get("routers", {}).get(f"seed{s}"), EXPECTED_ROUTER[s]))
        assets.append(("dncnn", snap.get("upstreams", {}).get("dncnn"), EXPECTED_DENOISE))
        assets.append(("Zero-DCE", snap.get("upstreams", {}).get("lowlight"), EXPECTED_LOWLIGHT))
        assets.append(("config", snap.get("config"), EXPECTED_CONFIG))
        assets.append(("test_manifest", snap.get("test_manifest"), EXPECTED_TEST_MANIFEST))
        for label, entry, exp_path in assets:
            if not isinstance(entry, dict) or not entry:
                snap_ok = False
                check(False, f"快照 {label} 缺失或非 dict")
                continue
            # path 一致性（不能只取 sha 后用硬编码路径验证）
            snap_p = entry.get("path")
            if not snap_p:
                snap_ok = False
                check(False, f"快照 {label} 缺 path")
            elif os.path.abspath(snap_p) != os.path.abspath(exp_path):
                snap_ok = False
                check(False, f"快照 {label} path={snap_p} 非预期 {exp_path}")
            # sha 一致性
            if not asset_hash_valid(entry.get("sha256"), snap_p if snap_p else exp_path):
                snap_ok = False
                check(False, f"快照 {label} sha256 为空/与磁盘不一致")
        # calibration 资产（快照只存 sha，路径固定 root 下）
        cal = snap.get("calibration", {})
        for label, fname in [("tau_calibration", "tau_calibration.json"),
                             ("threshold_calibration", "threshold_calibration.json")]:
            key = f"{label}_sha256"
            snap_sha = cal.get(key) if isinstance(cal, dict) else None
            disk_p = os.path.join(root, fname)
            if not asset_hash_valid(snap_sha, disk_p):
                snap_ok = False
                check(False, f"快照 {label} sha256 为空/与磁盘不一致")
        if snap_ok:
            check(True, "冻结快照 12 项资产哈希 + path 与当前一致")

    # ===== 15. 冻结快照 eval_manifests 13 项 fail-closed =====
    print("\n=== 15. 冻结快照 eval_manifests 13 项 fail-closed ===")
    if snap is not None:
        errs = validate_snapshot_eval_manifests(snap)
        if errs:
            for e in errs:
                check(False, e)
        else:
            check(True, "快照 eval_manifests 精确 13 项且 path/sha 全部与磁盘一致")

    # ===== 16. 快照 best_tau + selector 一致性 =====
    print("\n=== 16. 快照 best_tau + selector 一致性 ===")
    if snap is not None:
        snap_tau = snap.get("best_tau")
        if not os.path.exists(os.path.join(root, "tau_calibration.json")):
            check(False, "缺 tau_calibration.json（无法核对快照 best_tau）")
        else:
            cal_tau = json.load(open(os.path.join(root, "tau_calibration.json"), encoding="utf-8")).get("best_tau")
            if snap_tau is None:
                check(False, "快照缺 best_tau")
            elif not tau_equal(snap_tau, cal_tau):
                check(False, f"快照 best_tau={snap_tau} != tau_calibration best_tau={cal_tau}")
            else:
                check(True, f"快照 best_tau={snap_tau} 与 tau_calibration 一致")
        snap_selector = snap.get("selector_thresholds")
        cur_selector = {}
        try:
            import yaml as _yaml
            cur_selector = (_yaml.safe_load(open(EXPECTED_CONFIG, encoding="utf-8")) or {}).get("selector", {})
        except Exception:
            pass
        if snap_selector is None:
            check(False, "快照缺 selector_thresholds")
        elif snap_selector != cur_selector:
            check(False, "快照 selector_thresholds 与 config 当前 selector 不一致")
        else:
            check(True, "快照 selector_thresholds 与 config 一致")

    # ===== 17. 结果 → 快照 → 磁盘 eval_manifest 闭环 =====
    print("\n=== 17. 结果 → 快照 → 磁盘 eval_manifest 闭环（39 组） ===")
    if snap is not None:
        ems = snap.get("eval_manifests")
        loop_ok = True
        n_loop = 0
        if not isinstance(ems, dict):
            loop_ok = False
            check(False, "快照缺 eval_manifests（无法闭环）")
        else:
            for kind, name, seed, p, exp_manifest, cal_dir in _iter_result_files(root):
                if not os.path.exists(p):
                    continue
                d = json.load(open(p, encoding="utf-8"))
                em = d.get("run_meta", {}).get("eval_manifest", {})
                snap_entry = ems.get(name)
                if not isinstance(em, dict):
                    loop_ok = False
                    check(False, f"{name}/seed{seed} run_meta 缺 eval_manifest")
                    continue
                if not isinstance(snap_entry, dict):
                    loop_ok = False
                    check(False, f"{name}/seed{seed} 快照 eval_manifests 缺 {name}")
                    continue
                n_loop += 1
                # 结果 path == 快照 path == 磁盘
                if os.path.abspath(em.get("path", "")) != os.path.abspath(snap_entry.get("path", "")):
                    loop_ok = False
                    check(False, f"{name}/seed{seed} 结果 eval_manifest.path != 快照 path")
                elif os.path.abspath(em.get("path", "")) != os.path.abspath(exp_manifest):
                    loop_ok = False
                    check(False, f"{name}/seed{seed} 结果 eval_manifest.path != 预期 manifest")
                # 结果 sha == 快照 sha == 磁盘 sha
                if em.get("sha256") != snap_entry.get("sha256"):
                    loop_ok = False
                    check(False, f"{name}/seed{seed} 结果 eval_manifest.sha != 快照 sha")
                elif em.get("sha256") != sha256_file(em.get("path")):
                    loop_ok = False
                    check(False, f"{name}/seed{seed} 结果 eval_manifest.sha != 磁盘 sha")
            if loop_ok and n_loop == 39:
                check(True, "39 组结果 eval_manifest 与快照、磁盘三方一致")
            elif loop_ok:
                check(False, f"闭环组数 {n_loop}（期望 39）")

    print()
    if failures:
        print(f"校验失败：{len(failures)} 项")
        return 1
    print("全部通过 [OK]")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
