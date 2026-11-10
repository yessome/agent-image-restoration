"""CLI 入口：垃圾图像修复 Agent。

用法示例：
    # 垃圾分类（主线）
    python -m src.main --image garbage.jpg --intent classify

    # 离线实验（带真值标签，计算 Oracle/命中率/Regret）
    python -m src.main --image garbage.jpg --intent classify --ground-truth plastic

    # 指定设备、CPU 运行
    python -m src.main --image garbage.jpg --device cpu
"""
from __future__ import annotations

import argparse
import os
import sys

import cv2
import numpy as np
import yaml

from .agents.orchestrator import ImageRestorationAgent

DEFAULT_CONFIG = "config/iterative_agent_13tool.yaml"


def load_config(path: str, _seen=None) -> dict:
    path = os.path.abspath(path)
    seen = set(_seen or ())
    if path in seen:
        raise ValueError("配置继承存在循环")
    seen.add(path)
    with open(path, "r", encoding="utf-8") as f:
        config = yaml.safe_load(f) or {}
    base = config.pop("base_config", None)
    if not base:
        return config
    parent = load_config(os.path.join(os.path.dirname(path), base), seen)

    def merge(target, updates):
        for key, value in updates.items():
            if isinstance(value, dict) and isinstance(target.get(key), dict):
                merge(target[key], value)
            else:
                target[key] = value
        return target
    return merge(parent, config)


def _resolve_weight_paths(config: dict, config_dir: str) -> dict:
    """把 config 里的相对权重路径解析为绝对路径。

    pipelines.yaml 中的 weight_path（如 "weights/dncnn_gaussian_color.pth"）是相对
    项目根/配置文件的路径；将其解析为绝对路径，确保 `python -m src.main`
    与 demo.py 的行为一致（否则修复分支会因找不到权重而降级为原图）。
    """
    for section in ("upstreams", "downstreams"):
        registry = config.get(section)
        if not isinstance(registry, dict):
            continue
        for name, cfg in registry.items():
            if not isinstance(cfg, dict):
                continue
            wp = cfg.get("weight_path")
            if isinstance(wp, str) and wp and not wp.startswith(("http://", "https://")):
                if not os.path.isabs(wp):
                    wp = os.path.join(config_dir, wp)
                cfg["weight_path"] = os.path.normpath(wp)
    # 解析 router.learned_weight（learned router 权重，同样相对项目根）
    router_cfg = config.get("router")
    if isinstance(router_cfg, dict):
        wp = router_cfg.get("learned_weight")
        if isinstance(wp, str) and wp and not wp.startswith(("http://", "https://")):
            if not os.path.isabs(wp):
                wp = os.path.join(config_dir, wp)
            router_cfg["learned_weight"] = os.path.normpath(wp)
    return config


def main() -> int:
    parser = argparse.ArgumentParser(description="垃圾图像修复 Agent")
    parser.add_argument("--image", required=True, help="输入图像路径")
    parser.add_argument("--intent", default="classify", choices=["classify"],
                        help="下游任务意图（仅分类）")
    parser.add_argument("--labels", default=None, help="分类候选标签，逗号分隔")
    parser.add_argument("--ground-truth", default=None, dest="ground_truth",
                        help="真实标签（离线实验模式）")
    parser.add_argument("--config", default=DEFAULT_CONFIG,
                        help="配置文件路径（默认 13 分支 learned 反馈循环）")
    parser.add_argument("--device", default="cuda", help="cuda / cpu")
    parser.add_argument("--enable-model-toolbox", action="store_true",
                        help="启用配置中的修复工具箱，仍遵守 model_toolbox.exclude")
    toolbox_config = load_config(os.path.join(os.path.dirname(os.path.dirname(__file__)),
                                               "config", "pretrained_tools.yaml"))
    parser.add_argument("--restore-tool", choices=["denoise", "lowlight",
                                                    *toolbox_config.get("upstreams", {})],
                        help="直接调用指定修复器并保存图像，不运行分类 Agent")
    parser.add_argument("--output", default=None, help="--restore-tool 的输出图像路径")
    parser.add_argument("--sigma", type=float, default=25.0,
                        help="FFDNet 噪声等级（0-255），默认 25")
    parser.add_argument("--router", default="auto", choices=["learned", "rule", "auto"],
                        help="决策器：learned=可学习router，rule=规则，auto=有权重则learned否则rule")
    parser.add_argument("--routing-mode", default=None, choices=["iterative", "legacy", "pre", "rule_pre"],
                        help="调度模式：iterative=learned router 根据反馈逐步调用/停止；legacy=先全跑再选；"
                             "pre=运行前调度（6 维诊断特征选 1 个分支，只执行一次）；"
                             "rule_pre=规则单分支基线。缺省读 config/routing_pre.yaml（默认 legacy）")
    parser.add_argument("--pre-scheduler-weight", default=None, dest="pre_scheduler_weight",
                        help="覆盖运行前调度器权重路径（仅 --routing-mode pre 生效）")
    parser.add_argument("--iterative-router-weight", default=None,
                        help="覆盖循环 learned router 权重（必须匹配冻结分类器）")
    parser.add_argument("--learned-weight", default=None,
                        help="覆盖 legacy 执行后 learned router 权重")
    args = parser.parse_args()

    config = load_config(args.config) if args.config else {}
    # 解析权重路径（相对项目根目录，即 config 目录的上一级）
    config_dir = os.path.dirname(os.path.abspath(args.config)) if args.config else os.getcwd()
    project_root = os.path.dirname(config_dir) if os.path.basename(config_dir) == "config" else config_dir
    config = _resolve_weight_paths(config, project_root)
    extra_config_path = os.path.join(project_root, "config", "pretrained_tools.yaml")
    extra_config = load_config(extra_config_path)
    for name, upstream_cfg in extra_config.get("upstreams", {}).items():
        cfg = config.setdefault("upstreams", {}).setdefault(name, {})
        for key, value in upstream_cfg.items():
            if key == "weight_path" and isinstance(value, str) and not os.path.isabs(value):
                value = os.path.join(project_root, value)
            cfg.setdefault(key, value)
    if args.enable_model_toolbox:
        config.setdefault("model_toolbox", {})["enabled"] = True

    # 运行前调度配置：来自独立文件 config/routing_pre.yaml（不改动 pipelines.yaml）。
    # 只补齐 pipelines.yaml 里没定义的键，避免污染既有实验配置。
    routing_cfg_path = os.path.join(project_root, "config", "routing_pre.yaml")
    if os.path.isfile(routing_cfg_path):
        routing_file = load_config(routing_cfg_path) or {}
        for key, value in routing_file.items():
            if isinstance(value, dict):
                target = config.setdefault(key, {})
                for sub_key, sub_value in value.items():
                    target.setdefault(sub_key, sub_value)
            else:
                config.setdefault(key, value)
    if args.routing_mode:
        config.setdefault("routing", {})["mode"] = args.routing_mode
    if args.iterative_router_weight:
        config.setdefault("iterative_router", {})["weight_path"] = os.path.abspath(args.iterative_router_weight)
    if args.learned_weight:
        config.setdefault("router", {})["learned_weight"] = os.path.abspath(args.learned_weight)
    if args.pre_scheduler_weight:
        wp = args.pre_scheduler_weight
        if not os.path.isabs(wp):
            wp = os.path.join(project_root, wp)
        config.setdefault("pre_scheduler", {})["weight_path"] = wp
    if args.restore_tool and not args.output:
        parser.error("使用 --restore-tool 时必须同时提供 --output")
    if args.restore_tool == "ffdnet":
        config.setdefault("upstreams", {}).setdefault("ffdnet", {})["sigma"] = args.sigma

    # --router 开关：显式指定决策器，避免「以为在测 learned 实际在测 rule」。
    # learned：权重缺失/加载失败直接报错；auto：失败才允许回退规则。
    router_cfg = config.setdefault("router", {})
    iterative = config.get("routing", {}).get("mode") == "iterative"
    if args.restore_tool:
        config.setdefault("routing", {})["mode"] = "legacy"
        router_cfg["learned_weight"] = ""
        iterative = False
    elif iterative:
        if args.router == "rule":
            parser.error("iterative 模式必须使用 learned router；规则对照请显式指定 --routing-mode legacy")
        router_cfg["learned_weight"] = ""  # Separate checkpoint/schema; no legacy fallback.
    elif args.router == "rule":
        router_cfg["learned_weight"] = ""
    elif args.router == "learned":
        wp = router_cfg.get("learned_weight")
        if not wp:
            print("错误：--router learned 但 config 未配置 router.learned_weight",
                  file=sys.stderr)
            return 1
        if not os.path.exists(wp):
            print(f"错误：--router learned 但权重文件不存在：{wp}", file=sys.stderr)
            return 1

    image_bgr = cv2.imread(args.image)
    if image_bgr is None:
        print(f"错误：无法读取图像 {args.image}", file=sys.stderr)
        return 1
    image = cv2.cvtColor(image_bgr, cv2.COLOR_BGR2RGB)

    task_kwargs = {}
    if args.intent == "classify" and args.labels:
        task_kwargs["labels"] = [x.strip() for x in args.labels.split(",")]

    try:
        agent = ImageRestorationAgent(config=config, device=args.device)
    except (ValueError, OSError, RuntimeError) as exc:
        print(f"错误：{exc}", file=sys.stderr)
        return 1
    if args.restore_tool:
        result = agent._get_upstream(args.restore_tool).restore(image)
        if result.meta.get("fallback"):
            print(f"修复器不可用：{result.meta.get('note', '未知错误')}", file=sys.stderr)
            return 1
        os.makedirs(os.path.dirname(os.path.abspath(args.output)), exist_ok=True)
        if not cv2.imwrite(args.output, cv2.cvtColor(result.image, cv2.COLOR_RGB2BGR)):
            print(f"错误：无法写出图像 {args.output}", file=sys.stderr)
            return 1
        print(f"修复完成：{args.restore_tool} -> {args.output} | {result.meta}")
        return 0
    if args.router == "learned" and not iterative and agent._learned_router is None:
        print("错误：--router learned 但 learned router 加载失败（见上方警告）",
              file=sys.stderr)
        return 1
    result = agent.run(
        image,
        intent=args.intent,
        task_kwargs=task_kwargs,
        ground_truth=args.ground_truth,
    )

    print("=" * 60)
    print(f"诊断: {result.diagnosis}")
    if result.routing_mode != "legacy":
        # 运行前调度：显式报告「选了哪个分支 / 为什么 / 实际调用了几次修复模型」
        print(f"调度模式: {result.routing_mode}")
        print(f"选中分支: {result.selected_upstream or 'raw（不调用修复模型）'}")
        print(f"选择原因: {result.select_reason}")
        print(f"修复模型调用次数: {result.restore_calls}")
        for event in result.routing_trace:
            print(f"  轮次 {event['step']}: {event['action']} {event['upstream'] or 'raw'} ({event['reason']})")
    print(f"最终选择流水线: {result.chosen_pipeline}")
    print(f"决策: {result.decision}")
    print(f"说明: {result.explanation}")
    if result.vision_triggered:
        print("-" * 60)
        print(f"视觉复核: 已触发 | 请求成功={result.vision_succeeded} "
              f"| 耗时={result.vision_latency_ms:.0f}ms")
        if result.vision_succeeded and result.vision_review:
            rec = result.vision_review.get("recommended_pipeline", "?")
            conf = result.vision_review.get("visual_confidence", "?")
            print(f"  模型建议: {rec} (视觉置信度 {conf})")
        if result.vision_error:
            print(f"  错误: {result.vision_error}")
        if result.vision_reject_reason:
            print(f"  建议被拒绝: {result.vision_reject_reason}")
        print(f"  结果: {'采纳模型建议' if result.vision_applied else '维持规则结果'}")
    print("-" * 60)
    print("各候选方案:")
    for o in result.outcomes:
        flag = "  <- selected" if o.pipeline_id == result.chosen_pipeline else ""
        gt = f" 真值得分={o.ground_truth_score:.1f} 正确={o.is_correct}" if result.oracle_pipeline else ""
        print(f"  {o.pipeline_id:16s} 代理分={o.score:.4f}  fallback={o.fallback}{gt}{flag}")
    if result.oracle_pipeline:
        print("-" * 60)
        print(f"Oracle 最优分支: {result.oracle_pipeline}")
        print(f"选择命中: {result.hit}  |  Regret: {result.regret:.2f}")
    print("=" * 60)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
