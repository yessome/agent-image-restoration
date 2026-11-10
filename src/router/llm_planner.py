"""LLM 决策器（可选，Agent 的"大脑"增强）。

在规则路由基础上，用 DeepSeek 做两件事：
  1. 视觉复核（review_candidates）：仅在规则不确定的场景下，把原图 +
     候选修复图 + 结构化指标发给 deepseek-flash（原生多模态），
     返回白名单内的分支建议；本地安全校验通过后才可能被采纳。
  2. 结果解释（explain）：把最终决策翻译成自然语言。

设计原则（保证可复现与可降级）：
  - 无 API Key / 请求失败 → 静默降级，Agent 依然按纯规则运行；
  - 离线实验模式（有 ground_truth）永不调用视觉复核；
  - 模型建议必须通过白名单 + fallback + clip_ratio + 置信度校验，
    否则一律丢弃，维持规则结果。

默认接入 DeepSeek（OpenAI 兼容接口）：
  - api_base: https://api.deepseek.com
  - model: deepseek-flash（V4.1 Flash，原生支持图像输入）
  - 通过环境变量 DEEPSEEK_API_KEY 提供 key
"""
from __future__ import annotations

import base64
import json
import os
from typing import Dict, List, Optional

import cv2
import numpy as np

from .diagnose import Diagnosis
from .router import PipelineRoute
from .learned_router import is_overexposed


def image_to_data_url(image: np.ndarray, quality: int = 85,
                      max_side: int = 1024) -> str:
    """RGB uint8 图像转 JPEG Data URL（OpenAI 多模态 image_url 格式）。

    max_side 限制长边像素，避免高分辨率图导致请求体过大、上传慢、接口拒绝。
    """
    h, w = image.shape[:2]
    scale = min(1.0, max_side / max(h, w))
    if scale < 1.0:
        image = cv2.resize(
            image, (int(w * scale), int(h * scale)), interpolation=cv2.INTER_AREA
        )
    bgr = cv2.cvtColor(image, cv2.COLOR_RGB2BGR)
    ok, encoded = cv2.imencode(
        ".jpg", bgr, [int(cv2.IMWRITE_JPEG_QUALITY), quality]
    )
    if not ok:
        raise ValueError("图像编码失败")
    payload = base64.b64encode(encoded.tobytes()).decode("ascii")
    return f"data:image/jpeg;base64,{payload}"


def _load_env_file(path: Optional[str] = None) -> Dict[str, str]:
    """从项目根 .env 文件读取 KEY=VALUE（轻量实现，不依赖 python-dotenv）。

    支持注释（#）、空行、双/单引号包裹的值。文件不存在时返回空 dict。
    """
    if path is None:
        project_root = os.path.dirname(
            os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
        )
        path = os.path.join(project_root, ".env")
    env: Dict[str, str] = {}
    try:
        with open(path, "r", encoding="utf-8") as f:
            for line in f:
                line = line.strip()
                if not line or line.startswith("#") or "=" not in line:
                    continue
                k, v = line.split("=", 1)
                k = k.strip()
                v = v.strip().strip('"').strip("'")
                if k:
                    env[k] = v
    except OSError:
        pass
    return env


def _extract_json(text: str) -> Optional[Dict]:
    """从模型返回文本中稳健地提取 JSON 对象。

    兼容：markdown 代码块围栏、前后缀说明文字、以及 deepseek-flash
    推理模型把 JSON 放在 reasoning_content 里的情况。
    """
    if not text:
        return None
    text = text.strip()
    # 去掉 markdown 代码块围栏
    if text.startswith("```"):
        import re
        text = re.sub(r"^```[a-zA-Z]*\s*", "", text)
        text = re.sub(r"\s*```$", "", text)
        text = text.strip()
    try:
        return json.loads(text)
    except Exception:
        pass
    # 从第一个 { 到最后一个 } 截取
    s = text.find("{")
    e = text.rfind("}")
    if s != -1 and e != -1 and e > s:
        try:
            return json.loads(text[s:e + 1])
        except Exception:
            pass
    return None


def _normalize_review(review: Optional[Dict]) -> Optional[Dict]:
    """把模型自由发挥的字段名归一化到标准字段。

    deepseek-flash 常把 recommended_pipeline 输出成 selected/choice/pipeline，
    把 visual_confidence 输出成 confidence/score。这里统一映射，保证后续
    validate_review 能读到标准字段。
    """
    if not isinstance(review, dict):
        return None
    norm = dict(review)
    pid = (review.get("recommended_pipeline")
           or review.get("selected")
           or review.get("choice")
           or review.get("pipeline")
           or review.get("pipeline_id"))
    if pid is None:
        return None
    norm["recommended_pipeline"] = pid
    conf = (review.get("visual_confidence")
            or review.get("confidence")
            or review.get("score"))
    if conf is not None:
        norm["visual_confidence"] = conf
    return norm


class LLMPlanner:
    """可选的 LLM 决策器。未配置时静默降级为 None。

    默认走 DeepSeek deepseek-flash（V4.1 Flash，原生多模态）；
    也可通过 config 切换到其他 OpenAI 兼容的视觉模型服务。
    """

    def __init__(self, config: Optional[Dict] = None):
        config = config or {}
        self.enabled = bool(config.get("enabled", False))
        self.api_base = config.get("api_base", "https://api.deepseek.com")
        self.model = config.get("model", "deepseek-flash")
        # API Key 读取优先级：环境变量 > 项目根 .env 文件
        key_env_name = config.get("api_key_env", "DEEPSEEK_API_KEY")
        self.api_key = os.environ.get(key_env_name, "")
        if not self.api_key:
            self.api_key = _load_env_file().get(key_env_name, "")
        # 视觉复核配置
        self.vision_enabled = bool(config.get("vision_enabled", True))
        self.temperature = float(config.get("temperature", 0.0))
        self.max_tokens = int(config.get("max_tokens", 500))
        self.timeout = float(config.get("timeout_seconds", 30))
        trig = config.get("trigger") or {}
        self.trigger_uncertain = bool(trig.get("only_when_uncertain", True))
        self.trigger_conflict = bool(trig.get("review_prediction_conflict", True))
        self.trigger_ambiguous = bool(trig.get("review_ambiguous_diagnosis", True))
        self.allowed_pipelines = set(
            config.get("allowed_pipelines",
                       ["raw_cls", "denoise_cls", "lowlight_cls"])
        )
        # 本地安全校验参数
        guard = config.get("guard") or {}
        self.min_visual_confidence = float(guard.get("min_visual_confidence", 0.70))
        self._client = None
        self._last_error = ""  # 最近一次视觉复核的失败原因（不含 key/图片内容）

    @property
    def available(self) -> bool:
        return self.enabled and bool(self.api_key)

    @property
    def vision_available(self) -> bool:
        return self.available and self.vision_enabled

    def _ensure_client(self):
        if self._client is None and self.available:
            try:
                from openai import OpenAI
                self._client = OpenAI(
                    base_url=self.api_base,
                    api_key=self.api_key,
                    timeout=self.timeout,
                )
            except ImportError:
                self._client = None
        return self._client

    def plan(self, diagnosis: Diagnosis, fallback: PipelineRoute) -> PipelineRoute:
        """LLM 规划流水线；失败或不可用时返回规则路由结果。

        这里保持简单：LLM 主要做视觉复核与结果解读，路由仍以规则为准，
        保证行为可预期、可复现。
        """
        return fallback

    # ------------------------------------------------------------------
    # 视觉复核（deepseek-flash 原生多模态）
    # ------------------------------------------------------------------
    def review_candidates(
        self,
        original: np.ndarray,
        outcomes: list,
        diagnosis: Diagnosis,
    ) -> Optional[Dict]:
        """把原图 + 候选修复图 + 结构化指标发给视觉模型，返回复核意见。

        返回解析后的 dict（含 recommended_pipeline 等字段）；
        任何失败（网络/格式/解析）返回 None，调用方维持规则结果。
        """
        client = self._ensure_client()
        self._last_error = ""
        if client is None or not self.vision_available:
            self._last_error = "客户端未就绪（未启用视觉或无 API Key）"
            return None

        allowed = [o for o in (outcomes or []) if o.pipeline_id in self.allowed_pipelines]
        if not allowed:
            self._last_error = "无白名单内的候选分支"
            return None

        content = [
            {"type": "text", "text": (
                "你是垃圾图像修复 Agent 的视觉复核器。"
                "请比较原图和修复候选，判断哪一张最有利于垃圾分类。"
                f"只能从以下 pipeline_id 中选择："
                f"{sorted(o.pipeline_id for o in allowed)}。"
                "不要猜测真实标签，不要把视觉美观等同于分类收益。"
                "只输出 JSON 对象，格式："
                '{"recommended_pipeline": "...", "degradation": [...], '
                '"artifact_flags": [...], "visual_confidence": 0.0, "reason": "..."}。'
                "artifact_flags 只能用英文枚举：overexposure / color_shift / "
                "texture_loss / halo / blur，没有伪影则用空列表 []。"
            )},
            {"type": "text", "text": f"诊断结果：{diagnosis.to_dict()}"},
        ]

        # 原图（raw_cls 基准参照），始终发送，并附带 raw 的分类指标（信息对称）
        raw = next((o for o in allowed if o.upstream == ""), None)
        if original is not None:
            raw_desc = "图像：raw_cls（原图直通）"
            if raw is not None:
                raw_desc += (
                    f"；分类预测={raw.pred_label}；"
                    f"置信度={raw.downstream_confidence:.4f}；"
                    f"margin={raw.margin:.4f}"
                )
            content.append({"type": "text", "text": raw_desc})
            content.append({
                "type": "image_url",
                "image_url": {"url": image_to_data_url(original)},
            })

        # 修复候选图
        for o in allowed:
            if o.upstream and getattr(o, "restored_image", None) is not None:
                content.append({"type": "text", "text": (
                    f"候选：{o.pipeline_id}；分类预测={o.pred_label}；"
                    f"置信度={o.downstream_confidence:.4f}；"
                    f"margin={o.margin:.4f}；clip_ratio={o.clip_ratio:.4f}"
                )})
                content.append({
                    "type": "image_url",
                    "image_url": {"url": image_to_data_url(o.restored_image)},
                })

        if not any(c.get("type") == "image_url" for c in content):
            self._last_error = "无可发送的图像"
            return None  # 无图可发，复核无意义

        try:
            resp = client.chat.completions.create(
                model=self.model,
                messages=[
                    {"role": "system", "content": (
                        "你只能进行视觉质量复核和候选推荐。"
                        "不得输出白名单以外的流水线。只输出 JSON。"
                    )},
                    {"role": "user", "content": content},
                ],
                temperature=self.temperature,
                max_tokens=self.max_tokens,
            )
            msg = resp.choices[0].message
            text = (msg.content or "").strip()
            # deepseek-flash 为推理模型，正式答案可能落在 reasoning_content
            if not text:
                text = (getattr(msg, "reasoning_content", None) or "").strip()
            review = _normalize_review(_extract_json(text))
            if review is None:
                self._last_error = "返回内容无法解析为 JSON（或缺少 recommended_pipeline 字段）"
            return review
        except Exception as e:
            # 记录异常类型 + 截断消息（不记录 key 与图片内容）
            self._last_error = f"{type(e).__name__}: {str(e)[:200]}"
            return None  # 任何失败 → 降级，维持规则结果

    def validate_review(
        self,
        review: Optional[Dict],
        outcomes: list,
        chosen=None,
        raw=None,
        clip_max: float = 0.10,
        tau_uncert: float = 0.10,
        kappa: float = 1.5,
        conf_drop_tolerance: float = 0.10,
    ):
        """本地安全校验模型建议，返回 (candidate, reject_reason)。

        通过时 candidate 非 None、reason 为 ""；拒绝时 candidate 为 None、reason 说明原因。

        校验项（不信任模型裸输出，也不信任其自报置信度）：
        1. 白名单 + 候选确实存在 + 非降级；
        2. 无过曝伪影（is_overexposed(clip_ratio, clip_max)，统一从 selector.clip_max 读取 + 容差口径）；
        3. visual_confidence 必须在 [0,1] 且达到下限；
        4. artifact_flags 若含严重伪影（过曝/偏色/纹理损失）则拒绝；
        5. 被推荐分支分类置信度不得明显低于当前规则选择（防止强行换上低置信度分支）；
        6. 若推荐分支类别与 raw 不同（切类），必须满足原选择器的
           tau_uncert（raw 不确定）与 kappa（修复分支 margin 显著更大）约束。
        """
        if not isinstance(review, dict):
            return None, "复核意见非字典"
        available = {o.pipeline_id: o for o in (outcomes or [])}
        pipeline_id = str(review.get("recommended_pipeline", ""))
        if pipeline_id not in self.allowed_pipelines or pipeline_id not in available:
            return None, "白名单外或候选不存在"
        candidate = available[pipeline_id]
        if candidate.fallback:
            return None, "候选为降级分支"
        if is_overexposed(candidate.clip_ratio, clip_max):
            return None, f"过曝伪影 clip_ratio={candidate.clip_ratio:.3f}>{clip_max}"

        try:
            confidence = float(review.get("visual_confidence", -1.0))
        except (TypeError, ValueError):
            return None, "visual_confidence 非法"
        if not (0.0 <= confidence <= 1.0):
            return None, "visual_confidence 越界"
        if confidence < self.min_visual_confidence:
            return None, f"视觉置信度不足 {confidence:.2f}<{self.min_visual_confidence}"

        flags = review.get("artifact_flags") or []
        severe_en = {"overexposure", "color_shift", "texture_loss", "overexposed",
                     "halo", "blur"}
        severe_cn = {"过曝", "偏色", "纹理损失", "纹理抹除", "光晕", "模糊", "重影"}
        if isinstance(flags, list):
            for f in flags:
                fs = str(f).strip()
                if fs.lower() in severe_en or fs in severe_cn:
                    return None, "存在严重伪影"

        # 分类置信度不得明显下降（防强行换低置信度分支）
        if chosen is not None and getattr(chosen, "downstream_confidence", None) is not None:
            if candidate.downstream_confidence < chosen.downstream_confidence - conf_drop_tolerance:
                return None, "分类置信度明显下降"

        # 切类必须满足原选择器安全规则
        if raw is not None and candidate.pred_label != raw.pred_label:
            if raw.margin >= tau_uncert:
                return None, "切类但 raw 不确定度不足"
            if candidate.margin <= kappa * raw.margin + 1e-6:
                return None, "切类但 margin 增益不足"

        return candidate, ""

    # ------------------------------------------------------------------
    # 结果解释
    # ------------------------------------------------------------------
    def explain(self, diagnosis: Diagnosis, chosen, outcomes,
                vision_review: Optional[Dict] = None,
                force_local: bool = False) -> str:
        """生成结果解读。

        force_local=True（离线评测、视觉复核场景）时直接返回本地固定文案，
        不调用 LLM，保证可复现、零 API 费用。
        """
        score_summary = ", ".join(
            f"{o.pipeline_id}={o.score:.3f}" for o in (outcomes or [])
        )
        base = (
            f"图像诊断：{diagnosis.summary()}；"
            f"候选方案分数 [{score_summary}]；"
            f"最终选择「{chosen.pipeline_id}」"
            f"（上游 {chosen.upstream or '原图直通'}）。"
        )
        if vision_review:
            base += (
                f" 视觉复核建议 {vision_review.get('recommended_pipeline', '?')}"
                f"（置信度 {vision_review.get('visual_confidence', '?')}）。"
            )

        client = self._ensure_client()
        if force_local or client is None:
            return base

        review_note = ""
        if vision_review:
            review_note = (
                f"\n- 视觉复核意见（{self.model}）："
                f"{json.dumps(vision_review, ensure_ascii=False)}"
            )

        prompt = (
            "你是垃圾图像修复 Agent 的结果解释器。请用简洁中文解释本次决策：\n"
            f"- 图像质量诊断：{diagnosis.summary()}\n"
            f"- 各候选流水线及其下游置信度：{score_summary}\n"
            f"- 最终选择的流水线：{chosen.pipeline_id}（上游 "
            f"{chosen.upstream or '原图直通'}）{review_note}\n"
            "请说明：为什么这样选择、修复是否可能对垃圾分类有利、"
            "有无需要注意的风险。控制在 3 句话以内。"
        )
        try:
            resp = client.chat.completions.create(
                model=self.model,
                messages=[
                    {"role": "system", "content": "你是图像修复 Agent 的决策解释器。"},
                    {"role": "user", "content": prompt},
                ],
                temperature=0.2,
                max_tokens=200,
            )
            return resp.choices[0].message.content.strip()
        except Exception as e:  # 网络/鉴权失败时降级
            return (
                f"图像诊断：{diagnosis.summary()}；"
                f"候选方案分数 [{score_summary}]；"
                f"最终选择「{chosen.pipeline_id}」。"
                f"（LLM 解读失败，已降级：{e}）"
            )
