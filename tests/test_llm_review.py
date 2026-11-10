"""离线测试 DeepSeek 视觉复核结果的本地筛选。"""
from __future__ import annotations

import os
import sys
import unittest

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from src.router.llm_planner import LLMPlanner, _extract_json
from src.agents.orchestrator import PipelineOutcome, ImageRestorationAgent
from src.router.diagnose import Diagnosis


def _o(pid, up="", pred="plastic", conf=0.6, margin=0.2, clip=0.0, fallback=False):
    return PipelineOutcome(
        pipeline_id=pid, upstream=up, downstream="classify",
        pred_label=pred, downstream_confidence=conf,
        margin=margin, clip_ratio=clip, fallback=fallback,
    )


class TestJSONExtraction(unittest.TestCase):
    def test_markdown_fence(self):
        self.assertEqual(_extract_json('```json\n{"a": 1}\n```'), {"a": 1})

    def test_prefix_suffix(self):
        self.assertEqual(_extract_json('推理过程...{"a": 1}...结束'), {"a": 1})

    def test_invalid(self):
        self.assertIsNone(_extract_json("这不是 json"))
        self.assertIsNone(_extract_json(""))


class TestValidateReview(unittest.TestCase):
    def setUp(self):
        self.llm = LLMPlanner({"enabled": True, "guard": {"min_visual_confidence": 0.70}})
        self.raw = _o("raw_cls", pred="plastic", conf=0.6, margin=0.05)   # 不确定
        self.low = _o("lowlight_cls", "lowlight", pred="plastic", conf=0.65, margin=0.3)
        self.den = _o("denoise_cls", "denoise", pred="glass", conf=0.7, margin=0.5)
        self.outs = [self.raw, self.low, self.den]
        self.chosen = self.low

    def _v(self, **kw):
        base = {"recommended_pipeline": "lowlight_cls", "visual_confidence": 0.85}
        base.update(kw)
        return base

    def test_normal_pass(self):
        c, reason = self.llm.validate_review(self._v(), self.outs, chosen=self.chosen, raw=self.raw)
        self.assertIsNotNone(c)
        self.assertEqual(reason, "")

    def test_whitelist_outside_reject(self):
        c, reason = self.llm.validate_review(self._v(recommended_pipeline="hack_cls"), self.outs)
        self.assertIsNone(c)
        self.assertIn("白名单", reason)

    def test_fallback_reject(self):
        fb = _o("lowlight_cls", "lowlight", fallback=True)
        c, reason = self.llm.validate_review(self._v(), [self.raw, fb])
        self.assertIsNone(c)
        self.assertIn("降级", reason)

    def test_overexposure_reject(self):
        hi = _o("lowlight_cls", "lowlight", clip=0.15)
        c, reason = self.llm.validate_review(self._v(), [self.raw, hi], clip_max=0.10)
        self.assertIsNone(c)
        self.assertIn("伪影", reason)

    def test_confidence_out_of_range(self):
        c, reason = self.llm.validate_review(self._v(visual_confidence=1.5), self.outs)
        self.assertIsNone(c)
        self.assertIn("越界", reason)

    def test_confidence_below_min(self):
        c, reason = self.llm.validate_review(self._v(visual_confidence=0.5), self.outs)
        self.assertIsNone(c)
        self.assertIn("不足", reason)

    def test_artifact_english_reject(self):
        c, reason = self.llm.validate_review(self._v(artifact_flags=["overexposure"]), self.outs)
        self.assertIsNone(c)
        self.assertIn("伪影", reason)

    def test_artifact_chinese_reject(self):
        for cn in ["过曝", "偏色", "纹理损失", "模糊", "光晕"]:
            c, reason = self.llm.validate_review(self._v(artifact_flags=[cn]), self.outs)
            self.assertIsNone(c, f"中文伪影 {cn} 应被拒绝")

    def test_conf_drop_reject(self):
        bad = _o("lowlight_cls", "lowlight", conf=0.4)
        c, reason = self.llm.validate_review(
            self._v(), [self.raw, bad], chosen=self.chosen, raw=self.raw, conf_drop_tolerance=0.10)
        self.assertIsNone(c)
        self.assertIn("置信度", reason)

    def test_switch_class_raw_uncertain_reject(self):
        raw_conf = _o("raw_cls", pred="plastic", margin=0.5)  # raw 很确定
        den2 = _o("denoise_cls", "denoise", pred="glass", conf=0.8, margin=0.6)
        c, reason = self.llm.validate_review(
            self._v(recommended_pipeline="denoise_cls"),
            [raw_conf, den2], chosen=self.chosen, raw=raw_conf, tau_uncert=0.10, kappa=1.5)
        self.assertIsNone(c)
        self.assertIn("不确定度", reason)

    def test_switch_class_margin_gain_reject(self):
        raw_un = _o("raw_cls", pred="plastic", margin=0.05)
        den3 = _o("denoise_cls", "denoise", pred="glass", conf=0.8, margin=0.06)
        c, reason = self.llm.validate_review(
            self._v(recommended_pipeline="denoise_cls"),
            [raw_un, den3], chosen=self.chosen, raw=raw_un, tau_uncert=0.10, kappa=1.5)
        self.assertIsNone(c)
        self.assertIn("增益", reason)


class TestTriggerAndNoCall(unittest.TestCase):
    def setUp(self):
        self.agent = ImageRestorationAgent.__new__(ImageRestorationAgent)
        self.agent._select_cfg = {"tau_uncert": 0.10}
        self.agent.llm = LLMPlanner({"enabled": True})

    def test_raw_only_no_trigger(self):
        raw_only = _o("raw_cls", pred="plastic", margin=0.01)
        self.assertFalse(self.agent._needs_vision_review([raw_only], Diagnosis()))

    def test_conflict_trigger(self):
        raw = _o("raw_cls", pred="plastic", margin=0.5)
        den = _o("denoise_cls", "denoise", pred="glass", margin=0.6)
        self.assertTrue(self.agent._needs_vision_review([raw, den], Diagnosis()))

    def test_force_local_no_network(self):
        llm = LLMPlanner({"enabled": True})

        class BoomCompletions:
            def create(self, *a, **k):
                raise AssertionError("force_local 不应发起网络请求")
        class BoomChat:
            completions = BoomCompletions()
        class BoomClient:
            chat = BoomChat()
        llm._client = BoomClient()

        class Chosen:
            pipeline_id = "raw_cls"
            upstream = ""
        out = llm.explain(Diagnosis(), Chosen(), [], force_local=True)
        self.assertIsInstance(out, str)


if __name__ == "__main__":
    unittest.main(verbosity=2)
