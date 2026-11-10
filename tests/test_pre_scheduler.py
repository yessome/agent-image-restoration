"""运行前调度器（pre_scheduler）的单元测试。

约束：不联网、不加载真实权重、不启动 GPU 推理（用 stub 模型与 stub 执行函数）。
"""
from __future__ import annotations

import os
import sys
import unittest

import numpy as np

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from src.router.diagnose import Diagnosis
from src.router.pre_scheduler import (
    PRE_BRANCH_ORDER,
    PRE_FEATURE_DIM,
    PRE_SOURCE_INDICES,
    PreScheduler,
    build_pre_labels,
    build_pre_target,
    extract_pre_features,
    rule_select,
    validate_checkpoint,
)
from src.router.tool_registry import BRANCH_ORDER as ALL_BRANCH_ORDER


def _diag(noise=0.1, low=0.1, bright=120.0, nvar=20.0, dark=0.05, clip=0.0):
    return Diagnosis(
        noise_score=noise, lowlight_score=low,
        brightness_mean=bright, noise_var=nvar,
        dark_pixel_ratio=dark, clip_ratio=clip,
    )


class TestBranchScope(unittest.TestCase):
    def test_13_branches_raw_first(self):
        self.assertEqual(len(PRE_BRANCH_ORDER), 13)
        self.assertEqual(PRE_BRANCH_ORDER[0], "")

    def test_excludes_deblur_and_sr(self):
        joined = " ".join(PRE_BRANCH_ORDER)
        for excluded in ("restormer_defocus", "mprnet_deblur", "realesrgan_x2", "realesrgan_x4"):
            self.assertNotIn(excluded, joined)

    def test_source_indices_point_to_same_names(self):
        # 关键：映射必须按名称解析，绝不能靠「正好连续」的巧合
        for i, up in enumerate(PRE_BRANCH_ORDER):
            self.assertEqual(ALL_BRANCH_ORDER[PRE_SOURCE_INDICES[i]], up)
        self.assertEqual(len(set(PRE_SOURCE_INDICES)), 13)


class TestFeatures(unittest.TestCase):
    def test_dim_is_6(self):
        f = extract_pre_features(_diag())
        self.assertEqual(f.shape, (PRE_FEATURE_DIM,))
        self.assertEqual(f.dtype, np.float32)

    def test_matches_learned_router_prefix(self):
        """6 维口径必须与 learned_router 的诊断特征前 6 维完全一致。"""
        from src.router.learned_router import extract_features

        d = _diag(noise=0.7, low=0.3, bright=88.0, nvar=250.0, dark=0.31, clip=0.02)
        ours = extract_pre_features(d)
        full = extract_features(d, outcomes=[], include_diag=True)
        np.testing.assert_allclose(ours, full[:6], rtol=0, atol=0)

    def test_no_post_execution_input(self):
        """特征只能来自诊断：传入 outcomes 与否都不影响（接口上根本不接受 outcomes）。"""
        d = _diag()
        f = extract_pre_features(d)
        self.assertEqual(f.shape[0], 6)
        import inspect

        sig = inspect.signature(extract_pre_features)
        self.assertEqual(list(sig.parameters), ["diagnosis"])


class TestLabels(unittest.TestCase):
    def _y(self, correct_names):
        return np.array([1.0 if up in correct_names else 0.0 for up in ALL_BRANCH_ORDER],
                        dtype=np.float32)

    def test_pure_mode_keeps_all_correct(self):
        y = self._y({"", "denoise"})
        lab = build_pre_labels(y, raw_priority=False)
        self.assertEqual(lab[0], 1.0)
        self.assertEqual(lab[PRE_BRANCH_ORDER.index("denoise")], 1.0)

    def test_raw_priority_zeros_others_when_raw_ok(self):
        y = self._y({"", "denoise", "lowlight"})
        lab = build_pre_labels(y, raw_priority=True)
        self.assertEqual(lab[0], 1.0)
        self.assertTrue((lab[1:] == 0).all(), "raw 判对时其余分支标签必须清零")

    def test_raw_priority_keeps_tool_when_raw_wrong(self):
        y = self._y({"denoise"})
        lab = build_pre_labels(y, raw_priority=True)
        self.assertEqual(lab[0], 0.0)
        self.assertEqual(lab[PRE_BRANCH_ORDER.index("denoise")], 1.0)

    def test_rejects_wrong_dim(self):
        with self.assertRaises(ValueError):
            build_pre_labels(np.zeros(3, dtype=np.float32))

    def test_excludes_deblur_labels(self):
        """去模糊/超分即使正确，也不能出现在 13 维标签里。"""
        y = self._y({"restormer_defocus", "realesrgan_x4"})
        lab = build_pre_labels(y, raw_priority=False)
        self.assertTrue((lab == 0).all())


class TestSingleLabelTarget(unittest.TestCase):
    """方案 A：决策式单标签目标。"""

    def _y(self, correct_names):
        return np.array([1.0 if up in correct_names else 0.0 for up in ALL_BRANCH_ORDER],
                        dtype=np.float32)

    def test_raw_correct_maps_to_raw(self):
        y = self._y({"", "denoise"})
        self.assertEqual(build_pre_target(y), 0, "raw 判对必须标 raw（不切换）")

    def test_raw_wrong_picks_correct_tool(self):
        y = self._y({"denoise"})
        self.assertEqual(build_pre_target(y), PRE_BRANCH_ORDER.index("denoise"))

    def test_multiple_correct_tools_pick_earliest(self):
        y = self._y({"scunet", "denoise"})
        self.assertEqual(build_pre_target(y), PRE_BRANCH_ORDER.index("denoise"),
                         "多个工具都正确时取分支顺序最靠前的")

    def test_all_wrong_maps_to_raw(self):
        y = self._y(set())
        self.assertEqual(build_pre_target(y), 0)

    def test_deblur_only_correct_maps_to_raw(self):
        """去模糊正确但不在 13 分支范围内 → 只能标 raw。"""
        y = self._y({"restormer_defocus"})
        self.assertEqual(build_pre_target(y), 0)

    def test_rejects_wrong_dim(self):
        with self.assertRaises(ValueError):
            build_pre_target(np.zeros(5, dtype=np.float32))


class TestSoftmaxScoring(unittest.TestCase):
    def test_softmax_probs_sum_to_one(self):
        import torch

        s = PreScheduler(weight_path=None, device="cpu")
        s.output_activation = "softmax"
        s.model = lambda x: torch.zeros(x.shape[0], len(PRE_BRANCH_ORDER))
        probs = s.score(_diag())
        self.assertAlmostEqual(float(probs.sum()), 1.0, places=5)
        self.assertEqual(len(probs), 13)

    def test_checkpoint_accepts_softmax(self):
        validate_checkpoint({"feature_dim": 6, "branch_order": PRE_BRANCH_ORDER,
                             "output_activation": "softmax"})

    def test_checkpoint_rejects_unknown_activation(self):
        with self.assertRaises(ValueError):
            validate_checkpoint({"feature_dim": 6, "branch_order": PRE_BRANCH_ORDER,
                                 "output_activation": "relu"})


class TestRuleSelect(unittest.TestCase):
    def test_noise_high(self):
        up, reason = rule_select(_diag(noise=0.9, low=0.2), confidence_threshold=0.6)
        self.assertEqual(up, "denoise")
        self.assertIn("rule_pre", reason)

    def test_lowlight_dominates(self):
        up, _ = rule_select(_diag(noise=0.7, low=0.85), confidence_threshold=0.6)
        self.assertEqual(up, "lowlight")

    def test_clean_falls_back_to_raw(self):
        up, reason = rule_select(_diag(noise=0.1, low=0.1), confidence_threshold=0.6)
        self.assertEqual(up, "")
        self.assertIn("raw", reason)

    def test_always_single_branch(self):
        for n, l in [(0.0, 0.0), (0.9, 0.0), (0.0, 0.9), (0.8, 0.8)]:
            up, _ = rule_select(_diag(noise=n, low=l), confidence_threshold=0.6)
            self.assertIn(up, PRE_BRANCH_ORDER)


class TestCheckpointValidation(unittest.TestCase):
    def test_rejects_91_dim(self):
        with self.assertRaises(ValueError):
            validate_checkpoint({"feature_dim": 91, "branch_order": PRE_BRANCH_ORDER})

    def test_rejects_wrong_branch_order(self):
        with self.assertRaises(ValueError):
            validate_checkpoint({"feature_dim": 6, "branch_order": list(ALL_BRANCH_ORDER)})

    def test_rejects_missing_branch_order(self):
        with self.assertRaises(ValueError):
            validate_checkpoint({"feature_dim": 6})

    def test_accepts_correct(self):
        validate_checkpoint({"feature_dim": 6, "branch_order": PRE_BRANCH_ORDER})


class TestSelect(unittest.TestCase):
    def _sched_with_uniform_logits(self):
        import torch

        s = PreScheduler(weight_path=None, device="cpu")
        s.model = lambda x: torch.zeros(x.shape[0], len(PRE_BRANCH_ORDER))
        return s

    def test_tie_prefers_raw(self):
        s = self._sched_with_uniform_logits()
        up, reason, probs = s.select(_diag(noise=0.9, low=0.9))
        self.assertEqual(up, "", "概率并列时必须选 raw")
        self.assertEqual(len(probs), 13)
        self.assertIn("pre_scheduler", reason)

    def test_unavailable_falls_back_to_rule(self):
        s = PreScheduler(weight_path=None, device="cpu")
        self.assertFalse(s.available)
        up, reason, probs = s.select(_diag(noise=0.95, low=0.1))
        self.assertEqual(up, "denoise")
        self.assertIn("fallback_rule", reason)
        self.assertEqual(probs, [])

    def test_never_returns_branch_outside_scope(self):
        s = self._sched_with_uniform_logits()
        up, _, _ = s.select(_diag())
        self.assertIn(up, PRE_BRANCH_ORDER)


class TestOrchestratorPreMode(unittest.TestCase):
    """验证 pre 模式只执行 1 个分支（用 stub 替换真实推理）。"""

    def _agent(self, mode, weight=None):
        from src.agents.orchestrator import ImageRestorationAgent

        cfg = {"routing": {"mode": mode},
               "pre_scheduler": {"weight_path": weight or ""},
               "confidence_threshold": 0.6}
        agent = ImageRestorationAgent(config=cfg, device="cpu")
        agent.llm.explain = lambda *a, **k: "stub"  # 避免任何解释器副作用
        calls = []

        def _fake_exec(image, route, task_kwargs, ground_truth=None):
            from src.agents.orchestrator import PipelineOutcome

            calls.append(route.upstream)
            return PipelineOutcome(
                pipeline_id=route.pipeline_id, upstream=route.upstream,
                downstream=route.downstream, downstream_confidence=0.5,
                margin=0.1, clip_ratio=0.0, fallback=False,
                is_correct=True, ground_truth_score=1.0,
            )

        agent._execute_pipeline = _fake_exec
        return agent, calls

    def test_rule_pre_executes_exactly_once(self):
        agent, calls = self._agent("rule_pre")
        img = np.zeros((16, 16, 3), dtype=np.uint8)
        res = agent._run_pre_scheduled(img, _diag(noise=0.95, low=0.1), "classify", {}, None)
        self.assertEqual(len(calls), 1)
        self.assertEqual(calls[0], "denoise")
        self.assertEqual(res.restore_calls, 1)
        self.assertEqual(res.routing_mode, "rule_pre")
        self.assertEqual(len(res.outcomes), 1)

    def test_raw_choice_calls_zero_restorers(self):
        agent, calls = self._agent("rule_pre")
        img = np.zeros((16, 16, 3), dtype=np.uint8)
        res = agent._run_pre_scheduled(img, _diag(noise=0.05, low=0.05), "classify", {}, None)
        self.assertEqual(calls, [""])
        self.assertEqual(res.restore_calls, 0)

    def test_pre_without_weight_falls_back_with_reason(self):
        agent, calls = self._agent("pre", weight="weights/__not_exist__.pth")
        img = np.zeros((16, 16, 3), dtype=np.uint8)
        res = agent._run_pre_scheduled(img, _diag(noise=0.95, low=0.1), "classify", {}, None)
        self.assertEqual(len(calls), 1, "回退后仍只执行 1 个分支")
        self.assertEqual(res.routing_mode, "rule_pre")
        self.assertTrue(res.select_reason.startswith("fallback→"), res.select_reason)

    def test_legacy_mode_unchanged(self):
        agent, _ = self._agent("legacy")
        self.assertIsNone(agent._pre_scheduler)
        self.assertEqual(agent.routing_mode, "legacy")


if __name__ == "__main__":
    unittest.main(verbosity=2)
