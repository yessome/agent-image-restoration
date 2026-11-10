"""ImageRestorationAgent 主流程的单元测试（不加载任何模型权重）。

覆盖用户指定的五个场景：
  1. 纯噪声   → raw + 8 个去噪工具
  2. 纯低光   → raw + 4 个低光工具
  3. 两者同时 → raw + 8 + 4（独立判定，不互相遮蔽）
  4. raw-only → 只执行原图直通
  5. 所有修复分支都不合格 → 回退 raw 的分类结果（且不重复分类原图）

另外校验：
  - 每个修复器都独立接收**原图**（不是上一个分支的输出）
  - 每个候选只分类一次
  - 真实标签不参与在线选择
  - global_oracle 只补跑分支算上界，不改变 Agent 的实际选择
"""
from __future__ import annotations

import os
import sys
import unittest
from types import SimpleNamespace

import numpy as np

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from src.agents.orchestrator import ImageRestorationAgent, PipelineOutcome
from src.router.diagnose import Diagnosis
from src.router.tool_registry import TOOL_GROUPS, ALL_UPSTREAM_TOOLS


def _agent(**extra_cfg):
    cfg = {"model_toolbox": {"enabled": True}, "confidence_threshold": 0.6}
    cfg.update(extra_cfg)
    agent = ImageRestorationAgent(config=cfg, device="cpu")
    agent.llm.explain = lambda *a, **k: "stub"
    return agent


def _outcome(pid, upstream, conf=0.6, margin=0.2, label="glass", clip=0.0, fallback=False):
    return PipelineOutcome(
        pipeline_id=pid, upstream=upstream, downstream="classify",
        score=conf, downstream_confidence=conf, margin=margin,
        pred_label=label, clip_ratio=clip, fallback=fallback,
    )


class TestCandidateGeneration(unittest.TestCase):
    """噪声/低光必须独立判定，不能命中一种就漏掉另一种。"""

    def setUp(self):
        self.agent = _agent()

    def _ups(self, diag):
        return [c.upstream for c in self.agent._gen_candidates(diag)]

    def test_tool_counts(self):
        self.assertEqual(len(TOOL_GROUPS["noise"]), 8)
        self.assertEqual(len(TOOL_GROUPS["lowlight"]), 4)
        self.assertEqual(len(ALL_UPSTREAM_TOOLS), 16)
        for gone in ("restormer_derain", "mprnet_derain"):
            self.assertNotIn(gone, ALL_UPSTREAM_TOOLS, "去雨工具已移除，不应出现在注册表")

    def test_pure_noise(self):
        ups = self._ups(Diagnosis(noise_score=0.9, lowlight_score=0.1))
        self.assertEqual(ups, [""] + TOOL_GROUPS["noise"])
        self.assertEqual(len(ups), 9)

    def test_pure_lowlight(self):
        ups = self._ups(Diagnosis(noise_score=0.1, lowlight_score=0.9))
        self.assertEqual(ups, [""] + TOOL_GROUPS["lowlight"])
        self.assertEqual(len(ups), 5)

    def test_noise_and_lowlight_together(self):
        ups = self._ups(Diagnosis(noise_score=0.9, lowlight_score=0.9))
        self.assertEqual(ups, [""] + TOOL_GROUPS["noise"] + TOOL_GROUPS["lowlight"])
        self.assertEqual(len(ups), 13, "双退化必须同时包含两组的全部工具")

    def test_clean_is_raw_only(self):
        ups = self._ups(Diagnosis(noise_score=0.1, lowlight_score=0.1))
        self.assertEqual(ups, [""])

    def test_gray_zone_adds_all_tools(self):
        """灰区（两组都 ≥0.4 且都 < confidence_threshold）沿用当前规则：并行探索全部工具。"""
        ups = self._ups(Diagnosis(noise_score=0.5, lowlight_score=0.5))
        self.assertEqual(ups, [""] + ALL_UPSTREAM_TOOLS)
        self.assertEqual(len(ups), 17)

    def test_high_group_does_not_trigger_gray_for_other_group(self):
        """既有规则：一组明确命中时只加该组，另一组的灰区分数不再触发全量探索。

        这是项目当前行为（本次不改），单独记录以免后续误改：
          noise=0.9 / lowlight=0.5 → raw + 8 去噪（9 个），不含低光组。
        """
        ups = self._ups(Diagnosis(noise_score=0.9, lowlight_score=0.5))
        self.assertEqual(ups, [""] + TOOL_GROUPS["noise"])
        self.assertNotIn("lowlight", ups)


class TestExecutionSemantics(unittest.TestCase):
    """每个修复器独立处理原图；每个候选只分类一次。"""

    def _stub_agent(self):
        agent = _agent()
        seen_restore = []

        def get_upstream(name):
            class _U:
                def restore(self, image):
                    seen_restore.append((name, image.copy()))
                    bumped = np.clip(image.astype(np.int16) + 7, 0, 255).astype(np.uint8)
                    return SimpleNamespace(image=bumped, meta={})
            return _U()

        ds_calls = []

        class _DS:
            task_type = "classify"

            def process(self, image, **kwargs):
                ds_calls.append(image.copy())
                return SimpleNamespace(
                    task_type="classify",
                    result={"glass": 0.60, "paper": 0.30, "trash": 0.10},
                    meta={},
                )

        agent._get_upstream = get_upstream
        agent._get_downstream = lambda name: _DS()
        return agent, seen_restore, ds_calls

    def test_each_restorer_receives_original_image(self):
        agent, seen, _ = self._stub_agent()
        img = np.random.default_rng(0).integers(0, 255, (24, 32, 3), dtype=np.uint8)
        cands = agent._gen_candidates(Diagnosis(noise_score=0.9, lowlight_score=0.9))
        for c in cands:
            agent._execute_pipeline(img, c, {}, None)

        self.assertEqual(len(seen), 12, "8 去噪 + 4 低光，各自调用一次")
        for name, arr in seen:
            np.testing.assert_array_equal(arr, img, err_msg=f"{name} 收到的不是原图")

    def test_each_candidate_classified_exactly_once(self):
        agent, _, ds_calls = self._stub_agent()
        img = np.zeros((16, 16, 3), dtype=np.uint8)
        cands = agent._gen_candidates(Diagnosis(noise_score=0.9, lowlight_score=0.1))
        for c in cands:
            agent._execute_pipeline(img, c, {}, None)
        self.assertEqual(len(ds_calls), len(cands))

    def test_raw_only_classifies_once(self):
        agent, seen, ds_calls = self._stub_agent()
        img = np.zeros((16, 16, 3), dtype=np.uint8)
        cands = agent._gen_candidates(Diagnosis(noise_score=0.05, lowlight_score=0.05))
        self.assertEqual([c.upstream for c in cands], [""])
        for c in cands:
            agent._execute_pipeline(img, c, {}, None)
        self.assertEqual(seen, [], "raw-only 不应调用任何修复器")
        self.assertEqual(len(ds_calls), 1, "原图只分类一次")


class TestSelectionAndFallback(unittest.TestCase):
    """现有验收规则（_select_online）保持不变：不合格则回退 raw。"""

    def setUp(self):
        self.agent = _agent()

    def test_all_restorations_rejected_falls_back_to_raw(self):
        raw = _outcome("raw_cls", "", conf=0.70, margin=0.30, label="glass")
        # 同类但置信度提升不足（delta_conf 默认 0.05）
        weak = _outcome("denoise_cls", "denoise", conf=0.71, margin=0.31, label="glass")
        # 异类但 raw 不确定度足够高（margin 0.30 ≥ tau_uncert），margin 提升不显著
        swapper = _outcome("lowlight_cls", "lowlight", conf=0.90, margin=0.10, label="paper")
        chosen = self.agent._select_online([raw, weak, swapper], raw)
        self.assertIs(chosen, raw, "所有修复分支不合格时必须回退 raw 的既有结果")

    def test_raw_outcome_reused_not_reclassified(self):
        raw = _outcome("raw_cls", "", conf=0.70, margin=0.30, label="glass")
        chosen = self.agent._select_online([raw], raw)
        self.assertIs(chosen, raw, "回退时必须复用已算好的 raw 结果，不能重跑分类")

    def test_same_label_with_gain_is_adopted(self):
        raw = _outcome("raw_cls", "", conf=0.60, margin=0.15, label="glass")
        strong = _outcome("denoise_cls", "denoise", conf=0.75, margin=0.30, label="glass")
        chosen = self.agent._select_online([raw, strong], raw)
        self.assertIs(chosen, strong)

    def test_overexposed_branch_rejected(self):
        raw = _outcome("raw_cls", "", conf=0.60, margin=0.15, label="glass")
        blown = _outcome("lowlight_cls", "lowlight", conf=0.95, margin=0.50,
                         label="glass", clip=0.5)
        chosen = self.agent._select_online([raw, blown], raw)
        self.assertIs(chosen, raw, "过曝分支应被 clip_max 约束拒绝")


class TestOfflineIsolation(unittest.TestCase):
    """真实标签与 Oracle 只能用于离线评价，不得进入在线选择。"""

    def test_ground_truth_does_not_change_choice(self):
        agent = _agent()
        raw = _outcome("raw_cls", "", conf=0.70, margin=0.30, label="glass")
        weak = _outcome("denoise_cls", "denoise", conf=0.71, margin=0.31, label="glass")
        outcomes = [raw, weak]

        c1, o1, h1, r1 = agent._decide(outcomes, ground_truth="glass")
        c2, o2, h2, r2 = agent._decide(outcomes, ground_truth="trash")
        self.assertEqual(c1.pipeline_id, c2.pipeline_id,
                         "换真实标签不应改变 Agent 的选择")
        self.assertEqual(c1.pipeline_id, "raw_cls")

    def test_global_oracle_does_not_change_choice(self):
        agent = _agent()

        class _DS:
            _backend = "trashnet"
            task_type = "classify"

            def _assert_loaded(self):
                pass

        agent._get_downstream = lambda name: _DS()

        def fake_exec(image, route, task_kwargs, ground_truth=None):
            correct = route.upstream == "denoise"
            return PipelineOutcome(
                pipeline_id=route.pipeline_id, upstream=route.upstream,
                downstream="classify", score=0.6, downstream_confidence=0.60,
                margin=0.20, pred_label="glass", clip_ratio=0.0, fallback=False,
                is_correct=correct, ground_truth_score=1.0 if correct else 0.0,
            )

        agent._execute_pipeline = fake_exec
        # 固定诊断结果（避免依赖图像内容），得到 raw + 8 去噪 = 9 个候选
        agent.diagnoser.diagnose = lambda image: Diagnosis(noise_score=0.9, lowlight_score=0.1)
        img = np.full((32, 32, 3), 60, dtype=np.uint8)

        r_off = agent.run(img, intent="classify", ground_truth="glass", global_oracle=False)
        r_on = agent.run(img, intent="classify", ground_truth="glass", global_oracle=True)

        self.assertEqual(r_off.chosen_pipeline, r_on.chosen_pipeline,
                         "global_oracle 只补跑分支算上界，不得改变实际选择")
        self.assertGreater(len(r_on.all_outcomes), len(r_off.all_outcomes),
                           "global_oracle=True 应补跑更多分支用于上界计算")
        self.assertTrue(r_on.global_oracle_pipeline)


if __name__ == "__main__":
    unittest.main(verbosity=2)
