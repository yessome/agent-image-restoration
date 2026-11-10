"""Optional deblur availability; no trained tool execution or test-set reads."""
from pathlib import Path
import unittest

import numpy as np

from src.agents.orchestrator import ImageRestorationAgent, PipelineOutcome
from src.main import load_config
from src.router.diagnose import Diagnosis
from src.router.iterative_router import IterativeRouter, N_BRANCHES, STATE_DIM
from src.router.learned_router import FEATURE_DIM
from src.router.tool_registry import BRANCH_ORDER, TOOL_GROUPS, ALL_UPSTREAM_TOOLS


def make_agent(groups=("deblur",), toolbox=True):
    return ImageRestorationAgent({
        "routing": {"always_admit_groups": list(groups)},
        "model_toolbox": {"enabled": toolbox},
        "confidence_threshold": 0.6,
        "llm": {"enabled": False},
    }, device="cpu")


class AdmissionTests(unittest.TestCase):
    def names(self, noise=0.1, lowlight=0.1, groups=("deblur",)):
        return [r.upstream for r in make_agent(groups)._gen_candidates(
            Diagnosis(noise_score=noise, lowlight_score=lowlight))]

    def test_no_signal_still_admits_two_deblur_tools(self):
        self.assertEqual(self.names(), [""] + TOOL_GROUPS["deblur"])

    def test_default_admission_is_unchanged(self):
        self.assertEqual(self.names(groups=()), [""])
        self.assertEqual(self.names(0.9, groups=()), [""] + TOOL_GROUPS["noise"])

    def test_noise_and_deblur_both_remain_available(self):
        self.assertEqual(set(self.names(0.9)),
                         {"", *TOOL_GROUPS["noise"], *TOOL_GROUPS["deblur"]})
        self.assertEqual(len(self.names(0.9)), 11)

    def test_lowlight_and_deblur_both_remain_available(self):
        self.assertEqual(set(self.names(lowlight=0.9)),
                         {"", *TOOL_GROUPS["lowlight"], *TOOL_GROUPS["deblur"]})
        self.assertEqual(len(self.names(lowlight=0.9)), 7)

    def test_both_diagnoses_and_deblur(self):
        self.assertEqual(len(self.names(0.9, 0.9)), 15)

    def test_gray_candidates_are_complete_without_duplicates(self):
        names = self.names(0.5, 0.5)
        self.assertEqual(set(names), {"", *ALL_UPSTREAM_TOOLS})
        self.assertEqual(len(names), 17)

    def test_repeated_group_names_do_not_repeat_routes(self):
        self.assertEqual(self.names(groups=("deblur", "deblur")), self.names())

    def test_group_configuration_is_validated(self):
        for value in ("deblur", ["unknown"], [1], None):
            with self.subTest(value=value), self.assertRaises(ValueError):
                ImageRestorationAgent({
                    "routing": {"always_admit_groups": value},
                    "model_toolbox": {"enabled": True},
                }, device="cpu")
        with self.assertRaisesRegex(ValueError, "model_toolbox"):
            make_agent(toolbox=False)

    def test_final_profile_excludes_experimental_deblur_branches(self):
        root = Path(__file__).resolve().parents[1]
        cfg = load_config(str(root / "config/iterative_agent_13tool.yaml"))
        self.assertEqual(cfg["routing"]["mode"], "iterative")
        self.assertEqual(cfg["routing"].get("always_admit_groups", []), [])
        self.assertEqual(set(cfg["model_toolbox"]["exclude"]),
                         {"restormer_defocus", "mprnet_deblur", "realesrgan_x2", "realesrgan_x4"})
        self.assertEqual(cfg["iterative_router"]["weight_path"],
                         "weights/router_iterative_feedback_13tool_seed1.pth")
        self.assertFalse(cfg["llm"]["enabled"])
        self.assertEqual(STATE_DIM, 109)  # Full registry stays available for checkpoint validation.


class LearnedLoopTests(unittest.TestCase):
    def run_loop(self, repair_confidence=0.95, want_repair=True, truth=None):
        agent = make_agent()
        router = IterativeRouter.__new__(IterativeRouter)
        router.budget, router.call_cost, router.clip_max = 4, 0.01, 0.05
        first, second = TOOL_GROUPS["deblur"]
        i, j = BRANCH_ORDER.index(first), BRANCH_ORDER.index(second)
        seen = []

        def probabilities(state):
            branch = state[6:FEATURE_DIM].reshape(N_BRANCHES, 5)
            p = np.full(N_BRANCHES, 0.05)
            p[0] = 0.6
            if want_repair:
                p[i] = branch[i, 1] if branch[i, 0] else 0.9
                p[j] = branch[j, 1] if branch[j, 0] else 0.8
            return p

        def execute(image, route, kwargs, gt):
            seen.append((route.upstream, image.copy()))
            confidence = 0.6 if not route.upstream else (
                repair_confidence if route.upstream == first else 0.95)
            return PipelineOutcome(
                pipeline_id=route.pipeline_id, upstream=route.upstream,
                downstream="classify", pred_label="glass", score=confidence,
                downstream_confidence=confidence, margin=confidence / 2,
                ground_truth_score=float(gt == "glass"))

        def forbidden(*args, **kwargs):
            raise AssertionError("Rule/LLM must not decide the iterative result")

        router.probabilities = probabilities
        agent.routing_mode, agent._iterative_router = "iterative", router
        agent._execute_pipeline = execute
        agent._select_online = forbidden
        agent.llm.explain = forbidden
        image = np.full((16, 18, 3), 90, dtype=np.uint8)
        result = agent._run_iterative(image, Diagnosis(noise_score=0.1,
                                      lowlight_score=0.1), {}, truth)
        return result, seen, image

    def test_admission_does_not_force_a_repair(self):
        result, seen, _ = self.run_loop(want_repair=False)
        self.assertEqual([up for up, _ in seen], [""])
        self.assertEqual(result.restore_calls, 0)
        self.assertEqual(result.selected_upstream, "")
        self.assertEqual(result.to_dict()["routed_upstreams"],
                         [""] + TOOL_GROUPS["deblur"])

    def test_learned_router_can_call_deblur_then_stop(self):
        result, seen, image = self.run_loop()
        self.assertEqual([up for up, _ in seen], ["", TOOL_GROUPS["deblur"][0]])
        self.assertEqual(result.restore_calls, 1)
        self.assertEqual([e["action"] for e in result.routing_trace], ["call", "stop"])
        for _, received in seen:
            np.testing.assert_array_equal(received, image)

    def test_bad_feedback_can_request_the_other_deblur_tool(self):
        result, seen, image = self.run_loop(repair_confidence=0.1)
        self.assertEqual([up for up, _ in seen], [""] + TOOL_GROUPS["deblur"])
        self.assertEqual(result.restore_calls, 2)
        self.assertEqual(result.selected_upstream, TOOL_GROUPS["deblur"][1])
        for _, received in seen:
            np.testing.assert_array_equal(received, image)

    def test_ground_truth_cannot_change_deblur_actions(self):
        a, _, _ = self.run_loop(truth="glass")
        b, _, _ = self.run_loop(truth="paper")
        self.assertEqual(a.routing_trace, b.routing_trace)
        self.assertEqual(a.chosen_pipeline, b.chosen_pipeline)


if __name__ == "__main__":
    unittest.main()
