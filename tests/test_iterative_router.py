"""Behavior tests for feedback-driven tool calls; no restoration weights needed."""
from pathlib import Path
from tempfile import TemporaryDirectory
from types import SimpleNamespace
import unittest

import numpy as np

from src.agents.orchestrator import ImageRestorationAgent, PipelineOutcome
from src.router.diagnose import Diagnosis
from src.router.iterative_router import (IterativeRouter, N_BRANCHES, STATE_DIM,
                                         make_state, choose_action)
from src.router.learned_router import FEATURE_DIM
from src.router.tool_registry import BRANCH_ORDER
from src.main import load_config


class StateTests(unittest.TestCase):
    def state(self, observed=(0,), budget=4, remaining=4):
        full = np.zeros(FEATURE_DIM, np.float32)
        full[6:].reshape(N_BRANCHES, 5)[:, 0] = 1
        return make_state(full, np.ones(N_BRANCHES, bool),
                          np.isin(np.arange(N_BRANCHES), observed), remaining, budget)

    def test_unseen_feedback_never_enters_network(self):
        full = np.arange(FEATURE_DIM, dtype=np.float32)
        available = np.ones(N_BRANCHES, bool)
        observed = np.zeros(N_BRANCHES, bool)
        observed[0] = True
        state1 = make_state(full, available, observed, 4, 4)
        full[11:] = -10000  # Arbitrarily change all future tool outcomes.
        state2 = make_state(full, available, observed, 4, 4)
        np.testing.assert_array_equal(state1, state2)
        self.assertEqual(state1.shape, (STATE_DIM,))

    def test_probability_rank_selects_next_tool_not_registry_order(self):
        p = np.full(N_BRANCHES, 0.1)
        p[0], p[3] = 0.4, 0.9
        action = choose_action(p, self.state(), 0.01)
        self.assertEqual((action.kind, action.upstream), ("call", BRANCH_ORDER[3]))

    def test_budget_and_ties_stop(self):
        p = np.full(N_BRANCHES, 0.5)
        self.assertEqual(choose_action(p, self.state(), 0).kind, "stop")
        p[2] = 0.99
        a = choose_action(p, self.state(remaining=0), 0)
        self.assertEqual((a.kind, a.reason), ("stop", "budget_exhausted"))

    def test_overexposed_and_failed_results_cannot_be_returned(self):
        s = self.state(observed=(0, 1, 2), remaining=0)
        b = s[6:FEATURE_DIM].reshape(N_BRANCHES, 5)
        b[1, 3], b[2, 4] = 0.9, 1
        p = np.full(N_BRANCHES, 0.1)
        p[1:3] = 0.99
        self.assertEqual(choose_action(p, s, 0).upstream, "")

    def test_invalid_state_and_probabilities_rejected(self):
        with self.assertRaises(ValueError):
            choose_action(np.full(N_BRANCHES, np.nan), self.state(), 0)
        with self.assertRaises(ValueError):
            make_state(np.zeros(FEATURE_DIM), np.ones(N_BRANCHES),
                       np.zeros(N_BRANCHES), 4, 4)


def feedback_agent(first_conf, budget=4):
    agent = ImageRestorationAgent({"model_toolbox": {"enabled": True},
                                   "llm": {"enabled": False}}, device="cpu")
    router = IterativeRouter.__new__(IterativeRouter)
    router.budget, router.call_cost, router.clip_max = budget, 0.01, 0.05

    def probabilities(state):
        bf = state[6:FEATURE_DIM].reshape(N_BRANCHES, 5)
        p = np.full(N_BRANCHES, 0.1)
        p[0] = 0.4
        if not bf[1, 0]:
            p[1], p[2] = 0.8, 0.6
        else:
            # Test double: the very same policy changes action with tool feedback.
            p[1] = bf[1, 1]
            p[2] = bf[2, 1] if bf[2, 0] else 0.7
        return p

    router.probabilities = probabilities
    agent._iterative_router = router
    agent.routing_mode = "iterative"
    seen = []

    def execute(image, route, kwargs, gt):
        seen.append((route.upstream, image.copy()))
        confidence = {"": 0.4, "denoise": first_conf, "ffdnet": 0.96}.get(route.upstream, 0.1)
        return PipelineOutcome(pipeline_id=route.pipeline_id, upstream=route.upstream,
                               downstream="classify", pred_label="glass", score=confidence,
                               downstream_confidence=confidence, margin=confidence/2,
                               ground_truth_score=float(gt == "glass"))
    agent._execute_pipeline = execute
    return agent, seen


class LoopTests(unittest.TestCase):
    diagnosis = Diagnosis(noise_score=0.9, lowlight_score=0.1)

    def run_loop(self, conf, gt=None, budget=4, audit=False):
        agent, seen = feedback_agent(conf, budget)
        image = np.full((8, 9, 3), 7, dtype=np.uint8)
        result = agent._run_iterative(image, self.diagnosis, {}, gt, audit)
        return result, seen, image

    def test_good_feedback_stops_bad_feedback_calls_another_tool(self):
        good, seen_good, _ = self.run_loop(0.95)
        bad, seen_bad, _ = self.run_loop(0.1)
        self.assertEqual([x[0] for x in seen_good], ["", "denoise"])
        self.assertEqual([x[0] for x in seen_bad], ["", "denoise", "ffdnet"])
        self.assertEqual(good.selected_upstream, "denoise")
        self.assertEqual(bad.selected_upstream, "ffdnet")
        self.assertEqual([t["action"] for t in bad.routing_trace], ["call", "call", "stop"])

    def test_no_repeat_no_composition_and_no_extra_classification(self):
        result, seen, image = self.run_loop(0.1)
        self.assertEqual(result.restore_calls, 2)
        self.assertEqual(len(seen), 1 + result.restore_calls)
        self.assertEqual(len(set(name for name, _ in seen)), len(seen))
        for _, received in seen:
            np.testing.assert_array_equal(received, image)

    def test_budget_failure_returns_already_computed_raw(self):
        result, seen, _ = self.run_loop(0.1, budget=1)
        self.assertEqual(result.selected_upstream, "")
        self.assertEqual(result.routing_trace[-1]["reason"], "budget_exhausted")
        self.assertEqual([n for n, _ in seen].count(""), 1)

    def test_truth_changes_metrics_but_not_actions(self):
        a, _, _ = self.run_loop(0.1, "glass")
        b, _, _ = self.run_loop(0.1, "paper")
        self.assertEqual(a.routing_trace, b.routing_trace)
        self.assertEqual(a.chosen_pipeline, b.chosen_pipeline)
        self.assertTrue(a.outcomes[-1].ground_truth_score)
        self.assertFalse(b.outcomes[-1].ground_truth_score)

    def test_global_oracle_audit_cannot_change_loop(self):
        plain, _, _ = self.run_loop(0.1, "glass")
        audit, seen, _ = self.run_loop(0.1, "glass", audit=True)
        self.assertEqual(plain.routing_trace, audit.routing_trace)
        self.assertEqual(plain.chosen_pipeline, audit.chosen_pipeline)
        self.assertEqual(len(audit.outcomes), 3)
        self.assertEqual(len(seen), N_BRANCHES)
        self.assertEqual(plain.oracle_scope, "executed_only")
        self.assertEqual(audit.oracle_scope, "admitted_candidates")
        audit.to_dict()  # Trace must serialize without tool objects or label leakage.

    def test_loop_never_invokes_rule_selector_or_llm(self):
        agent, _ = feedback_agent(0.95)
        def forbidden(*a, **kw):
            raise AssertionError("Rule/LLM must not govern iterative decisions")
        agent._select_online = forbidden
        agent.llm.explain = forbidden
        agent._run_iterative(np.zeros((8, 8, 3), np.uint8), self.diagnosis, {}, None)


class ConfigTests(unittest.TestCase):
    def test_cli_profile_uses_learned_loop_and_frozen_classifier(self):
        cfg = load_config("config/iterative_agent_13tool.yaml")
        self.assertEqual(cfg["routing"]["mode"], "iterative")
        self.assertTrue(cfg["model_toolbox"]["enabled"])
        self.assertEqual(cfg["downstreams"]["classify"]["weight_path"], "weights/trashnet_resnet50_seed1.pth")

    def test_missing_checkpoint_fails_instead_of_rule_fallback(self):
        with self.assertRaisesRegex(ValueError, "不会回退规则"):
            ImageRestorationAgent({"routing": {"mode": "iterative"}}, "cpu")

    def test_old_checkpoint_is_rejected(self):
        import torch
        with TemporaryDirectory() as tmp:
            path = str(Path(tmp) / "old.pth")
            torch.save({"feature_dim": FEATURE_DIM, "branch_order": BRANCH_ORDER}, path)
            with self.assertRaisesRegex(ValueError, "partial-feedback"):
                IterativeRouter(path)


if __name__ == "__main__":
    unittest.main()
