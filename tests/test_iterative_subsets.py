"""Checkpoint-specific layouts and the 13-branch entry point; frozen tools stay idle."""
from copy import deepcopy
from pathlib import Path
from tempfile import TemporaryDirectory
import unittest

import numpy as np
import torch

from src.agents.orchestrator import ImageRestorationAgent, PipelineOutcome
from src.main import DEFAULT_CONFIG, load_config
from src.router import iterative_router as ir
from src.router._router_net import RouterNet
from src.router.diagnose import Diagnosis
from src.router.learned_router import extract_features


class SubsetTests(unittest.TestCase):
    def setUp(self):
        self.tmp = TemporaryDirectory()
        self.addCleanup(self.tmp.cleanup)
        self.addCleanup(ir.configure_branches)
        ir.configure_branches()
        self.root = Path(self.tmp.name)
        self.classifier = self.root / "frozen_classifier.pth"
        self.classifier.write_bytes(b"pairing fixture; never used for classification")
        self.paths = {}
        for n in (13, 17):
            model = RouterNet(7 + 6 * n, hidden=16, num_branches=n)
            ck = {"schema": ir.SCHEMA, "feature_dim": 7 + 6 * n,
                  "branch_order": ir.FULL_BRANCH_ORDER[:n], "hidden": 16,
                  "output_activation": "sigmoid", "state_dict": model.state_dict(),
                  "max_calls": 4, "call_cost": 0.01, "clip_max": 0.05,
                  "diag_mean": [0.] * 6, "diag_std": [1.] * 6,
                  "classifier_sha256": ir.file_hash(self.classifier)}
            self.paths[n] = self.root / f"router{n}.pth"
            torch.save(ck, self.paths[n])
        self.cfg = {
            "routing": {"mode": "iterative"}, "router": {"learned_weight": ""},
            "model_toolbox": {"enabled": True, "exclude": ir.FULL_BRANCH_ORDER[13:]},
            "iterative_router": {"weight_path": str(self.paths[13])},
            "downstreams": {"classify": {"weight_path": str(self.classifier)}},
            "llm": {"enabled": False},
        }

    def router(self, n):
        return ir.IterativeRouter(self.paths[n], classifier_weight=self.classifier)

    def test_default_profile_pairs_seed1_and_excludes_exactly_four_tools(self):
        cfg = load_config(DEFAULT_CONFIG)
        self.assertEqual(DEFAULT_CONFIG, "config/iterative_agent_13tool.yaml")
        self.assertEqual(set(cfg["model_toolbox"]["exclude"]), set(ir.FULL_BRANCH_ORDER[13:]))
        self.assertEqual(cfg["downstreams"]["classify"]["weight_path"],
                         "weights/trashnet_resnet50_seed1.pth")
        self.assertEqual(cfg["iterative_router"]["weight_path"],
                         "weights/router_iterative_feedback_13tool_seed1.pth")

    def test_subset_features_equal_exact_full_cache_prefix(self):
        diag = Diagnosis(noise_score=0.7, lowlight_score=0.2)
        outcomes = [PipelineOutcome(pipeline_id=f"{up}_cls", upstream=up,
                                   downstream="classify", downstream_confidence=i / 20)
                    for i, up in enumerate(ir.FULL_BRANCH_ORDER)]
        full = extract_features(diag, outcomes)
        np.testing.assert_array_equal(ir.extract_features_for(diag, outcomes, ir.FULL_BRANCH_ORDER), full)
        np.testing.assert_array_equal(ir.extract_features_for(diag, outcomes, ir.FULL_BRANCH_ORDER[:13]),
                                      full[:71])

    def test_13_state_rebuilds_availability_instead_of_truncating_old_state(self):
        features = np.arange(91, dtype=np.float32)
        observed = np.zeros(17, bool)
        observed[0] = True
        full = ir.make_state(features, np.ones(17, bool), observed, 4, 4,
                             branch_order=ir.FULL_BRANCH_ORDER)
        small = ir.make_state(features[:71], np.ones(13, bool), observed[:13], 4, 4,
                              branch_order=ir.FULL_BRANCH_ORDER[:13])
        self.assertEqual(small.shape, (85,))
        np.testing.assert_array_equal(small[71:84], np.ones(13))
        self.assertFalse(np.array_equal(full[:85], small))

    def test_full_and_subset_instances_survive_global_training_layout_switch(self):
        routers = [self.router(17), self.router(13)]
        diag = Diagnosis(noise_score=0.5, lowlight_score=0.5)
        raw = PipelineOutcome(pipeline_id="raw_cls", upstream="", downstream="classify",
                              downstream_confidence=0.4)
        before = [r.decide(diag, [raw], set(r.branch_order)) for r in routers]
        ir.configure_branches(ir.FULL_BRANCH_ORDER[13:])
        after = [r.decide(diag, [raw], set(r.branch_order)) for r in routers]
        self.assertEqual(before, after)
        self.assertEqual([r.state_dim for r in routers], [109, 85])

    def test_state_from_other_layout_is_rejected(self):
        with self.assertRaisesRegex(ValueError, "state shape"):
            self.router(13).probabilities(np.zeros(109, np.float32))

    def test_duplicate_or_dimension_mismatched_checkpoint_is_rejected(self):
        original = torch.load(self.paths[13], weights_only=False)
        for field, value in (("branch_order", [""] * 13), ("feature_dim", 109)):
            ck = dict(original)
            ck[field] = value
            bad = self.root / "bad.pth"
            torch.save(ck, bad)
            with self.assertRaises(ValueError):
                ir.IterativeRouter(bad)

    def test_agent_loads_without_global_setup_and_enforces_pool_pairing(self):
        agent = ImageRestorationAgent(deepcopy(self.cfg), "cpu")
        self.assertEqual(len(agent._iterative_router.branch_order), 13)
        cfg = deepcopy(self.cfg)
        cfg["model_toolbox"]["exclude"] = []
        with self.assertRaisesRegex(ValueError, "model_toolbox.exclude"):
            ImageRestorationAgent(cfg, "cpu")
        cfg = deepcopy(self.cfg)
        other = self.root / "other.pth"
        other.write_bytes(b"different classifier")
        cfg["downstreams"]["classify"]["weight_path"] = str(other)
        with self.assertRaisesRegex(ValueError, "Classifier does not match"):
            ImageRestorationAgent(cfg, "cpu")

    def test_gray_loop_only_calls_retained_tools_and_preserves_original_input(self):
        agent = ImageRestorationAgent(deepcopy(self.cfg), "cpu")
        router = agent._iterative_router

        def probabilities(state):
            observed = state[6:71].reshape(13, 5)[:, 0] > 0.5
            p = np.full(13, 0.1)
            p[0] = 0.4
            p[1] = 0.95 if observed[1] else 0.9
            return p

        router.probabilities = probabilities
        image = np.full((8, 9, 3), 7, np.uint8)
        seen = []

        def execute(received, route, kwargs, gt):
            np.testing.assert_array_equal(received, image)
            seen.append(route.upstream)
            return PipelineOutcome(pipeline_id=route.pipeline_id, upstream=route.upstream,
                                   downstream="classify", downstream_confidence=0.8)

        agent._execute_pipeline = execute
        result = agent._run_iterative(image, Diagnosis(noise_score=0.5, lowlight_score=0.5),
                                      {}, None, False)
        self.assertEqual(seen, ["", "denoise"])
        self.assertEqual(result.selected_upstream, "denoise")
        self.assertEqual(len(result.routed_upstreams), 13)

    def test_exclusions_override_explicit_admission_and_reject_typos(self):
        cfg = deepcopy(self.cfg)
        cfg["routing"]["always_admit_groups"] = ["deblur"]
        agent = ImageRestorationAgent(cfg, "cpu")
        candidates = agent._gen_candidates(Diagnosis(noise_score=0.1, lowlight_score=0.1))
        self.assertEqual([r.upstream for r in candidates], [""])
        cfg = deepcopy(self.cfg)
        cfg["model_toolbox"]["exclude"] = ["typo_tool"]
        with self.assertRaisesRegex(ValueError, "未知排除工具"):
            ImageRestorationAgent(cfg, "cpu")

    def test_13_branch_feedback_ablation_preserves_real_safety_values(self):
        from scripts.ablate_iterative_router import custom_decide

        router = self.router(13)
        seen = []

        def probabilities(state):
            seen.append(state.copy())
            p = np.full(13, 0.1)
            p[0], p[1] = 0.4, 0.99
            return p

        router.probabilities = probabilities
        outcomes = [PipelineOutcome(pipeline_id="raw_cls", upstream="", downstream="classify"),
                    PipelineOutcome(pipeline_id="denoise_cls", upstream="denoise", downstream="classify",
                                    downstream_confidence=0.9, clip_ratio=0.9, fallback=True)]
        action = custom_decide(router, Diagnosis(), outcomes, {"", "denoise"}, mask_feedback=True)
        np.testing.assert_array_equal(seen[0][12:16], np.zeros(4))
        self.assertEqual((action.kind, action.upstream), ("stop", ""))


if __name__ == "__main__":
    unittest.main()
