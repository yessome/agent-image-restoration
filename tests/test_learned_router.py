"""Learned Router 的单元测试（不联网、不加载真实权重）。

注意：分支数已从论文三分支扩展到当前工具集（raw + 16 个修复工具 = 17 分支）。
测试按**分支名**索引构造 17 维向量，避免写死下标；生产代码不做任何兼容性回退。
"""
from __future__ import annotations

import os
import sys
import tempfile
import unittest
from unittest.mock import patch

import numpy as np

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from src.router.learned_router import (
    LearnedRouter, extract_features, BRANCH_ORDER, FEATURE_DIM, N_BRANCHES,
    N_DIAG_FEATURES,
)
from src.router.diagnose import Diagnosis
from src.agents.orchestrator import PipelineOutcome

N = len(BRANCH_ORDER)


def _o(pid, up, conf=0.6, margin=0.2, clip=0.0, fallback=False):
    return PipelineOutcome(
        pipeline_id=pid, upstream=up, downstream="classify",
        downstream_confidence=conf, margin=margin,
        clip_ratio=clip, fallback=fallback,
    )


def _idx(name: str) -> int:
    """按分支名取下标；raw 用 "raw" 表示（对应 upstream=""）。"""
    return BRANCH_ORDER.index("" if name == "raw" else name)


def _vec(**values) -> np.ndarray:
    """构造 17 维向量（未指定的分支为 0），键为分支名或 raw。"""
    v = np.zeros(N, dtype=np.float32)
    for name, val in values.items():
        v[_idx(name)] = val
    return v


class TestFeatures(unittest.TestCase):
    def test_branch_count_and_dim(self):
        self.assertEqual(N_BRANCHES, N)
        self.assertEqual(N, 17, "当前工具集为 raw + 16 工具")
        self.assertEqual(FEATURE_DIM, N_DIAG_FEATURES + N * 5)

    def test_branch_order(self):
        expected = [
            "", "denoise", "ffdnet", "scunet",
            "restormer_real", "restormer_blind",
            "restormer_sigma15", "restormer_sigma25", "restormer_sigma50",
            "lowlight", "zero_dce_plus_plus", "sci_lowlight", "retinexformer",
            "restormer_defocus", "mprnet_deblur",
            "realesrgan_x2", "realesrgan_x4",
        ]
        self.assertEqual(BRANCH_ORDER, expected)

    def test_feature_dim(self):
        d = Diagnosis()
        outs = [_o("raw_cls", ""), _o("denoise_cls", "denoise"),
                _o("lowlight_cls", "lowlight")]
        f = extract_features(d, outs)
        self.assertEqual(f.shape, (FEATURE_DIM,))

    def test_missing_branch_present_zero(self):
        d = Diagnosis()
        outs = [_o("raw_cls", ""), _o("denoise_cls", "denoise")]  # 缺 lowlight
        f = extract_features(d, outs)
        # 每分支 5 维：第 0 维是 present
        self.assertEqual(f[N_DIAG_FEATURES + _idx("lowlight") * 5], 0.0)
        self.assertEqual(f[N_DIAG_FEATURES + _idx("denoise") * 5], 1.0)
        self.assertEqual(f[N_DIAG_FEATURES + _idx("raw") * 5], 1.0)

    def test_normalize(self):
        d = Diagnosis(brightness_mean=255.0, noise_var=0.0)
        f = extract_features(d, [_o("raw_cls", "")])
        self.assertAlmostEqual(f[2], 1.0, places=5)  # brightness_mean/255
        self.assertAlmostEqual(f[3], 0.0, places=5)  # log1p(0)=0


class TestSelectConstraints(unittest.TestCase):
    """select 的安全约束：排除 fallback / 过曝 / 不存在的分支。"""

    def setUp(self):
        self.router = LearnedRouter()  # 无权重，model=None
        self.diag = Diagnosis()

    def _mock_score(self, probs, present):
        def fake(diagnosis, outcomes):
            return (np.asarray(probs, dtype=np.float32),
                    np.asarray(present, dtype=np.float32))
        return fake

    def test_excludes_fallback(self):
        raw = _o("raw_cls", "")
        den = _o("denoise_cls", "denoise", fallback=True)  # denoise 概率最高但 fallback
        outs = [raw, den]
        with patch.object(self.router, "score",
                          self._mock_score(_vec(raw=0.1, denoise=0.9),
                                           _vec(raw=1, denoise=1))):
            pid, _ = self.router.select(self.diag, outs, clip_max=0.10)
        self.assertEqual(pid, "raw_cls")  # denoise 被排除，选 raw

    def test_excludes_clip_ratio(self):
        raw = _o("raw_cls", "")
        low = _o("lowlight_cls", "lowlight", clip=0.5)  # 过曝
        outs = [raw, low]
        with patch.object(self.router, "score",
                          self._mock_score(_vec(raw=0.1, lowlight=0.9),
                                           _vec(raw=1, lowlight=1))):
            pid, _ = self.router.select(self.diag, outs, clip_max=0.10)
        self.assertEqual(pid, "raw_cls")  # lowlight 过曝被排除，选 raw

    def test_excludes_missing_branch(self):
        raw = _o("raw_cls", "")
        outs = [raw]  # 只有 raw
        with patch.object(self.router, "score",
                          self._mock_score(_vec(raw=0.1, denoise=0.9, lowlight=0.5),
                                           _vec(raw=1))):
            pid, _ = self.router.select(self.diag, outs, clip_max=0.10)
        self.assertEqual(pid, "raw_cls")  # 不存在分支 present=0，只能选 raw

    def test_selects_highest_valid(self):
        raw = _o("raw_cls", "")
        den = _o("denoise_cls", "denoise")
        low = _o("lowlight_cls", "lowlight")
        outs = [raw, den, low]
        with patch.object(self.router, "score",
                          self._mock_score(_vec(raw=0.1, denoise=0.9, lowlight=0.5),
                                           _vec(raw=1, denoise=1, lowlight=1))):
            pid, _ = self.router.select(self.diag, outs, clip_max=0.10)
        self.assertEqual(pid, "denoise_cls")

    def test_selects_toolbox_branch_beyond_three(self):
        """17 分支下，非论文三分支（如 restormer_blind）也应能被选中。"""
        raw = _o("raw_cls", "")
        blind = _o("restormer_blind_cls", "restormer_blind")
        outs = [raw, blind]
        with patch.object(self.router, "score",
                          self._mock_score(_vec(raw=0.2, restormer_blind=0.85),
                                           _vec(raw=1, restormer_blind=1))):
            pid, _ = self.router.select(self.diag, outs, clip_max=0.10)
        self.assertEqual(pid, "restormer_blind_cls")


class TestLoadValidation(unittest.TestCase):
    def test_feature_dim_mismatch(self):
        import torch
        ckpt = {"state_dict": {}, "feature_dim": 99, "branch_order": BRANCH_ORDER}
        with tempfile.NamedTemporaryFile(suffix=".pth", delete=False) as f:
            torch.save(ckpt, f.name)
            path = f.name
        router = LearnedRouter()
        with self.assertRaises(ValueError):
            router.load(path)
        os.remove(path)

    def test_branch_order_mismatch(self):
        import torch
        # 3 分支的旧接口必须被拒绝（不允许把生产代码退回三分支）
        ckpt = {"state_dict": {}, "feature_dim": FEATURE_DIM,
                "branch_order": ["", "denoise", "lowlight"]}
        with tempfile.NamedTemporaryFile(suffix=".pth", delete=False) as f:
            torch.save(ckpt, f.name)
            path = f.name
        router = LearnedRouter()
        with self.assertRaises(ValueError):
            router.load(path)
        os.remove(path)

    def test_missing_file(self):
        router = LearnedRouter()
        with self.assertRaises(FileNotFoundError):
            router.load("nonexistent_path.pth")


if __name__ == "__main__":
    unittest.main(verbosity=2)
