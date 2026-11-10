"""validate_final.py 的 fail-closed 校验逻辑单元测试。

验证两个关键 fail-closed 行为：
1. 快照资产（asset_hash_valid）：快照 sha 为空 / 磁盘文件缺失 / 哈希不匹配时都必须失败，
   不允许 None==None 误通过。
2. eval_manifest（eval_manifest_valid）：path/sha 缺失、path 与期望不一致、sha 与磁盘
   不一致时都必须失败。
"""
from __future__ import annotations

import hashlib
import os
import sys
import tempfile
import unittest

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from scripts.validate_final import asset_hash_valid, eval_manifest_valid, sha256_file


def _write(path, content=b"hello"):
    os.makedirs(os.path.dirname(path), exist_ok=True)
    with open(path, "wb") as f:
        f.write(content)


def _sha(content=b"hello"):
    return hashlib.sha256(content).hexdigest()


class TestAssetHashValid(unittest.TestCase):
    def setUp(self):
        self.tmp = tempfile.mkdtemp()
        self.disk = os.path.join(self.tmp, "asset.pth")
        _write(self.disk, b"asset-content")

    def test_missing_snapshot_sha_fails(self):
        # 快照 sha 为空（None 或 ""）必须失败，不能 None==None 通过
        self.assertFalse(asset_hash_valid(None, self.disk))
        self.assertFalse(asset_hash_valid("", self.disk))

    def test_missing_disk_file_fails(self):
        # 磁盘文件缺失必须失败，即使快照 sha 有值
        self.assertFalse(asset_hash_valid("abc123", os.path.join(self.tmp, "nope.pth")))

    def test_hash_mismatch_fails(self):
        self.assertFalse(asset_hash_valid("deadbeef", self.disk))

    def test_match_passes(self):
        self.assertTrue(asset_hash_valid(sha256_file(self.disk), self.disk))


class TestEvalManifestValid(unittest.TestCase):
    def setUp(self):
        self.tmp = tempfile.mkdtemp()
        self.manifest = os.path.join(self.tmp, "manifest_eval.tsv")
        _write(self.manifest, b"degraded\tlabel\tsid\tsrc\tparams\n")
        self.sha = sha256_file(self.manifest)

    def test_missing_fields_fails(self):
        self.assertFalse(eval_manifest_valid(None, self.manifest))
        self.assertFalse(eval_manifest_valid({}, self.manifest))
        self.assertFalse(eval_manifest_valid({"path": self.manifest}, self.manifest))  # 缺 sha
        self.assertFalse(eval_manifest_valid({"sha256": self.sha}, self.manifest))     # 缺 path

    def test_path_mismatch_fails(self):
        self.assertFalse(eval_manifest_valid(
            {"path": os.path.join(self.tmp, "other.tsv"), "sha256": self.sha},
            self.manifest))

    def test_sha_mismatch_fails(self):
        self.assertFalse(eval_manifest_valid(
            {"path": self.manifest, "sha256": "deadbeef"}, self.manifest))

    def test_match_passes(self):
        self.assertTrue(eval_manifest_valid(
            {"path": self.manifest, "sha256": self.sha}, self.manifest))


if __name__ == "__main__":
    unittest.main()
