from __future__ import annotations

import unittest

import numpy as np

from src.upstream.toolbox import TOOLS, apply_tool
from src.upstream.deepinv_denoise import _tiled_inference


class TestImageToolbox(unittest.TestCase):
    def test_tool_count_and_output_contract(self):
        self.assertGreaterEqual(len(TOOLS), 12)
        image = np.full((32, 40, 3), 96, dtype=np.uint8)
        image[8:24, 10:30] = (180, 70, 30)
        for name in TOOLS:
            with self.subTest(tool=name):
                output = apply_tool(name, image)
                self.assertEqual(output.dtype, np.uint8)
                self.assertEqual(output.ndim, 3)
                self.assertEqual(output.shape[2], 3)
                expected = (64, 80) if name.startswith("upscale-") else (32, 40)
                self.assertEqual(output.shape[:2], expected)

    def test_unknown_tool_is_actionable(self):
        with self.assertRaisesRegex(ValueError, "可用工具"):
            apply_tool("missing", np.zeros((2, 2, 3), dtype=np.uint8))

    def test_tiled_inference_preserves_image_and_shape(self):
        import torch

        for shape in ((3, 31, 47), (3, 270, 285)):
            with self.subTest(shape=shape):
                image = torch.rand((1, *shape))
                output, height, width = _tiled_inference(
                    torch.nn.Identity(), image, "cpu", tile_size=256, overlap=32,
                )
                self.assertEqual((height, width), shape[1:])
                torch.testing.assert_close(output[:, :, :height, :width], image)


if __name__ == "__main__":
    unittest.main()
