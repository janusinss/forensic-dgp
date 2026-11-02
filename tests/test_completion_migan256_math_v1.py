"""Pure NumPy contract checks; no torch import, network construction, or inference."""
import importlib.util
from pathlib import Path
import sys
import unittest

import numpy as np

RUNNER = Path(__file__).resolve().parents[1] / "scripts/run_completion_migan256_v1.py"
SPEC = importlib.util.spec_from_file_location("migan_math_contract_v1", RUNNER)
MODULE = importlib.util.module_from_spec(SPEC)
SPEC.loader.exec_module(MODULE)


class MiganMathContract(unittest.TestCase):
    def test_import_has_no_neural_runtime(self):
        self.assertNotIn("torch", sys.modules)

    def test_mask_polarity_and_channel_order(self):
        image = np.full((256, 256, 3), 128, dtype=np.uint8)
        removal = np.zeros((256, 256), dtype=bool)
        removal[0, 0] = True
        actual = MODULE.numpy_input(image, removal)
        self.assertEqual(actual.shape, (1, 4, 256, 256))
        self.assertEqual(actual.dtype, np.float32)
        np.testing.assert_array_equal(actual[0, :, 0, 0], [-.5, 0, 0, 0])
        self.assertEqual(actual[0, 0, 1, 1], .5)
        expected_rgb = np.float32(128) * np.float32(2) / np.float32(255) - np.float32(1)
        np.testing.assert_array_equal(actual[0, 1:, 1, 1], np.full(3, expected_rgb, dtype=np.float32))

    def test_truncation_clip_and_exact_outside(self):
        image = np.full((256, 256, 3), 132, dtype=np.uint8)
        removal = np.zeros((256, 256), dtype=bool)
        removal[0, 0] = True
        raw = np.full((256, 256, 3), -.5, dtype=np.float32)
        raw[0, 0] = [-2, -.5, 2]
        decoded, completed, final = MODULE.compose(image, removal, raw)
        np.testing.assert_array_equal(final[0, 0], [0, 63, 255])
        np.testing.assert_array_equal(final[~removal], image[~removal])
        np.testing.assert_array_equal(completed[~removal], (image.astype(np.float32) / 255)[~removal])
        np.testing.assert_array_equal(decoded[0, 0], [0, .25, 1])

    def test_nonbinary_and_nonfinite_rejected(self):
        image = np.zeros((256, 256, 3), dtype=np.uint8)
        mask = np.zeros((256, 256), dtype=np.uint8)
        mask[0, 0] = 1
        with self.assertRaises(ValueError):
            MODULE.check_pixels(image, mask)
        raw = np.zeros((256, 256, 3), dtype=np.float32)
        raw[0, 0, 0] = np.nan
        with self.assertRaises(ValueError):
            MODULE.compose(image, mask == 255, raw)


if __name__ == "__main__":
    unittest.main()
