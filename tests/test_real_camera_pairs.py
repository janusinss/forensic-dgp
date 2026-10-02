import unittest

import numpy as np

from real_camera_pairs import camera_pair


class CameraPairContract(unittest.TestCase):
    def sample(self):
        rgb = np.full((64, 64, 3), 96, np.uint8)
        yy, xx = np.indices((64, 48))
        rgb[:, 8:56] = np.stack(((xx * 5) % 256, (yy * 3) % 256, (xx + yy) % 256), axis=-1)
        source = np.zeros((64, 64), bool)
        source[:, 8:56] = True
        valid = source.copy()
        valid[20:26, 20:26] = False  # Unknown observed pixels, not neutral padding.
        target = np.zeros((64, 64), bool)
        target[30:45, 22:39] = True
        return rgb, target, valid, source

    def test_input_only_deterministic_and_preserves_unknown_rgb_context(self):
        sample = self.sample()
        before = [a.copy() for a in sample]
        first, second = (camera_pair(*sample, seed=42) for _ in range(2))
        self.assertTrue(np.array_equal(first["input"], second["input"]))
        self.assertTrue(np.any(first["input"][sample[3]] != sample[0][sample[3]]))
        for actual, original in zip(sample, before):
            self.assertTrue(np.array_equal(actual, original))
        for field, original in zip(("mask", "valid", "source_valid"), sample[1:]):
            self.assertTrue(np.array_equal(first[field], original))
        self.assertTrue(np.all(first["input"][~sample[3]] == 96))
        unknown = sample[3] & ~sample[2]
        self.assertTrue(np.any(first["input"][unknown] != 96))
        self.assertFalse(first["camera"]["target_or_support_changed"])

    def test_positive_outside_valid_and_fabricated_padding_rejected(self):
        rgb, target, valid, source = self.sample()
        target[21, 21] = True
        with self.assertRaises(ValueError):
            camera_pair(rgb, target, valid, source, 42)
        target[21, 21] = False
        rgb[0, 0] = 0
        with self.assertRaises(ValueError):
            camera_pair(rgb, target, valid, source, 42)

    def test_nonrectangular_source_is_not_silently_filled(self):
        rgb, target, valid, source = self.sample()
        rgb[5, 15] = 96
        source[5, 15] = valid[5, 15] = False
        with self.assertRaises(ValueError):
            camera_pair(rgb, target, valid, source, 42)


if __name__ == "__main__":
    unittest.main()
