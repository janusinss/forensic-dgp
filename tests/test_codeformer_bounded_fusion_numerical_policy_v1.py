import unittest
import numpy as np
from research.codeformer_bounded_fusion_feasibility_v1.numerical_policy import canary_replay


class CanaryReplayTests(unittest.TestCase):
    def test_zero_original_delta(self):
        base = np.full((1, 256, 32, 32), .25, dtype=np.float32)
        row = canary_replay(base, base.copy(), .25)
        self.assertTrue(np.array_equal(row['used_delta'], np.zeros_like(base)))
        self.assertTrue(np.all(row['tolerance'] > 0))

    def test_gate_error_is_not_rounding(self):
        base = np.ones((1, 256, 64, 64), dtype=np.float32)
        other = base + np.float32(.2)
        expected = canary_replay(base, other, .25)
        wrong = canary_replay(base, other, 1.)
        self.assertTrue(np.all(np.abs(wrong['used_delta'] - expected['used_delta']) > expected['tolerance']))
        self.assertEqual(expected['n'], 4096)

    def test_reference_matches_declared_float32_order(self):
        base = np.linspace(-1, 1, 256 * 32 * 32, dtype=np.float32).reshape(1, 256, 32, 32)
        other = base + np.float32(.013)
        row = canary_replay(base, other, .83)
        scale = np.float32(.1) * np.sqrt(np.square(base).mean((-2, -1), keepdims=True, dtype=np.float32) + np.float32(1e-6))
        reference = (np.tanh((other - base) / scale) * scale) * np.float32(.83)
        self.assertTrue(np.array_equal(row['used_delta'], reference))
        self.assertTrue(np.all(np.abs(row['used_delta']) <= row['scale'] * np.float32(.83)))

    def test_reject_bad_geometry_dtype_and_gate(self):
        good = np.ones((1, 256, 32, 32), dtype=np.float32)
        for base, other, gate in [(good.astype(np.float64), good, 1.),
                                  (good[:, :, :2], good[:, :, :2], 1.),
                                  (good, good, float('nan')), (good, good, True), (good, good, 0.)]:
            with self.assertRaises(ValueError):
                canary_replay(base, other, gate)


if __name__ == '__main__':
    unittest.main()
