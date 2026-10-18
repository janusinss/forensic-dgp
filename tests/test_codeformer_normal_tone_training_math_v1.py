"""Independent CPU mathematics for the finite, checkpoint-producing hypothesis.

These tests use NumPy and toy scalars only. They neither import the training
implementation nor construct a neural model, framework, optimizer or VM task.
"""
import ast
from pathlib import Path
import unittest

import numpy as np


WEIGHTS = np.asarray([.2126, .7152, .0722], dtype=np.float64).reshape(1, 3, 1, 1)


def loss(error):
    value = np.asarray(error, dtype=np.float64)
    mu = (value * WEIGHTS).sum(axis=1, keepdims=True).mean(axis=(2, 3), keepdims=True)
    return float(np.abs(value - mu).mean())


def derivative(error):
    value = np.asarray(error, dtype=np.float64)
    batch, _, height, width = value.shape
    pixels = height * width
    mu = (value * WEIGHTS).sum(axis=1, keepdims=True).mean(axis=(2, 3), keepdims=True)
    signs = np.sign(value - mu)
    return (signs / (batch * 3 * pixels) -
            WEIGHTS * signs.mean(axis=(1, 2, 3), keepdims=True) / (batch * pixels))


def asymmetric():
    return np.asarray([.25, -.4, .15, .31, -.2, .1, .35, -.12,
                       .42, -.18, .08, -.27], dtype=np.float32).reshape(1, 3, 2, 2)


def chroma():
    return np.broadcast_to(np.asarray([.3, -.1, .05], dtype=np.float32)
                           .reshape(1, 3, 1, 1), (1, 3, 2, 2)).copy()


class IndependentTrainingMath(unittest.TestCase):
    def finite_difference(self, value):
        result = np.empty_like(value, dtype=np.float64)
        value = value.astype(np.float64)
        epsilon = 1e-6
        for index in np.ndindex(value.shape):
            plus, minus = value.copy(), value.copy()
            plus[index] += epsilon
            minus[index] -= epsilon
            result[index] = (loss(plus) - loss(minus)) / (2 * epsilon)
        return result

    def test_asymmetric_prerequisite_analytic_gradient(self):
        np.testing.assert_allclose(derivative(asymmetric()), self.finite_difference(asymmetric()),
                                   rtol=1e-8, atol=1e-10)

    def test_chroma_prerequisite_analytic_gradient(self):
        np.testing.assert_allclose(derivative(chroma()), self.finite_difference(chroma()),
                                   rtol=1e-8, atol=1e-10)

    def test_both_prerequisites_avoid_L1_kinks(self):
        for value in (asymmetric(), chroma()):
            mu = (value.astype(np.float64) * WEIGHTS).sum(axis=1).mean()
            self.assertGreater(float(np.abs(value - mu).min()), .001)

    def test_differentiated_mean_rejects_detached_mean(self):
        value = chroma().astype(np.float64)
        mu = (value * WEIGHTS).sum(axis=1).mean()
        detached = np.sign(value - mu) / value.size
        correct = derivative(value)
        self.assertGreater(float(np.abs(detached - correct).max()), .01)
        self.assertGreater(abs(float(detached.sum())), .3)
        self.assertAlmostEqual(float(correct.sum()), 0., places=14)

    def test_uniform_RGB_shift_invariant_and_direction_zero(self):
        for value in (asymmetric(), chroma()):
            for offset in (-.75, .125, 1.25):
                self.assertAlmostEqual(loss(value), loss(value.astype(np.float64) + offset), places=13)
            self.assertAlmostEqual(float(derivative(value).sum()), 0., places=14)

    def test_uniform_chroma_remains_constrained(self):
        value = chroma().astype(np.float64)
        self.assertGreater(loss(value), .01)
        self.assertEqual(float(np.abs(value - value.mean(axis=(2, 3), keepdims=True)).mean()), 0.)
        self.assertGreater(float(np.abs(derivative(value)).max()), .01)

    def test_per_observation_mean_and_gradient_batch_scaling(self):
        first, second = asymmetric().astype(np.float64), chroma().astype(np.float64) + .7
        combined = np.concatenate((first, second))
        self.assertAlmostEqual(loss(combined), (loss(first) + loss(second)) / 2, places=14)
        np.testing.assert_allclose(derivative(combined)[0], derivative(first)[0] / 2, atol=1e-16)
        np.testing.assert_allclose(derivative(combined)[1], derivative(second)[0] / 2, atol=1e-16)
        wrong_mu = (combined * WEIGHTS).sum(axis=1).mean()
        self.assertGreater(abs(float(np.abs(combined - wrong_mu).mean()) - loss(combined)), .01)

    def test_native_pool_and_native_minus1_plus1_chain(self):
        pooled_rgb = np.arange(3 * 16 * 16, dtype=np.float64).reshape(1, 3, 16, 16) / 256
        rgb = np.repeat(np.repeat(pooled_rgb, 32, axis=2), 32, axis=3)
        native = rgb * 2 - 1
        reconstructed_rgb = (native + 1) / 2
        pooled = reconstructed_rgb.reshape(1, 3, 16, 32, 16, 32).mean(axis=(3, 5))
        np.testing.assert_array_equal(pooled, pooled_rgb)
        self.assertGreater(float(rgb.max()), 1.)
        self.assertNotEqual(loss(pooled), loss(np.clip(pooled, 0, 1)))
        # For every pixel in a32x32 block, d(pool((native+1)/2))/d(native)=1/2048.
        self.assertEqual(.5 / (32 * 32), 1 / 2048)

    def test_training_weight4_divisor4_preserves_all_four_slots(self):
        normal_a, dark_a, normal_b, dark_b = asymmetric(), chroma(), chroma(), asymmetric()
        chosen = [loss(normal_a), float(np.abs(dark_a).mean()),
                  loss(normal_b), float(np.abs(dark_b).mean())]
        total = sum(4 * term / 4 for term in chosen)
        self.assertAlmostEqual(total, sum(chosen), places=14)
        self.assertNotAlmostEqual(total, sum(4 * term / 16 for term in chosen), places=6)
        self.assertNotAlmostEqual(total, sum(4 * term / 2 for term in chosen), places=6)

    def test_centered_L1_not_an_additive_DC_decomposition(self):
        value = np.broadcast_to(np.asarray([.15, -.1, .08], dtype=np.float64)
                                .reshape(1, 3, 1, 1), (1, 3, 2, 2))
        mu = float((value * WEIGHTS).sum(axis=1).mean())
        self.assertGreater(abs(float(np.abs(value).mean()) - (abs(mu) + loss(value))), .01)

    def test_new_call_census_has_two_nonneural_queries(self):
        eligible = sum(2 + (2 if (update - 1) % 4 >= 2 else 0) for update in range(1, 51))
        self.assertEqual(eligible, 148)
        forwards = 200 + eligible + 50 + 391 + 2 * 406 + 4 + 4 + 1 + 2 + 2
        self.assertEqual(forwards, 1614)
        self.assertEqual(1 + 8 + 2, 11)
        self.assertEqual(forwards + 11 + 200 + 50 + 200, 2075)

    def test_initial_disk_floor_covers_caps_and_checkpoint_temp(self):
        payload, archive, temporary, reserve = 3758096384, 3865470566, 1024**3, 4 * 1024**3
        self.assertLess(payload + archive + temporary + reserve, 16 * 1024**3)

    def test_fixtures_import_no_training_or_neural_modules(self):
        tree = ast.parse(Path(__file__).read_text(encoding='utf-8'))
        imports = [item.name for node in ast.walk(tree) if isinstance(node, ast.Import) for item in node.names]
        imports += [node.module or '' for node in ast.walk(tree) if isinstance(node, ast.ImportFrom)]
        self.assertTrue(all(not name.startswith(('research.', 'torch')) for name in imports))


if __name__ == '__main__':
    unittest.main()
