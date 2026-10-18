"""Independent NumPy mathematics for a prospective normal-tone diagnostic.

No worker imports, saved research payload, neural framework, autograd, model,
optimizer or VM. Finite differences below test a small algebraic formula only.
"""
import ast
from pathlib import Path
import sys
import unittest

import numpy as np


W = np.asarray([0.2126, 0.7152, 0.0722], dtype=np.float64)
ORDER = ('token_ce', 'latent_mse', 'pixel_l1_rgb', 'lpips_vgg',
         'coarse_rgb_l1', 'behavior_rgb_l1', 'behavior_token_kl')


def centered_scalar(error):
    """HWC reference mathematics; every spatial position belongs to one image."""
    error = np.asarray(error, dtype=np.float64)
    mu = np.mean(error @ W)
    return float(np.mean(np.abs(error - mu)))


def independent_derivative(error):
    """Analytic derivative away from the L1 kink, including the mean's path."""
    error = np.asarray(error, dtype=np.float64)
    pixels = int(np.prod(error.shape[:-1]))
    signs = np.sign(error - np.mean(error @ W))
    return signs / (3 * pixels) - W * np.mean(signs) / pixels


def sequential_fp32(values):
    result = None
    for value in values:
        if value is None:
            continue
        if result is None:
            result = np.asarray(value, dtype=np.float32).copy()
        else:
            np.add(result, np.asarray(value, dtype=np.float32), out=result)
    return result


class NormalToneFormulaFixtures(unittest.TestCase):
    @staticmethod
    def asymmetric_error():
        # Eight positive versus four negative residual coordinates keep the
        # incorrectly detached-mean derivative observably different.
        return np.asarray([.15, -.1, .08]) + np.asarray([
            [[-.02, .01, .015], [.015, -.005, -.01]],
            [[.01, -.015, .005], [-.005, .01, -.015]],
        ])

    def test_uniform_RGB_offset_invariant_but_full_coarse_is_not(self):
        error = self.asymmetric_error()
        for offset in (-.6, .2, 1.4):
            self.assertAlmostEqual(centered_scalar(error), centered_scalar(error + offset), places=13)
        self.assertNotAlmostEqual(float(np.abs(error).mean()), float(np.abs(error + .2).mean()), places=8)

    def test_uniform_chroma_cast_is_retained_while_per_channel_centering_loses_it(self):
        cast = np.asarray([.1, -.1 * W[0] / W[1], 0.])
        self.assertAlmostEqual(float(cast @ W), 0., places=15)
        error = np.broadcast_to(cast, (2, 2, 3)).copy()
        self.assertGreater(centered_scalar(error), .04)
        wrong = np.mean(np.abs(error - error.mean(axis=(0, 1))))
        self.assertEqual(wrong, 0.)

    def test_spatial_luminance_pattern_is_retained(self):
        spatial = np.asarray([[-.2, .1], [.15, -.05]])
        self.assertAlmostEqual(float(spatial.mean()), 0., places=15)
        error = np.repeat(spatial[..., None], 3, axis=2)
        self.assertAlmostEqual(centered_scalar(error), float(np.abs(spatial).mean()), places=14)
        self.assertGreater(centered_scalar(error), 0.)

    def test_mean_is_per_observation_instead_of_across_normal_batch(self):
        errors = [np.full((2, 2, 3), value) for value in (.2, .4)]
        self.assertTrue(all(abs(centered_scalar(error)) < 1e-14 for error in errors))
        batch = np.stack(errors)
        wrong_shared_mu = np.mean(batch @ W)
        self.assertGreater(float(np.abs(batch - wrong_shared_mu).mean()), .09)

    def test_analytic_derivative_matches_finite_difference_and_rejects_detached_mean(self):
        error = self.asymmetric_error()
        analytic = independent_derivative(error)
        finite = np.empty_like(error)
        step = 1e-6
        for index in np.ndindex(error.shape):
            plus, minus = error.copy(), error.copy()
            plus[index] += step
            minus[index] -= step
            finite[index] = (centered_scalar(plus) - centered_scalar(minus)) / (2 * step)
        np.testing.assert_allclose(analytic, finite, rtol=1e-8, atol=1e-10)
        self.assertAlmostEqual(float(analytic.sum()), 0., places=14)
        signs = np.sign(error - np.mean(error @ W))
        detached_wrong = signs / error.size
        self.assertGreater(abs(float(detached_wrong.sum())), .3)
        self.assertGreater(float(np.max(np.abs(detached_wrong - finite))), .01)

    def test_L1_does_not_additively_split_into_DC_and_centered_error(self):
        error = self.asymmetric_error()
        mu = float(np.mean(error @ W))
        full = float(np.mean(np.abs(error)))
        self.assertGreater(abs(full - (abs(mu) + centered_scalar(error))), .01)

    def test_native_pool_grid_and_no_clipping_are_distinct_from_delivery(self):
        blocks = np.arange(3 * 16 * 16, dtype=np.float32).reshape(3, 16, 16) / 256
        native = np.repeat(np.repeat(blocks, 32, axis=1), 32, axis=2)
        pooled = native.reshape(3, 16, 32, 16, 32).mean(axis=(2, 4), dtype=np.float32)
        np.testing.assert_array_equal(pooled, blocks)
        self.assertEqual(pooled.shape, (3, 16, 16))
        unclipped_error = np.full((3, 16, 16), 1.2) - .6
        clipped_error = np.clip(np.full((3, 16, 16), 1.2), 0, 1) - .6
        self.assertAlmostEqual(float(np.abs(unclipped_error).mean()), .6, places=14)
        self.assertAlmostEqual(float(np.abs(clipped_error).mean()), .4, places=14)


class NormalToneAccumulationFixtures(unittest.TestCase):
    def test_dark_subsequence_cannot_be_recovered_by_rounded_subtraction(self):
        values = [np.asarray([v], dtype=np.float32) for v in (1e8, 1., -1e8, 1.)]
        all_ordered = sequential_fp32(values)
        normal = sequential_fp32([values[0], values[2]])
        dark = sequential_fp32([values[1], values[3]])
        self.assertEqual(float(all_ordered[0]), 1.)
        self.assertEqual(float(normal[0]), 0.)
        self.assertEqual(float(dark[0]), 2.)
        self.assertFalse(np.array_equal(all_ordered - normal, dark))
        self.assertFalse(np.array_equal(normal + dark, all_ordered))

    def test_hybrid_uses_original_interleaved_case_order(self):
        # Normal positions 0/2 use centered gradients; dark positions 1/3 keep
        # their original gradients. Grouping by exposure changes FP32 results.
        hybrid_cases = [np.asarray([v], dtype=np.float32) for v in (2e8, 1., -2e8, 1.)]
        canonical = sequential_fp32(hybrid_cases)
        grouped = sequential_fp32([sequential_fp32(hybrid_cases[::2]),
                                   sequential_fp32(hybrid_cases[1::2])])
        self.assertEqual(float(canonical[0]), 1.)
        self.assertEqual(float(grouped[0]), 2.)

    def test_normal_subgroup_keeps_original_weight4_divisor16(self):
        per_case = np.asarray([.5, -.25], dtype=np.float32)
        expected = np.float32(4.) * per_case / np.float32(16.)
        wrong_renormalized = np.float32(4.) * per_case / np.float32(8.)
        np.testing.assert_array_equal(expected, [.125, -.0625])
        np.testing.assert_array_equal(wrong_renormalized, expected * 2)

    def test_coarse_replacement_stays_at_existing_component_slot_and_preserves_cached_values(self):
        components = {name: None for name in ORDER}
        components['token_ce'] = np.asarray([1e8, 2.], dtype=np.float32)
        components['pixel_l1_rgb'] = np.asarray([1., 3.], dtype=np.float32)
        components['coarse_rgb_l1'] = np.asarray([-1e8, 4.], dtype=np.float32)
        components['behavior_rgb_l1'] = np.asarray([1., 5.], dtype=np.float32)
        before = {name: None if value is None else value.copy() for name, value in components.items()}
        canonical = sequential_fp32([components[name] for name in ORDER])
        appended = sequential_fp32([components[name] for name in ORDER if name != 'coarse_rgb_l1'] +
                                   [components['coarse_rgb_l1']])
        self.assertFalse(np.array_equal(canonical, appended))
        np.testing.assert_array_equal(canonical, [1., 14.])
        for name in ORDER:
            if before[name] is None:
                self.assertIsNone(components[name])
            else:
                np.testing.assert_array_equal(before[name], components[name])

    def test_momentum_zero_is_explicit_and_not_missing(self):
        missing = None
        real_zero = np.zeros(4, dtype=np.float32)
        self.assertIsNone(missing)
        self.assertIsNotNone(real_zero)
        self.assertEqual(real_zero.dtype, np.dtype('float32'))
        self.assertEqual(float(real_zero.sum()), 0.)

    def test_fixture_imports_no_neural_framework_or_worker(self):
        self.assertFalse(any(name == 'torch' or name.startswith('torch.') for name in sys.modules))
        tree = ast.parse(Path(__file__).read_text(encoding='utf-8'))
        imports = []
        for node in ast.walk(tree):
            if isinstance(node, ast.Import):
                imports.extend(item.name for item in node.names)
            elif isinstance(node, ast.ImportFrom):
                imports.append(node.module or '')
        self.assertTrue(all(not name.startswith(('research.', 'torch')) for name in imports))


if __name__ == '__main__':
    unittest.main()
