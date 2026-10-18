"""CPU saved-pixel scientific invariants; no models or returned code executes."""
from copy import deepcopy
from pathlib import Path
import sys
import unittest

import numpy as np

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
from scripts import audit_codeformer_role_preserving_training_r2_pixels_v1 as a


def bank():
    result = {}
    for i in range(61 + 53 + 53 + 53):
        profile = 'clear' if i < 61 else 'blur' if i < 114 else 'lowlight' if i < 167 else 'compound'
        metrics = {key: .1 if key != 'ssim' else .6 for key in a.WORKER_METRICS}
        result[str(i)] = {'id': str(i), 'profile': profile, 'source': 'FFHQ validation',
            'evidence': 'paired_synthetic', 'metrics': {view: dict(metrics) for view in a.VIEWS}}
    return result


class PixelScientificFixtures(unittest.TestCase):
    def test_no_neural_framework_import(self):
        self.assertFalse(any(name in sys.modules for name in ('torch', 'tensorflow', 'jax')))

    def test_native_target_is_null_and_fabricated_target_rejected(self):
        image = np.zeros((256, 256, 3), np.float32)
        mask = np.ones((256, 256), bool)
        values = a.metrics(image, None, mask, native=True)
        self.assertTrue(all(value is None for value in values.values()))
        with self.assertRaises(ValueError):
            a.metrics(image, image, mask, native=True)

    def test_exact_paired_pixels_have_zero_errors_and_unit_SSIM(self):
        rng = np.random.default_rng(317)
        image = rng.random((256, 256, 3), dtype=np.float32)
        values = a.metrics(image, image, np.ones((256, 256), bool), native=False)
        for metric in ('mae', 'mse', 'luminance_bias', 'signed_luminance_bias', 'coarse_rgb_mae', 'gradient_vector_mae'):
            self.assertEqual(values[metric], 0.)
        self.assertEqual(values['ssim'], 1.)
        self.assertEqual(values['masked_paired_SSIM'], 1.)
        self.assertIsNone(values['psnr'])

    def test_known_constant_error_and_partial_support_do_not_invent_full_SSIM(self):
        image = np.full((256, 256, 3), .75, np.float32)
        target = np.full_like(image, .25)
        mask = np.zeros((256, 256), bool); mask[32:224, 32:224] = True
        values = a.metrics(image, target, mask, native=False)
        self.assertEqual(values['mae'], .5)
        self.assertEqual(values['mse'], .25)
        self.assertAlmostEqual(values['psnr'], 6.020599913279624)
        self.assertAlmostEqual(values['luminance_bias'], .5)
        self.assertAlmostEqual(values['signed_luminance_bias'], .5)
        self.assertIsNone(values['ssim'])
        self.assertIsNone(values['coarse_rgb_mae'])
        self.assertIsNotNone(values['masked_paired_SSIM'])

    def test_delivered_PNG_is_floor_and_exact_input_outside_observed_support(self):
        raw = np.full((256, 256, 3), .5, np.float32)
        inp = np.full((256, 256, 3), 19, np.uint8)
        mask = np.ones((256, 256), bool); mask[:10] = False
        png = a.expected_png(raw, inp, mask)
        self.assertTrue(np.all(png[mask] == 127))
        self.assertTrue(np.array_equal(png[~mask], inp[~mask]))

    def test_clear_worst_case_failure_cannot_hide_under_passing_mean(self):
        original = bank(); candidate = deepcopy(original)
        candidate['0']['metrics']['native_png']['mae'] += .030001
        result = a.preservation(candidate, original)
        self.assertFalse(result['pass'])
        row = result['clear']['views']['native_png']['mae']
        self.assertLess(row['mean_delta'], .01)
        self.assertEqual(row['cases_over_0_03'], 1)
        self.assertFalse(row['pass'])

    def test_clear_and_blur_RAW_PNG_are_separate_constraints(self):
        original = bank(); candidate = deepcopy(original)
        candidate['61']['metrics']['raw']['coarse_rgb_mae'] += .030001
        result = a.preservation(candidate, original)
        self.assertFalse(result['normal_blur']['views']['raw']['coarse_rgb_mae']['pass'])
        self.assertTrue(result['normal_blur']['views']['native_png']['coarse_rgb_mae']['pass'])
        self.assertFalse(result['pass'])

    def test_dark_retention_needs_MAE_and_SSIM_for_both_full_profiles(self):
        original = bank(); candidate = deepcopy(original)
        self.assertTrue(a.dark_retention(candidate, original)['pass'])
        candidate['114']['metrics']['native_png']['mae'] -= .1
        candidate['114']['metrics']['native_png']['ssim'] -= .001
        result = a.dark_retention(candidate, original)
        self.assertFalse(result['profiles']['lowlight']['pass'])
        self.assertTrue(result['profiles']['compound']['pass'])
        self.assertFalse(result['pass'])

    def test_compare_tree_detects_nested_gate_and_scalar_errors(self):
        self.assertEqual(a.compare_tree({'x': {'y': [1., None, True]}}, {'x': {'y': [1. + 1e-12, None, True]}}), [])
        self.assertTrue(a.compare_tree({'x': {'y': [1., None, True]}}, {'x': {'y': [1. + 1e-5, None, True]}}))
        self.assertTrue(a.compare_tree({'pass': True}, {'pass': False}))


if __name__ == '__main__':
    unittest.main(verbosity=2)
