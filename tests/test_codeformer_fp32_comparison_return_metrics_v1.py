"""Small CPU scalar/image fixtures; no model, gradient or VM execution."""
import unittest

import numpy as np

from research.codeformer_fp32_comparison_v1 import return_metrics as audit


class ReturnMetricTests(unittest.TestCase):
    def setUp(self):
        self.observed = np.ones((256, 256), dtype=bool)
        self.target = np.full((256, 256, 3), .25, dtype=np.float32)

    def test_identity_ssim_and_zero_errors(self):
        result = audit.paired_secondary(self.target, self.target, self.observed, evidence='paired_synthetic')
        self.assertEqual(result['ssim'], 1.)
        for key in ('signed_luminance_bias', 'contrast_absolute_error', 'gradient_vector_mae', 'gradient_magnitude_mae'):
            self.assertEqual(result[key], 0.)

    def test_brightness_shift_does_not_invent_detail(self):
        result = audit.paired_secondary(self.target + .125, self.target, self.observed, evidence='paired_synthetic')
        self.assertAlmostEqual(result['signed_luminance_bias'], .125)
        self.assertEqual(result['gradient_vector_mae'], 0.)
        self.assertEqual(result['contrast_absolute_error'], 0.)
        self.assertLess(result['ssim'], 1.)

    def test_native_reference_metrics_remain_null(self):
        result = audit.paired_secondary(self.target, None, self.observed, evidence='native_unpaired')
        self.assertTrue(all(value is None for value in result.values()))
        with self.assertRaises(ValueError):
            audit.paired_secondary(self.target, self.target, self.observed, evidence='native_unpaired')

    def test_padding_does_not_contribute_to_observed_window_metrics(self):
        observed = np.zeros_like(self.observed); observed[35:220, 45:190] = True
        rng = np.random.default_rng(19)
        target = rng.random((256, 256, 3)).astype(np.float32)
        prediction = np.clip(target + .02, 0, 1)
        a = audit.paired_secondary(prediction, target, observed, evidence='paired_synthetic')
        target[~observed] = 1; prediction[~observed] = 0
        b = audit.paired_secondary(prediction, target, observed, evidence='paired_synthetic')
        self.assertEqual(a, b)

    def test_PNG_floor_protects_original_unobserved_pixels(self):
        raw = np.full((256, 256, 3), .5, dtype=np.float32)
        pixels = np.full((256, 256, 3), 73, dtype=np.uint8)
        support = np.zeros_like(self.observed); support[20:240, 20:240] = True
        result = audit.v1.expected_native_png(raw, pixels, support)
        self.assertTrue((result[support] == 127).all())
        self.assertTrue((result[~support] == 73).all())

    def test_clear_gate_retains_per_case_failure_despite_good_mean(self):
        baseline = {str(i): {'raw': .01, 'native_png': .01} for i in range(61)}
        candidate = {key: dict(value) for key, value in baseline.items()}
        candidate['1']['native_png'] = .05
        gate = audit.v1.independently_clear_gate(candidate, baseline,
            {'mean_clear_mae_delta_stop': .01, 'worst_per_case_clear_mae_delta_stop': .03})
        self.assertFalse(gate['pass'])
        self.assertTrue(gate['views']['raw']['pass'])
        self.assertLess(gate['views']['native_png']['mean_mae_delta'], .01)

    def test_blur_gate_keeps_both_raw_and_PNG_bias_and_coarse(self):
        baseline = {str(i): {view: {'luminance_bias': .01, 'coarse_rgb_mae': .01}
                             for view in audit.VIEWS} for i in range(53)}
        candidate = {key: {view: dict(value) for view, value in values.items()} for key, values in baseline.items()}
        for value in candidate.values():
            value['raw']['luminance_bias'] = .021
        gate = audit.v2.independent_blur_gate(candidate, baseline,
            {'mean_blur_photometric_delta_stop': .01, 'worst_per_case_blur_photometric_delta_stop': .03})
        self.assertFalse(gate['pass'])
        self.assertFalse(gate['views']['raw']['luminance_bias']['pass'])
        self.assertTrue(gate['views']['native_png']['coarse_rgb_mae']['pass'])

    def test_more_texture_can_have_worse_detail_error(self):
        stripe = np.broadcast_to((((np.arange(256) // 2) % 2) * .1)[None, :, None], self.target.shape)
        result = audit.paired_secondary(self.target + stripe, self.target, self.observed, evidence='paired_synthetic')
        self.assertGreater(result['gradient_vector_mae'], 0)
        self.assertLess(result['ssim'], 1)


if __name__ == '__main__':
    unittest.main()
