"""CPU fixtures: saved-pixel arithmetic and honest diagnostic scope only."""
import unittest
import numpy as np

from research.codeformer_bounded_fusion_feasibility_v1 import return_pixels as p


class BoundedReturnPixelTests(unittest.TestCase):
    def test_identical_target_has_unit_SSIM_zero_error_and_NULL_perfect_PSNR(self):
        image = np.full((256, 256, 3), .25, dtype=np.float32)
        result = p.paired_metrics(image, image)
        self.assertEqual(result['ssim'], 1.)
        self.assertEqual(result['mae'], 0.)
        self.assertIsNone(result['psnr'])
        self.assertEqual(result['gradient_vector_mae'], 0.)

    def test_brightness_change_is_not_invented_gradient_or_contrast_detail(self):
        target = np.full((256, 256, 3), .25, dtype=np.float32)
        result = p.paired_metrics(target + .125, target)
        self.assertAlmostEqual(result['signed_luminance_bias'], .125)
        self.assertEqual(result['gradient_vector_mae'], 0.)
        self.assertEqual(result['contrast_absolute_error'], 0.)
        self.assertLess(result['ssim'], 1.)

    def test_native_unbounded_prediction_is_not_clipped_in_metrics(self):
        target = np.zeros((512, 512, 3), dtype=np.float32)
        result = p.paired_metrics(np.full_like(target, 2.), target)
        self.assertEqual(result['mae'], 2.)
        self.assertEqual(result['mse'], 4.)

    def test_delivery_clamps_before_downsampling(self):
        native = np.zeros((512, 512, 3), dtype=np.float32)
        native[::2, ::2] = 4.
        native[1::2, ::2] = -1.
        raw = np.full((256, 256, 3), .25, dtype=np.float32)
        self.assertTrue(p.delivery_check(native, raw)['pointwise_pass'])
        wrong = np.full_like(raw, .75)
        with self.assertRaisesRegex(ValueError, 'clamp-before'):
            p.delivery_check(native, wrong)

    def test_delivery_rejects_mean_small_but_one_pixel_large_error(self):
        native = np.full((512, 512, 3), .5, dtype=np.float32)
        raw = np.full((256, 256, 3), .5, dtype=np.float32)
        raw[0, 0, 0] += .0001
        with self.assertRaises(ValueError):
            p.delivery_check(native, raw)

    def test_gate_is_RGB8_observed_central_support_only(self):
        image = np.full((256, 256, 3), 128, dtype=np.uint8)
        original = p.gate(image)
        image[:32] = 255; image[224:] = 0
        self.assertEqual(p.gate(image), original)
        with self.assertRaises(ValueError):
            p.gate(image.astype(np.float32))

    def test_prospective_scalar_roundoff_accepts_only_with_same_applied_FP32_gate(self):
        rebuilt = {'central192_sRGB_mean_luminance': .3, 'gate': .8}
        saved = {key: value + 3e-16 for key, value in rebuilt.items()}
        self.assertLess(max(p.gate_metadata(saved, rebuilt, 1e-12).values()), 1e-12)

    def test_prospective_scalar_roundoff_rejects_beyond_frozen_bound(self):
        rebuilt = {'central192_sRGB_mean_luminance': .3, 'gate': .8}
        saved = dict(rebuilt); saved['gate'] += 2e-12
        with self.assertRaisesRegex(ValueError, 'frozen scalar-roundoff'):
            p.gate_metadata(saved, rebuilt, 1e-12)
        with self.assertRaises(ValueError):
            p.gate_metadata(rebuilt, rebuilt, 1e-9)

    def test_clear_subset_keeps_outlier_failure_and_never_claims_full_DEV_guard(self):
        rows = []
        for rid in range(4):
            for mode in p.MODES:
                for view in ('raw256', 'native_png'):
                    rows.append({'case_id': str(rid), 'mode': mode, 'view': view,
                        'profile': 'clear_normal', 'section': 'paths',
                        'mae': .01 + (.031 if rid == 0 and mode == 'bounded_fixed' else 0)})
        result = p.clear_subset(rows)
        self.assertFalse(result['full61_clear_and53_blur_DEV_guards_evaluated'])
        selected = next(r for r in result['subsets'] if r['mode'] == 'bounded_fixed' and r['view'] == 'raw256')
        self.assertLess(selected['mean_mae_delta'], .01)
        self.assertFalse(selected['descriptive_subset_within_thresholds'])

    def test_exact_five_paths_and_ten_controls_are_not_extra_training(self):
        ids = ['case_%d' % i for i in range(32)]
        plan = p.expected_plan(ids, [ids[0], ids[-1]])
        self.assertEqual(len(plan), 170)
        self.assertEqual(sum(section == 'paths' for _, _, section in plan), 160)
        self.assertEqual(sum(section == 'zero_delta' for _, _, section in plan), 4)
        self.assertEqual(sum(section == 'identity' for _, _, section in plan), 2)

    def test_256_and_512_grid_photometry_agree_for_constant_shift(self):
        results = []
        for size in (256, 512):
            target = np.full((size, size, 3), .25, dtype=np.float32)
            results.append(p.paired_metrics(target + .125, target))
        self.assertEqual(results[0]['coarse_rgb_mae'], results[1]['coarse_rgb_mae'])
        self.assertEqual(results[0]['gradient_vector_mae'], results[1]['gradient_vector_mae'])


if __name__ == '__main__':
    unittest.main()
