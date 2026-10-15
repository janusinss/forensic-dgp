"""Pure scalar/NumPy activation fixtures; no model or derivative execution."""
import copy
import json
from pathlib import Path
import unittest

import numpy as np

from cctv_dgp_decoder_activation_v1_math import (
    METRIC_DEFINITIONS, finite_objective, relu_right_derivative,
    affine_hinge_events, one_sided_objective, mean_control_right_mse_at_initial,
)


def fixture():
    current = {'MSE': np.ones(5), 'SSIM_loss': np.full(5, 2.),
               'ArcFace_loss': np.full(5, 3.), 'landmark_structure': np.full(5, 4.)}
    baseline = {key: current[key].copy() for key in ['MSE', 'SSIM_loss', 'ArcFace_loss']}
    photo = {'old_MSE': np.ones(5), 'actual_MSE': np.ones(5), 'mean_only_MSE': np.ones(5)}
    slopes = {key: np.zeros(5) for key in current}
    return current, baseline, photo, slopes


class ActivationArithmetic(unittest.TestCase):
    def test_baseline_matches_hand_computed_full_objective(self):
        current, baseline, photo, _ = fixture()
        value = finite_objective(current, baseline, [1, 2, 3, 4], [1, 1], photometric=photo)
        self.assertAlmostEqual(value['objective'], 5.2)
        self.assertEqual(value['regression_barriers_RGB_SSIM_recognizer'], [0., 0., 0.])
        self.assertEqual(value['photometric_guard'], 0.)

    def test_real_offsets_are_subtracted_before_regression_hinges(self):
        current, baseline, photo, _ = fixture()
        current['MSE'][2] += 2e-12
        current['SSIM_loss'][3] += 2e-6
        value = finite_objective(current, baseline, [1, 2, 3, 4], [1, 1], photometric=photo)
        self.assertAlmostEqual(value['regression_barriers_RGB_SSIM_recognizer'][0],
                               max(0., current['MSE'][2]-1.-1e-12), places=20)
        self.assertAlmostEqual(value['regression_barriers_RGB_SSIM_recognizer'][1],
                               max(0., current['SSIM_loss'][3]-2.-1e-6)/2, places=16)

    def test_photometric_values_are_explicit_not_inferred_from_canonical_MSE(self):
        current, baseline, photo, _ = fixture()
        photo['actual_MSE'][1:] = .9
        photo['mean_only_MSE'][1:] = .95
        value = finite_objective(current, baseline, [1, 2, 3, 4], [1, 1], photometric=photo)
        self.assertAlmostEqual(value['photometric_guard'], .3)
        self.assertAlmostEqual(value['objective'], 5.5)

    def test_clear_profile_is_excluded_from_brightness_guard(self):
        current, baseline, photo, _ = fixture()
        photo['mean_only_MSE'][0] = 0.
        value = finite_objective(current, baseline, [1, 2, 3, 4], [1, 1], photometric=photo)
        self.assertEqual(value['photometric_guard'], 0.)

    def test_inactive_margin_has_zero_infinitesimal_slope_but_finite_activation(self):
        residual = np.array([-1e-12, -1e-6])
        slope = np.array([1., 2.])
        np.testing.assert_array_equal(relu_right_derivative(residual, slope), [0., 0.])
        events = affine_hinge_events(residual, slope)
        self.assertEqual(events[0]['activation_distance'], 1e-12)
        self.assertEqual(events[1]['activation_distance'], 5e-7)
        self.assertEqual(max(0., residual[0]+.5e-12*slope[0]), 0.)
        self.assertGreater(max(0., residual[0]+2e-12*slope[0]), 0.)

    def test_exact_and_active_boundary_directions_have_correct_signs(self):
        np.testing.assert_array_equal(relu_right_derivative(
            np.array([0., 0., 1., -1.]), np.array([2., -2., -2., 2.])), [2., 0., -2., 0.])

    def test_nested_photo_right_rule_and_sign_reversal_are_not_linear(self):
        current, baseline, photo, slopes = fixture()
        positive = one_sided_objective(current, baseline, slopes, [1, 2, 3, 4], [1, 1],
            photometric=photo, photometric_slopes={'actual_MSE': np.full(5, -1.),
                'mean_only_MSE': np.full(5, -.5)}, mean_control_derivative_kind='analytic_scalar_fixture')
        negative = one_sided_objective(current, baseline, slopes, [1, 2, 3, 4], [1, 1],
            photometric=photo, photometric_slopes={'actual_MSE': np.ones(5),
                'mean_only_MSE': np.full(5, .5)}, mean_control_derivative_kind='analytic_scalar_fixture')
        self.assertAlmostEqual(positive['photometric_guard_directional'], 3.)
        self.assertEqual(negative['photometric_guard_directional'], 0.)
        self.assertFalse(positive['literal_float32_program_derivative_certified'])

    def test_active_brightness_penalty_can_have_negative_right_slope(self):
        current, baseline, photo, slopes = fixture()
        photo['actual_MSE'][:] = .9
        photo['mean_only_MSE'][:] = .95
        value = one_sided_objective(current, baseline, slopes, [1, 2, 3, 4], [1, 1],
            photometric=photo, photometric_slopes={'actual_MSE': np.zeros(5),
                'mean_only_MSE': np.full(5, .1)}, mean_control_derivative_kind='analytic_scalar_fixture')
        self.assertAlmostEqual(value['photometric_guard_directional'], -1.)

    def test_scalar_autograd_mean_control_is_not_silently_certified(self):
        current, baseline, photo, slopes = fixture()
        value = one_sided_objective(current, baseline, slopes, [1, 2, 3, 4], [1, 1],
            photometric=photo, photometric_slopes={'actual_MSE': np.zeros(5), 'mean_only_MSE': np.zeros(5)})
        self.assertFalse(value['photometric_directional_supported'])
        self.assertIsNone(value['photometric_guard_directional'])
        self.assertIsNone(value['objective_directional'])
        self.assertTrue(value['unsupported_reason'])

    def test_idealized_clip_handles_inward_outward_and_interior_directions(self):
        baseline = np.empty((5, 3, 256, 256), np.float32)
        target = np.empty_like(baseline)
        for i, (a, b) in enumerate(zip([0., 0., 1., 1., .5], [1., 1., 0., 0., 0.])):
            baseline[i] = a; target[i] = b
        slopes = np.repeat(np.asarray([-2., 2., 2., -2., 2.])[:, None], 3, axis=1)
        mask = np.ones((5, 1, 256, 256), np.float32)
        value = mean_control_right_mse_at_initial(baseline, target, mask, slopes)
        np.testing.assert_array_equal(value['mean_only_MSE_slopes'], [0., -4., 0., -4., 2.])
        self.assertEqual(value['mean_control_derivative_kind'], 'idealized_pixel_clip')
        self.assertFalse(value['literal_float32_program_derivative_certified'])

    def test_nonfinite_shape_and_invalid_normalization_are_rejected(self):
        current, baseline, photo, _ = fixture()
        for bad in [float('nan'), float('inf'), -float('inf')]:
            altered = copy.deepcopy(current); altered['MSE'][0] = bad
            with self.subTest(bad=bad), self.assertRaises(ValueError):
                finite_objective(altered, baseline, [1, 2, 3, 4], [1, 1], photometric=photo)
        for normalizers in [[0, 2, 3, 4], [1, 2, 3], [1, 2, float('nan'), 4]]:
            with self.subTest(normalizers=normalizers), self.assertRaises(ValueError):
                finite_objective(current, baseline, normalizers, [1, 1], photometric=photo)
        with self.assertRaises(ValueError):
            relu_right_derivative(np.zeros(5), np.zeros(4))

    def test_mean_control_ignores_unobserved_pixels_and_rejects_invalid_masks(self):
        baseline = np.full((5, 3, 256, 256), .5, np.float32)
        target = np.zeros_like(baseline)
        mask = np.zeros((5, 1, 256, 256), np.float32); mask[:, :, :128] = 1.
        baseline[:, :, 128:] = 1.; target[:, :, 128:] = 1.
        value = mean_control_right_mse_at_initial(baseline, target, mask, np.ones((5, 3)))
        np.testing.assert_array_equal(value['mean_only_MSE_slopes'], np.ones(5))
        self.assertEqual(value['observed_pixels_per_profile'], [128*256]*5)
        with self.assertRaises(ValueError):
            mean_control_right_mse_at_initial(baseline, target, np.zeros_like(mask), np.ones((5, 3)))
        with self.assertRaises(ValueError):
            mean_control_right_mse_at_initial(baseline, target, mask, np.full((5, 3), np.inf))

    def test_nested_event_distances_retain_inner_branch_validity_warning(self):
        current, baseline, photo, slopes = fixture()
        photo['mean_only_MSE'][:] = 1.1
        result = one_sided_objective(current, baseline, slopes, [1, 2, 3, 4], [1, 1],
            photometric=photo, photometric_slopes={'actual_MSE': np.zeros(5),
                'mean_only_MSE': np.full(5, -.5)}, mean_control_derivative_kind='analytic_scalar_fixture')
        nested = result['nested_photometric_model']
        self.assertAlmostEqual(nested['mean_gain_events'][0]['activation_distance'], .2)
        self.assertEqual(result['photometric_guard_directional'], 0.)
        self.assertIn('only while the initial inner hinge branches remain valid', nested['outer_event_scope'])

    def test_landmark_luminance_gain_definition_is_separate_from_ArcFace(self):
        self.assertIn('luminance', METRIC_DEFINITIONS['structure_error'])
        self.assertIn('landmark_high_frequency_MSE', METRIC_DEFINITIONS['structure_gain'])
        self.assertIn('separate', METRIC_DEFINITIONS['ArcFace_observed_fixed'])

    def test_retained_120_component_rows_match_scalar_formula_without_neural_calls(self):
        root = Path(__file__).resolve().parent.parent
        stage = root/'outputs/cctv_dgp_decoder_hinge_repair_v1_return/outputs/decoder_hinge_repair_v1'
        if not stage.is_dir(): self.skipTest('Retained local return not included in tiny fixture environment')
        old = json.loads((root/'outputs/cctv_dgp_spatial_readout_v1_return/protocol.json').read_text(encoding='utf-8'))
        load = lambda label: json.loads((stage/label/'objective_components.json').read_text(encoding='utf-8'))['rows']
        initial = {row['reference']: row for row in load('baseline')}
        checked = 0
        for label in ['baseline', 'initial', 'legacy_A1', 'fit_u001', 'probe_1_16', 'probe_1_4']:
            for row in load(label):
                before = initial[row['reference']]
                current = {key: row[name] for key, name in [('MSE', 'raw_RGB_MSE'),
                    ('SSIM_loss', 'raw_SSIM_loss'), ('ArcFace_loss', 'raw_recognizer_loss'),
                    ('landmark_structure', 'raw_structure_loss')]}
                baseline = {key: before[name] for key, name in [('MSE', 'raw_RGB_MSE'),
                    ('SSIM_loss', 'raw_SSIM_loss'), ('ArcFace_loss', 'raw_recognizer_loss')]}
                actual = finite_objective(current, baseline, old['normalizers'], old['active_anchor_normalizers'],
                    photometric=row['per_image_photometric'])
                for key in ['objective', 'photometric_guard', 'normalized_weighted_RGB_SSIM_recognizer_structure',
                            'regression_barriers_RGB_SSIM_recognizer', 'clear_blur_RGB_anchors']:
                    np.testing.assert_allclose(actual[key], row[key], rtol=1e-12, atol=1e-12)
                checked += 1
        self.assertEqual(checked, 120)


if __name__ == '__main__': unittest.main(verbosity=2)
