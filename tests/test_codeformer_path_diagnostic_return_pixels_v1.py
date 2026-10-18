"""Small CPU image/token fixtures; no model, neural forward, gradient or VM."""
import io
from pathlib import Path
import unittest
from unittest.mock import patch

import numpy as np

from research.codeformer_path_diagnostic_v1 import return_pixels as audit


class SavedPixelAuditTests(unittest.TestCase):
    def test_floor_conversion_including_boundary(self):
        value = np.array([0, .5, 1], dtype=np.float32).reshape(1, 1, 3)
        np.testing.assert_array_equal(audit.delivered_png(value), [[[0, 127, 255]]])

    def test_raw_invalid_range_nonfinite_and_wrong_dtype_fail(self):
        for value in (np.array([0, .5, 1.1], np.float32), np.array([0, np.nan, 1], np.float32),
                      np.array([0, .5, 1], np.float64)):
            with self.assertRaises(ValueError):
                audit.delivered_png(value.reshape(1, 1, 3))

    def test_paired_shift_separates_mean_contrast_and_derivative(self):
        target = np.zeros((2, 2, 3), dtype=np.float32)
        target[1] = .25
        shifted = target + .125
        result = audit.metrics(shifted, target)
        self.assertAlmostEqual(result['signed_luminance_bias'], .125)
        self.assertAlmostEqual(result['RGB_MAE'], .125)
        self.assertAlmostEqual(result['RGB_MSE'], .125 ** 2)
        self.assertAlmostEqual(result['mean_absolute_first_difference_error'], 0)
        self.assertAlmostEqual(result['luminance_std'], result['target_luminance_std'])

    def test_first_difference_reports_spatial_change(self):
        target = np.zeros((2, 2, 3), dtype=np.float32)
        predicted = target.copy(); predicted[1] = 1
        self.assertAlmostEqual(audit.metrics(predicted, target)['mean_absolute_first_difference_error'], .5)

    def test_captured_tied_selection_not_replaced_with_top2_first(self):
        tokens = np.zeros(256, np.int64); tokens[0] = 7
        indices = np.tile([0, 1], (256, 1)).astype(np.int64)
        probability = np.tile([.5, .25], (256, 1)).astype(np.float32); probability[0] = [.5, .5]
        logits = np.tile([2., 1.], (256, 1)).astype(np.float32)
        result = audit.token_statistics(tokens, probability, indices, logits)
        self.assertEqual(result['selected_index_not_top2_first_due_to_ties'], 1)
        self.assertEqual(result['native_probability_top2_zero_margin_tokens'], 1)
        self.assertFalse(result['actual_selection_independently_recomputed_from_full_logits'])

    def test_strict_winner_disagreement_is_rejected(self):
        tokens = np.zeros(256, np.int64); tokens[0] = 1
        indices = np.tile([0, 1], (256, 1)).astype(np.int64)
        probability = np.tile([.5, .25], (256, 1)).astype(np.float32)
        logits = np.tile([2., 1.], (256, 1)).astype(np.float32)
        with self.assertRaises(ValueError):
            audit.token_statistics(tokens, probability, indices, logits)

    def test_strict_array_rejects_trailing_payload_and_dtype(self):
        buffer = io.BytesIO()
        np.save(buffer, np.zeros((2, 2), dtype=np.float32))
        payload = buffer.getvalue()
        path = Path('fixture.npy')
        with patch.object(Path, 'open', side_effect=lambda *a, **k: io.BytesIO(payload)):
            audit.strict_array(path, (2, 2), np.float32)
            with self.assertRaises(ValueError):
                audit.strict_array(path, (2, 2), np.float64)
        with patch.object(Path, 'open', side_effect=lambda *a, **k: io.BytesIO(payload + b'extra')):
            with self.assertRaises(ValueError):
                audit.strict_array(path, (2, 2), np.float32)

    def test_conditional_factorial_interaction(self):
        metric = lambda value: {key: float(value) for key in ('luminance_mean', 'luminance_std',
            'signed_luminance_bias', 'RGB_MAE', 'RGB_MSE', 'mean_absolute_first_difference_error')}
        result = audit.factorial_contrasts({'C': metric(1), 'M': metric(3), 'F': metric(4), 'MF': metric(8)})
        self.assertEqual(result['RGB_MAE']['M_minus_C'], 2)
        self.assertEqual(result['RGB_MAE']['F_minus_C'], 3)
        self.assertEqual(result['RGB_MAE']['interaction_MF_minus_M_minus_F_plus_C'], 2)

    def test_expected_full_condition_census(self):
        cases = [{'id': 'u%s_s%s' % (u, s), 'update': u, 'slot': s} for u in (1, 24) for s in range(1, 5)]
        rows = audit.expected_conditions(cases)
        self.assertEqual(len(rows), 104)
        self.assertEqual(sum(row[0] == 'path' for row in rows), 64)
        self.assertEqual(sum(row[0] == 'repeat' for row in rows), 16)
        self.assertEqual(sum(row[0] == 'intervention' for row in rows), 24)

    def test_numerical_tolerance_does_not_accept_material_change(self):
        audit.scalar_equal(.1, .1 + 1e-9, 'tiny derived scalar')
        with self.assertRaises(ValueError):
            audit.scalar_equal(.1, .101, 'material difference')

    def test_bilinear_geometry_is_plain_average_for_factor2(self):
        clean = np.arange(512 * 512 * 3, dtype=np.float32).reshape(512, 512, 3)
        clean /= np.float32(clean.max())
        expected = (clean[0::2, 0::2] + clean[0::2, 1::2] + clean[1::2, 0::2] + clean[1::2, 1::2]) * np.float32(.25)
        result = audit.target256(clean)
        self.assertEqual(result.dtype, np.float32)
        self.assertEqual(result.shape, (256, 256, 3))
        np.testing.assert_allclose(result, expected, rtol=0, atol=1.2e-7)


if __name__ == '__main__':
    unittest.main()
