"""Synthetic arithmetic/scalar fixtures only; no saved model/head application."""
import copy
import unittest

import numpy as np

from research.codeformer_u64_gate_aligned_pilot_v1 import arithmetic, quality


def gradients(values):
    return np.tile(np.asarray(values, dtype=np.float64), (8, 1))


def means(value=.1):
    return {view: {role: {'mae': value, 'adjacent_rgb_error': value}
                   for role in quality.ROLES} for view in quality.VIEWS}


def inherited_result(passed=True, terminal_pass=True):
    return {'pass': passed, 'original_terminal': {'pass': terminal_pass, 'binding': False}}


class PilotArithmeticTests(unittest.TestCase):
    def test_fixed_scalar_actual_fp32_bounds_and_no_input_mutation(self):
        current = np.array([.1, -.2], dtype=np.float32)
        original, direction, gates = current.copy(), np.array([-1., -2.]), gradients([1., 1.])
        before = [a.tobytes() for a in (current, original, direction, gates)]
        result = arithmetic.propose_fp32_vector_step(current, original, direction, gates)
        self.assertEqual(result['certificate']['step_scalar'], .99 * 2.5e-5 / 2.)
        self.assertTrue(result['certificate']['all8_actual_displacement_dots_strict_negative'])
        self.assertLessEqual(np.max(np.abs(result['actual_delta'])), 2.5e-5)
        self.assertEqual(before, [a.tobytes() for a in (current, original, direction, gates)])
        self.assertFalse(np.shares_memory(result['candidate'], current))
        self.assertFalse(result['certificate']['parameter_applied'])

    def test_cumulative_bound_rejects_without_retry(self):
        current, original = np.array([9e-5], np.float32), np.zeros(1, np.float32)
        with self.assertRaisesRegex(ValueError, 'cumulative'):
            arithmetic.propose_fp32_vector_step(current, original, np.ones(1), gradients([-1.]))

    def test_rounding_overshoot_rejects_without_retry(self):
        current = np.array([257.], np.float32)
        with self.assertRaisesRegex(ValueError, 'per-coordinate'):
            arithmetic.propose_fp32_vector_step(current, current.copy(), np.ones(1), gradients([-1.]))

    def test_rounding_can_destroy_direction_sign_and_is_rejected(self):
        current = np.array([1., 1e-10], np.float32)
        with self.assertRaisesRegex(ValueError, 'Rounded actual displacement'):
            arithmetic.propose_fp32_vector_step(current, current.copy(), np.array([-1., 1.]), gradients([1., .9997]))

    def test_rounding_to_no_displacement_rejects(self):
        current = np.array([1e8], np.float32)
        with self.assertRaisesRegex(ValueError, 'Rounded actual displacement'):
            arithmetic.propose_fp32_vector_step(current, current.copy(), np.array([-1.]), gradients([1.]))

    def test_missing_gate_nonfinite_wrong_precision_and_zero_reject(self):
        current = np.zeros(2, np.float32)
        bad = ((np.zeros(2), gradients([1., 1.])),
               (np.array([-1., -1.], np.float32), gradients([1., 1.])),
               (np.array([-1., -1.]), np.ones((7, 2))),
               (np.array([-1., np.nan]), gradients([1., 1.])),
               (np.array([-1., -1.]), np.zeros((8, 2))))
        for direction, gates in bad:
            with self.subTest(shape=gates.shape, dtype=direction.dtype):
                with self.assertRaises(ValueError):
                    arithmetic.propose_fp32_vector_step(current, current.copy(), direction, gates)

    def test_exact_head_schema_rejects_missing_name(self):
        with self.assertRaisesRegex(ValueError, 'Exact36-head namespace'):
            arithmetic.propose_fp32_step({}, {}, np.ones(1), np.ones((8, 1)))

    def test_exact_head_adapter_on_synthetic_arrays_only(self):
        shapes = arithmetic.head_shapes()
        current = {n: np.zeros(s, np.float32) for n, s in shapes.items()}
        original = {n: a.copy() for n, a in current.items()}
        result = arithmetic.propose_fp32_step(current, original,
            -np.ones(168960), np.ones((8, 168960)))
        self.assertEqual(set(result['candidate']), set(shapes))
        self.assertEqual(result['actual_delta'].shape, (168960,))
        for name, shape in shapes.items():
            self.assertEqual(result['candidate'][name].shape, shape)
            self.assertEqual(result['candidate'][name].dtype, np.float32)
            self.assertTrue(np.all(current[name] == 0))
            self.assertTrue(np.all(original[name] == 0))
            self.assertFalse(np.shares_memory(result['candidate'][name], current[name]))


class PilotQualityTests(unittest.TestCase):
    def test_equal_nonterminal_means_pass_without_modifying_inherited(self):
        inherited = {'pass': True, 'individual_cases': {'preserved': True}}
        before = copy.deepcopy(inherited)
        result = quality.evaluate_quality(means(), means(), means(), False, inherited)
        self.assertTrue(result['pass'])
        self.assertEqual(len(result['role_mean_nonincrease_checks']), 32)
        self.assertEqual(inherited, before)
        self.assertEqual(result['inherited_gate_result'], inherited)
        self.assertIsNot(result['inherited_gate_result'], inherited)

    def test_increase_against_either_baseline_stops(self):
        original, prior, current = means(.2), means(.1), means(.15)
        result = quality.evaluate_quality(current, original, prior, False, {'pass': True})
        self.assertFalse(result['pass'])
        self.assertTrue(result['quality_stop'])

    def test_tiny_increase_has_no_tolerance(self):
        current = means()
        current['raw']['clear']['mae'] = np.nextafter(.1, np.inf)
        self.assertFalse(quality.evaluate_quality(current, means(), means(), False, {'pass': True})['pass'])

    def test_terminal_requires_strict_clear_lowlight_mae_in_both_views(self):
        equal = quality.evaluate_quality(means(), means(), means(), True, inherited_result())
        self.assertFalse(equal['pass'])
        self.assertEqual(len(equal['terminal_clear_lowlight_strict_MAE_checks']), 4)
        current = means(.09)
        self.assertTrue(quality.evaluate_quality(current, means(), means(), True, inherited_result())['pass'])
        current['native_png']['clear']['mae'] = .1
        self.assertFalse(quality.evaluate_quality(current, means(), means(), True, inherited_result())['pass'])

    def test_inherited_failure_cannot_be_overridden(self):
        result = quality.evaluate_quality(means(.09), means(), means(), True,
            {'pass': False, 'original_terminal': {'pass': True}, 'legacy': {'pass': False}})
        self.assertFalse(result['pass'])
        self.assertTrue(result['additional_role_mean_checks_pass'])
        self.assertFalse(result['earlier_state_fallback_or_retry_authorized'])

    def test_original_terminal_gate_binds_even_when_inherited_u4_passes(self):
        inherited = inherited_result(passed=True, terminal_pass=False)
        result = quality.evaluate_quality(means(.09), means(), means(), True, inherited)
        self.assertFalse(result['pass'])
        self.assertTrue(result['additional_role_mean_checks_pass'])
        self.assertTrue(result['inherited_original_terminal_binding']['required'])
        self.assertFalse(result['inherited_original_terminal_binding']['pass'])
        self.assertEqual(result['inherited_gate_result'], inherited)
        with self.assertRaises(ValueError):
            quality.evaluate_quality(means(.09), means(), means(), True, {'pass': True})

    def test_missing_nonfinite_boolean_and_bad_inherited_reject(self):
        bad = means()
        bad['raw']['clear']['mae'] = float('nan')
        with self.assertRaises(ValueError):
            quality.evaluate_quality(bad, means(), means(), False, {'pass': True})
        bad = means()
        bad['raw']['clear']['mae'] = True
        with self.assertRaises(ValueError):
            quality.evaluate_quality(bad, means(), means(), False, {'pass': True})
        with self.assertRaises(ValueError):
            quality.evaluate_quality(means(), means(), means(), False, {'pass': 1})
        bad = means()
        del bad['native_png']['clear']
        with self.assertRaises(ValueError):
            quality.evaluate_quality(bad, means(), means(), False, {'pass': True})


if __name__ == '__main__':
    unittest.main()
