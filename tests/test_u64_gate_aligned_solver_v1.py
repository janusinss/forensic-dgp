"""Analytical NumPy fixtures; no framework, model, optimizer or parameters."""
import math
import json
from pathlib import Path
import unittest
from unittest import mock

import numpy as np

from research.codeformer_u64_gate_aligned_v1 import contract, solver


class GateAlignedSolverTests(unittest.TestCase):
    def test_orthogonal_closed_form(self):
        result = solver.solve_common_descent(np.eye(2))
        np.testing.assert_allclose(result['simplex_coefficients'], [.5, .5], atol=1e-12)
        np.testing.assert_allclose(result['direction'], [-math.sqrt(.5)] * 2, atol=1e-12)
        self.assertEqual(result['face_count'], 3)
        self.assertGreater(result['minimum_normalized_descent_margin'], .7)

    def test_normalization_prevents_norm_domination(self):
        result = solver.solve_common_descent(np.array([[1000., 0.], [0., .001]]))
        np.testing.assert_allclose(result['simplex_coefficients'], [.5, .5], atol=1e-12)
        self.assertTrue(np.all(result['original_gradient_dot_direction'] < 0))

    def test_singular_duplicate_faces_are_checked(self):
        result = solver.solve_common_descent(np.array([[1., 0.], [2., 0.], [3., 0.]]))
        np.testing.assert_allclose(result['direction'], [-1., 0.], atol=1e-12)
        self.assertEqual(result['face_count'], 7)
        self.assertEqual(result['accepted_faces'], 7)

    def test_conflicting_but_feasible_closed_form(self):
        result = solver.solve_common_descent(np.array([[1., 0.], [-.5, math.sqrt(.75)]]))
        np.testing.assert_allclose(result['direction'], [-.5, -math.sqrt(.75)], atol=1e-12)
        self.assertAlmostEqual(result['minimum_normalized_descent_margin'], .5, places=12)

    def test_exact_cancellation_rejects_without_fallback(self):
        with self.assertRaises(solver.CommonDescentError):
            solver.solve_common_descent(np.array([[1., 0.], [-1., 0.]]))

    def test_insufficient_strict_margin_rejects(self):
        with self.assertRaises(solver.CommonDescentError):
            solver.solve_common_descent(np.array([[1., 0.], [-1., 1e-8]]))

    def test_zero_nonfinite_wrong_dimension_and_overscope_reject(self):
        for data in (np.array([[0., 0.]]), np.array([[np.nan, 1.]]),
                     np.array([[np.inf, 1.]]), np.ones((9, 2)), np.ones(3),
                     np.ones((2, 2), dtype=np.int64)):
            with self.subTest(shape=data.shape):
                with self.assertRaises(solver.CommonDescentError):
                    solver.solve_common_descent(data)

    def test_eight_vector_face_cap_and_wrapper(self):
        bank = {r: {q: np.array([1., (i + 1.) / 100.]) for q in contract.GATES}
                for i, r in enumerate(contract.ROLES)}
        result = solver.solve_eight_pooled_gates(bank)
        self.assertEqual(result['face_count'], 255)
        self.assertEqual(result['accepted_faces'] + result['rejected_faces'], 255)
        del bank[contract.ROLES[-1]]['gate_adjacent']
        with self.assertRaises(solver.CommonDescentError):
            solver.solve_eight_pooled_gates(bank)

    def test_expired_wall_budget_rejects(self):
        with mock.patch.object(solver.time, 'monotonic', side_effect=[0., 11.]):
            with self.assertRaises(solver.CommonDescentError):
                solver.solve_common_descent(np.eye(2))

    def test_coordinate_scope_rejects_above_headspace(self):
        with self.assertRaises(solver.CommonDescentError):
            solver.solve_common_descent(np.ones((1, solver.MAX_COORDINATES + 1)))

    def test_actual_bound_contract_preserves_original_quality(self):
        root = Path(__file__).resolve().parents[1]
        folder = root / 'outputs/cctv_codeformer_u64_role_gradients_v1_vm'
        standard_raw = (folder / 'evidence/standard_protocol.json').read_bytes()
        source = json.loads(standard_raw)
        design = contract.prospective_contract((folder / 'protocol.json').read_bytes(), standard_raw)
        self.assertEqual(design['quality']['inherited_policy_exact'], source['quality_policy'])
        self.assertEqual(design['quality']['inherited_legacy24_guard_definitions_exact'], source['guards'])
        self.assertEqual(len(design['selection']['optimization']), 64)
        self.assertEqual(design['limits']['worker_seconds'], 1800)
        self.assertEqual(design['limits']['external_seconds'], 1860)
        self.assertEqual(design['limits']['maximum_RSS_bytes'], 10 * 1024**3)
        self.assertFalse(design['worker_implemented'])
        self.assertFalse(design['manual_packet_ready'])
        self.assertFalse(design['pilot_selected_or_authorized'])
        self.assertTrue(design['prior_failures_remain_binding'])

    def test_contract_counts_and_no_executable_claim(self):
        limits = contract.operation_limits()
        self.assertEqual(limits['full_forward_attempts'], 540)
        self.assertEqual(limits['gradient_query_attempts'], 512)
        self.assertEqual(limits['parameter_applications'], 4)
        self.assertEqual(limits['optimizer_constructions'], 0)
        self.assertEqual(limits['maximum_absolute_coordinate_step'] * 4,
                         limits['maximum_cumulative_absolute_coordinate_delta'])
        with self.assertRaises(ValueError):
            contract.prospective_contract(b'{}', b'{}')


if __name__ == '__main__':
    unittest.main()
