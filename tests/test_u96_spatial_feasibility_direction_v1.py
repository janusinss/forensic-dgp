"""Analytical adversarial feasibility fixtures; no model/framework/VM calls."""
import ast
import copy
from pathlib import Path
import unittest

import numpy as np

from research.codeformer_u96_spatial_feasibility_v1 import direction as D


class CommonDirectionFixtures(unittest.TestCase):
    def gradients(self, vectors):
        rows = []
        for items in vectors:
            row = np.zeros(D.PARAMETER_COUNT, dtype=np.float64)
            for index, value in items.items():
                row[index] = value
            rows.append(row)
        return dict(zip(D.TASK_NAMES, rows))

    def aligned(self, index=0):
        return self.gradients([{index: i + 1.0} for i in range(8)])

    def test_exact_parameter_packing_and_fixed_readout_mask(self):
        schema = D.parameter_schema()
        self.assertEqual(len(schema), 10)
        self.assertEqual(schema[0][0], 'base.generator.blocks.22.conv1.bias')
        self.assertEqual(schema[-1], ('base.generator.blocks.24.weight', (3, 64, 3, 3)))
        mask = D.readout_subspace_mask(np=np)
        self.assertEqual(mask.dtype, np.dtype(bool))
        self.assertEqual(int(mask.sum()), 1731)
        self.assertFalse(mask[:74112].any())
        self.assertTrue(mask[74112:].all())

    def test_aligned_singular_tasks_have_known_common_direction(self):
        result = D.solve_common_descent(self.aligned(), np=np)
        self.assertTrue(result['certified'])
        self.assertEqual(result['support_solves'], 255)
        self.assertTrue(np.allclose(result['gram'], 1, atol=1e-14, rtol=0))
        self.assertAlmostEqual(result['combined_vector'][0], 1, places=13)
        self.assertEqual(np.count_nonzero(result['combined_vector'][1:]), 0)
        self.assertTrue(all(value > .999999999 for value in result['unit_gradient_dot_candidate']))

    def test_orthogonal_tasks_analytic_hull_minimum(self):
        result = D.solve_common_descent(self.gradients([{i: 1} for i in range(8)]), np=np)
        self.assertTrue(result['certified'])
        np.testing.assert_allclose(result['weights'], np.full(8, .125), atol=2e-13, rtol=0)
        self.assertAlmostEqual(result['candidate_norm_squared'], .125, places=12)
        np.testing.assert_allclose(result['combined_vector'][:8], .125, atol=2e-13, rtol=0)

    def test_opposing_tasks_cannot_be_reported_certified(self):
        result = D.solve_common_descent(self.gradients([{0: 1}] * 4 + [{0: -1}] * 4), np=np)
        self.assertFalse(result['certified'])
        self.assertFalse(result['failure_is_infeasibility_proof'])
        self.assertLess(result['candidate_norm_squared'], 1e-20)

    def test_zero_task_is_not_discarded(self):
        gradients = self.aligned()
        gradients[D.TASK_NAMES[7]][:] = 0
        result = D.solve_common_descent(gradients, np=np)
        self.assertFalse(result['certified'])
        self.assertEqual(result['support_solves'], 0)

    def test_nan_and_infinity_stop_without_lstsq(self):
        for value in (np.nan, np.inf, -np.inf):
            with self.subTest(value=value):
                gradients = self.aligned()
                gradients[D.TASK_NAMES[0]][0] = value
                result = D.solve_common_descent(gradients, np=np)
                self.assertFalse(result['certified'])
                self.assertEqual(result['support_solves'], 0)

    def test_subnormal_scale_stops_but_small_normal_values_do_not_underflow(self):
        tiny = self.gradients([{0: 1e-45}] * 8)
        self.assertFalse(D.solve_common_descent(tiny, np=np)['certified'])
        normal = self.gradients([{0: 1e-30}] * 8)
        self.assertTrue(D.solve_common_descent(normal, np=np)['certified'])

    def test_nearly_opposing_direction_is_numerically_uncertified(self):
        gradients = self.gradients([{0: 1, 1: 1e-8}] * 4 + [{0: -1, 1: 1e-8}] * 4)
        result = D.solve_common_descent(gradients, np=np)
        self.assertFalse(result['certified'])
        self.assertFalse(result['failure_is_infeasibility_proof'])

    def test_broader_path_can_offer_common_direction_absent_from_readout(self):
        gradients = self.gradients([{0: 1, D.READOUT_START: 1}] * 4 +
                                   [{0: 1, D.READOUT_START: -1}] * 4)
        result = D.compare_subspaces(gradients, np=np)
        self.assertTrue(result['expanded']['certified'])
        self.assertFalse(result['readout']['certified'])
        self.assertFalse(result['selection_after_observing_results'])
        self.assertTrue(np.allclose(result['expanded']['combined_vector'][D.READOUT_START:], 0,
                                    atol=1e-12, rtol=0))

    def test_readout_zero_projection_is_reported_not_silently_replaced(self):
        result = D.compare_subspaces(self.aligned(), np=np)
        self.assertTrue(result['expanded']['certified'])
        self.assertFalse(result['readout']['certified'])
        self.assertEqual(result['readout']['support_solves'], 0)

    def test_each_task_positive_rescaling_preserves_unit_direction(self):
        gradients = self.gradients([{i: i + 1, 8: 1} for i in range(8)])
        baseline = D.solve_common_descent(gradients, np=np)
        changed = {name: row * 10.0 ** (i - 4) for i, (name, row) in enumerate(gradients.items())}
        scaled = D.solve_common_descent(changed, np=np)
        self.assertTrue(baseline['certified'])
        self.assertTrue(scaled['certified'])
        np.testing.assert_allclose(baseline['combined_vector'], scaled['combined_vector'], atol=1e-12, rtol=0)

    def test_exact_tasks_and_extent_are_mandatory(self):
        gradients = self.aligned()
        gradients['undeclared'] = gradients.pop(D.TASK_NAMES[-1])
        with self.assertRaises(ValueError):
            D.solve_common_descent(gradients, np=np)
        gradients = self.aligned()
        gradients[D.TASK_NAMES[0]] = np.ones(8)
        with self.assertRaises(ValueError):
            D.solve_common_descent(gradients, np=np)
        with self.assertRaises(ValueError):
            D.solve_common_descent(self.aligned(), np=np, subspace='favorable_subset')

    def test_two_fp32_proposals_use_one_unchanged_initializer(self):
        gradients = self.aligned(D.READOUT_START)
        solution = D.solve_common_descent(gradients, np=np)
        initializer = np.full(D.PARAMETER_COUNT, .125, np.float32)
        before = initializer.copy()
        result = D.build_fp32_proposals(initializer, gradients, solution, np=np)
        self.assertTrue(result['certified'])
        self.assertFalse(result['stop_before_any_proposal'])
        self.assertEqual([p['maxabs_scale'] for p in result['proposals']], [1e-5, 5e-5])
        np.testing.assert_array_equal(initializer, before)
        for proposal in result['proposals']:
            realized = proposal['realized_displacement_float64']
            np.testing.assert_array_equal(realized, proposal['parameters_fp32'].astype(np.float64) - before)
            self.assertLessEqual(abs(realized[D.READOUT_START]), proposal['maxabs_scale'])
            self.assertTrue(all(value < 0 for value in proposal['mean_gradient_dot_realized_displacement']))
        # The second proposal is five times the intended step, not six times
        # from cumulative application of the first.
        self.assertEqual(result['proposals'][1]['intended_displacement_float64'][D.READOUT_START], -5e-5)

    def test_nearest_fp32_overshoot_is_rounded_toward_initializer(self):
        gradients = self.aligned()
        solution = D.solve_common_descent(gradients, np=np)
        initializer = np.zeros(D.PARAMETER_COUNT, np.float32)
        result = D.build_fp32_proposals(initializer, gradients, solution, np=np)
        self.assertTrue(result['certified'])
        for proposal in result['proposals']:
            self.assertLessEqual(proposal['maximum_actual_absolute_displacement'], proposal['maxabs_scale'])
            self.assertGreaterEqual(proposal['parameters_fp32'][0], -proposal['maxabs_scale'])

    def test_fp32_spacing_can_erase_direction_and_must_stop_both(self):
        gradients = self.aligned()
        solution = D.solve_common_descent(gradients, np=np)
        initializer = np.full(D.PARAMETER_COUNT, 1e8, np.float32)
        result = D.build_fp32_proposals(initializer, gradients, solution, np=np)
        self.assertFalse(result['certified'])
        self.assertTrue(result['stop_before_any_proposal'])
        self.assertTrue(all(not row['certified'] for row in result['proposals']))
        self.assertTrue(all(row['maximum_actual_absolute_displacement'] == 0 for row in result['proposals']))

    def test_fp32_selective_rounding_can_turn_descent_into_ascent(self):
        # Both task directions share a descent vector with a positive first
        # and negative second coordinate. Erasing the first coordinate can
        # make the second task's original mean ascend instead.
        gradients = self.gradients([{0: 1, 1: 1}] * 4 + [{0: 1, 1: -2}] * 4)
        solution = D.solve_common_descent(gradients, np=np)
        self.assertTrue(solution['certified'])
        initializer = np.zeros(D.PARAMETER_COUNT, np.float32)
        initializer[0] = 1e8
        result = D.build_fp32_proposals(initializer, gradients, solution, np=np)
        self.assertFalse(result['certified'])
        self.assertTrue(result['stop_before_any_proposal'])
        self.assertTrue(any(any(value >= 0 for value in row['mean_gradient_dot_realized_displacement'])
                            for row in result['proposals']))

    def test_binding_and_candidate_tamper_are_refused(self):
        gradients = self.aligned()
        solution = D.solve_common_descent(gradients, np=np)
        initializer = np.zeros(D.PARAMETER_COUNT, np.float32)
        changed = {name: row.copy() for name, row in gradients.items()}
        changed[D.TASK_NAMES[0]][0] = 9
        with self.assertRaises(ValueError):
            D.build_fp32_proposals(initializer, changed, solution, np=np)
        bad = copy.deepcopy(solution)
        bad['combined_vector'][0] = 0
        with self.assertRaises(ValueError):
            D.build_fp32_proposals(initializer, gradients, bad, np=np)

    def test_failed_solution_cannot_be_applied_or_force_certified(self):
        gradients = self.gradients([{0: 1}] * 4 + [{0: -1}] * 4)
        solution = D.solve_common_descent(gradients, np=np)
        initializer = np.zeros(D.PARAMETER_COUNT, np.float32)
        result = D.build_fp32_proposals(initializer, gradients, solution, np=np)
        self.assertEqual(result['proposals'], [])
        self.assertTrue(result['stop_before_any_proposal'])
        solution['certified'] = True
        result = D.build_fp32_proposals(initializer, gradients, solution, np=np)
        self.assertEqual(result['proposals'], [])

    def test_source_imports_no_framework_numpy_or_execution_modules(self):
        source = Path(D.__file__).read_text(encoding='utf-8')
        tree = ast.parse(source)
        imports = {node.module for node in ast.walk(tree) if isinstance(node, ast.ImportFrom)}
        imports.update(alias.name for node in ast.walk(tree) if isinstance(node, ast.Import) for alias in node.names)
        self.assertEqual(imports, {'hashlib', 'math', 'objective'})


if __name__ == '__main__':
    unittest.main()
