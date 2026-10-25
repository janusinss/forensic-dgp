"""Pure protocol/NumPy geometry fixtures; no Torch, autograd or model."""
import builtins
import copy
import importlib.util
import json
import math
from pathlib import Path
import sys
import unittest
from unittest import mock

import numpy as np

from research.codeformer_u64_role_gradients_v1 import analysis as A, contract as C

ROOT = Path(__file__).resolve().parents[1]
PARENT = (ROOT / 'outputs' / C.PARENT_STAGE / 'protocol.json').read_bytes()
STANDARD = (ROOT / 'outputs' / C.PARENT_STAGE / 'evidence/u64_parent_protocol.json').read_bytes()


def tensor_map(value):
    return {name: np.full(shape, value, np.float32) for name, shape in C.head_shapes().items()}


def tiny_bank(objectives):
    return {role: {'objective_div4': np.asarray(objectives[index], np.float64),
                   'gate_mae': np.array([1., 0.]), 'gate_adjacent': np.array([0., 1.])}
            for index, role in enumerate(C.ROLES)}


class RoleGradientArithmetic(unittest.TestCase):
    def test_import_is_inert_under_Torch_and_numpy_import_blocker(self):
        original = builtins.__import__
        def blocked(name, *args, **kwargs):
            if name.split('.')[0] in ('torch', 'numpy'):
                self.fail('Numerical framework import in inert source')
            return original(name, *args, **kwargs)
        path = ROOT / 'research/codeformer_u64_role_gradients_v1/analysis.py'
        spec = importlib.util.spec_from_file_location('research.codeformer_u64_role_gradients_v1.inert_analysis_fixture', path)
        with mock.patch('builtins.__import__', side_effect=blocked):
            spec.loader.exec_module(importlib.util.module_from_spec(spec))
        self.assertNotIn('torch', sys.modules)

    def test_exact_two_protocol_selection_and_matched_four_role_order(self):
        selected = C.select_cases(PARENT, STANDARD)
        self.assertEqual(tuple(len(selected[key]) for key in ('optimization', 'monitors', 'guards')), (64, 32, 24))
        self.assertEqual(len(set(selected['optimization_reference_order'])), 16)
        for offset in range(0, 64, 4):
            rows = selected['optimization'][offset:offset + 4]
            self.assertEqual([row['role'] for row in rows], list(C.ROLES))
            self.assertEqual(len({row['reference_id'] for row in rows}), 1)
            self.assertTrue(all(row['target'] == rows[0]['target'] for row in rows))
        self.assertFalse(selected['independent_holdout'])

    def test_changed_protocol_bytes_and_nonfinite_duplicate_JSON_refused(self):
        for parent, standard in ((PARENT + b' ', STANDARD), (PARENT, STANDARD[:-1] + b' ')):
            with self.assertRaises(ValueError): C.select_cases(parent, standard)
        for raw in (b'{"x":1,"x":2}', b'{"x":NaN}', b'{"x":1e999}'):
            with self.assertRaises(ValueError): C.strict_json(raw)

    def test_exact72_forward192_query_no_update_contract(self):
        selected = C.select_cases(PARENT, STANDARD); plan = C.operation_plan(selected)
        self.assertEqual(len(plan), 72)
        self.assertEqual(sum(len(row['gradient_queries']) for row in plan), 192)
        self.assertEqual([row['id'] for row in plan[:4]], [row['id'] for row in plan[4:8]])
        self.assertTrue(all(row['gradient_queries'] == [] for row in plan[:8]))
        self.assertEqual([row['id'] for row in plan[8:]], [row['id'] for row in selected['optimization']])
        self.assertEqual([row['metric_region'] for row in plan[8:12]], ['central128', 'whole', 'whole', 'whole'])
        self.assertEqual((C.LIMITS['worker_seconds'], C.LIMITS['external_seconds'], C.LIMITS['export_seconds'], C.LIMITS['export_external_seconds']), (900, 960, 180, 210))
        self.assertTrue(all(C.LIMITS[name] == 0 for name in ('optimizer_constructions', 'optimizer_updates', 'parameter_applications')))

    def test_monitor_or_reordered_case_cannot_receive_measurement_queries(self):
        selected = C.select_cases(PARENT, STANDARD)
        for change in ('monitor', 'order', 'target'):
            bad = copy.deepcopy(selected)
            if change == 'monitor': bad['optimization'][0] = copy.deepcopy(bad['monitors'][0])
            elif change == 'order': bad['optimization'][0], bad['optimization'][1] = bad['optimization'][1], bad['optimization'][0]
            else: bad['optimization'][1]['target']['sha256'] = '0' * 64
            with self.assertRaises(ValueError): C.operation_plan(bad)

    def test_independent36_head_schema_and_C_order_pack(self):
        shapes = C.head_shapes()
        self.assertEqual((len(shapes), sum(math.prod(s) for s in shapes.values())), (36, 168960))
        values = tensor_map(0)
        ordered = list(shapes)
        values[ordered[0]].flat[0] = 3
        values[ordered[-1]].flat[-1] = 7
        packed = A.pack_gradients(values, np=np)
        self.assertEqual((packed[0], packed[-1], np.count_nonzero(packed)), (3, 7, 2))

    def test_missing_wrongdtype_shape_nonfinite_disconnected_tensor_refused(self):
        good = tensor_map(0); name = next(iter(good))
        for change in ('missing', 'dtype', 'shape', 'nonfinite', 'none'):
            bad = dict(good)
            if change == 'missing': bad.pop(name)
            elif change == 'dtype': bad[name] = good[name].astype(np.float64)
            elif change == 'shape': bad[name] = np.zeros(1, np.float32)
            elif change == 'nonfinite': bad[name] = np.full(good[name].shape, np.inf, np.float32)
            else: bad[name] = None
            with self.assertRaises(ValueError): A.pack_gradients(bad, np=np)

    def test_dot_norm_cosine_and_zero_are_not_omitted(self):
        result = A.comparison(np.array([3., 4.]), np.array([4., -3.]), np=np)
        self.assertEqual((result['dot'], result['left_norm'], result['right_norm'], result['cosine']), (0., 5., 5., 0.))
        zero = A.comparison(np.zeros(2), np.ones(2), np=np)
        self.assertTrue(zero['left_zero']); self.assertIsNone(zero['cosine'])

    def test_objective_div4_is_summed_once_and_sign_is_metric_descent(self):
        bank = tiny_bank([[1., 0.]] * 4)
        result = A.summarize_vectors(bank, np=np)
        self.assertEqual(result['joint_objective_div4_sum_norm'], 4.)
        self.assertEqual(result['joint_first_order_derivatives']['gate_mae']['clear']['steepest_descent_first_order_derivative'], -4.)
        self.assertEqual(result['joint_first_order_derivatives']['gate_adjacent']['clear']['steepest_descent_first_order_derivative'], 0.)
        self.assertFalse(result['finite_step_or_PNG_prediction'])

    def test_conflict_cancellation_and_scale_are_separate(self):
        result = A.summarize_vectors(tiny_bank([[1., 0.], [-1., 0.], [0., 0.], [0., 0.]]), np=np)
        self.assertEqual(result['objective_role_pairwise'][C.ROLES[0]][C.ROLES[1]]['cosine'], -1.)
        self.assertTrue(result['joint_objective_zero'])
        self.assertIsNone(result['joint_first_order_derivatives']['gate_mae']['clear']['cosine_with_joint_objective'])
        scaled = A.summarize_vectors(tiny_bank([[1., 0.], [0., 100.], [0., 0.], [0., 0.]]), np=np)
        self.assertEqual(scaled['gradient_norms'][C.ROLES[1]]['objective_div4'], 100.)
        self.assertEqual(scaled['joint_first_order_derivatives']['gate_adjacent']['clear']['steepest_descent_first_order_derivative'], -100.)

    def test_fresh_Adam_direction_is_only_arithmetic_with_clipping_and_epsilon(self):
        g = np.array([3., 4.]); before = g.copy(); value = A.hypothetical_fresh_adam(g, np=np)
        clipped = g / (5. + 1e-6)
        np.testing.assert_allclose(value['direction'], -2.5e-5 * clipped / (np.abs(clipped) + 1e-8), rtol=0, atol=1e-20)
        self.assertTrue(np.array_equal(g, before)); self.assertFalse(value['optimizer_constructed']); self.assertFalse(value['parameter_applied'])
        tiny = A.hypothetical_fresh_adam(np.array([1e-12]), np=np)
        self.assertLess(abs(tiny['direction'][0]), 3e-9)
        self.assertEqual(A.hypothetical_fresh_adam(np.zeros(2), np=np)['direction'].tolist(), [0., 0.])

    def test_finite_extremes_cannot_produce_nonfinite_or_underflow_zero_report(self):
        with np.errstate(all='ignore'):
            for value in (np.array([1e308, 1e308]), np.array([1e-300])):
                with self.assertRaises(ValueError): A.hypothetical_fresh_adam(value, np=np)
            with self.assertRaises(ValueError): A.comparison(np.array([1e308]), np.array([1e308]), np=np)

    def test_full64_photo_and_pooled_cancellation_preserve_every_case(self):
        selected = C.select_cases(PARENT, STANDARD)
        plus, minus = tensor_map(1), tensor_map(-1)
        rows = []
        for index, row in enumerate(selected['optimization']):
            obj = minus if row['role'] == 'clear' and (index // 4) % 2 else plus
            rows.append({'id': row['id'], 'reference_id': row['reference_id'], 'role': row['role'],
                'metric_region': C.metric_region(row['role']),
                'gradients': {'objective_div4': obj, 'gate_mae': plus, 'gate_adjacent': plus}})
        before = next(iter(plus.values())).copy()
        result = A.analyze(rows, selected, np=np)
        self.assertEqual((len(result['per_photograph']), len(result['original_offset_sampler_at_common_state'])), (16, 16))
        cancellation = result['pooled_cancellation_diagnostics']['clear']['objective_div4']
        self.assertGreater(cancellation['mean_individual_norm'], 0)
        self.assertEqual(cancellation['pooled_mean_norm'], 0)
        self.assertEqual(result['pooled_role_mean']['joint_first_order_derivatives']['gate_mae']['clear']['steepest_descent_first_order_derivative'], -3 * 168960)
        self.assertTrue(np.array_equal(before, next(iter(plus.values()))))
        json.dumps(result, allow_nan=False)
        self.assertFalse(result['model_qualified']); self.assertFalse(result['common_descent_or_feasibility_certificate'])

    def test_incomplete_or_wrong_metadata_bank_is_refused_before_imputation(self):
        selected = C.select_cases(PARENT, STANDARD)
        with self.assertRaises(ValueError): A.analyze([], selected, np=np)
        bad = [{'id': 'wrong', 'reference_id': 'wrong', 'role': 'clear', 'metric_region': 'whole', 'gradients': {}}] * 64
        with self.assertRaises(ValueError): A.analyze(bad, selected, np=np)
        bank = tiny_bank([[1., 0.]] * 4); bank['clear']['gate_mae'] = None
        with self.assertRaises(ValueError): A.summarize_vectors(bank, np=np)


if __name__ == '__main__':
    unittest.main()
