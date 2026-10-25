"""Saved-state NumPy and passive-pickle miniatures; no Torch/model imports."""
import ast
import builtins
import copy
import hashlib
import importlib.util
import io
import math
from pathlib import Path
import pickle
import random
import struct
import sys
import tempfile
import unittest
from unittest import mock
import zipfile

import numpy as np

ROOT = Path(__file__).resolve().parents[1]
SOURCE = ROOT / 'scripts/audit_matched_feature_return_state_v1.py'
spec = importlib.util.spec_from_file_location('matched_state_audit_under_test', SOURCE)
A = importlib.util.module_from_spec(spec); spec.loader.exec_module(A)


def unicode(value):
    raw = value.encode(); return b'X' + struct.pack('<I', len(raw)) + raw


def tensor_pickle(*, count=2, shape=(2,), stride=(1,), offset=0, requires_grad=False):
    def integer(value):
        return b'J' + struct.pack('<i', value)
    def tuple_value(values):
        return b'(' + b''.join(integer(v) for v in values) + b't'
    return (b'\x80\x02}' + unicode('x') + b'ctorch._utils\n_rebuild_tensor_v2\n(' +
            b'(' + unicode('storage') + b'ctorch\nFloatStorage\n' + unicode('0') + unicode('cpu') + integer(count) + b'tQ' +
            integer(offset) + tuple_value(shape) + tuple_value(stride) +
            (b'\x88' if requires_grad else b'\x89') + b'ccollections\nOrderedDict\n)RtRs.')


def optimizer(update, schema):
    names = sorted(schema)
    opt = {'state': {}, 'param_groups': [{'params': list(range(len(names))),
        'lr': A.learning_rate(update), 'betas': (.9, .99), 'eps': 1e-8, 'weight_decay': 0.,
        'amsgrad': False, 'foreach': False, 'fused': False, 'capturable': False,
        'differentiable': False, 'maximize': False}]}
    if update:
        opt['state'] = {i: {'step': np.array(update, np.float32),
            'exp_avg': np.zeros(schema[n], np.float32), 'exp_avg_sq': np.zeros(schema[n], np.float32)}
            for i, n in enumerate(names)}
    return opt


def graph(arm):
    return {'status': 'WITNESSED', 'complete': True, 'grad_enabled': True, 'arm': arm,
        'sites': {site: {branch: {'pass': True, 'expected_present': arm == 'B',
            'current_adapter_leaves_reached': 6 if arm == 'B' else 0,
            'autograd_queries_or_backward_calls': 0, 'nonzero_or_finite_branch_gradient_proven': False}
            for branch in ('stock_SFT', 'head_encoder')} for site in ('32', '64', '128', '256')}}


class SavedStateMiniatures(unittest.TestCase):
    def setUp(self):
        (ROOT / 'scratch').mkdir(exist_ok=True)
        self.temp = tempfile.TemporaryDirectory(dir=ROOT / 'scratch', prefix='matched_state_unit_')
        self.addCleanup(self.temp.cleanup); self.root = Path(self.temp.name)

    def zip(self, raw, *, count=2, extra=None):
        path = self.root / 'closed.pt'
        with zipfile.ZipFile(path, 'w', compression=zipfile.ZIP_STORED) as archive:
            archive.writestr('archive/data.pkl', raw)
            archive.writestr('archive/byteorder', 'little')
            archive.writestr('archive/version', '3\n')
            if count: archive.writestr('archive/data/0', b'\0' * count * 4)
            if extra: archive.writestr(extra[0], extra[1])
        return path

    def test_import_is_stdlib_only_even_with_neural_numpy_blocker(self):
        original = builtins.__import__
        def blocked(name, *args, **kwargs):
            if name.split('.')[0] in ('torch', 'numpy', 'PIL') or name.endswith(('.model', '.worker')):
                raise AssertionError('Forbidden source import ' + name)
            return original(name, *args, **kwargs)
        fresh = importlib.util.spec_from_file_location('matched_state_fresh', SOURCE)
        with mock.patch('builtins.__import__', side_effect=blocked):
            fresh.loader.exec_module(importlib.util.module_from_spec(fresh))

    def test_independent_arm_schemas_exact_counts_coordinates_and_subset(self):
        a, b = A.shapes('A'), A.shapes('B')
        self.assertEqual((len(a), sum(math.prod(s) for s in a.values())), (36, 168960))
        self.assertEqual((len(b), sum(math.prod(s) for s in b.values())), (60, 220288))
        self.assertEqual({k: v for k, v in b.items() if k.startswith('heads.')}, a)
        self.assertEqual(len(set(b) - set(a)), 24)
        for value in ('a', None, True, 'C'):
            with self.assertRaises(ValueError): A.shapes(value)

    def test_learning_rate_finite_initial_and_four_update_warmup(self):
        self.assertEqual([A.learning_rate(u) for u in (0, 1, 2, 3, 4, 8, 48)],
                         [1e-5, 2.5e-6, 5e-6, 7.500000000000001e-6, 1e-5, 1e-5, 1e-5])
        for value in (-1, 49, True, 1.0, '1'):
            with self.assertRaises(ValueError): A.learning_rate(value)

    def test_passive_protocol2_primitive_and_tensor_pickle_are_supported(self):
        primitive = pickle.dumps({'key': [1, False, None, .5], 'tuple': (2, 'cpu')}, protocol=2)
        result = A.preflight_pickle(primitive, {})
        self.assertTrue(result['passive_only']); self.assertFalse(result['reducer_execution'])
        tensor = A.preflight_pickle(tensor_pickle(), {'0': 8})
        self.assertEqual((tensor['storage_count'], tensor['tensor_count']), (1, 1))

    def test_whitelisted_reducers_are_never_executed_during_preflight(self):
        with mock.patch('builtins.__import__', side_effect=AssertionError('Unexpected import')):
            self.assertEqual(A.preflight_pickle(tensor_pickle(), {'0': 8})['tensor_count'], 1)

    def test_pickle_rejects_any_unlisted_global(self):
        for raw in (b'\x80\x02cos\nsystem\n.', b'\x80\x02cbuiltins\neval\n.',
                    b'\x80\x02ctorch\nLongStorage\n.'):
            with self.assertRaisesRegex(ValueError, 'GLOBAL'):
                A.preflight_pickle(raw, {})

    def test_pickle_rejects_unsupported_protocol_opcodes_and_trailing_bytes(self):
        for raw in (pickle.dumps({}, protocol=4), tensor_pickle() + b'hidden', b'\x80\x02}b.',
                    b'\x80\x02}q\x00h\x01s.', b'\x80\x02cos\nsystem\n)R.'):
            with self.subTest(raw=raw), self.assertRaises((ValueError, IndexError)):
                A.preflight_pickle(raw, {})

    def test_tensor_storage_must_match_exact_verified_bytes(self):
        for mapping in ({'0': 4}, {}, {'0': 8, '1': 4}):
            with self.subTest(mapping=mapping), self.assertRaises(ValueError):
                A.preflight_pickle(tensor_pickle(), mapping)

    def test_tensor_view_refuses_out_of_storage_and_huge_shapes(self):
        for kwargs in ({'shape': (3,)}, {'offset': 2}, {'stride': (3,)}, {'shape': (0,)},
                       {'shape': (65537,)}, {'requires_grad': True}, {'count': -1}):
            with self.subTest(kwargs=kwargs), self.assertRaises(ValueError):
                A.preflight_pickle(tensor_pickle(**kwargs), {'0': 8})

    def test_scalar_tensor_view_is_allowed_with_one_float_storage(self):
        result = A.preflight_pickle(tensor_pickle(count=1, shape=(), stride=()), {'0': 4})
        self.assertEqual(result['tensor_count'], 1)

    def test_real_zip_preflight_checks_crc_metadata_and_payload_scope_without_torch(self):
        path = self.zip(tensor_pickle())
        result = A.pt_preflight(path)
        self.assertEqual(result['tensor_count'], 1)
        self.assertTrue(result['CPU_only_and_weights_only_required'])
        self.assertNotIn('torch', sys.modules)

    def test_zip_rejects_unknown_member_and_storage_mismatch(self):
        for extra in (('../escape', b'x'), ('archive/unexpected', b'x')):
            path = self.zip(tensor_pickle(), extra=extra)
            with self.assertRaises(ValueError): A.pt_preflight(path)
            path.unlink()
        path = self.zip(tensor_pickle(), count=3)
        with self.assertRaises(ValueError): A.pt_preflight(path)

    def test_zip_rejects_compressed_duplicate_or_appended_zip_payload(self):
        path = self.zip(tensor_pickle())
        path.write_bytes(path.read_bytes() + b'junk')
        with self.assertRaises(ValueError): A.pt_preflight(path)
        path.unlink()
        with zipfile.ZipFile(path, 'w', compression=zipfile.ZIP_DEFLATED) as archive:
            archive.writestr('archive/data.pkl', pickle.dumps({}, protocol=2))
        with self.assertRaises(ValueError): A.pt_preflight(path)

    def test_exact_saved_optimizer_zero_and_complete_B_moments(self):
        for arm in ('A', 'B'):
            schema = A.shapes(arm); names = sorted(schema)
            for update in (0, 1, 8):
                actual = A.validate_optimizer(optimizer(update, schema), update, names, schema, np)
                self.assertEqual(set(actual), set(names))
                self.assertTrue(all(a.dtype == np.float32 for pair in actual.values() for a in pair))

    def test_optimizer_refuses_changed_policy_order_missing_state_and_bool_updates(self):
        schema = {'heads.test': (2,)}; names = sorted(schema)
        changes = [lambda x: x['param_groups'][0].update(lr=.0001),
                   lambda x: x['param_groups'][0].update(params=[1]),
                   lambda x: x['param_groups'][0].update(fused=True),
                   lambda x: x['state'].clear(),
                   lambda x: x['state'][0].update(step=np.array(2, np.float32)),
                   lambda x: x['state'][0].update(exp_avg=np.zeros((3,), np.float32)),
                   lambda x: x['state'][0].update(exp_avg_sq=np.array([-1, 0], np.float32))]
        for change in changes:
            value = optimizer(1, schema); change(value)
            with self.assertRaises(ValueError): A.validate_optimizer(value, 1, names, schema, np)
        with self.assertRaises(ValueError): A.validate_optimizer(optimizer(1, schema), True, names, schema, np)

    def test_saved_array_Adam_recurrence_and_tolerance_detects_tampering(self):
        H = A.helper(); p = np.array([.1, -.3], np.float32); g = np.array([.2, -.5], np.float32)
        zero = np.zeros(2, np.float32)
        m, v, updated = H.adam_predict(p, g, zero, zero, 1, A.learning_rate(1), np)
        np.testing.assert_allclose(m, g * .1, atol=1e-9, rtol=1e-6)
        np.testing.assert_allclose(v, g * g * .01, atol=1e-9, rtol=1e-6)
        self.assertGreater(np.max(np.abs(p - updated)), 0)
        self.assertEqual(H.compare(updated, updated.copy(), np, 'parameter_update'), 0)
        with self.assertRaises(ValueError): H.compare(updated, updated + .01, np, 'parameter_update')

    def test_actual_displacement_has_no_added_quality_tolerance(self):
        zero = {'x': np.zeros(2, np.float32)}
        after = {'x': np.array([.00001, -.00001], np.float32)}
        self.assertTrue(A.independent_displacement(zero, after, zero, np)['pass'])
        self.assertFalse(A.independent_displacement(zero, {'x': np.array([.01, 0], np.float32)}, zero, np)['pass'])
        self.assertFalse(A.independent_displacement(zero, zero, zero, np)['pass'])

    def test_diagnostic_norm_representation_eight_ULPs_only_and_nine_refused(self):
        reference = .03; changed = reference
        for _ in range(8): changed = math.nextafter(changed, math.inf)
        result = A.diagnostic_norm_check({'site': changed}, {'site': reference})
        self.assertTrue(result['pass']); self.assertEqual(result['sites']['site']['ULP_distance'], 8)
        self.assertTrue(result['representation_only_not_quality_epsilon'])
        changed = math.nextafter(changed, math.inf)
        self.assertFalse(A.diagnostic_norm_check({'site': changed}, {'site': reference})['pass'])

    def test_diagnostic_norm_zero_finiteness_positive_admission_remain_strict(self):
        self.assertTrue(A.diagnostic_norm_check({'s': 0.}, {'s': 0.})['pass'])
        self.assertFalse(A.diagnostic_norm_check({'s': math.ulp(0.)}, {'s': 0.})['pass'])
        for value in (True, -1., math.nan, math.inf):
            with self.subTest(value=value), self.assertRaises(ValueError):
                A.diagnostic_norm_check({'s': value}, {'s': 1.})
        with self.assertRaises(ValueError): A.diagnostic_norm_check({'s': 1.}, {'other': 1.})

    def test_graph_receipts_require_both_branches_four_sites_and_nonzero_claim_false(self):
        for arm in ('A', 'B'): A.graph_witness(graph(arm), arm)
        changes = [lambda x: x['sites'].pop('32'),
                   lambda x: x['sites']['32'].pop('head_encoder'),
                   lambda x: x['sites']['32']['stock_SFT'].update(current_adapter_leaves_reached=0),
                   lambda x: x['sites']['32']['head_encoder'].update(autograd_queries_or_backward_calls=1),
                   lambda x: x['sites']['32']['head_encoder'].update(nonzero_or_finite_branch_gradient_proven=True)]
        for change in changes:
            value = graph('B'); change(value)
            with self.assertRaises(ValueError): A.graph_witness(value, 'B')

    def test_safe_streamed_NPZ_rejects_shape_and_object_payloads(self):
        H = A.helper(); path = self.root / 'state.npz'; np.savez(path, **{'heads.x': np.ones((2,), np.float32)})
        actual = H.read_npz(path, {'heads.x': (2,)}, np)
        self.assertEqual(actual['counts']['heads'], 1)
        with self.assertRaises(ValueError): H.read_npz(path, {'heads.x': (3,)}, np)
        path.unlink(); np.savez(path, **{'heads.x': np.array([{}], dtype=object)})
        with self.assertRaises(ValueError): H.read_npz(path, {'heads.x': (1,)}, np)

    def test_json_duplicate_nonfinite_and_historicalhelper_tamper_fail_closed(self):
        for raw in (b'{"x":1,"x":2}', b'{"x":1e999}', b'[NaN]'):
            with self.assertRaises(ValueError): A.strict_json(raw)
        with mock.patch.object(A, 'HELPER_PIN', (A.HELPER_PIN[0], '0' * 64)), self.assertRaises(ValueError):
            A.helper()

    def test_no_neural_execution_and_CPU_decode_flags_are_literal(self):
        tree = ast.parse(SOURCE.read_text(), feature_version=(3, 10))
        calls = [n for n in ast.walk(tree) if isinstance(n, ast.Call)]
        forbidden = {'step', 'backward', 'grad', 'jvp', 'vjp', 'forward', 'train', 'Adam', 'CodeFormer', 'build_restorer'}
        self.assertFalse({getattr(c.func, 'attr', getattr(c.func, 'id', '')) for c in calls} & forbidden)
        load = [c for c in calls if isinstance(c.func, ast.Attribute) and c.func.attr == 'load']
        self.assertEqual(len(load), 1)
        self.assertEqual({k.arg: ast.literal_eval(k.value) for k in load[0].keywords}, {'map_location': 'cpu', 'weights_only': True})
        self.assertNotIn('torch', sys.modules)


if __name__ == '__main__':
    unittest.main()
