"""Pure saved-array/metadata miniatures; no Torch, model, optimizer or archive."""
import ast
import builtins
import copy
import hashlib
import importlib.util
import io
import json
import os
from pathlib import Path
import pickle
import random
import subprocess
import tempfile
import unittest
from unittest import mock
import zipfile

import numpy as np

ROOT = Path(__file__).resolve().parents[1]
SOURCE = ROOT/'scripts/audit_u64_cleananchors_return_state_v1.py'
SPEC = importlib.util.spec_from_file_location('u64_saved_state_audit_under_test', SOURCE)
A = importlib.util.module_from_spec(SPEC)
SPEC.loader.exec_module(A)
REAL_CHILD_DIAGNOSTIC = None


def rng():
    return {'Python': repr(random.Random(7).getstate()),
            'NumPy': {'algorithm': 'MT19937', 'keys_hex': '00'*(624*4), 'position': 624,
                      'has_gauss': 0, 'cached_gaussian': 0.},
            'CPU_hex': '00'*5056, 'CUDA_hex': ['00'*16]}


def optimizer(update, shapes):
    names = sorted(shapes)
    result = {'state': {}, 'param_groups': [{'params': list(range(len(names))), 'lr': A.learning_rate(update),
              'betas': (.9, .99), 'eps': 1e-8, 'weight_decay': 0., 'amsgrad': False,
              'foreach': False, 'fused': False, 'capturable': False, 'differentiable': False, 'maximize': False}]}
    if update:
        result['state'] = {i: {'step': np.array(update, np.float32),
                              'exp_avg': np.zeros(shapes[n], np.float32),
                              'exp_avg_sq': np.zeros(shapes[n], np.float32)} for i, n in enumerate(names)}
    return result


class SavedValueAuditTests(unittest.TestCase):
    def setUp(self):
        self.temp = tempfile.TemporaryDirectory(dir=ROOT/'outputs', prefix='u64_state_audit_test_')
        self.addCleanup(self.temp.cleanup)
        self.folder = Path(self.temp.name)

    def npz(self, arrays, name='values.npz', compressed=False):
        path = self.folder/name
        (np.savez_compressed if compressed else np.savez)(path, **arrays)
        return path

    def test_import_inert_even_with_framework_import_blocker(self):
        original = builtins.__import__
        def blocked(name, *args, **kwargs):
            if name.split('.')[0] in ('torch', 'numpy'):
                self.fail('Audit module attempted numerical import on import')
            return original(name, *args, **kwargs)
        spec = importlib.util.spec_from_file_location('u64_saved_state_fresh_inert', SOURCE)
        with mock.patch('builtins.__import__', side_effect=blocked):
            spec.loader.exec_module(importlib.util.module_from_spec(spec))

    def test_schema_and_checkpoint_counts_before_quality_snapshots(self):
        schema = A.head_shapes()
        self.assertEqual((len(schema), sum(np.prod(s) for s in schema.values())), (36, 168960))
        self.assertEqual([A.expected_counts(u)['full_forward_returns'] for u in (0, 4, 5, 8, 9, 16)], [60, 76, 136, 148, 208, 236])
        np.testing.assert_allclose([A.learning_rate(u) for u in (0, 1, 2, 3, 4, 16)], [.0001, .000025, .00005, .000075, .0001, .0001], rtol=0, atol=1e-19)
        for value in (-1, 17, True):
            with self.assertRaises(ValueError): A.expected_counts(value)

    def test_known_u8_stop_and_partial_u9_are_distinguished(self):
        c = A.expected_counts(8); c['full_forward_attempts'] += 56; c['full_forward_returns'] += 56
        c['candidate_snapshot_forwards_after_initializer'] += 56
        result = A.operation_coverage(c, [], 8, False)
        self.assertTrue(result['covered']); self.assertEqual(result['later_unattempted'], list(range(9, 17)))
        c['full_forward_attempts'] += 1; c['full_forward_returns'] += 1; c['candidate_training_forwards'] += 1
        result = A.operation_coverage(c, [], 8, False)
        self.assertEqual(result['partially_attempted_update'], 9)
        self.assertEqual(result['later_unattempted'], list(range(10, 17)))

    def test_missing_corrupt_and_unknown_status_never_invent_zero(self):
        for c in (None, {}, {'optimizer_updates': 0}):
            with self.assertRaises(ValueError): A.operation_coverage(c, [], 0, False)
        c = A.expected_counts(8); c['optimizer_update_attempts'] = 9
        result = A.operation_coverage(c, [{'kind': 'optimizer_update'}], 8, True)
        self.assertTrue(result['optimizer_unknown']); self.assertFalse(result['covered'])
        self.assertIsNone(result['later_unattempted'])
        with self.assertRaises(ValueError): A.operation_coverage(c, [{'kind': 'optimizer_update'}], 8, False)

    def test_known_u8_stop_rejects_changed_closing_full575_heads(self):
        coverage = {'covered': True, 'optimizer_unknown': False, 'partially_attempted_update': None}
        good = [{'update': 8, 'heads_match_last_verified_checkpoint': True}]
        self.assertTrue(A.closing_relationship(good, 8, 8, coverage, [])['closing_head_trajectory_verified'])
        bad = [{'update': 8, 'heads_match_last_verified_checkpoint': False}]
        with self.assertRaisesRegex(ValueError, 'Known committed-stop closing575 heads'):
            A.closing_relationship(bad, 8, 8, coverage, [])
        with self.assertRaisesRegex(ValueError, 'cursor binds'):
            A.closing_relationship([{'update': 9, 'heads_match_last_verified_checkpoint': True}], 8, 8, coverage, [])

    def test_closing_unknown_or_later_unverified_state_is_explicit(self):
        bad = [{'update': 9, 'heads_match_last_verified_checkpoint': False}]
        for pending in ([{'kind': 'optimizer_update'}], []):
            result = A.closing_relationship(bad, 8, 9, {'covered': False}, pending)
            self.assertFalse(result['head_equality_required'])
            self.assertFalse(result['closing_head_trajectory_verified'])
            self.assertIsNotNone(result['unvalidated_relationship_exception'])
        # A later forward/backward without a new optimizer commit cannot excuse
        # changed parameters once its operation has returned and none is pending.
        with self.assertRaises(ValueError):
            A.closing_relationship(bad, 8, 9, {'covered': True, 'partially_attempted_update': 9}, [])

    def test_partial_cursor_mutation_cannot_skip_required_Adam_RNG_check(self):
        status = {'active_update': 8}; failure = {'active_update': 8}; coverage = {'covered': True}
        self.assertTrue(A.partial_cursor({'new_update': 8}, status, failure, 8, coverage, [])['committed_Adam_RNG_equality_required'])
        for cursor in (9, 7, True, None):
            with self.subTest(cursor=cursor), self.assertRaisesRegex(ValueError, 'new_update must equal'):
                A.partial_cursor({'new_update': cursor}, status, failure, 8, coverage, [])
        with self.assertRaisesRegex(ValueError, 'Failure/status active-update'):
            A.partial_cursor({'new_update': 8}, status, {'active_update': 9}, 8, coverage, [])

    def test_genuine_partial_cursor_keeps_unvalidated_relationship(self):
        result = A.partial_cursor({'new_update': 9}, {'active_update': 9}, {'active_update': 9},
                                  8, {'covered': False}, [{'kind': 'optimizer_update'}])
        self.assertFalse(result['committed_Adam_RNG_equality_required'])
        self.assertIsNotNone(result['unvalidated_relationship_exception'])

    def test_npz_strict_schema_canonical_and_selective_retention(self):
        values = {'heads.a': np.array([1., 2.], np.float32), 'base.x': np.ones((2, 2), np.float32)}
        path = self.npz(values)
        result = A.read_npz(path, {n: a.shape for n, a in values.items()}, np, retain={'heads.a'})
        self.assertEqual(set(result['arrays']), {'heads.a'})
        self.assertEqual(result['digests']['all'], A.canonical(values, np))
        self.assertEqual(result['digests']['base'], A.canonical({'x': values['base.x']}, np))
        self.assertEqual(result['rows'][0]['bytes_sha256'], hashlib.sha256(values['base.x'].tobytes()).hexdigest())

    def test_npz_rejects_compressed_objects_float64_nan_wrong_shape(self):
        variants = [(np.array([1.], np.float32), True), (np.array([object()], object), False),
                    (np.array([1.], np.float64), False), (np.array([np.nan], np.float32), False),
                    (np.array([1., 2.], np.float32), False)]
        for i, (array, compressed) in enumerate(variants):
            with self.subTest(i=i):
                path = self.npz({'heads.a': array}, str(i)+'.npz', compressed)
                with self.assertRaises(ValueError): A.read_npz(path, {'heads.a': (1,)}, np)

    def test_npz_rejects_trailing_bytes_duplicate_keys_and_oversized_declared_shape(self):
        path = self.npz({'heads.a': np.ones((1,), np.float32)})
        with path.open('ab') as f: f.write(b'junk')
        with self.assertRaises(ValueError): A.read_npz(path, {'heads.a': (1,)}, np)
        stream = io.BytesIO(); np.save(stream, np.ones((1,), np.float32), allow_pickle=False)
        path = self.folder/'duplicate.npz'
        with zipfile.ZipFile(path, 'w') as z:
            z.writestr('heads.a.npy', stream.getvalue()); z.writestr('unexpected.npy', stream.getvalue())
        with self.assertRaises(ValueError): A.read_npz(path, {'heads.a': (1,)}, np)
        with self.assertRaises(ValueError): A.read_npz(path, {'heads.a': (65536, 65536)}, np)

    def test_fresh_Adam_and_policy_are_not_historical_resume(self):
        shapes = {'heads.a': (2,)}
        opt = optimizer(0, shapes); moments = A.validate_optimizer(opt, 0, ['heads.a'], shapes, np)
        self.assertTrue(np.array_equal(moments['heads.a'][0], np.zeros(2, np.float32)))
        for mutate in (lambda o: o['state'].update({0: {}}),
                       lambda o: o['param_groups'][0].update(lr=.00005),
                       lambda o: o['param_groups'][0].update(fused=True),
                       lambda o: o['param_groups'][0].update(params=[1])):
            bad = copy.deepcopy(opt); mutate(bad)
            with self.assertRaises(ValueError): A.validate_optimizer(bad, 0, ['heads.a'], shapes, np)

    def test_moment_census_step_and_finiteness(self):
        shapes = {'heads.a': (2,)}; good = optimizer(8, shapes)
        A.validate_optimizer(good, 8, ['heads.a'], shapes, np)
        for mutate in (lambda o: o['state'][0].update(step=np.array(7, np.float32)),
                       lambda o: o['state'][0].update(step=np.array(8, np.float64)),
                       lambda o: o['state'][0].update(exp_avg_sq=np.array([-1., 0.], np.float32)),
                       lambda o: o['state'][0].update(exp_avg=np.array([np.nan, 0.], np.float32))):
            bad = copy.deepcopy(good); mutate(bad)
            with self.assertRaises(ValueError): A.validate_optimizer(bad, 8, ['heads.a'], shapes, np)

    def test_Adam_recurrence_two_commits_and_corruption_detection(self):
        p = np.array([.1, -.2], np.float32); g = np.array([.25, -.125], np.float32); z = np.zeros(2, np.float32)
        m, v, p1 = A.adam_predict(p, g, z, z, 1, .000025, np)
        np.testing.assert_allclose(m, [.025, -.0125], rtol=1e-6)
        np.testing.assert_allclose(v, [.000625, .00015625], rtol=1e-6)
        np.testing.assert_allclose(p1, p-np.sign(g)*np.float32(.000025), rtol=1e-6, atol=1e-8)
        m2, v2, p2 = A.adam_predict(p1, g, m, v, 2, .00005, np)
        self.assertLess(A.compare(p2, p1-np.sign(g)*np.float32(.00005), np, 'parameter_update'), 6e-8)
        with self.assertRaises(ValueError): A.compare(m2+np.float32(.001), m2, np, 'first_moment')
        with self.assertRaises(ValueError): A.compare(v2+np.float32(.001), v2, np, 'second_moment')

    def test_both_gradient_files_bind_actual_clip(self):
        pre = {'heads.a': np.array([3., 4.], np.float32)}; coef = 1/(5+1e-6)
        post = {'heads.a': pre['heads.a']*np.float32(coef)}
        result = A.check_clipping(pre, post, 5., coef, np)
        self.assertAlmostEqual(result['preclip_L2_float64'], 5.)
        self.assertLessEqual(result['postclip_L2_float64'], 1.000001)
        with self.assertRaises(ValueError): A.check_clipping(pre, post, 4., 1/(4+1e-6), np)
        with self.assertRaises(ValueError): A.check_clipping(pre, {'heads.a': np.zeros(2, np.float32)}, 5., coef, np)
        with self.assertRaises(ValueError): A.check_clipping(pre, post, 5., .2, np)

    def test_RNG_schema_and_corruption(self):
        self.assertEqual(len(A.validate_rng(rng())), 64)
        for key, value in (('CPU_hex', 'xyz'), ('CUDA_hex', []), ('Python', '__import__("os")')):
            bad = rng(); bad[key] = value
            with self.assertRaises((ValueError, SyntaxError)): A.validate_rng(bad)
        bad = rng(); bad['NumPy']['position'] = 625
        with self.assertRaises(ValueError): A.validate_rng(bad)

    def test_new_photographic_objective_scalar_arithmetic(self):
        self.assertAlmostEqual(A.objective_scalar('clear', {'photographic_image_anchor_SmoothL1': .1, 'photographic_adjacent_anchor_SmoothL1': .2}), .3)
        self.assertAlmostEqual(A.objective_scalar('lowlight_linear_0p35', {'target_image_L1': .1, 'target_adjacent_L1': .2, 'photographic_block16_luminance_anchor_SmoothL1': .3}), .75)
        self.assertAlmostEqual(A.objective_scalar('local_gaussian_1p2', {'inside_image_L1': .1, 'outside_image_L1': .2, 'inside_adjacent_L1': .3, 'outside_adjacent_L1': .4}), .2375)
        with self.assertRaises((KeyError, ValueError)): A.objective_scalar('clear', {'U96_image_anchor_SmoothL1': .1})

    def test_independent292_plan_and_one_rolecase_pass(self):
        order = ['opt'+str(i) for i in range(16)]; monitors = ['mon'+str(i) for i in range(8)]
        cases = [{'reference_id': r, 'role': role, 'id': r+'__'+role} for r in order+monitors for role in A.ROLES]
        guards = [{'reference_id': 'guard'+str(i), 'role': role, 'id': 'guard'+str(i)+'__'+role} for i in range(8) for role in A.ROLES[:3]]
        parent = {'learning': {'optimization_reference_order': order, 'monitor_reference_order': monitors}, 'cases': cases, 'guards': guards}
        calls, rows = A.independent_plan(parent, parent)
        training = [x for x in calls if x['backward']]
        self.assertEqual((len(calls), len(training), len({x['id'] for x in training}), len(rows)), (292, 64, 64, 120))
        self.assertEqual([x['slot'] for x in calls if x['new_update'] <= 8][-1], 204)
        self.assertEqual([x['id'].split('__')[0] for x in training[:4]], ['opt0', 'opt4', 'opt8', 'opt12'])

    def test_resume_exact_cursor_frozen_protocol_and_rng(self):
        shapes = {'heads.a': (2,)}; r = {'format': 'u64-cleananchors-closed-finite-state-v1', 'protocol_sha256': A.PROTOCOL_SHA,
            'update': 0, 'automatic_resume_permitted': False, 'optimizer': optimizer(0, shapes), 'optimizer_parameter_names': ['heads.a'],
            'RNG': rng(), 'sampler_next_new_update': 1, 'initial_head_canonical_sha256': A.HEAD_SHA,
            'frozen539_canonical_sha256': 'frozen', 'counters': A.expected_counts(0), 'model_qualified': False}
        A.validate_resume(r, 0, 'frozen', np, shapes)
        for key, value in (('protocol_sha256', '0'*64), ('initial_head_canonical_sha256', '0'*64), ('sampler_next_new_update', 65), ('automatic_resume_permitted', True)):
            bad = copy.deepcopy(r); bad[key] = value
            with self.assertRaises(ValueError): A.validate_resume(bad, 0, 'frozen', np, shapes)

    def test_PT_preflight_is_nonexecuting_and_rejects_globals_storage_bombs(self):
        def write(metadata, name):
            path = self.folder/name
            with zipfile.ZipFile(path, 'w') as z: z.writestr('archive/data.pkl', metadata); z.writestr('archive/data/0', b'0000')
            return path
        A.pt_preflight(write(pickle.dumps({'value': 1.}, protocol=2), 'simple.pt'))
        for i, metadata in enumerate((b'cos\nsystem\n.', pickle.dumps({'x': 2**63}, protocol=2), pickle.dumps({'x': float('inf')}, protocol=2))):
            with self.assertRaises(ValueError): A.pt_preflight(write(metadata, str(i)+'.pt'))

    def test_manifest_exactsubset_and_changed_bytes(self):
        path = self.folder/'evidence'; path.mkdir(); value = path/'small.json'; value.write_bytes(b'{}\n')
        binding = A.pin(value); manifest = {'complete': True, 'evidence_root': path.relative_to(ROOT).as_posix(), 'files': {'small.json': binding, 'empty.log': {'bytes': 0, 'sha256': hashlib.sha256(b'').hexdigest()}}, 'extracted_files': {'small.json': binding, 'empty.log': {'bytes': 0, 'sha256': hashlib.sha256(b'').hexdigest()}}}
        mp = self.folder/'inventory.json'; mp.write_text(json.dumps(manifest))
        e = A.Evidence(mp); self.assertEqual(e.json('small.json'), {})
        value.write_bytes(b'{"changed":true}\n')
        with self.assertRaises(ValueError): e.json('small.json')
        with self.assertRaises(ValueError): e.file('../small.json')

    def test_deserialization_guard_and_no_model_operations_in_AST(self):
        manifest = self.folder/'inventory.json'; manifest.write_text('{}')
        review = self.folder/'review.json'; review.write_text(json.dumps({'complete': False}))
        with self.assertRaises(ValueError): A.admission(review, manifest)
        tree = ast.parse(SOURCE.read_text()); loads = []
        for node in ast.walk(tree):
            if isinstance(node, ast.Call) and isinstance(node.func, ast.Attribute):
                if isinstance(node.func.value, ast.Name) and node.func.value.id == 'torch' and node.func.attr == 'load': loads.append(node)
                self.assertNotIn(node.func.attr, ('backward', 'step', 'forward', 'load_state_dict'))
        self.assertEqual(len(loads), 1)
        self.assertEqual({k.arg: ast.literal_eval(k.value) for k in loads[0].keywords}, {'map_location': 'cpu', 'weights_only': True})
        with mock.patch.dict('os.environ', {'FORENSIC_U64_STATE_AUDIT_CHILD': '', 'CUDA_VISIBLE_DEVICES': ''}), mock.patch('sys.argv', ['audit', '--_cpu-child', '--manifest', str(manifest), '--source-review', str(review)]):
            with self.assertRaises(ValueError): A.main()

    def windows_context(self):
        return {'windows': True, 'executable': r'C:\repo\venv\Scripts\python.exe',
                'base_executable': r'C:\Python313\python.exe', 'image': r'C:\Python313\python.exe',
                'prefix': r'C:\repo\venv', 'base_prefix': r'C:\Python313',
                'site_paths': [r'C:\repo\venv\Lib\site-packages'], 'parent_pid': 42, 'environ': {}}

    def test_Windows_direct_binary_plan_preserves_venv_context(self):
        context = self.windows_context(); binary, environment = A.child_launch_plan(**context)
        self.assertEqual(binary, context['image'])
        self.assertEqual(environment['__PYVENV_LAUNCHER__'], context['executable'])
        child = {k: context[k] for k in ('windows', 'executable', 'image', 'prefix', 'base_prefix', 'site_paths', 'parent_pid')}
        result = A.verify_child_context(environ=environment, **child)
        self.assertTrue(result['direct_parent_verified']); self.assertTrue(result['Windows_launcher_bypassed'])
        self.assertEqual(result['venv_prefix'], context['prefix'])
        bad = dict(context, image=r'C:\Unrelated\python.exe')
        with self.assertRaisesRegex(ValueError, 'Actual running image'): A.child_launch_plan(**bad)

    def test_wrapper_and_unrelated_parent_still_reject(self):
        context = self.windows_context(); _, environment = A.child_launch_plan(**context)
        child = {k: context[k] for k in ('windows', 'executable', 'image', 'prefix', 'base_prefix', 'site_paths', 'parent_pid')}
        for parent in (84, 99, 0):
            with self.subTest(parent=parent), self.assertRaisesRegex(ValueError, 'direct-parent'):
                A.verify_child_context(environ=environment, **dict(child, parent_pid=parent))
        for field, value in (('executable', r'C:\Python313\python.exe'), ('image', r'C:\Unrelated\python.exe'),
                             ('prefix', r'C:\OtherVenv'), ('base_prefix', r'C:\OtherPython'), ('site_paths', [])):
            with self.subTest(field=field), self.assertRaises(ValueError):
                A.verify_child_context(environ=environment, **dict(child, **{field: value}))

    def test_normal_POSIX_plan_keeps_venv_executable_and_direct_guard(self):
        context = {'windows': False, 'executable': '/repo/venv/bin/python', 'base_executable': '/usr/bin/python3',
                   'image': '/usr/bin/python3', 'prefix': '/repo/venv', 'base_prefix': '/usr',
                   'site_paths': ['/repo/venv/lib/python3.13/site-packages'], 'parent_pid': 42,
                   'environ': {'__PYVENV_LAUNCHER__': 'stale'}}
        binary, environment = A.child_launch_plan(**context)
        self.assertEqual(binary, context['executable']); self.assertNotIn('__PYVENV_LAUNCHER__', environment)
        child = {k: context[k] for k in ('windows', 'executable', 'image', 'prefix', 'base_prefix', 'site_paths', 'parent_pid')}
        self.assertTrue(A.verify_child_context(environ=environment, **child)['direct_parent_verified'])

    def test_tiny_real_child_verifies_direct_parent_and_venv_without_frameworks(self):
        global REAL_CHILD_DIAGNOSTIC
        binary, environment = A.current_launch_plan()
        code = '''import builtins,importlib.util,json,os,sys
original=builtins.__import__
def blocked(name,*args,**kwargs):
 if name.split('.')[0] in ('torch','numpy'):raise RuntimeError('Forbidden numerical import')
 return original(name,*args,**kwargs)
builtins.__import__=blocked
spec=importlib.util.spec_from_file_location('actual_no_framework_child',sys.argv[1]);module=importlib.util.module_from_spec(spec);spec.loader.exec_module(module)
context=module.current_child_context()
print(json.dumps({'context':context,'pid':os.getpid(),'ppid':os.getppid(),'Torch_imported':'torch' in sys.modules,'NumPy_imported':'numpy' in sys.modules,'NumPy_spec':importlib.util.find_spec('numpy').origin,'Torch_spec':importlib.util.find_spec('torch').origin}))
'''
        process = subprocess.run([binary, '-B', '-c', code, str(SOURCE)], env=environment,
                                 capture_output=True, text=True, timeout=10)
        self.assertEqual(process.returncode, 0, process.stderr); self.assertFalse(process.stderr)
        result = json.loads(process.stdout)
        self.assertEqual(result['ppid'], os.getpid()); self.assertTrue(result['context']['direct_parent_verified'])
        self.assertFalse(result['Torch_imported']); self.assertFalse(result['NumPy_imported'])
        for name in ('NumPy_spec', 'Torch_spec'):
            self.assertTrue(Path(result[name]).resolve().is_relative_to(Path(result['context']['venv_prefix'])))
        REAL_CHILD_DIAGNOSTIC = result


if __name__ == '__main__':
    unittest.main()
