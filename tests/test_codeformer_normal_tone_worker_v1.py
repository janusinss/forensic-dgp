"""CPU-only worker control fixtures; no framework or optimizer objects/calls."""
import ast
import builtins
from contextlib import redirect_stdout
from copy import deepcopy
import hashlib
import io
import json
from pathlib import Path
import sys
import tempfile
import time
from types import SimpleNamespace
import unittest
from unittest.mock import patch

import numpy as np

from research.codeformer_normal_tone_v1 import contract as c
from research.codeformer_normal_tone_v1 import worker as w


ROOT = Path(__file__).resolve().parents[1]


class FakeCuda:
    def __init__(self):
        self.reserved = 1024
        self.allocated = 512
        self.peak_reserved = 2048
        self.peak_allocated = 1024

    def memory_reserved(self):
        return self.reserved

    def max_memory_reserved(self):
        return self.peak_reserved

    def memory_allocated(self):
        return self.allocated

    def max_memory_allocated(self):
        return self.peak_allocated


class FakeHelper:
    @staticmethod
    def atomic_json(path, value):
        path = Path(path)
        path.parent.mkdir(parents=True, exist_ok=True)
        with path.open('x', encoding='utf-8') as stream:
            json.dump(value, stream, allow_nan=False)
            stream.write('\n')

    @staticmethod
    def sha256(path):
        return hashlib.sha256(Path(path).read_bytes()).hexdigest()


class WorkerSourceTests(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        cls.tree = ast.parse(Path(w.__file__).read_text(encoding='utf-8'))

    def test_import_is_standard_library_only(self):
        top_imports = [node for node in self.tree.body if isinstance(node, (ast.Import, ast.ImportFrom))]
        roots = {alias.name.split('.')[0] for node in top_imports if isinstance(node, ast.Import)
                 for alias in node.names}
        roots.update(node.module.split('.')[0] for node in top_imports if isinstance(node, ast.ImportFrom))
        self.assertTrue(roots.isdisjoint({'torch', 'numpy', 'PIL', 'cv2', 'research'}))
        self.assertFalse(any(isinstance(node, ast.Call) for node in self.tree.body))

    def test_exact_call_sites_and_finite_arithmetic(self):
        calls = [node for node in ast.walk(self.tree) if isinstance(node, ast.Call)]
        attrs = [node.func.attr for node in calls if isinstance(node.func, ast.Attribute)]
        self.assertEqual(attrs.count('infer'), 3)  # One source/probe/repeat site.
        self.assertEqual(attrs.count('step'), 1)   # One REAL step site, no test execution.
        self.assertEqual(attrs.count('grad'), 1)   # One query helper; two declared uses.
        self.assertEqual(attrs.count('backward'), 0)
        self.assertFalse(set(attrs) & {'forward_training_terms', 'load_teacher',
                                      'load_behavior_teacher', 'load_perceptual'})
        self.assertEqual(w.COUNTS, c.COUNTS)
        self.assertEqual(list(w.BRANCHES), c.BRANCHES)
        self.assertEqual(list(w.COMPONENTS), c.COMPONENTS)
        self.assertEqual(w.COUNTS['candidate_source_forwards'], 2 * 16)
        self.assertEqual(w.COUNTS['candidate_probe_forwards'], 2 * 5 * 16)
        self.assertEqual(w.COUNTS['restoration_repeat_forwards'], 2 * 16)
        self.assertEqual(w.COUNTS['coarse_gradient_queries'], 2 * 16)
        self.assertEqual(w.COUNTS['centered_normal_gradient_queries'], 2 * 8)
        self.assertEqual(w.COUNTS['disposable_optimizer_steps'], 2 * 4)
        self.assertEqual(w.call_totals(w.COUNTS)['neural_forwards'], 224)
        self.assertEqual(w.call_totals(w.COUNTS)['gradient_queries'], 48)

    def test_guard_precedes_framework_import_and_checkpoint_load(self):
        run = next(node for node in self.tree.body if isinstance(node, ast.FunctionDef) and node.name == 'run')
        calls = [node for node in ast.walk(run) if isinstance(node, ast.Call)]
        guard = min(node.lineno for node in calls if isinstance(node.func, ast.Attribute) and
                    node.func.attr == 'manual_root')
        verify = min(node.lineno for node in calls if isinstance(node.func, ast.Attribute) and
                     node.func.attr == 'verify_protocol')
        framework = min(node.lineno for node in ast.walk(run) if isinstance(node, ast.Import) and
                        any(alias.name == 'torch' for alias in node.names))
        loads = [node.lineno for node in calls if isinstance(node.func, ast.Attribute) and
                 node.func.attr in ('load_trainable', 'load')]
        self.assertLess(guard, framework)
        self.assertLess(verify, framework)
        self.assertTrue(all(framework < line for line in loads))
        self.assertLess(guard, min(loads))

    def test_windows_run_refuses_before_any_framework_import(self):
        imported = []
        original = builtins.__import__
        def guarded_import(name, *args, **kwargs):
            if name.split('.')[0] in {'torch', 'PIL', 'cv2'}:
                imported.append(name)
                raise AssertionError('Framework import attempted before manual VM guard')
            return original(name, *args, **kwargs)
        with patch.object(c.platform, 'system', return_value='Windows'), \
                patch('builtins.__import__', side_effect=guarded_import):
            with self.assertRaisesRegex(ValueError, 'Manual NVIDIA L4 VM execution only'):
                w.run(ROOT, {}, '0' * 64)
        self.assertEqual(imported, [])

    def test_centered_mean_not_detached_or_channel_centered(self):
        function = next(node for node in self.tree.body if isinstance(node, ast.FunctionDef) and node.name == 'coarse_terms')
        text = ast.unparse(function)
        self.assertIn('sum(dim=1, keepdim=True).mean(dim=(2, 3), keepdim=True)', text)
        self.assertIn('(error - mean_luminance).abs().mean()', text)
        self.assertNotIn('.detach(', text)
        self.assertIn('functional.avg_pool2d(rgb, 32, 32)', text)
        self.assertIn('functional.l1_loss(predicted_grid, target_grid)', text)

    def test_preflight_stdout_omits_4630_large_dependency_records(self):
        maps = {alias: {'path%04d' % index: {'sha256': '1' * 64, 'bytes': 12345}
                       for index in range(count)} for alias, count in (('adain', 2011), ('loss_response', 2619))}
        transfer = {'format': 'codeformer-normal-tone-transfer-v1', 'complete': True,
            'protocol_sha256': '2' * 64, 'protocol': {'dependencies': maps}, 'dependencies': maps,
            'dependencies_verified': {'adain': 2011, 'loss_response': 2619},
            'static_files': {'source%02d.py' % n: {'sha256': '3' * 64, 'bytes': 100} for n in range(16)},
            'cases_verified': 16, 'readonly': True, 'neural_calls': 0, 'gradient_queries': 0, 'optimizer_steps': 0}
        summary = w.transfer_cli_summary(transfer)
        self.assertLess(len(json.dumps(summary).encode('utf-8')), 4096)
        self.assertNotIn('protocol', summary)
        self.assertNotIn('dependencies', summary)
        self.assertEqual(summary['dependencies_verified'], {'adain': 2011, 'loss_response': 2619})
        self.assertEqual(summary['static_files_verified_including_protocol'], 17)

    def test_mocked_preflight_main_stdout_is_bounded_without_torch(self):
        from research.codeformer_normal_tone_v1 import transport
        scratch = ROOT / 'scratch'
        scratch.mkdir(exist_ok=True)
        maps = {alias: {'file%04d' % n: {'sha256': '4' * 64, 'bytes': n}
                       for n in range(count)} for alias, count in (('adain', 2011), ('loss_response', 2619))}
        transfer = {'format': 'codeformer-normal-tone-transfer-v1', 'complete': True,
            'protocol_sha256': '2' * 64, 'protocol': {'dependencies': maps}, 'dependencies': maps,
            'dependencies_verified': {'adain': 2011, 'loss_response': 2619}, 'static_files': {},
            'cases_verified': 16, 'readonly': True, 'neural_calls': 0, 'gradient_queries': 0, 'optimizer_steps': 0}
        original, path_before = builtins.__import__, list(sys.path)
        def guarded_import(name, *args, **kwargs):
            if name.split('.')[0] in {'torch', 'PIL', 'cv2'}:
                raise AssertionError('Static main imported neural runtime')
            return original(name, *args, **kwargs)
        try:
            with tempfile.TemporaryDirectory(prefix='normal-tone-preflight-cpu-', dir=scratch) as temp, \
                    patch.object(transport, 'verify_transfer', return_value=transfer) as verified, \
                    patch('builtins.__import__', side_effect=guarded_import), redirect_stdout(io.StringIO()) as captured:
                w.main(['--root', temp, '--protocol-sha', '2' * 64, '--verify-transfer'])
                self.assertEqual(verified.call_count, 1)
                stdout = captured.getvalue()
                self.assertLess(len(stdout.encode('utf-8')), 4096)
                value = json.loads(stdout)
                self.assertNotIn('protocol', value)
                self.assertNotIn('dependencies', value)
                self.assertEqual(value['dependencies_verified'], {'adain': 2011, 'loss_response': 2619})
        finally:
            sys.path[:] = path_before

    def test_driver_provenance_precedes_model_construction(self):
        text = Path(w.__file__).read_text(encoding='utf-8')
        self.assertIn('--query-gpu=driver_version,name,uuid', text)
        self.assertIn('check=True, timeout=10', text)
        self.assertLess(text.index('driver = subprocess.run('), text.index('net, census = model.load_trainable('))
        self.assertIn("'NVIDIA_driver': driver_version", text)
        self.assertIn("'GPU_UUID': driver_uuid", text)

    def test_metric_target256_bound_before_model_construction(self):
        text = Path(w.__file__).read_text(encoding='utf-8')
        self.assertIn("target_prefix + 'clean256.npy'", text)
        self.assertIn("retained_target_record['clean256_array_sha256']", text)
        self.assertIn('np.array_equal(target, retained_target)', text)
        self.assertLess(text.index('cases, arrays = load_cases('), text.index('net, census = model.load_trainable('))


class ArrayMathTests(unittest.TestCase):
    def test_accumulation_copies_preserves_none_and_source_order(self):
        names = ['encoder.a', 'fuse_convs_dict.a']
        total = dict.fromkeys(names)
        current = {'encoder.a': None, 'fuse_convs_dict.a': np.array([1e20], np.float32)}
        original = current['fuse_convs_dict.a'].copy()
        w.accumulate(total, current, names, np)
        self.assertIsNone(total['encoder.a'])
        self.assertIsNot(total['fuse_convs_dict.a'], current['fuse_convs_dict.a'])
        for value in (3., -1e20):
            w.accumulate(total, {'encoder.a': None, 'fuse_convs_dict.a': np.array([value], np.float32)}, names, np)
        self.assertEqual(float(total['fuse_convs_dict.a'][0]), 0.)
        np.testing.assert_array_equal(current['fuse_convs_dict.a'], original)
        # Canonical all - normal is zero; the actual dark subsequence is3.
        normals = {'encoder.a': None, 'fuse_convs_dict.a': np.array([0.], np.float32)}
        dark = np.array([3.], np.float32)
        self.assertFalse(np.array_equal(total['fuse_convs_dict.a'] - normals['fuse_convs_dict.a'], dark))

    def test_compose_fixed_slot_order_immutability_and_explicit_zero(self):
        name, names = 'fuse_convs_dict.a', ['fuse_convs_dict.a']
        components = {term: {name: None} for term in w.COMPONENTS}
        components['token_ce'][name] = np.array([1e20], np.float32)
        components['latent_mse'][name] = np.array([-1e20], np.float32)
        coarse = {name: np.array([3.], np.float32)}
        saved = deepcopy(components)
        result = w.compose(components, coarse, names, np)
        self.assertEqual(float(result[name][0]), 3.)
        self.assertEqual(float(((np.float32(1e20) + np.float32(3.)) + np.float32(-1e20))), 0.)
        result[name][0] = 9
        for term in w.COMPONENTS:
            if saved[term][name] is None:
                self.assertIsNone(components[term][name])
            else:
                np.testing.assert_array_equal(saved[term][name], components[term][name])
        self.assertEqual(float(coarse[name][0]), 3.)
        zeros = w.compose(components, coarse, names, np, momentum_only=True)
        self.assertIsNotNone(zeros[name])
        self.assertEqual(zeros[name].dtype, np.float32)
        self.assertFalse(np.any(zeros[name]))

    def test_none_cannot_be_forged_as_real_zero(self):
        expected = {'a': None, 'b': np.zeros(2, np.float32)}
        actual = {'a': None, 'b': np.zeros(2, np.float32)}
        self.assertTrue(w.exact_arrays(actual, expected, ['a', 'b'], np)['pass'])
        with self.assertRaisesRegex(ValueError, 'None topology'):
            w.exact_arrays({'a': np.zeros(1, np.float32), 'b': expected['b']}, expected, ['a', 'b'], np)
        actual['b'][0] = np.nextafter(np.float32(0), np.float32(1))
        result = w.exact_arrays(actual, expected, ['a', 'b'], np)
        self.assertFalse(result['pass'])
        self.assertGreater(result['maximum_absolute'], 0)
        self.assertEqual(result['accepted_maximum_error'], 0.)


class BudgetTests(unittest.TestCase):
    def setUp(self):
        scratch = ROOT / 'scratch'
        scratch.mkdir(exist_ok=True)
        self.temporary = tempfile.TemporaryDirectory(prefix='normal-tone-worker-cpu-', dir=scratch)
        self.addCleanup(self.temporary.cleanup)
        self.root = Path(self.temporary.name)
        self.out = self.root / 'outputs'
        self.out.mkdir()
        self.cuda = FakeCuda()
        self.free = 20 * 1024**3
        self.rss = 1000
        self.disk_patch = patch.object(w.shutil, 'disk_usage', side_effect=lambda path: SimpleNamespace(free=self.free))
        self.resource_patch = patch.object(w, 'resource', SimpleNamespace(RUSAGE_SELF=0,
            getrusage=lambda kind: SimpleNamespace(ru_maxrss=self.rss)))
        self.disk_patch.start(); self.resource_patch.start()
        self.addCleanup(self.disk_patch.stop); self.addCleanup(self.resource_patch.stop)
        self.budget = w.Budget(self.root, self.out, deepcopy(c.LIMITS), SimpleNamespace(cuda=self.cuda),
            {'signal': None}, FakeHelper, time.monotonic())

    def test_call_reservations_are_prospective_and_counts_survive_failure(self):
        self.budget.reserve('coarse_gradient_queries', {'fixture': True})
        trace = json.loads((self.out / 'call_trace' / '00001.json').read_text())
        self.assertEqual(trace['before_call'], 'coarse_gradient_queries')
        self.assertEqual(trace['counts']['coarse_gradient_queries'], 1)
        self.assertEqual(trace['actual_completed_disposable_Adam_steps'], 0)
        self.budget.counts['coarse_gradient_queries'] = w.COUNTS['coarse_gradient_queries']
        with self.assertRaisesRegex(ValueError, 'Finite call cap'):
            self.budget.reserve('coarse_gradient_queries')
        self.assertEqual(self.budget.event, 1)
        self.budget.reserve('optimizer_step_attempts', {'fixture': 'reservation only; no optimizer'})
        self.budget.optimizer_inflight = True
        self.assertEqual(self.budget.counts['optimizer_step_attempts'], 1)
        self.assertEqual(self.budget.counts['disposable_optimizer_steps'], 0)
        self.assertEqual(self.budget.actual_steps, 0)
        self.assertTrue(self.budget.optimizer_inflight)
        self.assertEqual(w.call_totals(self.budget.counts)['actual_completed_disposable_Adam_steps'], 0)

    def test_write_and_disk_reservations_refuse_before_file(self):
        self.budget.limits['result_payload_bytes'] = 1024**2 + 4
        target = self.out / 'must_not_exist.npy'
        with self.assertRaisesRegex(ValueError, 'Prospective output'):
            self.budget.array(target, np.zeros(2, np.float32), np)
        self.assertFalse(target.exists())
        self.budget.limits['result_payload_bytes'] = c.LIMITS['result_payload_bytes']
        self.free = c.LIMITS['reserve_free_bytes'] + 100
        with self.assertRaisesRegex(ValueError, 'disk reserve'):
            self.budget.reserve_write(101)
        self.assertEqual(self.budget.measurements()['measured_minimum_sampled_free_bytes'], self.free)

    def test_measured_peaks_and_failure_tick_are_retained(self):
        self.rss = 2000
        self.cuda.peak_reserved = 4096
        self.cuda.peak_allocated = 3072
        self.free = 19 * 1024**3
        self.budget.check()
        record = self.budget.measurements()
        self.assertEqual(record['measured_peak_RSS_bytes'], 2000 * 1024)
        self.assertEqual(record['measured_peak_CUDA_reserved_bytes'], 4096)
        self.assertEqual(record['measured_peak_CUDA_allocated_bytes'], 3072)
        self.assertEqual(record['measured_minimum_sampled_free_bytes'], self.free)
        self.assertTrue(record['free_disk_minimum_is_sampled_not_continuous'])
        self.assertEqual(record['free_disk_measurement_path'], str(self.root))
        self.assertIn('Worker root filesystem only', record['free_disk_scope'])
        self.cuda.peak_reserved = c.LIMITS['CUDA_reserved_bytes'] + 1
        with self.assertRaisesRegex(w.DiagnosticStop, 'CUDA reserved'):
            self.budget.check()
        self.assertEqual(self.budget.measurements()['measured_peak_CUDA_reserved_bytes'], self.cuda.peak_reserved)
        self.assertGreaterEqual(self.budget.measurements()['measurement_samples'], 3)

    def test_file_metadata_and_manual_stop_caps(self):
        self.budget.regular_files = self.budget.limits['regular_file_cap']
        with self.assertRaisesRegex(ValueError, 'file cap'):
            self.budget.reserve_write(0)
        self.budget.regular_files = 0
        self.budget.metadata_bytes = self.budget.limits['metadata_bytes']
        with self.assertRaisesRegex(ValueError, 'metadata cap'):
            self.budget.json(self.out / 'no.json', {'fixture': True})
        self.assertFalse((self.out / 'no.json').exists())
        self.budget.metadata_bytes = 0
        self.budget.request['signal'] = 15
        with self.assertRaisesRegex(w.DiagnosticStop, 'Manual signal'):
            self.budget.check()


class RetainedMetadataTests(unittest.TestCase):
    def test_small_saved_json_schema_matches_initializer_and_control_assumptions(self):
        # Read only retained JSON; never load a checkpoint or neural array.
        base = ROOT / 'outputs/codeformer_loss_response_v1_return/outputs/loss_response_v1'
        initialization = c.read_json(base / 'initialization.json')
        self.assertEqual(initialization['census']['trainable_parameters'], 59788608)
        groups = initialization['optimizer_default_metadata']
        self.assertEqual([group['name'] for group in groups], ['encoder_token', 'fusion'])
        self.assertEqual([group['lr'] for group in groups], [1e-5, 2e-5])
        self.assertEqual([group['base_lr'] for group in groups], [1e-5, 2e-5])
        for group in groups:
            self.assertEqual(group['betas'], [.9, .99])
            self.assertEqual(group['eps'], 1e-8)
            self.assertIsNone(group['foreach'])
            self.assertIsNone(group['fused'])
            self.assertFalse(group['differentiable'])
        for state in ('original0', 'stopped50'):
            source = c.read_json(base / state / 'source_summary.json')
            branch = c.read_json(base / state / 'conditions/full/branch.json')
            w.validate_prior_branch(branch, state, 'full', source['base_model_sha256'],
                                    source['base_Adam_sha256'], source['base_RNG_sha256'], c)
            self.assertEqual(len(branch['step']['assigned_gradient_sha256_by_tensor']), 350)
            manifest = c.read_json(base / state / 'gradients/coarse_rgb_l1/manifest.json')
            self.assertEqual(sum(not row['is_none'] for row in manifest['parameters'].values()), 72)
            self.assertEqual(manifest['aggregation_divisor'], 16)
            self.assertEqual(len(manifest['accumulation_order']), 16)
            self.assertEqual(c.cached_basis_paths()[state]['coarse_rgb_l1']['manifest_path'],
                             'outputs/loss_response_v1/' + state + '/gradients/coarse_rgb_l1/manifest.json')


if __name__ == '__main__':
    unittest.main()
