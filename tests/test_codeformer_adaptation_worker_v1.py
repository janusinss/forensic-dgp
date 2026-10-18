"""Safety/schema/packaging regressions; no neural forward, gradient or training."""
import ast
from copy import deepcopy
import gzip
import hashlib
import importlib.util
import json
from pathlib import Path
import shutil
import sys
import tarfile
import uuid
from types import SimpleNamespace
import unittest
from unittest.mock import patch


SOURCE = Path(__file__).resolve().parents[1] / 'research/codeformer_adaptation_v1/worker.py'
SPEC = importlib.util.spec_from_file_location('codeformer_worker_static_tests', SOURCE)
worker = importlib.util.module_from_spec(SPEC)
SPEC.loader.exec_module(worker)


def fixture(root):
    def put(relative, payload):
        target = root / relative
        target.parent.mkdir(parents=True, exist_ok=True)
        target.write_bytes(payload)
        return hashlib.sha256(payload).hexdigest()
    worker_sha = put('research/codeformer_adaptation_v1/worker.py', SOURCE.read_bytes())
    source_sha = put('data/sources/fixture.png', b'static source bytes; image decoder must never run')
    target_sha = put('data/targets/fixture.png', b'static target bytes')
    input_sha = put('data/comparison/input.png', b'static comparison bytes')
    mask_sha = put('data/comparison/observed.png', b'static observed mask bytes')
    files = {'research/codeformer_adaptation_v1/worker.py': worker_sha,
             'data/sources/fixture.png': source_sha, 'data/targets/fixture.png': target_sha,
             'data/comparison/input.png': input_sha, 'data/comparison/observed.png': mask_sha}
    rows = []
    for role, count in [('train', 391), ('validation', 53)]:
        for index in range(count):
            rows.append({'id': ('tr_' if role == 'train' else 'va_') + str(index),
                         'historical_role': role, 'source_path': 'data/sources/fixture.png',
                         'source_sha256': source_sha, 'existing_target_path': 'data/targets/fixture.png',
                         'existing_target_sha256': target_sha})
    files['references.json'] = put('references.json', json.dumps({'references': rows}).encode())
    cases = []
    for index in range(88):
        row = {'id': 'case_%03d' % index, 'evidence': 'native_unpaired' if index < 48 else 'paired_synthetic',
               'prepared_path': 'data/comparison/input.png', 'prepared_sha256': input_sha,
               'observed_path': 'data/comparison/observed.png', 'observed_sha256': mask_sha,
               'profile': 'clear', 'source': 'static-fixture'}
        if index >= 48:
            row.update(target_path='data/targets/fixture.png', target_sha256=target_sha)
        cases.append(row)
    files['comparison_cases.json'] = put('comparison_cases.json', json.dumps({'cases': cases}).encode())
    files['LICENSE'] = put('LICENSE', b'Owned static test license fixture')
    protocol = deepcopy(worker.DEFAULT_PROTOCOL)
    protocol['files_sha256'] = files
    protocol['assets'] = {}
    for label in sorted(worker.ASSET_LABELS):
        relative = 'assets/' + label + '.pth'
        payload = ('not a tensor: ' + label).encode()
        digest = put(relative, payload)
        protocol['assets'][label] = {'path': relative, 'sha256': digest, 'bytes': len(payload)}
    protocol['export_files'] = ['research/codeformer_adaptation_v1/worker.py',
                                'references.json', 'comparison_cases.json', 'LICENSE']
    path = root / 'protocol.json'
    path.write_text(json.dumps(protocol), encoding='utf-8')
    return protocol, worker.sha256(path)


class WorkerStaticTests(unittest.TestCase):
    def setUp(self):
        scratch = SOURCE.parents[2] / 'scratch'
        scratch.mkdir(exist_ok=True)
        self.root = scratch / ('codeformer_worker_static_' + uuid.uuid4().hex)
        self.root.mkdir()

    def tearDown(self):
        exact = self.root.resolve(strict=True)
        self.assertEqual(exact.parent, (SOURCE.parents[2] / 'scratch').resolve())
        self.assertTrue(exact.name.startswith('codeformer_worker_static_'))
        self.assertFalse(self.root.is_symlink())
        shutil.rmtree(exact)  # Literal owned fixture, resolved and scoped above.

    def test_python310_syntax_and_no_top_level_neural_import(self):
        tree = ast.parse(SOURCE.read_text(), feature_version=(3, 10))
        imports = [node for node in tree.body if isinstance(node, (ast.Import, ast.ImportFrom))]
        for node in imports:
            names = [alias.name for alias in node.names] if isinstance(node, ast.Import) else [node.module]
            self.assertFalse(any(name and name.split('.')[0] in {'torch', 'numpy', 'PIL', 'cv2'} for name in names))

    def test_exact_static_verification_never_imports_torch(self):
        _, digest = fixture(self.root)
        original_import = __import__
        def restricted(name, *args, **kwargs):
            if name.split('.')[0] in {'torch', 'numpy', 'PIL', 'cv2'}:
                raise AssertionError('Static verification attempted a neural/image import')
            return original_import(name, *args, **kwargs)
        with patch('builtins.__import__', side_effect=restricted):
            _, receipt = worker.verify_transfer(self.root, digest)
        self.assertTrue(receipt['complete'])
        self.assertEqual(receipt['optimizer_updates'], 0)
        self.assertEqual(receipt['gradient_queries'], 0)
        self.assertEqual(receipt['model_calls'], 0)

    def test_changed_source_and_wrong_protocol_rejected(self):
        _, digest = fixture(self.root)
        with self.assertRaisesRegex(ValueError, 'Protocol SHA256 differs'):
            worker.verify_transfer(self.root, '0' * 64)
        (self.root / 'data/sources/fixture.png').write_bytes(b'changed')
        with self.assertRaisesRegex(ValueError, 'Packet file differs'):
            worker.verify_transfer(self.root, digest)

    def test_finite_recipe_and_no_resume_are_immutable(self):
        protocol, _ = fixture(self.root)
        for section, key, value in [('training', 'max_updates', 1201), ('training', 'resume', True),
                                    ('limits', 'initial_free_bytes', 12 * 1024**3),
                                    ('evaluation', 'mean_clear_mae_delta_stop', 0.1)]:
            changed = deepcopy(protocol); changed[section][key] = value
            with self.assertRaisesRegex(ValueError, 'V1 protocol control differs'):
                worker.validate_protocol(changed)

    def test_path_escape_and_new_file_bounds(self):
        (self.root / 'existing').write_text('safe')
        for name in ['../escape', '/absolute', 'C:/escape', 'a\\escape', 'a/../escape']:
            with self.assertRaises(ValueError):
                worker.bounded(self.root, name, must_exist=False)
        self.assertEqual(worker.bounded(self.root, 'new/output', must_exist=False), self.root / 'new/output')

    def test_schedule_warmup_and_cosine_end(self):
        self.assertAlmostEqual(worker.learning_rate(1, 1e-5), 2e-7)
        self.assertAlmostEqual(worker.learning_rate(50, 1e-5), 1e-5)
        self.assertAlmostEqual(worker.learning_rate(1200, 1e-5), 2e-6)
        self.assertGreater(worker.learning_rate(600, 2e-5), worker.learning_rate(1199, 2e-5))
        with self.assertRaises(ValueError):
            worker.learning_rate(1201, 1e-5)

    def test_clear_gate_preserves_raw_and_png_separately(self):
        baseline = {'a': {'raw': 0.1, 'native_png': 0.1}, 'b': {'raw': 0.2, 'native_png': 0.2}}
        candidate = deepcopy(baseline)
        self.assertTrue(worker.clear_quality_gate(candidate, baseline)['pass'])
        candidate['a']['native_png'] = 0.07
        candidate['b']['native_png'] = 0.231
        result = worker.clear_quality_gate(candidate, baseline)
        self.assertTrue(result['views']['raw']['pass'])
        self.assertLess(result['views']['native_png']['mean_mae_delta'], 0.01)
        self.assertFalse(result['pass'])
        with self.assertRaises(ValueError):
            worker.clear_quality_gate({'a': candidate['a']}, baseline)

    def test_pid_lock_refuses_existing_and_does_not_delete_stale_lock(self):
        with worker.pid_lock(self.root, 'a' * 64):
            self.assertTrue((self.root / 'adaptation_v1.lock').is_file())
            with self.assertRaises(FileExistsError):
                with worker.pid_lock(self.root, 'a' * 64):
                    pass
        self.assertFalse((self.root / 'adaptation_v1.lock').exists())
        old = self.root / 'adaptation_v1.lock'; old.write_text('retained stale marker')
        with self.assertRaises(FileExistsError):
            with worker.pid_lock(self.root, 'a' * 64):
                pass
        self.assertEqual(old.read_text(), 'retained stale marker')

    def test_training_rejected_before_torch_import_on_windows(self):
        with patch.object(worker.platform, 'system', return_value='Windows'):
            with self.assertRaisesRegex(ValueError, 'Linux L4 VM only'):
                worker.run(self.root, deepcopy(worker.DEFAULT_PROTOCOL), 'a' * 64)
        self.assertFalse((self.root / 'outputs').exists())

    def test_missing_vm_dependency_exports_structured_zero_update_failure(self):
        original_import = __import__
        def missing_torch(name, *args, **kwargs):
            if name == 'torch':
                raise ModuleNotFoundError('Owned dependency-stop fixture')
            return original_import(name, *args, **kwargs)
        with patch.object(worker, 'require_vm_path', return_value=self.root):
            with patch('builtins.__import__', side_effect=missing_torch):
                with self.assertRaises(ModuleNotFoundError):
                    worker.run(self.root, deepcopy(worker.DEFAULT_PROTOCOL), 'a' * 64)
        result = json.loads((self.root / 'outputs/adaptation_v1/failure.json').read_text())
        self.assertFalse(result['complete'])
        self.assertEqual(result['completed_updates'], 0)
        self.assertEqual(result['counts']['backward_calls'], 0)
        self.assertEqual(result['counts']['gradient_queries'], 0)
        self.assertFalse((self.root / 'adaptation_v1.lock').exists())

    def test_atomic_write_refuses_history_replacement(self):
        path = self.root / 'receipt.json'
        worker.atomic_json(path, {'complete': True})
        with self.assertRaises(ValueError):
            worker.atomic_json(path, {'complete': False})
        self.assertTrue(json.loads(path.read_text())['complete'])
        self.assertFalse(list(self.root.glob('*.tmp-*')))

    def test_export_excludes_immutable_binary_inputs_and_models(self):
        protocol, digest = fixture(self.root)
        out = self.root / protocol['outputs']; out.mkdir(parents=True)
        (out / 'failure.json').write_text(json.dumps({'complete': False, 'optimizer_updates': 0}))
        partial = out / 'checkpoint.pt.tmp-retained'; partial.write_bytes(b'incomplete owned fixture')
        (self.root / 'trainer.log').write_text('Closed trainer log fixture\n')
        result = worker.export_results(self.root, protocol, digest, destination=self.root)
        self.assertTrue(result['complete']); self.assertTrue(result['failure_present'])
        self.assertFalse(result['model_qualified'])
        self.assertTrue(result['training_success_not_implied'])
        archive = self.root / (protocol['export_prefix'] + '-results.tar.gz')
        with tarfile.open(archive) as stream:
            names = stream.getnames()
            self.assertIn('protocol.json', names)
            self.assertIn('research/codeformer_adaptation_v1/worker.py', names)
            self.assertIn('trainer.log', names)
            self.assertFalse(any(name.startswith(('data/', 'assets/')) for name in names))
            manifest = json.load(stream.extractfile(result['manifest_member']))
        self.assertIn('data/sources/fixture.png', manifest['retained_not_exported_sha256'])
        relative = partial.relative_to(self.root).as_posix()
        self.assertFalse(manifest['incomplete_temp_files'][relative]['resume_safe'])
        self.assertTrue(partial.exists())
        self.assertFalse(manifest['model_qualified'])

    def test_bad_validation_role_and_unpaired_target_refused(self):
        protocol, digest = fixture(self.root)
        data = json.loads((self.root / 'comparison_cases.json').read_text())
        data['cases'][0]['target_path'] = 'data/targets/fixture.png'
        path = self.root / 'comparison_cases.json'; path.write_text(json.dumps(data))
        protocol['files_sha256']['comparison_cases.json'] = worker.sha256(path)
        (self.root / 'protocol.json').write_text(json.dumps(protocol))
        with self.assertRaisesRegex(ValueError, 'Native CCTV must remain unpaired'):
            worker.verify_transfer(self.root, worker.sha256(self.root / 'protocol.json'))

    def test_export_space_stop_retains_structured_failure_and_original(self):
        protocol, digest = fixture(self.root)
        out = self.root / protocol['outputs']; out.mkdir(parents=True)
        original = out / 'failure.json'; original.write_text('{"complete":false,"reason":"retained"}')
        before = original.read_bytes()
        with patch.object(worker.shutil, 'disk_usage', return_value=SimpleNamespace(free=0)):
            with self.assertRaisesRegex(ValueError, 'Insufficient conservative export space'):
                worker.export_results(self.root, protocol, digest, destination=self.root)
        self.assertEqual(original.read_bytes(), before)
        stops = list(out.glob('export_stop_*.json'))
        self.assertEqual(len(stops), 1)
        self.assertFalse(json.loads(stops[0].read_text())['model_qualified'])
        self.assertFalse(list(self.root.glob('*.tar.gz')))

    def test_export_active_lock_refusal_and_dead_lock_preservation(self):
        lock = self.root / 'adaptation_v1.lock'
        lock.write_text(json.dumps({'pid': 12345, 'host': worker.socket.gethostname()}))
        saved = lock.read_bytes()
        with patch.object(worker.platform, 'system', return_value='Linux'):
            with patch.object(worker.os, 'kill', return_value=None):
                with self.assertRaisesRegex(worker.PilotStop, 'Active worker PID'):
                    worker.export_lock_state(self.root)
            with patch.object(worker.os, 'kill', side_effect=ProcessLookupError):
                result = worker.export_lock_state(self.root)
        self.assertTrue(result['dead_PID_lock_retained'])
        self.assertEqual(lock.read_bytes(), saved)
        with patch.object(worker.platform, 'system', return_value='Windows'):
            with patch.object(worker.os, 'kill') as signal_call:
                with self.assertRaisesRegex(ValueError, 'Linux process inspection'):
                    worker.export_lock_state(self.root)
                signal_call.assert_not_called()


if __name__ == '__main__':
    unittest.main()
