"""Safety/schema/packaging regressions; no neural forward, gradient or training."""
import ast
from copy import deepcopy
import gzip
import hashlib
import importlib.util
import io
import json
from pathlib import Path
import shutil
import sys
import tarfile
import uuid
from types import SimpleNamespace
import unittest
from unittest.mock import patch


SOURCE = Path(__file__).resolve().parents[1] / 'research/codeformer_adaptation_v2/worker.py'
SPEC = importlib.util.spec_from_file_location('codeformer_worker_static_tests', SOURCE)
worker = importlib.util.module_from_spec(SPEC)
SPEC.loader.exec_module(worker)


def fixture(root):
    def put(relative, payload):
        target = root / relative
        target.parent.mkdir(parents=True, exist_ok=True)
        target.write_bytes(payload)
        return hashlib.sha256(payload).hexdigest()
    worker_sha = put('research/codeformer_adaptation_v2/worker.py', SOURCE.read_bytes())
    source_sha = put('data/sources/fixture.png', b'static source bytes; image decoder must never run')
    target_sha = put('data/targets/fixture.png', b'static target bytes')
    input_sha = put('data/comparison/input.png', b'static comparison bytes')
    mask_sha = put('data/comparison/observed.png', b'static observed mask bytes')
    files = {'research/codeformer_adaptation_v2/worker.py': worker_sha,
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
    design_path = worker.DEFAULT_PROTOCOL['design']['path']
    files[design_path] = put(design_path, (SOURCE.parents[2] / 'CCTV_CODEFORMER_ADAPTATION_V2_DESIGN.md').read_bytes())
    readback_path = worker.DEFAULT_PROTOCOL['design']['readback_path']
    files[readback_path] = put(readback_path, json.dumps({
        'status': 'APPROVED_FOR_PROSPECTIVE_IMPLEMENTATION_ONLY',
        'design_sha256': worker.DEFAULT_PROTOCOL['design']['sha256'],
        'implementation_authorized': True, 'training_authorized_by_this_review': False}).encode())
    saved_items, original_cases = [], {}
    references_sha = files['references.json']
    for row in rows:
        if row['historical_role'] != 'validation':
            continue
        for profile in worker.DEFAULT_PROTOCOL['evaluation']['profiles']:
            identifier = 'validation_' + row['id'] + '_' + profile
            input_path = 'data/validation_v1/' + identifier + '/input.png'
            record_path = 'data/validation_v1/' + identifier + '/pair_record.json'
            files[input_path] = put(input_path, b'owned exact VM validation PNG fixture; no decoder')
            files[record_path] = put(record_path, json.dumps({
                'reference_id': row['id'], 'profile': profile, 'role': 'validation',
                'source_sha256': row['source_sha256'], 'reference_manifest_sha256': references_sha}).encode())
            old = {'path': 'old-v1-output/' + identifier + '/input.png',
                   'sha256': files[input_path], 'pair_record_path': 'old-v1-output/' + identifier + '/pair_record.json'}
            original_cases[identifier] = old
            saved_items.append({'id': identifier, 'input_path': input_path, 'input_sha256': files[input_path],
                'pair_record_path': record_path, 'pair_record_sha256': files[record_path],
                'target_path': row['existing_target_path'], 'target_sha256': row['existing_target_sha256'],
                'original_VM_paths': {'input': old['path'], 'pair_record': old['pair_record_path']}})
    original_path = 'provenance/v1/validation_inputs_v1.json'
    files[original_path] = put(original_path, json.dumps({'complete': True, 'cases_count': 318,
                                                        'cases': original_cases}).encode())
    saved_path = 'validation_saved_v1.json'
    files[saved_path] = put(saved_path, json.dumps({'format': 'codeformer-adaptation-saved-validation-v1',
        'source_protocol_sha256': worker.SOURCE_PROTOCOL_V1, 'case_count': 318, 'items': saved_items,
        'retained_original_manifest_path': original_path,
        'retained_original_manifest_sha256': files[original_path]}).encode())
    baseline_path = 'provenance/v1/snapshot_0000_metrics.json'
    files[baseline_path] = put(baseline_path, b'owned baseline metrics fixture; transfer mode only hashes this')
    protocol = deepcopy(worker.DEFAULT_PROTOCOL)
    protocol['validation_saved'] = {'manifest': saved_path, 'sha256': files[saved_path],
        'source_protocol_sha256': worker.SOURCE_PROTOCOL_V1, 'case_count': 318}
    protocol['prior_baseline'] = {'metrics': baseline_path, 'sha256': files[baseline_path],
        'source_protocol_sha256': worker.SOURCE_PROTOCOL_V1}
    protocol['files_sha256'] = files
    protocol['assets'] = {}
    for label in sorted(worker.ASSET_LABELS):
        relative = 'assets/' + label + '.pth'
        payload = ('not a tensor: ' + label).encode()
        digest = put(relative, payload)
        protocol['assets'][label] = {'path': relative, 'sha256': digest, 'bytes': len(payload)}
    protocol['export_files'] = ['research/codeformer_adaptation_v2/worker.py',
                                'references.json', 'comparison_cases.json', 'LICENSE', design_path, readback_path,
                                saved_path, original_path, baseline_path]
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
        for section, key, value in [('training', 'max_updates', 601), ('training', 'resume', True),
                                    ('limits', 'initial_free_bytes', 12 * 1024**3),
                                    ('evaluation', 'mean_clear_mae_delta_stop', 0.1)]:
            changed = deepcopy(protocol); changed[section][key] = value
            with self.assertRaisesRegex(ValueError, 'V2 protocol control differs'):
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
        self.assertAlmostEqual(worker.learning_rate(600, 1e-5), 2e-6)
        self.assertGreater(worker.learning_rate(300, 2e-5), worker.learning_rate(599, 2e-5))
        with self.assertRaises(ValueError):
            worker.learning_rate(601, 1e-5)

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
            self.assertTrue((self.root / 'adaptation_v2.lock').is_file())
            with self.assertRaises(FileExistsError):
                with worker.pid_lock(self.root, 'a' * 64):
                    pass
        self.assertFalse((self.root / 'adaptation_v2.lock').exists())
        old = self.root / 'adaptation_v2.lock'; old.write_text('retained stale marker')
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
        result = json.loads((self.root / 'outputs/adaptation_v2/failure.json').read_text())
        self.assertFalse(result['complete'])
        self.assertEqual(result['completed_updates'], 0)
        self.assertEqual(result['counts']['backward_calls'], 0)
        self.assertEqual(result['counts']['gradient_queries'], 0)
        self.assertFalse((self.root / 'adaptation_v2.lock').exists())

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
            self.assertIn('research/codeformer_adaptation_v2/worker.py', names)
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
        lock = self.root / 'adaptation_v2.lock'
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

    def test_blur_guard_checks_signed_per_case_regressions_in_both_views(self):
        baseline = {str(i): {view: {'luminance_bias': .1, 'coarse_rgb_mae': .2}
                               for view in ['raw', 'native_png']} for i in range(53)}
        candidate = deepcopy(baseline)
        candidate['0']['raw']['luminance_bias'] = .0
        candidate['1']['native_png']['coarse_rgb_mae'] = .231
        report = worker.blur_quality_gate(candidate, baseline)
        self.assertFalse(report['pass'])
        self.assertTrue(report['views']['raw']['luminance_bias']['pass'])
        failed = report['views']['native_png']['coarse_rgb_mae']
        self.assertLess(failed['mean_delta'], .01)
        self.assertGreater(failed['worst_per_case_delta'], .03)
        with self.assertRaisesRegex(ValueError, 'namespace'):
            worker.blur_quality_gate({'0': candidate['0']}, baseline)

    def test_bounded_writes_refuse_before_overflow_and_keep_partial_evidence(self):
        stream = io.BytesIO()
        capped = worker.CappedWriter(stream, 5)
        capped.write(b'1234')
        with self.assertRaises(worker.PilotStop):
            capped.write(b'56')
        self.assertEqual(stream.getvalue(), b'1234')
        target = self.root / 'owned.png'
        with self.assertRaises(worker.PilotStop):
            worker._atomic_binary(target, lambda file: file.write(b'123456'), byte_cap=5)
        self.assertFalse(target.exists())
        retained = list(self.root.glob('owned.png.tmp-*'))
        self.assertEqual(len(retained), 1)
        self.assertLessEqual(retained[0].stat().st_size, 5)
        fake_torch = SimpleNamespace(save=lambda value, file: file.write(b'123456'))
        checkpoint = self.root / 'owned_checkpoint.pt'
        with self.assertRaises(worker.PilotStop):
            worker._atomic_torch(checkpoint, {'metadata_fixture': True}, fake_torch, byte_cap=5)
        self.assertFalse(checkpoint.exists())
        self.assertEqual(len(list(self.root.glob('owned_checkpoint.pt.tmp-*'))), 1)

    def test_finite_counter_reserves_before_any_extra_invocation(self):
        counts = {'fake': 0}
        worker.reserve_call(counts, 'fake', 1)
        with self.assertRaisesRegex(ValueError, 'cap reached'):
            worker.reserve_call(counts, 'fake', 1)
        self.assertEqual(counts['fake'], 1)

    def test_saved_development_record_cannot_switch_target_or_profile(self):
        protocol, _ = fixture(self.root)
        path = self.root / protocol['validation_saved']['manifest']
        manifest = json.loads(path.read_text())
        manifest['items'][0]['target_sha256'] = '0' * 64
        path.write_text(json.dumps(manifest))
        protocol['files_sha256'][path.name] = worker.sha256(path)
        protocol['validation_saved']['sha256'] = worker.sha256(path)
        (self.root / 'protocol.json').write_text(json.dumps(protocol))
        with self.assertRaisesRegex(ValueError, 'Saved validation file not bound'):
            worker.verify_transfer(self.root, worker.sha256(self.root / 'protocol.json'))

    def test_result_payload_reservation_includes_actual_existing_bytes(self):
        out = self.root / 'owned_outputs'; out.mkdir()
        (out / 'first.npy').write_bytes(b'12345')
        limits = {'result_payload_bytes': 10}
        worker.require_payload_room(out, limits, 5)
        with self.assertRaisesRegex(ValueError, '7 GiB'):
            worker.require_payload_room(out, limits, 6)

    def test_failed_export_partial_refuses_unbounded_repeat(self):
        protocol, digest = fixture(self.root)
        out = self.root / protocol['outputs']; out.mkdir(parents=True)
        (out / 'failure.json').write_text('{"complete":false}')
        partial = self.root / (protocol['export_prefix'] + '-results.tar.gz.partial-owned')
        partial.write_bytes(b'retained incomplete archive')
        with self.assertRaisesRegex(ValueError, 'partial archive exists'):
            worker.export_results(self.root, protocol, digest, destination=self.root)
        self.assertEqual(partial.read_bytes(), b'retained incomplete archive')

    def test_source_has_no_dev_generation_and_commits_after_optimizer(self):
        text = SOURCE.read_text()
        self.assertNotIn('data.make_pair(', text)
        self.assertNotIn('profile_weights', text)
        self.assertLess(text.index('optimizer.step()'), text.index('sampler.commit(selection)'))
        self.assertLess(text.index('sampler.commit(selection)'), text.index('committed_rng, committed_sampler =',
                        text.index('sampler.commit(selection)')))
        self.assertIn("counts['training_input_writes'] == 2400", text)

    def test_worker_data_camera_policy_and_design_bindings_agree(self):
        from research.codeformer_adaptation_v2 import data
        self.assertEqual(data.CAMERA_POLICY, worker.DEFAULT_PROTOCOL['training']['pair_policy'])
        self.assertEqual(data.DESIGN_SHA256, worker.DEFAULT_PROTOCOL['design']['sha256'])
        self.assertEqual(data.MAX_UPDATES, worker.DEFAULT_PROTOCOL['training']['max_updates'])
        self.assertEqual(list(data.A_EDGES), worker.DEFAULT_PROTOCOL['training']['severe_edges'])
        self.assertEqual(list(data.B_VARIANTS), worker.DEFAULT_PROTOCOL['training']['control_variants'])

    def test_interrupted_committed_step_retains_row_even_before_history_append(self):
        tree = ast.parse(SOURCE.read_text())
        run = next(node for node in tree.body if isinstance(node, ast.FunctionDef) and node.name == 'run')
        text = ast.get_source_segment(SOURCE.read_text(), run)
        committed = text.index("pending_step_record['optimizer_committed'] = True")
        self.assertLess(committed, text.index('budget()', committed))
        self.assertIn("'pending_step_record': pending_step_record", text)
        self.assertIn("'pending_step_record_preserves_committed_row_before_history_abort': True", text)


if __name__ == '__main__':
    unittest.main()
