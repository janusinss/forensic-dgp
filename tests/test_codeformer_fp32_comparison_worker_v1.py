"""Finite-worker CPU metadata/image fixtures; no model, autograd or optimizer."""
import ast
from contextlib import contextmanager
from copy import deepcopy
import io
import json
from pathlib import Path
import subprocess
import sys
import tempfile
import shutil
import tarfile
from types import SimpleNamespace
import unittest
from unittest.mock import patch

import numpy as np
from PIL import Image

from research.codeformer_fp32_comparison_v1 import worker as w
from research.codeformer_adaptation_v2 import worker as old


@contextmanager
def fixture_directory():
    # Python3.13's Windows700 temporary-directory ACL excludes this harness's
    # restricted token; default workspace directory permissions are appropriate.
    scratch = (Path.cwd() / 'scratch').resolve(strict=True)
    root = scratch / ('fp32-worker-fixture-' + w.uuid.uuid4().hex)
    root.mkdir()
    try:
        yield str(root)
    finally:
        checked = root.resolve(strict=True)
        assert checked.is_relative_to(scratch) and checked.parent == scratch and not root.is_symlink()
        shutil.rmtree(checked)


def valid_dtypes():
    return {'logits_dtype': 'torch.float32', 'probabilities_dtype': 'torch.float32',
            'native_output_dtype': 'torch.float32', 'autocast_enabled': False}


class FP32WorkerTests(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        (Path.cwd() / 'scratch').mkdir(exist_ok=True)

    def test_first_fifty_lr_exact_with_original_600_horizon(self):
        self.assertEqual(w.DEFAULT_PROTOCOL['training']['max_updates'], 50)
        self.assertEqual(w.DEFAULT_PROTOCOL['training']['schedule_horizon_updates'], 600)
        for update in range(1, 51):
            for base in (1e-5, 2e-5):
                self.assertEqual(w.learning_rate(update, base), old.learning_rate(update, base))
        with self.assertRaises(ValueError): w.learning_rate(51, 1e-5)

    def test_exact_reduced_census_derived_from_eligibility(self):
        self.assertEqual(w.expected_call_caps(), w.DEFAULT_PROTOCOL['training']['call_caps'])
        self.assertEqual(w.expected_call_caps()['perceptual_training_forwards'], 148)
        self.assertEqual(w.DEFAULT_PROTOCOL['training']['snapshots'], [0, 50])

    def test_unchanged_quality_guard_code_and_evaluation(self):
        self.assertEqual(w.DEFAULT_PROTOCOL['evaluation'], old.DEFAULT_PROTOCOL['evaluation'])
        for name in ('clear_quality_gate', 'blur_quality_gate', 'validate_training_examples'):
            left = ast.parse(Path(w.__file__).read_text())
            right = ast.parse(Path(old.__file__).read_text())
            fn = lambda tree: next(x for x in tree.body if isinstance(x, ast.FunctionDef) and x.name == name)
            self.assertEqual(ast.dump(fn(left)), ast.dump(fn(right)))
        baseline = {'x': {'raw': .1, 'native_png': .1}}
        changed = {'x': {'raw': .14, 'native_png': .1}}
        self.assertFalse(w.clear_quality_gate(changed, baseline)['pass'])

    def test_call_refused_before_extra_step(self):
        count = {'optimizer_step_attempts': 49}
        w.reserve_call(count, 'optimizer_step_attempts', 50)
        with self.assertRaises(ValueError): w.reserve_call(count, 'optimizer_step_attempts', 50)
        self.assertEqual(count['optimizer_step_attempts'], 50)

    def test_checkpoint_writer_caps_follow_actual_adam_state_not_update_number(self):
        empty_optimizer = SimpleNamespace(state={})
        partial_first_step = SimpleNamespace(state={'first_parameter': {'step': 0}})
        self.assertEqual(w.checkpoint_byte_cap(bool(empty_optimizer.state)), 384 * 1024**2)
        self.assertEqual(w.checkpoint_byte_cap(bool(partial_first_step.state)), 832 * 1024**2)
        self.assertLess(w.checkpoint_byte_cap(True), w.DEFAULT_PROTOCOL['limits']['checkpoint_temporary_bytes'])
        with self.assertRaises(ValueError): w.checkpoint_byte_cap(0)

    def test_regular_file_namespace_is_counted_for_export_bound(self):
        with fixture_directory() as temporary:
            root = Path(temporary)
            (root / 'one.json').write_text('{}')
            (root / 'two.npy.tmp-fixture').write_bytes(b'abcd')
            accounting = w.output_accounting(root)
            self.assertEqual(accounting['regular_files'], 2)
            self.assertEqual(accounting['temporary_bytes'], 4)
            self.assertEqual(w.REGULAR_OUTPUT_FILE_CAP, 7000)

    def test_initial_gate_accepts_tiny_signed_kl_but_no_drift(self):
        self.assertTrue(w.initial_path_gate(0., -1e-8, valid_dtypes())['pass'])
        for image, kl in ((1.1e-6, 0), (0, 1.1e-6), (float('nan'), 0), (-1, 0)):
            with self.assertRaises(ValueError): w.initial_path_gate(image, kl, valid_dtypes())

    def test_initial_gate_refuses_any_autocast_or_wrong_dtype(self):
        for key, value in (('autocast_enabled', True), ('logits_dtype', 'torch.bfloat16'),
                           ('probabilities_dtype', 'torch.bfloat16'), ('native_output_dtype', 'torch.bfloat16')):
            current = valid_dtypes(); current[key] = value
            with self.assertRaises(ValueError): w.initial_path_gate(0, 0, current)

    def test_runtime_free_space_changes_allowed_core_version_not_allowed(self):
        expected = {name: 'fixture' for name in w.RUNTIME_KEYS}
        expected.update(initial_free_bytes=123, pip_freeze='old inventory')
        actual = dict(expected, initial_free_bytes=987, pip_freeze='new inventory')
        self.assertTrue(w.verify_runtime_parity(actual, expected)['complete'])
        actual['torch'] = 'different version'
        with self.assertRaises(ValueError): w.verify_runtime_parity(actual, expected)

    def test_local_run_guard_precedes_import_and_io(self):
        with patch.object(w.platform, 'system', return_value='Windows'):
            with self.assertRaises(ValueError): w.run(Path('missing'), {}, 'a' * 64)

    def test_exact_vm_directory_and_manual_tmux_required(self):
        with fixture_directory() as temporary:
            home = Path(temporary); base = home / 'forensic-dgp'; base.mkdir()
            correct = base / w.PACKAGE_DIR; correct.mkdir()
            incorrect = base / 'historical'; incorrect.mkdir()
            with patch.object(w.platform, 'system', return_value='Linux'), patch.object(w.Path, 'home', return_value=home):
                with patch.dict(w.os.environ, {}, clear=True):
                    with self.assertRaises(ValueError): w.require_vm_path(correct)
                with patch.dict(w.os.environ, {'TMUX': 'human-session'}):
                    self.assertEqual(w.require_vm_path(correct), correct.resolve())
                    with self.assertRaises(ValueError): w.require_vm_path(incorrect)

    def test_source_readiness_covers_executables_and_exact_new_sources(self):
        names = ['research/codeformer_fp32_comparison_v1/worker.py',
            'research/codeformer_fp32_comparison_v1/__init__.py',
            'scripts/prepare_codeformer_fp32_comparison_v1.py',
            'scripts/supervise_codeformer_fp32_comparison_v1.py',
            'scripts/seal_codeformer_fp32_comparison_v1.py',
            'tests/test_codeformer_fp32_comparison_worker_v1.py']
        protocol = {'files_sha256': {name: 'a' * 64 for name in names}}
        receipt = {'complete': True, 'approved_for_protocol_freeze': True,
                   'training_authorized': False, 'source_sha256': dict(protocol['files_sha256'])}
        w.verify_source_readiness(protocol, receipt)
        receipt['source_sha256'][names[0]] = 'b' * 64
        with self.assertRaises(ValueError): w.verify_source_readiness(protocol, receipt)
        receipt['source_sha256'] = dict(protocol['files_sha256']); del receipt['source_sha256'][names[1]]
        with self.assertRaises(ValueError): w.verify_source_readiness(protocol, receipt)

    def test_namespace_rejects_unbound_importable_files(self):
        with fixture_directory() as temporary:
            root = Path(temporary); (root / 'protocol.json').write_text('{}')
            protocol = {'files_sha256': {}, 'dependency_files_sha256': {}}
            w.verify_packet_namespace(root, protocol)
            (root / 'dependency_install_receipt.json').write_text('{}')
            w.verify_packet_namespace(root, protocol)
            (root / 'torch.py').write_text('raise RuntimeError()')
            with self.assertRaises(ValueError): w.verify_packet_namespace(root, protocol)

    def test_capped_atomic_file_preserves_previous_output(self):
        with fixture_directory() as temporary:
            path = Path(temporary) / 'safe.bin'
            w._atomic_binary(path, lambda stream: stream.write(b'1234'), byte_cap=4)
            with self.assertRaises(ValueError): w._atomic_binary(path, lambda stream: stream.write(b'X'), byte_cap=4)
            self.assertEqual(path.read_bytes(), b'1234')
            with self.assertRaises(w.PilotStop):
                w._atomic_binary(Path(temporary) / 'partial.bin', lambda stream: stream.write(b'12345'), byte_cap=4)
            self.assertEqual(len(list(Path(temporary).glob('partial.bin.tmp-*'))), 1)

    def test_exact_pair_tensor_and_saved_png_required(self):
        with fixture_directory() as temporary:
            root = Path(temporary)
            pixels = np.full((256, 256, 3), 113, dtype=np.uint8)
            clean_pixels = np.full((512, 512, 3), 129, dtype=np.uint8)
            value = pixels.astype(np.float32) / np.float32(255)
            clean = clean_pixels.astype(np.float32) / np.float32(255)
            path = root / 'saved.png'; Image.fromarray(pixels).save(path)
            expected = {'input_path': 'saved.png', 'input_sha256': w.sha256(path),
                'input256_rgb8_sha256': w.array_sha(pixels, np), 'input256_float32_sha256': w.array_sha(value, np),
                'clean512_rgb8_sha256': w.array_sha(clean_pixels, np), 'clean512_float32_sha256': w.array_sha(clean, np),
                'update': 1, 'slot': 1, 'reference_id': 'tr_ffhq_00001', 'source_sha256': 'a' * 64,
                'profile': 'severe_normal', 'image_eligible': False}
            pair = SimpleNamespace(record=dict(expected, role='train'), input256=value,
                                   clean512=clean, parameters={'image_eligible': False})
            self.assertTrue(w.verify_pair_parity(root, expected, pair, np, Image)['complete'])
            pair.clean512 = clean.copy(); pair.clean512[0, 0, 0] += .001
            with self.assertRaises(ValueError): w.verify_pair_parity(root, expected, pair, np, Image)

    def test_existing_anchor_hash_and_actual_array_comparison(self):
        with fixture_directory() as temporary:
            root = Path(temporary)
            actuals = {'delivered': np.zeros((1, 3, 256, 256), np.float32),
                       'logits': np.zeros((1, 256, 1024), np.float32)}
            expected = {'update': 1, 'input256_float32_sha256': 'a' * 64}
            for name, array in actuals.items():
                path = root / (name + '.npy'); np.save(path, array, allow_pickle=False)
                expected[name + '_path'] = path.name; expected[name + '_sha256'] = w.sha256(path)
            self.assertTrue(w.verify_anchor_parity(root, expected, actuals['delivered'], actuals['logits'], np)['complete'])
            wrong = actuals['logits'].copy(); wrong[0, 0, 0] = .1
            difference = w.verify_anchor_parity(root, expected, actuals['delivered'], wrong, np)
            self.assertFalse(difference['comparisons']['logits']['exact_array_equal'])
            used = w.load_retained_anchor_arrays(root, expected, np)
            self.assertEqual(used['logits'][0, 0, 0], 0.)
            expected['logits_sha256'] = 'b' * 64
            with self.assertRaises(ValueError): w.load_retained_anchor_arrays(root, expected, np)

    def test_capture_uses_existing_calls_restores_hooks_on_exception(self):
        removed = []
        hook = SimpleNamespace(remove=lambda: removed.append(True))
        idx = SimpleNamespace(register_forward_hook=lambda callback: hook)
        original_lookup = lambda indices, shape: 'original result'
        calls = []
        original_softmax = lambda *args, **kwargs: calls.append(1) or SimpleNamespace(dtype='torch.float32')
        functional = SimpleNamespace(softmax=original_softmax)
        fake_torch = SimpleNamespace(nn=SimpleNamespace(functional=functional))
        net = SimpleNamespace(quantize=SimpleNamespace(get_codebook_feat=original_lookup), idx_pred_layer=idx)
        with self.assertRaises(RuntimeError):
            with w.capture_candidate_path(net, fake_torch) as captured:
                captured['logits_dtype'] = 'torch.float32'
                functional.softmax('already budgeted input')
                self.assertEqual(captured['probabilities_dtype'], 'torch.float32')
                raise RuntimeError('test restoration')
        self.assertIs(net.quantize.get_codebook_feat, original_lookup)
        self.assertIs(functional.softmax, original_softmax)
        self.assertEqual(calls, [1]); self.assertEqual(removed, [True])

    def test_export_refuses_live_worker_and_includes_closed_companions(self):
        with fixture_directory() as temporary:
            root = Path(temporary); (root / 'outputs/supervision').mkdir(parents=True)
            out = root / 'outputs/fp32_comparison_v1'; out.mkdir()
            destination = root / 'download'; destination.mkdir()
            sha = 'a' * 64
            lock = root / 'fp32_comparison_v1.lock'
            lock.write_text(json.dumps({'pid': 12345, 'host': w.socket.gethostname()}))
            with patch.object(w.platform, 'system', return_value='Linux'), patch.object(w.os, 'kill', return_value=None):
                with self.assertRaises(w.PilotStop): w.export_lock_state(root)
            lock.unlink()
            (root / 'dependency_install_receipt.json').write_text(json.dumps({'complete': True, 'protocol_sha256': sha}))
            (root / 'dependency_install_journal.jsonl').write_text('{"event":"copied"}\n')
            for log in ('preflight.log', 'trainer.log'):
                (root / 'outputs/supervision' / log).write_text('closed\n')
            run = {'protocol_sha256': sha, 'training_process_exit_code': 2,
                'stages': [{'stage': '--verify-transfer', 'exit_code': 0}, {'stage': '--run', 'exit_code': 2}]}
            (root / 'outputs/supervision/run_receipt.json').write_text(json.dumps(run))
            (root / 'outputs/supervision/export.log').write_text('still writing; excluded\n')
            operational = w.closed_export_companions(root, sha)
            self.assertEqual(len(operational), 5)
            self.assertNotIn('outputs/supervision/export.log', operational)
            for path, digest in operational.items(): self.assertEqual(w.sha256(root / path), digest)
            (root / 'protocol.json').write_text('{}')
            (root / 'source.py').write_text('"""no execution"""\n')
            (out / 'failure.json').write_text('{"complete":false,"quality_stop":true}')
            protocol = {'outputs': 'outputs/fp32_comparison_v1', 'export_prefix': 'fixture-only',
                'files_sha256': {'source.py': w.sha256(root / 'source.py')}, 'assets': {},
                'export_files': ['source.py'], 'limits': dict(w.DEFAULT_PROTOCOL['limits'], reserve_free_bytes=0)}
            real_sha = w.sha256(root / 'protocol.json')
            for relative in ('dependency_install_receipt.json', 'outputs/supervision/run_receipt.json'):
                record = json.loads((root / relative).read_text()); record['protocol_sha256'] = real_sha
                (root / relative).write_text(json.dumps(record))
            receipt = w._export_results(root, protocol, real_sha, destination=destination)
            self.assertTrue(receipt['complete']); self.assertTrue(receipt['failure_present'])
            with tarfile.open(destination / 'fixture-only-results.tar.gz') as bundle:
                names = bundle.getnames()
                self.assertIn('outputs/supervision/trainer.log', names)
                self.assertIn('dependency_install_receipt.json', names)
                self.assertNotIn('outputs/supervision/export.log', names)
                manifest = json.load(bundle.extractfile(receipt['manifest_member']))
                self.assertEqual(len(manifest['closed_operational_companions_sha256']), 5)
                self.assertEqual(manifest['files_sha256']['source.py'], protocol['files_sha256']['source.py'])

    def test_import_and_help_have_no_heavy_framework_dependency(self):
        code = "import sys; from research.codeformer_fp32_comparison_v1 import worker; assert not any(n in sys.modules for n in ('torch','numpy','PIL','cv2'))"
        result = subprocess.run([sys.executable, '-B', '-c', code], capture_output=True, text=True, timeout=20)
        self.assertEqual(result.returncode, 0, result.stderr)
        result = subprocess.run([sys.executable, '-B', '-m', 'research.codeformer_fp32_comparison_v1.worker', '--help'],
                                capture_output=True, text=True, timeout=20)
        self.assertEqual(result.returncode, 0, result.stderr)
        self.assertNotIn('--resume', result.stdout)


if __name__ == '__main__':
    unittest.main()
