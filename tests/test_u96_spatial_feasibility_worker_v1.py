"""Pure durable-ledger/state-format/source guards; never run the VM worker."""
import ast
import hashlib
import importlib
import io
import json
from pathlib import Path
import subprocess
import sys
import tempfile
import unittest
from unittest import mock

import numpy as np

from research.codeformer_u96_spatial_feasibility_v1 import worker as W


ROOT = Path(__file__).resolve().parents[1]


class DurableOperationEvidence(unittest.TestCase):
    def test_return_event_kind_cannot_be_overridden_by_pending_metadata(self):
        events = []
        ledger = W.OperationLedger(events.append)
        token = ledger.reserve('forward', category='independent_parity_forwards', id='case1')
        self.assertEqual(events[0]['kind'], 'forward_reserved')
        self.assertEqual(ledger.snapshot()['unknown_reservations'][0]['kind'], 'forward')
        ledger.returned(token, observed=True)
        self.assertEqual(events[1]['kind'], 'forward_returned')
        self.assertEqual(events[1]['token'], events[0]['token'])
        self.assertEqual(events[1]['id'], 'case1')
        self.assertFalse(ledger.snapshot()['operation_outcome_unknown'])
        self.assertEqual(ledger.counters['full_forward_returns'], 1)
        with self.assertRaises(ValueError):
            ledger.returned(token)

    def test_counter_metadata_collision_is_rejected(self):
        ledger = W.OperationLedger(lambda row: None)
        with self.assertRaises(ValueError):
            ledger.reserve('gradient', counters={})
        self.assertEqual(ledger.counters['gradient_query_attempts'], 0)
        token = ledger.reserve('gradient')
        with self.assertRaises(ValueError):
            ledger.returned(token, counters={})

    def test_failed_reservation_write_blocks_caller_and_leaves_unknown(self):
        def fail(row):
            raise OSError('fsync failure')
        ledger = W.OperationLedger(fail)
        with self.assertRaises(OSError):
            ledger.reserve('gradient', task='clear/rgb')
        self.assertEqual(ledger.counters['gradient_query_attempts'], 1)
        self.assertEqual(ledger.counters['gradient_queries'], 0)
        self.assertTrue(ledger.snapshot()['operation_outcome_unknown'])

    def test_failed_return_write_never_claims_durable_known_outcome(self):
        events = []
        ledger = W.OperationLedger(events.append)
        token = ledger.reserve('application')
        def fail(row):
            raise OSError('return receipt incomplete')
        ledger.persist = fail
        with self.assertRaises(OSError):
            ledger.returned(token)
        self.assertEqual(ledger.counters['proposal_applications'], 0)
        self.assertTrue(ledger.snapshot()['operation_outcome_unknown'])

    def test_exact_all_forward_categories_and_finite_global_caps(self):
        ledger = W.OperationLedger(lambda row: None)
        for category, count in W.CATEGORY_CAPS.items():
            for _ in range(count):
                ledger.returned(ledger.reserve('forward', category=category))
            with self.assertRaises(ValueError):
                ledger.reserve('forward', category=category)
        self.assertEqual(ledger.counters['full_forward_attempts'], 218)
        self.assertEqual(ledger.counters['full_forward_returns'], 218)
        for kind, attempt, returned in (
            ('gradient', 'gradient_query_attempts', 'gradient_queries'),
            ('application', 'proposal_application_attempts', 'proposal_applications')):
            for _ in range(W.OPERATION_CAPS[kind]):
                ledger.returned(ledger.reserve(kind))
            with self.assertRaises(ValueError):
                ledger.reserve(kind)
            self.assertEqual(ledger.counters[attempt], W.OPERATION_CAPS[kind])
            self.assertEqual(ledger.counters[returned], W.OPERATION_CAPS[kind])
        self.assertEqual(ledger.counters['optimizer_steps'], 0)
        self.assertEqual(ledger.counters['optimizer_constructions'], 0)

    def test_undeclared_operation_and_category_refused(self):
        ledger = W.OperationLedger(lambda row: None)
        with self.assertRaises(ValueError):
            ledger.reserve('optimizer')
        with self.assertRaises(ValueError):
            ledger.reserve('forward', category='extra_probe')
        self.assertFalse(any(ledger.counters.values()))


class ExclusiveBoundedArtifacts(unittest.TestCase):
    def setUp(self):
        (ROOT / 'scratch').mkdir(exist_ok=True)
        self.temporary = tempfile.TemporaryDirectory(prefix='spatial-worker-pure-', dir=ROOT / 'scratch')
        self.root = Path(self.temporary.name)
        def pin(root, path):
            value = path.read_bytes()
            return {'path': path.relative_to(root).as_posix(), 'bytes': len(value),
                    'sha256': hashlib.sha256(value).hexdigest()}
        self.store = W.DurableStore(self.root, self.root / 'outputs' / 'run', 1024 * 1024, pin)

    def tearDown(self):
        self.temporary.cleanup()

    def test_exclusive_payload_hash_and_no_hardlinks(self):
        binding = self.store.write('state/one.bin', b'123')
        self.assertEqual(binding['sha256'], hashlib.sha256(b'123').hexdigest())
        path = self.store.out / 'state/one.bin'
        self.assertEqual(path.stat().st_nlink, 1)
        self.assertEqual(self.store.audit_namespace(), {'complete': True, 'files': 1, 'bytes': 3})
        with self.assertRaises(ValueError):
            self.store.write('state/one.bin', b'changed')
        self.assertEqual(path.read_bytes(), b'123')

    def test_interrupted_payload_is_retained_and_not_accepted(self):
        with mock.patch.object(W.os, 'fsync', side_effect=OSError('interrupted fsync')):
            with self.assertRaises(OSError):
                self.store.write('partial.bin', b'unfinished')
        self.assertEqual((self.store.out / 'partial.bin').read_bytes(), b'unfinished')
        self.assertIn('partial.bin', self.store.partial_files)
        self.assertEqual(self.store.bytes, len(b'unfinished'))
        with self.assertRaises(ValueError):
            self.store.write('partial.bin', b'retry')
        self.store.audit_namespace()

    def test_output_cap_preserves_failure_receipt_reserve(self):
        with self.assertRaises(ValueError):
            self.store.write('too-large.bin', b'x' * (1024 * 1024))
        self.assertFalse((self.store.out / 'too-large.bin').exists())
        binding = self.store.write('failure.bin', b'x' * (1024 * 1024), emergency=True)
        self.assertEqual(binding['bytes'], 1024 * 1024)

    def test_status_owns_only_its_existing_regular_file(self):
        self.store.status({'phase': 'first'})
        self.store.status({'phase': 'second'})
        self.assertEqual(json.loads((self.store.out / 'status.json').read_text())['phase'], 'second')
        self.assertFalse((self.store.out / 'status.pending.json').exists())
        self.store.audit_namespace()

    def test_interrupted_status_pending_is_preserved(self):
        self.store.status({'phase': 'first'})
        with mock.patch.object(W.os, 'replace', side_effect=OSError('interrupted publish')):
            with self.assertRaises(OSError):
                self.store.status({'phase': 'second'})
        self.assertTrue((self.store.out / 'status.pending.json').exists())
        self.assertEqual(json.loads((self.store.out / 'status.json').read_text())['phase'], 'first')
        with self.assertRaises(ValueError):
            self.store.status({'phase': 'third'})
        self.store.audit_namespace()

    def test_append_and_namespace_detect_external_changes(self):
        self.store.write('events.jsonl', b'')
        self.store.append('events.jsonl', {'kind': 'reserved'})
        self.store.append('events.jsonl', {'kind': 'returned'})
        events = [json.loads(line) for line in (self.store.out / 'events.jsonl').read_text().splitlines()]
        self.assertEqual([row['kind'] for row in events], ['reserved', 'returned'])
        with (self.store.out / 'events.jsonl').open('ab') as stream:
            stream.write(b'outsider')
        with self.assertRaises(ValueError):
            self.store.append('events.jsonl', {})
        with self.assertRaises(ValueError):
            self.store.audit_namespace()

    def test_scope_escape_and_preexisting_run_refused(self):
        for relative in ('../escape.bin', str(self.root / 'absolute.bin')):
            with self.subTest(relative=relative), self.assertRaises(ValueError):
                self.store.write(relative, b'x')
        with self.assertRaises(ValueError):
            W.DurableStore(self.root, self.store.out, 1024 * 1024, lambda a, b: None)


class PortableStateAndStaticGuards(unittest.TestCase):
    def test_npz_state_canonical_is_name_shape_and_signed_zero_sensitive(self):
        arrays = {'base.a': np.array([1, 0], np.float32), 'adapters.b': np.array([[2]], np.float32)}
        payload, receipt = W.npz_state_payload(arrays, np=np)
        with np.load(io.BytesIO(payload), allow_pickle=False) as saved:
            self.assertEqual(set(saved.files), set(arrays))
            self.assertEqual(W.canonical_arrays({key: saved[key] for key in saved.files}, np=np)[0],
                             receipt['canonical_sha256'])
        renamed = {'base.changed': arrays['base.a'], 'adapters.b': arrays['adapters.b']}
        self.assertNotEqual(W.canonical_arrays(renamed, np=np)[0], receipt['canonical_sha256'])
        signed = {key: value.copy() for key, value in arrays.items()}
        signed['base.a'][1] = np.float32(-0.0)
        self.assertNotEqual(W.canonical_arrays(signed, np=np)[0], receipt['canonical_sha256'])

    def test_npz_refuses_nonfinite_object_or_pathlike_array_names(self):
        for name, value in (('x', np.array([np.nan], np.float32)), ('x', np.array([1], object)),
                            ('../escape', np.array([1], np.float32))):
            with self.subTest(name=name, dtype=value.dtype), self.assertRaises(ValueError):
                W.npz_state_payload({name: value}, np=np)

    def test_vm_supervisor_acknowledgement_binds_actual_parent(self):
        expected = 'a' * 64
        environ = {'FORENSIC_U96_SPATIAL_SUPERVISED': '1', 'FORENSIC_U96_SPATIAL_PROTOCOL': expected,
                   'FORENSIC_U96_SPATIAL_SUPERVISOR_PID': '123'}
        self.assertTrue(W.supervisor_context(expected, environ=environ, parent_pid=123))
        for changed in ({}, {**environ, 'FORENSIC_U96_SPATIAL_SUPERVISOR_PID': '124'},
                        {**environ, 'FORENSIC_U96_SPATIAL_PROTOCOL': 'b' * 64}):
            with self.assertRaises(ValueError):
                W.supervisor_context(expected, environ=changed, parent_pid=123)

    def test_failure_capture_has_separate_finite_budget_inside_external_stop(self):
        self.assertEqual(W.failure_capture_budget(1200), 30)
        self.assertEqual(W.failure_capture_budget(1240), 10)
        self.assertEqual(W.failure_capture_budget(1250), 0)
        self.assertEqual(W.failure_capture_budget(1300), 0)
        for elapsed in (0, 1199, 1200, 1235, 1249):
            self.assertLessEqual(elapsed + W.failure_capture_budget(elapsed) + 10, max(elapsed + 10, 1260))
        for invalid in (-1, float('nan'), float('inf')):
            with self.assertRaises(ValueError):
                W.failure_capture_budget(invalid)

    def test_import_has_no_numpy_torch_PIL_or_model_side_effect(self):
        code = ("import sys; from research.codeformer_u96_spatial_feasibility_v1 import worker; "
                "assert 'torch' not in sys.modules; assert 'numpy' not in sys.modules; "
                "assert 'PIL' not in sys.modules; "
                "assert 'research.codeformer_u96_spatial_feasibility_v1.model' not in sys.modules")
        result = subprocess.run([sys.executable, '-B', '-c', code], cwd=ROOT, capture_output=True, text=True, timeout=10)
        self.assertEqual(result.returncode, 0, result.stderr)

    def test_source_prospective_call_schedule_has_no_epochs_optimizer_or_per_case_clock(self):
        source = Path(W.__file__).read_text(encoding='utf-8')
        tree = ast.parse(source)
        run = next(node for node in tree.body if isinstance(node, ast.FunctionDef) and node.name == 'run')
        imports = [node for node in ast.walk(run) if isinstance(node, ast.Import)]
        torch_import = next(node.lineno for node in imports if any(alias.name == 'torch' for alias in node.names))
        calls = [node for node in ast.walk(run) if isinstance(node, ast.Call)]
        for name in ('verify_static', 'require_vm'):
            line = next(node.lineno for node in calls if isinstance(node.func, ast.Attribute) and node.func.attr == name)
            self.assertLess(line, torch_import)
        self.assertNotIn('torch.optim', source)
        self.assertNotIn('torch.save(', source)
        self.assertNotIn('bounded_phase(\'initializer_\' + active_case)', source)
        self.assertNotIn("bounded_phase(active_state + '_' + active_case)", source)
        self.assertIn("bounded_phase(active_state + '_72_case_bank')", source)
        self.assertIn("('initializer_gradient16'", source)
        self.assertIn("('initializer_monitor_guard56'", source)
        self.assertIn("bounded_phase('independent_frozen_U96_parity2')", source)
        self.assertIn('prior_proposal_passed=prior_proposal_passed', source)
        self.assertIn('prior_proposal_passed = bool(gate[\'pass\'])', source)

    def test_linux_rss_and_nonfinite_JSON_guards(self):
        self.assertEqual(W.linux_current_rss_bytes('100 5 3', 4096), 20480)
        with self.assertRaises(ValueError):
            W.linux_current_rss_bytes('100 bad', 4096)
        with self.assertRaises(ValueError):
            W.finite_json({'bad': float('nan')})


if __name__ == '__main__':
    unittest.main()
