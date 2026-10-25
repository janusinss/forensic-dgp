"""Synthetic stdlib transport/status tests; no worker, model or VM execution.

Miniature archive fixtures patch all manual/static context and contain only
temporary synthetic bytes. Child-process control tests use inert mock children.
These checks do not execute the real supervisor, exporter CLI or worker.
"""
import ast
import contextlib
import copy
import hashlib
import importlib.abc
import io
import json
from pathlib import Path
import runpy
import signal
import subprocess
import sys
import tarfile
import tempfile
import time
from types import SimpleNamespace
import unittest
from unittest import mock


class ImportBlocker(importlib.abc.MetaPathFinder):
    attempted = []

    def find_spec(self, fullname, path=None, target=None):
        if fullname.split('.')[0] in {'torch', 'numpy', 'PIL', 'tensorflow', 'jax', 'jaxlib', 'keras'} or fullname.endswith(('.worker', '.model')):
            self.attempted.append(fullname)
            raise RuntimeError('Forbidden entry-test import:' + fullname)
        return None


BLOCKER = ImportBlocker()
sys.meta_path.insert(0, BLOCKER)
from research.codeformer_matched_feature_learning_v1 import entry as T, execution as E, contract as C, packet as P


ROOT = Path(__file__).resolve().parents[1]
PREP = ROOT / 'outputs/codeformer_matched_feature_learning_v1_preparation'
BOOTSTRAP = ROOT / 'scripts/run_matched_feature_learning_v1.py'
SHA = 'a' * 64


def json_bytes(value):
    return (json.dumps(value, sort_keys=True, separators=(',', ':'), allow_nan=False) + '\n').encode()


class StableStatus(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        cls.input_raws = ((PREP / 'data_admission_v1.json').read_bytes(),
            (PREP / 'candidate_data_v1.json').read_bytes(),
            (ROOT / 'outputs/cctv_codeformer_u96_decoder_readout_v1_vm/protocol.json').read_bytes())
        cls.selection = C.select_cases(*cls.input_raws)

    def setUp(self):
        self.temp = tempfile.TemporaryDirectory(dir=ROOT / 'scratch')
        self.addCleanup(self.temp.cleanup)
        self.root = Path(self.temp.name)
        self.out = self.root / P.OUTPUT_RELATIVE
        self.out.mkdir(parents=True)
        protocol = {'files': {}}
        for field, raw in zip(('admission', 'candidate_data', 'parent_protocol'), self.input_raws):
            path = self.root / (field + '.json'); path.write_bytes(raw)
            protocol[field] = P.pin(self.root, path)
            protocol['files'][path.name] = {key: protocol[field][key] for key in ('bytes', 'sha256')}
        raw = json_bytes(protocol)
        (self.root / 'protocol.json').write_bytes(raw)
        self.sha = hashlib.sha256(raw).hexdigest()
        self.lines = []
        self.ledger = E.ExecutionLedger(self.selection, persist=self.lines.append)
        self.addCleanup(self.ledger.close)
        self.write_status()

    def write_status(self, **changes):
        value = {'complete': False, 'protocol_sha256': self.sha, 'model_qualified': False,
                 'phase': 'synthetic_scalar_status', **self.ledger.summary()}
        value.update(changes)
        (self.out / 'operations.jsonl').write_bytes(b''.join(self.lines))
        (self.out / 'status.json').write_bytes(json_bytes(value))
        return value

    def read(self):
        def module(root, files, name):
            return E if name.endswith('.execution') else C
        with mock.patch.object(P, '_module', side_effect=module):
            return T.status(self.root, self.sha)

    def test_exact_real_ledger_counter_schema_matches_entry_caps(self):
        result = self.read()
        self.assertTrue(result['status_schema_and_protocol_verified'])
        self.assertEqual(set(result['confirmed_counters']), set(T.COUNTER_CAPS))
        self.assertTrue(result['journal_full_replay'])
        self.assertFalse(result['operation_outcome_unknown'])
        self.assertFalse(result['journal_independent_pixel_or_state_audit'])

    def test_pending_reserved_forward_is_explicitly_unknown_until_returned(self):
        token = self.ledger.reserve_forward(self.ledger.next_forward())
        self.write_status()
        result = self.read()
        self.assertTrue(result['operation_outcome_unknown'])
        self.assertEqual(result['confirmed_counters']['full_forward_attempts'], 1)
        self.assertEqual(result['confirmed_counters']['full_forward_returns'], 0)
        self.ledger.returned(token)
        self.write_status()
        self.assertFalse(self.read()['operation_outcome_unknown'])

    def test_newer_durable_reservation_cannot_hide_behind_stale_zero_status(self):
        self.ledger.reserve_forward(self.ledger.next_forward())
        (self.out / 'operations.jsonl').write_bytes(b''.join(self.lines))
        result = self.read()
        self.assertFalse(result['status_schema_and_protocol_verified'])
        self.assertIsNone(result['confirmed_counters'])
        self.assertTrue(result['operation_outcome_unknown'])

    def test_missing_partial_or_corrupt_journal_never_yields_false_zero_counters(self):
        for raw in (b'', b'{}', b'{"partial":', b'{}\n'):
            (self.out / 'operations.jsonl').write_bytes(raw)
            with self.subTest(raw=raw):
                result = self.read()
                self.assertIsNone(result['confirmed_counters'])
                self.assertTrue(result['operation_outcome_unknown'])
        (self.out / 'operations.jsonl').unlink()
        self.assertIsNone(self.read()['confirmed_counters'])

    def test_pending_status_or_unknown_malformed_numeric_fields_fail_closed(self):
        (self.out / 'status.pending.json').write_bytes(b'partial')
        self.assertTrue(self.read()['operation_outcome_unknown'])
        (self.out / 'status.pending.json').unlink()
        for field, value in (('journal_records', True), ('journal_records', 1.0),
                             ('journal_bytes', True), ('journal_bytes', 0),
                             ('journal_sha256', 'A' * 64), ('complete', 1), ('model_qualified', True)):
            self.write_status(**{field: value})
            with self.subTest(field=field, value=value):
                self.assertIsNone(self.read()['confirmed_counters'])
        value = self.write_status()
        for bad in (True, 0.0, -1, 1737):
            changed = copy.deepcopy(value)
            changed['counters']['full_forward_attempts'] = bad
            (self.out / 'status.json').write_bytes(json_bytes(changed))
            with self.subTest(counter=bad): self.assertIsNone(self.read()['confirmed_counters'])

    def test_status_or_journal_mutation_during_readback_is_rejected(self):
        real_pin = P.pin
        counts = {}
        def changed(root, path):
            result = real_pin(root, path)
            counts[path.name] = counts.get(path.name, 0) + 1
            if path.name == 'operations.jsonl' and counts[path.name] == 2:
                result['sha256'] = '0' * 64
            return result
        with mock.patch.object(P, 'pin', side_effect=changed):
            result = self.read()
        self.assertIsNone(result['confirmed_counters'])
        self.assertIn('changed during readback', result['status_read_error'])

    def test_absolute_alarm_is_never_swallowed_by_status_fallback(self):
        with mock.patch.object(T, '_checked_status', side_effect=TimeoutError('absolute alarm')):
            with self.assertRaises(TimeoutError): T.status(self.root, self.sha)


class QualityStopJoin(unittest.TestCase):
    def fixture(self):
        stops = {'A': {'event': 'quality_stop', 'arm': 'A', 'reason': 'bound checkpoint failed',
            'receipt_sha256': 'c' * 64, 'checkpoint': 24, 'skipped_count': 400,
            'skipped_slots_sha256': 'd' * 64}}
        result = {'complete': True, 'protocol_sha256': SHA, 'quality_stop': True,
            'model_qualified': False, 'quality_stops': {'A': {'update': 24,
                'reason': stops['A']['reason'], 'gate_receipt': {'path': 'synthetic/bank.json',
                    'bytes': 100, 'sha256': stops['A']['receipt_sha256']}}},
            'execution': {'quality_stops': copy.deepcopy(stops), 'status': 'FINISHED',
                'bookkeeping_complete': True}}
        status = {'status_schema_and_protocol_verified': True, 'quality_stops': copy.deepcopy(stops),
            'recorded_ledger_status': 'FINISHED', 'operation_outcome_unknown': False}
        return result, status

    def test_joined_matched_result_and_replay_report_quality_stop_without_failure_json(self):
        result, status = self.fixture()
        evidence = T.quality_stop_evidence(result, None, status, SHA)
        self.assertTrue(evidence['quality_stop_recorded'])
        self.assertEqual(evidence['recorded_quality_stop_arms'], ['A'])
        self.assertEqual(evidence['quality_stop_source'], 'matched_result_and_replayed_journal')

    def test_missing_mismatched_or_unverified_result_cannot_infer_stop_from_exit(self):
        result, status = self.fixture()
        for field, value in (('complete', False), ('complete', 1), ('protocol_sha256', 'e' * 64),
                             ('quality_stop', False), ('quality_stops', {}), ('model_qualified', True)):
            changed = copy.deepcopy(result); changed[field] = value
            with self.subTest(field=field):
                self.assertFalse(T.quality_stop_evidence(changed, None, status, SHA)['quality_stop_recorded'])
        bad = copy.deepcopy(result); bad['execution']['quality_stops'] = {}
        self.assertFalse(T.quality_stop_evidence(bad, None, status, SHA)['quality_stop_recorded'])
        for bad in ({}, {'status_schema_and_protocol_verified': False, 'quality_stops': result['quality_stops']}):
            self.assertFalse(T.quality_stop_evidence(result, None, bad, SHA)['quality_stop_recorded'])

    def test_fatal_failure_with_prior_arm_stop_is_not_clean_quality_completion(self):
        result, status = self.fixture()
        failure = {'fatal': True, 'execution': {'quality_stops': result['quality_stops']}}
        evidence = T.quality_stop_evidence(None, failure, status, SHA)
        self.assertFalse(evidence['quality_stop_recorded'])
        self.assertTrue(evidence['fatal_failure_present'])
        self.assertFalse(T.quality_stop_evidence(result, failure, status, SHA)['quality_stop_recorded'])

    def test_worker_stop_receipt_update_reason_and_hash_must_match_journal(self):
        result, status = self.fixture()
        for field, value in (('update', True), ('update', 48), ('reason', 'wrong reason'),
                             ('gate_receipt', {'sha256': 'e' * 64})):
            changed = copy.deepcopy(result); changed['quality_stops']['A'][field] = value
            with self.subTest(field=field):
                self.assertFalse(T.quality_stop_evidence(changed, None, status, SHA)['quality_stop_recorded'])
        for field, value in (('recorded_ledger_status', 'FATAL'), ('operation_outcome_unknown', True)):
            changed = copy.deepcopy(status); changed[field] = value
            with self.subTest(field=field):
                self.assertFalse(T.quality_stop_evidence(result, None, changed, SHA)['quality_stop_recorded'])


class SyntheticExportAndCleanup(unittest.TestCase):
    def setUp(self):
        self.temp = tempfile.TemporaryDirectory(dir=ROOT / 'scratch')
        self.addCleanup(self.temp.cleanup)
        self.home = Path(self.temp.name)
        self.root = self.home / 'synthetic-stage'; self.root.mkdir()
        (self.root / 'protocol.json').write_bytes(b'{"synthetic":true}\n')
        self.archive = self.home / (T.PREFIX + '-results.tar.gz')

    @contextlib.contextmanager
    def patched_export(self):
        with mock.patch.object(T, '_export_context'), \
             mock.patch.object(P, 'verify_static', return_value={'stage_name': P.STAGE_NAME}), \
             mock.patch.object(T.platform, 'system', return_value='Linux'), \
             mock.patch.object(T.shutil, 'disk_usage', return_value=SimpleNamespace(free=40 * 1024**3)), \
             mock.patch.object(Path, 'home', return_value=self.home), \
             mock.patch.object(T, 'status', return_value={'confirmed_counters': None,
                 'operation_outcome_unknown': True, 'status_schema_and_protocol_verified': False}), \
             contextlib.redirect_stdout(io.StringIO()):
            yield

    def export(self):
        with self.patched_export(): return T._export(self.root, SHA, time.monotonic())

    def test_small_synthetic_export_uses_plain_stream_tar_correct_three_names_and_hashes(self):
        (self.root / 'synthetic.bin').write_bytes(b'evidence' * 25)
        (self.root / 'weights').mkdir(); (self.root / 'weights/model.bin').write_bytes(b'omit weights')
        result = self.export()
        self.assertTrue(result['complete'])
        self.assertFalse(result['model_qualified'])
        self.assertTrue(result['training_success_not_implied'])
        self.assertEqual(result['archive_sha256'], hashlib.sha256(self.archive.read_bytes()).hexdigest())
        self.assertEqual(self.archive.with_name(self.archive.name + '.sha256').read_bytes(),
            (result['archive_sha256'] + '  ' + self.archive.name + '\n').encode())
        self.assertTrue((self.home / (T.PREFIX + '-export.json')).is_file())
        with tarfile.open(self.archive, 'r:gz') as archive:
            names = archive.getnames()
            self.assertEqual(names, [P.STAGE_NAME + '/protocol.json', P.STAGE_NAME + '/synthetic.bin'])
            self.assertEqual(archive.extractfile(names[1]).read(), b'evidence' * 25)
        self.assertEqual(T.EXPORT_SECONDS, 600)
        self.assertEqual(T.EXPORT_EXTERNAL_SECONDS, 660)
        self.assertEqual(T.EXPORT_BYTES, 12 * 1024**3)
        self.assertEqual(P.MAX_FILE, 2 * 1024**3)

    def test_unknown_worker_writer_blocks_before_archive_creation(self):
        for relative in ('run.log', 'supervisor.worker.started.json', P.OUTPUT_RELATIVE + '/started.json'):
            path = self.root / relative; path.parent.mkdir(parents=True, exist_ok=True); path.write_bytes(b'{}\n')
            with self.subTest(relative=relative), self.assertRaisesRegex(ValueError, 'Worker outcome unknown'):
                self.export()
            self.assertFalse(self.archive.exists())
            path.unlink()

    def test_worker_cleanup_schema_must_confirm_child_reaped_and_group_absent(self):
        for cleanup in (None, {}, {'child_reaped': True, 'process_group_absent': False},
                        {'child_reaped': False, 'process_group_absent': True}):
            (self.root / 'supervisor.worker.json').write_bytes(json_bytes({
                'complete': True, 'spawn_returned': True, 'exit_code': 2, 'cleanup': cleanup}))
            with self.subTest(cleanup=cleanup), self.assertRaisesRegex(ValueError, 'Known writer cleanup'):
                self.export()
            self.assertFalse(self.archive.exists())

    def test_previous_partial_archive_is_retained_exactly(self):
        self.archive.write_bytes(b'previous partial archive')
        with self.assertRaisesRegex(ValueError, 'Prior exports immutable'): self.export()
        self.assertEqual(self.archive.read_bytes(), b'previous partial archive')

    def test_changed_payload_hash_retains_partial_archive_and_omits_checksum(self):
        real_pin = P.pin
        calls = {}
        def changed(root, path):
            pin = real_pin(root, path)
            calls[path.name] = calls.get(path.name, 0) + 1
            if path.name == 'protocol.json' and calls[path.name] == 2:
                pin['sha256'] = '0' * 64
            return pin
        with mock.patch.object(P, 'pin', side_effect=changed), self.assertRaisesRegex(ValueError, 'changed after read'):
            self.export()
        self.assertTrue(self.archive.is_file())
        self.assertFalse(self.archive.with_name(self.archive.name + '.sha256').exists())

    def test_inventory_and_archive_aggregate_caps_fail_closed(self):
        with mock.patch.object(T, 'EXPORT_BYTES', 1), self.assertRaisesRegex(ValueError, 'payload exceeds'):
            self.export()
        self.assertFalse(self.archive.exists())

    def test_per_file_cap_is_separate_and_compressed_output_cap_retains_partial(self):
        with mock.patch.object(P, 'MAX_FILE', 1), self.assertRaisesRegex(ValueError, 'Bounded single-link'):
            self.export()
        self.assertFalse(self.archive.exists())
        with mock.patch.object(T, 'EXPORT_BYTES', 32), self.assertRaisesRegex(ValueError, 'archive exceeds'):
            self.export()
        self.assertTrue(self.archive.exists())
        self.assertFalse(self.archive.with_name(self.archive.name + '.sha256').exists())

    def test_absolute_alarm_during_metadata_parse_propagates(self):
        out = self.root / P.OUTPUT_RELATIVE; out.mkdir(parents=True)
        failure = b'{"fatal":true}\n'; (out / 'failure.json').write_bytes(failure)
        worker = {'complete': True, 'spawn_returned': True, 'exit_code': 1,
            'cleanup': {'child_reaped': True, 'process_group_absent': True}}
        (self.root / 'supervisor.worker.json').write_bytes(json_bytes(worker))
        real_parse = P.strict_json
        def parse(raw):
            if raw == failure: raise TimeoutError('synthetic absolute export alarm')
            return real_parse(raw)
        with mock.patch.object(P, 'strict_json', side_effect=parse), self.assertRaises(TimeoutError): self.export()
        self.assertTrue(self.archive.exists())

    def test_mock_child_timeout_records_PID_and_confirms_cleanup_without_real_process(self):
        child = mock.Mock(pid=45678)
        child.wait.side_effect = subprocess.TimeoutExpired('synthetic', 1)
        cleanup = {'child_reaped': True, 'process_group_absent': True, 'actions': ['synthetic_cleanup']}
        with mock.patch.object(T.subprocess, 'Popen', return_value=child) as spawn, \
             mock.patch.object(T, '_stop_group', return_value=cleanup), \
             contextlib.redirect_stdout(io.StringIO()):
            result = T._child(self.root, ['synthetic-never-executed'], phase='worker', timeout=1,
                log_name='mock.log', env={'TEST': 'only'})
        self.assertTrue(result['timeout']); self.assertEqual(result['exit_code'], 124)
        self.assertTrue(spawn.call_args.kwargs['start_new_session'])
        self.assertEqual(json.loads((self.root / 'supervisor.worker.started.json').read_bytes())['child_pid'], 45678)

    def test_mock_owned_group_cleanup_never_targets_other_PIDs(self):
        child = mock.Mock(pid=77777)
        child.poll.return_value = 0
        with mock.patch.object(T, '_group_alive', side_effect=[True, True, False]), \
             mock.patch.object(T.os, 'killpg', create=True) as kill, \
             mock.patch.object(T.signal, 'SIGKILL', 9, create=True):
            result = T._stop_group(child, 20)
        self.assertTrue(result['child_reaped']); self.assertTrue(result['process_group_absent'])
        self.assertEqual([call.args[0] for call in kill.call_args_list], [77777, 77777])
        self.assertEqual(result['actions'], ['SIGTERM', 'SIGKILL'])

    def test_mock_supervision_exports_quality_exit2_only_after_known_writer_cleanup(self):
        worker = {'complete': True, 'exit_code': 2, 'timeout': False, 'spawn_returned': True,
            'cleanup': {'child_reaped': True, 'process_group_absent': True}}
        exported = {'complete': True, 'exit_code': 0, 'timeout': False}
        with mock.patch.object(P, 'require_vm'), \
             mock.patch.object(T, '_child', side_effect=[worker, exported]) as child, \
             mock.patch.object(T, 'status', return_value={'operation_outcome_unknown': False}), \
             contextlib.redirect_stdout(io.StringIO()):
            code = T.supervise(self.root, {'limits': C.LIMITS}, SHA)
        self.assertEqual(code, 2); self.assertEqual(child.call_count, 2)
        self.assertEqual(child.call_args_list[0].kwargs['timeout'], 5460)
        self.assertEqual(child.call_args_list[1].kwargs['timeout'], 660)
        self.assertEqual(child.call_args_list[0].args[1][-1], '--run')
        self.assertEqual(child.call_args_list[1].args[1][-1], '--export')
        env = child.call_args_list[0].kwargs['env']
        self.assertEqual(env[T.ENV_SUPERVISED], '1')
        self.assertEqual(env[T.ENV_PROTOCOL], SHA)
        self.assertEqual(env[T.ENV_PARENT], str(T.os.getpid()))

    def test_mock_supervision_unknown_cleanup_skips_export_without_inventing_exit(self):
        worker = {'complete': True, 'exit_code': 124, 'timeout': True, 'spawn_returned': True,
            'cleanup': {'child_reaped': False, 'process_group_absent': False}, 'orphan_cleanup_unknown': True}
        with mock.patch.object(P, 'require_vm'), \
             mock.patch.object(T, '_child', return_value=worker) as child, \
             mock.patch.object(T, 'status', return_value={'operation_outcome_unknown': True}), \
             contextlib.redirect_stdout(io.StringIO()):
            code = T.supervise(self.root, {'limits': C.LIMITS}, SHA)
        self.assertEqual(code, 124); self.assertEqual(child.call_count, 1)
        result = json.loads((self.root / 'supervisor.export.json').read_bytes())
        self.assertTrue(result['skipped']); self.assertIsNone(result['exit_code'])
        self.assertTrue(result['active_writer_outcome_unknown'])

    def test_prior_supervisor_artifact_blocks_any_mock_child_dispatch(self):
        (self.root / 'supervisor.start.json').write_bytes(b'previous partial')
        with mock.patch.object(P, 'require_vm'), mock.patch.object(T, '_child') as child:
            with self.assertRaisesRegex(ValueError, 'Prior launch immutable'):
                T.supervise(self.root, {'limits': C.LIMITS}, SHA)
        child.assert_not_called()


class ManualBootstrapBoundaries(unittest.TestCase):
    def test_bootstrap_authenticates_all_parents_and_direct_imports(self):
        bootstrap = runpy.run_path(str(BOOTSTRAP), run_name='synthetic_bootstrap_import_only')
        names = ('scripts/run_matched_feature_learning_v1.py', 'research/__init__.py',
                 P.PACKAGE + '/__init__.py', P.PACKAGE + '/packet.py', P.PACKAGE + '/entry.py')
        with tempfile.TemporaryDirectory(dir=ROOT / 'scratch') as temp:
            root = Path(temp); files = {}
            for name in names:
                path = root / name; path.parent.mkdir(parents=True, exist_ok=True)
                path.write_bytes(('synthetic:' + name).encode())
                files[name] = {'bytes': path.stat().st_size, 'sha256': hashlib.sha256(path.read_bytes()).hexdigest()}
            bootstrap['authenticate'](root, {'files': files})
            for name in names:
                changed = copy.deepcopy(files); changed[name]['sha256'] = '0' * 64
                with self.subTest(name=name), self.assertRaisesRegex(ValueError, 'Sealed bootstrap source differs'):
                    bootstrap['authenticate'](root, {'files': changed})

    def test_help_subprocess_import_blocker_prevents_any_staged_package_import(self):
        code = '''import importlib.abc,runpy,sys
class Block(importlib.abc.MetaPathFinder):
 def find_spec(self,fullname,path=None,target=None):
  if fullname=='research' or fullname.startswith('research.') or fullname.split('.')[0] in {'torch','numpy','PIL'}:
   raise RuntimeError('Forbidden help import:'+fullname)
  return None
sys.meta_path.insert(0,Block())
sys.argv=[sys.argv[1],'--help']
runpy.run_path(sys.argv[0],run_name='__main__')
'''
        result = subprocess.run([sys.executable, '-B', '-c', code, str(BOOTSTRAP)],
            cwd=ROOT, capture_output=True, text=True, timeout=10)
        self.assertEqual(result.returncode, 0, result.stdout + result.stderr)
        self.assertIn('--verify-transfer', result.stdout)

    def test_bootstrap_duplicate_nonfinite_or_overflow_JSON_is_rejected(self):
        bootstrap = runpy.run_path(str(BOOTSTRAP), run_name='synthetic_bootstrap_import_only')
        for raw in (b'{"a":1,"a":2}', b'{"a":NaN}', b'{"a":Infinity}', b'{"a":1e999}'):
            with self.subTest(raw=raw), self.assertRaises(ValueError): bootstrap['strict_json'](raw)

    def test_export_context_requires_manual_exact_stage_but_allows_existing_outputs(self):
        home = ROOT / 'scratch/synthetic-export-home'
        root = home / 'forensic-dgp' / P.STAGE_NAME
        def identity(path): return Path(path)
        with mock.patch.object(P, 'directory', side_effect=identity), \
             mock.patch.object(Path, 'home', return_value=home), \
             mock.patch.object(Path, 'cwd', return_value=root), \
             mock.patch.object(T, '__file__', str(root / P.PACKAGE / 'entry.py')), \
             mock.patch.object(T.sys, 'prefix', str(home / 'forensic-dgp/cctv_dgp_vm_bundle/.venv')), \
             mock.patch.object(T.platform, 'system', return_value='Linux'), \
             mock.patch.dict(T.os.environ, {'TMUX': 'synthetic'}):
            T._export_context(root)
            with mock.patch.dict(T.os.environ, {'TMUX': ''}), self.assertRaisesRegex(ValueError, 'Exact manual Linux'):
                T._export_context(root)
            with mock.patch.object(Path, 'cwd', return_value=Path('/wrong')), self.assertRaises(ValueError):
                T._export_context(root)

    def test_source_order_and_Python310_compatible_stream_tar_are_preserved(self):
        entry_tree = ast.parse(Path(T.__file__).read_bytes(), feature_version=(3, 10))
        bootstrap_text = BOOTSTRAP.read_text(encoding='utf-8')
        ast.parse(bootstrap_text, feature_version=(3, 10))
        self.assertLess(bootstrap_text.index('authenticate(root, protocol)\n'),
                        bootstrap_text.index("module = importlib.import_module('research."))
        main = next(node for node in entry_tree.body if isinstance(node, ast.FunctionDef) and node.name == 'main')
        worker_import = next(node for node in ast.walk(main) if isinstance(node, ast.ImportFrom)
                             and any(alias.name == 'worker' for alias in node.names))
        verify = next(node for node in ast.walk(main) if isinstance(node, ast.Call)
                      and isinstance(node.func, ast.Attribute) and node.func.attr == 'verify_static')
        self.assertLess(verify.lineno, worker_import.lineno)
        source_lines = Path(T.__file__).read_text(encoding='utf-8').splitlines()
        worker_tail = '\n'.join(source_lines[worker_import.lineno - 1:])
        self.assertIn('type(result) is int', worker_tail)
        self.assertIn('result in (0, 1, 2)', worker_tail)
        self.assertNotIn('else 0', worker_tail)
        tar_calls = [node for node in ast.walk(entry_tree) if isinstance(node, ast.Call)
                     and isinstance(node.func, ast.Attribute) and isinstance(node.func.value, ast.Name)
                     and node.func.value.id == 'tarfile' and node.func.attr == 'open']
        self.assertEqual(len(tar_calls), 1)
        self.assertNotIn('compresslevel', {keyword.arg for keyword in tar_calls[0].keywords})
        self.assertTrue(any(keyword.arg == 'mode' and isinstance(keyword.value, ast.Constant)
                            and keyword.value.value == 'w|' for keyword in tar_calls[0].keywords))
        shell = (ROOT / P.PACKAGE / 'run_matched.sh').read_text(encoding='utf-8')
        self.assertIn('python -B -u scripts/run_matched_feature_learning_v1.py', shell)
        self.assertIn('--supervise', shell); self.assertIn('exit "$worker_status"', shell)
        self.assertNotIn('|| true', shell)
        self.assertEqual(BLOCKER.attempted, [])


if __name__ == '__main__':
    unittest.main()
