"""Small byte/transport fixtures only; no model, neural, VM or optimizer work."""
import contextlib
import copy
import hashlib
import io
import json
import os
from pathlib import Path
import subprocess
import stat
import sys
import tarfile
import tempfile
import time
from types import SimpleNamespace
import unittest
from unittest import mock

from research.codeformer_u96_spatial_feasibility_v1 import contract
from research.codeformer_u96_spatial_feasibility_v1 import entry, install_weights, packet

ROOT = Path(__file__).resolve().parents[1]
PARENT = ROOT / 'outputs/cctv_codeformer_u96_decoder_readout_v1_vm/protocol.json'
PACKAGE = 'research/codeformer_u96_spatial_feasibility_v1'


def byte_pin(name, raw):
    return {'path': name, 'bytes': len(raw), 'sha256': hashlib.sha256(raw).hexdigest()}


class SmallPacket(unittest.TestCase):
    def setUp(self):
        scratch = ROOT / 'scratch'
        scratch.mkdir(exist_ok=True)
        self.temp = tempfile.TemporaryDirectory(dir=scratch)
        self.addCleanup(self.temp.cleanup)
        self.root = Path(self.temp.name) / packet.STAGE_NAME
        self.root.mkdir()
        parent = json.loads(PARENT.read_bytes())
        self.files = {}
        def add(name, raw):
            p = self.root / name
            p.parent.mkdir(parents=True, exist_ok=True)
            p.write_bytes(raw)
            binding = byte_pin(name, raw)
            self.files[name] = {k: binding[k] for k in ('bytes', 'sha256')}
            return binding
        shared = add('assets/synthetic.bin', b'synthetic data fixture only')
        document = add('evidence/data/synthetic.json', b'{}\n')
        def replace_bindings(value):
            if type(value) is dict:
                if set(value) == {'path', 'bytes', 'sha256'}:
                    return copy.deepcopy(shared)
                return {k: replace_bindings(v) for k, v in value.items()}
            if type(value) is list:
                return [replace_bindings(v) for v in value]
            return value
        parent['cases'] = replace_bindings(parent['cases'])
        parent['guards'] = replace_bindings(parent['guards'])
        parent['data'] = {k: copy.deepcopy(document) for k in parent['data']}
        parent['model'] = {key: add('weights/' + key + '.pth', ('fake-' + key).encode())
                           for key in ('r2', 'u96')}
        model_fixed = {k: (v['bytes'], v['sha256']) for k, v in parent['model'].items()}
        prototype = {key: add(PACKAGE + '/' + key + '.py', (ROOT / PACKAGE / (key + '.py')).read_bytes())
                     for key in packet.PROTOTYPE_FIXED}
        parent_raw = (json.dumps(parent, indent=2) + '\n').encode()
        parent_pin = add('evidence/spatial_parent_protocol.json', parent_raw)
        self.addCleanup(mock.patch.stopall)
        mock.patch.object(packet, 'PARENT_PROTOCOL_FIXED', (len(parent_raw), parent_pin['sha256'])).start()
        mock.patch.object(packet, 'MODEL_FIXED', model_fixed).start()
        mock.patch.object(contract, 'PARENT_PROTOCOL_BYTES', len(parent_raw)).start()
        mock.patch.object(contract, 'PARENT_PROTOCOL_SHA256', parent_pin['sha256']).start()
        self.protocol = {
            'format': packet.FORMAT, 'stage_name': packet.STAGE_NAME,
            'files': self.files, 'model': parent['model'], 'prototype': prototype,
            'parent_protocol': parent_pin, 'data': parent['data'],
            'cases': parent['cases'], 'guards': parent['guards'],
            'quality_policy': parent['quality_policy'], 'limits': copy.deepcopy(packet.LIMITS),
            'feasibility': contract.prospective_contract(), 'manual_launch_only': True,
            'automatic_retry_resume': False, 'model_qualified': False,
            'training_success_not_implied': True,
        }
        self.reseal()

    def reseal(self):
        raw = (json.dumps(self.protocol, indent=2) + '\n').encode()
        (self.root / 'protocol.json').write_bytes(raw)
        self.sha = hashlib.sha256(raw).hexdigest()

    def test_small_exact_table_verifies_and_never_imports_torch(self):
        p = packet.verify_static(self.root, self.sha)
        self.assertEqual(len(p['cases']), 96)
        self.assertEqual(len(p['guards']), 24)
        self.assertEqual(p['limits']['optimizer_steps'], 0)
        self.assertNotIn('torch', sys.modules)

    def test_missing_weights_only_permitted_during_install(self):
        for value in self.protocol['model'].values():
            (self.root / value['path']).unlink()
        packet.verify_static(self.root, self.sha, missing_weights_for_install=True)
        with self.assertRaises(FileNotFoundError):
            packet.verify_static(self.root, self.sha)

    def test_changed_payload_rejected_even_when_protocol_is_same(self):
        (self.root / 'assets/synthetic.bin').write_bytes(b'changed')
        with self.assertRaises(ValueError):
            packet.verify_static(self.root, self.sha)

    def test_changed_case_or_limit_refused_even_if_resealed(self):
        self.protocol['cases'][0]['role'] = 'unadmitted'
        self.reseal()
        with self.assertRaises(ValueError):
            packet.verify_static(self.root, self.sha)
        self.protocol['cases'][0]['role'] = 'local_gaussian_1p2'
        self.protocol['limits']['full_forward_attempts'] = 219
        self.reseal()
        with self.assertRaises(ValueError):
            packet.verify_static(self.root, self.sha)

    def test_extra_file_and_cache_directory_refused(self):
        extra = self.root / 'unreviewed.py'
        extra.write_text('pass\n')
        with self.assertRaises(ValueError):
            packet.verify_static(self.root, self.sha)
        extra.unlink()
        (self.root / '__pycache__').mkdir()
        with self.assertRaises(ValueError):
            packet.verify_static(self.root, self.sha)

    def test_runtime_output_is_preserved_without_allowing_sibling_namespace(self):
        out = self.root / packet.OUTPUT_RELATIVE
        out.mkdir(parents=True)
        (out / 'partial.json').write_bytes(b'partial unknown outcome')
        packet.verify_static(self.root, self.sha)
        sibling = self.root / 'outputs/unrelated'
        sibling.mkdir()
        with self.assertRaises(ValueError):
            packet.verify_static(self.root, self.sha)

    def test_exact_core_pin_and_LF_protocol_required(self):
        self.protocol['prototype']['model']['sha256'] = '0' * 64
        self.reseal()
        with self.assertRaises(ValueError):
            packet.verify_static(self.root, self.sha)
        raw = (self.root / 'protocol.json').read_bytes().replace(b'\n', b'\r\n')
        (self.root / 'protocol.json').write_bytes(raw)
        with self.assertRaises(ValueError):
            packet.verify_static(self.root, hashlib.sha256(raw).hexdigest())


class Helpers(unittest.TestCase):
    def test_duplicate_nonfinite_JSON_and_unsafe_paths_refused(self):
        for raw in (b'{"a":1,"a":2}', b'{"a":NaN}', b'{"a":1e999}'):
            with self.assertRaises(ValueError):
                packet.strict_json(raw)
            with self.assertRaises(ValueError):
                install_weights.strict_json(raw)
        for path in ('../a', '/a', './a', 'a//b', 'C:/a', 'a\\b', '.'):
            with self.assertRaises(ValueError):
                packet.relative(path)
        self.assertEqual(packet.relative('a/b').as_posix(), 'a/b')

    def test_hardlinks_are_refused(self):
        with tempfile.TemporaryDirectory(dir=ROOT / 'scratch') as tmp:
            p = Path(tmp) / 'one'
            p.write_bytes(b'a')
            alias = Path(tmp) / 'two'
            try:
                os.link(p, alias)
            except PermissionError:
                # Windows sandbox can forbid link creation. Exercise the exact lstat
                # rejection branch without claiming a real linked-file fixture ran.
                original = Path.lstat
                def observed(path):
                    if path in (p, alias):
                        return SimpleNamespace(st_mode=stat.S_IFREG | 0o600, st_nlink=2,
                                               st_size=1, st_file_attributes=0)
                    return original(path)
                with mock.patch.object(Path, 'lstat', observed):
                    for item in (p, alias):
                        with self.assertRaises(ValueError):
                            packet.regular(item)
            else:
                for item in (p, alias):
                    with self.assertRaises(ValueError):
                        packet.regular(item)

    def test_exclusive_stream_copy_and_partial_retention(self):
        with tempfile.TemporaryDirectory(dir=ROOT / 'scratch') as tmp:
            root = Path(tmp); source = root / 'source'; target = root / 'target'
            raw = b'a' * (1024**2 + 20); source.write_bytes(raw)
            pin = byte_pin('target', raw)
            install_weights.copy_verified(source, target, pin, lambda: None)
            self.assertEqual(target.read_bytes(), raw)
            with self.assertRaises(FileExistsError):
                install_weights.copy_verified(source, target, pin, lambda: None)
            second = root / 'partial'; calls = 0
            def stop():
                nonlocal calls
                calls += 1
                if calls == 2:
                    raise TimeoutError('fixture bounded stop')
            with self.assertRaises(TimeoutError):
                install_weights.copy_verified(source, second, pin, stop)
            self.assertEqual(second.stat().st_size, 1024**2)
            self.assertEqual(source.read_bytes(), raw)

    def test_missing_corrupt_and_pending_status_keep_unknown(self):
        with tempfile.TemporaryDirectory(dir=ROOT / 'scratch') as tmp:
            root = Path(tmp)
            out = root / packet.OUTPUT_RELATIVE
            out.mkdir(parents=True)
            missing = entry.status(root, 'a' * 64)
            self.assertIsNone(missing['confirmed_counters'])
            self.assertTrue(missing['operation_outcome_unknown'])
            (out / 'status.json').write_bytes(b'{broken')
            corrupt = entry.status(root, 'a' * 64)
            self.assertIsNone(corrupt['confirmed_counters'])
            (out / 'status.json').write_text(json.dumps({'protocol_sha256': 'a' * 64,
                'counters': {name: 0 for name in entry.COUNTER_CAPS}}))
            (out / 'events.jsonl').write_text(json.dumps({'kind': 'phase_returned',
                'counters': {name: 0 for name in entry.COUNTER_CAPS}}) + '\n')
            self.assertFalse(entry.status(root, 'a' * 64)['operation_outcome_unknown'])
            (out / 'status.pending.json').write_bytes(b'unknown')
            self.assertTrue(entry.status(root, 'a' * 64)['operation_outcome_unknown'])

    def test_new_durable_reservation_cannot_be_hidden_by_stale_status(self):
        with tempfile.TemporaryDirectory(dir=ROOT / 'scratch') as tmp:
            root = Path(tmp); out = root / packet.OUTPUT_RELATIVE; out.mkdir(parents=True)
            counters = {name: 0 for name in entry.COUNTER_CAPS}
            (out / 'status.json').write_text(json.dumps({'protocol_sha256': 'a' * 64, 'counters': counters}))
            journal = out / 'events.jsonl'
            newer = {**counters, 'full_forward_attempts': 1}
            journal.write_text(json.dumps({'kind': 'forward_reserved', 'counters': newer}) + '\n')
            result = entry.status(root, 'a' * 64)
            self.assertTrue(result['operation_outcome_unknown'])
            self.assertFalse(result['journal_tail_reconciled'])
            self.assertEqual(result['confirmed_counters']['full_forward_attempts'], 0)
            journal.write_text(json.dumps({'kind': 'phase_returned', 'counters': counters}) + '\n')
            self.assertFalse(entry.status(root, 'a' * 64)['operation_outcome_unknown'])
            journal.write_bytes(b'{"interrupted":')
            self.assertTrue(entry.status(root, 'a' * 64)['operation_outcome_unknown'])

    def test_attempt_return_gap_and_bad_counter_are_unknown(self):
        with tempfile.TemporaryDirectory(dir=ROOT / 'scratch') as tmp:
            root = Path(tmp); out = root / packet.OUTPUT_RELATIVE; out.mkdir(parents=True)
            value = {'protocol_sha256': 'a' * 64, 'counters': {name: 0 for name in entry.COUNTER_CAPS}}
            value['counters']['proposal_application_attempts'] = 1
            path = out / 'status.json'; path.write_text(json.dumps(value))
            answer = entry.status(root, 'a' * 64)
            self.assertTrue(answer['status_schema_and_protocol_verified'])
            self.assertTrue(answer['operation_outcome_unknown'])
            value['counters']['optimizer_steps'] = 1
            path.write_text(json.dumps(value))
            self.assertFalse(entry.status(root, 'a' * 64)['status_schema_and_protocol_verified'])

    def test_CLI_help_is_inert(self):
        result = subprocess.run([sys.executable, '-B', '-m',
            'research.codeformer_u96_spatial_feasibility_v1.entry', '--help'],
            cwd=ROOT, capture_output=True, text=True, timeout=10, check=False)
        self.assertEqual(result.returncode, 0, result.stderr)
        self.assertIn('--verify-transfer', result.stdout)
        self.assertIn('--status', result.stdout)
        self.assertNotIn('torch', sys.modules)

    def test_real_staged_package_bootstrap_help_uses_scripts_bridge(self):
        with tempfile.TemporaryDirectory(dir=ROOT / 'scratch') as tmp:
            stage = Path(tmp) / packet.STAGE_NAME
            scripts = stage / 'scripts'; scripts.mkdir(parents=True)
            package = stage / PACKAGE; package.mkdir(parents=True)
            for name in ('__init__.py', 'stage_entry.py', 'entry.py', 'packet.py'):
                (package / name).write_bytes((ROOT / PACKAGE / name).read_bytes())
            bridge = scripts / 'run_u96_spatial_feasibility_v1.py'
            bridge.write_text("import sys\nfrom pathlib import Path\nsys.dont_write_bytecode=True\n"
                              "root=Path(__file__).resolve().parents[1]\nsys.path.insert(0,str(root))\n"
                              "from research.codeformer_u96_spatial_feasibility_v1.stage_entry import main\n"
                              "raise SystemExit(main())\n", encoding='utf-8')
            result = subprocess.run([sys.executable, '-B', str(bridge), '--help'], cwd=stage,
                                    capture_output=True, text=True, timeout=10, check=False)
            self.assertEqual(result.returncode, 0, result.stderr)
            self.assertIn('--supervise', result.stdout)
            self.assertFalse(list(stage.rglob('__pycache__')))


class MiniatureExport(unittest.TestCase):
    def test_missing_supervisor_start_cannot_bypass_existing_worker_evidence(self):
        for evidence in ('run.log', packet.OUTPUT_RELATIVE + '/started.json',
                         'supervisor.worker.started.json'):
            with tempfile.TemporaryDirectory(dir=ROOT / 'scratch') as tmp:
                root = Path(tmp)
                item = root / evidence
                item.parent.mkdir(parents=True, exist_ok=True)
                item.write_bytes(b'{}\n')
                with mock.patch.object(entry.packet, 'verify_static', return_value={'stage_name': packet.STAGE_NAME}), \
                     mock.patch.object(entry.platform, 'system', return_value='Linux'), \
                     mock.patch.object(entry.shutil, 'disk_usage', return_value=SimpleNamespace(free=20 * 1024**3)), \
                     mock.patch.object(Path, 'home', return_value=root):
                    with self.assertRaisesRegex(ValueError, 'Worker outcome unknown'):
                        entry._export(root, 'a' * 64, time.monotonic())
                self.assertFalse(list(root.glob('*.tar.gz')))

    def test_manual_export_cannot_bypass_unknown_writer_cleanup(self):
        with tempfile.TemporaryDirectory(dir=ROOT / 'scratch') as tmp:
            root = Path(tmp)
            (root / 'supervisor.start.json').write_bytes(b'{}\n')
            with mock.patch.object(entry.packet, 'verify_static', return_value={'stage_name': packet.STAGE_NAME}), \
                 mock.patch.object(entry.platform, 'system', return_value='Linux'), \
                 mock.patch.object(entry.shutil, 'disk_usage', return_value=SimpleNamespace(free=20 * 1024**3)), \
                 mock.patch.object(Path, 'home', return_value=root):
                with self.assertRaises(ValueError):
                    entry._export(root, 'a' * 64, time.monotonic())
                (root / 'supervisor.worker.json').write_text(json.dumps({'complete': True,
                    'spawn_returned': True, 'cleanup': {'child_reaped': False, 'process_group_absent': False}}))
                with self.assertRaises(ValueError):
                    entry._export(root, 'a' * 64, time.monotonic())
            self.assertFalse(list(root.glob('*.tar.gz')))

    def test_streamed_export_preserves_failure_and_skips_initializer_aliases(self):
        with tempfile.TemporaryDirectory(dir=ROOT / 'scratch') as tmp:
            home = Path(tmp); root = home / 'stage'; root.mkdir()
            (root / 'protocol.json').write_bytes(b'{}\n')
            out = root / packet.OUTPUT_RELATIVE; out.mkdir(parents=True)
            raw = b'full-state miniature bytes'; (out / 'state.npz').write_bytes(raw)
            (out / 'failure.json').write_text(json.dumps({'status': 'QUALITY_STOP'}))
            (root / 'supervisor.worker.json').write_text(json.dumps({'complete': True, 'exit_code': 2,
                'spawn_returned': True, 'cleanup': {'child_reaped': True, 'process_group_absent': True}}))
            weights = root / 'weights'; weights.mkdir(); (weights / 'r2.pth').write_bytes(b'not exported alias')
            with mock.patch.object(entry.packet, 'verify_static', return_value={'stage_name': packet.STAGE_NAME}), \
                 mock.patch.object(entry.platform, 'system', return_value='Linux'), \
                 mock.patch.object(entry.shutil, 'disk_usage', return_value=SimpleNamespace(free=20 * 1024**3)), \
                 mock.patch.object(Path, 'home', return_value=home), contextlib.redirect_stdout(io.StringIO()):
                result = entry._export(root, 'a' * 64, time.monotonic())
                with self.assertRaises(ValueError):
                    entry._export(root, 'a' * 64, time.monotonic())
            archive = home / (entry.PREFIX + '-results.tar.gz')
            sha = hashlib.sha256(archive.read_bytes()).hexdigest()
            self.assertEqual(result['archive_sha256'], sha)
            self.assertTrue(result['quality_stop_exit2'])
            self.assertTrue(result['failure_present'])
            self.assertTrue(result['training_success_not_implied'])
            self.assertFalse(result['model_qualified'])
            self.assertIsNone(result['confirmed_counters'])
            self.assertEqual(Path(str(archive) + '.sha256').read_bytes(),
                             (sha + '  ' + archive.name + '\n').encode('ascii'))
            with tarfile.open(archive, 'r:gz') as tar:
                names = tar.getnames()
                self.assertTrue(all(name.startswith(packet.STAGE_NAME + '/') for name in names))
                self.assertFalse(any('/weights/' in name for name in names))
                member = packet.STAGE_NAME + '/' + packet.OUTPUT_RELATIVE + '/state.npz'
                self.assertEqual(tar.extractfile(member).read(), raw)

    def test_payload_budget_refusal_does_not_delete_failure(self):
        with tempfile.TemporaryDirectory(dir=ROOT / 'scratch') as tmp:
            root = Path(tmp); (root / 'evidence.bin').write_bytes(b'x' * 100)
            with mock.patch.object(entry.packet, 'verify_static', return_value={'stage_name': packet.STAGE_NAME}), \
                 mock.patch.object(entry.platform, 'system', return_value='Linux'), \
                 mock.patch.object(entry.shutil, 'disk_usage', return_value=SimpleNamespace(free=20 * 1024**3)), \
                 mock.patch.object(Path, 'home', return_value=root), mock.patch.object(entry, 'EXPORT_BYTES', 10):
                with self.assertRaises(ValueError):
                    entry._export(root, 'a' * 64, time.monotonic())
            self.assertEqual((root / 'evidence.bin').read_bytes(), b'x' * 100)


class BoundedSupervision(unittest.TestCase):
    def test_any_prior_worker_evidence_blocks_spawn_even_without_start_receipt(self):
        for evidence in ('supervisor.worker.started.json', 'supervisor.worker.json',
                         packet.OUTPUT_RELATIVE + '/failure.json'):
            with tempfile.TemporaryDirectory(dir=ROOT / 'scratch') as tmp:
                root = Path(tmp)
                item = root / evidence
                item.parent.mkdir(parents=True, exist_ok=True)
                item.write_bytes(b'{}\n')
                with mock.patch.object(entry.packet, 'require_vm'), \
                     mock.patch.object(entry, '_child', side_effect=AssertionError('No spawn allowed')) as child:
                    with self.assertRaisesRegex(ValueError, 'Prior launch immutable'):
                        entry.supervise(root, {'limits': dict(packet.LIMITS)}, 'a' * 64)
                    child.assert_not_called()

    def test_owned_group_gets_TERM_KILL_and_known_absence(self):
        child = SimpleNamespace(pid=31337,
            wait=mock.Mock(side_effect=[subprocess.TimeoutExpired('fixture', 10), -9]),
            poll=mock.Mock(return_value=-9))
        with mock.patch.object(entry, '_group_alive', side_effect=[True, True, False]), \
             mock.patch.object(entry.os, 'killpg', create=True) as kill, \
             mock.patch.object(entry.signal, 'SIGKILL', 9, create=True):
            result = entry._stop_group(child, 20)
        self.assertEqual(result['actions'], ['SIGTERM', 'SIGKILL'])
        self.assertTrue(result['child_reaped'])
        self.assertTrue(result['process_group_absent'])
        self.assertEqual([x.args[0] for x in kill.call_args_list], [31337, 31337])

    def test_quality_exit_retains_PID_and_parent_cleanup_receipt(self):
        with tempfile.TemporaryDirectory(dir=ROOT / 'scratch') as tmp:
            root = Path(tmp)
            child = SimpleNamespace(pid=31337, wait=mock.Mock(return_value=2))
            cleanup = {'actions': [], 'child_reaped': True, 'process_group_absent': True, 'seconds': 0.}
            with mock.patch.object(entry.subprocess, 'Popen', return_value=child) as create, \
                 mock.patch.object(entry, '_stop_group', return_value=cleanup), \
                 contextlib.redirect_stdout(io.StringIO()):
                result = entry._child(root, ['fixture'], phase='worker', timeout=1260,
                                      log_name='run.log', env={'ONLY': 'fixture'})
            started = json.loads((root / 'supervisor.worker.started.json').read_text())
            self.assertEqual(started['child_pid'], 31337)
            self.assertEqual(result['exit_code'], 2)
            self.assertEqual(result['cleanup'], cleanup)
            self.assertTrue(create.call_args.kwargs['start_new_session'])

    def test_timeout_with_unknown_cleanup_stays_failure(self):
        with tempfile.TemporaryDirectory(dir=ROOT / 'scratch') as tmp:
            root = Path(tmp)
            child = SimpleNamespace(pid=31337,
                wait=mock.Mock(side_effect=subprocess.TimeoutExpired('fixture', 1260)))
            cleanup = {'actions': ['SIGTERM', 'SIGKILL'], 'child_reaped': False,
                       'process_group_absent': False, 'seconds': 20.}
            with mock.patch.object(entry.subprocess, 'Popen', return_value=child), \
                 mock.patch.object(entry, '_stop_group', return_value=cleanup), \
                 contextlib.redirect_stdout(io.StringIO()):
                result = entry._child(root, ['fixture'], phase='worker', timeout=1260,
                                      log_name='run.log', env={})
            self.assertEqual(result['exit_code'], 124)
            self.assertTrue(result['timeout'])
            self.assertTrue(result['orphan_cleanup_unknown'])
            self.assertFalse(result['model_qualified'])

    def test_quality_stop_still_attempts_export_without_changing_exit(self):
        with tempfile.TemporaryDirectory(dir=ROOT / 'scratch') as tmp:
            root = Path(tmp)
            worker = {'complete': True, 'exit_code': 2, 'timeout': False, 'spawn_returned': True,
                      'cleanup': {'child_reaped': True, 'process_group_absent': True}}
            exported = {'complete': True, 'exit_code': 0, 'timeout': False}
            with mock.patch.object(entry.packet, 'require_vm'), \
                 mock.patch.object(entry, '_child', side_effect=[worker, exported]) as children, \
                 mock.patch.object(entry, 'status', return_value={'operation_outcome_unknown': True}), \
                 contextlib.redirect_stdout(io.StringIO()):
                code = entry.supervise(root, {'limits': dict(packet.LIMITS)}, 'a' * 64)
            self.assertEqual(code, 2)
            self.assertEqual(children.call_count, 2)
            self.assertEqual(children.call_args_list[0].args[1][-1], '--run')
            self.assertEqual(children.call_args_list[1].args[1][-1], '--export')
            self.assertEqual(children.call_args_list[0].kwargs['timeout'], 1260)
            self.assertEqual(children.call_args_list[1].kwargs['timeout'], 330)
            result = json.loads((root / 'supervisor.json').read_text())
            self.assertTrue(result['quality_stop_exit2'])
            self.assertFalse(result['model_qualified'])

    def test_unknown_cleanup_blocks_export_process_without_fabricating_exit(self):
        with tempfile.TemporaryDirectory(dir=ROOT / 'scratch') as tmp:
            root = Path(tmp)
            worker = {'complete': True, 'exit_code': 124, 'timeout': True, 'spawn_returned': True,
                      'orphan_cleanup_unknown': True,
                      'cleanup': {'child_reaped': False, 'process_group_absent': False}}
            with mock.patch.object(entry.packet, 'require_vm'), \
                 mock.patch.object(entry, '_child', return_value=worker) as children, \
                 mock.patch.object(entry, 'status', return_value={'operation_outcome_unknown': True}), \
                 contextlib.redirect_stdout(io.StringIO()):
                code = entry.supervise(root, {'limits': dict(packet.LIMITS)}, 'a' * 64)
            self.assertEqual(code, 124)
            self.assertEqual(children.call_count, 1)
            exported = json.loads((root / 'supervisor.export.json').read_text())
            self.assertTrue(exported['skipped'])
            self.assertIsNone(exported['exit_code'])
            self.assertTrue(exported['active_writer_outcome_unknown'])
            result = json.loads((root / 'supervisor.json').read_text())
            self.assertFalse(result['export_attempted'])
            self.assertIsNone(result['export_exit'])


if __name__ == '__main__':
    unittest.main()
