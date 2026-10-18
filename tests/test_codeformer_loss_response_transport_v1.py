"""Synthetic stdlib-only transport/supervisor fixtures; no VM/neural execution."""
from contextlib import ExitStack, contextmanager
import hashlib
import importlib.util
import json
import os
import stat
from pathlib import Path
from types import SimpleNamespace
import tarfile
import unittest
from unittest import mock
import uuid

from research.codeformer_loss_response_v1 import transport as t
from scripts import supervise_codeformer_loss_response_v1 as s


def digest(data):
    return hashlib.sha256(data).hexdigest()


class TransferExportTests(unittest.TestCase):
    def fixture(self, *, failed=False):
        base = Path(__file__).resolve().parents[1] / 'scratch' / ('loss_transport_' + uuid.uuid4().hex)
        root, parent, home = base / 'new', base / 'parent', base / 'home'
        for path in (root, parent, home):
            path.mkdir(parents=True)
        (parent / 'protocol.json').write_bytes(b'{"retained":true}\n')
        (parent / 'original.pth').write_bytes(b'never instantiate or deserialize this synthetic model')
        files = {'research/synthetic.py': b'raise RuntimeError("must never execute")\n',
                 'cases.json': b'[]\n'}
        for name, payload in files.items():
            path = root / name; path.parent.mkdir(parents=True, exist_ok=True); path.write_bytes(payload)
        p = {'format': 'codeformer-loss-response-v1', 'parent_directory': 'parent',
             'parent_protocol_sha256': t.c.sha256(parent / 'protocol.json'), 'cases_file': 'cases.json',
             'files_sha256': {name: digest(data) for name, data in files.items()},
             'files_bytes': {name: len(data) for name, data in files.items()},
             'dependencies': {'original.pth': {'sha256': t.c.sha256(parent / 'original.pth'),
                                               'bytes': (parent / 'original.pth').stat().st_size}},
             'outputs': t.OUTPUTS}
        (root / 'protocol.json').write_text(json.dumps(p), encoding='utf-8')
        protocol_sha = t.c.sha256(root / 'protocol.json')
        out = root / t.OUTPUTS; out.mkdir(parents=True)
        completion = {'complete': not failed, 'protocol_sha256': protocol_sha,
                      'actual_completed_disposable_Adam_steps': 1 if failed else 18,
                      'counts': {'optimizer_step_attempts': 2 if failed else 18}}
        (out / ('failure.json' if failed else 'completion.json')).write_text(json.dumps(completion), encoding='utf-8')
        (out / 'raw.npy').write_bytes(b'synthetic array bytes; no ndarray loading')
        supervision = root / 'outputs/supervision'; supervision.mkdir()
        for name in ('preflight.log', 'diagnostic.log', 'export_invocation.json'):
            (supervision / name).write_bytes(b'{}\n')
        (supervision / 'run_receipt.json').write_text(json.dumps({'worker_terminal': True,
                                                               'protocol_sha256': protocol_sha}), encoding='utf-8')
        (supervision / 'export.log').write_bytes(b'active stream excluded explicitly')
        return SimpleNamespace(root=root, parent=parent, home=home, p=p, sha=protocol_sha)

    @contextmanager
    def fake_contract(self, f):
        with ExitStack() as stack:
            stack.enter_context(mock.patch.object(t.c, 'verify_protocol', side_effect=lambda p: p))
            stack.enter_context(mock.patch.object(t.c, 'verify_cases', side_effect=lambda p: p))
            stack.enter_context(mock.patch.object(t.c, 'resolve_parent', return_value=f.parent))
            stack.enter_context(mock.patch.object(t.c, 'manual_root', side_effect=lambda root: Path(root).resolve()))
            stack.enter_context(mock.patch.object(t, '_process_rows', return_value=[]))
            stack.enter_context(mock.patch.object(Path, 'home', return_value=f.home))
            stack.enter_context(mock.patch.object(t.shutil, 'disk_usage', return_value=SimpleNamespace(free=32 * 1024**3)))
            yield

    def rewrite(self, f):
        (f.root / 'protocol.json').write_text(json.dumps(f.p), encoding='utf-8')
        f.sha = t.c.sha256(f.root / 'protocol.json')

    def test_verify_is_readonly_and_does_not_load_parent_model(self):
        f = self.fixture()
        before = {p.relative_to(f.root).as_posix(): (p.stat().st_size, p.stat().st_mtime_ns)
                  for p in f.root.rglob('*') if p.is_file()}
        with self.fake_contract(f):
            receipt = t.verify_transfer(f.root, f.sha)
        after = {p.relative_to(f.root).as_posix(): (p.stat().st_size, p.stat().st_mtime_ns)
                 for p in f.root.rglob('*') if p.is_file()}
        self.assertEqual(before, after)
        self.assertTrue(receipt['readonly']); self.assertEqual(receipt['protocol'], f.p)
        self.assertEqual(receipt['dependencies_verified'], 1)
        self.assertEqual(receipt['neural_calls'], 0)

    def test_static_hash_and_size_and_parent_hash_are_binding(self):
        for location in ('source', 'source_size', 'parent'):
            f = self.fixture()
            if location == 'source':
                (f.root / 'research/synthetic.py').write_bytes(b'changed')
            elif location == 'source_size':
                f.p['files_bytes']['cases.json'] += 1; self.rewrite(f)
            else:
                (f.parent / 'original.pth').write_bytes(b'changed parent')
            with self.subTest(location=location), self.fake_contract(f), self.assertRaises(ValueError):
                t.verify_transfer(f.root, f.sha)

    def test_protocol_hash_and_unbound_source_refused(self):
        f = self.fixture()
        with self.fake_contract(f), self.assertRaisesRegex(ValueError, 'protocol hash'):
            t.verify_transfer(f.root, '0' * 64)
        (f.root / 'unbound.py').write_bytes(b'')
        with self.fake_contract(f), self.assertRaisesRegex(ValueError, 'Unbound'):
            t.verify_transfer(f.root, f.sha)

    def test_path_portability_and_case_collisions(self):
        for name in ('../escape', '/absolute', './bad', 'a//b', 'a\\b', 'C:/x', 'a/NUL.txt',
                     'a/trailing.', 'a/trailing ', 'a/COM¹.txt', 'a/control\x01', 'a/e\u0301'):
            with self.subTest(name=name), self.assertRaises(ValueError):
                t.canonical(name)
        f = self.fixture(); f.p['files_sha256']['CASES.json'] = digest(b'[]\n')
        f.p['files_bytes']['CASES.json'] = 3; self.rewrite(f)
        with self.fake_contract(f), self.assertRaises(ValueError):
            t.verify_transfer(f.root, f.sha)

    def test_linked_new_packet_file_refused(self):
        f = self.fixture()
        try:
            os.link(f.root / 'cases.json', f.root / 'linked.json')
        except OSError as error:
            self.skipTest('Hardlink unavailable: ' + str(error))
        with self.fake_contract(f), self.assertRaisesRegex(ValueError, 'link|regular'):
            t.verify_transfer(f.root, f.sha)

    def test_symlink_new_packet_file_refused(self):
        f = self.fixture()
        try:
            os.symlink(f.root / 'cases.json', f.root / 'linked.json')
        except OSError as error:
            self.skipTest('Symlink privilege unavailable: ' + str(error))
        with self.fake_contract(f), self.assertRaisesRegex(ValueError, 'link|reparse'):
            t.verify_transfer(f.root, f.sha)

    def test_link_metadata_guards_without_windows_link_privilege(self):
        f = self.fixture()
        with mock.patch.object(t, '_no_links', side_effect=lambda path: Path(path)), mock.patch.object(t.c, 'safe_path', return_value=f.root / 'cases.json'), mock.patch.object(Path, 'lstat', return_value=SimpleNamespace(st_mode=stat.S_IFREG | 0o644, st_nlink=2)), self.assertRaisesRegex(ValueError, 'Hard-linked'):
            t._regular(f.root, 'cases.json')
        with mock.patch.object(Path, 'lstat', return_value=SimpleNamespace(st_mode=stat.S_IFLNK | 0o777, st_file_attributes=0)), self.assertRaisesRegex(ValueError, 'Linked'):
            t._no_links(f.root / 'cases.json')

    def test_complete_archive_exact_namespace_hashes_checksum_and_boundary(self):
        f = self.fixture()
        parent_before = (f.parent / 'original.pth').read_bytes()
        with self.fake_contract(f):
            receipt = t.export_results(f.root, f.sha)
        archive = f.home / (t.PREFIX + '-results.tar.gz')
        self.assertTrue(receipt['complete_measurement'])
        self.assertEqual(receipt['actual_disposable_Adam_steps'], 18)
        self.assertEqual((Path(str(archive) + '.sha256')).read_bytes(),
                         (t.c.sha256(archive) + '  ' + archive.name + '\n').encode('ascii'))
        with tarfile.open(archive, 'r:gz') as bundle:
            self.assertTrue(all(member.isfile() for member in bundle.getmembers()))
            names = bundle.getnames(); self.assertEqual(len(names), len(set(names)))
            manifest = json.loads(bundle.extractfile(t.OUTPUTS + '/export_manifest.json').read())
            self.assertEqual(set(names), set(manifest['files_sha256']) | {t.OUTPUTS + '/export_manifest.json'})
            self.assertNotIn(t.OUTPUTS + '/export_manifest.json', manifest['files_sha256'])
            self.assertNotIn('outputs/supervision/export.log', names)
            self.assertNotIn('original.pth', names)
            self.assertIn(t.OUTPUTS + '/dependency_verification.json', names)
            for name, sha in manifest['files_sha256'].items():
                payload = bundle.extractfile(name).read()
                self.assertEqual(digest(payload), sha)
                self.assertEqual(len(payload), manifest['files_bytes'][name])
            self.assertEqual(sum(member.size for member in bundle.getmembers()), receipt['payload_bytes'])
        self.assertEqual(parent_before, (f.parent / 'original.pth').read_bytes())
        self.assertFalse(receipt['model_qualified'])

    def test_failed_run_exports_without_faking_zero_disposable_steps(self):
        f = self.fixture(failed=True)
        with self.fake_contract(f):
            receipt = t.export_results(f.root, f.sha)
        self.assertFalse(receipt['complete_measurement']); self.assertTrue(receipt['failure_present'])
        self.assertEqual(receipt['actual_disposable_Adam_steps'], 1)
        self.assertEqual(receipt['worker_recorded_counts']['optimizer_step_attempts'], 2)
        self.assertEqual(receipt['committed_training_updates'], 0)

    def test_every_terminal_receipt_requires_current_protocol_binding(self):
        for name in (t.OUTPUTS + '/completion.json', t.OUTPUTS + '/failure.json',
                     'outputs/supervision/run_receipt.json'):
            for stale in (None, '0' * 64):
                f = self.fixture(failed=name.endswith('/failure.json'))
                path = f.root / name
                value = json.loads(path.read_text())
                if stale is None:
                    value.pop('protocol_sha256')
                else:
                    value['protocol_sha256'] = stale
                path.write_text(json.dumps(value), encoding='utf-8')
                with self.subTest(name=name, stale=stale), self.fake_contract(f), self.assertRaisesRegex(ValueError, 'protocol binding'):
                    t.export_results(f.root, f.sha)
                self.assertFalse((f.home / (t.PREFIX + '-results.tar.gz')).exists())

    def test_worker_failure_inflight_flag_and_partial_attempt_survive_export(self):
        f = self.fixture(failed=True)
        path = f.root / t.OUTPUTS / 'failure.json'
        value = json.loads(path.read_text())
        value['optimizer_inflight_at_failure'] = True
        value['partial_disposable_step_possible'] = True
        path.write_text(json.dumps(value), encoding='utf-8')
        with self.fake_contract(f):
            receipt = t.export_results(f.root, f.sha)
        self.assertIs(receipt['worker_recorded_optimizer_inflight_at_failure'], True)
        self.assertEqual(receipt['worker_recorded_disposable_step_attempts'], 2)
        self.assertEqual(receipt['worker_recorded_completed_disposable_Adam_steps'], 1)

    def test_existing_partial_export_is_never_overwritten(self):
        f = self.fixture(); archive = f.home / (t.PREFIX + '-results.tar.gz'); archive.write_bytes(b'partial')
        with self.fake_contract(f), self.assertRaisesRegex(ValueError, 'Preserve existing'):
            t.export_results(f.root, f.sha)
        self.assertEqual(archive.read_bytes(), b'partial')

    def test_forbidden_checkpoint_and_unknown_output_namespace_refused(self):
        for name in (t.OUTPUTS + '/weights.pt', 'outputs/other/private.json'):
            f = self.fixture(); path = f.root / name; path.parent.mkdir(parents=True, exist_ok=True); path.write_bytes(b'x')
            with self.subTest(name=name), self.fake_contract(f), self.assertRaisesRegex(ValueError, 'forbidden|namespace'):
                t.export_results(f.root, f.sha)

    def test_live_worker_and_live_lock_block_export(self):
        f = self.fixture()
        with self.fake_contract(f), mock.patch.object(t, '_process_rows', return_value=[(987654, 'python -m ' + t.MODULE + ' --run')]), self.assertRaisesRegex(ValueError, 'live'):
            t.export_results(f.root, f.sha)
        (f.root / 'loss_response_v1.lock').write_text(json.dumps({'pid': 987654}), encoding='utf-8')
        with self.fake_contract(f), mock.patch.object(t.os, 'kill', return_value=None), self.assertRaisesRegex(ValueError, 'alive'):
            t.export_results(f.root, f.sha)

    def test_terminal_receipt_and_supervisor_terminal_flag_required(self):
        f = self.fixture(); (f.root / 'outputs/supervision/run_receipt.json').write_text('{}', encoding='utf-8')
        with self.fake_contract(f), self.assertRaisesRegex(ValueError, 'Supervisor'):
            t.export_results(f.root, f.sha)

    def test_payload_metadata_members_archive_and_disk_caps(self):
        for name, value in (('PAYLOAD_CAP', 1), ('METADATA_CAP', 1), ('MEMBER_CAP', 1), ('ARCHIVE_CAP', 1)):
            f = self.fixture()
            with self.subTest(cap=name), self.fake_contract(f), mock.patch.object(t, name, value), self.assertRaisesRegex(ValueError, 'cap|bound'):
                t.export_results(f.root, f.sha)
        f = self.fixture()
        with self.fake_contract(f), mock.patch.object(t.shutil, 'disk_usage', return_value=SimpleNamespace(free=t.RESERVE)), self.assertRaisesRegex(ValueError, 'reserve'):
            t.export_results(f.root, f.sha)
        self.assertFalse((f.root / t.OUTPUTS / 'dependency_verification.json').exists())

    def test_streaming_bound_and_time_bound(self):
        stream = mock.Mock(); stream.write.side_effect = lambda payload: len(payload)
        bounded = t.BoundedArchive(stream, 20, Path('.'))
        with mock.patch.object(t.time, 'monotonic', return_value=30), self.assertRaisesRegex(ValueError, 'time'):
            bounded.write(b'x')
        with mock.patch.object(t.time, 'monotonic', return_value=10), mock.patch.object(t.shutil, 'disk_usage', return_value=SimpleNamespace(free=32 * 1024**3)), mock.patch.object(t, 'ARCHIVE_CAP', 1), self.assertRaisesRegex(ValueError, 'cap'):
            bounded.write(b'xx')
        stream.write.assert_not_called()

    def test_no_neural_or_subprocess_calls_during_transfer(self):
        f = self.fixture()
        with self.fake_contract(f), mock.patch.object(t.subprocess, 'run', side_effect=AssertionError('No process during static verify')):
            result = t.verify_transfer(f.root, f.sha)
        self.assertEqual(result['optimizer_steps'], 0)
        self.assertNotIn('torch', t.__dict__)


class SupervisorTests(unittest.TestCase):
    def test_full_python_commands_detect_workers_and_ignore_idle_tmux(self):
        rows = [(1, 'tmux new-session -d -s idle python -m research.old --run'),
                (2, '/bin/bash'), (3, 'python -B -u -m research.codeformer_old.worker --run'),
                (4, 'timeout 1800 /venv/bin/python train_cctv_dgp_old.py'),
                (5, 'python scripts/supervise_codeformer_old.py --root /x'),
                (6, 'python app.py'), (7, 'python train.py'),
                (8, 'python scripts/supervise_codeformer_loss_response_v1.py --root /new')]
        conflicts = s.competing_commands(rows, 8)
        self.assertEqual([row['pid'] for row in conflicts], [3, 4, 5, 7])

    def test_unparseable_python_command_fails_closed(self):
        with self.assertRaisesRegex(ValueError, 'Cannot classify'):
            s.competing_commands([(1, 'python "unterminated')], 2)

    def test_occupied_gpu_refuses_without_killing_anything(self):
        with mock.patch.object(s.subprocess, 'run', return_value=SimpleNamespace(stdout='99,python,10\n')), mock.patch.object(s.os, 'killpg', create=True) as kill, self.assertRaisesRegex(ValueError, 'GPU occupied'):
            s.idle()
        kill.assert_not_called()

    def test_manual_guard_precedes_any_idle_or_stage(self):
        with mock.patch.object(s.c, 'manual_root', side_effect=ValueError('Linux guard')), mock.patch.object(s, 'idle') as idle, mock.patch.object(s, 'stage') as stage, self.assertRaisesRegex(ValueError, 'Linux guard'):
            s.main(['--root', '.', '--protocol-sha', 'a' * 64])
        idle.assert_not_called(); stage.assert_not_called()

    def test_failed_run_still_exports_with_exact_caps(self):
        f = TransferExportTests().fixture()
        # A separate fresh execution root, preserving every synthetic fixture.
        root = f.root.parent / 'supervisor_new'; root.mkdir()
        calls = []
        def stage(root, sha, flag, seconds, grace, log):
            calls.append((flag, seconds, grace)); log.write_bytes(b'')
            return {'exit_code': 2 if flag == '--run' else 0, 'worker_terminal': True}
        with mock.patch.object(s.c, 'manual_root', return_value=root), mock.patch.dict(s.os.environ, {'TMUX': 'synthetic'}), mock.patch.object(s.shutil, 'which', return_value='/usr/bin/timeout'), mock.patch.object(s, 'idle', return_value={}), mock.patch.object(s, 'stage', side_effect=stage):
            code = s.main(['--root', str(root), '--protocol-sha', 'a' * 64])
        self.assertEqual(code, 1)
        self.assertEqual(calls, [('--verify-transfer', 300, 30), ('--run', 1800, 60), ('--export', 600, 30)])
        receipt = json.loads((root / 'outputs/supervision/run_receipt.json').read_text())
        self.assertTrue(receipt['worker_terminal'])
        self.assertIsNone(receipt['actual_disposable_Adam_steps'])
        self.assertEqual(receipt['maximum_disposable_Adam_steps'], 18)

    def test_static_sources_never_import_worker_or_neural_framework(self):
        self.assertNotIn('torch', s.__dict__)
        source = Path(s.__file__).read_text(encoding='utf-8')
        self.assertNotIn('import worker', source)
        self.assertIn('start_new_session=True', source)
        self.assertIn("log.open('xb')", source)


if __name__ == '__main__':
    unittest.main()
