"""Bounded no-neural tests for V2 packaging, portable TAR and manual supervisor."""
import ast
import hashlib
import io
import json
from pathlib import Path
import shutil
import tarfile
import unittest
import uuid

from scripts import prepare_codeformer_adaptation_packet_v2 as prep
from scripts import seal_codeformer_adaptation_packet_v2 as seal
from scripts import supervise_codeformer_adaptation_v2 as supervisor

ROOT = Path(__file__).resolve().parents[1]


class PacketTests(unittest.TestCase):
    def setUp(self):
        self.work = ROOT / 'scratch' / ('codeformer_v2_packet_static_' + uuid.uuid4().hex)
        self.work.mkdir()
        self.packet = self.work / 'packet'; self.packet.mkdir()

    def tearDown(self):
        target = self.work.resolve(strict=True)
        assert target.parent == (ROOT / 'scratch').resolve(strict=True)
        assert target.name.startswith('codeformer_v2_packet_static_') and not self.work.is_symlink()
        shutil.rmtree(target)

    def test_python310_and_no_neural_imports(self):
        for module in (prep, seal, supervisor):
            tree = ast.parse(Path(module.__file__).read_text(encoding='utf-8'), feature_version=(3, 10))
            for node in ast.walk(tree):
                if isinstance(node, ast.Import):
                    self.assertFalse({'torch', 'numpy', 'PIL', 'cv2', 'tensorflow'} &
                                     {x.name.split('.')[0] for x in node.names})
                if isinstance(node, ast.ImportFrom):
                    self.assertNotIn((node.module or '').split('.')[0], {'torch', 'numpy', 'PIL', 'cv2'})

    def test_portable_paths_reject_traversal_and_devices(self):
        for name in ('', '.', '../x', 'x/../y', '/root', 'x//y', 'x/./y', 'C:/x', 'x\\y',
                     'NUL.txt', 'x/COM¹', 'x/file.', 'x/file ', 'x/a?.png', 'x/q".json'):
            with self.subTest(name=name), self.assertRaises(ValueError):
                prep.canonical(name)
        self.assertEqual(prep.canonical('data/crop_1/input.png'), 'data/crop_1/input.png')

    def test_copy_exact_hash_and_journal(self):
        source = self.work / 'original.bin'; source.write_bytes(b'original research bytes')
        pin = prep.sha256(source)
        bindings, copies = {}, {}
        with (self.work / 'journal.jsonl').open('x', encoding='utf-8', newline='\n') as journal:
            prep.copy_exact(self.work, self.packet, 'original.bin', 'data/copy.bin', pin, bindings, copies, journal)
        self.assertEqual((self.packet / 'data/copy.bin').read_bytes(), source.read_bytes())
        self.assertEqual(bindings, {'data/copy.bin': pin})
        events = [json.loads(x) for x in (self.work / 'journal.jsonl').read_text().splitlines()]
        self.assertEqual([x['event'] for x in events], ['intent', 'copied'])
        self.assertEqual(copies['data/copy.bin']['sha256'], pin)
        self.assertEqual(prep.exact_namespace(self.packet), {'data/copy.bin'})

    def test_copy_wrong_hash_refuses_before_write(self):
        (self.work / 'original.bin').write_bytes(b'original')
        with (self.work / 'journal').open('x', encoding='utf-8') as journal, self.assertRaises(ValueError):
            prep.copy_exact(self.work, self.packet, 'original.bin', 'copy.bin', '0' * 64, {}, {}, journal)
        self.assertFalse((self.packet / 'copy.bin').exists())

    def test_existing_evidence_never_overwritten(self):
        target = self.work / 'retained.json'; target.write_bytes(b'retained')
        with self.assertRaises(FileExistsError):
            prep.write_json(target, {'new': True})
        self.assertEqual(target.read_bytes(), b'retained')
        (self.work / 'outputs').mkdir()
        (self.work / 'outputs' / prep.NAME).mkdir()
        with self.assertRaises(ValueError):
            prep.stage_inputs(self.work)

    def test_export_policy_keeps_notices_and_no_model_data(self):
        for name in ('LICENSE', 'third_party/codeformer/LICENSE', 'README.md',
                     'research/codeformer_adaptation_v2/worker.py', 'validation_saved_v1.json'):
            self.assertTrue(prep.export_companion(name))
        for name in ('data/input.png', 'data/pair_record.json', 'weights/model.pth', 'model.pth', 'a.log'):
            self.assertFalse(prep.export_companion(name))

    def test_no_neural_import_guard_blocks_worker_dependency(self):
        with prep.no_neural_imports(), self.assertRaises(ValueError):
            __import__('torch')
        path = self.packet / 'research/codeformer_adaptation_v2/worker.py'
        path.parent.mkdir(parents=True)
        path.write_text('import torch\n', encoding='utf-8')
        with self.assertRaises(ValueError):
            prep.load_worker(self.packet)

    def test_checksum_and_runner_are_LF_only(self):
        path = self.work / 'fixture.tar.gz.sha256'
        seal.write_lf_sidecar(path, 'a' * 64, 'fixture.tar.gz')
        self.assertEqual(path.read_bytes(), ('a' * 64 + '  fixture.tar.gz\n').encode('ascii'))
        runner = prep.shell_runner()
        self.assertNotIn(b'\r', runner)
        self.assertIn(b'set -euo pipefail', runner)
        self.assertIn(b'supervise_codeformer_adaptation_v2.py', runner)
        self.assertNotIn(b'--run', runner)

    def archive(self, rows):
        path = self.work / 'fixture.tar.gz'
        with tarfile.open(path, 'w:gz') as out:
            for name, payload, kind in rows:
                member = tarfile.TarInfo(name); member.size = len(payload); member.type = kind
                if kind != tarfile.REGTYPE:
                    member.linkname = '../../escape'
                out.addfile(member, io.BytesIO(payload))
        return path

    def test_archive_full_payload_and_exact_namespace(self):
        payload = b'source fixture\n'
        path = self.archive([(prep.NAME + '/worker.py', payload, tarfile.REGTYPE)])
        pin = hashlib.sha256(payload).hexdigest()
        result = seal.inspect_archive(path, prep.NAME, {'worker.py': pin})
        self.assertEqual(result['members'], 1)
        self.assertEqual(result['payload_bytes'], len(payload))
        with self.assertRaises(ValueError):
            seal.inspect_archive(path, prep.NAME, {'worker.py': 'f' * 64})
        with self.assertRaises(ValueError):
            seal.inspect_archive(path, prep.NAME, {'different.py': pin})

    def test_archive_traversal_links_and_case_collision(self):
        samples = [
            [(prep.NAME + '/../outside', b'x', tarfile.REGTYPE)],
            [(prep.NAME + '/linked', b'', tarfile.SYMTYPE)],
            [(prep.NAME + '/hard', b'', tarfile.LNKTYPE)],
            [(prep.NAME + '/dev', b'', tarfile.CHRTYPE)],
            [(prep.NAME + '/Case', b'a', tarfile.REGTYPE), (prep.NAME + '/case', b'b', tarfile.REGTYPE)],
        ]
        for rows in samples:
            with self.subTest(rows=rows):
                path = self.archive(rows)
                pins = {name.split('/', 1)[1]: hashlib.sha256(payload).hexdigest() for name, payload, _ in rows}
                with self.assertRaises(ValueError):
                    seal.inspect_archive(path, prep.NAME, pins)

    def test_finalization_requires_independent_approval(self):
        path = self.work / 'readiness.json'
        prep.write_json(path, {'complete': True, 'approved_for_protocol_freeze': False})
        pin = prep.sha256(path)
        with self.assertRaises(ValueError):
            prep.finalize(self.work, 'readiness.json', pin)
        self.assertFalse((self.work / 'outputs' / prep.NAME / 'protocol.json').exists())

    def test_manual_linux_tmux_guard_and_own_idle_session(self):
        base = self.work / 'forensic-dgp'; base.mkdir()
        packet = base / 'new_packet'; packet.mkdir()
        with self.assertRaises(ValueError):
            supervisor.guard_environment(packet, 'a' * 64, system='Windows', home=self.work, environment={'TMUX': 'x'})
        with self.assertRaises(ValueError):
            supervisor.guard_environment(packet, 'a' * 64, system='Linux', home=self.work, environment={})
        self.assertEqual(supervisor.guard_environment(packet, 'a' * 64, system='Linux',
            home=self.work, environment={'TMUX': '/tmp/tmux,0,0'}), packet.resolve())
        rows = [(1, 'tmux new-session -A -s dgp_training'), (2, 'python audit.py'),
                (3, 'python -B scripts/supervise_codeformer_adaptation_v2.py --root .')]
        self.assertEqual(supervisor.competing_python_commands(rows, 3), [])

    def test_competing_training_and_finite_stage_command(self):
        rows = [(10, 'python -m research.codeformer_adaptation_v1.worker --root . --run'),
                (11, '/venv/bin/python train_cctv_dgp_test.py'), (12, 'python analyze.py --run'),
                (13, 'tmux dgp_training')]
        self.assertEqual({x['pid'] for x in supervisor.competing_python_commands(rows, 99)}, {10, 11})
        command = supervisor.stage_command(self.packet, 'a' * 64, '--run', 7200, 180)
        self.assertEqual(command[:4], ['timeout', '--signal=TERM', '--kill-after=180', '7200'])
        self.assertEqual(command[-1], '--run')
        self.assertIn('research.codeformer_adaptation_v2.worker', command)
        self.assertNotIn('resume', ' '.join(command))

    def test_duplicate_or_nonfinite_json_refused(self):
        path = self.work / 'bad.json'
        for payload in (b'{"x":1,"x":2}', b'{"x":NaN}'):
            path.write_bytes(payload)
            with self.assertRaises(ValueError):
                prep.read_json(path)

    def test_full_evidence_storage_ledger_includes_inputs_and_partial(self):
        protocol = {'limits': {'checkpoint_temporary_bytes': 1024**3, 'PNG_file_bytes': 198656,
            'metadata_bytes': 128 * 1024**2, 'result_payload_bytes': 7 * 1024**3,
            'archive_bytes': 7623566951, 'reserve_free_bytes': 4 * 1024**3,
            'initial_free_bytes': 20 * 1024**3}}
        ledger = prep.storage_ledger(protocol, 8 * 1024**2)
        self.assertEqual(ledger['components_upper_bytes']['all_training_input_PNG_perfile_cap'], 476774400)
        self.assertGreater(ledger['one_replacing_partial_state_scenario_bytes'], ledger['completed_state_scenario_bytes'])
        self.assertLessEqual(ledger['prospective_result_upper_bytes'], 7 * 1024**3)
        self.assertLessEqual(ledger['prospective_archive_upper_bytes'], 7623566951)
        with self.assertRaises(ValueError):
            prep.storage_ledger(protocol, 17 * 1024**2)
        protocol['limits']['result_payload_bytes'] = 6 * 1024**3
        with self.assertRaises(ValueError):
            prep.storage_ledger(protocol, 0)


if __name__ == '__main__':
    unittest.main()
