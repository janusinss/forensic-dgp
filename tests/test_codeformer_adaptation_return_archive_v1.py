"""Small adversarial return-transport fixtures; no models or neural imports."""
import hashlib
import importlib
import io
import json
from pathlib import Path
import shutil
import tarfile
import unittest
from unittest.mock import patch
import uuid


ROOT = Path(__file__).resolve().parents[1]
AUDITOR = importlib.import_module('research.codeformer_adaptation_v1.return_archive')


def digest(payload):
    return hashlib.sha256(payload).hexdigest()


def encoded(value):
    return (json.dumps(value, sort_keys=True) + '\n').encode('utf-8')


class ReturnArchiveFixtures(unittest.TestCase):
    def setUp(self):
        self.work = ROOT / 'scratch' / ('codeformer_return_static_' + uuid.uuid4().hex)
        self.work.mkdir()
        self.packet = self.work / 'packet'
        self.packet.mkdir()
        self.destination = self.work / 'imported'
        self.receipt = self.work / 'receipt.json'
        self.archive = self.work / 'returned.tar.gz'
        self.sidecar = self.work / 'returned.tar.gz.sha256'
        self.export_path = self.work / 'export.json'
        self.manifest_name = 'outputs/adaptation_v1/export_manifest_fixture.json'
        self.protocol = {'outputs': 'outputs/adaptation_v1',
                         'export_files': ['research/source.py', 'LICENSE'],
                         'files_sha256': {'research/source.py': digest(b'# retained source\n'),
                                          'LICENSE': digest(b'fixture terms\n'),
                                          'data/input.bin': digest(b'retained input')},
                         'assets': {}}
        for name, payload in {'research/source.py': b'# retained source\n',
                              'LICENSE': b'fixture terms\n',
                              'data/input.bin': b'retained input'}.items():
            target = self.packet / name
            target.parent.mkdir(parents=True, exist_ok=True)
            target.write_bytes(payload)
        self.protocol_bytes = encoded(self.protocol)
        (self.packet / 'protocol.json').write_bytes(self.protocol_bytes)
        self.protocol_sha = digest(self.protocol_bytes)
        self.files = {'protocol.json': self.protocol_bytes,
                      'research/source.py': b'# retained source\n', 'LICENSE': b'fixture terms\n',
                      'outputs/adaptation_v1/failure.json': b'{"complete": false}\n'}

    def tearDown(self):
        exact = self.work.resolve(strict=True)
        assert exact.parent == (ROOT / 'scratch').resolve(strict=True)
        assert exact.name.startswith('codeformer_return_static_') and not self.work.is_symlink()
        shutil.rmtree(exact)

    def build(self, *, payloads=None, listed=None, extras=None, manifest_change=None, types=None):
        payloads = dict(self.files if payloads is None else payloads)
        listed = {name: digest(value) for name, value in payloads.items()} if listed is None else dict(listed)
        all_inputs = dict(self.protocol['files_sha256'])
        manifest = {'complete': True, 'protocol_sha256': self.protocol_sha, 'model_qualified': False,
                    'files_sha256': listed, 'failure_present': True, 'run_results_present': False,
                    'retained_not_exported_sha256': {name: value for name, value in all_inputs.items()
                                                    if name not in listed}}
        if manifest_change:
            manifest.update(manifest_change)
        manifest_bytes = encoded(manifest)
        members = dict(payloads)
        members[self.manifest_name] = manifest_bytes
        members.update(extras or {})
        with tarfile.open(self.archive, 'w:gz') as target:
            for name, payload in members.items():
                member = tarfile.TarInfo(name)
                member.size = len(payload)
                if types and name in types:
                    member.type = types[name]
                    member.linkname = '../../outside'
                target.addfile(member, io.BytesIO(payload))
        archive_sha = digest(self.archive.read_bytes())
        self.sidecar.write_text(archive_sha + '  ' + self.archive.name + '\n', encoding='ascii')
        self.export_path.write_bytes(encoded({'complete': True, 'model_qualified': False,
            'protocol_sha256': self.protocol_sha, 'archive_sha256': archive_sha,
            'bytes': self.archive.stat().st_size, 'manifest_member': self.manifest_name,
            'manifest_sha256': digest(manifest_bytes), 'failure_present': True,
            'run_results_present': False, 'retained_not_exported_files': len(manifest['retained_not_exported_sha256'])}))

    def inspect(self):
        with patch.object(AUDITOR, 'PROTOCOL_SHA256', self.protocol_sha), \
                patch.object(AUDITOR.shutil, 'disk_usage', return_value=shutil._ntuple_diskusage(64 << 30, 0, 64 << 30)):
            return AUDITOR.inspect_and_import(self.archive, self.sidecar, self.export_path,
                                              self.packet, self.destination, self.receipt)

    def test_valid_exact_import_and_zero_neural_execution(self):
        self.build()
        receipt = self.inspect()
        self.assertTrue(receipt['complete'])
        self.assertEqual(receipt['members'], len(self.files) + 1)
        self.assertEqual(receipt['retained_inputs_verified'], 1)
        self.assertEqual(receipt['gradient_queries'], 0)
        self.assertEqual(receipt['optimizer_updates'], 0)
        self.assertEqual(receipt['VM_calls'], 0)
        self.assertFalse(receipt['model_qualified'])
        for name, payload in self.files.items():
            self.assertEqual((self.destination / name).read_bytes(), payload)
        with self.assertRaises(ValueError):
            self.inspect()

    def test_path_traversal_and_noncanonical_names_rejected(self):
        for name in ('../escape', '/absolute', 'a/../b', 'a//b', 'a/./b', 'a\\b', 'C:/drive', ''):
            with self.subTest(name=name), self.assertRaises(ValueError):
                AUDITOR.canonical_member(name)

    def test_windows_device_and_invalid_names_rejected(self):
        for name in ('.', 'CON', 'a/NUL.txt', 'a/COM1.log', 'LPT9', 'a/file.', 'a/file ',
                     'a/q?.png', 'a/s*.png', 'a/less<.png', 'a/greater>.png', 'a/pipe|.png',
                     'a/quote\".png', 'a/COM\u00b9.txt', 'a/LPT\u00b2.txt'):
            with self.subTest(name=name), self.assertRaises(ValueError):
                AUDITOR.canonical_member(name)

    def test_case_collision_rejected_before_import(self):
        files = dict(self.files)
        files['outputs/adaptation_v1/Case.bin'] = b'a'
        files['outputs/adaptation_v1/case.bin'] = b'b'
        self.build(payloads=files)
        with self.assertRaises(ValueError):
            self.inspect()
        self.assertFalse(self.destination.exists())

    def test_link_and_device_members_rejected(self):
        for kind in (tarfile.SYMTYPE, tarfile.LNKTYPE, tarfile.CHRTYPE):
            with self.subTest(kind=kind):
                self.build(types={'outputs/adaptation_v1/failure.json': kind})
                with self.assertRaises(ValueError):
                    self.inspect()
                self.assertFalse(self.destination.exists())

    def test_unlisted_payload_rejected(self):
        self.build(extras={'outputs/adaptation_v1/unlisted.bin': b'x'})
        with self.assertRaises(ValueError):
            self.inspect()
        self.assertFalse(self.destination.exists())

    def test_payload_tamper_retained_as_failed_evidence(self):
        expected = {name: digest(value) for name, value in self.files.items()}
        files = dict(self.files)
        files['outputs/adaptation_v1/failure.json'] = b'{"tampered": true}\n'
        self.build(payloads=files, listed=expected)
        with self.assertRaises(ValueError):
            self.inspect()
        self.assertTrue(self.archive.is_file())
        self.assertTrue(self.destination.is_dir())
        self.assertFalse(self.receipt.exists())
        self.assertTrue(self.receipt.with_name('receipt_failure.json').is_file())

    def test_mutated_frozen_source_rejected_before_import(self):
        files = dict(self.files)
        files['research/source.py'] = b'print("mutated")\n'
        self.build(payloads=files)
        with self.assertRaises(ValueError):
            self.inspect()
        self.assertFalse(self.destination.exists())

    def test_mutated_local_protocol_and_retained_input_rejected(self):
        self.build()
        original = (self.packet / 'protocol.json').read_bytes()
        (self.packet / 'protocol.json').write_bytes(original + b' ')
        with self.assertRaises(ValueError):
            self.inspect()
        (self.packet / 'protocol.json').write_bytes(original)
        (self.packet / 'data/input.bin').write_bytes(b'changed input')
        with self.assertRaises(ValueError):
            self.inspect()
        self.assertFalse(self.destination.exists())

    def test_omitted_frozen_export_companion_rejected(self):
        files = dict(self.files)
        del files['research/source.py']
        self.build(payloads=files)
        with self.assertRaises(ValueError):
            self.inspect()

    def test_receipt_manifest_failure_disagreement_rejected(self):
        self.build(manifest_change={'failure_present': False})
        with self.assertRaises(ValueError):
            self.inspect()

    def test_checksum_and_json_duplicate_rejected(self):
        self.build()
        self.sidecar.write_text('0' * 64 + '  ' + self.archive.name + '\n', encoding='ascii')
        with self.assertRaises(ValueError):
            self.inspect()
        with self.assertRaises(ValueError):
            AUDITOR._json(b'{"complete":true,"complete":false}')
        with self.assertRaises(ValueError):
            AUDITOR._json(b'{"number":NaN}')


if __name__ == '__main__':
    unittest.main()
