"""Synthetic stdlib transport fixtures; no real archive import or neural calls."""
import hashlib
from contextlib import contextmanager
import importlib.util
import io
import json
from pathlib import Path
import tarfile
import unittest
from unittest import mock
import uuid

from scripts import audit_codeformer_path_diagnostic_v1_transport as a


def encoded(value):
    return json.dumps(value, sort_keys=True).encode()


def sha(payload):
    return hashlib.sha256(payload).hexdigest()


class TransportFixtureTests(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        cls.scratch = Path(__file__).resolve().parents[1] / 'scratch'
        cls.scratch.mkdir(exist_ok=True)

    @contextmanager
    def directory(self):
        # Python 3.13 TemporaryDirectory uses a Windows mode-0700 ACL that the
        # restricted test token cannot subsequently traverse. Ordinary workspace
        # directories preserve synthetic fixtures without touching ACLs/deletion.
        path = self.scratch / ('path_transport_fixture_' + uuid.uuid4().hex)
        path.mkdir()
        yield path

    def fixture(self, mutate=None, extra=None):
        source = b'raise RuntimeError("returned code must never execute")\n'
        protocol = {'parent_files_sha256': {'parent.py': 'a' * 64},
                    'files_sha256': {'research/diagnostic.py': sha(source)},
                    'export_files': ['research/diagnostic.py']}
        protocol_bytes = encoded(protocol)
        files = {'protocol.json': protocol_bytes, 'research/diagnostic.py': source,
                 a.OUTPUTS + '/measurement.json': encoded({'complete': True})}
        m = {'format': 'codeformer-path-diagnostic-return-v1',
             'protocol_sha256': sha(protocol_bytes),
             'files_sha256': {k: sha(v) for k, v in files.items()},
             'files_bytes': {k: len(v) for k, v in files.items()},
             'parent_files_sha256': protocol['parent_files_sha256'],
             'complete_measurement': True, 'failure_present': False,
             'optimizer_updates': 0, 'model_qualified': False,
             'native_CCTV_qualification': False, 'automatic_training': False}
        if mutate:
            mutate(files, m, protocol)
        manifest_bytes = encoded(m)
        files[a.MANIFEST] = manifest_bytes
        stream = io.BytesIO()
        with tarfile.open(fileobj=stream, mode='w:gz') as tar:
            for name, payload in files.items():
                info = tarfile.TarInfo(name); info.size = len(payload)
                tar.addfile(info, io.BytesIO(payload))
            for info, payload in extra or []:
                tar.addfile(info, io.BytesIO(payload) if payload is not None else None)
        stream.seek(0)
        receipt = {'measurement_complete': True, 'failure_present': False}
        return stream, protocol, protocol_bytes, receipt, sha(manifest_bytes)

    def validate(self, fixture):
        return a.validate_tar(*fixture)

    def test_valid_all_members_before_import(self):
        fixture = self.fixture()
        observed, m, size = self.validate(fixture)
        self.assertEqual(len(observed), 4)
        self.assertEqual(size, sum(x['bytes'] for x in observed.values()))
        self.assertTrue(m['complete_measurement'])

    def test_import_never_executes_returned_code(self):
        fixture = self.fixture(); observed, _, _ = self.validate(fixture)
        fixture[0].seek(0)
        with self.directory() as tmp:
            dest = Path(tmp) / 'new'; log = []
            self.assertEqual(a.import_verified(fixture[0], dest, observed, lambda: None, log.append), 4)
            self.assertEqual(len(log), 4)
            self.assertIn('raise RuntimeError', (dest / 'research/diagnostic.py').read_text())

    def test_destination_must_be_fresh(self):
        fixture = self.fixture(); observed, _, _ = self.validate(fixture)
        fixture[0].seek(0)
        with self.directory() as tmp:
            with self.assertRaisesRegex(ValueError, 'already exists'):
                a.import_verified(fixture[0], Path(tmp), observed, lambda: None, lambda _: None)

    def test_partial_import_preserved_on_failure(self):
        fixture = self.fixture(); observed, _, _ = self.validate(fixture)
        fixture[0].seek(0)
        events = []
        def journal(row):
            events.append(row)
            raise RuntimeError('synthetic interruption')
        with self.directory() as tmp:
            dest = Path(tmp) / 'new'
            with self.assertRaisesRegex(RuntimeError, 'interruption'):
                a.import_verified(fixture[0], dest, observed, lambda: None, journal)
            self.assertTrue((dest / 'protocol.json').is_file())
            self.assertEqual(len(events), 1)

    def test_member_paths(self):
        for name in ('/abs', '../escape', 'x/../escape', './bad', 'x//y', 'C:/drive',
                     'x\\y', 'x:y', 'x/NUL.txt', 'x/trailing.', 'x/trailing ', 'x/a\x01b',
                     'x/CONOUT$', 'x/COM¹.txt', 'x/checkpoint.PTH', 'x/weights.pt'):
            with self.subTest(name=name), self.assertRaises(ValueError):
                a.canonical(name)

    def test_duplicate_member(self):
        info = tarfile.TarInfo('protocol.json'); info.size = 1
        with self.assertRaisesRegex(ValueError, 'Duplicate'):
            self.validate(self.fixture(extra=[(info, b'x')]))

    def test_case_collision(self):
        info = tarfile.TarInfo('Protocol.json'); info.size = 1
        with self.assertRaisesRegex(ValueError, 'colliding'):
            self.validate(self.fixture(extra=[(info, b'x')]))

    def test_symlink_hardlink_directory_and_device_rejected(self):
        for kind in (tarfile.SYMTYPE, tarfile.LNKTYPE, tarfile.DIRTYPE, tarfile.CHRTYPE):
            info = tarfile.TarInfo(a.OUTPUTS + '/bad.json'); info.type = kind
            if kind in (tarfile.SYMTYPE, tarfile.LNKTYPE):
                info.linkname = 'protocol.json'
            with self.subTest(kind=kind), self.assertRaisesRegex(ValueError, 'Nonregular'):
                self.validate(self.fixture(extra=[(info, None)]))

    def test_sparse_extension_rejected(self):
        info = tarfile.TarInfo(a.OUTPUTS + '/bad.json'); info.size = 0
        info.pax_headers = {'GNU.sparse.name': 'sparse'}
        with self.assertRaises(ValueError):
            self.validate(self.fixture(extra=[(info, b'')]))

    def test_namespace_extra_rejected(self):
        info = tarfile.TarInfo('outputs/other/secret.json'); info.size = 0
        with self.assertRaisesRegex(ValueError, 'namespace'):
            self.validate(self.fixture(extra=[(info, b'')]))

    def test_exact_manifest_namespace(self):
        for action in ('hash_missing', 'size_extra'):
            def mutate(files, m, p):
                if action == 'hash_missing':
                    del m['files_sha256']['protocol.json']
                else:
                    m['files_bytes']['extra'] = 0
            with self.subTest(action=action), self.assertRaisesRegex(ValueError, 'namespace'):
                self.validate(self.fixture(mutate))

    def test_hash_and_size_replay(self):
        for key in ('files_sha256', 'files_bytes'):
            def mutate(files, m, p):
                m[key][a.OUTPUTS + '/measurement.json'] = '0' * 64 if key == 'files_sha256' else 9
            with self.subTest(key=key), self.assertRaisesRegex(ValueError, 'hash/size'):
                self.validate(self.fixture(mutate))

    def test_source_companion_binding(self):
        def mutate(files, m, p):
            payload = b'changed source'
            files['research/diagnostic.py'] = payload
            m['files_sha256']['research/diagnostic.py'] = sha(payload)
            m['files_bytes']['research/diagnostic.py'] = len(payload)
        with self.assertRaisesRegex(ValueError, 'Source companion'):
            self.validate(self.fixture(mutate))

    def test_parent_binding(self):
        def mutate(files, m, p):
            m['parent_files_sha256'] = {'changed.py': 'b' * 64}
        with self.assertRaisesRegex(ValueError, 'Parent bindings'):
            self.validate(self.fixture(mutate))

    def test_optimizer_and_qualification_status(self):
        for key, val in (('optimizer_updates', 1), ('model_qualified', True),
                         ('native_CCTV_qualification', True), ('automatic_training', True)):
            def mutate(files, m, p):
                m[key] = val
            with self.subTest(key=key), self.assertRaisesRegex(ValueError, 'status'):
                self.validate(self.fixture(mutate))

    def test_receipt_status_consistency(self):
        def mutate(files, m, p):
            m['complete_measurement'] = False
        with self.assertRaisesRegex(ValueError, 'measurement status'):
            self.validate(self.fixture(mutate))

    def test_caps_are_enforced(self):
        with mock.patch.object(a, 'MAX_MEMBERS', 2), self.assertRaisesRegex(ValueError, 'cap'):
            self.validate(self.fixture())
        with mock.patch.object(a, 'RESULT_CAP', 100), self.assertRaisesRegex(ValueError, 'size|cap'):
            self.validate(self.fixture())

    def test_json_duplicate_and_nonfinite(self):
        for data in (b'{"x":1,"x":2}', b'{"x":NaN}', b'{"x":Infinity}'):
            with self.subTest(data=data), self.assertRaises(ValueError):
                a.decode_json(data)

    def test_import_time_has_no_io_or_neural_imports(self):
        source = Path(a.__file__).read_bytes()
        spec = importlib.util.spec_from_file_location('synthetic_transport_import', a.__file__)
        module = importlib.util.module_from_spec(spec)
        with mock.patch('builtins.open', side_effect=AssertionError('unexpected I/O')):
            exec(compile(source, a.__file__, 'exec'), module.__dict__)
        self.assertNotIn('torch', module.__dict__)

    def test_import_rehash_catches_changed_payload(self):
        fixture = self.fixture(); observed, _, _ = self.validate(fixture)
        observed['protocol.json']['sha256'] = '0' * 64
        fixture[0].seek(0)
        with self.directory() as tmp:
            dest = Path(tmp) / 'new'
            with self.assertRaisesRegex(ValueError, 'Import bytes'):
                a.import_verified(fixture[0], dest, observed, lambda: None, lambda _: None)
            self.assertTrue((dest / 'protocol.json').is_file())

    def test_manifest_pin(self):
        fixture = list(self.fixture()); fixture[-1] = '0' * 64
        with self.assertRaisesRegex(ValueError, 'Manifest hash'):
            self.validate(fixture)

    def test_export_receipt_and_sidecar_bindings(self):
        with self.directory() as tmp:
            archive = tmp / 'artifact.tar.gz'; archive.write_bytes(b'return archive')
            sidecar = tmp / 'artifact.tar.gz.sha256'
            receipt = tmp / 'export.json'; digest = sha(archive.read_bytes())
            r = {'complete': True, 'archive_sha256': digest, 'bytes': archive.stat().st_size,
                 'manifest_member': a.MANIFEST, 'manifest_sha256': 'e' * 64,
                 'protocol_sha256': a.PROTOCOL_SHA, 'optimizer_updates': 0,
                 'model_qualified': False, 'training_success_not_implied': True}
            sidecar.write_bytes((digest + '  artifact.tar.gz\n').encode())
            receipt.write_bytes(encoded(r))
            self.assertEqual(a.check_receipts(archive, sidecar, receipt, digest,
                                             archive.stat().st_size, 'e' * 64), r)
            sidecar.write_bytes((digest + '  artifact.tar.gz\r\n').encode())
            with self.assertRaisesRegex(ValueError, 'sidecar'):
                a.check_receipts(archive, sidecar, receipt, digest, archive.stat().st_size, 'e' * 64)
            sidecar.write_bytes((digest + '  artifact.tar.gz\n').encode())
            for key, value in (('archive_sha256', 'f' * 64), ('bytes', 4),
                               ('protocol_sha256', 'f' * 64), ('manifest_sha256', 'f' * 64),
                               ('optimizer_updates', 2), ('optimizer_updates', False),
                               ('model_qualified', True)):
                changed = dict(r); changed[key] = value; receipt.write_bytes(encoded(changed))
                with self.subTest(key=key, value=value), self.assertRaisesRegex(ValueError, 'binding'):
                    a.check_receipts(archive, sidecar, receipt, digest, archive.stat().st_size, 'e' * 64)

    def test_pinned_local_packet_and_companions(self):
        with self.directory() as tmp:
            payload = b'source retained, never executed'
            (tmp / 'source.py').write_bytes(payload)
            p = {'format': 'codeformer-path-diagnostic-finite-manual-v1', 'outputs': a.OUTPUTS,
                 'export_prefix': a.PREFIX, 'optimizer_updates': 0, 'no_automatic_training': True,
                 'limits': {'result_bytes': a.RESULT_CAP, 'archive_bytes': a.ARCHIVE_CAP},
                 'files_sha256': {'source.py': sha(payload)}, 'export_files': ['source.py']}
            protocol_bytes = encoded(p); (tmp / 'protocol.json').write_bytes(protocol_bytes)
            self.assertEqual(a.packet_contract(tmp, sha(protocol_bytes))[0], p)
            with self.assertRaisesRegex(ValueError, 'Pinned protocol'):
                a.packet_contract(tmp, '0' * 64)
            (tmp / 'source.py').write_bytes(b'modified')
            with self.assertRaisesRegex(ValueError, 'Packet asset'):
                a.packet_contract(tmp, sha(protocol_bytes))


if __name__ == '__main__':
    unittest.main()
