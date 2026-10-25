"""Synthetic framing/namespace tests; no real archive or numerical imports."""
import gzip
import hashlib
import importlib.util
import io
import json
from pathlib import Path
import tarfile
import tempfile
import unittest
from unittest import mock

ROOT = Path(__file__).resolve().parents[1]
SPEC = importlib.util.spec_from_file_location('matched_transport_unit', ROOT / 'scripts/audit_matched_feature_return_transport_v1.py')
M = importlib.util.module_from_spec(SPEC); SPEC.loader.exec_module(M)


def wire(rows):
    stream = io.BytesIO()
    with tarfile.open(fileobj=stream, mode='w', format=tarfile.PAX_FORMAT) as tar:
        for name, payload, kind in rows:
            info = tarfile.TarInfo(M.STAGE + '/' + name)
            info.size = len(payload)
            info.type = kind
            if kind in (tarfile.SYMTYPE, tarfile.LNKTYPE): info.linkname = 'escape'
            tar.addfile(info, io.BytesIO(payload))
    return stream.getvalue()


class Transport(unittest.TestCase):
    def setUp(self):
        self.protocol_raw = b'{"files":{}}\n'
        self.static = b'not executable\n'
        self.protocol = {'files': {'a.txt': {'bytes': len(self.static), 'sha256': M.sha(self.static)}}}
        self.rows = [('a.txt', self.static, tarfile.REGTYPE),
                     ('protocol.json', self.protocol_raw, tarfile.REGTYPE),
                     (M.RUN + 'status.json', b'{"complete":false}\n', tarfile.REGTYPE)]
        self.sha_patch = mock.patch.object(M, 'PROTOCOL_SHA', M.sha(self.protocol_raw)); self.sha_patch.start()
        self.addCleanup(self.sha_patch.stop)

    def scan(self, rows=None, *, raw=None, root=None):
        data = gzip.compress(wire(self.rows if rows is None else rows)) if raw is None else raw
        return M.inspect_stream(io.BytesIO(data), self.protocol, retained_root=root)

    def test_full_synthetic_members_payload_and_compressed_hash(self):
        raw = gzip.compress(wire(self.rows)); index, metadata, framing = self.scan(raw=raw)
        self.assertEqual(len(index), 3)
        self.assertEqual(metadata[M.RUN + 'status.json'], {'complete': False})
        self.assertEqual(framing['compressed_sha256'], hashlib.sha256(raw).hexdigest())
        self.assertEqual(framing['payload_bytes'], sum(len(r[1]) for r in self.rows))
        self.assertTrue(framing['single_complete_gzip_CRC_ISIZE_verified'])

    def test_namespace_escape_absolute_backslash_colon_and_alias_refused(self):
        for path in ('../escape', '/absolute', 'x/../y', 'x//y', './x', 'x\\y', 'x:y', 'x\ny'):
            with self.subTest(path=path), self.assertRaises(ValueError): self.scan(self.rows + [(path, b'x', tarfile.REGTYPE)])

    def test_duplicate_and_windows_case_alias_refused(self):
        for name in ('a.txt', 'A.TXT'):
            with self.subTest(name=name), self.assertRaisesRegex(ValueError, 'Duplicate'):
                self.scan(self.rows + [(name, b'x', tarfile.REGTYPE)])

    def test_symlink_hardlink_directory_and_fifo_refused(self):
        for kind in (tarfile.SYMTYPE, tarfile.LNKTYPE, tarfile.DIRTYPE, tarfile.FIFOTYPE):
            with self.subTest(kind=kind), self.assertRaises(ValueError): self.scan(self.rows + [(M.RUN + 'bad', b'', kind)])

    def test_returned_executable_in_run_namespace_never_retained(self):
        for suffix in ('.py', '.sh', '.exe', '.ps1', '.cmd'):
            with self.subTest(suffix=suffix), self.assertRaisesRegex(ValueError, 'Executable'):
                self.scan(self.rows + [(M.RUN + 'bad' + suffix, b'ignored', tarfile.REGTYPE)])

    def test_unplanned_namespace_or_weight_return_refused(self):
        for name in ('unexpected.json', 'weights/model.pth'):
            with self.subTest(name=name), self.assertRaisesRegex(ValueError, 'namespace'):
                self.scan(self.rows + [(name, b'x', tarfile.REGTYPE)])

    def test_changed_or_missing_static_source_refused(self):
        changed = list(self.rows); changed[0] = ('a.txt', b'wrong', tarfile.REGTYPE)
        with self.assertRaisesRegex(ValueError, 'sealed static'): self.scan(changed)
        with self.assertRaisesRegex(ValueError, 'static payload missing'): self.scan(self.rows[1:])

    def test_protocol_digest_mismatch_refused(self):
        rows = list(self.rows); rows[1] = ('protocol.json', b'{}\n', tarfile.REGTYPE)
        with self.assertRaisesRegex(ValueError, 'protocol differs'): self.scan(rows)

    def test_truncated_gzip_bad_crc_or_concatenated_gzip_refused(self):
        good = gzip.compress(wire(self.rows)); bad_crc = bytearray(good); bad_crc[-8] ^= 1
        for value in (good[:-1], bytes(bad_crc), good + good, good + b'garbage'):
            with self.subTest(size=len(value)), self.assertRaises((ValueError, EOFError, tarfile.ReadError, __import__('zlib').error)):
                self.scan(raw=value)

    def test_hidden_tar_payload_after_end_is_refused(self):
        first = wire(self.rows)
        for value in (first + wire([('hidden', b'x', tarfile.REGTYPE)]), first + b'nonzero'):
            with self.subTest(size=len(value)), self.assertRaises(ValueError): self.scan(raw=gzip.compress(value))

    def test_partial_or_nonfinite_metadata_refused(self):
        for value in (b'{"a":NaN}', b'{"a":1e999}', b'{"a":1,"a":2}'):
            rows = list(self.rows); rows[2] = (M.RUN + 'status.json', value, tarfile.REGTYPE)
            with self.subTest(value=value), self.assertRaises(ValueError): self.scan(rows)
        with self.assertRaisesRegex(ValueError, 'Partial JSONL'):
            self.scan(self.rows + [(M.RUN + 'events.jsonl', b'{}', tarfile.REGTYPE)])

    def test_caps_enforced_for_members_total_and_expansion(self):
        for attribute, limit in (('MAX_MEMBER', 2), ('MAX_FILES', 1), ('MAX_BYTES', 1), ('MAX_EXPANDED', 1)):
            with self.subTest(attribute=attribute), mock.patch.object(M, attribute, limit), self.assertRaises(ValueError): self.scan()

    def test_data_only_retention_uses_exclusive_fresh_root(self):
        (ROOT / 'scratch').mkdir(exist_ok=True)
        with tempfile.TemporaryDirectory(prefix='matched_return_unit_', dir=ROOT / 'scratch') as temp:
            path = Path(temp) / 'retained'
            index, _, _ = self.scan(root=path)
            self.assertFalse((path / 'a.txt').exists())
            self.assertEqual((path / 'protocol.json').read_bytes(), self.protocol_raw)
            self.assertTrue((path / M.RUN / 'status.json').is_file())
            with self.assertRaisesRegex(ValueError, 'immutable'): self.scan(root=path)

    def test_direct_runtime_bindings_reject_wrong_payload_hash(self):
        name = M.RUN + 'raw.npy'; value = b'pretend-array'
        binding = {'path': name, 'bytes': len(value), 'sha256': M.sha(value)}
        index = {name: {k: binding[k] for k in ('bytes', 'sha256')}}
        self.assertEqual(M.direct_joins({M.RUN + 'record.json': {'output': binding}}, index), 1)
        binding['sha256'] = '0' * 64
        with self.assertRaisesRegex(ValueError, 'binding differs'): M.direct_joins({M.RUN + 'record.json': binding}, index)


if __name__ == '__main__': unittest.main()
