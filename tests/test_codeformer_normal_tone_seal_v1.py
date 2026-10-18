"""CPU-only hostile source-archive fixtures; never invoke a worker or model."""
import gzip
import hashlib
import importlib.util
import io
from pathlib import Path
import tarfile
import tempfile
import unittest

ROOT = Path(__file__).resolve().parents[1]
SPEC = importlib.util.spec_from_file_location('normal_tone_sealer_fixture',
    ROOT / 'scripts/seal_codeformer_normal_tone_v1.py')
SEAL = importlib.util.module_from_spec(SPEC)
SPEC.loader.exec_module(SEAL)


class SourceArchiveBoundaries(unittest.TestCase):
    def setUp(self):
        self.temporary = tempfile.TemporaryDirectory(dir=ROOT / 'scratch')
        self.addCleanup(self.temporary.cleanup)
        self.path = Path(self.temporary.name) / 'packet.tar.gz'
        self.top = 'cctv_codeformer_normal_tone_v1_vm'
        self.payload = b'print("source only")\n'
        self.hashes = {'worker.py': hashlib.sha256(self.payload).hexdigest()}
        self.sizes = {'worker.py': len(self.payload)}

    def archive(self, entries=None):
        stream = io.BytesIO()
        with tarfile.open(fileobj=stream, mode='w', format=tarfile.USTAR_FORMAT) as packed:
            for name, payload, kind in entries or [('worker.py', self.payload, tarfile.REGTYPE)]:
                info = tarfile.TarInfo(self.top + '/' + name)
                info.type = kind
                info.size = len(payload) if kind == tarfile.REGTYPE else 0
                if kind == tarfile.SYMTYPE:
                    info.linkname = '/tmp/outside'
                packed.addfile(info, io.BytesIO(payload) if info.isfile() else None)
        self.path.write_bytes(gzip.compress(stream.getvalue(), mtime=0))

    def verify(self):
        return SEAL.verify_archive(self.path, self.top, self.hashes, self.sizes)

    def test_exact_regular_source_roundtrip(self):
        self.archive()
        result = self.verify()
        self.assertEqual(result['members'], 1)
        self.assertTrue(result['single_gzip_stream_CRC_checked'])
        self.assertFalse(result['extraction_or_execution'])

    def test_payload_change_refused(self):
        self.archive([('worker.py', b'x' * len(self.payload), tarfile.REGTYPE)])
        with self.assertRaises(ValueError):
            self.verify()

    def test_duplicate_member_refused(self):
        self.archive([('worker.py', self.payload, tarfile.REGTYPE)] * 2)
        with self.assertRaises(ValueError):
            self.verify()

    def test_link_refused(self):
        self.archive([('worker.py', b'', tarfile.SYMTYPE)])
        with self.assertRaises(ValueError):
            self.verify()

    def test_unknown_member_refused(self):
        self.archive([('other.py', self.payload, tarfile.REGTYPE)])
        with self.assertRaises(ValueError):
            self.verify()

    def test_path_traversal_refused(self):
        self.archive([('../worker.py', self.payload, tarfile.REGTYPE)])
        with self.assertRaises(ValueError):
            self.verify()

    def test_extra_gzip_stream_refused(self):
        self.archive()
        with self.path.open('ab') as stream:
            stream.write(gzip.compress(b'second stream', mtime=0))
        with self.assertRaises(ValueError):
            self.verify()

    def test_truncated_gzip_refused(self):
        self.archive()
        self.path.write_bytes(self.path.read_bytes()[:-5])
        with self.assertRaises(ValueError):
            self.verify()

    def test_nonzero_inside_tar_end_padding_refused(self):
        self.archive()
        raw = bytearray(gzip.decompress(self.path.read_bytes()))
        raw[-1] = 65
        self.path.write_bytes(gzip.compress(bytes(raw), mtime=0))
        with self.assertRaises(ValueError):
            self.verify()

    def test_valid_but_noncanonical_header_metadata_refused(self):
        self.archive()
        raw = bytearray(gzip.decompress(self.path.read_bytes()))
        header = tarfile.TarInfo.frombuf(bytes(raw[:512]), 'utf-8', 'strict')
        header.uid = 1000
        raw[:512] = header.tobuf(format=tarfile.USTAR_FORMAT)
        self.path.write_bytes(gzip.compress(bytes(raw), mtime=0))
        with self.assertRaises(ValueError):
            self.verify()


if __name__ == '__main__':
    unittest.main()
