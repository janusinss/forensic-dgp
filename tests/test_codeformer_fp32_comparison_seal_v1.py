"""Synthetic archive-boundary tests; no models, gradients or VM calls."""
import hashlib
import importlib.util
import io
from pathlib import Path
import tarfile
import unittest
import uuid


ROOT = Path(__file__).resolve().parents[1]
spec = importlib.util.spec_from_file_location('fp32_sealer', ROOT / 'scripts/seal_codeformer_fp32_comparison_v1.py')
sealer = importlib.util.module_from_spec(spec)
spec.loader.exec_module(sealer)


class ArchiveTests(unittest.TestCase):
    def setUp(self):
        self.folder = ROOT / 'scratch' / ('fp32-seal-fixture-' + uuid.uuid4().hex)
        self.folder.mkdir(parents=True)
        self.packet = self.folder / 'packet'
        self.packet.mkdir()
        (self.packet / 'protocol.json').write_bytes(b'{}\n')
        self.digest = hashlib.sha256(b'{}\n').hexdigest()

    def bundle(self, rows):
        path = self.folder / ('fixture-' + uuid.uuid4().hex + '.tar.gz')
        with tarfile.open(path, 'w:gz') as archive:
            for name, content, kind in rows:
                info = tarfile.TarInfo(name)
                info.type = kind
                info.size = len(content) if kind == tarfile.REGTYPE else 0
                if kind == tarfile.SYMTYPE:
                    info.linkname = '/etc/passwd'
                archive.addfile(info, io.BytesIO(content) if info.size else None)
        return path

    def check(self, rows, expected=None):
        return sealer.archive_readback(self.bundle(rows), self.packet,
                                       {'protocol.json': self.digest} if expected is None else expected)

    def test_exact_regular_payload(self):
        result = self.check([(sealer.PACKAGE + '/protocol.json', b'{}\n', tarfile.REGTYPE)])
        self.assertEqual(result['members'], 1)
        self.assertEqual(result['payload_bytes'], 3)

    def test_wrong_payload_is_rejected(self):
        with self.assertRaisesRegex(ValueError, 'hash'):
            self.check([(sealer.PACKAGE + '/protocol.json', b'[]\n', tarfile.REGTYPE)])

    def test_duplicate_member_is_rejected(self):
        row = (sealer.PACKAGE + '/protocol.json', b'{}\n', tarfile.REGTYPE)
        with self.assertRaisesRegex(ValueError, 'duplicate'):
            self.check([row, row])

    def test_traversal_is_rejected(self):
        with self.assertRaises(ValueError):
            self.check([(sealer.PACKAGE + '/../protocol.json', b'{}\n', tarfile.REGTYPE)])

    def test_symlink_is_rejected(self):
        with self.assertRaisesRegex(ValueError, 'regular'):
            self.check([(sealer.PACKAGE + '/protocol.json', b'', tarfile.SYMTYPE)])

    def test_wrong_root_is_rejected(self):
        with self.assertRaisesRegex(ValueError, 'root'):
            self.check([('old_v2/protocol.json', b'{}\n', tarfile.REGTYPE)])

    def test_missing_member_is_rejected(self):
        with self.assertRaisesRegex(ValueError, 'omits'):
            self.check([])

    def test_windows_device_and_noncanonical_paths_rejected(self):
        for path in ('data/CON.txt', 'data/CONIN$', 'data/CONOUT$.txt', 'data/COM¹', 'data/LPT².txt',
                     'data/a\\b', 'data/x:stream', 'data/file.', '/absolute',
                     'data//double', 'data/../escape', 'data/./dot'):
            with self.subTest(path=path), self.assertRaises(ValueError):
                sealer.canonical(path)


if __name__ == '__main__':
    unittest.main()
