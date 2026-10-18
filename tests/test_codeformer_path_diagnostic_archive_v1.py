"""Synthetic small archive fixtures only; no real weights or VM execution."""
from io import BytesIO
from pathlib import Path
import tarfile
import unittest
from unittest.mock import patch

from scripts import seal_codeformer_path_diagnostic_v1 as seal
from tests.test_codeformer_path_diagnostic_worker_v1 import owned_fixture_directory


class ArchiveReadback(unittest.TestCase):
    def fixture(self, directory, members):
        packet = directory / seal.c.PACKAGE_DIR
        packet.mkdir()
        (packet / 'source.txt').write_bytes(b'owned fixture')
        (packet / 'protocol.json').write_bytes(b'{}')
        p = {'files_sha256': {'source.txt': seal.c.sha256(packet / 'source.txt')}}
        pin = seal.c.sha256(packet / 'protocol.json')
        archive = directory / 'owned.tar.gz'
        with tarfile.open(archive, 'w:gz') as bundle:
            for name, payload, typeflag in members:
                info = tarfile.TarInfo(name)
                info.type = typeflag
                info.size = len(payload)
                bundle.addfile(info, BytesIO(payload))
        return packet, p, pin, archive

    def test_exact_member_namespace_and_hash(self):
        with owned_fixture_directory() as directory:
            members = [(seal.c.PACKAGE_DIR + '/source.txt', b'owned fixture', tarfile.REGTYPE),
                       (seal.c.PACKAGE_DIR + '/protocol.json', b'{}', tarfile.REGTYPE)]
            packet, p, pin, archive = self.fixture(directory, members)
            with patch.object(seal.c, 'verify_transfer', return_value=p):
                result = seal.archive_readback(archive, packet, pin)
            self.assertTrue(result['exact_namespace_and_every_member_hash'])
            self.assertEqual(result['members'], 2)

    def test_corrupt_duplicate_noncanonical_and_nonregular_refused(self):
        variants = [
            [(seal.c.PACKAGE_DIR + '/source.txt', b'wrong fixture', tarfile.REGTYPE)],
            [(seal.c.PACKAGE_DIR + '/source.txt', b'owned fixture', tarfile.REGTYPE)] * 2,
            [(seal.c.PACKAGE_DIR + '//source.txt', b'owned fixture', tarfile.REGTYPE)],
            [(seal.c.PACKAGE_DIR + '/source.txt', b'', tarfile.SYMTYPE)],
            [(seal.c.PACKAGE_DIR + '/../source.txt', b'owned fixture', tarfile.REGTYPE)],
        ]
        for members in variants:
            with self.subTest(members=members), owned_fixture_directory() as directory:
                packet, p, pin, archive = self.fixture(directory, members)
                with patch.object(seal.c, 'verify_transfer', return_value=p):
                    with self.assertRaises(ValueError):
                        seal.archive_readback(archive, packet, pin)


if __name__ == '__main__':
    unittest.main()
