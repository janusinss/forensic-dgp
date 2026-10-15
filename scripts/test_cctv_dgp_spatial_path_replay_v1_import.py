"""Archive import boundaries; tiny retained fixtures, zero neural calls."""
import hashlib
import io
import json
from pathlib import Path
import tarfile
import unittest
import uuid

from import_cctv_dgp_spatial_path_replay_v1_return import (
    namespace, read_manifest, check_payload, extract_checked, safe_name, STAGE,
)

BASE = Path(__file__).resolve().parent.parent / 'outputs/cctv_dgp_spatial_path_replay_v1_design/import_fixtures'
PIN = '1' * 64


def fixture(extra=None, overrides=None):
    directory = BASE / uuid.uuid4().hex
    directory.mkdir(parents=True)
    files = {'protocol.json': b'protocol', 'scripts/worker.py': b'never executed', STAGE + 'failure.json': b'{"fixture_only":true}'}
    if extra:
        files.update(extra)
    packet = {name: hashlib.sha256(value).hexdigest() for name, value in files.items() if not name.startswith(STAGE)}
    manifest = dict(protocol_sha256=PIN, model_quality_not_implied=True, new_optimizer_updates=0,
        files={name: dict(bytes=len(value), sha256=hashlib.sha256(value).hexdigest()) for name, value in files.items()})
    if overrides:
        overrides(manifest)
    files['result_manifest.json'] = json.dumps(manifest).encode()
    path = directory / 'fixture.tar.gz'
    with tarfile.open(path, 'w:gz') as tar:
        for name, value in files.items():
            row = tarfile.TarInfo(name); row.size = len(value); tar.addfile(row, io.BytesIO(value))
    return directory, path, packet


class Boundaries(unittest.TestCase):
    def test_valid_import_preserves_exact_bytes_without_executing_code(self):
        directory, path, packet = fixture()
        with tarfile.open(path, 'r:gz') as tar:
            names = namespace(tar, 65536)
            manifest = read_manifest(tar, names, PIN, packet, 65536)
            check_payload(tar, names, manifest)
            extract_checked(tar, names, manifest, directory / 'return')
        self.assertEqual((directory / 'return/scripts/worker.py').read_bytes(), b'never executed')

    def test_windows_and_posix_unsafe_names_rejected(self):
        for name in ['../escape', '/absolute', 'C:/file', 'a\\b', './a', 'a//b', 'CON.txt', 'a/NUL',
                     'a.', 'a ', 'a\x00b', 'a:b', 'a?b', 'a/../b']:
            with self.subTest(name=name), self.assertRaises(AssertionError):
                safe_name(name)

    def test_case_and_parent_file_collisions_rejected(self):
        for extra in [{'Protocol.json': b'x'}, {'scripts': b'x'}]:
            _, path, _ = fixture(extra)
            with tarfile.open(path, 'r:gz') as tar, self.assertRaises(AssertionError):
                namespace(tar, 65536)

    def test_links_rejected(self):
        directory = BASE / uuid.uuid4().hex; directory.mkdir(parents=True)
        for kind in [tarfile.SYMTYPE, tarfile.LNKTYPE]:
            path = directory / (str(kind) + '.tar.gz')
            with tarfile.open(path, 'w:gz') as tar:
                row = tarfile.TarInfo('link'); row.type = kind; row.linkname = 'target'; tar.addfile(row)
            with tarfile.open(path, 'r:gz') as tar, self.assertRaises(AssertionError):
                namespace(tar, 65536)

    def test_uncompressed_limit_rejected(self):
        _, path, _ = fixture({STAGE + 'oversized.bin': b'x' * 128})
        with tarfile.open(path, 'r:gz') as tar, self.assertRaises(AssertionError):
            namespace(tar, 64)

    def test_wrong_protocol_or_mutated_packet_rejected(self):
        for change in [lambda m: m.update(protocol_sha256='2' * 64),
                       lambda m: m['files']['protocol.json'].update(sha256='3' * 64)]:
            _, path, packet = fixture(overrides=change)
            with tarfile.open(path, 'r:gz') as tar, self.assertRaises(AssertionError):
                read_manifest(tar, namespace(tar, 65536), PIN, packet, 65536)

    def test_missing_binding_rejected_before_extract(self):
        directory, path, packet = fixture(overrides=lambda m: m['files'].pop(STAGE + 'failure.json'))
        with tarfile.open(path, 'r:gz') as tar, self.assertRaises(AssertionError):
            read_manifest(tar, namespace(tar, 65536), PIN, packet, 65536)
        self.assertFalse((directory / 'return').exists())

    def test_corrupted_output_rejected_before_extract(self):
        directory, path, packet = fixture(overrides=lambda m: m['files'][STAGE + 'failure.json'].update(sha256='4' * 64))
        with tarfile.open(path, 'r:gz') as tar:
            names = namespace(tar, 65536); manifest = read_manifest(tar, names, PIN, packet, 65536)
            with self.assertRaises(AssertionError):
                check_payload(tar, names, manifest)
        self.assertFalse((directory / 'return').exists())

    def test_existing_import_not_overwritten(self):
        directory, path, packet = fixture(); target = directory / 'return'; target.mkdir()
        (target / 'retained').write_bytes(b'keep')
        with tarfile.open(path, 'r:gz') as tar:
            names = namespace(tar, 65536); manifest = read_manifest(tar, names, PIN, packet, 65536)
            with self.assertRaises(AssertionError):
                extract_checked(tar, names, manifest, target)
        self.assertEqual((target / 'retained').read_bytes(), b'keep')


if __name__ == '__main__':
    unittest.main(verbosity=2)
