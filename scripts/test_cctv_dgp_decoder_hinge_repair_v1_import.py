"""Tiny retained archive fixtures; no returned Python is imported or executed."""
import hashlib
import io
import json
from pathlib import Path
import tarfile
import unittest
import uuid

from import_cctv_dgp_decoder_hinge_repair_v1_return import (
    namespace, read_manifest, check_payload, extract_checked, safe_name, STAGE,
    check_result_metadata, validate_export_receipt,
    local_preparation,
)

BASE = Path(__file__).resolve().parent.parent/'outputs/cctv_dgp_decoder_hinge_repair_v1_design/import_fixtures'
PIN = hashlib.sha256(b'protocol').hexdigest()


def fixture(extra=None, change=None, optimizer_updates=1, extra_members=None):
    directory = BASE/uuid.uuid4().hex; directory.mkdir(parents=True)
    files = {'protocol.json': b'protocol', 'scripts/worker.py': b'never executed',
        STAGE+'failure.json': json.dumps(dict(complete=False, counts=dict(optimizer_updates=optimizer_updates))).encode()}
    packet = {name: hashlib.sha256(value).hexdigest() for name, value in files.items() if not name.startswith(STAGE)}
    if extra: files.update(extra)
    manifest = dict(complete=True, protocol_sha256=PIN, training_success_not_implied=True,
        model_quality_not_implied=True, model_qualified=False, optimizer_updates=optimizer_updates,
        files={name: dict(bytes=len(value), sha256=hashlib.sha256(value).hexdigest()) for name, value in files.items()})
    if change: change(manifest)
    files['result_manifest.json'] = json.dumps(manifest).encode()
    path = directory/'fixture.tar.gz'
    with tarfile.open(path, 'w:gz') as tar:
        for name, value in list(files.items())+(extra_members or []):
            member = tarfile.TarInfo(name); member.size = len(value); tar.addfile(member, io.BytesIO(value))
    return directory, path, packet


def exported(path, updates=1):
    return dict(complete=True, protocol_sha256=PIN, archive_sha256=hashlib.sha256(path.read_bytes()).hexdigest(),
        bytes=path.stat().st_size, optimizer_updates=updates, training_success_not_implied=True,
        model_quality_not_implied=True, model_qualified=False, run_results_present=False, failure_present=True)


class ImportBoundaries(unittest.TestCase):
    def test_actual_frozen_preparation_and_independent_proof_resolve_before_return(self):
        pin, protocol, prepared_path, proof_path = local_preparation()
        self.assertEqual(pin, hashlib.sha256(prepared_path.parent.parent.joinpath('cctv_dgp_decoder_hinge_repair_v1_vm/protocol.json').read_bytes()).hexdigest())
        self.assertTrue(proof_path.is_file())
        self.assertEqual(len(protocol['assets_sha256']), 18)
        self.assertEqual(len(protocol['replay_required_sha256']), 136)

    def test_valid_archive_checked_before_any_destination_and_keeps_payload_bytes(self):
        directory, path, packet = fixture(); export = exported(path)
        with tarfile.open(path, 'r:gz') as tar:
            names = namespace(tar, 65536); manifest = read_manifest(tar, names, PIN, packet, 65536, export)
            check_payload(tar, names, manifest); check_result_metadata(tar, names, manifest, export)
            self.assertFalse((directory/'return').exists())
            extract_checked(tar, names, manifest, directory/'return')
        self.assertEqual((directory/'return/scripts/worker.py').read_bytes(), b'never executed')

    def test_unsafe_traversal_windows_names_and_invalid_utf8_controls_rejected(self):
        for name in ['../x', '/x', 'C:/x', 'a\\b', './a', 'a//b', 'NUL', 'COM1.txt', 'a/../b', 'a.', 'a ', 'a\x00b', 'a?b']:
            with self.subTest(name=name), self.assertRaises(AssertionError): safe_name(name)

    def test_duplicate_casefold_and_parent_file_collision_rejected(self):
        for extra, members in [({'Protocol.json': b'x'}, None), ({'scripts': b'x'}, None), (None, [('protocol.json', b'duplicate')])]:
            _, path, _ = fixture(extra=extra, extra_members=members)
            with tarfile.open(path, 'r:gz') as tar, self.assertRaises(AssertionError): namespace(tar, 65536)

    def test_links_rejected_before_extraction(self):
        directory = BASE/uuid.uuid4().hex; directory.mkdir(parents=True)
        for kind in [tarfile.SYMTYPE, tarfile.LNKTYPE]:
            path = directory/(str(kind)+'.tar.gz')
            with tarfile.open(path, 'w:gz') as tar:
                member = tarfile.TarInfo('link'); member.type = kind; member.linkname = 'target'; tar.addfile(member)
            with tarfile.open(path, 'r:gz') as tar, self.assertRaises(AssertionError): namespace(tar, 65536)

    def test_uncompressed_and_manifest_member_limits_rejected(self):
        _, path, _ = fixture(extra={STAGE+'oversized.bin': b'x'*128})
        with tarfile.open(path, 'r:gz') as tar, self.assertRaises(AssertionError): namespace(tar, 64)

    def test_wrong_protocol_missing_member_extra_root_and_altered_packet_rejected(self):
        changes = [lambda m: m.update(protocol_sha256='2'*64), lambda m: m['files'].pop(STAGE+'failure.json'),
                   lambda m: m['files']['protocol.json'].update(sha256='3'*64)]
        for change in changes:
            directory, path, packet = fixture(change=change)
            with tarfile.open(path, 'r:gz') as tar, self.assertRaises(AssertionError):
                read_manifest(tar, namespace(tar, 65536), PIN, packet, 65536)
            self.assertFalse((directory/'return').exists())
        _, path, packet = fixture(extra={'unregistered.py': b'no execution'})
        with tarfile.open(path, 'r:gz') as tar, self.assertRaises(AssertionError):
            read_manifest(tar, namespace(tar, 65536), PIN, packet, 65536)

    def test_corrupted_payload_rejected_before_extraction(self):
        directory, path, packet = fixture(change=lambda m: m['files'][STAGE+'failure.json'].update(sha256='4'*64))
        with tarfile.open(path, 'r:gz') as tar:
            names = namespace(tar, 65536); manifest = read_manifest(tar, names, PIN, packet, 65536)
            with self.assertRaises(AssertionError): check_payload(tar, names, manifest)
        self.assertFalse((directory/'return').exists())

    def test_optimizer_limits_and_export_manifest_mismatch_rejected(self):
        for updates in [-1, 51, True, 1.5]:
            _, path, packet = fixture(optimizer_updates=updates)
            with tarfile.open(path, 'r:gz') as tar, self.assertRaises(AssertionError):
                read_manifest(tar, namespace(tar, 65536), PIN, packet, 65536)
        _, path, packet = fixture()
        with tarfile.open(path, 'r:gz') as tar, self.assertRaises(AssertionError):
            read_manifest(tar, namespace(tar, 65536), PIN, packet, 65536, exported(path, updates=2))

    def test_result_payload_update_count_and_presence_bound_to_manifest(self):
        _, path, packet = fixture(change=lambda m: m.update(optimizer_updates=2))
        with tarfile.open(path, 'r:gz') as tar:
            names = namespace(tar, 65536); manifest = read_manifest(tar, names, PIN, packet, 65536)
            with self.assertRaises(AssertionError): check_result_metadata(tar, names, manifest, exported(path, updates=2))
        _, path, packet = fixture()
        export = exported(path); export['failure_present'] = False
        with tarfile.open(path, 'r:gz') as tar:
            names = namespace(tar, 65536); manifest = read_manifest(tar, names, PIN, packet, 65536)
            with self.assertRaises(AssertionError): check_result_metadata(tar, names, manifest, export)

    def test_exact_LF_sidecar_and_explicit_nonqualification_required(self):
        _, path, _ = fixture(); value = exported(path)
        sidecar = (value['archive_sha256']+'  '+path.name+'\n').encode()
        validate_export_receipt(value, PIN, path, sidecar)
        with self.assertRaises(AssertionError): validate_export_receipt(value, PIN, path, sidecar.replace(b'\n', b'\r\n'))
        value['model_qualified'] = True
        with self.assertRaises(AssertionError): validate_export_receipt(value, PIN, path, sidecar)

    def test_existing_import_retained(self):
        directory, path, packet = fixture(); target = directory/'return'; target.mkdir(); (target/'keep').write_bytes(b'keep')
        with tarfile.open(path, 'r:gz') as tar:
            names = namespace(tar, 65536); manifest = read_manifest(tar, names, PIN, packet, 65536)
            with self.assertRaises(AssertionError): extract_checked(tar, names, manifest, target)
        self.assertEqual((target/'keep').read_bytes(), b'keep')


if __name__ == '__main__': unittest.main(verbosity=2)
