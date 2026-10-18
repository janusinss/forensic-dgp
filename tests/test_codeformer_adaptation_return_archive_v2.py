"""Synthetic-byte fixtures only for the prepared local V2 transport auditor."""
import ast
import hashlib
import io
import json
from pathlib import Path, PurePosixPath
import shutil
import tarfile
import unittest
from unittest.mock import patch
import uuid

from research.codeformer_adaptation_v2 import return_archive as auditor

ROOT = Path(__file__).resolve().parents[1]


def digest(payload):
    return hashlib.sha256(payload).hexdigest()


def encode(value):
    return (json.dumps(value, sort_keys=True) + '\n').encode('utf-8')


class ReturnV2Tests(unittest.TestCase):
    def setUp(self):
        self.work = ROOT / 'scratch' / ('codeformer_return_v2_fixture_' + uuid.uuid4().hex)
        self.work.mkdir()
        self.packet = self.work / 'packet'; self.packet.mkdir()
        self.archive = self.work / auditor.RESULT_ARCHIVE
        self.sidecar = self.work / (auditor.RESULT_ARCHIVE + '.sha256')
        self.export_path = self.work / auditor.EXPORT_RECEIPT
        self.destination = self.work / 'new_import'
        self.receipt_path = self.work / 'transport.json'
        self.manifest_name = auditor.OUTPUTS + '/export_manifest_20261015T120000Z-1234abcd.json'
        self.originals = {'research/source.py': b'# original frozen fixture\n',
                          'LICENSE': b'fixture terms\n', 'data/input.bin': b'original input',
                          'weights/model.pth': b'synthetic bytes, never torch-loaded'}
        for name, value in self.originals.items():
            path = self.packet / name; path.parent.mkdir(parents=True, exist_ok=True); path.write_bytes(value)
        self.protocol = {'format': 'codeformer-adaptation-finite-manual-v2', 'outputs': auditor.OUTPUTS,
            'export_prefix': 'cctv-codeformer-adaptation-v2',
            'export_files': ['research/source.py', 'LICENSE'],
            'files_sha256': {n: digest(v) for n, v in self.originals.items()},
            'assets': {'initializer': {'path': 'weights/model.pth',
                                      'sha256': digest(self.originals['weights/model.pth']),
                                      'bytes': len(self.originals['weights/model.pth'])}},
            'limits': {'result_payload_bytes': auditor.MAX_FINAL_BYTES,
                'checkpoint_temporary_bytes': auditor.MAX_TEMP_BYTES, 'archive_bytes': auditor.MAX_ARCHIVE_BYTES,
                'metadata_bytes': auditor.MAX_METADATA_BYTES, 'reserve_free_bytes': auditor.RESERVE_BYTES}}
        self.protocol_bytes = encode(self.protocol)
        (self.packet / 'protocol.json').write_bytes(self.protocol_bytes)
        self.pin = digest(self.protocol_bytes)
        self.files = {'protocol.json': self.protocol_bytes, 'research/source.py': self.originals['research/source.py'],
                      'LICENSE': self.originals['LICENSE'], auditor.OUTPUTS + '/failure.json': b'{"complete":false}\n'}

    def tearDown(self):
        exact = self.work.resolve(strict=True)
        assert exact.parent == (ROOT / 'scratch').resolve(strict=True)
        assert exact.name.startswith('codeformer_return_v2_fixture_') and not self.work.is_symlink()
        shutil.rmtree(exact)

    def build(self, *, files=None, listed=None, extras=None, manifest_change=None, export_change=None, types=None):
        files = dict(self.files if files is None else files)
        listed = {n: digest(v) for n, v in files.items()} if listed is None else dict(listed)
        incomplete = {n: {'sha256': digest(v), 'bytes': len(v), 'resume_safe': False,
                         'reason': 'Incomplete fixture write retained as failure evidence'}
                      for n, v in files.items() if auditor._incomplete(n)}
        retained = {n: v for n, v in self.protocol['files_sha256'].items() if n not in listed}
        manifest = {'complete': True, 'protocol_sha256': self.pin, 'model_qualified': False,
            'training_success_not_implied': True, 'immutable_inputs_and_original_models_not_duplicated': True,
            'run_results_present': False, 'failure_present': True, 'files_sha256': listed,
            'retained_not_exported_sha256': retained, 'incomplete_temp_files': incomplete,
            'incomplete_temp_files_are_not_resumable_checkpoints': True}
        if manifest_change:
            manifest.update(manifest_change)
        manifest_payload = encode(manifest)
        members = dict(files, **{self.manifest_name: manifest_payload})
        with tarfile.open(self.archive, 'w:gz') as out:
            for name, value in {**members, **(extras or {})}.items():
                member = tarfile.TarInfo(name); member.size = len(value)
                if types and name in types:
                    member.type = types[name]; member.linkname = '../../outside'
                out.addfile(member, io.BytesIO(value))
        total = sum(map(len, members.values()))
        temporary = sum(len(v) for n, v in members.items()
                        if n.startswith(auditor.OUTPUTS + '/') and '.tmp-' in PurePosixPath(n).name)
        output_final = sum(len(v) for n, v in members.items()
                           if n.startswith(auditor.OUTPUTS + '/') and '.tmp-' not in PurePosixPath(n).name)
        output_metadata = sum(len(v) for n, v in members.items() if n.startswith(auditor.OUTPUTS + '/') and
            PurePosixPath(n).suffix.lower() in {'.json', '.jsonl', '.log', '.txt', '.sha256'})
        exported_metadata = sum(len(v) for n, v in members.items() if auditor._metadata(n))
        tar_upper = sum(((len(v) + 511) // 512) * 512 + 4096 for v in members.values()) + 1024**2
        archive_upper = tar_upper + (tar_upper // 1000 + 1) + 1024**2
        export = {'complete': True, 'protocol_sha256': self.pin, 'archive_sha256': digest(self.archive.read_bytes()),
            'bytes': self.archive.stat().st_size, 'manifest_member': self.manifest_name,
            'manifest_sha256': digest(manifest_payload), 'run_results_present': False, 'failure_present': True,
            'model_qualified': False, 'training_success_not_implied': True,
            'incomplete_temp_files': len(incomplete), 'retained_not_exported_files': len(retained),
            'upper_payload_bytes': total, 'conservative_archive_upper_bytes': archive_upper,
            'result_accounting': {'final_bytes': output_final, 'temporary_bytes': temporary,
                                  'metadata_bytes': output_metadata},
            'exported_metadata_bytes': exported_metadata}
        if export_change:
            export.update(export_change)
        self.export_path.write_bytes(encode(export))
        self.sidecar.write_bytes((export['archive_sha256'] + '  ' + self.archive.name + '\n').encode('ascii'))
        return members

    def inspect(self, **kwargs):
        with patch.object(auditor, 'PROTOCOL_SHA256', self.pin), patch.object(auditor.shutil, 'disk_usage',
                return_value=shutil._ntuple_diskusage(64 << 30, 0, 64 << 30)):
            return auditor.inspect_and_import(self.archive, self.sidecar, self.export_path, self.packet,
                kwargs.get('destination', self.destination), kwargs.get('receipt_path', self.receipt_path))

    def rejected_before_import(self):
        with self.assertRaises((ValueError, tarfile.TarError, EOFError)):
            self.inspect()
        self.assertFalse(self.destination.exists())
        self.assertTrue(self.archive.is_file())
        self.assertTrue(self.receipt_path.with_name('transport_failure.json').is_file())

    def test_preparation_status_is_not_actual_V2_audit(self):
        value = auditor.preparation_status()
        self.assertFalse(value['actual_V2_return_audited'])
        self.assertFalse(value['model_qualified'])
        self.assertEqual(value['VM_calls'], 0)
        source = Path(auditor.__file__).read_text(encoding='utf-8')
        tree = ast.parse(source, feature_version=(3, 10))
        for node in ast.walk(tree):
            if isinstance(node, ast.Import):
                self.assertFalse({n.name.split('.')[0] for n in node.names} & {'torch', 'numpy', 'PIL', 'cv2', 'tensorflow'})
            if isinstance(node, ast.ImportFrom):
                self.assertNotIn((node.module or '').split('.')[0], {'torch', 'numpy', 'PIL', 'cv2'})

    def test_valid_import_exact_bytes_journal_and_zero_neural(self):
        members = self.build()
        receipt = self.inspect()
        self.assertTrue(receipt['complete'])
        self.assertEqual(receipt['members'], len(members))
        self.assertEqual(receipt['retained_inputs_verified'], 2)
        self.assertEqual(receipt['all_original_packet_inputs_verified'], 4)
        self.assertTrue(receipt['all_members_verified_before_destination_created'])
        self.assertTrue(receipt['failure_present'])
        self.assertFalse(receipt['run_results_present'])
        self.assertFalse(receipt['model_qualified'])
        self.assertEqual(receipt['neural_calls'], 0)
        for name, payload in members.items():
            self.assertEqual((self.destination / name).read_bytes(), payload)
        journal = [json.loads(x) for x in Path(receipt['journal']).read_text().splitlines()]
        self.assertEqual(len(journal), len(members) * 2)
        self.assertEqual({x['file'] for x in journal if x['event'] == 'imported'}, set(members))
        with self.assertRaises(ValueError):
            self.inspect()

    def test_incomplete_checkpoint_retained_and_classified_not_resumable(self):
        files = dict(self.files, **{auditor.OUTPUTS + '/checkpoint.pt.tmp-1234': b'incomplete checkpoint fixture'})
        self.build(files=files)
        receipt = self.inspect()
        self.assertEqual(receipt['incomplete_temp_files'], 1)
        self.assertTrue(receipt['incomplete_files_preserved_not_resumable'])
        self.assertEqual((self.destination / auditor.OUTPUTS / 'checkpoint.pt.tmp-1234').read_bytes(),
                         b'incomplete checkpoint fixture')

    def test_windows_paths_and_devices_refused(self):
        for name in ('', '.', '../a', '/a', 'a/../b', 'a//b', 'a/./b', 'a\\b', 'C:/a', 'a/nul.txt',
                     'CON', 'COM¹.txt', 'LPT²', 'CONIN$', 'a/file ', 'a/file.', 'a/q?.png', 'a/s*.png',
                     'a/q".txt', 'a/less<.txt', 'a/pipe|.txt', 'a/\x00.txt'):
            with self.subTest(name=name), self.assertRaises(ValueError):
                auditor.canonical_member(name)

    def test_case_and_file_directory_collisions_refused(self):
        for names in (['outputs/X', 'outputs/x'], ['outputs/X', 'outputs/x/child']):
            with self.subTest(names=names), self.assertRaises(ValueError):
                auditor._namespace(names)
        self.build(files=dict(self.files, **{auditor.OUTPUTS + '/Case.bin': b'a', auditor.OUTPUTS + '/case.bin': b'b'}))
        self.rejected_before_import()

    def test_symlink_hardlink_device_and_sparse_refused(self):
        for kind in (tarfile.SYMTYPE, tarfile.LNKTYPE, tarfile.CHRTYPE, tarfile.BLKTYPE, tarfile.DIRTYPE):
            info = tarfile.TarInfo('safe'); info.type = kind
            with self.subTest(kind=kind), self.assertRaises(ValueError):
                auditor._member_regular(info)
        info = tarfile.TarInfo('safe'); info.sparse = [(0, 1)]
        with self.assertRaises(ValueError):
            auditor._member_regular(info)
        info = tarfile.TarInfo('safe'); info.pax_headers = {'GNU.sparse.size': '1'}
        with self.assertRaises(ValueError):
            auditor._member_regular(info)
        self.build(types={auditor.OUTPUTS + '/failure.json': tarfile.SYMTYPE})
        self.rejected_before_import()

    def test_unlisted_member_refused(self):
        self.build(extras={auditor.OUTPUTS + '/unlisted.bin': b'not declared'})
        self.rejected_before_import()

    def test_tampered_payload_refused_before_any_import(self):
        listed = {n: digest(v) for n, v in self.files.items()}
        files = dict(self.files, **{auditor.OUTPUTS + '/failure.json': b'changed but unlisted hash'})
        self.build(files=files, listed=listed)
        self.rejected_before_import()

    def test_mutated_returned_frozen_source_refused(self):
        self.build(files=dict(self.files, **{'research/source.py': b'mutated source'}))
        self.rejected_before_import()

    def test_missing_mandatory_companion_refused(self):
        files = dict(self.files); del files['LICENSE']
        self.build(files=files)
        self.rejected_before_import()

    def test_manifest_status_disagreement_refused(self):
        self.build(manifest_change={'failure_present': False})
        self.rejected_before_import()

    def test_mutated_local_protocol_refused(self):
        self.build()
        (self.packet / 'protocol.json').write_bytes(self.protocol_bytes + b' ')
        self.rejected_before_import()

    def test_mutated_retained_input_refused(self):
        self.build()
        (self.packet / 'data/input.bin').write_bytes(b'changed original')
        self.rejected_before_import()

    def test_mutated_local_exported_original_also_refused(self):
        self.build()
        (self.packet / 'research/source.py').write_bytes(b'changed local source')
        self.rejected_before_import()

    def test_incomplete_receipt_must_not_claim_resumable(self):
        name = auditor.OUTPUTS + '/checkpoint.pt.tmp-1234'
        value = b'partial'
        self.build(files=dict(self.files, **{name: value}), manifest_change={'incomplete_temp_files': {
            name: {'sha256': digest(value), 'bytes': len(value), 'resume_safe': True, 'reason': 'invalid'}}})
        self.rejected_before_import()

    def test_accounting_and_finite_archive_bound_refused(self):
        self.build(export_change={'upper_payload_bytes': 0})
        self.rejected_before_import()
        with patch.object(auditor, 'MAX_ARCHIVE_BYTES', 1), self.assertRaises(ValueError):
            self.inspect(receipt_path=self.work / 'second.json')

    def test_LF_checksum_exact_and_bad_hash_refused(self):
        self.build()
        self.sidecar.write_bytes(self.sidecar.read_bytes().replace(b'\n', b'\r\n'))
        self.rejected_before_import()

    def test_compressed_payload_checksum_refused(self):
        self.build()
        with self.archive.open('ab') as output:
            output.write(b'changed')
        self.rejected_before_import()

    def test_existing_journal_and_failure_receipts_preserved(self):
        self.build()
        journal = self.work / 'new_import_import_journal.jsonl'; journal.write_bytes(b'prior journal')
        with self.assertRaises(ValueError):
            self.inspect()
        self.assertEqual(journal.read_bytes(), b'prior journal')
        self.assertFalse(self.destination.exists())

    def test_disk_reserve_refused_without_destination(self):
        self.build()
        with patch.object(auditor, 'PROTOCOL_SHA256', self.pin), patch.object(auditor.shutil, 'disk_usage',
                return_value=shutil._ntuple_diskusage(1, 1, 0)), self.assertRaises(ValueError):
            auditor.inspect_and_import(self.archive, self.sidecar, self.export_path, self.packet,
                                       self.destination, self.receipt_path)
        self.assertFalse(self.destination.exists())

    def test_duplicate_nonfinite_json_and_wrong_protocol_family_refused(self):
        for payload in (b'{"a":1,"a":2}', b'{"a":NaN}', b'{"a":Infinity}'):
            with self.assertRaises(ValueError):
                auditor._json(payload)
        self.build()
        with patch.object(auditor, 'PROTOCOL_SHA256', 'f' * 64), self.assertRaises(ValueError):
            auditor.inspect_and_import(self.archive, self.sidecar, self.export_path, self.packet,
                                       self.destination, self.receipt_path)


if __name__ == '__main__':
    unittest.main()
