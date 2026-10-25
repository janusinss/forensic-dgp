"""Malformed miniature archive regressions; no model or returned source execution."""
import gzip
import importlib.util
import io
import copy
import json
from pathlib import Path
import tarfile
from types import SimpleNamespace
import unittest


ROOT = Path(__file__).resolve().parents[1]
SPEC = importlib.util.spec_from_file_location('independent_spatial_execution_audit',
    ROOT / 'scripts/audit_u96_spatial_feasibility_execution_v1.py')
A = importlib.util.module_from_spec(SPEC)
SPEC.loader.exec_module(A)


def fixture(entries, *, tail_mutation=False, linked=False):
    stream = io.BytesIO()
    with tarfile.open(fileobj=stream, mode='w') as archive:
        for name, payload in entries:
            member = tarfile.TarInfo(name)
            member.size = len(payload)
            if linked:
                member.type = tarfile.SYMTYPE
                member.linkname = '../outside'
                member.size = 0
            archive.addfile(member, io.BytesIO(payload))
    plain = bytearray(stream.getvalue())
    if tail_mutation:
        plain[1536] = 1
    return gzip.compress(plain, mtime=0)


class ExecutionAudit(unittest.TestCase):
    def test_complete_single_member_and_zero_tail(self):
        payload = b'{"fixture_only":true}\n'
        raw = fixture([(A.STAGE + '/protocol.json', payload)])
        members, contents, framing = A.inspect_stream(io.BytesIO(raw))
        self.assertEqual(contents['protocol.json'], payload)
        self.assertEqual(members['protocol.json']['bytes'], len(payload))
        self.assertTrue(framing['gzip_single_frame_complete'])
        self.assertEqual(framing['complete_zero_tar_padding_bytes'], 9216)

    def test_buffered_nonzero_tar_tail_is_refused(self):
        raw = fixture([(A.STAGE + '/file.txt', b'x')], tail_mutation=True)
        with self.assertRaises(ValueError):
            A.inspect_stream(io.BytesIO(raw))

    def test_extra_gzip_frame_suffix_crc_and_truncation_refused(self):
        raw = fixture([(A.STAGE + '/file.txt', b'x')])
        corrupt = bytearray(raw)
        corrupt[-8] ^= 1
        for bad in (raw + gzip.compress(b'extra'), raw + b'x', raw[:-3], bytes(corrupt)):
            with self.assertRaises((ValueError, EOFError, OSError, tarfile.ReadError, __import__('zlib').error)):
                A.inspect_stream(io.BytesIO(bad))

    def test_duplicate_namespace_escape_and_wrong_prefix_refused(self):
        for entries in (
            [(A.STAGE + '/file.txt', b'a'), (A.STAGE + '/file.txt', b'b')],
            [(A.STAGE + '/../outside', b'a')],
            [('historical_stage/file.txt', b'a')],
            [(A.STAGE + '/a\\b', b'a')],
        ):
            with self.assertRaises(ValueError):
                A.inspect_stream(io.BytesIO(fixture(entries)))

    def test_links_refused(self):
        with self.assertRaises(ValueError):
            A.inspect_stream(io.BytesIO(fixture([(A.STAGE + '/linked', b'')], linked=True)))

    def test_payload_cap_is_enforced(self):
        before = A.MAX_MEMBER
        try:
            A.MAX_MEMBER = 4
            with self.assertRaises(ValueError):
                A.inspect_stream(io.BytesIO(fixture([(A.STAGE + '/large', b'12345')])))
        finally:
            A.MAX_MEMBER = before

    def test_decompression_cap_applies_before_large_PAX_header_parse(self):
        stream = io.BytesIO()
        with tarfile.open(fileobj=stream, mode='w', format=tarfile.PAX_FORMAT) as archive:
            member = tarfile.TarInfo(A.STAGE + '/small.txt')
            member.pax_headers = {'comment': 'x' * 20000}
            member.size = 1
            archive.addfile(member, io.BytesIO(b'x'))
        before = A.MAX_EXPANDED
        try:
            A.MAX_EXPANDED = 1024
            with self.assertRaisesRegex(ValueError, 'before tar parsing'):
                A.inspect_stream(io.BytesIO(gzip.compress(stream.getvalue())))
        finally:
            A.MAX_EXPANDED = before

    def bound_fixture(self):
        parent = {'model': {'r2': 'fixed'}, 'data': {'source': 'fixed'},
                  'cases': [{'id': 'fixed'}], 'guards': [{'id': 'guard'}],
                  'quality_policy': {'unchanged': True}}
        raw = json.dumps(parent).encode()
        review_raw = json.dumps({'complete': True, 'source_review_pass': True,
                                'unresolved_blockers': [], 'sources': []}).encode()
        name = 'evidence/spatial_parent_protocol.json'
        review_name = 'evidence/spatial_runner_source_review_v1.json'
        contents = {name: raw, review_name: review_raw}
        members = {key: {'bytes': len(value), 'sha256': A.sha(value)} for key, value in contents.items()}
        p = {**copy.deepcopy(parent), 'parent_protocol': {'path': name, **members[name]}}
        contract = SimpleNamespace(PARENT_PROTOCOL_BYTES=len(raw), PARENT_PROTOCOL_SHA256=A.sha(raw),
                                   select_parent_cases=lambda data: {'actual_parent_bytes': data})
        return p, members, contents, contract

    def test_approval_is_read_from_archived_bytes_and_pins_cannot_drift(self):
        p, members, contents, contract = self.bound_fixture()
        selected, review = A.parent_review_readback(p, members, contents, contract=contract)
        self.assertTrue(review['source_review_pass'])
        contents['evidence/spatial_runner_source_review_v1.json'] = b'{"source_review_pass":true}'
        with self.assertRaisesRegex(ValueError, 'archived JSON byte binding'):
            A.parent_review_readback(p, members, contents, contract=contract)

    def test_self_resealed_green_approval_cannot_replace_independent_review(self):
        name = 'evidence/spatial_runner_source_review_v1.json'
        trusted = b'{"complete":true,"sources":["reviewed original worker"]}'
        replacement = b'{"complete":true,"sources":["changed unreviewed worker"]}'
        A.bind_trusted_review({name: trusted}, trusted)
        with self.assertRaisesRegex(ValueError, 'independently trusted'):
            A.bind_trusted_review({name: replacement}, trusted)

    def test_all_five_archived_parent_fields_are_binding(self):
        for field in ('model', 'data', 'cases', 'guards', 'quality_policy'):
            p, members, contents, contract = self.bound_fixture()
            p[field] = {'tampered': True}
            with self.assertRaisesRegex(ValueError, 'parent field'):
                A.parent_review_readback(p, members, contents, contract=contract)

    def test_parent_binding_cannot_refer_to_unrelated_local_or_archive_alias(self):
        p, members, contents, contract = self.bound_fixture()
        p['parent_protocol']['path'] = 'evidence/loose_local_parent.json'
        with self.assertRaisesRegex(ValueError, 'immutable parent binding'):
            A.parent_review_readback(p, members, contents, contract=contract)

    def test_parent_data_and_code_bytes_cannot_be_self_resealed(self):
        for name in ('assets/input.png', 'third_party/codeformer/codeformer_arch.py'):
            original = b'fixed parent bytes'
            parent_raw = json.dumps({'files': {name: {'bytes': len(original), 'sha256': A.sha(original)}}}).encode()
            parent_name = 'evidence/spatial_parent_protocol.json'
            p = {'parent_protocol': {'path': parent_name},
                 'files': {name: {'bytes': 7, 'sha256': A.sha(b'changed')}}}
            members = {parent_name: {'bytes': len(parent_raw), 'sha256': A.sha(parent_raw)},
                       name: p['files'][name]}
            contents = {parent_name: parent_raw, 'research/__init__.py': b'"""Inert research namespace."""\n'}
            with self.assertRaisesRegex(ValueError, 'Immutable retained parent'):
                A.retained_namespace_readback(p, members, contents)

    def test_namespace_initializer_or_new_unreviewed_location_refused(self):
        parent_raw = b'{"files":{}}'
        name = 'evidence/spatial_parent_protocol.json'
        p = {'parent_protocol': {'path': name}, 'files': {}}
        members = {name: {'bytes': len(parent_raw), 'sha256': A.sha(parent_raw)}}
        contents = {name: parent_raw, 'research/__init__.py': b'import unexpected\n'}
        with self.assertRaisesRegex(ValueError, 'inert research namespace'):
            A.retained_namespace_readback(p, members, contents)
        contents['research/__init__.py'] = b'"""Inert research namespace."""\n'
        members['unreviewed/new.py'] = {'bytes': 0, 'sha256': A.sha(b'')}
        with self.assertRaisesRegex(ValueError, 'Unexpected new executable'):
            A.retained_namespace_readback(p, members, contents)

    def test_reviewed_package_namespace_and_omitted_weights_are_exact(self):
        approved = {A.PACKAGE + '/worker.py': {}, A.PACKAGE + '/script_entry.py': {},
                    A.PACKAGE + '/run_feasibility.sh': {}}
        models = {'r2': {'path': 'weights/r2.pth', 'bytes': 1, 'sha256': 'a'},
                  'u96': {'path': 'weights/u96.pth', 'bytes': 1, 'sha256': 'b'}}
        p = {'model': models, 'files': {row['path']: {key: row[key] for key in ('bytes', 'sha256')}
                                      for row in models.values()}}
        members = dict.fromkeys((A.PACKAGE + '/worker.py', 'scripts/run_u96_spatial_feasibility_v1.py',
                                 'scripts/run_feasibility.sh'))
        A.reviewed_namespace_readback(p, members, approved)
        for changed in ({**members, A.PACKAGE + '/unreviewed.txt': {}},
                        {key: val for key, val in members.items() if not key.endswith('/worker.py')}):
            with self.assertRaisesRegex(ValueError, 'complete reviewed installed namespace'):
                A.reviewed_namespace_readback(p, changed, approved)
        p['files']['weights/unbound.pth'] = {'bytes': 1, 'sha256': 'c'}
        with self.assertRaisesRegex(ValueError, 'two fixed initializer'):
            A.reviewed_namespace_readback(p, members, approved)


if __name__ == '__main__':
    unittest.main()
