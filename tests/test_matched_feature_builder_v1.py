"""Tiny supplied-file/mock inventory checks; never build a stage or archive."""
import ast
import copy
import hashlib
import importlib.abc
import importlib.util
import io
import json
from pathlib import Path
import stat
import sys
import tempfile
from types import SimpleNamespace
import unittest
from unittest import mock


class FrameworkBlocker(importlib.abc.MetaPathFinder):
    attempted = []

    def find_spec(self, fullname, path=None, target=None):
        if fullname.split('.')[0] in {
                'torch', 'torchvision', 'torchaudio', 'numpy', 'PIL',
                'tensorflow', 'jax', 'jaxlib', 'keras'} or fullname.endswith(('.model', '.worker')):
            self.attempted.append(fullname)
            raise AssertionError('No neural/model/worker import in builder tests: ' + fullname)
        return None


BLOCKER = FrameworkBlocker()
sys.meta_path.insert(0, BLOCKER)
ROOT = Path(__file__).resolve().parents[1]
SPEC = importlib.util.spec_from_file_location('matched_builder_test_module',
    ROOT / 'scripts/build_codeformer_matched_feature_learning_v1.py')
B = importlib.util.module_from_spec(SPEC)
SPEC.loader.exec_module(B)


class BuilderPrimitives(unittest.TestCase):
    def setUp(self):
        (ROOT / 'scratch').mkdir(exist_ok=True)
        self.temporary = tempfile.TemporaryDirectory(prefix='matched_builder_unit_', dir=ROOT / 'scratch')
        self.root = Path(self.temporary.name)
        self.addCleanup(self.temporary.cleanup)

    def file(self, name, raw=b'tiny source\n'):
        path = self.root / name
        path.parent.mkdir(parents=True, exist_ok=True)
        path.write_bytes(raw)
        return path

    def receipt_fixture(self):
        required = {'scripts/new.py', 'tests/test_new.py'}
        inherited = {'outputs/parent/research/old.py': {'bytes': 4, 'sha256': hashlib.sha256(b'old\n').hexdigest()}}
        for name in required:
            self.file(name)
        self.file(next(iter(inherited)), b'old\n')
        proof = 'outputs/proof.json'
        self.file(proof, b'{"complete":true}\n')
        pins = {name: B.content_pin(self.root / name) for name in required | set(inherited)}
        common = {'complete': True, 'exact_current_sources': True,
                  'neural_model_gradient_optimizer_VM_calls': 0,
                  'source_pins': pins, 'evidence_files': {proof: B.content_pin(self.root / proof)}}
        review = {**copy.deepcopy(common), 'independent_review': True,
                  'source_review_pass': True, 'blockers': []}
        tests = {**copy.deepcopy(common), 'passed': True, 'failures': 0, 'errors': 0,
                 'tests_run': 5, 'passed_count': 4, 'skipped': 1}
        return required, inherited, proof, review, tests

    def join(self, values):
        required, inherited, proof, review, tests = values
        with mock.patch.object(B, 'CORE_PROOFS', frozenset((proof,))):
            return B.validate_reviews(self.root, review, tests, required, inherited)

    def test_no_argument_cannot_call_build_or_create_stage(self):
        with mock.patch.object(B, 'build', side_effect=AssertionError('Build not authorized')) as build:
            with self.assertRaisesRegex(ValueError, 'Explicit --build'):
                B.main([])
            build.assert_not_called()

    def test_exact_content_hash_and_pin_inside_declared_base(self):
        path = self.file('data/a', b'known\x00bytes\n')
        expected = {'bytes': len(path.read_bytes()), 'sha256': hashlib.sha256(path.read_bytes()).hexdigest()}
        self.assertEqual(B.content_pin(path), expected)
        self.assertEqual(B.pin(path, self.root), {'path': 'data/a', **expected})
        unrelated = self.root / 'unrelated'; unrelated.mkdir()
        with self.assertRaisesRegex(ValueError, 'escapes'):
            B.pin(path, unrelated)

    def test_hash_change_during_read_is_rejected(self):
        path = self.file('changing', b'a')
        original = Path.open
        def altered(item, *args, **kwargs):
            if item == path and args and args[0] == 'rb':
                with original(item, 'wb') as stream:
                    stream.write(b'changed')
            return original(item, *args, **kwargs)
        with mock.patch.object(Path, 'open', altered), self.assertRaisesRegex(ValueError, 'changed during'):
            B.content_pin(path)

    def test_relative_names_reject_escape_alias_and_control_characters(self):
        self.assertEqual(str(B.relative('evidence/source.json')), 'evidence/source.json')
        for value in ('', '.', '..', '../a', 'a/../b', '/a', 'a//b', 'a/', './a', 'C:/a',
                      'a\\b', 'a\x00b', 'a\nb', True, None):
            with self.subTest(value=value), self.assertRaises(ValueError):
                B.relative(value)

    def test_mock_link_and_reparse_guards_fail_closed(self):
        path = self.file('guard')
        original = Path.lstat
        for mode, links, attributes in ((stat.S_IFREG | 0o600, 2, 0),
                                         (stat.S_IFLNK | 0o777, 1, 0),
                                         (stat.S_IFREG | 0o600, 1, 1024)):
            def altered(item, *args, **kwargs):
                if item == path:
                    return SimpleNamespace(st_mode=mode, st_nlink=links,
                                           st_file_attributes=attributes, st_size=1)
                return original(item, *args, **kwargs)
            with self.subTest(mode=mode, links=links), mock.patch.object(Path, 'lstat', altered):
                with self.assertRaises(ValueError):
                    B.ordinary(path)

    def test_strict_JSON_rejects_duplicates_nonfinite_overflow_and_depth(self):
        for raw in (b'{"x":1,"x":2}', b'[NaN]', b'[Infinity]', b'[1e999]',
                    b'[' * 66 + b'0' + b']' * 66, '{}', bytearray(b'{}')):
            with self.subTest(raw=repr(raw)[:40]), self.assertRaises(ValueError):
                B.strict_json(raw)
        self.assertEqual(B.strict_json(b'{"x":true,"v":[1,null]}'), {'x': True, 'v': [1, None]})

    def test_join_requires_two_current_independent_passing_receipts(self):
        values = self.receipt_fixture()
        result = self.join(values)
        self.assertEqual(set(result['sources']), values[0] | set(values[1]))
        self.assertEqual(set(result['evidence']), {values[2]})
        for receipt_index, field, value in ((3, 'complete', False), (3, 'independent_review', False),
                                          (3, 'source_review_pass', False), (3, 'blockers', ['P2']),
                                          (4, 'passed', False), (4, 'failures', 1),
                                          (4, 'errors', 1), (4, 'skipped', -1),
                                          (4, 'neural_model_gradient_optimizer_VM_calls', True)):
            changed = copy.deepcopy(values)
            changed[receipt_index][field] = value
            with self.subTest(field=field), self.assertRaises(ValueError):
                self.join(changed)

    def test_stale_omitted_extra_or_conflicting_reviewed_sources_are_refused(self):
        for kind in ('stale', 'missing', 'extra', 'different_receipts', 'wrong_inherited'):
            values = self.receipt_fixture()
            name = sorted(values[0])[0]
            if kind == 'stale':
                (self.root / name).write_bytes(b'changed after review')
            elif kind == 'missing':
                del values[3]['source_pins'][name]
            elif kind == 'extra':
                values[3]['source_pins']['unreviewed.py'] = values[3]['source_pins'][name]
            elif kind == 'different_receipts':
                values[4]['source_pins'][name]['sha256'] = '0' * 64
            else:
                values[1][next(iter(values[1]))]['sha256'] = '0' * 64
            with self.subTest(kind=kind), self.assertRaises(ValueError):
                self.join(values)

    def test_readiness_proofs_must_exist_match_and_agree_between_receipts(self):
        for kind in ('missing', 'stale', 'disagree', 'wrong_schema'):
            values = self.receipt_fixture()
            proof = values[2]
            if kind == 'missing':
                values[3]['evidence_files'] = {}
            elif kind == 'stale':
                (self.root / proof).write_bytes(b'changed proof')
            elif kind == 'disagree':
                values[4]['evidence_files'][proof]['sha256'] = '0' * 64
            else:
                values[3]['evidence_files'][proof]['bytes'] = True
            with self.subTest(kind=kind), self.assertRaises(ValueError):
                self.join(values)

    def test_core_proofs_include_every_independent_execution_and_visual_review(self):
        prefix = 'outputs/codeformer_matched_feature_learning_v1_source_preparation/'
        expected = {'independent_factory_graph_review_v1.json', 'independent_contract_review_v1.json',
                    'independent_objective_quality_repair_review_v1.json', 'independent_records_review_v1.json',
                    'independent_execution_repair_review_v1.json', 'independent_packet_bootstrap_repair_review_v1.json',
                    'independent_entry_review_v1.json', 'independent_worker_review_v1.json',
                    'independent_visual_plan_review_v1.json', 'independent_builder_review_v1.json'}
        self.assertTrue({prefix + name for name in expected} <= B.CORE_PROOFS)
        for required_proof in B.CORE_PROOFS:
            values = self.receipt_fixture()
            # Prove the real required name cannot be omitted while a harmless
            # unrelated proof is present. No readiness review is fabricated.
            with self.subTest(proof=required_proof), mock.patch.object(B, 'CORE_PROOFS', frozenset((required_proof,))):
                with self.assertRaisesRegex(ValueError, 'proofs and visual plan'):
                    B.validate_reviews(self.root, values[3], values[4], values[0], values[1])

    def test_verified_tiny_copy_is_exclusive_and_checks_authorized_before_creation(self):
        source = self.file('source', b'payload\n')
        stage = self.root / 'tiny_target'; stage.mkdir()
        target = stage / 'data/result'
        expected = B.content_pin(source)
        installed = B.copy_verified(source, target, expected, source_base=self.root, target_base=stage)
        self.assertEqual(installed, {'path': 'data/result', **expected})
        with self.assertRaises(FileExistsError):
            B.copy_verified(source, target, expected, source_base=self.root, target_base=stage)
        bad_target = stage / 'missing'
        with self.assertRaisesRegex(ValueError, 'authorized pin'):
            B.copy_verified(source, bad_target, {'bytes': 8, 'sha256': '0' * 64},
                            source_base=self.root, target_base=stage)
        self.assertFalse(bad_target.exists())

    def test_copy_cannot_escape_stage_and_no_existing_bytes_overwritten(self):
        source = self.file('source')
        stage = self.root / 'target'; stage.mkdir()
        with self.assertRaisesRegex(ValueError, 'escapes'):
            B.copy_verified(source, self.root / 'outside', B.content_pin(source),
                            source_base=self.root, target_base=stage)

    def test_only_two_named_weights_are_omitted_and_heads_remain(self):
        row = {'bytes': 1, 'sha256': hashlib.sha256(b'x').hexdigest()}
        table = {name: dict(row) for name in B.MODEL_PATHS | {'weights/terminal_u0064_heads.npz', 'a'}}
        result = B.included_archive_files(table)
        self.assertEqual(set(table) - set(result), set(B.MODEL_PATHS))
        self.assertIn('weights/terminal_u0064_heads.npz', result)
        del table['weights/terminal_u0064_heads.npz']
        with self.assertRaises(ValueError):
            B.included_archive_files(table)

    def archive_fixture(self, *, name='a', raw=b'x', kind='regular', size=None):
        path = self.file('mock_archive.gz', b'not a real archive; mocked readers only')
        member = SimpleNamespace(name=B.STAGE_NAME + '/' + name,
            size=len(raw) if size is None else size, linkname='', sparse=None, pax_headers={},
            isfile=lambda: kind == 'regular', issym=lambda: kind == 'symlink',
            islnk=lambda: kind == 'hardlink')
        archive = mock.MagicMock()
        archive.__enter__.return_value = archive
        archive.__iter__.return_value = iter((member,))
        archive.extractfile.return_value = io.BytesIO(raw)
        expected = {'a': {'bytes': 1, 'sha256': hashlib.sha256(b'x').hexdigest()}}
        return path, archive, expected, member

    @staticmethod
    def wire_fixture(raw=b'x'):
        # Minimal raw-block fixture for the independent terminator scanner.
        # Semantic header parsing remains the separately mocked tarfile pass.
        header = bytearray(512)
        header[:1] = b'a'
        header[124:136] = ('%011o' % len(raw)).encode('ascii') + b'\0'
        padding = b'\0' * ((512 - len(raw) % 512) % 512)
        return bytes(header) + raw + padding + b'\0' * 1024

    def test_mock_archive_inventory_streams_every_member_and_complete_gzip(self):
        path, archive, expected, _ = self.archive_fixture()
        with mock.patch.object(B.tarfile, 'open', return_value=archive), mock.patch.object(
                B.gzip, 'open', return_value=io.BytesIO(self.wire_fixture())):
            result = B.audit_archive(path, expected)
        self.assertEqual(result['members'], 1)
        self.assertTrue(result['all_payloads_readback'])
        self.assertTrue(result['full_gzip_CRC_footer_readback'])
        archive.extractfile.assert_called_once()

    def test_mock_archive_rejects_path_link_duplicate_missing_size_or_content(self):
        for kind in ('escape', 'symlink', 'hardlink', 'duplicate', 'missing', 'wrong_size', 'wrong_content'):
            path, archive, expected, member = self.archive_fixture()
            if kind == 'escape': member.name = B.STAGE_NAME + '/../a'
            elif kind in ('symlink', 'hardlink'):
                member.isfile = lambda: False
            elif kind == 'duplicate': archive.__iter__.return_value = iter((member, member))
            elif kind == 'missing': archive.__iter__.return_value = iter(())
            elif kind == 'wrong_size': member.size = 2
            else: archive.extractfile.return_value = io.BytesIO(b'y')
            with self.subTest(kind=kind), mock.patch.object(B.tarfile, 'open', return_value=archive), mock.patch.object(
                    B.gzip, 'open', return_value=io.BytesIO(self.wire_fixture())):
                with self.assertRaises(ValueError):
                    B.audit_archive(path, expected)

    def test_mock_archive_caps_sparse_and_custom_metadata_refused(self):
        for kind in ('size_cap', 'sparse', 'metadata'):
            path, archive, expected, member = self.archive_fixture()
            if kind == 'sparse': member.sparse = [(0, 1)]
            elif kind == 'metadata': member.pax_headers = {'unknown': 'bad'}
            with self.subTest(kind=kind), mock.patch.object(B.tarfile, 'open', return_value=archive), mock.patch.object(
                    B.gzip, 'open', return_value=io.BytesIO(self.wire_fixture())), mock.patch.object(
                    B, 'MAX_ARCHIVE', 1 if kind == 'size_cap' else B.MAX_ARCHIVE):
                with self.assertRaises(ValueError):
                    B.audit_archive(path, expected)

    def test_tar_wire_scanner_rejects_hidden_suffix_padding_truncation_or_missing_end(self):
        good = self.wire_fixture()
        nonzero_padding = bytearray(good); nonzero_padding[1023] = 1
        for raw in (good + self.wire_fixture(b'hidden'), bytes(nonzero_padding),
                    good[:-1], good[:1024], good[:1536], good + b'garbage'):
            with self.subTest(size=len(raw)), self.assertRaises(ValueError):
                B.audit_tar_terminator(io.BytesIO(raw))
        result = B.audit_tar_terminator(io.BytesIO(good + b'\0' * 1024))
        self.assertEqual(result['raw_tar_headers'], 1)
        self.assertTrue(result['nonzero_suffix_after_tar_EOF_refused'])

    def test_tar_wire_scanner_handles_short_reads_and_enforces_expanded_cap(self):
        class Short(io.BytesIO):
            def read(self, size=-1):
                return super().read(min(size, 7))
        good = self.wire_fixture(b'x' * 600)
        self.assertEqual(B.audit_tar_terminator(Short(good))['gzip_expanded_bytes'], len(good))
        with mock.patch.object(B, 'MAX_EXPANDED', 1000), self.assertRaisesRegex(ValueError, 'Bounded'):
            B.audit_tar_terminator(io.BytesIO(good))

    def test_tar_wire_scanner_rejects_noncanonical_size_and_large_payload_header(self):
        for size in (b'not-octal!!!', b'77777777777\0'):
            bad = bytearray(self.wire_fixture())
            bad[124:136] = size
            with self.subTest(size=size), self.assertRaises(ValueError):
                B.audit_tar_terminator(io.BytesIO(bad))

    def test_build_source_is_gated_and_never_executes_old_or_new_workers(self):
        tree = ast.parse(Path(B.__file__).read_text(), feature_version=(3, 10))
        imports = []
        for node in ast.walk(tree):
            if isinstance(node, ast.Import): imports.extend(alias.name for alias in node.names)
            elif isinstance(node, ast.ImportFrom): imports.append(node.module or '')
        self.assertFalse(any(name.split('.')[0] in {'torch', 'numpy', 'PIL'} or
                             name.endswith(('.worker', '.model')) for name in imports))
        build = next(node for node in tree.body if isinstance(node, ast.FunctionDef) and node.name == 'build')
        call_lines = {name: [] for name in ('validate_reviews', 'mkdir', 'import_module')}
        for node in ast.walk(build):
            if isinstance(node, ast.Call):
                name = node.func.id if isinstance(node.func, ast.Name) else getattr(node.func, 'attr', '')
                if name in call_lines: call_lines[name].append(node.lineno)
        self.assertLess(min(call_lines['validate_reviews']), min(call_lines['import_module']))
        self.assertLess(min(call_lines['validate_reviews']), min(call_lines['mkdir']))
        self.assertIn("'--verify-transfer'", Path(B.__file__).read_text())
        self.assertEqual(BLOCKER.attempted, [])
        self.assertFalse({'torch', 'numpy', 'PIL'} & set(sys.modules))


if __name__ == '__main__':
    unittest.main()
