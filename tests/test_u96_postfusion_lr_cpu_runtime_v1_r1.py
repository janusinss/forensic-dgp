"""Pure archive/schema/arithmetic fixtures; never constructs a real model."""
import ast
import copy
import gzip
import hashlib
import importlib.util
import io
import json
from pathlib import Path
import subprocess
import sys
import tarfile
import tempfile
import unittest
from unittest.mock import patch

import numpy as np

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / 'scripts'))
import export_u96_postfusion_lr_inference_v1 as E
import audit_u96_postfusion_lr_cpu_runtime_v1_r1 as R


def pin_bytes(path, raw):
    return {'path': path, 'bytes': len(raw), 'sha256': hashlib.sha256(raw).hexdigest()}


class InertSourceTests(unittest.TestCase):
    def test_inert_import_and_help_refuse_numerical_dependencies(self):
        code = r'''
import ast,importlib,sys
sys.path.insert(0,sys.argv[1])
class Block:
 def find_spec(self,name,path=None,target=None):
  if name.split('.')[0] in {'torch','numpy','PIL','psutil'}: raise AssertionError(name)
sys.meta_path.insert(0,Block())
for name in ['export_u96_postfusion_lr_inference_v1','audit_u96_postfusion_lr_cpu_runtime_v1_r1']:
 m=importlib.import_module(name)
 ast.parse(open(m.__file__,encoding='utf-8').read(),feature_version=(3,10))
 sys.argv=[m.__file__,'--help']
 try:m.main()
 except SystemExit as e:assert e.code==0
 else:raise AssertionError('Expected help exit')
assert 'torch'not in sys.modules and'numpy'not in sys.modules
'''
        result = subprocess.run([sys.executable, '-B', '-c', code, str(ROOT / 'scripts')], capture_output=True, text=True, timeout=15)
        self.assertEqual(result.returncode, 0, result.stderr)

    def test_missing_execute_flag_fails_before_any_source_or_artifact_read(self):
        args = ['runtime', '--source-review', 'missing', '--export-receipt', 'missing', '--plan-sha', '0' * 64,
                '--r2-artifact', 'missing', '--u96-artifact', 'missing']
        with patch.object(sys, 'argv', args), patch.object(E, 'reviewed_source', side_effect=AssertionError('must not read')):
            with self.assertRaisesRegex(ValueError, 'Runtime HOLD'): R.main()


    def test_R1_ROOT_import_path_and_package_resolution_are_explicit_without_Torch(self):
        saved = list(sys.path)
        try:
            sys.path[:] = [p for p in sys.path if p != str(ROOT)]
            result = R.validated_workspace_imports()
            self.assertEqual(sys.path[0], str(ROOT))
            self.assertEqual(result['retained_third_party_package'], str(ROOT / 'third_party'))
            self.assertNotIn('torch', sys.modules)
        finally: sys.path[:] = saved

    def test_prior_failed_source_ledger_retained_and_R1_namespace_is_separate(self):
        for row in R.PRIOR_RUNTIME_FAILURE_EVIDENCE: E.require_pin(row)
        self.assertEqual(R.OUT.name, 'codeformer_u96_postfusion_smoothanchors_lr_v1_cpu_runtime_r1')
        self.assertNotEqual(R.OUT.name, 'codeformer_u96_postfusion_smoothanchors_lr_v1_cpu_runtime')
        self.assertFalse(R.OUT.exists())

    def test_source_and_artifact_pins_match_actual_audited_terminal_metadata(self):
        checkpoint = E.read(E.RETURN / 'returned_metadata/checkpoints/u0064/checkpoint.json')
        self.assertEqual(checkpoint['heads']['file']['sha256'], E.HEAD_SHA)
        self.assertEqual(checkpoint['head_canonical_sha256'], E.HEAD_CANONICAL)
        self.assertEqual(E.pin(E.INFERENCE)['sha256'], E.INFERENCE_SHA)
        self.assertNotIn('torch', sys.modules)

    def test_source_review_requires_current_pins_of_both_scripts_and_inference(self):
        rows = [E.pin(ROOT / path) for path in ['scripts/export_u96_postfusion_lr_inference_v1.py',
                'scripts/audit_u96_postfusion_lr_cpu_runtime_v1.py', 'research/codeformer_u96_postfusion_lr_inference_v1.py']]
        good = {'complete': True, 'source_review_pass': True, 'sources': rows}
        with patch.object(E, 'read', return_value=good): E.reviewed_source('fixture')
        for broken in [{'complete': True, 'source_review_pass': False, 'sources': rows},
                       {'complete': True, 'source_review_pass': True, 'sources': rows[:-1]}]:
            with patch.object(E, 'read', return_value=broken), self.assertRaises(ValueError): E.reviewed_source('fixture')


class VisualAndPlanTests(unittest.TestCase):
    def visual_fixture(self):
        files = [pin_bytes(f'outputs/fixture/sheet{i:02d}.png', bytes([i])) for i in range(30)]
        manifest = {'planned_sheets': 30, 'sheets': [{'file': row} for row in files]}
        manifest_pin = pin_bytes((E.RETURN / 'visual_manifest_v1.json').relative_to(ROOT).as_posix(), b'manifest')
        counts = [28] * 14 + [20] * 16
        root = {'complete': True, 'all_planned_sheets_reviewed': True, 'generated_PNG_cells_reviewed': 712,
                'sheets_reviewed': 30, 'visual_manifest': manifest_pin,
                'observations': [{'file': row, 'personally_viewed_by_root': True, 'generated_cells_reviewed': n}
                                 for row, n in zip(files, counts)]}
        return root, manifest, manifest_pin

    def call_visual(self, root, manifest, manifest_pin):
        def read(path): return manifest if str(path).endswith('visual_manifest_v1.json') else root
        with patch.object(E, 'read', side_effect=read), patch.object(E, 'pin', return_value=manifest_pin), patch.object(E, 'require_pin'):
            return E.visual_closure('fixture-root-closure.json')

    def test_full_root_visual_closure_is_required(self):
        root, manifest, row = self.visual_fixture()
        self.assertTrue(self.call_visual(root, manifest, row)['complete'])
        for name in ['complete', 'all_planned_sheets_reviewed']:
            bad = copy.deepcopy(root); bad[name] = False
            with self.assertRaises(ValueError): self.call_visual(bad, manifest, row)
        bad = copy.deepcopy(root); bad['observations'][0]['personally_viewed_by_root'] = False
        with self.assertRaises(ValueError): self.call_visual(bad, manifest, row)

    def test_missing_duplicate_or_wrong_sheet_and_cell_counts_rejected(self):
        root, manifest, row = self.visual_fixture()
        for mutate in ['missing', 'duplicate', 'cellcount', 'manifestpin']:
            bad = copy.deepcopy(root)
            if mutate == 'missing': bad['observations'].pop()
            elif mutate == 'duplicate': bad['observations'][0] = bad['observations'][1]
            elif mutate == 'cellcount': bad['observations'][0]['generated_cells_reviewed'] = 27
            else: bad['visual_manifest']['sha256'] = '0' * 64
            with self.assertRaises(ValueError): self.call_visual(bad, manifest, row)

    def test_plan_uses_all_four_roles_of_same_first_monitor_photo_only(self):
        protocol = E.read(E.STAGE / 'protocol.json'); audit = E.read(E.RETURN / 'saved_pixel_audit_v1.json')
        rows = E.four_cases(protocol, audit)
        self.assertEqual([x['case']['role'] for x in rows], E.ROLES)
        self.assertEqual({x['case']['reference_id'] for x in rows}, {'tr_ffhq_08698'})
        self.assertTrue(all(x['case']['cohort'] == 'monitor' and not x['case']['optimization_permitted'] for x in rows))
        self.assertTrue(all('/snapshots/u0064/' in x['saved_L4_RAW']['path'] for x in rows))


class ArchiveAndDifferenceTests(unittest.TestCase):
    def archive(self, entries):
        raw = io.BytesIO()
        with tarfile.open(fileobj=raw, mode='w') as archive:
            for relative, data, kind in entries:
                member = tarfile.TarInfo(E.STAGE.name + '/' + relative); member.size = len(data); member.type = kind
                archive.addfile(member, io.BytesIO(data))
        return gzip.compress(raw.getvalue(), mtime=0)

    def invoke(self, raw, expected, selected):
        with tempfile.TemporaryDirectory(dir=ROOT / 'scratch') as directory:
            path = Path(directory) / 'synthetic-not-real-results.tar.gz'; path.write_bytes(raw)
            with patch.object(E, 'ARCHIVE_BYTES', len(raw)), patch.object(E, 'ARCHIVE_SHA', hashlib.sha256(raw).hexdigest()):
                return E.stream_selected(path, expected, selected)

    def test_only_selected_artifact_retained_and_full_archive_pins_bound(self):
        expected = {k: pin_bytes(k, v) for k, v in [('one.bin', b'one'), ('two.bin', b'two')]}
        raw = self.archive([(k, value, tarfile.REGTYPE) for k, value in [('one.bin', b'one'), ('two.bin', b'two')]])
        kept, receipt = self.invoke(raw, expected, {'two.bin'})
        self.assertEqual(kept, {'two.bin': b'two'})
        self.assertEqual(receipt['members'], 2)
        self.assertTrue(receipt['gzip_CRC_footer_drained'])
        self.assertFalse(receipt['unsafe_extract_used'])
        expected['one.bin']['sha256'] = '0' * 64
        with self.assertRaises(ValueError): self.invoke(raw, expected, {'two.bin'})

    def test_duplicate_unsafe_unexpected_and_sparse_members_rejected(self):
        expected = {'one.bin': pin_bytes('one.bin', b'x')}
        for entries in [[('one.bin', b'x', tarfile.REGTYPE)] * 2,
                        [('../escape', b'x', tarfile.REGTYPE)],
                        [('unlisted', b'x', tarfile.REGTYPE)],
                        [('one.bin', b'x', tarfile.GNUTYPE_SPARSE)]]:
            with self.assertRaises((ValueError, tarfile.TarError)):
                self.invoke(self.archive(entries), expected, {'one.bin'})

    def test_truncated_gzip_rejected(self):
        raw = self.archive([('one.bin', b'x', tarfile.REGTYPE)])[:-5]
        with self.assertRaises((ValueError, EOFError, OSError)):
            self.invoke(raw, {'one.bin': pin_bytes('one.bin', b'x')}, {'one.bin'})

    def test_nonzero_suffix_inside_first_TAR_read_ahead_block_rejected(self):
        compressed = self.archive([('one.bin', b'x', tarfile.REGTYPE)])
        raw = bytearray(gzip.decompress(compressed))
        # First file is header512 + one-byte body padded512; terminator begins1024.
        # This byte is after its first zero header and still within read-ahead10KiB.
        raw[1024 + 512 + 7] = 1
        with self.assertRaisesRegex(ValueError, 'buffered TAR suffix'):
            self.invoke(gzip.compress(bytes(raw), mtime=0), {'one.bin': pin_bytes('one.bin', b'x')}, {'one.bin'})

    def test_RAW_header_allocation_bounds_and_comparison_quantification(self):
        raw = np.full((256, 256, 3), .5, np.float32); stream = io.BytesIO(); np.save(stream, raw, allow_pickle=False)
        self.assertTrue(np.array_equal(R.decode_raw(stream.getvalue()), raw))
        bomb = io.BytesIO(); np.lib.format.write_array_header_1_0(bomb, {'descr': '<f4', 'fortran_order': False, 'shape': (2**34,)})
        with self.assertRaises(ValueError): R.decode_raw(bomb.getvalue())
        shifted = raw.copy(); shifted[0, 0, 0] = .51
        result = R.difference(shifted, raw)
        self.assertEqual(result['RAW_components_different'], 1)
        self.assertEqual(result['floored_PNG_pixels_different'], 1)
        self.assertAlmostEqual(result['maximum_absolute_RAW_difference'], float(np.float32(.51) - np.float32(.5)), places=15)
        self.assertTrue(result['no_bitwise_parity_or_quality_tolerance_invented'])
        self.assertNotIn('pass', result)

    def test_strict_JSON_rejects_ambiguous_nonfinite_values(self):
        for value in [b'{"a":1,"a":2}', b'{"a":NaN}', b'{"a":1e999}']:
            with self.assertRaises(ValueError): E.strict_json(value)

    def test_durable_reserved_forward_exception_stays_unknown(self):
        with tempfile.TemporaryDirectory(dir=ROOT / 'scratch') as directory:
            ledger = R.RuntimeLedger(Path(directory) / 'synthetic-ledger-only')
            ledger.update('construct', 'model_construction_attempts')
            ledger.update('constructed', 'model_construction_returns')
            ledger.update('reserved', 'forward_attempts')
            ledger.fail('SYNTHETIC_EXCEPTION_NOT_REAL_RUNTIME', ValueError('fixture'))
            status = E.read(ledger.directory / 'status.json'); failure = E.read(ledger.directory / 'failure_v1.json')
            self.assertTrue(status['reserved_model_or_forward_outcome_unknown'])
            self.assertEqual(failure['counters']['forward_attempts'], 1)
            self.assertEqual(failure['counters']['forward_returns'], 0)
            self.assertFalse((ledger.directory / 'status.pending.json').exists())
            self.assertFalse(status['model_qualified'])

    def test_ledger_four_returns_are_known_but_never_qualified(self):
        with tempfile.TemporaryDirectory(dir=ROOT / 'scratch') as directory:
            ledger = R.RuntimeLedger(Path(directory) / 'synthetic-ledger-only')
            for _ in range(4):
                ledger.update('reserved', 'forward_attempts'); ledger.update('returned', 'forward_returns')
            self.assertFalse(E.read(ledger.directory / 'status.json')['reserved_model_or_forward_outcome_unknown'])
            with self.assertRaises(ValueError): ledger.update('forbidden fifth', 'forward_attempts')


if __name__ == '__main__':
    unittest.main()
