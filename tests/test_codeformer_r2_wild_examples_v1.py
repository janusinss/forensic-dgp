"""Adversarial CPU fixtures for the five-example recording/download audit."""
import hashlib
import importlib.util
import io
from pathlib import Path
import tempfile
import unittest
import warnings
import zipfile

import numpy as np

SPEC = importlib.util.spec_from_file_location('wild_audit',
    Path(__file__).resolve().parents[1] / 'scripts/audit_codeformer_r2_wild_examples_v1.py')
m = importlib.util.module_from_spec(SPEC); SPEC.loader.exec_module(m)


def manifest():
    result = {'complete': True, 'provenance': m.PROVENANCE, 'evidence': m.EVIDENCE,
        'paired_metrics_available': False, 'new_training': False, 'model_qualified': False,
        'source_pins': dict.fromkeys(m.SOURCE_FILES, '0' * 64), 'cases': []}
    for i in range(1, 6):
        usable = i <= 3
        result['cases'].append({'id': 'wild' + str(i), 'source_path': 'input%d.png' % i,
            'source_sha256': '0' * 64, 'source_bytes': 1, 'provenance': m.PROVENANCE,
            'evidence': m.EVIDENCE, 'input_review': 'usable' if usable else 'needs_clearer',
            'review_reason': 'Input-only judgment', 'review_uncertainty': 'Explicit uncertainty',
            'request': {'method': 'POST', 'endpoint': '/restoration-test/compare', 'selection': 'both',
                'status': 200 if usable else 422, 'elapsed_seconds': 1,
                'response_path': 'response%d.json' % i, 'response_sha256': '0' * 64},
            'downloads': {key: {'path': '%d-%s' % (i, key), 'sha256': '0' * 64, 'bytes': 1}
                for key in ('original_png', 'r2_png', 'bundle_zip')} if usable else {},
            'refused_without_result': not usable, 'no_result_download_links': not usable})
    return result


class WildAuditFixtures(unittest.TestCase):
    def test_prospective_unique_five_case_schema(self):
        self.assertEqual(len(m.validate_manifest(manifest())), 5)

    def test_unknown_images_cannot_become_native_CCTV(self):
        value = manifest(); value['cases'][0]['evidence'] = 'native_unpaired'
        with self.assertRaisesRegex(ValueError, 'relabeled'):
            m.validate_manifest(value)

    def test_insufficient_example_cannot_be_bypassed(self):
        value = manifest(); value['cases'][4]['input_review'] = 'usable'
        with self.assertRaisesRegex(ValueError, 'prospectively'):
            m.validate_manifest(value)

    def test_duplicate_artifacts_refused(self):
        value = manifest()
        value['cases'][1]['request']['response_path'] = value['cases'][0]['request']['response_path']
        with self.assertRaisesRegex(ValueError, 'unique artifact'):
            m.validate_manifest(value)

    def test_refusal_cannot_have_downloads(self):
        value = manifest(); value['cases'][4]['downloads']['r2_png'] = {'path': 'incorrect'}
        with self.assertRaisesRegex(ValueError, 'Download artifacts'):
            m.validate_manifest(value)

    def test_hash_bound_file_detects_same_length_mutation(self):
        with tempfile.TemporaryDirectory(prefix='wild-audit-fixture-', dir=m.ROOT / 'scratch') as folder:
            self.assertTrue(Path(folder).resolve().is_relative_to(m.ROOT / 'scratch'))
            path = Path(folder) / 'file.bin'; path.write_bytes(b'abc')
            pin = {'path': str(path), 'sha256': hashlib.sha256(b'abc').hexdigest(), 'bytes': 3}
            self.assertEqual(m.read_file(pin), b'abc')
            path.write_bytes(b'abd')
            with self.assertRaisesRegex(ValueError, 'fingerprint'):
                m.read_file(pin)

    def test_asymmetric_padding_observed_not_a_removal_mask(self):
        rgb = np.full((165, 138, 3), 37, np.uint8)
        canvas, observed, geometry = m.prepare(rgb)
        self.assertEqual(geometry['native_center_offset'], [13, 0])
        self.assertEqual(geometry['observed_bounds_256'], [20, 0, 234, 256])
        self.assertEqual(int(observed.sum()), 54784)
        self.assertEqual(canvas.shape, (256, 256, 3))
        self.assertEqual(geometry['removal_mask_sha256'], hashlib.sha256(bytes(256 * 256)).hexdigest())

    def test_nonfinite_raw_and_trailing_bytes_refused(self):
        raw = np.zeros((256, 256, 3), np.float32)
        stream = io.BytesIO(); np.save(stream, raw, allow_pickle=False)
        self.assertTrue(np.array_equal(m.raw_array(stream.getvalue()), raw))
        with self.assertRaisesRegex(ValueError, 'Trailing'):
            m.raw_array(stream.getvalue() + b'x')
        raw[0, 0, 0] = np.nan; stream = io.BytesIO(); np.save(stream, raw, allow_pickle=False)
        with self.assertRaisesRegex(ValueError, 'Finite'):
            m.raw_array(stream.getvalue())

    def test_duplicate_zip_names_refused(self):
        names = ('original-upload.png', 'resize-control.png', 'observed-support.png',
            'original.png', 'r2.png', 'original-raw.npy', 'r2-raw.npy', 'processing.json')
        stream = io.BytesIO()
        with warnings.catch_warnings():
            warnings.simplefilter('ignore', UserWarning)
            with zipfile.ZipFile(stream, 'w') as archive:
                for name in names:
                    archive.writestr(name, b'x')
                archive.writestr('original.png', b'y')
        with self.assertRaisesRegex(ValueError, 'unique namespace'):
            m.zip_members(stream.getvalue())


if __name__ == '__main__':
    unittest.main()
