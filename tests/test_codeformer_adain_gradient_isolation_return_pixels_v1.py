"""Small meaningful CPU fixtures for audit boundaries; no model execution."""
import ast
from pathlib import Path
import unittest
import uuid

import numpy as np

from research.codeformer_adain_gradient_isolation_v1 import return_pixels as audit


class ReturnPixelsTests(unittest.TestCase):
    def test_native_descriptors_are_input_relative_not_reference_scores(self):
        x = np.full((256, 256, 3), .5, np.float32)
        original = np.full((256, 256, 3), 128, np.uint8)
        row = audit.unpaired_descriptors(x, np.ones((256, 256), bool), original)
        self.assertFalse(row['clean_reference_metrics_computed'])
        self.assertFalse(row['is_quality_or_identity_evidence'])
        self.assertAlmostEqual(row['observed_mean_sRGB_luminance'], .5)
        self.assertGreater(row['observed_MAE_change_from_input_not_target'], 0)

    def test_per_case_limit_is_strict_and_mean_stop_not_overridden(self):
        candidate = {str(i): {'raw': .03, 'native_png': .03} for i in range(61)}
        baseline = {str(i): {'raw': 0., 'native_png': 0.} for i in range(61)}
        criteria = {'mean_clear_mae_delta_stop': .01, 'worst_per_case_clear_mae_delta_stop': .03}
        gate = audit.old.v1.independently_clear_gate(candidate, baseline, criteria)
        self.assertFalse(gate['pass'])
        self.assertEqual(audit.make_failure_rows(gate, 'clear', 'test'), [])
        candidate['0']['native_png'] = .0301
        gate = audit.old.v1.independently_clear_gate(candidate, baseline, criteria)
        self.assertEqual(audit.make_failure_rows(gate, 'clear', 'test')[0]['id'], '0')

    def test_native_score_null_requirement_includes_every_secondary_metric(self):
        row = dict.fromkeys(audit.METRICS)
        self.assertTrue(audit.native_metric_nulls(row))
        row['ssim'] = .9
        self.assertFalse(audit.native_metric_nulls(row))

    def test_witness_array_rejects_trailing_data_and_wrong_dtype(self):
        directory = Path(__file__).resolve().parents[1] / 'scratch' / ('adain_pixel_fixture_' + uuid.uuid4().hex)
        directory.mkdir(parents=True)
        path = directory / 'a.npy'
        np.save(path, np.zeros((2,), np.float32), allow_pickle=False)
        self.assertEqual(audit.load_array(path, (2,), 'float32').shape, (2,))
        with self.assertRaises(ValueError):
            audit.load_array(path, (2,), 'int64')
        with path.open('ab') as stream:
            stream.write(b'x')
        with self.assertRaises(ValueError):
            audit.load_array(path, (2,), 'float32')

    def test_no_neural_framework_import_in_new_auditor(self):
        source = Path(audit.__file__).read_text(encoding='utf-8')
        tree = ast.parse(source)
        imported = []
        for node in ast.walk(tree):
            if isinstance(node, ast.Import):
                imported.extend(alias.name for alias in node.names)
            elif isinstance(node, ast.ImportFrom):
                imported.append(node.module or '')
        self.assertFalse(any(name.startswith(('torch', 'tensorflow', 'jax')) for name in imported))


if __name__ == '__main__':
    unittest.main()
