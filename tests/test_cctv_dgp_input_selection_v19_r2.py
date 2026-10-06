import copy
import json
from pathlib import Path
import unittest

import numpy as np

import cctv_dgp_input_selection_v19_r2 as q
from dgp_input_normalization_v19_r2 import retained_input, spatial_input, spatial_cpu_replay_input

ROOT = Path(__file__).resolve().parents[1]
RETURN = ROOT / 'outputs/cctv_dgp_input_selection_v19_parity_diagnostic_return_r1'


class NormalizationCorrectionTests(unittest.TestCase):
    def test_actual_returned_encodings_reproduced_without_network_forwards(self):
        from PIL import Image
        diag = json.loads((RETURN / 'results.json').read_text())
        plan = json.loads((ROOT / 'outputs/cctv_dgp_input_selection_vm_v19/input_selection_protocol_v19.json').read_text())
        cases = {c['id']: c for c in plan['cases'] + plan['training_parity_cases']}
        for row in diag['rows']:
            with Image.open(ROOT / 'outputs/cctv_dgp_mixed_vm_v9_r2' / cases[row['id']]['input']) as im:
                camera = np.asarray(im).copy()
            canonical = retained_input(camera)[0].permute(1, 2, 0).numpy()
            scalar = spatial_cpu_replay_input(camera)[0].permute(1, 2, 0).numpy()
            self.assertTrue(np.array_equal(canonical, np.load(RETURN / row['routes']['numpy_before_GPU']['input'])))
            self.assertTrue(np.array_equal(scalar, np.load(RETURN / row['routes']['CUDA_scalar_division']['input'])))
            self.assertFalse(np.array_equal(canonical, scalar))
            self.assertEqual(retained_input(camera).stride(), tuple(row['routes']['numpy_before_GPU']['tensor_stride']))
            self.assertFalse(retained_input(camera).requires_grad)

    def test_all_byte_values_and_illegal_input_shapes(self):
        camera = np.broadcast_to(np.arange(256, dtype=np.uint8)[None, :, None], (256, 256, 3)).copy()
        a = retained_input(camera)[0].permute(1, 2, 0).numpy()
        b = spatial_cpu_replay_input(camera)[0].permute(1, 2, 0).numpy()
        self.assertTrue(np.array_equal(a, camera.astype(np.float32) / 255))
        self.assertTrue(np.array_equal(b, camera.astype(np.float32) * np.float32(1 / 255)))
        self.assertEqual(np.count_nonzero(a[0, :, 0] != b[0, :, 0]), 126)
        for bad in [camera.astype(np.float32), camera[:128], camera[..., 0], None]:
            for fn in [retained_input, spatial_input, spatial_cpu_replay_input]:
                with self.assertRaisesRegex(ValueError, 'RGB256'):
                    fn(bad)

    def test_original_scientific_requirements_cohort_counts_and_aliases_unchanged(self):
        for key, value in q.original.DESIGN.items():
            self.assertEqual(q.DESIGN[key], value)
        expected = q.original.expected_counts()
        self.assertEqual(q.expected_counts()['dgp'], expected['dgp'] + 520)
        self.assertEqual({k: v for k, v in q.expected_counts().items() if k != 'dgp'},
                         {k: v for k, v in expected.items() if k != 'dgp'})
        item = {'raw': 'raw/a.npy', 'prediction': 'a.png'}
        row = {'decision': {'branch': 'retained_dgp_v2'}, 'arms': {key: copy.deepcopy(item) for key in q.ARMS}}
        q.validate_automatic_alias(row)
        row['arms']['automatic_v19']['raw'] = 'changed.npy'
        with self.assertRaisesRegex(ValueError, 'alias'):
            q.validate_automatic_alias(row)

    def test_relaxed_normalization_scope_training_and_native_claims_rejected(self):
        r = {'complete': True, 'protocol_sha256': 'pin', 'seconds': 1,
             'training': False, 'validation_used': True, 'optimizer_updates': 0, 'backward_calls': 0,
             'teacher_used': False, 'native_used': False, 'native_reserved_used': False,
             'best_checkpoint_selected': False, 'thresholds_refitted': False, 'production_promoted': False,
             'normalization_policy': q.NORMALIZATION_POLICY, 'original_scientific_gates_changed': False,
             'internal_spatial_base_raw_exports': 520}
        q.validate_result_scope(r, 'pin')
        for key, value in [('original_scientific_gates_changed', True), ('normalization_policy', {}),
                           ('internal_spatial_base_raw_exports', 519), ('training', True),
                           ('native_reserved_used', True), ('thresholds_refitted', True), ('optimizer_updates', 1)]:
            with self.assertRaises(ValueError):
                q.validate_result_scope({**r, key: value}, 'pin')


if __name__ == '__main__':
    unittest.main()
