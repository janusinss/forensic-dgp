import importlib.util
import math
from pathlib import Path
import unittest

from cctv_dgp_structure_audit_numeric_v18_r1 import (
    RELATIVE_ACCUMULATION_TOLERANCE, verify_training_weighting)
import cctv_dgp_structure_v18 as q


class PortableAccumulationTests(unittest.TestCase):
    def setUp(self):
        self.means = {'group' + str(i): .002 * (i + 1) for i in range(10)}
        self.weighting = {'raw_MSE_means': self.means.copy(),
                          'weights': q.normalized_group_weights(self.means)}

    def test_roundoff_can_pass_while_coefficients_stay_exact(self):
        local = {key: value * (1 + RELATIVE_ACCUMULATION_TOLERANCE / 4) for key, value in self.means.items()}
        expected, receipt = verify_training_weighting(self.weighting, local, self.means, q.normalized_group_weights)
        self.assertEqual(expected, self.weighting['weights'])
        self.assertTrue(receipt['exact_recorded_coefficient_derivation'])

    def test_material_mean_change_rejected(self):
        for role in ['local', 'recorded']:
            local = self.means.copy()
            weighting = {'raw_MSE_means': self.means.copy(), 'weights': self.weighting['weights']}
            target = local if role == 'local' else weighting['raw_MSE_means']
            target['group3'] *= 1 + 10 * RELATIVE_ACCUMULATION_TOLERANCE
            with self.assertRaisesRegex(ValueError, 'accumulation tolerance'):
                verify_training_weighting(weighting, local, self.means, q.normalized_group_weights)

    def test_any_coefficient_change_rejected(self):
        self.weighting['weights']['group3'] += 1e-10
        with self.assertRaisesRegex(ValueError, 'exact frozen derivation'):
            verify_training_weighting(self.weighting, self.means, self.means, q.normalized_group_weights)

    def test_nonfinite_nonpositive_and_missing_group_rejected(self):
        for bad in [math.nan, math.inf, 0, -.001]:
            local = self.means.copy()
            local['group1'] = bad
            with self.assertRaises(ValueError):
                verify_training_weighting(self.weighting, local, self.means, q.normalized_group_weights)
        local = self.means.copy()
        local.pop('group2')
        with self.assertRaisesRegex(ValueError, 'ten training-only'):
            verify_training_weighting(self.weighting, local, self.means, q.normalized_group_weights)

    def test_correction_is_limited_to_one_frozen_assertion(self):
        root = Path(__file__).resolve().parents[1]
        spec = importlib.util.spec_from_file_location('external_audit_correction', root / 'scripts/recover_cctv_dgp_structure_v18_audit_r1.py')
        recovery = importlib.util.module_from_spec(spec)
        spec.loader.exec_module(recovery)
        source = (root / 'outputs/cctv_dgp_structure_vm_v18/scripts/audit_cctv_dgp_structure_v18.py').read_text()
        fixed = recovery.corrected_source(source)
        self.assertEqual(fixed.replace(recovery.NEW, recovery.OLD), source)
        with self.assertRaises(ValueError):
            recovery.corrected_source(fixed)
        self.assertIn('delta <= 5e-5', fixed)
        self.assertIn('delta <= 2e-6', fixed)
        self.assertIn('q.strict_preservation(summary, baseline_summary)', fixed)


if __name__ == '__main__':
    unittest.main()
