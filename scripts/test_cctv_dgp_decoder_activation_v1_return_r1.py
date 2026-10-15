"""Regression against retained real gate schema; arrays and models stay unread."""
import copy
import json
from pathlib import Path
import unittest
import numpy as np

import audit_cctv_dgp_decoder_activation_v1_return as original
import audit_cctv_dgp_decoder_activation_v1_return_r1 as repaired

ROOT = Path(__file__).resolve().parents[1]
REPAIR = ROOT/'outputs/cctv_dgp_decoder_hinge_repair_v1_return/outputs/decoder_hinge_repair_v1'
SOURCE = ROOT/'outputs/cctv_dgp_spatial_readout_v1_return'


class RetainedGateSchema(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        cls.before = json.loads((REPAIR/'baseline/metrics.json').read_text())['groups']
        cls.states = {}
        for label in ['initial','probe_1_16','probe_1_4','fit_u001']:
            cls.states[label] = json.loads((REPAIR/label/'metrics.json').read_text())['groups']

    def test_original_actual_nine_key_schema_is_reproduced_and_rejected(self):
        for label in ['probe_1_16','probe_1_4','fit_u001']:
            retained = json.loads((REPAIR/(label+'_outcome.json')).read_text())['gates']
            for mode in ['raw','png']:
                with self.subTest(state=label,mode=mode):
                    old = original.unchanged_gate(self.before[mode],self.states[label][mode])
                    self.assertEqual(len(old),5)
                    self.assertEqual(len(retained[mode]),9)
                    with self.assertRaises(AssertionError):
                        original.close_tree(old,retained[mode],atol=1e-12,rtol=1e-12)

    def test_all_four_states_equal_exact_frozen_parent_gate(self):
        frozen = repaired.load_frozen_gate(SOURCE)
        for label,groups in self.states.items():
            for mode in ['raw','png']:
                with self.subTest(state=label,mode=mode):
                    computed = repaired.repaired_gate(self.before[mode],groups[mode])
                    self.assertEqual(computed,frozen(self.before[mode],groups[mode]))
                    self.assertEqual(len(computed),9)

    def test_actual_retained_outcome_schema_passes_without_numerical_change(self):
        for label in ['probe_1_16','probe_1_4','fit_u001']:
            retained = json.loads((REPAIR/(label+'_outcome.json')).read_text())['gates']
            for mode in ['raw','png']:
                repaired.compare_gate_record(repaired.repaired_gate(self.before[mode],self.states[label][mode]),retained[mode])

    def test_changed_metadata_thresholds_and_numerics_are_rejected(self):
        baseline = repaired.repaired_gate(self.before['raw'],self.states['probe_1_16']['raw'])
        for key,value in [('numerical_sample_gate_only',False),('model_qualified',True),
                          ('automatic_extra_epochs',True),('global_structure_gain',baseline['global_structure_gain']+.001),
                          ('pass',True)]:
            changed = copy.deepcopy(baseline);changed[key] = value
            with self.subTest(key=key),self.assertRaises(AssertionError):
                repaired.compare_gate_record(baseline,changed)
        for key in baseline['scientific_thresholds']:
            changed = copy.deepcopy(baseline);changed['scientific_thresholds'][key] *= 2
            with self.subTest(threshold=key),self.assertRaises(AssertionError):
                repaired.compare_gate_record(baseline,changed)
        for missing in baseline:
            changed = copy.deepcopy(baseline);del changed[missing]
            with self.subTest(missing=missing),self.assertRaises(AssertionError):
                repaired.compare_gate_record(baseline,changed)
        changed = copy.deepcopy(baseline);changed['extra'] = False
        with self.assertRaises(AssertionError):
            repaired.compare_gate_record(baseline,changed)

    def test_exact_direction_proof_preserves_approximate_metadata_discrepancy(self):
        direction = np.array([3.,4.],np.float64)
        record = dict(direction_sha256=original.array_sha(direction),dtype='float64',elements=2,
                      new_parameter_proposal=False,norm_L2=float(np.nextafter(5.,0.)))
        proof = repaired.exact_direction_radius(direction,record,dict(common_radius_L2=5.))
        self.assertTrue(proof['exact_squared_norm_le_unchanged_squared_radius'])
        self.assertFalse(proof['metadata_bitwise_replay'] or proof['saved_norm_matches_correct_rounded'])
        self.assertTrue(proof['saved_norm_approximate_unverified'])
        self.assertFalse(proof['saved_metadata_rewritten'] or proof['ULP_acceptance_tolerance_used'])
        self.assertEqual(proof['converged_correct_rounded_norm'],5.)
        self.assertEqual(record['norm_L2'],float(np.nextafter(5.,0.)))

    def test_exact_radius_boundary_rejects_one_ULP_larger_vector(self):
        direction = np.array([np.nextafter(5.,np.inf)],np.float64)
        record = dict(direction_sha256=original.array_sha(direction),dtype='float64',elements=1,
                      new_parameter_proposal=False,norm_L2=5.)
        with self.assertRaisesRegex(AssertionError,'unchanged scientific radius'):
            repaired.exact_direction_radius(direction,record,dict(common_radius_L2=5.))

    def test_nonfinite_negative_norm_metadata_and_changed_array_rejected(self):
        direction = np.array([3.,4.],np.float64)
        record = dict(direction_sha256=original.array_sha(direction),dtype='float64',elements=2,
                      new_parameter_proposal=False,norm_L2=5.)
        for value in [float('nan'),float('inf'),-5.,0.]:
            altered = dict(record,norm_L2=value)
            with self.subTest(value=value),self.assertRaises(AssertionError):
                repaired.exact_direction_radius(direction,altered,dict(common_radius_L2=5.))
        changed = direction.copy();changed[0] = 2.
        with self.assertRaises(AssertionError):
            repaired.exact_direction_radius(changed,record,dict(common_radius_L2=5.))

    def test_local_ast_changes_only_known_norm_assertion_and_uses_schema_adapter(self):
        proofs = []
        module,source = repaired.load_adapter_module(proofs)
        self.assertIs(module.unchanged_gate,repaired.repaired_gate)
        self.assertIn('assert _r1_verify_exact_direction_radius(direction, direction_record, p)',source)
        self.assertEqual(proofs,[])

    def test_objective_only_existing_absolute_bound_accepts_reduction_rounding(self):
        actual = dict(objective=3.0008850148634116,other_scalar=0.,nested=dict(value=[1.,2.]))
        saved = copy.deepcopy(actual);saved['objective']=3.000885014863412
        proof = repaired.compare_finite_objective_only(actual,saved,1e-10)
        self.assertFalse(proof['original_exact_objective_check_passed'])
        self.assertTrue(proof['all_other_leaves_exact'])
        self.assertEqual(proof['existing_absolute_bound'],1e-10)

    def test_other_scalar_change_of_one_e_minus16_is_rejected(self):
        actual = dict(objective=3.,other_scalar=0.,nested=dict(value=[1.,2.]))
        saved = copy.deepcopy(actual);saved['other_scalar']=1e-16
        with self.assertRaises(AssertionError):
            repaired.compare_finite_objective_only(actual,saved,1e-10)

    def test_objective_change_over_existing_bound_is_rejected(self):
        actual = dict(objective=3.,other_scalar=0.,nested=dict(value=[1.,2.]))
        saved = copy.deepcopy(actual);saved['objective']+=1.01e-10
        with self.assertRaises(AssertionError):
            repaired.compare_finite_objective_only(actual,saved,1e-10)
        with self.assertRaises(AssertionError):
            repaired.compare_finite_objective_only(actual,actual,2e-10)


if __name__ == '__main__':
    unittest.main()
