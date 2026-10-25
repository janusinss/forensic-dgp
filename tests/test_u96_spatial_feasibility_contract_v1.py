import hashlib
from pathlib import Path
import sys
import unittest

from research.codeformer_u96_spatial_feasibility_v1 import contract as c


ROOT = Path(__file__).resolve().parents[1]


class Contract(unittest.TestCase):
    def test_actual_parent_selection_admitted_and_disjoint(self):
        data = (ROOT / 'outputs/cctv_codeformer_u96_decoder_readout_v1_vm/protocol.json').read_bytes()
        answer = c.select_parent_cases(data)
        self.assertEqual(len(answer['optimization']), 16)
        self.assertEqual(len(answer['monitors']), 32)
        self.assertEqual(len(answer['guards']), 24)
        self.assertEqual([row['reference_id'] for row in answer['optimization'][::4]],
                         list(c.OPTIMIZATION_REFERENCES))
        self.assertFalse(answer['independent_holdout'])
        self.assertFalse(answer['FINAL_or_native_pixels_used'])
        calls = c.operation_plan(answer)
        self.assertEqual(len(calls), 218)
        self.assertEqual([row['id'] for row in calls[72:74]], list(c.PARITY_IDS))
        self.assertEqual(sum(len(row['gradient_components']) for row in calls), 32)
        self.assertTrue(all(not row['gradient_components'] for row in calls
                            if row['kind'] in ('legacy_guard', 'parity', 'adaptive_TRAIN_monitor')))
        answer['optimization'][0]['role'] = 'corrupted'
        self.assertNotEqual(c.select_parent_cases(data)['optimization'][0]['role'], 'corrupted')

    def test_changed_parent_byte_or_type_is_refused(self):
        data = (ROOT / 'outputs/cctv_codeformer_u96_decoder_readout_v1_vm/protocol.json').read_bytes()
        for invalid in (data + b'\n', bytes([data[0] ^ 1]) + data[1:], bytearray(data)):
            with self.assertRaises(ValueError):
                c.select_parent_cases(invalid)

    def test_no_hidden_extra_forwards_or_gradient_queries(self):
        p = c.prospective_contract()
        limits = p['limits']
        self.assertEqual(16 + 32 + 24 + 2 + 2 * (16 + 32 + 24), 218)
        self.assertEqual(limits['full_forward_attempts'], 218)
        self.assertEqual(limits['gradient_queries'], 32)
        self.assertEqual(limits['proposal_applications'], 2)
        self.assertEqual(limits['optimizer_steps'], 0)
        self.assertEqual(p['extra_neural_hook_probes'], 0)

    def test_no_implicit_training_qualification_or_resource_mutability(self):
        p = c.prospective_contract()
        self.assertFalse(p['runtime_observed'])
        self.assertFalse(p['model_qualified'])
        self.assertFalse(p['worker_supervisor_exporter_installer_builder_and_guide_ready'])
        self.assertEqual(p['steps']['maximum_absolute_FP32_displacements'], [1e-5, 5e-5])
        self.assertFalse(p['steps']['cumulative_training'])
        p['limits']['full_forward_attempts'] = 100000
        self.assertEqual(c.prospective_contract()['limits']['full_forward_attempts'], 218)
        self.assertNotIn('torch', sys.modules)

    def test_unknown_or_failed_check_blocks_next_proposal(self):
        valid = dict(attempted=0, initial_checks_passed=True, direction_certified=True,
                     actual_FP32_direction_passed=True, prior_proposal_passed=True)
        self.assertTrue(c.permit_next_proposal(**valid))
        for key in ('initial_checks_passed', 'direction_certified',
                    'actual_FP32_direction_passed', 'prior_proposal_passed'):
            self.assertFalse(c.permit_next_proposal(**{**valid, key: False}))
            with self.assertRaises(ValueError):
                c.permit_next_proposal(**{**valid, key: None})
        self.assertFalse(c.permit_next_proposal(**{**valid, 'attempted': 2}))
        self.assertFalse(c.permit_next_proposal(**{**valid, 'attempted': 1,
                                                'prior_proposal_passed': False}))


if __name__ == '__main__':
    unittest.main()
