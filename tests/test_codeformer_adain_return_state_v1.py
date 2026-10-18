import copy
import json
from pathlib import Path
import unittest
from types import SimpleNamespace

from research.codeformer_adain_gradient_isolation_v1 import return_state as audit


class SavedStateArithmetic(unittest.TestCase):
    def test_complete_reservation_plan_and_witness_precede_snapshot_and_training(self):
        p = json.loads(Path('outputs/cctv_codeformer_adain_gradient_isolation_v1_vm/protocol.json').read_text())
        rows, counts = audit.events(p)
        self.assertEqual(len(rows), 2073)
        self.assertEqual(counts['gradient_queries'], 9)
        self.assertEqual(counts['optimizer_updates'], 50)
        self.assertEqual(sum(counts[k] for k in counts if k.endswith('_forwards')), 1614)
        witnessed = [r for r in rows if r['before_call'] == 'route_witness_forwards']
        self.assertEqual(len(witnessed), 4)
        self.assertTrue(all(r['committed_update'] == 0 and r['counts']['optimizer_step_attempts'] == 0 for r in witnessed))
        self.assertLess(witnessed[-1]['event'], next(r['event'] for r in rows if r['before_call'] == 'snapshot_forwards'))

    @staticmethod
    def summary():
        return {'parameters': {'encoder.a': {'is_none': True, 'nonzero': False, 'l2': 0., 'maximum_absolute': 0., 'sha256': None},
            'fuse_convs_dict.a': {'is_none': False, 'nonzero': True, 'l2': 3., 'maximum_absolute': 3.,
                                 'sha256': '1' * 64, 'shape': [1], 'dtype': 'float32'},
            'position_emb': {'is_none': True, 'nonzero': False, 'l2': 0., 'maximum_absolute': 0., 'sha256': None}},
            'groups': {'encoder': {'tensors': 1, 'none': 1, 'nonzero': 0, 'l2': 0., 'maximum_absolute': 0.},
                       'predictor': {'tensors': 1, 'none': 1, 'nonzero': 0, 'l2': 0., 'maximum_absolute': 0.},
                       'fusion': {'tensors': 1, 'none': 0, 'nonzero': 1, 'l2': 3., 'maximum_absolute': 3.}},
            'full_gradient_vectors_exported': False,
            'norms_are_source_bound_VM_readouts_not_independent_gradient_replay': True}

    def test_none_gradient_is_not_a_connected_zero_gradient(self):
        value = self.summary()
        names = {key: SimpleNamespace(shape=(1,)) for key in value['parameters']}
        groups = audit.gradients(value, names)
        self.assertEqual(groups['encoder']['none'], 1)
        value['parameters']['encoder.a']['sha256'] = '0' * 64
        with self.assertRaises(ValueError): audit.gradients(value, names)

    def test_reported_group_norm_cannot_hide_tensor_norm(self):
        value = self.summary(); names = {key: SimpleNamespace(shape=(1,)) for key in value['parameters']}
        value['groups']['fusion']['l2'] = 2.
        with self.assertRaises(ValueError): audit.gradients(value, names)

    def test_nonfinite_norm_is_rejected(self):
        value = self.summary(); names = {key: SimpleNamespace(shape=(1,)) for key in value['parameters']}
        value['parameters']['fuse_convs_dict.a']['l2'] = float('nan')
        with self.assertRaises(ValueError): audit.gradients(value, names)


if __name__ == '__main__': unittest.main()
