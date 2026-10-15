"""Saved-vector regression fixtures; no models, parameter installation or gradients."""
import copy
import json
from pathlib import Path
import unittest

import numpy as np

from audit_cctv_dgp_decoder_hinge_repair_v1_return_r1 import (
    exact_reserve, verify_projection_journal, ORIGINAL_PROJECTION,
)

ROOT=Path(__file__).resolve().parent.parent


class ReserveReference(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        source=ROOT/'outputs/cctv_dgp_spatial_readout_v1_return/outputs/spatial_readout_v1'
        out=ROOT/'outputs/cctv_dgp_decoder_hinge_repair_v1_return/outputs/decoder_hinge_repair_v1'
        cls.initial=np.load(source/'A_initial_vector.npy',allow_pickle=False)
        with np.load(out/'trajectory/update_001.npz',allow_pickle=False) as pack:
            cls.proposal,cls.actual=[pack[k].copy() for k in ['unconstrained','actual']]
        cls.row=json.loads((out/'trajectory/trace.jsonl').read_text().splitlines()[0])['projection']
        cls.radius=cls.row['radius']

    def test_original_exact_numpy_metadata_comparison_reproduces_stop(self):
        with self.assertRaises(AssertionError):
            ORIGINAL_PROJECTION(self.initial,self.proposal,self.actual,self.row,self.radius)

    def test_actual_record_matches_deterministic_exact_reference(self):
        value,proof=exact_reserve(self.initial)
        self.assertEqual(value,self.row['float32_roundoff_reserve'])
        self.assertEqual(value.hex(),'0x1.d8874314ebff3p-18')
        self.assertEqual(proof['decimal_precisions'],[110,160])
        verify_projection_journal(self.initial,self.proposal,self.actual,self.row,self.radius)

    def test_even_one_ULP_metadata_drift_is_rejected(self):
        for destination in [-np.inf,np.inf]:
            row=copy.deepcopy(self.row)
            row['float32_roundoff_reserve']=float(np.nextafter(row['float32_roundoff_reserve'],destination))
            with self.assertRaises(AssertionError):
                verify_projection_journal(self.initial,self.proposal,self.actual,row,self.radius)

    def test_changed_weight_bytes_and_radius_are_rejected(self):
        changed=self.actual.copy();changed[0]=np.nextafter(changed[0],np.float32(np.inf))
        with self.assertRaises(AssertionError):
            verify_projection_journal(self.initial,self.proposal,changed,self.row,self.radius)
        with self.assertRaises(AssertionError):
            verify_projection_journal(self.initial,self.proposal,self.actual,self.row,self.radius*1.01)

    def test_nonfinite_and_wrong_bound_reference_are_rejected(self):
        for change in [dict(float32_roundoff_reserve=float('nan')),dict(bound_reference='previous update')]:
            row=self.row|change
            with self.assertRaises(AssertionError):
                verify_projection_journal(self.initial,self.proposal,self.actual,row,self.radius)

    def test_short_power_of_two_reference(self):
        value,_=exact_reserve(np.array([1.],np.float32))
        self.assertEqual(value,2*float(np.spacing(np.float32(1.))))


if __name__=='__main__':unittest.main(verbosity=2)
