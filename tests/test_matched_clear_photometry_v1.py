"""Synthetic diagnostic identities only; never run a restoration or training."""
import ast
import importlib.util
from pathlib import Path
import sys
import unittest
import numpy as np

ROOT=Path(__file__).resolve().parents[1]
spec=importlib.util.spec_from_file_location('photometric_diagnosis_under_test',ROOT/'scripts/diagnose_matched_clear_photometry_v1.py')
D=importlib.util.module_from_spec(spec);spec.loader.exec_module(D)
BLOCKER=D.Blocker();sys.meta_path.insert(0,BLOCKER)


class PhotometricIdentities(unittest.TestCase):
    def test_block_RGB_means_preserve_known_grid_and_sign(self):
        value=np.zeros((256,256,3),np.float64);value[:16,:16]=[1.,-2.,3.]
        actual=D.block_mean(value,np=np)
        self.assertEqual(actual.shape,(16,16,3));np.testing.assert_array_equal(actual[0,0],[1.,-2.,3.])
        self.assertEqual(np.count_nonzero(actual),3)

    def test_global_constant_delta_has_zero_spatial_energy_and_global_attribution(self):
        original=np.full((256,256,3),.25,np.float32);current=np.full_like(original,.26);target=np.full_like(original,.5)
        result,delta,error=D.field_diagnosis(original,current,target,np=np)
        self.assertAlmostEqual(result['constant_mean_fraction_of_output_delta_energy'],1.,places=12)
        self.assertLess(result['remaining_zero_mean_spatial_delta_energy'],1e-25)
        self.assertLess(abs(result['raw_MAE_attribution']['remaining_spatial_delta_attribution']),1e-14)
        self.assertLess(result['raw_MAE_attribution']['global_channel_mean_delta_attribution'],0)
        self.assertEqual(error.shape,(256,256));self.assertEqual(delta.shape,(256,256,3))

    def test_balanced_spatial_delta_has_zero_global_energy_and_spatial_attribution(self):
        original=np.full((256,256,3),.5,np.float32);current=original.copy();current[:128]+=.125;current[128:]-=.125
        result,_,_=D.field_diagnosis(original,current,original,np=np)
        self.assertEqual(result['constant_mean_fraction_of_output_delta_energy'],0.)
        self.assertEqual(result['raw_MAE_attribution']['global_channel_mean_delta_attribution'],0.)
        self.assertGreater(result['raw_MAE_attribution']['remaining_spatial_delta_attribution'],0.)

    def test_zero_delta_remains_finite_zero_every_attribution(self):
        original=np.full((256,256,3),.5,np.float32)
        result,_,_=D.field_diagnosis(original,original,original,np=np)
        self.assertEqual(result['mean_square_output_delta'],0.)
        self.assertEqual(result['constant_mean_fraction_of_output_delta_energy'],0.)
        self.assertEqual(result['top26_of256_blocks_fraction_of_output_delta_energy'],0.)
        self.assertEqual(result['raw_MAE_attribution']['delta_positive_worse'],0.)

    def test_nonlinear_MAE_symmetric_parts_sum_exact_delta_without_causal_claim(self):
        rng=np.random.default_rng(2550);base=rng.random((256,256,3));change=rng.normal(0,.01,base.shape);target=rng.random(base.shape)
        actual=D.symmetric_attribution(base,change,target,lambda x:float(np.abs(x).mean()),np=np)
        self.assertAlmostEqual(actual['global_channel_mean_delta_attribution']+actual['remaining_spatial_delta_attribution'],actual['delta_positive_worse'],places=15)
        self.assertTrue(actual['counterfactual_values_are_unclipped_scalar_diagnostics_only'])

    def test_integer_error_all_worse_counts_and_channel_sum_exact(self):
        original=np.zeros((256,256,3),np.uint8);current=np.ones_like(original)
        actual,diff,movement=D.integer_error_decomposition(original,current,original,np=np)
        self.assertEqual(actual['net_target_error_code_units'],256*256*3)
        self.assertEqual(actual['worsened_error_code_units'],256*256*3)
        self.assertEqual(actual['improved_error_code_units'],0)
        self.assertEqual(actual['brightened_component_error_contribution'],256*256*3)
        self.assertEqual(sum(c['net_target_error_code_units']for c in actual['by_channel']),actual['net_target_error_code_units'])
        self.assertTrue((diff==1).all());self.assertTrue((movement==1).all())

    def test_integer_cross_target_equal_errors_and_mixed_signs(self):
        before=np.zeros((256,256,3),np.uint8);after=before.copy();target=np.zeros_like(before)
        before[0,0]=[0,2,0];after[0,0]=[2,0,0];target[0,0]=[1,1,0]
        actual,_,_=D.integer_error_decomposition(before,after,target,np=np)
        self.assertEqual(actual['net_target_error_code_units'],0)
        self.assertEqual(actual['changed_but_equal_error_components'],2)
        self.assertEqual(actual['changed_components'],2)
        self.assertEqual(actual['unchanged_component_error_contribution'],0)

    def test_integer_decomposition_refuses_float_and_wrong_geometry(self):
        value=np.zeros((256,256,3),np.uint8)
        for bad in(value.astype(np.float32),value[:128]):
            with self.assertRaises(ValueError):D.integer_error_decomposition(bad,value,value,np=np)

    def test_signed_diagnostic_map_white_zero_red_positive_blue_negative(self):
        value=np.zeros((256,256),np.float64);value[0,0]=1.;value[0,1]=-1.;value[0,2]=2.
        result=D.signed_map(value,1.,np=np)
        np.testing.assert_array_equal(result[1,1],[255,255,255]);np.testing.assert_array_equal(result[0,0],[210,38,38])
        np.testing.assert_array_equal(result[0,1],[38,80,210]);np.testing.assert_array_equal(result[0,2],[210,38,38])
        self.assertEqual(result.dtype,np.uint8)

    def test_diagnostic_source_has_no_neural_execution(self):
        tree=ast.parse(Path(D.__file__).read_bytes(),feature_version=(3,10));calls={n.func.attr for n in ast.walk(tree)if isinstance(n,ast.Call)and isinstance(n.func,ast.Attribute)}
        self.assertFalse({'backward','step','train','forward','cuda','load_state_dict'}&calls)
        self.assertEqual(BLOCKER.attempted,[]);self.assertFalse({'torch','tensorflow','jax'}&{n.split('.')[0]for n in sys.modules})


if __name__=='__main__':unittest.main()
