"""Forward-only and file-contract checks. Never perform training/backward locally."""
import copy
import importlib.util
import json
from pathlib import Path
import unittest
from unittest.mock import patch
import sys

import numpy as np
import torch

ROOT=Path(__file__).resolve().parents[1]
sys.path.insert(0,str(ROOT))
sys.path.insert(0,str(ROOT/'scripts'))
from cctv_dgp_pilot import identity_crop,grid112,exported_pixel_metrics,qualifies,verify_bundle,sha,write
from run_cctv_dgp_pilot_vm import require_vm
from audit_cctv_dgp_pilot_results import safe_result_path,same


class PilotContracts(unittest.TestCase):
    def test_no_cuda_guard_before_model_or_output(self):
        with patch('torch.cuda.is_available',return_value=False),patch('torch.optim.Adam') as optimizer:
            with self.assertRaisesRegex(RuntimeError,'CUDA required'):
                require_vm(ROOT)
            optimizer.assert_not_called()

    def test_non_vm_host_rejected_even_with_cuda(self):
        with patch('torch.cuda.is_available',return_value=True),patch('run_cctv_dgp_pilot_vm.sys.platform','win32'):
            with self.assertRaisesRegex(RuntimeError,'Linux VM'):
                require_vm(ROOT)

    def test_fixed_identity_crop_masks_uncaptured_context(self):
        torch.set_num_threads(2)
        matrix=[[1.,0.,-60.],[0.,1.,-70.]]
        grid=torch.from_numpy(grid112(matrix))[None]
        rng=np.random.default_rng(9);rgb=torch.tensor(rng.random((1,3,256,256)),dtype=torch.float32)
        mask=torch.zeros(1,1,256,256);mask[:,:,80:180,80:150]=1
        # Integer affine is intentionally used to test exact mask/context behavior.
        altered=rgb*mask+(1-rgb)*(1-mask)
        with torch.no_grad():
            a,b=identity_crop(rgb,mask,grid),identity_crop(altered,mask,grid)
        np.testing.assert_allclose(a.numpy(),b.numpy(),atol=1e-7,rtol=0)
        np.testing.assert_allclose(a[:,:,20:100,20:90].numpy(),rgb[:,:,90:170,80:150].numpy(),atol=1e-7,rtol=0)
        self.assertEqual(float(a[0,0,0,0]),float(torch.tensor(128/255,dtype=torch.float32)))

    def test_metric_basis_ignores_padding_and_marks_perfect(self):
        target=np.full((256,256,3),128,np.uint8);mask=np.zeros((256,256),bool);mask[50:200,70:180]=True
        prediction=target.copy();prediction[~mask]=0
        metrics=exported_pixel_metrics(prediction,target,mask)
        self.assertEqual(metrics['MSE'],0.);self.assertIsNone(metrics['PSNR']);self.assertTrue(metrics['perfect_match'])
        self.assertAlmostEqual(metrics['SSIM'],1.,places=6)

    def test_selection_rejects_source_or_clear_regression_and_missing_pairs(self):
        groups=['degraded','clear','dataset/asian_faces/compound_lr24','dataset/thumbnails128x128/clear']
        baseline={g:{'cases':10,'identity_pairs':10,'MSE':.01,'SSIM':.8,'ArcFace_observed_fixed':.5} for g in groups}
        candidate=copy.deepcopy(baseline);candidate['degraded']['MSE']=.009
        self.assertTrue(qualifies(candidate,baseline,baseline))
        for group,metric,value in [('clear','MSE',.0101),('dataset/asian_faces/compound_lr24','ArcFace_observed_fixed',.49),
                                   ('clear','SSIM',.79),('degraded','identity_pairs',9),('degraded','MSE',float('nan'))]:
            bad=copy.deepcopy(candidate);bad[group][metric]=value
            self.assertFalse(qualifies(bad,baseline,baseline))
        bad=copy.deepcopy(candidate);bad.pop('clear');self.assertFalse(qualifies(bad,baseline,baseline))
        perfect=copy.deepcopy(baseline);perfect['degraded']['MSE']=0
        self.assertFalse(qualifies(perfect,perfect,perfect))

    def test_invalid_identity_transform_rejected(self):
        with self.assertRaises(ValueError):
            grid112([[0,0,0],[0,0,0]])

    def test_result_paths_and_numeric_tampering_rejected(self):
        for name in ['../escape.png','/absolute.png','C:/absolute.png','outputs\\escape.png']:
            with self.assertRaises(ValueError):
                safe_result_path(ROOT,name)
        with self.assertRaises(ValueError):
            same({'MSE':float('nan')},{'MSE':.1})

    def test_changed_protocol_rejected_before_asset_load(self):
        with patch.object(Path,'read_text',return_value='0'*64+'\n'),patch('cctv_dgp_pilot.sha',return_value='1'*64):
            with self.assertRaisesRegex(ValueError,'fingerprint'):
                verify_bundle(ROOT/'scratch/nonexistent_protocol_fixture')


if __name__=='__main__':
    unittest.main()
