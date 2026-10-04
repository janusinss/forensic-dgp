"""Boundary and fidelity checks for the inference-only DGP adapter; no backward."""
from pathlib import Path
import sys
import unittest
from unittest.mock import patch

import numpy as np
import torch

ROOT=Path(__file__).resolve().parents[1]
sys.path.insert(0,str(ROOT))
from dgp_face_restoration import (DGPFaceRestoration,PHASE3_SHA256,load_dgp_restorer,
                                  prepare_crop,png_rgb,restore_crop)


class CountingNet(torch.nn.Module):
    def __init__(self):
        super().__init__();self.calls=0;self.norm=torch.nn.BatchNorm2d(3)

    def forward(self,image):
        self.calls+=1
        return self.norm(image).clamp(0,1)


class DGPAdapterChecks(unittest.TestCase):
    def test_adapter_stays_frozen_even_when_train_requested(self):
        torch.set_num_threads(2)
        net=CountingNet();model=DGPFaceRestoration(net)
        before={k:v.clone() for k,v in net.state_dict().items()}
        model.train(True);x=torch.ones(1,3,256,256,requires_grad=True)*.4
        y=model(x)
        self.assertFalse(model.training);self.assertFalse(net.norm.training)
        self.assertFalse(y.requires_grad)
        self.assertTrue(all(torch.equal(v,before[k]) for k,v in net.state_dict().items()))
        self.assertTrue(all(not p.requires_grad and p.grad is None for p in model.parameters()))

    def test_invalid_batch_and_values_do_not_reach_network(self):
        net=CountingNet();model=DGPFaceRestoration(net)
        for x in [torch.zeros(0,3,256,256),torch.zeros(1,3,128,128),torch.zeros(1,1,256,256),
                  torch.full((1,3,256,256),float('nan')),torch.ones(1,3,256,256)*1.01,
                  torch.zeros(1,3,256,256,dtype=torch.float64)]:
            with self.assertRaises(ValueError):
                model(x)
        self.assertEqual(net.calls,0)

    def test_invalid_outputs_raise_without_silent_clamp_or_resize(self):
        class BadNet(torch.nn.Module):
            def __init__(self,value):
                super().__init__();self.value=value
            def forward(self,image):
                return torch.full_like(image,self.value)
        for value in [float('nan'),1.2,-.2]:
            with self.assertRaises(FloatingPointError):
                DGPFaceRestoration(BadNet(value))(torch.zeros(1,3,256,256))

    def test_aspect_padding_and_mask_support_are_not_stretched_or_dilated(self):
        rgb=np.full((128,64,3),[40,70,90],np.uint8);mask=np.zeros((128,64),np.uint8)
        mask[40:48,12:20]=1
        canvas,observed,removal,meta=prepare_crop(rgb,mask)
        self.assertEqual(canvas.shape,(256,256,3));self.assertEqual(meta['observed_bounds_256'],[64,0,192,256])
        self.assertEqual(int(observed.sum()),128*256)
        self.assertEqual(int(removal.sum()),8*8*4)
        self.assertTrue(np.array_equal(canvas[:,0],np.full((256,3),128,np.uint8)))
        self.assertFalse(removal[~observed].any())
        for invalid in [mask*.5,np.zeros((64,64)),[],np.ones((128,64))*2]:
            with self.assertRaises(ValueError):
                prepare_crop(rgb,invalid)

    def test_missing_or_changed_checkpoint_has_no_model_fallback(self):
        with patch('models.DGPSynthesizer') as net:
            with self.assertRaises(FileNotFoundError):
                load_dgp_restorer(ROOT/'scratch/absent-dgp-model.pth')
            with patch('dgp_face_restoration.sha',return_value='0'*64):
                with self.assertRaisesRegex(ValueError,'fingerprint'):
                    load_dgp_restorer(ROOT/'checkpoints/dgp_zamboanga_final.pth')
            with self.assertRaisesRegex(ValueError,'SHA256'):
                load_dgp_restorer(ROOT/'checkpoints/dgp_zamboanga_final.pth',expected_sha256=None)
            net.assert_not_called()

    def test_nonfinite_checkpoint_is_rejected_before_model_construction(self):
        with patch('models.DGPSynthesizer') as net,patch('dgp_face_restoration.sha',return_value=PHASE3_SHA256),\
                patch('torch.load',return_value={'weight':torch.tensor(float('nan'))}):
            with self.assertRaisesRegex(ValueError,'Nonfinite'):
                load_dgp_restorer('does-not-need-to-exist')
            net.assert_not_called()

    def test_raw_and_observed_output_are_distinct_and_png_basis_declared(self):
        class Constant(torch.nn.Module):
            def forward(self,x):
                return torch.full_like(x,.3)
        rgb=np.full((80,40,3),100,np.uint8)
        result=restore_crop(DGPFaceRestoration(Constant()),rgb)
        self.assertEqual(result['raw_rgb'].shape,(256,256,3))
        self.assertTrue(np.all(result['raw_rgb']==np.float32(.3)))
        self.assertTrue(np.array_equal(result['observed_rgb'][~result['observed']],result['input'][~result['observed']].astype(np.float32)/255))
        self.assertTrue(np.array_equal(png_rgb(result['observed_rgb'])[~result['observed']],result['input'][~result['observed']]))
        self.assertEqual(int(png_rgb(result['raw_rgb'])[0,0,0]),76)


if __name__=='__main__':
    unittest.main()
