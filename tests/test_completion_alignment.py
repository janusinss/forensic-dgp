import unittest
import numpy as np
import torch
from completion_alignment import SelectiveEyeAlignment


class Fill(torch.nn.Module):
    def forward(self, x, m):
        return x*(1-m)+m*.4


class AlignmentTests(unittest.TestCase):
    def run_case(self, eyes):
        model=SelectiveEyeAlignment(Fill(),lambda image: eyes)
        x=torch.rand(1,3,64,64);m=torch.zeros(1,1,64,64);m[:,:,36:60,8:56]=1
        return model,x,m

    def test_large_crop_aligns_and_preserves_visible_pixels(self):
        model,x,m=self.run_case(np.array([[19,29],[45,29]]))
        y=model(x,m)
        self.assertEqual(model.events[-1]['decision'],'aligned')
        self.assertTrue(torch.equal(y*(1-m),x*(1-m)))
        self.assertTrue(torch.isfinite(y).all())

    def test_missing_covered_and_normal_eyes_fall_back(self):
        for eyes in [None,np.array([[19,40],[45,40]]),np.array([[24,30],[40,30]])]:
            model,x,m=self.run_case(eyes)
            self.assertTrue(torch.equal(model(x,m),Fill()(x,m)))
            self.assertEqual(model.events[-1]['decision'],'fallback')

    def test_cover_content_cannot_leak_through_warp(self):
        model,x,m=self.run_case(np.array([[19,29],[45,29]]))
        self.assertTrue(torch.equal(model(x*(1-m),m),model(x*(1-m)+m,m)))

    def test_empty_mask_bypasses_detection(self):
        model=SelectiveEyeAlignment(Fill(),lambda image: self.fail('detector called'))
        x=torch.rand(1,3,64,64)
        self.assertTrue(torch.equal(model(x,torch.zeros(1,1,64,64)),x))
