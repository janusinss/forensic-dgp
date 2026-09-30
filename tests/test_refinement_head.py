import unittest
import torch
from refinement_head import RefinementHead
from scripts.compare_pixel_heads_vm import PixelHead


class RefinementTests(unittest.TestCase):
    def setUp(self):
        torch.set_num_threads(2)
        torch.manual_seed(42)
        self.parent=PixelHead(3)
        self.f=torch.randn(1,256,64,64)
        self.rgb=torch.rand(1,3,256,256)

    def test_initial_exact_parent_and_frozen_backbone(self):
        model=RefinementHead(self.parent.state_dict(),use_rgb=True).train()
        self.assertFalse(model.parent.training)
        self.assertTrue(all(not p.requires_grad for p in model.parent.parameters()))
        with torch.no_grad():
            self.assertTrue(torch.equal(model(self.f,self.rgb),self.parent(self.f)))

    def test_control_invariant_to_rgb_after_nonzero_residual(self):
        model=RefinementHead(self.parent.state_dict(),use_rgb=False)
        torch.nn.init.normal_(model.output.weight,std=.01)
        self.assertTrue(torch.equal(model(self.f,self.rgb),model(self.f,1-self.rgb)))

    def test_rgb_branch_can_affect_output_and_gradients(self):
        model=RefinementHead(self.parent.state_dict(),use_rgb=True)
        torch.nn.init.normal_(model.output.weight,std=.01)
        self.assertFalse(torch.equal(model(self.f,self.rgb),model(self.f,1-self.rgb)))
        model(self.f,self.rgb).mean().backward()
        self.assertTrue(all(p.grad is None for p in model.parent.parameters()))
        self.assertGreater(float(model.output.weight.grad.abs().sum()),0)

    def test_invalid_input_rejected(self):
        model=RefinementHead(self.parent.state_dict(),use_rgb=True)
        with self.assertRaises(ValueError):model(self.f,self.rgb[:,:,::2,::2])
        with self.assertRaises(ValueError):model(self.f,self.rgb*0+float('nan'))
