"""Inference-boundary checks only; no pretrained model, backward or optimizer."""
import unittest

import torch
from torch import nn

from dgp_broader_code_conditioner_v16 import BroaderCodeConditioner


class BoundaryTests(unittest.TestCase):
    def test_zero_head_preserves_logits_without_gradients(self):
        torch.set_num_threads(4);torch.manual_seed(20261004)
        model=BroaderCodeConditioner()
        self.assertEqual(sum(p.numel() for p in model.parameters()),2422432)
        self.assertFalse(model.training);self.assertFalse(any(p.requires_grad for p in model.parameters()))
        image=torch.zeros(2,3,256,256);features=torch.zeros(2,256,16,16)
        base=torch.randn(2,256,1024)
        with torch.inference_mode():value=model(image,image,features,base)
        self.assertTrue(torch.equal(value,base));self.assertFalse(value.requires_grad)
        self.assertFalse(hasattr(model,'stats_projection'))

    def test_training_entry_and_bypass_rejected_locally(self):
        model=BroaderCodeConditioner()
        with self.assertRaisesRegex(RuntimeError,'local training'):model.train(True)
        with self.assertRaises(RuntimeError):model.enable_vm_training('.')
        nn.Module.train(model,True)
        with self.assertRaisesRegex(RuntimeError,'explicit VM'):
            model(torch.zeros(1,3,256,256),torch.zeros(1,3,256,256),
                  torch.zeros(1,256,16,16),torch.zeros(1,256,1024))

    def test_differentiable_cached_inputs_rejected(self):
        model=BroaderCodeConditioner()
        image=torch.zeros(1,3,256,256);features=torch.zeros(1,256,16,16,requires_grad=True)
        with self.assertRaisesRegex(ValueError,'detached'):
            model(image,image,features,torch.zeros(1,256,1024))


if __name__=='__main__':unittest.main()
