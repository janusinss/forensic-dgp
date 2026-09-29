import unittest
from unittest.mock import patch
import torch
from torch.nn import functional as F
from scripts.train_expanded_border_vm import border_loss, main


class BorderLossTests(unittest.TestCase):
    def test_control_equals_original_pixel_loss_and_gradient(self):
        x = torch.tensor([[[[-2., 1.], [.3, -.5]]], [[[.2, -.3], [1., 2.]]]], requires_grad=True)
        t = torch.tensor([[[[1., 1.], [0., 0.]]], [[[0., 0.], [0., 0.]]]])
        b = t.bool()
        p = x.sigmoid()
        dice = (1-(2*(p*t).sum((1,2,3))+1)/(p.sum((1,2,3))+t.sum((1,2,3))+1))*t.flatten(1).any(1)
        expected = F.binary_cross_entropy_with_logits(x,t)+dice.mean()
        actual = border_loss(x,t,b,1)
        torch.testing.assert_close(actual,expected)
        torch.testing.assert_close(torch.autograd.grad(actual,x,retain_graph=True)[0],torch.autograd.grad(expected,x)[0])

    def test_weight_two_uses_per_image_normalization(self):
        x=torch.tensor([[[[-2.,2.]]]]);t=torch.ones_like(x);b=torch.tensor([[[[True,False]]]])
        delta=border_loss(x,t,b,2)-border_loss(x,t,b,1)
        losses=F.binary_cross_entropy_with_logits(x,t,reduction='none').flatten()
        torch.testing.assert_close(delta,(2*losses[0]+losses[1])/3-losses.mean())
        with self.assertRaises(ValueError):border_loss(x,torch.zeros_like(t),b,2)

    def test_cpu_refused_before_files(self):
        with patch('torch.cuda.is_available',return_value=False),patch('pathlib.Path.resolve',side_effect=AssertionError('access')):
            with self.assertRaisesRegex(RuntimeError,'VM GPU'):main('.')
