import unittest
from unittest.mock import patch
import torch
from scripts.compare_presence_heads_vm import PresenceHead, main

class PresenceComparisonTests(unittest.TestCase):
    def test_heads_detach_encoder_features(self):
        for grid in (1,4):
            x=torch.randn(2,256,8,8,requires_grad=True)
            head=PresenceHead(grid)
            y=head(x)
            self.assertEqual(tuple(y.shape),(2,))
            y.sum().backward()
            self.assertIsNone(x.grad)
            self.assertIsNotNone(head.linear.weight.grad)

    def test_spatial_head_retains_location(self):
        x=torch.zeros(1,256,8,8);x[0,0,0,0]=1
        shifted=torch.roll(x,4,2)
        torch.manual_seed(42)
        with torch.no_grad():
            global_head=PresenceHead(1);spatial=PresenceHead(4)
            self.assertTrue(torch.equal(global_head(x),global_head(shifted)))
            self.assertFalse(torch.equal(spatial(x),spatial(shifted)))

    @patch('torch.cuda.is_available',return_value=False)
    def test_cpu_execution_refused_before_filesystem(self,_):
        with patch('pathlib.Path.read_text',side_effect=AssertionError('data read')):
            with self.assertRaisesRegex(RuntimeError,'VM GPU'):main('.')
