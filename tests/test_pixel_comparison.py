import unittest
from unittest.mock import patch
import torch
from feature_detector import FrozenFeatureHead
from scripts.compare_pixel_heads_vm import PixelHead,main

class PixelComparisonTests(unittest.TestCase):
    def test_center_initialization_matches_existing_head(self):
        torch.manual_seed(42)
        baseline=FrozenFeatureHead()
        x=torch.randn(2,256,8,8,requires_grad=True)
        for kernel in (1,3):
            head=PixelHead(kernel);head.initialize(baseline.state_dict())
            y=head(x,(32,32))
            torch.testing.assert_close(y,baseline(x,(32,32)),atol=2e-6,rtol=2e-5)
            y.sum().backward()
            self.assertIsNone(x.grad)
            self.assertIsNotNone(head.head[0].weight.grad)

    @patch('torch.cuda.is_available',return_value=False)
    def test_cpu_refused_before_data(self,_):
        with patch('pathlib.Path.read_text',side_effect=AssertionError('data read')):
            with self.assertRaisesRegex(RuntimeError,'VM GPU'):main('.')
