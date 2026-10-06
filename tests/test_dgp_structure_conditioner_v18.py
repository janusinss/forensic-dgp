"""Inference-only interface regressions; no local graph/backward/update."""
import sys
from pathlib import Path
import unittest

import torch

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
from dgp_structure_conditioner_v18 import DGPStructureResidualHead


class StructureInterfaceTests(unittest.TestCase):
    def setUp(self):
        torch.set_num_threads(4)
        torch.manual_seed(20261005)
        self.head = DGPStructureResidualHead()
        self.camera = torch.rand(1, 3, 256, 256)
        self.base = torch.rand_like(self.camera)
        self.prior = torch.randn(1, 256, 64, 64)

    def test_zero_residual_exact_dgp_even_at_range_boundaries(self):
        self.base[:, :, 0] = 0
        self.base[:, :, -1] = 1
        before = {k: v.clone() for k, v in self.head.state_dict().items()}
        with torch.inference_mode():
            actual = self.head(self.camera, self.base, self.prior)
            ablated = self.head(self.camera, self.base, self.prior, prior_ablation=True)
        self.assertTrue(torch.equal(actual, self.base))
        self.assertTrue(torch.equal(ablated, self.base))
        self.assertTrue(all(torch.equal(before[k], v) for k, v in self.head.state_dict().items()))
        self.assertFalse(actual.requires_grad)

    def test_inputs_must_be_detached_valid_rgb_and_feature_schema(self):
        for camera, base, prior in [(self.camera[:, :, :128], self.base, self.prior),
            (self.camera, self.base.double(), self.prior),
            (self.camera, self.base, self.prior[:, :, :32]),
            (self.camera.clone().requires_grad_(), self.base, self.prior)]:
            with self.assertRaises(ValueError):
                self.head(camera, base, prior)
        self.prior[0, 0, 0, 0] = float('nan')
        with self.assertRaises(ValueError):
            self.head(self.camera, self.base, self.prior)

    def test_local_training_attempt_rejected_before_forward(self):
        with self.assertRaises(RuntimeError):
            self.head.train(True)
        self.head.requires_grad_(True)
        with self.assertRaisesRegex(RuntimeError, 'explicit existing-L4'):
            self.head(self.camera, self.base, self.prior)
        self.assertFalse(self.head.training)
        self.assertIsNone(self.head._training_root)
        self.assertTrue(all(p.grad is None for p in self.head.parameters()))

    def test_vm_authorization_rejects_foreign_root_without_enabling_parameters(self):
        with self.assertRaises(RuntimeError):
            self.head.enable_vm_training(Path('/not-forensic-dgp'))
        self.assertFalse(self.head.training)
        self.assertTrue(all(not p.requires_grad for p in self.head.parameters()))
        self.assertIsNone(self.head._training_root)

    def test_spatial_prior_and_visible_input_reach_the_residual(self):
        # Assigned inference probe weights isolate connectivity, not learned fitting.
        with torch.no_grad():
            self.head.final.weight.normal_(0, .01)
        with torch.inference_mode():
            actual = self.head(self.camera, self.base, self.prior)
            ablated = self.head(self.camera, self.base, self.prior, prior_ablation=True)
            other_input = self.head(1 - self.camera, self.base, self.prior)
        self.assertGreater((actual - ablated).abs().max().item(), 1e-5)
        self.assertGreater((actual - other_input).abs().max().item(), 1e-5)
        self.assertGreater((actual - self.base).abs().max().item(), 1e-5)
        self.assertTrue(torch.isfinite(actual).all())
        self.assertGreaterEqual(actual.min().item(), 0)
        self.assertLessEqual(actual.max().item(), 1)
        self.assertTrue(all(p.grad is None for p in self.head.parameters()))


if __name__ == '__main__':
    unittest.main()
