"""Inference/guard tests only; never call backward or an optimizer locally."""
from pathlib import Path
import sys
import unittest

import torch

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
from dgp_direct_face_code_v14 import DirectCodeConditioner, render_codes
from third_party.codeformer.codeformer_arch import calc_mean_std


class DirectCodeBoundaryTests(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        torch.set_num_threads(4)
        torch.manual_seed(14)
        cls.head = DirectCodeConditioner()

    def test_zero_heads_preserve_logits_and_observed_statistics(self):
        image = torch.rand(1, 3, 256, 256)
        features = torch.randn(1, 256, 16, 16)
        logits = torch.randn(1, 256, 1024)
        with torch.inference_mode():
            predicted = self.head(image, image, features, logits)
        mean, std = calc_mean_std(features)
        self.assertTrue(torch.equal(predicted['logits'], logits))
        self.assertTrue(torch.equal(predicted['mean'], mean.flatten(1)))
        self.assertTrue(torch.equal(predicted['std'], std.flatten(1)))
        self.assertEqual(sum(p.numel() for p in self.head.parameters()), 2619808)

    def test_local_training_entry_leaves_parameters_disabled(self):
        with self.assertRaises(RuntimeError):
            self.head.enable_vm_training(ROOT)
        self.assertFalse(any(p.requires_grad for p in self.head.parameters()))
        with self.assertRaises(RuntimeError):
            self.head.train(True)

    def test_actual_trainer_rejects_local_before_creating_output(self):
        import uuid
        from scripts.train_cctv_dgp_direct_codes_v14_r2 import train, CONTEXT
        destination = ROOT / 'outputs' / ('v14_forbidden_local_' + uuid.uuid4().hex)
        with self.assertRaisesRegex(RuntimeError, 'only on forensic-dgp-thesis'):
            train(destination, ROOT / 'outputs/cctv_dgp_face_code_fit_vm_v12_r2', '0' * 64)
        self.assertFalse(destination.exists())
        self.assertIsNone(CONTEXT['out'])
        self.assertIsNone(CONTEXT['model'])
        self.assertEqual(CONTEXT['backwards'], 0)
        self.assertEqual(CONTEXT['updates'], 0)

    def test_renderer_rejects_invalid_codes_before_accessing_prior(self):
        with self.assertRaises(ValueError):
            render_codes(None, torch.full((1, 256), 1024, dtype=torch.int64))
        with self.assertRaises(ValueError):
            render_codes(None, torch.zeros((1, 256), dtype=torch.float32))


if __name__ == '__main__':
    unittest.main()
