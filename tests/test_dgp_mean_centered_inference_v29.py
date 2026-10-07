"""Safety and projection contracts without constructing a neural restoration net."""
import unittest
from unittest.mock import patch

import numpy as np
import torch
from torch import nn

from dgp_mean_centered_inference_v29 import (
    MeanCenteredDGPInferenceV29, load_mean_centered_dgp_v29, project_observed_delta,
)


class Stub(nn.Module):
    def __init__(self, offset=0):
        super().__init__(); self.offset = offset; self.calls = 0; self.eval()

    def forward(self, image):
        self.calls += 1
        return (image + self.offset).clamp(0, 1)


class Contracts(unittest.TestCase):
    def setUp(self):
        self.x = torch.full((2, 3, 256, 256), .5, dtype=torch.float32)
        self.s = torch.ones((2, 1, 256, 256), dtype=torch.bool)

    def test_initial_exact_and_no_autograd(self):
        out = project_observed_delta(self.x, self.x, self.x, self.s)
        self.assertTrue(torch.equal(out, self.x)); self.assertFalse(out.requires_grad)
        self.assertIsNone(out.grad_fn)

    def test_independent_case_channel_means_and_observed_support(self):
        self.s[1, :, :, :64] = False
        p = self.x.clone(); p[0, 0, :, 128:] = .625; p[1, 2, :, 192:] = .75
        actual = project_observed_delta(p, self.x, self.x, self.s).numpy()
        for i in range(2):
            support = self.s[i, 0].numpy()
            delta = (p[i].numpy() - self.x[i].numpy()).astype(np.float64)
            expected = self.x[i].numpy().astype(np.float64) + delta - delta[:, support].mean(1)[:, None, None]
            expected = np.where(support[None], np.clip(expected, 0, 1), self.x[i].numpy())
            self.assertLessEqual(float(np.abs(actual[i] - expected).max()), 1e-7)
        self.assertTrue(torch.equal(torch.from_numpy(actual[1, :, :, :64]), self.x[1, :, :, :64]))

    def test_saturation_is_explicit_and_bounded(self):
        baseline = self.x.clone(); baseline[:, :, :, :128] = 1
        candidate = baseline.clone(); candidate[:, :, :, 128:] = 0
        out = project_observed_delta(candidate, baseline, self.x, self.s)
        self.assertEqual(float(out.max()), 1); self.assertEqual(float(out.min()), .25)
        self.assertNotEqual(float((out - baseline).mean()), 0)

    def test_invalid_support_before_any_neural_call(self):
        original, candidate = Stub(), Stub(.02)
        model = MeanCenteredDGPInferenceV29(original, candidate)
        for invalid in (self.s.float() * .2, torch.zeros_like(self.s), self.s[:1]):
            with self.assertRaises(ValueError): model(self.x, invalid)
        self.assertEqual(original.calls + candidate.calls, 0)

    def test_nonfinite_or_wrong_scale_before_neural_call(self):
        original, candidate = Stub(), Stub()
        model = MeanCenteredDGPInferenceV29(original, candidate)
        for invalid in (self.x * 255, self.x.double(), self.x * float('nan')):
            with self.assertRaises(ValueError): model(invalid, self.s)
        self.assertEqual(original.calls + candidate.calls, 0)

    def test_training_request_stays_frozen_and_two_required_forwards(self):
        original, candidate = Stub(), Stub(.02)
        model = MeanCenteredDGPInferenceV29(original, candidate)
        model.train(True); out = model(self.x, self.s)
        self.assertTrue(all(not m.training for m in model.modules()))
        self.assertFalse(out.requires_grad); self.assertEqual((original.calls, candidate.calls), (1, 1))
        self.assertTrue(torch.equal(out, self.x))

    def test_changed_checkpoint_rejected_before_model_construction(self):
        with patch('dgp_mean_centered_inference_v29.sha', return_value='0' * 64), patch('dgp_mean_centered_inference_v29.load_frozen_dgp_restorer') as load:
            with self.assertRaises(ValueError): load_mean_centered_dgp_v29('unused-original', 'unused-candidate')
            load.assert_not_called()


if __name__ == '__main__': unittest.main()
