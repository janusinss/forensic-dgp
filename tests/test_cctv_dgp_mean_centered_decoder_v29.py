"""Forward-only tests of a fixed input-conditioned projection and retained gates."""
import importlib.util
from pathlib import Path
import sys
import unittest

import numpy as np
import torch

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT/'outputs/cctv_dgp_active_original_decoder_vm_v28'))
sys.path.insert(0, str(ROOT/'scripts'))
from cctv_dgp_mean_centered_decoder_v29 import center_observed_delta
from audit_cctv_dgp_active_original_decoder_v28_return import capacity, validate_members


class Projection(unittest.TestCase):
    @torch.no_grad()
    def test_zero_delta_preserves_every_baseline_bit(self):
        base = torch.linspace(0, 1, 48).reshape(1, 3, 4, 4)
        image = torch.full_like(base, .7)
        mask = torch.ones(1, 1, 4, 4); mask[:, :, 0] = 0
        answer = center_observed_delta(base, base, image, mask)
        self.assertTrue(torch.equal(answer, torch.where(mask.bool(), base, image)))

    @torch.no_grad()
    def test_constant_RGB_shift_is_removed(self):
        base = torch.full((2, 3, 4, 4), .4)
        shift = torch.tensor([.125, -.125, .25]).reshape(1, 3, 1, 1)
        image = base.clone(); mask = torch.ones(2, 1, 4, 4)
        answer = center_observed_delta(base+shift, base, image, mask)
        self.assertTrue(torch.allclose(answer, base, rtol=0, atol=torch.finfo(torch.float32).eps))

    @torch.no_grad()
    def test_structure_and_unobserved_pixels_survive(self):
        base = torch.full((1, 3, 4, 4), .5)
        delta = torch.zeros_like(base); delta[:, :, :, 1] = .125; delta[:, :, :, 2] = -.125
        mask = torch.ones(1, 1, 4, 4); mask[:, :, :, 0] = 0
        image = torch.full_like(base, .8)
        answer = center_observed_delta(base+delta, base, image, mask)
        self.assertTrue(torch.equal(answer[:, :, :, 0], image[:, :, :, 0]))
        self.assertTrue(torch.equal(answer[:, :, :, 1:], (base+delta)[:, :, :, 1:]))

    @torch.no_grad()
    def test_clipping_is_explicit_not_a_mean_preservation_claim(self):
        base = torch.tensor([0., .5, .5, .5]).repeat(3).reshape(1, 3, 1, 4)
        pred = torch.tensor([0., 1., .5, .5]).repeat(3).reshape(1, 3, 1, 4)
        answer = center_observed_delta(pred, base, base, torch.ones(1, 1, 1, 4))
        self.assertTrue(bool(((answer >= 0) & (answer <= 1)).all()))
        self.assertGreater(float((answer-base).mean()), 0)

    @torch.no_grad()
    def test_empty_or_nonbinary_support_rejected(self):
        a = torch.full((1, 3, 2, 2), .5)
        for mask in [torch.zeros(1, 1, 2, 2), torch.full((1, 1, 2, 2), .5)]:
            with self.assertRaises(AssertionError): center_observed_delta(a, a, a, mask)

    def test_local_derivatives_rejected_before_calculation(self):
        a = torch.full((1, 3, 2, 2), .5)
        with self.assertRaises(RuntimeError):
            center_observed_delta(a, a, a, torch.ones(1, 1, 2, 2))


class RetainedGates(unittest.TestCase):
    def group(self):
        return {'cases': 10, 'MSE': .1, 'SSIM': .8, 'ArcFace_observed_fixed': .8,
                'landmark_high_frequency_MSE': .1, 'constant_mean_shift_only_MSE': .1}

    def test_structure_gain_cannot_hide_preservation_failure(self):
        baseline = {'degraded': self.group(), 'source/degraded': self.group(), 'source/clear': self.group()}
        final = {key: dict(value) for key, value in baseline.items()}
        for key in ['degraded', 'source/degraded']:
            final[key].update(MSE=.09, landmark_high_frequency_MSE=.08)
        final['source/clear']['SSIM'] -= .0001
        failures, gain, _, _, passed = capacity(baseline, final)
        self.assertGreater(gain, .1); self.assertFalse(passed)
        self.assertTrue(any(row['metric'] == 'SSIM' for row in failures))

    def test_brightness_gate_is_not_weakened(self):
        baseline = {'degraded': self.group(), 'source/degraded': self.group()}
        final = {key: dict(value, MSE=.09, landmark_high_frequency_MSE=.08, constant_mean_shift_only_MSE=.095) for key, value in baseline.items()}
        failures, _, _, fraction, passed = capacity(baseline, final)
        self.assertGreater(fraction, .2); self.assertFalse(passed)
        self.assertTrue(any(row['metric'] == 'brightness_gain_fraction' for row in failures))


if __name__ == '__main__':
    unittest.main()
