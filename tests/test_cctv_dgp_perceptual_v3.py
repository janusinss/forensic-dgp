"""Forward-only checks for signed features preceding in-place activation."""
from pathlib import Path
import sys
import unittest

import torch
from torch import nn

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
from cctv_dgp_pilot import PilotPerceptual
from cctv_dgp_perceptual_v3 import PerceptualV3


def fixture(policy):
    model = PerceptualV3.__new__(PerceptualV3)
    nn.Module.__init__(model)
    layers = [nn.Identity() for _ in range(27)]
    for before, after in [(2, 3), (7, 8), (16, 17), (25, 26)]:
        layers[before] = nn.Conv2d(3, 3, 1, bias=False)
        with torch.no_grad():
            layers[before].weight.copy_(torch.eye(3).reshape(3, 3, 1, 1))
        layers[after] = nn.ReLU(inplace=True)
    model.features = nn.Sequential(*layers)
    model.register_buffer('mean', torch.zeros(1, 3, 1, 1))
    model.register_buffer('std', torch.ones(1, 3, 1, 1))
    model.feature_policy = policy
    return model.eval().requires_grad_(False)


class SignedFeatureChecks(unittest.TestCase):
    def setUp(self):
        self.input = torch.tensor([-1., .25, .5]).reshape(1, 3, 1, 1).expand(1, 3, 4, 4).clone()

    def test_signed_first_tap_survives_later_inplace_relu(self):
        model = fixture('preactivation')
        with torch.no_grad():
            features = model.taps(self.input)
        self.assertEqual(len(features), 4)
        torch.testing.assert_close(features[0], self.input, rtol=0, atol=0)
        self.assertLess(float(features[0].min()), 0.)

    def test_control_is_exactly_the_historical_postactivation_loss(self):
        model = fixture('postactivation')
        with torch.no_grad():
            actual = model.taps(self.input)
            expected = PilotPerceptual.taps(model, self.input)
        for generated, original in zip(actual, expected):
            torch.testing.assert_close(generated, original, rtol=0, atol=0)

    def test_invalid_feature_policy_is_rejected(self):
        with self.assertRaisesRegex(ValueError, 'feature policy'):
            fixture('invented').taps(self.input)


if __name__ == '__main__':
    unittest.main()
