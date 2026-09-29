import unittest
import torch
from feature_detector import FrozenFeatureHead


class FeatureHeadTests(unittest.TestCase):
    def test_shapes_and_frozen_input(self):
        model = FrozenFeatureHead(8, 4)
        features = torch.randn(2, 8, 8, 8, requires_grad=True)
        logits = model(features, (32, 32))
        self.assertEqual(tuple(logits.shape), (2, 1, 32, 32))
        logits.square().mean().backward()
        self.assertIsNone(features.grad)
        self.assertTrue(any(p.grad is not None for p in model.parameters()))

    def test_rejects_wrong_feature_channels(self):
        with self.assertRaises(ValueError):
            FrozenFeatureHead(8, 4)(torch.zeros(1, 7, 8, 8), (32, 32))

    def test_learns_separable_feature_fixture(self):
        torch.manual_seed(42)
        torch.set_num_threads(1)
        x = torch.randn(2, 8, 8, 8)
        y = (x[:, :1] > 0).float()
        model = FrozenFeatureHead(8, 8)
        optimizer = torch.optim.AdamW(model.parameters(), lr=.03)
        initial = torch.nn.functional.binary_cross_entropy_with_logits(model(x, (8, 8)), y).item()
        for _ in range(60):
            optimizer.zero_grad()
            loss = torch.nn.functional.binary_cross_entropy_with_logits(model(x, (8, 8)), y)
            loss.backward()
            optimizer.step()
        self.assertLess(loss.item(), initial * .5)
