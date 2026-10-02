import unittest

import torch

from pretrained_face_restoration import CodeFormerRestoration


class DummyNet(torch.nn.Module):
    def __init__(self):
        super().__init__()
        self.calls = []

    def forward(self, image, w, adain):
        self.calls.append((image.shape, w, adain))
        return (image,)


class RestorationAdapterTests(unittest.TestCase):
    def test_official_normalization_and_fidelity_arguments(self):
        net = DummyNet()
        adapter = CodeFormerRestoration(net)
        image = torch.full((1, 3, 64, 64), .3)
        output = adapter(image, fidelity=.5)
        self.assertEqual(output.shape, image.shape)
        self.assertTrue(torch.allclose(image, output, atol=1e-6))
        self.assertEqual(net.calls, [(torch.Size((1, 3, 512, 512)), .5, True)])
        self.assertFalse(output.requires_grad)

    def test_invalid_inputs_do_not_reach_network(self):
        net = DummyNet()
        adapter = CodeFormerRestoration(net)
        with self.assertRaises(ValueError):
            adapter(torch.zeros(1, 3, 32, 48))
        with self.assertRaises(ValueError):
            adapter(torch.full((1, 3, 64, 64), float("nan")))
        with self.assertRaises(ValueError):
            adapter(torch.zeros(1, 3, 64, 64), fidelity=2)
        self.assertEqual(net.calls, [])

    def test_nonfinite_output_is_rejected(self):
        class BadNet(DummyNet):
            def forward(self, image, w, adain):
                return (torch.full_like(image, float("nan")),)
        with self.assertRaises(FloatingPointError):
            CodeFormerRestoration(BadNet())(torch.zeros(1, 3, 64, 64))


if __name__ == "__main__":
    unittest.main()
