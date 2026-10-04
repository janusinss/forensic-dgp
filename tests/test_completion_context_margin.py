"""Context-support counterexamples; tensor fixtures only, no pretrained weights."""
import unittest
import torch

from completion_context_margin import ContextMarginCompletion, conditioning_mask


class FixtureBackend(torch.nn.Module):
    def __init__(self):
        super().__init__()
        self.calls = 0
        self.mask = None

    def forward(self, image, mask):
        self.calls += 1
        self.mask = mask.clone()
        return torch.full_like(image, .875)


class ContextMarginTests(unittest.TestCase):
    def test_context_expands_but_exact_output_support_and_inputs_are_preserved(self):
        image = torch.rand(1, 3, 256, 256)
        mask = torch.zeros(1, 1, 256, 256)
        mask[:, :, 100:104, 100:104] = 1
        original = image.clone()
        before = mask.clone()
        backend = FixtureBackend()
        wrapper = ContextMarginCompletion(backend, 2).eval()
        output = wrapper(image, mask)
        self.assertEqual(int(backend.mask.sum()), 64)
        self.assertEqual(int(mask.sum()), 16)
        self.assertTrue(torch.equal(image, original))
        self.assertTrue(torch.equal(mask, before))
        self.assertTrue(torch.equal(output[~mask.bool().expand_as(image)], image[~mask.bool().expand_as(image)]))
        self.assertTrue((output[mask.bool().expand_as(image)] == .875).all())
        self.assertFalse(output.requires_grad)

    def test_radius_zero_and_scaled_context_geometry(self):
        for size in (128, 256, 512):
            mask = torch.zeros(1, 1, size, size)
            mask[:, :, size // 2, size // 2] = 1
            self.assertTrue(torch.equal(conditioning_mask(mask, 0), mask))
            radius = round(2 * size / 256)
            self.assertEqual(int(conditioning_mask(mask, 2).sum()), (2 * radius + 1) ** 2)

    def test_empty_mask_bypasses_backend_and_nearly_hidden_context_is_rejected_before_forward(self):
        backend = FixtureBackend()
        image = torch.rand(1, 3, 256, 256)
        wrapper = ContextMarginCompletion(backend, 6)
        self.assertTrue(torch.equal(wrapper(image, torch.zeros(1, 1, 256, 256)), image))
        self.assertEqual(backend.calls, 0)
        mask = torch.zeros(1, 1, 256, 256)
        mask[:, :, 10:246, 10:246] = 1
        self.assertLess(float(mask.mean()), .85)
        with self.assertRaisesRegex(ValueError, "too little visible face"):
            wrapper(image, mask)
        self.assertEqual(backend.calls, 0)

    def test_fractional_nonfinite_and_invalid_radius_inputs_are_rejected(self):
        image = torch.rand(1, 3, 256, 256)
        for value in (.5, float("nan")):
            mask = torch.full((1, 1, 256, 256), value)
            with self.assertRaises(ValueError):
                ContextMarginCompletion(FixtureBackend(), 2)(image, mask)
        for radius in (True, 2.5, -1, 7):
            with self.assertRaises(ValueError):
                ContextMarginCompletion(FixtureBackend(), radius)

    def test_nonfinite_generated_pixels_are_rejected(self):
        class InvalidBackend(FixtureBackend):
            def forward(self, image, mask):
                return torch.full_like(image, float("nan"))
        image = torch.rand(1, 3, 256, 256)
        mask = torch.zeros(1, 1, 256, 256)
        mask[:, :, 100:104, 100:104] = 1
        with self.assertRaises(FloatingPointError):
            ContextMarginCompletion(InvalidBackend(), 2)(image, mask)


if __name__ == "__main__":
    unittest.main()
