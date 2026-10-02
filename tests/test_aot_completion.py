import tempfile
import unittest
from pathlib import Path

import torch

from aot_completion import AOTCompletion, load_aot


class RecordingNet(torch.nn.Module):
    def forward(self, image, mask):
        self.image, self.mask = image.clone(), mask.clone()
        return torch.zeros_like(image)


class AOTCompletionTests(unittest.TestCase):
    def setUp(self):
        torch.set_num_threads(2)

    def test_white_glare_and_visible_white_pixels_use_different_mask_values(self):
        image = torch.ones(1, 3, 64, 64)
        mask = torch.zeros(1, 1, 64, 64)
        mask[:, :, 18:44, 17:47] = 1
        net = RecordingNet()
        output = AOTCompletion(net)(image, mask)
        self.assertEqual(net.image.shape, (1, 3, 512, 512))
        self.assertEqual(net.mask.shape, (1, 1, 512, 512))
        self.assertTrue((output[mask.expand_as(image) == 1] == .5).all())
        self.assertTrue(torch.equal(output[mask.expand_as(image) == 0], image[mask.expand_as(image) == 0]))

    def test_covered_colors_cannot_leak_during_resize_and_empty_members_bypass(self):
        image = torch.full((2, 3, 67, 61), .25)
        mask = torch.zeros(2, 1, 67, 61)
        mask[0, :, 21:44, 15:41] = 1
        net = RecordingNet()
        adapter = AOTCompletion(net)
        first = adapter(image, mask)
        network_image = net.image.clone()
        other = torch.where(mask.bool(), torch.ones_like(image), image)
        self.assertTrue(torch.equal(first, adapter(other, mask)))
        self.assertTrue(torch.equal(network_image, net.image))
        self.assertEqual(net.image.shape[0], 1)
        self.assertTrue(torch.equal(first[1], image[1]))
        fresh = RecordingNet()
        self.assertTrue(torch.equal(AOTCompletion(fresh)(image, torch.zeros_like(mask)), image))
        self.assertFalse(hasattr(fresh, "image"))

    def test_invalid_masks_and_nonfinite_outputs_rejected(self):
        image = torch.rand(1, 3, 64, 64)
        for mask in (torch.ones(1, 1, 64, 64), torch.full((1, 1, 64, 64), .3)):
            with self.assertRaises(ValueError):
                AOTCompletion(RecordingNet())(image, mask)
        class Invalid(torch.nn.Module):
            def forward(self, image, mask):
                return torch.full_like(image, float("nan"))
        mask = torch.zeros(1, 1, 64, 64)
        mask[:, :, 12:23, 21:35] = 1
        with self.assertRaises(FloatingPointError):
            AOTCompletion(Invalid())(image, mask)

    def test_checkpoint_fingerprint_checked_before_loading(self):
        with tempfile.TemporaryDirectory() as folder:
            path = Path(folder) / "incorrect.pt"
            path.write_bytes(b"incorrect model")
            with self.assertRaisesRegex(ValueError, "SHA256"):
                load_aot(path)


if __name__ == "__main__":
    unittest.main()
