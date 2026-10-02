import tempfile
import unittest
from pathlib import Path

import torch

from lama_completion import LaMaCompletion, load_lama


class RecordingNet(torch.nn.Module):
    def forward(self, value):
        self.received = value.clone()
        return torch.full_like(value[:, :3], .25)


class LaMaCompletionTests(unittest.TestCase):
    def setUp(self):
        torch.set_num_threads(2)

    def test_white_pixels_and_covered_colors_cannot_override_explicit_mask(self):
        net = RecordingNet()
        adapter = LaMaCompletion(net)
        image = torch.ones(1, 3, 64, 64)
        mask = torch.zeros(1, 1, 64, 64)
        mask[:, :, 21:43, 17:48] = 1
        output = adapter(image, mask)
        first = net.received.clone()
        other = torch.where(mask.bool(), torch.zeros_like(image), image)
        self.assertTrue(torch.equal(output, adapter(other, mask)))
        self.assertTrue(torch.equal(first, net.received))
        self.assertEqual(first.shape[1], 4)
        self.assertTrue(torch.equal(first[:, 3:4], mask))
        self.assertTrue(torch.equal(first[:, :3] * mask, torch.zeros_like(image)))
        self.assertTrue(torch.equal(output[mask.expand_as(image) == 0], image[mask.expand_as(image) == 0]))

    def test_empty_batch_members_skip_network_and_padding_preserves_dimensions(self):
        net = RecordingNet()
        adapter = LaMaCompletion(net)
        image = torch.rand(2, 3, 45, 51)
        mask = torch.zeros(2, 1, 45, 51)
        self.assertTrue(torch.equal(adapter(image, mask), image))
        self.assertFalse(hasattr(net, "received"))
        mask[0, :, 13:24, 16:30] = 1
        output = adapter(image, mask)
        self.assertEqual(net.received.shape, (1, 4, 64, 64))
        self.assertEqual(output.shape, image.shape)
        self.assertTrue(torch.equal(output[1], image[1]))
        self.assertTrue(torch.equal(output[mask.expand_as(image) == 0], image[mask.expand_as(image) == 0]))

    def test_invalid_masks_and_nonfinite_network_outputs_fail(self):
        adapter = LaMaCompletion(RecordingNet())
        image = torch.rand(1, 3, 64, 64)
        for mask in (torch.ones(1, 1, 64, 64), torch.full((1, 1, 64, 64), .5)):
            with self.assertRaises(ValueError):
                adapter(image, mask)
        class Invalid(torch.nn.Module):
            def forward(self, value):
                return torch.full_like(value[:, :3], float("nan"))
        mask = torch.zeros(1, 1, 64, 64)
        mask[:, :, 10:20, 10:20] = 1
        with self.assertRaises(FloatingPointError):
            LaMaCompletion(Invalid())(image, mask)

    def test_checksum_checked_before_checkpoint_loading(self):
        with tempfile.TemporaryDirectory() as folder:
            path = Path(folder) / "incorrect.pth"
            path.write_bytes(b"not a published checkpoint")
            with self.assertRaisesRegex(ValueError, "SHA256 mismatch"):
                load_lama(path)


if __name__ == "__main__":
    unittest.main()
