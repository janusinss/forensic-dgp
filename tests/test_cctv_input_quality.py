"""Input-only signal tests; no model loading, inference or backward calls."""
import unittest
from unittest.mock import patch

import cv2
import numpy as np
import torch

from cctv_input_quality import observed_quality_signals, quality_for_crop


class ObservedInputQualityTests(unittest.TestCase):
    def test_padding_changes_cannot_change_visible_quality(self):
        image = np.full((256, 256, 3), 40, np.uint8)
        observed = np.zeros((256, 256), bool)
        observed[:, 64:192] = True
        removal = np.zeros((256, 256), bool)
        first = observed_quality_signals(image, observed, removal)
        image[~observed] = 255
        second = observed_quality_signals(image, observed, removal)
        for key in ('blur_variance', 'noise_sigma_255', 'suggest_restoration', 'visible_signal_pixels'):
            self.assertEqual(first[key], second[key])
        self.assertEqual(first['blur_variance'], 0.)
        self.assertTrue(first['suggest_restoration'])

    def test_reviewed_covering_pixels_do_not_contribute_to_signals(self):
        image = np.full((256, 256, 3), 90, np.uint8)
        observed = np.ones((256, 256), bool)
        removal = np.zeros((256, 256), bool)
        removal[140:230, 50:206] = True
        before = observed_quality_signals(image, observed, removal)
        image[removal] = np.random.default_rng(3).integers(0, 256, (removal.sum(), 3), dtype=np.uint8)
        after = observed_quality_signals(image, observed, removal)
        self.assertEqual(before, after)

    def test_detail_blur_and_noise_are_still_distinguished_on_full_support(self):
        yy, xx = np.mgrid[:256, :256]
        gray = (110 + 20*np.sin(xx*.8) + 20*np.sin(yy*.8)).clip(0, 255).astype(np.uint8)
        detail = np.repeat(gray[..., None], 3, -1)
        support = np.ones((256, 256), bool)
        removal = np.zeros((256, 256), bool)
        self.assertFalse(observed_quality_signals(detail, support, removal)['suggest_restoration'])
        blurred = cv2.GaussianBlur(detail, (13, 13), 3)
        self.assertTrue(observed_quality_signals(blurred, support, removal)['suggest_restoration'])
        noisy = np.clip(detail.astype(float) + np.random.default_rng(7).normal(0, 20, detail.shape), 0, 255).round().astype(np.uint8)
        self.assertGreaterEqual(observed_quality_signals(noisy, support, removal)['noise_sigma_255'], 8)

    def test_small_native_crop_preparation_is_not_a_structural_qualification(self):
        crop = np.full((27, 23, 3), 100, np.uint8)
        with patch('torch.load') as load, patch('torch.autograd.grad') as gradient, \
                patch('models.DGPSynthesizer') as model:
            signals = quality_for_crop(crop)
        load.assert_not_called()
        gradient.assert_not_called()
        model.assert_not_called()
        self.assertEqual(signals['geometry']['native_size'], [23, 27])
        self.assertEqual(signals['geometry']['output_size'], [256, 256])
        self.assertFalse(signals['structure_qualification_established'])
        self.assertFalse(signals['usable_face_confirmed'])

    def test_all_hidden_support_requests_signal_override_without_claiming_face_structure(self):
        image = np.full((256, 256, 3), 100, np.uint8)
        signals = observed_quality_signals(image, np.ones((256, 256), bool), np.ones((256, 256), bool))
        self.assertIsNone(signals['blur_variance'])
        self.assertFalse(signals['suggest_restoration'])
        self.assertEqual(signals['visible_signal_pixels'], 0)
        self.assertFalse(signals['structure_qualification_established'])

    def test_invalid_canvas_support_or_removal_is_rejected(self):
        image = np.full((256, 256, 3), 100, np.uint8)
        observed = np.ones((256, 256), bool)
        removal = np.zeros((256, 256), bool)
        for args in [(image.astype(float), observed, removal), (image[:128], observed, removal),
                     (image, observed[:128], removal), (image, observed, np.full((256, 256), .5))]:
            with self.assertRaises(ValueError):
                observed_quality_signals(*args)
        observed[:, :20] = False
        removal[:, :20] = True
        with self.assertRaisesRegex(ValueError, 'observed'):
            observed_quality_signals(image, observed, removal)


if __name__ == '__main__':
    unittest.main()
