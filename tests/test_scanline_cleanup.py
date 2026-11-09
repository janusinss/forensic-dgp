"""Check scanline suppression against a source with known pixel values."""

import unittest

import numpy as np

from implementation.scanline_cleanup import detect_scanlines, prepare_scanlines


class ScanlineCleanupTests(unittest.TestCase):
    def test_reduces_horizontal_banding_and_rejects_unrelated_directions(self):
        y, x = np.indices((180, 216))
        clean = 128 + 25 * np.sin(x / 20) + 12 * np.cos(y / 30)
        scanlines = 24 * np.sin(2 * np.pi * y / 5.1)
        noisy = np.repeat(np.rint(clean + scanlines)[..., None].astype(np.uint8), 3, axis=2)
        clean = np.repeat(clean[..., None], 3, axis=2)
        prepared, report = prepare_scanlines(noisy)
        self.assertAlmostEqual(report['spacing_native_pixels'], 5.1, delta=.1)
        self.assertEqual(prepared.shape, noisy.shape)
        self.assertLess(np.mean(np.abs(prepared - clean)), .25 * np.mean(np.abs(noisy - clean)))
        self.assertIsNone(detect_scanlines(noisy.transpose(1, 0, 2)))
        self.assertIsNone(detect_scanlines(np.rint(clean).astype(np.uint8)))
