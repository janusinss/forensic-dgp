"""Check scanline suppression against a source with known pixel values."""

import unittest

import numpy as np

from implementation.scanline_cleanup import (
    clean_residual_scanline_comb,
    detect_and_clean_spatial_comb,
    detect_scanlines,
    prepare_scanlines,
)


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

    def test_residual_scanline_comb_cleanup_and_spatial_comb_gate(self):
        y, x = np.indices((160, 160))
        base = 120 + 28 * np.sin(x / 11) + 14 * np.cos(y / 17)
        clean = np.repeat(np.rint(base)[..., None].astype(np.uint8), 3, axis=2)
        out_clean, stats = clean_residual_scanline_comb(clean, source=clean)
        self.assertEqual(out_clean.shape, clean.shape)
        self.assertIn('comb_ela_masked_pixels', stats)
        self.assertIn('chroma_cleanup_pixels', stats)
        self.assertEqual(detect_and_clean_spatial_comb(clean), (None, None, {}))

        combed = clean.astype(np.int16)
        combed[::2, :, :] += 14
        combed[1::2, :, :] -= 14
        combed_u8 = np.clip(combed, 0, 255).astype(np.uint8)
        spatial_out, spatial_report, spatial_ev = detect_and_clean_spatial_comb(combed_u8)
        self.assertIsNotNone(spatial_out)
        self.assertIsNotNone(spatial_report)
        self.assertEqual(spatial_report['policy'], 'spatial-comb-ela-v1')
        self.assertIn('spatial-comb-cleaned.png', spatial_ev)
