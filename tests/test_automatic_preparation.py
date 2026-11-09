"""Spatial preservation and conservative block-pattern routing checks."""
import io
import unittest
from unittest.mock import patch

import numpy as np
from PIL import Image, ImageFilter
from scipy.ndimage import gaussian_filter

from implementation import automatic_preparation as automatic


class AutomaticPreparationTests(unittest.TestCase):
    def test_strength_can_increase_for_coarse_blocks_and_defaults_without_evidence(self):
        rng = np.random.default_rng(92)
        coarse = rng.integers(0, 256, (10, 10, 3), dtype=np.uint8)
        blocks = Image.fromarray(coarse).resize((256, 256), Image.Resampling.NEAREST)
        mask = np.ones((256, 256), bool)
        choice = automatic.choose_pixelation_radius(blocks, mask)
        self.assertEqual(choice['basis'], 'measured_artifact_detail_tradeoff')
        self.assertEqual(choice['selected_radius_at256'], 3.5)
        flat = Image.new('RGB', (256, 256), (140, 140, 140))
        fallback = automatic.choose_pixelation_radius(flat, mask)
        self.assertEqual(fallback['basis'], 'default_weak_evidence')
        self.assertEqual(fallback['selected_radius_at256'], 3.0)

    def test_shifted_compression_grid_requires_evidence_in_both_directions(self):
        rng = np.random.default_rng(2026)
        texture = gaussian_filter(rng.normal(127, 140, (144, 160)), 2)
        gray = np.clip(127 + (texture - 127) * 2.2, 0, 255).astype(np.uint8)
        stream = io.BytesIO()
        Image.fromarray(gray).save(stream, format='JPEG', quality=9)
        decoded = np.asarray(Image.open(stream), dtype=np.float32)
        self.assertEqual(automatic.compression_grid_phases(decoded[5:125, 3:139]), [2, 4])
        self.assertIsNone(automatic.compression_grid_phases(gray[5:125, 3:139].astype(np.float32)))

    def test_regular_blocks_require_both_axes_and_visible_contrast(self):
        coarse = np.random.default_rng(12).integers(0, 256, (12, 12, 3), dtype=np.uint8)
        block = np.asarray(Image.fromarray(coarse).resize((96, 96), Image.Resampling.NEAREST))
        signal = automatic.pixelation_signal(block, [0, 0, 96, 96])
        self.assertTrue(signal['suspected'])
        self.assertEqual(signal['period_native'], 8)
        softened = np.asarray(Image.fromarray(block).filter(ImageFilter.GaussianBlur(4)))
        self.assertFalse(automatic.pixelation_signal(softened, [0, 0, 96, 96])['suspected'])
        stripes = np.tile(block[:1], (96, 1, 1))
        self.assertFalse(automatic.pixelation_signal(stripes, [0, 0, 96, 96])['suspected'])
        self.assertFalse(automatic.pixelation_signal(np.full_like(block, 120), [0, 0, 96, 96])['suspected'])

    def test_face_smoothing_keeps_crop_geometry_and_every_surrounding_pixel(self):
        source = np.random.default_rng(3).integers(0, 256, (160, 180, 3), dtype=np.uint8)
        rectangle = [20, 10, 130, 140]
        native = source[10:150, 20:150]
        decision = {'route': 'face_gaussian', 'reason': 'suspected_pixelation',
                    'degradation': {'period_native': 12}}
        with patch.object(automatic, 'decide', return_value=decision):
            contents, metadata, evidence, report = automatic.prepare(native, source, [40, 30, 80, 90], rectangle)
        prepared = np.asarray(Image.open(io.BytesIO(contents)))
        apply = np.zeros(native.shape[:2], bool)
        apply[20:110, 20:100] = True
        self.assertEqual(prepared.shape, native.shape)
        np.testing.assert_array_equal(prepared[~apply], native[~apply])
        self.assertGreater(np.any(prepared[apply] != native[apply], axis=1).mean(), .99)
        self.assertFalse(metadata['automatic_protection'])
        self.assertEqual(report['Gaussian_radius_at256'], 3.0)
        self.assertEqual(evidence['automatic-crop-input.png'], automatic.png(native))

    def test_coarse_rectangular_blocks_survive_noninteger_resizing(self):
        coarse = np.random.default_rng(28).integers(0, 256, (7, 6, 3), dtype=np.uint8)
        block = Image.fromarray(coarse).resize((282, 287), Image.Resampling.NEAREST)
        resized = block.resize((259, 271), Image.Resampling.BILINEAR)
        for image, expected in ((block, (47, 41)), (resized, (259 / 6, 271 / 7))):
            with self.subTest(size=image.size):
                signal = automatic.pixelation_signal(np.asarray(image), [0, 0, *image.size])
                self.assertEqual(signal.get('pattern'), 'coarse_or_resized_grid')
                self.assertAlmostEqual(signal['block_width_native'], expected[0], delta=2)
                self.assertAlmostEqual(signal['block_height_native'], expected[1], delta=2)
