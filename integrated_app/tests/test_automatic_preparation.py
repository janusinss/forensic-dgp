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

    def test_rotation_fallback_preserves_pixels_outside_its_remapped_mask(self):
        rng = np.random.default_rng(82)
        source = Image.new('RGB', (256, 256), (115, 112, 109))
        tiles = Image.fromarray(rng.integers(20, 230, (20, 18, 3), dtype=np.uint8))
        source.paste(tiles.resize((144, 160), Image.Resampling.NEAREST), (56, 48))
        for angle in (9, 15):
            with self.subTest(angle=angle):
                rotated = np.asarray(source.rotate(angle, resample=Image.Resampling.BICUBIC,
                                                   fillcolor=(115, 112, 109)))
                prepared, report, evidence = automatic.rotated_pixelation(rotated)
                self.assertEqual(report['rotation_correction_degrees'], -angle)
                self.assertEqual(report['Gaussian_radius_at256'], 3.0)
                mask = np.asarray(Image.open(io.BytesIO(evidence['full-image-pixelation-mask.png']))) > 0
                self.assertGreater(np.count_nonzero(mask), 10000)
                np.testing.assert_array_equal(prepared[~mask], rotated[~mask])

    def test_spectral_mosaic_is_diagnostic_and_rejects_sharp_blocks(self):
        rng = np.random.default_rng(84)
        tiles = Image.fromarray(rng.integers(30, 220, (20, 18), dtype=np.uint8))
        sharp = tiles.resize((144, 160), Image.Resampling.NEAREST)
        image = Image.new('L', (256, 256), 115)
        image.paste(sharp, (56, 48))
        self.assertEqual(automatic.softened_pixelation_evidence(np.asarray(image.convert('RGB'))),
                         (None, None, {}))
        image.paste(sharp.filter(ImageFilter.GaussianBlur(2.1)), (56, 48))
        prepared, report, evidence = automatic.softened_pixelation_evidence(np.asarray(image.convert('RGB')))
        self.assertIsNone(prepared)
        self.assertGreater(report['masked_native_pixels'], 10000)
        self.assertEqual(report['mask_sha256'], automatic.sha(evidence['softened-mosaic-mask.png']))

    def test_tiny_crop_needs_repeating_native_seams_in_both_axes(self):
        rng = np.random.default_rng(35)
        canvas = Image.new('RGB', (40, 40), (125, 128, 130))
        tiles = Image.fromarray(rng.integers(20, 220, (9, 9, 3), dtype=np.uint8))
        canvas.paste(tiles.resize((30, 30), Image.Resampling.NEAREST), (5, 5))
        source = np.asarray(canvas)
        prepared, report, evidence = automatic.tiny_crop_pixelation(source)
        self.assertAlmostEqual(report['tiny_grid_period_native'][0], 3.33, delta=.3)
        mask = np.asarray(Image.open(io.BytesIO(evidence['full-image-pixelation-mask.png']))) > 0
        self.assertGreater(np.count_nonzero(mask), 144)
        np.testing.assert_array_equal(prepared[~mask], source[~mask])
        stripes = np.repeat(source[:1], 40, axis=0)
        self.assertEqual(automatic.tiny_crop_pixelation(stripes), (None, None, {}))
        y, x = np.indices((40, 40))
        checker = (120 + 45 * ((x // 4 + y // 4) % 2)).astype(np.uint8)
        checker_rgb = np.repeat(checker[:, :, None], 3, axis=2)
        self.assertEqual(automatic.tiny_crop_pixelation(checker_rgb), (None, None, {}))

    def test_scanline_cleanup_uses_masked_comb_ela_instead_of_full_frame_blur(self):
        y, x = np.indices((180, 216))
        clean = 128 + 25 * np.sin(x / 18) + 12 * np.cos(y / 28)
        scanlines = 24 * np.sin(2 * np.pi * y / 5.1)
        source = np.repeat(np.rint(clean + scanlines)[..., None].astype(np.uint8), 3, axis=2)
        prepared, report, evidence = automatic.prepare_source(source)
        self.assertIsNotNone(prepared)
        self.assertNotIn('Gaussian_radius_at256', report)
        self.assertIn('comb_ela_masked_pixels', report)
        self.assertIn('chroma_cleanup_pixels', report)
        self.assertIn('scanline-cleaned.png', evidence)

    def test_composable_spatial_comb_respects_allow_flag(self):
        y, x = np.indices((160, 160))
        base = 120 + 28 * np.sin(x / 11) + 14 * np.cos(y / 17)
        combed = np.repeat(np.rint(base)[..., None].astype(np.int16), 3, axis=2)
        combed[::2, :, :] += 14
        combed[1::2, :, :] -= 14
        combed_u8 = np.clip(combed, 0, 255).astype(np.uint8)

        out, stages, reports, evidence = automatic.prepare_composables(combed_u8, allow_spatial_comb=True)
        self.assertIsNotNone(out)
        self.assertEqual(stages[0], 'spatial_comb')
        self.assertIn('spatial_comb', reports)
        self.assertIn('spatial-comb-cleaned.png', evidence)

        out_disabled, stages_disabled, _, _ = automatic.prepare_composables(combed_u8, allow_spatial_comb=False)
        self.assertNotIn('spatial_comb', stages_disabled)
        if out_disabled is not None:
            self.assertGreater(len(stages_disabled), 0)

    def test_strip_dct_macroblocking_preserves_flat_mosaic_and_strips_dct_patch(self):
        rng = np.random.default_rng(7)
        flat_canvas = np.full((192, 192, 3), 120, dtype=np.uint8)
        tiles = Image.fromarray(rng.integers(25, 230, (8, 8, 3), dtype=np.uint8))
        flat_canvas[48:144, 48:144] = np.asarray(tiles.resize((96, 96), Image.Resampling.NEAREST))
        flat_mask = np.zeros((192, 192), dtype=bool)
        flat_mask[48:144, 48:144] = True
        kept = automatic._strip_dct_macroblocking_components(flat_canvas, flat_mask)
        self.assertEqual(int(kept.sum()), int(flat_mask.sum()))

        det = {'suspected': True, 'regional': True, 'mask_fraction': 0.12, 'is_flat_privacy_mosaic': False}
        with patch.object(automatic, 'is_dct_macroblocking_patch', return_value=True):
            self.assertTrue(automatic._det_passes_macroblock_gate(flat_canvas, det))
