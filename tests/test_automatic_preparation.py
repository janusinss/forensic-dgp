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

    def test_detect_pasted_degraded_face_patch_positive_and_zero_fp_controls(self):
        """Straight-row paste-seam detector fires on pasted grain seams and stays 0-FP on clean portraits/curves."""
        from pathlib import Path
        from PIL import ImageOps

        # 1) Synthetic pasted fine-grain rectangle starting at y=70 on a smooth 320x320 portrait
        rng = np.random.default_rng(2027)
        yy, xx = np.mgrid[0:320, 0:320].astype(np.float32)
        base = np.stack([
            np.clip(165 + 18 * np.sin(xx / 45.0) + 12 * np.cos(yy / 55.0), 0, 255),
            np.clip(125 + 14 * np.sin(xx / 45.0) + 10 * np.cos(yy / 55.0), 0, 255),
            np.clip(105 + 10 * np.sin(xx / 45.0) + 8 * np.cos(yy / 55.0), 0, 255),
        ], axis=-1).astype(np.float32)
        pasted = base.copy()
        pasted[70:312, 20:300] = np.clip(
            base[70:312, 20:300] + 22.0 + rng.normal(0.0, 14.0, (242, 280, 3)), 0, 255
        )
        synth_mask = automatic.detect_pasted_degraded_face_patch(pasted.astype(np.uint8))
        self.assertIsNotNone(synth_mask)
        self.assertGreater(float(synth_mask.mean()), 0.50)

        # 2) Synthetic curved hairline with strong contrast must NOT trigger (lacks straight-row continuity)
        curved = base.copy()
        hair_y = 65.0 + 28.0 * np.cos((xx - 160.0) / 90.0)
        is_hair = yy < hair_y
        curved[is_hair] = np.array([35.0, 28.0, 25.0], dtype=np.float32)
        curved[~is_hair] = np.clip(curved[~is_hair] + rng.normal(0.0, 6.0, curved[~is_hair].shape), 0, 255)
        self.assertIsNone(automatic.detect_pasted_degraded_face_patch(curved.astype(np.uint8)))

        # 3) Real fixtures: S5:low light phone positive; ALL 5 clean default portraits strictly 0-FP
        test_dir = Path(__file__).resolve().parents[2] / 'test_images'
        s5_lowlt = test_dir / 'face 5 - low light phone.png'
        if s5_lowlt.exists():
            rgb_s5 = np.asarray(ImageOps.exif_transpose(Image.open(s5_lowlt)).convert('RGB'))
            mask_s5 = automatic.detect_pasted_degraded_face_patch(rgb_s5)
            self.assertIsNotNone(mask_s5)
            self.assertGreater(float(mask_s5.mean()), 0.45)

        for subj in range(1, 6):
            clean_path = test_dir / f'face {subj} - default.jpg'
            if clean_path.exists():
                rgb_clean = np.asarray(ImageOps.exif_transpose(Image.open(clean_path)).convert('RGB'))
                self.assertIsNone(
                    automatic.detect_pasted_degraded_face_patch(rgb_clean),
                    f'False positive on clean portrait {clean_path.name}',
                )

    def test_condition_degraded_face_patch_skips_clean_color_mosaic_and_preserves_outside_mask(self):
        """Clean-color mosaics bypass degraded-patch conditioning; conditioned patches keep ~mask bit-exact."""
        rng = np.random.default_rng(303)
        yy, xx = np.mgrid[0:320, 0:320].astype(np.float32)
        smooth = np.stack([
            np.clip(110 + 70 * np.sin(xx / 37.0) + 40 * np.cos(yy / 41.0), 0, 255),
            np.clip(120 + 60 * np.cos(xx / 43.0) + 35 * np.sin(yy / 31.0), 0, 255),
            np.clip(130 + 50 * np.sin((xx + yy) / 53.0), 0, 255),
        ], axis=-1).astype(np.uint8)
        mask = np.zeros((320, 320), dtype=bool)
        mask[32:288, 32:288] = True

        # Clean-color smooth portrait (neither block noise nor collapsed palette) -> returns None
        self.assertIsNone(automatic._condition_degraded_face_patch(smooth, mask))

        # Multi-pixel block noise inside mask -> conditioned, and ~final_mask stays 100% bit-exact
        noisy = smooth.copy()
        blk = rng.integers(35, 220, (32, 32, 3), dtype=np.uint8)
        noisy[32:288, 32:288] = np.asarray(Image.fromarray(blk).resize((256, 256), Image.Resampling.NEAREST))
        conditioned = automatic._condition_degraded_face_patch(noisy, mask)
        self.assertIsNotNone(conditioned)
        prepared, final_mask, meta = conditioned
        self.assertEqual(meta['regime'], 'multipixel_block_noise')
        np.testing.assert_array_equal(prepared[~final_mask], noisy[~final_mask])

        # Clean-color 24x24 and 32x32 privacy mosaics on real portraits must NOT trigger chroma/shadow conditioning
        from pathlib import Path
        import cv2
        from PIL import ImageOps

        test_dir = Path(__file__).resolve().parents[2] / 'test_images'
        for subj in (1, 3, 5):
            clean_path = test_dir / f'face {subj} - default.jpg'
            if not clean_path.exists():
                continue
            rgb0 = np.asarray(ImageOps.exif_transpose(Image.open(clean_path)).convert('RGB'))
            h, w = rgb0.shape[:2]
            y0, y1, x0, x1 = int(h * 0.12), int(h * 0.88), int(w * 0.12), int(w * 0.88)
            for blk_sz in (24, 32):
                rgb_mos = rgb0.copy()
                sub = rgb_mos[y0:y1, x0:x1]
                sh, sw = sub.shape[:2]
                sm = cv2.resize(sub, (max(4, sw // blk_sz), max(4, sh // blk_sz)), interpolation=cv2.INTER_AREA)
                rgb_mos[y0:y1, x0:x1] = cv2.resize(sm, (sw, sh), interpolation=cv2.INTER_NEAREST)
                m = np.zeros((h, w), dtype=bool)
                m[y0:y1, x0:x1] = True
                self.assertIsNone(
                    automatic._condition_degraded_face_patch(rgb_mos, m),
                    f'Clean {blk_sz}x{blk_sz} mosaic on face {subj} falsely triggered degraded-patch conditioning',
                )

    def test_condition_degraded_face_patch_upper_band_seam_bounds_safety(self):
        """Upper-band mask with ty_seam > y1 must never invert bounds or raise IndexError."""
        rng = np.random.default_rng(404)
        h, w = 1200, 1200
        img = np.full((h, w, 3), 120, dtype=np.uint8)
        # Put a strong horizontal step at y=305 (inside 4%-26% band: 48..312) below an upper-band mask [0..300]
        img[306:, :] = 210
        blk = rng.integers(35, 220, (38, 125, 3), dtype=np.uint8)
        img[0:300, 100:1100] = np.asarray(Image.fromarray(blk).resize((1000, 300), Image.Resampling.NEAREST))
        mask = np.zeros((h, w), dtype=bool)
        mask[0:300, 100:1100] = True
        # Must not raise IndexError; either safely conditions valid bounds or returns None
        res = automatic._condition_degraded_face_patch(img, mask)
        if res is not None:
            prepared, final_mask, _ = res
            self.assertTrue(final_mask.any())
            np.testing.assert_array_equal(prepared[~final_mask], img[~final_mask])
        # Invalid inputs must return None cleanly without crashing
        self.assertIsNone(automatic._condition_degraded_face_patch(img.astype(np.float32), mask))
        self.assertIsNone(automatic._condition_degraded_face_patch(img, np.zeros((10, 10), dtype=bool)))


class DegradedFacePatchRoutingTests(unittest.TestCase):
    """Verify all 20 degraded face test images route cleanly through degraded-region conditioning."""

    def _run_subjects(self, subjects, conds=('full breakdown', 'low light phone', 'motion capture', 'raw sensor')):
        from pathlib import Path
        from PIL import ImageOps
        from implementation.conditioning_v2 import compute_v2_conditioning_numpy
        from implementation.runtime import prepare_crop

        test_dir = Path(__file__).resolve().parents[2] / 'test_images'
        checked = 0
        for subj in subjects:
            for cond in conds:
                alt = cond.replace('low light ', 'low light  ')
                path = next(
                    (
                        test_dir / name
                        for name in (
                            f'face {subj} - {cond}.png',
                            f'face {subj} - {alt}.png',
                            f'face {subj}- {cond}.png',
                        )
                        if (test_dir / name).exists()
                    ),
                    None,
                )
                if path is None:
                    continue
                checked += 1
                with self.subTest(image=path.name):
                    rgb = np.asarray(ImageOps.exif_transpose(Image.open(path)).convert('RGB'))
                    prepared, report, evidence = automatic.prepare_source(rgb)
                    self.assertIsNotNone(prepared, f'{path.name} was not routed to degraded-region conditioning')
                    self.assertIsNotNone(report)
                    self.assertIn('full-image-pixelation-mask.png', evidence)
                    mask = np.asarray(Image.open(io.BytesIO(evidence['full-image-pixelation-mask.png']))) > 0
                    np.testing.assert_array_equal(prepared[~mask], rgb[~mask])
                    prep_256, _, _, _ = prepare_crop(prepared)
                    v2c = compute_v2_conditioning_numpy(prep_256.astype(np.float32) / 255.0, np=np)
                    self.assertGreaterEqual(
                        float(v2c['g_blur']),
                        0.85,
                        f'{path.name} yielded low g_blur={v2c["g_blur"]}',
                    )
        if checked == 0:
            self.skipTest('Degraded face test fixtures not present in test_images/')
        self.assertEqual(checked, len(subjects) * len(conds))

    def test_subject_1a_degraded_face_patches_route_with_high_g_blur(self):
        self._run_subjects((1,), conds=('full breakdown', 'low light phone'))

    def test_subject_1b_degraded_face_patches_route_with_high_g_blur(self):
        self._run_subjects((1,), conds=('motion capture', 'raw sensor'))

    def test_subject_2_degraded_face_patches_route_with_high_g_blur(self):
        self._run_subjects((2,))

    def test_subject_3_degraded_face_patches_route_with_high_g_blur(self):
        self._run_subjects((3,))

    def test_subject_4a_degraded_face_patches_route_with_high_g_blur(self):
        self._run_subjects((4,), conds=('full breakdown', 'low light phone'))

    def test_subject_4b_degraded_face_patches_route_with_high_g_blur(self):
        self._run_subjects((4,), conds=('motion capture', 'raw sensor'))

    def test_subject_5a_degraded_face_patches_route_with_high_g_blur(self):
        self._run_subjects((5,), conds=('full breakdown', 'low light phone'))

    def test_subject_5b_degraded_face_patches_route_with_high_g_blur(self):
        self._run_subjects((5,), conds=('motion capture',))

    def test_subject_5c_degraded_face_patches_route_with_high_g_blur(self):
        self._run_subjects((5,), conds=('raw sensor',))
