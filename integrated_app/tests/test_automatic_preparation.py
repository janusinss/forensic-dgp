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

    def test_face_smoothing_keeps_crop_geometry_and_applies_uniform_whole_crop_blur(self):
        source = np.random.default_rng(3).integers(0, 256, (160, 180, 3), dtype=np.uint8)
        rectangle = [20, 10, 130, 140]
        native = source[10:150, 20:150]
        decision = {'route': 'face_gaussian', 'reason': 'suspected_pixelation',
                    'degradation': {'period_native': 12}}
        with patch.object(automatic, 'decide', return_value=decision):
            contents, metadata, evidence, report = automatic.prepare(native, source, [40, 30, 80, 90], rectangle)
        prepared = np.asarray(Image.open(io.BytesIO(contents)))
        face_box = np.zeros(native.shape[:2], bool)
        face_box[20:110, 20:100] = True
        expected_radius_native = 3.0 * max(native.shape[:2]) / 256.0
        expected_uniform = np.asarray(Image.fromarray(native).filter(ImageFilter.GaussianBlur(expected_radius_native)))
        self.assertEqual(prepared.shape, native.shape)
        np.testing.assert_array_equal(prepared, expected_uniform)
        self.assertGreater(np.any(prepared[face_box] != native[face_box], axis=1).mean(), .99)
        self.assertGreater(np.any(prepared[~face_box] != native[~face_box], axis=1).mean(), .99)
        self.assertFalse(metadata['automatic_protection'])
        self.assertTrue(metadata['uniform_crop_smoothing'])
        self.assertTrue(report['uniform_crop_smoothing'])
        self.assertEqual(report['Gaussian_radius_at256'], 3.0)
        self.assertAlmostEqual(report['crop_Gaussian_radius_native'], round(expected_radius_native, 4), places=4)
        self.assertEqual(metadata['crop_Gaussian_radius_native'], report['crop_Gaussian_radius_native'])
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


class AutoBlurOverhaulTests(unittest.TestCase):
    @staticmethod
    def _make_sharp_grid_with_face(size=320, face_blur_sigma=0.0, uniform_blur_sigma=0.0, partial_eye_bar=False):
        rng = np.random.default_rng(4242)
        yy, xx = np.mgrid[0:size, 0:size].astype(np.float32)
        # Sharp background height-chart grid + crisp texture
        bg = np.full((size, size, 3), 215.0, dtype=np.float32)
        bg[(yy.astype(int) % 16) < 2] = 65.0
        bg[(xx.astype(int) % 24) < 2] = 90.0
        bg += rng.normal(0.0, 12.0, bg.shape)
        # Synthetic face with sharp eyes, brows, nose, mouth inside [100:220, 90:230]
        face = [100, 90, 120, 140]
        fx, fy, fw, fh = face
        cx, cy = fx + fw / 2.0, fy + fh / 2.0
        oval = ((xx - cx) / (fw * 0.48)) ** 2 + ((yy - cy) / (fh * 0.48)) ** 2 <= 1.0
        skin = np.stack([
            195.0 - 25.0 * np.hypot((xx - cx) / fw, (yy - cy) / fh),
            152.0 - 20.0 * np.hypot((xx - cx) / fw, (yy - cy) / fh),
            128.0 - 16.0 * np.hypot((xx - cx) / fw, (yy - cy) / fh),
        ], axis=-1)
        img = bg.copy()
        img[oval] = skin[oval]
        # Sharp facial features (eyes, brows, mouth)
        img[fy + 38:fy + 50, fx + 24:fx + 48] = [30, 28, 32]
        img[fy + 38:fy + 50, fx + 72:fx + 96] = [30, 28, 32]
        img[fy + 28:fy + 33, fx + 20:fx + 52] = [45, 35, 30]
        img[fy + 28:fy + 33, fx + 68:fx + 100] = [45, 35, 30]
        img[fy + 96:fy + 108, fx + 36:fx + 84] = [125, 55, 55]
        if partial_eye_bar:
            # Keep the face box corners filled with textured skin + organic curved features
            img[fy:fy + fh, fx:fx + fw] = skin[fy:fy + fh, fx:fx + fw] + rng.normal(0.0, 4.0, (fh, fw, 3))
            le = ((xx - (fx + 36)) / 11.0) ** 2 + ((yy - (fy + 44)) / 6.0) ** 2 <= 1.0
            re = ((xx - (fx + 84)) / 11.0) ** 2 + ((yy - (fy + 44)) / 6.0) ** 2 <= 1.0
            mo = ((xx - (fx + 60)) / 22.0) ** 2 + ((yy - (fy + 102)) / 7.0) ** 2 <= 1.0
            img[le | re] = [30, 28, 32]
            img[mo] = [125, 55, 55]
        img = np.clip(np.rint(img), 0, 255).astype(np.uint8)
        if face_blur_sigma > 0:
            blurred = np.asarray(Image.fromarray(img).filter(ImageFilter.GaussianBlur(face_blur_sigma)))
            zone = ((xx >= fx - 6) & (xx < fx + fw + 6) & (yy >= fy - 6) & (yy < fy + fh + 6))
            img[zone] = blurred[zone]
        elif partial_eye_bar:
            blurred = np.asarray(Image.fromarray(img).filter(ImageFilter.GaussianBlur(10.0)))
            bar = ((xx >= fx + 6) & (xx < fx + fw - 6) & (yy >= fy + 22) & (yy < fy + 76))
            img[bar] = blurred[bar]
        if uniform_blur_sigma > 0:
            img = np.asarray(Image.fromarray(img).filter(ImageFilter.GaussianBlur(uniform_blur_sigma)))
        return img, face

    def test_heavy_gaussian_face_blur_triggers_protect_auto_and_empirical_surround_match(self):
        img, face = self._make_sharp_grid_with_face(face_blur_sigma=12.0)
        decision = automatic.decide(img, face)
        self.assertEqual(decision['route'], 'protect_auto')
        self.assertIn('hf_deficit', decision)
        self.assertTrue(decision['hf_deficit']['clear'])

        cand, meta, ev, report = automatic.prepare(img, img, face, None)
        self.assertIsNotNone(cand)
        self.assertEqual(report['route'], 'protect_auto')
        bm = meta['algorithm']['blur_matching']
        self.assertEqual(bm['status'], 'estimated')
        self.assertGreaterEqual(bm['target_radius_at256'], 2.5)
        self.assertTrue(meta['every_protected_native_RGB_exact'])
        pmask = np.asarray(Image.open(io.BytesIO(ev['effective-Protect-native.png']))) > 0
        derived = np.asarray(Image.open(io.BytesIO(cand)))
        np.testing.assert_array_equal(derived[pmask], img[pmask])
        self.assertGreater(int(np.any(derived[~pmask] != img[~pmask], axis=1).sum()), 0)

    def test_partial_eye_bar_blur_triggers_protect_auto(self):
        img, face = self._make_sharp_grid_with_face(partial_eye_bar=True)
        decision = automatic.decide(img, face)
        self.assertEqual(decision['route'], 'protect_auto')

    def test_clean_sharp_face_and_uniformly_blurred_face_never_trigger_auto_blur(self):
        clean, face = self._make_sharp_grid_with_face(face_blur_sigma=0.0)
        self.assertEqual(automatic.decide(clean, face)['route'], 'direct')

        uniform, face_u = self._make_sharp_grid_with_face(face_blur_sigma=0.0, uniform_blur_sigma=6.0)
        self.assertEqual(automatic.decide(uniform, face_u)['route'], 'direct')

    def test_background_grid_guard_distinguishes_blurred_face_from_pixelated_face(self):
        img, face = self._make_sharp_grid_with_face(face_blur_sigma=12.0)
        fx, fy, fw, fh = face
        bg_mask = np.ones(img.shape[:2], dtype=bool)
        bg_mask[fy:fy + fh, fx:fx + fw] = False
        self.assertTrue(automatic.is_background_grid_with_blurred_face(img, bg_mask, face))

        # Real pixelated face inside face box must NOT be bypassed by the background grid guard
        rng = np.random.default_rng(19)
        mosaic_img = img.copy()
        blk = rng.integers(20, 235, (10, 10, 3), dtype=np.uint8)
        mosaic_img[fy:fy + fh, fx:fx + fw] = np.asarray(Image.fromarray(blk).resize((fw, fh), Image.Resampling.NEAREST))
        face_mask = np.zeros(img.shape[:2], dtype=bool)
        face_mask[fy:fy + fh, fx:fx + fw] = True
        self.assertFalse(automatic.is_background_grid_with_blurred_face(mosaic_img, face_mask, face))

    def test_blur_island_fallback_locates_heavy_blur_and_rejects_uniform_images(self):
        img, face = self._make_sharp_grid_with_face(face_blur_sigma=12.0)
        island = automatic.detect_blur_island_face_box(img)
        self.assertIsNotNone(island)
        ix, iy, iw, ih = island
        fx, fy, fw, fh = face
        # Island center should fall inside the true blurred face box
        self.assertTrue(fx <= ix + iw / 2.0 <= fx + fw)
        self.assertTrue(fy <= iy + ih / 2.0 <= fy + fh)

        uniform, _ = self._make_sharp_grid_with_face(uniform_blur_sigma=6.0)
        self.assertIsNone(automatic.detect_blur_island_face_box(uniform))

    def test_overhaul_regressions_small_crop_period_radius_white_backdrop_and_coarse_chroma(self):
        """Verify 48-64px dead-zone closure, period-coupled radius, white-backdrop haze guard, and coarse chroma denoise."""
        import cv2
        from implementation.cast_dehaze import detect_cast_dehaze
        from implementation.irregular_mosaic import irregular_mosaic_mask
        from implementation.noise_denoise import detect_noise, prepare_noise_denoise

        # 1) 56x56 small-crop mosaic (was in 48-60px dead zone) now detected cleanly
        rng = np.random.default_rng(515)
        small_canvas = Image.new('RGB', (56, 56), (120, 122, 125))
        tiles = Image.fromarray(rng.integers(20, 235, (10, 10, 3), dtype=np.uint8))
        small_canvas.paste(tiles.resize((40, 40), Image.Resampling.NEAREST), (8, 8))
        t_out, t_rep, _ = automatic.tiny_crop_pixelation(np.asarray(small_canvas))
        self.assertIsNotNone(t_out)
        self.assertIsNotNone(t_rep)

        # 2) Period-coupled native Gaussian radius prevents over-blurring 12px blocks in a 1024px scene
        big_scene = np.full((1024, 1024, 3), 120, dtype=np.uint8)
        blk12 = Image.fromarray(rng.integers(25, 230, (12, 12, 3), dtype=np.uint8)).resize((144, 144), Image.Resampling.NEAREST)
        big_scene[400:544, 400:544] = np.asarray(blk12)
        mask12 = np.zeros((1024, 1024), dtype=bool)
        mask12[400:544, 400:544] = True
        _, rep12, _ = automatic.apply_pixelation_mask(big_scene, mask12, (1024, 1024), 0)
        self.assertIsNotNone(rep12)
        self.assertLessEqual(rep12['native_Gaussian_radius'], 7.0)
        self.assertGreaterEqual(rep12['native_Gaussian_radius'], 3.5)

        # 3) Clean portrait on bright white studio backdrop must NOT trigger haze_dehaze
        studio = np.full((320, 320, 3), 242, dtype=np.uint8)
        yy, xx = np.mgrid[0:320, 0:320]
        oval = ((xx - 160) / 70.0) ** 2 + ((yy - 160) / 90.0) ** 2 <= 1.0
        studio[oval] = np.array([155, 110, 88], dtype=np.uint8)
        studio[(yy > 235) & (np.abs(xx - 160) < 95)] = np.array([45, 42, 48], dtype=np.uint8)
        self.assertIsNone(detect_cast_dehaze(studio))

        # 4) 2x coarse chroma blotch noise triggers chroma-only denoise leaving Y untouched
        base_ycrcb = np.full((160, 160, 3), [110, 134, 118], dtype=np.uint8)
        blotch_cr = cv2.resize(rng.normal(0, 14.0, (80, 80)).astype(np.float32), (160, 160), interpolation=cv2.INTER_LINEAR)
        blotch_cb = cv2.resize(rng.normal(0, 14.0, (80, 80)).astype(np.float32), (160, 160), interpolation=cv2.INTER_LINEAR)
        noisy_ycrcb = base_ycrcb.astype(np.float32)
        noisy_ycrcb[..., 1] = np.clip(noisy_ycrcb[..., 1] + blotch_cr, 0, 255)
        noisy_ycrcb[..., 2] = np.clip(noisy_ycrcb[..., 2] + blotch_cb, 0, 255)
        blotch_rgb = cv2.cvtColor(noisy_ycrcb.astype(np.uint8), cv2.COLOR_YCrCb2RGB)
        n_det = detect_noise(blotch_rgb)
        self.assertIsNotNone(n_det)
        self.assertTrue(n_det['coarse_chroma_triggered'])
        n_out, _, n_rep = prepare_noise_denoise(blotch_rgb)
        self.assertIsNotNone(n_out)
        self.assertTrue(n_rep['coarse_chroma_triggered'])
        y_before = cv2.cvtColor(blotch_rgb, cv2.COLOR_RGB2YCrCb)[..., 0].astype(np.float32)
        y_after = cv2.cvtColor(n_out, cv2.COLOR_RGB2YCrCb)[..., 0].astype(np.float32)
        self.assertLessEqual(float(np.abs(y_after - y_before).mean()), 1.0)

        # 5) Rippled Voronoi mosaic (+-2 LSB interior ripple, missed by strict tol=1) detected by irregular_mosaic_mask
        xs = np.arange(4, 160, 9)
        ys = np.arange(4, 160, 9)
        sy, sx = np.meshgrid(ys, xs, indexing='ij')
        py = sy.ravel().astype(np.float32) + rng.uniform(-2.0, 2.0, sy.size).astype(np.float32)
        px = sx.ravel().astype(np.float32) + rng.uniform(-2.0, 2.0, sx.size).astype(np.float32)
        yy2, xx2 = np.mgrid[0:160, 0:160].astype(np.float32)
        nearest = np.argmin((xx2[..., None] - px[None, None, :]) ** 2 + (yy2[..., None] - py[None, None, :]) ** 2, axis=2)
        vor = rng.integers(30, 230, (py.size, 3), dtype=np.uint8)[nearest]
        vor_rippled = np.clip(vor.astype(np.int16) + rng.integers(-1, 2, vor.shape), 0, 255).astype(np.uint8)
        self.assertGreaterEqual(float(irregular_mosaic_mask(vor_rippled).mean()), 0.25)

    def test_protect_auto_softens_narrow_sharp_ruler_ticks_beside_blurred_head(self):
        """Sparse sharp exterior ticks on a smooth wall beside a blurred face must be surround-matched."""
        size = 256
        yy, xx = np.mgrid[0:size, 0:size].astype(np.float32)
        # Smooth bright mugshot wall with narrow sharp black ruler ticks along left/right borders
        scene = np.full((size, size, 3), 225.0, dtype=np.float32)
        for y_tick in range(12, size - 12, 16):
            scene[y_tick:y_tick + 2, 2:14] = 25.0
            scene[y_tick:y_tick + 2, size - 14:size - 2] = 25.0
        # Blurred oval head in center
        cx, cy = size * 0.5, size * 0.5
        head = ((xx - cx) / 72.0) ** 2 + ((yy - cy) / 86.0) ** 2 <= 1.0
        scene[head] = np.array([150.0, 118.0, 98.0], dtype=np.float32)
        scene[96:110, 96:118] = 45.0
        scene[96:110, 138:160] = 45.0
        scene[154:166, 108:148] = [120.0, 62.0, 62.0]
        scene_u8 = np.clip(np.rint(scene), 0, 255).astype(np.uint8)
        blurred = np.asarray(Image.fromarray(scene_u8).filter(ImageFilter.GaussianBlur(10.0)))
        blur_zone = ((xx - cx) / 84.0) ** 2 + ((yy - cy) / 96.0) ** 2 <= 1.0
        scene_u8[blur_zone] = blurred[blur_zone]

        face = [68, 52, 120, 152]
        cand, meta, ev, dec = automatic.prepare(scene_u8, scene_u8, face, None)
        self.assertIsNotNone(cand)
        self.assertEqual(dec['route'], 'protect_auto')
        self.assertTrue(meta['every_protected_native_RGB_exact'])
        pmask = np.asarray(Image.open(io.BytesIO(ev['effective-Protect-native.png']))) > 0
        derived = np.asarray(Image.open(io.BytesIO(cand)))
        np.testing.assert_array_equal(derived[pmask], scene_u8[pmask])
        # Narrow sharp ticks along left & right edges must be softened (not held out!)
        left_tick_diff = np.abs(derived[:, :16].astype(np.float32) - scene_u8[:, :16].astype(np.float32)).max()
        right_tick_diff = np.abs(derived[:, -16:].astype(np.float32) - scene_u8[:, -16:].astype(np.float32)).max()
        self.assertGreater(left_tick_diff, 30.0)
        self.assertGreater(right_tick_diff, 30.0)

    def test_face_gaussian_calibrates_from_face_box_and_blurs_entire_crop_uniformly(self):
        """face_gaussian measures coarse block strength on the face box and applies that sigma across 100% of crop."""
        rng = np.random.default_rng(717)
        native = rng.integers(30, 225, (256, 256, 3), dtype=np.uint8)
        coarse_face = Image.fromarray(rng.integers(0, 256, (6, 6, 3), dtype=np.uint8)).resize(
            (144, 144), Image.Resampling.NEAREST
        )
        native[56:200, 56:200] = np.asarray(coarse_face)
        face = [56, 56, 144, 144]
        with patch.object(
            automatic,
            'decide',
            return_value={'route': 'face_gaussian', 'reason': 'suspected_pixelation', 'degradation': {'period_native': 24}},
        ):
            cand, meta, ev, report = automatic.prepare(native, native, face, None)
        self.assertIsNotNone(cand)
        self.assertTrue(report['uniform_crop_smoothing'])
        self.assertTrue(meta['uniform_crop_smoothing'])
        self.assertEqual(report['Gaussian_radius_at256'], 3.5)
        self.assertAlmostEqual(report['crop_Gaussian_radius_native'], 3.5, places=4)
        prepared = np.asarray(Image.open(io.BytesIO(cand)))
        expected = np.asarray(Image.fromarray(native).filter(ImageFilter.GaussianBlur(3.5)))
        np.testing.assert_array_equal(prepared, expected)
        # Exterior sharp hair/background outside face box is softened identically to face interior
        self.assertGreater(float(np.any(prepared[:48, :] != native[:48, :], axis=2).mean()), 0.99)
