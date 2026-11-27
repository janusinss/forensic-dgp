"""Offline and regression tests for face detection, landmark sanity, and +32% padded auto-crops."""
import json
import subprocess
import sys
import unittest
from pathlib import Path

import cv2
import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from implementation.capture_worker import (
    AUTO_CROP_MARGIN_RATIO,
    _padded_face_crop,
    _valid_face_landmarks,
    native_face_boxes,
    scan_face_records,
    scan_faces,
)


APP_ROOT = Path(__file__).resolve().parents[1]
TEST_IMAGES_DIR = APP_ROOT.parent / 'test_images'


class NativeFaceBoxTests(unittest.TestCase):
    def test_uses_each_axis_scale_and_encloses_fractional_edges(self):
        rows = [(10.25, 20.5, 8.5, 10.25)]

        self.assertEqual(native_face_boxes(rows, (1000, 750), (400, 250)), [[25, 61, 22, 32]])

    def test_clips_boxes_to_the_frame(self):
        rows = [(-4.2, 90, 30, 30), (84.25, -4, 20, 30)]

        self.assertEqual(native_face_boxes(rows, (100, 110), (100, 110)), [[0, 90, 26, 20], [84, 0, 16, 26]])

    def test_ignores_invalid_and_too_small_boxes(self):
        rows = [
            (float('nan'), 20, 25, 25),
            (20, 20, float('inf'), 25),
            (20, 20, -5, 25),
            (101, 20, 25, 25),
            (85, 20, 25, 25),
            (20, 85, 25, 25),
            (10, 10, 16, 16),
        ]

        self.assertEqual(native_face_boxes(rows, (100, 100), (100, 100)), [[10, 10, 16, 16]])

    def test_caps_valid_boxes_after_filtering(self):
        rows = [(0, 0, 1, 1)] + [(x, 0, 16, 16) for x in range(102)]

        boxes = native_face_boxes(rows, (200, 200), (200, 200))

        self.assertEqual(boxes, [[x, 0, 16, 16] for x in range(100)])

    def test_accepts_empty_detector_output(self):
        for rows in (None, []):
            with self.subTest(rows=rows):
                self.assertEqual(native_face_boxes(rows, (100, 100), (100, 100)), [])


class FaceLandmarkAndCropTests(unittest.TestCase):
    def setUp(self):
        # Typical frontal face inside a [100, 80, 100, 120] box.
        self.valid_row = [
            100.0, 80.0, 100.0, 120.0,
            132.0, 126.0,  # right eye
            168.0, 126.0,  # left eye
            150.0, 148.0,  # nose
            135.0, 168.0,  # right mouth
            165.0, 168.0,  # left mouth
            0.92,
        ]

    def test_valid_face_landmarks_enforces_geometry_and_rejects_degenerate_rows(self):
        self.assertTrue(_valid_face_landmarks(self.valid_row, 100.0, 120.0))

        cases = [
            ('short_row', self.valid_row[:10]),
            ('nan_landmark', [*self.valid_row[:4], float('nan'), *self.valid_row[5:]]),
            ('out_of_box_left', [*self.valid_row[:4], 75.0, *self.valid_row[5:]]),
            ('out_of_box_bottom', [*self.valid_row[:11], 230.0, *self.valid_row[12:]]),
            ('collapsed_eyes', [*self.valid_row[:4], 149.0, 126.0, 151.0, 126.0, *self.valid_row[8:]]),
            ('oversized_eyes', [100.0, 80.0, 100.0, 120.0, 95.0, 126.0, 195.0, 126.0, 150.0, 148.0, 135.0, 168.0, 165.0, 168.0]),
            ('collapsed_eye_mouth', [100.0, 80.0, 100.0, 120.0, 132.0, 130.0, 168.0, 130.0, 150.0, 134.0, 135.0, 136.0, 165.0, 136.0]),
            ('nose_above_band', [*self.valid_row[:8], 150.0, 112.0, *self.valid_row[10:]]),
            ('nose_below_band', [*self.valid_row[:8], 150.0, 185.0, *self.valid_row[10:]]),
        ]
        for label, row in cases:
            with self.subTest(case=label):
                self.assertFalse(_valid_face_landmarks(row, 100.0, 120.0))
        self.assertFalse(_valid_face_landmarks(self.valid_row, 0.0, 120.0))

    def test_padded_face_crop_applies_thirty_two_percent_margin_and_clamps(self):
        self.assertAlmostEqual(AUTO_CROP_MARGIN_RATIO, 0.32)
        self.assertEqual(_padded_face_crop([100, 80, 100, 120], 400, 400), [62, 42, 176, 196])
        # Edge-clamped crops preserve full head crown (top == 0) on wild3 and wild5 geometry.
        self.assertEqual(_padded_face_crop([55, 88, 289, 382], 460, 578), [0, 0, 460, 578])
        self.assertEqual(_padded_face_crop([87, 110, 278, 355], 455, 523), [0, 0, 455, 523])
        # Tiny corner box clamps safely within image bounds without negative coords or overflow.
        cx, cy, cw, ch = _padded_face_crop([0, 0, 16, 16], 40, 40)
        self.assertEqual((cx, cy, cw, ch), (0, 0, 21, 21))
        self.assertGreaterEqual(cx, 0)
        self.assertGreaterEqual(cy, 0)
        self.assertLessEqual(cx + cw, 40)
        self.assertLessEqual(cy + ch, 40)

    def test_padded_face_crop_caps_side_at_1600(self):
        cx, cy, cw, ch = _padded_face_crop([100, 100, 1500, 1500], 2400, 2400)
        self.assertEqual((cx, cy, cw, ch), (0, 0, 1600, 1600))


class MultiPassFaceScanTests(unittest.TestCase):
    @staticmethod
    def _box_iou(a, b):
        ix1, iy1 = max(a[0], b[0]), max(a[1], b[1])
        ix2, iy2 = min(a[0] + a[2], b[0] + b[2]), min(a[1] + a[3], b[1] + b[3])
        inter = max(0, ix2 - ix1) * max(0, iy2 - iy1)
        union = a[2] * a[3] + b[2] * b[3] - inter
        return inter / union if union > 0 else 0.0

    def test_non_face_noise_stripes_and_flat_textures_produce_zero_false_positives(self):
        rng = np.random.default_rng(77)
        yy, xx = np.mgrid[0:320, 0:320]
        textures = {
            'white_noise': rng.integers(0, 256, (320, 320, 3), dtype=np.uint8),
            'blurred_noise': cv2.GaussianBlur(rng.integers(0, 256, (320, 320, 3), dtype=np.uint8), (0, 0), 6.0),
            'horizontal_stripes': np.repeat((((yy // 8) % 2) * 200 + 20).astype(np.uint8)[..., None], 3, axis=2),
            'diagonal_stripes': np.repeat(((((xx + yy) // 10) % 2) * 190 + 25).astype(np.uint8)[..., None], 3, axis=2),
            'checkerboard': np.repeat(((((xx // 16) + (yy // 16)) % 2) * 220 + 15).astype(np.uint8)[..., None], 3, axis=2),
            'flat_gray': np.full((256, 256, 3), 128, dtype=np.uint8),
        }
        for name, img in textures.items():
            with self.subTest(texture=name):
                self.assertEqual(scan_face_records(img), [])
                self.assertEqual(scan_faces(img), [])

    @unittest.skipUnless(
        all((TEST_IMAGES_DIR / n).exists() for n in ('wild3.png', 'wild4.png', 'wild5.png')),
        'wild test images missing',
    )
    def test_wild3_and_wild4_detect_single_mosaic_face_with_auto_crop_margin(self):
        for name in ('wild3.png', 'wild4.png'):
            with self.subTest(image=name):
                img = cv2.imread(str(TEST_IMAGES_DIR / name))
                h, w = img.shape[:2]
                records = scan_face_records(img)
                self.assertEqual(len(records), 1)
                self.assertEqual(scan_faces(img), [records[0]['box']])
                bx, by, bw, bh = records[0]['box']
                cx, cy, cw, ch = records[0]['crop']
                # Must detect the main central mosaic face, not a tiny corner hair ghost.
                self.assertGreaterEqual(bw, 200)
                self.assertGreaterEqual(bh, 250)
                margin = int(max(bw, bh) * AUTO_CROP_MARGIN_RATIO + 0.5)
                self.assertEqual(records[0]['crop'], [max(0, bx - margin), max(0, by - margin),
                                                      min(w, bx + bw + margin) - max(0, bx - margin),
                                                      min(h, by + bh + margin) - max(0, by - margin)])
                self.assertEqual(cy, 0)
                self.assertGreaterEqual(cx, 0)
                self.assertLessEqual(cx + cw, w)
                self.assertLessEqual(cy + ch, h)
                if name == 'wild4.png':
                    # Balanced framing preserves crown without over-zooming out into mosaic degradation.
                    self.assertGreaterEqual(cw, 500)
                    self.assertLessEqual(cw, 580)

    @unittest.skipUnless((TEST_IMAGES_DIR / 'wild5.png').exists(), 'wild5.png missing')
    def test_heavy_gaussian_and_motion_blurred_faces_detected_accurately(self):
        base = cv2.imread(str(TEST_IMAGES_DIR / 'wild5.png'))
        ref = scan_face_records(base)
        self.assertEqual(len(ref), 1)
        ref_box = ref[0]['box']
        # Wild5 auto-crop must preserve top head crown (no top clipping).
        self.assertEqual(ref[0]['crop'][1], 0)

        motion_kernel = np.eye(15, dtype=np.float32) / 15.0
        degraded = {
            'gaussian_sigma_8': cv2.GaussianBlur(base, (0, 0), sigmaX=8.0, sigmaY=8.0),
            'gaussian_sigma_12': cv2.GaussianBlur(base, (0, 0), sigmaX=12.0, sigmaY=12.0),
            'motion_15px': cv2.filter2D(base, -1, motion_kernel),
        }
        for label, img in degraded.items():
            with self.subTest(degradation=label):
                recs = scan_face_records(img)
                self.assertEqual(len(recs), 1)
                self.assertGreaterEqual(self._box_iou(recs[0]['box'], ref_box), 0.50)
                bx, by, bw, bh = recs[0]['box']
                cx, cy, cw, ch = recs[0]['crop']
                margin = int(max(bw, bh) * AUTO_CROP_MARGIN_RATIO + 0.5)
                expected_left, expected_top = max(0, bx - margin), max(0, by - margin)
                self.assertEqual([cx, cy, cw, ch], [
                    expected_left, expected_top,
                    min(img.shape[1], bx + bw + margin) - expected_left,
                    min(img.shape[0], by + bh + margin) - expected_top,
                ])

    @unittest.skipUnless(
        (TEST_IMAGES_DIR / 'wild5.png').exists()
        and (TEST_IMAGES_DIR / '851128671_1090321427166748_3093066005956191228_n.png').exists(),
        'multi-face fixture images missing',
    )
    def test_keeps_real_secondary_face_and_detect_details_cli(self):
        small = cv2.imread(str(TEST_IMAGES_DIR / '851128671_1090321427166748_3093066005956191228_n.png'))
        main = cv2.resize(cv2.imread(str(TEST_IMAGES_DIR / 'wild5.png')), (300, 345))
        canvas = np.full((600, 800, 3), 80, dtype=np.uint8)
        canvas[50:50 + small.shape[0], 40:40 + small.shape[1]] = small
        canvas[150:495, 420:720] = cv2.GaussianBlur(main, (0, 0), sigmaX=8.0, sigmaY=8.0)

        recs = scan_face_records(canvas)
        self.assertEqual(len(recs), 2)

        ok, encoded = cv2.imencode('.png', canvas)
        self.assertTrue(ok)
        worker = APP_ROOT / 'implementation' / 'capture_worker.py'
        proc = subprocess.run(
            [sys.executable, '-B', str(worker), 'detect-details'],
            input=encoded.tobytes(),
            stdout=subprocess.PIPE,
            stderr=subprocess.PIPE,
            timeout=15,
            check=True,
        )
        cli_recs = json.loads(proc.stdout)
        self.assertEqual(len(cli_recs), 2)
        for cli_rec, py_rec in zip(cli_recs, recs):
            self.assertEqual(cli_rec['box'], py_rec['box'])
            self.assertEqual(cli_rec['crop'], py_rec['crop'])
            self.assertAlmostEqual(cli_rec['score'], py_rec['score'], places=5)
            self.assertEqual(len(cli_rec['landmarks']), 5)
            self.assertIsInstance(cli_rec['roll_deg'], float)

    @unittest.skipUnless((TEST_IMAGES_DIR / 'wild5.png').exists(), 'wild5.png missing')
    def test_pass2b_backlit_dark_face_lift_recovers_face_without_fp_on_dark_noise(self):
        """Pass 2b gamma lift recovers backlit/darkened faces while staying 0-FP on dark noise."""
        base = cv2.imread(str(TEST_IMAGES_DIR / 'wild5.png'))
        # Create a backlit scene: bright 235 border around a darkened 0.22x foreground face.
        h, w = base.shape[:2]
        canvas = np.full((h + 120, w + 180, 3), 235, dtype=np.uint8)
        dark_face = np.clip(np.rint(base.astype(np.float32) * 0.22), 0, 255).astype(np.uint8)
        canvas[60 : 60 + h, 90 : 90 + w] = dark_face
        recs = scan_face_records(canvas)
        self.assertGreaterEqual(len(recs), 1)
        # Ensure the detected face center lies inside the darkened portrait region.
        bx, by, bw, bh = recs[0]['box']
        self.assertTrue(90 <= bx + bw * 0.5 <= 90 + w)
        self.assertTrue(60 <= by + bh * 0.5 <= 60 + h)

        # Dark random noise + bright border must still yield 0 false positives.
        rng = np.random.default_rng(4040)
        dark_noise = np.full((320, 320, 3), 235, dtype=np.uint8)
        dark_noise[40:280, 40:280] = rng.integers(0, 35, (240, 240, 3), dtype=np.uint8)
        self.assertEqual(scan_face_records(dark_noise), [])


class Phase2ScrfdAndFiqaTests(unittest.TestCase):
    """Phase 2 runtime-safety, dual-backend, spotlight regression, and FIQA scorer tests."""

    def tearDown(self):
        from implementation import fiqa_scorer, scrfd_detector

        scrfd_detector.reset_scrfd_cache()
        fiqa_scorer.reset_fiqa_cache()

    def test_ordinary_file_rejects_symlink_and_windows_reparse_attributes(self):
        import stat
        import tempfile
        from types import SimpleNamespace
        from unittest.mock import patch
        from implementation import fiqa_scorer, scrfd_detector

        with tempfile.TemporaryDirectory() as tmpdir:
            dummy = Path(tmpdir) / 'model.onnx'
            dummy.write_bytes(b'abc')

            # 1) Symlink mode on file -> rejected by both verifiers.
            fake_link = SimpleNamespace(st_mode=stat.S_IFLNK | 0o777, st_file_attributes=0)
            with patch.object(Path, 'lstat', return_value=fake_link):
                self.assertFalse(scrfd_detector._is_ordinary_entry(dummy, directory=False))
                self.assertFalse(fiqa_scorer._is_ordinary_entry(dummy, directory=False))
                self.assertFalse(scrfd_detector._verify_onnx_file(dummy, 3, '0' * 64))
                self.assertFalse(fiqa_scorer._verify_onnx_file(dummy, 3, '0' * 64))

            # 2) Windows reparse point attribute (1024) on ancestor dir -> rejected.
            orig_lstat = Path.lstat
            parent_norm = str(dummy.parent.resolve()).lower()

            def _reparse_parent_lstat(path_self):
                if str(path_self.resolve()).lower() == parent_norm:
                    return SimpleNamespace(st_mode=stat.S_IFDIR | 0o755, st_file_attributes=1024)
                return orig_lstat(path_self)

            with patch.object(Path, 'lstat', _reparse_parent_lstat):
                self.assertFalse(scrfd_detector._is_ordinary_entry(dummy.parent, directory=True))
                self.assertFalse(fiqa_scorer._is_ordinary_entry(dummy.parent, directory=True))
                self.assertFalse(scrfd_detector._verify_onnx_file(dummy, 3, '0' * 64))
                self.assertFalse(fiqa_scorer._verify_onnx_file(dummy, 3, '0' * 64))

    def test_sha256_mismatch_degrades_to_none_and_falls_back_to_yunet(self):
        import tempfile
        from unittest.mock import patch
        from implementation import fiqa_scorer, scrfd_detector

        with tempfile.TemporaryDirectory() as tmpdir:
            bad_scrfd = Path(tmpdir) / scrfd_detector.SCRFD_FILENAME
            bad_scrfd.write_bytes(b'corrupted-scrfd-weights')
            bad_fiqa = Path(tmpdir) / fiqa_scorer.FIQA_FILENAME
            bad_fiqa.write_bytes(b'corrupted-fiqa-weights')

            self.assertFalse(
                scrfd_detector._verify_onnx_file(bad_scrfd, scrfd_detector.SCRFD_BYTES, scrfd_detector.SCRFD_SHA256)
            )
            self.assertFalse(
                fiqa_scorer._verify_onnx_file(bad_fiqa, fiqa_scorer.FIQA_BYTES, fiqa_scorer.FIQA_SHA256)
            )

            scrfd_detector.reset_scrfd_cache()
            with patch.object(scrfd_detector, '_resolve_scrfd_model_path', return_value=None):
                img = np.zeros((128, 128, 3), dtype=np.uint8)
                self.assertIsNone(scrfd_detector.scan_scrfd_face_records(img))
                if (TEST_IMAGES_DIR / 'wild5.png').exists():
                    w5 = cv2.imread(str(TEST_IMAGES_DIR / 'wild5.png'))
                    recs = scan_face_records(w5)
                    self.assertEqual(len(recs), 1)
                    self.assertNotEqual(recs[0].get('detector'), scrfd_detector.DETECTOR_TAG)

            fiqa_scorer.reset_fiqa_cache()
            with patch.object(fiqa_scorer, '_resolve_fiqa_model_path', return_value=None):
                self.assertIsNone(fiqa_scorer.score_face_crop_fiqa(np.full((64, 64, 3), 120, dtype=np.uint8)))

    def test_real_ort_pybind11_and_cv2_error_fallbacks_are_bounded(self):
        from unittest.mock import patch
        import onnxruntime as ort
        from onnxruntime.capi import onnxruntime_pybind11_state as ort_state
        from implementation import fiqa_scorer, scrfd_detector

        sample_bgr = (
            cv2.imread(str(TEST_IMAGES_DIR / 'wild5.png'))
            if (TEST_IMAGES_DIR / 'wild5.png').exists()
            else np.full((128, 128, 3), 120, dtype=np.uint8)
        )
        sample_rgb = cv2.cvtColor(sample_bgr, cv2.COLOR_BGR2RGB)

        exc_cases = [
            ('NoSuchFile', ort_state.NoSuchFile('missing model')),
            ('Fail', ort_state.Fail('ort execution failed')),
            ('cv2.error', cv2.error('dnn forward failed')),
        ]
        for label, exc in exc_cases:
            with self.subTest(exc=label):
                scrfd_detector.reset_scrfd_cache()
                with patch.object(ort, 'InferenceSession', side_effect=exc):
                    self.assertIsNone(scrfd_detector.scan_scrfd_face_records(sample_bgr))
                    if (TEST_IMAGES_DIR / 'wild5.png').exists():
                        fallback_recs = scan_face_records(sample_bgr)
                        self.assertEqual(len(fallback_recs), 1)
                        self.assertNotEqual(fallback_recs[0].get('detector'), scrfd_detector.DETECTOR_TAG)

                fiqa_scorer.reset_fiqa_cache()
                with patch.object(ort, 'InferenceSession', side_effect=exc):
                    self.assertIsNone(fiqa_scorer.score_face_crop_fiqa(sample_rgb))

    @unittest.skipUnless((TEST_IMAGES_DIR / 'wild5.png').exists(), 'wild5.png missing')
    def test_cv2_dnn_backend_fallback_when_onnxruntime_import_fails(self):
        import builtins
        from unittest.mock import patch
        from implementation import scrfd_detector

        w5 = cv2.imread(str(TEST_IMAGES_DIR / 'wild5.png'))
        scrfd_detector.reset_scrfd_cache()
        ort_recs = scrfd_detector.scan_scrfd_face_records(w5)
        self.assertIsNotNone(ort_recs)
        self.assertEqual(len(ort_recs), 1)

        real_import = builtins.__import__

        def _block_ort(name, globals=None, locals=None, fromlist=(), level=0):
            if name == 'onnxruntime' or name.startswith('onnxruntime.'):
                raise ImportError('simulated missing onnxruntime in isolated capture venv')
            return real_import(name, globals, locals, fromlist, level)

        scrfd_detector.reset_scrfd_cache()
        with patch.object(builtins, '__import__', side_effect=_block_ort):
            backend = scrfd_detector._load_backend()
            self.assertIsNotNone(backend)
            self.assertEqual(backend[0], 'cv2_dnn')
            cv_recs = scrfd_detector.scan_scrfd_face_records(w5)
        self.assertIsNotNone(cv_recs)
        self.assertEqual(len(cv_recs), 1)
        self.assertGreaterEqual(MultiPassFaceScanTests._box_iou(ort_recs[0]['box'], cv_recs[0]['box']), 0.98)

    def test_nan_and_inf_outputs_are_sanitized_cleanly(self):
        from unittest.mock import MagicMock, patch
        from implementation import fiqa_scorer, scrfd_detector

        # SCRFD NaN/Inf raw outputs -> sanitized to 0.0 -> empty list (no crash / no NaN).
        nan_outs = [
            np.full((12800, 1), np.nan, dtype=np.float32),
            np.full((3200, 1), np.inf, dtype=np.float32),
            np.full((800, 1), -np.inf, dtype=np.float32),
            np.full((12800, 4), np.nan, dtype=np.float32),
            np.full((3200, 4), np.nan, dtype=np.float32),
            np.full((800, 4), np.nan, dtype=np.float32),
            np.full((12800, 10), np.nan, dtype=np.float32),
            np.full((3200, 10), np.nan, dtype=np.float32),
            np.full((800, 10), np.nan, dtype=np.float32),
        ]
        fake_sess = MagicMock()
        fake_sess.run.return_value = nan_outs
        fake_backend = ('ort', fake_sess, 'input.1', scrfd_detector.SCRFD_OUT_NAMES)
        with patch.object(scrfd_detector, '_load_backend', return_value=fake_backend):
            recs = scrfd_detector.scan_scrfd_face_records(np.full((128, 128, 3), 100, dtype=np.uint8))
        self.assertEqual(recs, [])

        # FIQA NaN/Inf raw outputs -> sanitized finite floats.
        fake_fiqa = MagicMock()
        fake_fiqa.run.return_value = [np.array([[np.nan], [np.inf], [-np.inf]], dtype=np.float32)]
        with patch.object(fiqa_scorer, '_load_fiqa_session', return_value=(fake_fiqa, 'input', 'output')):
            patches = [np.full((64, 64, 3), 120, dtype=np.uint8) for _ in range(3)]
            scores = fiqa_scorer.score_face_crops_fiqa(patches)
        self.assertEqual(len(scores), 3)
        self.assertTrue(all(isinstance(s, float) and np.isfinite(s) for s in scores))

    def test_capture_install_probe_runs_under_runpy(self):
        import runpy
        from implementation import capture_install

        worker = APP_ROOT / 'implementation' / 'capture_worker.py'
        ns = runpy.run_path(str(worker))
        self.assertEqual(ns['scan_faces'](np.zeros((64, 64, 3), dtype=np.uint8)), [])
        py = capture_install.capture_python()
        if py.is_file():
            self.assertTrue(capture_install.verified(py))

    def test_real_image_spotlight_regressions_and_mosaic08_yunet_fallback(self):
        from implementation import scrfd_detector

        rest_dir = TEST_IMAGES_DIR / 'internet_samples' / 'restoration'
        cov_dir = TEST_IMAGES_DIR / 'internet_samples' / 'covering'

        c06 = rest_dir / 'cctv_06_archival_grainy.jpg'
        if c06.is_file():
            recs = scan_face_records(cv2.imread(str(c06)))
            self.assertEqual(len(recs), 1)
            self.assertEqual(recs[0].get('detector'), scrfd_detector.DETECTOR_TAG)
            # Locks onto red-circled suspect (around x=432), not top-right corner face (x>=1000).
            self.assertLess(recs[0]['box'][0], 900)

        m04 = rest_dir / 'mosaic_04_police_street.png'
        if m04.is_file():
            recs = scan_face_records(cv2.imread(str(m04)))
            self.assertGreaterEqual(len(recs), 1)
            self.assertEqual(recs[0].get('detector'), scrfd_detector.DETECTOR_TAG)
            wheel_fp = [182, 303, 97, 103]
            for r in recs:
                self.assertLess(MultiPassFaceScanTests._box_iou(r['box'], wheel_fp), 0.25)

        m03 = cov_dir / 'mask_03_kn95_sunglasses.jpg'
        if m03.is_file():
            recs = scan_face_records(cv2.imread(str(m03)))
            self.assertEqual(len(recs), 1)
            self.assertEqual(recs[0].get('detector'), scrfd_detector.DETECTOR_TAG)
            self.assertLess(abs(recs[0]['roll_deg']), 8.0)

        c05 = cov_dir / 'compound_05_cctv_mask_hood.jpg'
        if c05.is_file():
            recs = scan_face_records(cv2.imread(str(c05)))
            self.assertEqual(len(recs), 1)
            self.assertEqual(recs[0].get('detector'), scrfd_detector.DETECTOR_TAG)

        m08 = rest_dir / 'mosaic_08_indoor_coarse.png'
        if m08.is_file():
            recs = scan_face_records(cv2.imread(str(m08)))
            self.assertGreaterEqual(len(recs), 1)
            self.assertNotEqual(recs[0].get('detector'), scrfd_detector.DETECTOR_TAG)

    @unittest.skipUnless((TEST_IMAGES_DIR / 'wild5.png').exists(), 'wild5.png missing')
    def test_fiqa_blur_monotonicity_single_batch_and_video_lucky_frame_ranking(self):
        from unittest.mock import patch
        from implementation import fiqa_scorer

        bgr = cv2.imread(str(TEST_IMAGES_DIR / 'wild5.png'))
        rgb = cv2.cvtColor(bgr, cv2.COLOR_BGR2RGB)
        recs = scan_face_records(bgr)
        self.assertGreaterEqual(len(recs), 1)
        bx, by, bw, bh = recs[0]['box']
        sharp = rgb[by:by + bh, bx:bx + bw]
        blur_mid = cv2.GaussianBlur(sharp, (0, 0), sigmaX=3.5)
        blur_heavy = cv2.GaussianBlur(sharp, (0, 0), sigmaX=8.0)

        # Single batched ORT call scores all 3 crops monotonically.
        sess_tuple = fiqa_scorer._load_fiqa_session()
        self.assertIsNotNone(sess_tuple)
        sess, in_name, out_name = sess_tuple
        orig_run = sess.run
        call_count = {'n': 0}

        def _counting_run(*args, **kwargs):
            call_count['n'] += 1
            return orig_run(*args, **kwargs)

        with patch.object(sess, 'run', side_effect=_counting_run):
            scores = fiqa_scorer.score_face_crops_fiqa([sharp, blur_mid, blur_heavy])
        self.assertEqual(call_count['n'], 1)
        self.assertTrue(all(s is not None for s in scores))
        self.assertGreater(scores[0], scores[1])
        self.assertGreater(scores[1], scores[2])

        # Video Lucky-Frame ranking with eDifFIQA-T primary and Laplacian fallback.
        frames = [
            {'frame_id': 'heavy', 'frame': cv2.GaussianBlur(rgb, (0, 0), sigmaX=8.0), 'box': [bx, by, bw, bh]},
            {'frame_id': 'sharp', 'frame': rgb, 'box': [bx, by, bw, bh]},
            {'frame_id': 'mid', 'frame': cv2.GaussianBlur(rgb, (0, 0), sigmaX=3.5), 'box': [bx, by, bw, bh]},
        ]
        ranked = fiqa_scorer.rank_video_frames_fiqa(frames)
        self.assertEqual([r['frame_id'] for r in ranked], ['sharp', 'mid', 'heavy'])
        self.assertEqual(ranked[0]['scorer'], 'ediffiqa_tiny')

        with patch.object(fiqa_scorer, '_load_fiqa_session', return_value=None):
            ranked_fb = fiqa_scorer.rank_video_frames_fiqa(frames)
        self.assertEqual(ranked_fb[0]['frame_id'], 'sharp')
        self.assertEqual(ranked_fb[0]['scorer'], 'laplacian_fallback')
        self.assertIsNone(ranked_fb[0]['fiqa_score'])

        # Malformed box (NaN) and cv2.error inside Laplacian fallback degrade cleanly to 0.0.
        malformed = [{'frame_id': 'bad', 'frame': rgb, 'box': [float('nan'), 0, 10, 10]}]
        with patch.object(fiqa_scorer, '_load_fiqa_session', return_value=None):
            ranked_bad = fiqa_scorer.rank_video_frames_fiqa(malformed)
        self.assertEqual(len(ranked_bad), 1)
        self.assertEqual(ranked_bad[0]['laplacian_var'], 0.0)
        with patch.object(fiqa_scorer, '_load_fiqa_session', return_value=None), \
             patch.object(cv2, 'Laplacian', side_effect=cv2.error('laplacian fail')):
            ranked_err = fiqa_scorer.rank_video_frames_fiqa(frames[:1])
        self.assertEqual(ranked_err[0]['laplacian_var'], 0.0)

    def test_scrfd_roll_refinement_survives_yunet_fallback_runtime_error(self):
        from unittest.mock import patch
        from implementation import capture_worker

        dummy = np.zeros((120, 120, 3), dtype=np.uint8)
        scrfd_recs = [{
            'box': [40, 40, 32, 32],
            'crop': [29, 29, 54, 54],
            'score': 0.65,
            'landmarks': [[48.0, 50.0], [64.0, 56.0], [56.0, 58.0], [50.0, 64.0], [62.0, 66.0]],
            'roll_deg': 20.5,
            'detector': 'scrfd_2.5g',
        }]
        for label, exc in (
            ('RuntimeError', RuntimeError('missing yunet')),
            ('cv2.error', cv2.error('yunet dnn failure')),
        ):
            with self.subTest(exc=label):
                with patch.object(capture_worker, '_scan_yunet_face_records', side_effect=exc):
                    refined = capture_worker._refine_scrfd_roll_jitter(dummy, scrfd_recs)
                self.assertEqual(len(refined), 1)
                self.assertEqual(refined[0]['box'], [40, 40, 32, 32])
                self.assertLessEqual(abs(refined[0]['roll_deg']), 12.0)


