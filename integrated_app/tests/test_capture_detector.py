"""Offline and regression tests for face detection, landmark sanity, and +20% padded auto-crops."""
import json
import subprocess
import sys
import unittest
from pathlib import Path

import cv2
import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from implementation.capture_worker import (
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

    def test_padded_face_crop_applies_twenty_percent_margin_and_clamps(self):
        self.assertEqual(_padded_face_crop([100, 80, 100, 120], 400, 400), [76, 56, 148, 168])
        # Edge-clamped crop matching previous auto crop on wild3 geometry.
        self.assertEqual(_padded_face_crop([55, 88, 289, 382], 460, 578), [0, 12, 420, 534])

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
    def test_wild3_and_wild4_detect_single_mosaic_face_with_previous_auto_zoom(self):
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
                margin = int(max(bw, bh) * 0.20 + 0.5)
                self.assertEqual(records[0]['crop'], [max(0, bx - margin), max(0, by - margin),
                                                      min(w, bx + bw + margin) - max(0, bx - margin),
                                                      min(h, by + bh + margin) - max(0, by - margin)])
                self.assertGreaterEqual(cx, 0)
                self.assertGreaterEqual(cy, 0)
                self.assertLessEqual(cx + cw, w)
                self.assertLessEqual(cy + ch, h)

    @unittest.skipUnless((TEST_IMAGES_DIR / 'wild5.png').exists(), 'wild5.png missing')
    def test_heavy_gaussian_and_motion_blurred_faces_detected_accurately(self):
        base = cv2.imread(str(TEST_IMAGES_DIR / 'wild5.png'))
        ref = scan_face_records(base)
        self.assertEqual(len(ref), 1)
        ref_box = ref[0]['box']

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
                margin = int(max(bw, bh) * 0.20 + 0.5)
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
