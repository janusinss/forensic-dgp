"""Offline tests for mapping detected face boxes back to the frozen frame."""
import sys
import unittest
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from implementation.capture_worker import native_face_boxes


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
