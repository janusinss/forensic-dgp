"""Geometry, fallback and read-only inference counterexamples for one crop rule."""
import unittest
import numpy as np
import torch
from torch import nn

from face_crop_diagnostic import make_crop, map_prediction, warp_target, infer_crop_masks


class PixelDetector(nn.Module):
    def __init__(self, mutate=False):
        super().__init__(); self.register_buffer('calls', torch.tensor(0)); self.mutate = mutate
    def detect(self, x):
        if self.mutate: self.calls.add_(1)
        return (x[:, :1]-.5)*20


class FaceCropTests(unittest.TestCase):
    def image(self): return np.full((64, 64, 3), 180, dtype=np.uint8)
    def transform(self): return make_crop(self.image(), np.array([[16., 16., 48., 48., .9]]))

    def test_absent_multiple_low_confidence_and_nonfinite_faces_keep_input(self):
        image = self.image()
        for boxes in (np.empty((0, 5)), np.tile([[16, 16, 48, 48, .9]], (2, 1)),
                      np.array([[16, 16, 48, 48, .59]]), np.array([[16, 16, 48, np.nan, .9]]),
                      np.array([[0, 0, 64, 64, .9]])):
            result = make_crop(image, boxes)
            self.assertFalse(result['cropped'])
            np.testing.assert_array_equal(result['rgb'], image)

    def test_one_fixed_uniform_transform_preserves_points_and_ratio(self):
        result = self.transform()
        self.assertTrue(result['cropped']); self.assertEqual(result['bounds'], [12, 12, 52, 52])
        affine = result['affine']; inverse = result['inverse']
        self.assertEqual(affine[0, 0], affine[1, 1])
        points = np.array([[12., 12.], [51., 51.], [26., 40.]])
        moved = points@affine[:, :2].T+affine[:, 2]
        np.testing.assert_allclose(moved[:2], [[0, 0], [255, 255]], atol=1e-12)
        np.testing.assert_allclose(moved@inverse[:, :2].T+inverse[:, 2], points, atol=1e-12)

    def test_composition_retains_original_pixels_outside_input_derived_roi(self):
        result = self.transform(); original = np.zeros((64, 64), dtype=bool)
        original[2, 3] = True; original[25, 25] = True
        empty = np.zeros((256, 256), dtype=bool)
        mapped = map_prediction(empty, original, result)
        self.assertTrue(mapped[2, 3]); self.assertFalse(mapped[25, 25])
        full = map_prediction(~empty, original, result)
        self.assertEqual(int(full.sum()), 40*40+1)
        np.testing.assert_array_equal(full & ~result['roi'], original & ~result['roi'])

    def test_target_mapping_is_binary_and_does_not_modify_source(self):
        result = self.transform(); target = np.zeros((64, 64), dtype=bool)
        target[22:30, 22:30] = True; before = target.copy()
        warped = warp_target(target, result)
        self.assertEqual(warped.shape, (256, 256)); self.assertEqual(warped.dtype, np.bool_)
        self.assertTrue(warped.any()); np.testing.assert_array_equal(target, before)

    def test_near_border_crop_uses_neutral_padding_without_stretch(self):
        image = self.image(); result = make_crop(image, np.array([[-8., 16., 24., 48., .9]]))
        self.assertTrue(result['cropped']); self.assertEqual(result['bounds'], [-12, 12, 28, 52])
        np.testing.assert_array_equal(result['rgb'][:, 0], np.full((256, 3), 96, np.uint8))
        self.assertEqual(int(result['roi'].sum()), 28*40)

    def test_invalid_pixel_types_and_target_shapes_are_rejected(self):
        with self.assertRaises(ValueError): make_crop(self.image().astype(float), np.empty((0, 5)))
        result = self.transform(); original = np.zeros((64, 64), dtype=bool)
        with self.assertRaises(ValueError): map_prediction(np.zeros((256, 256), np.uint8), original, result)
        with self.assertRaises(ValueError): warp_target(original[:32], result)

    def test_fallback_constructs_no_forward_and_crop_inference_preserves_state(self):
        image = self.image(); baseline = np.zeros((64, 64), dtype=bool)
        items = [{'id': 'fallback', 'original': baseline, 'transform': make_crop(image, np.empty((0, 5)))},
                 {'id': 'crop', 'original': baseline, 'transform': self.transform()}]
        model = PixelDetector(); masks, report = infer_crop_masks(model, items, progress=False)
        self.assertEqual(report['forward_count'], 1); self.assertTrue(report['model_state_unchanged'])
        self.assertEqual(int(masks['crop'].sum()), 1600)
        np.testing.assert_array_equal(masks['fallback'], baseline)
        self.assertFalse(any(p.requires_grad for p in model.parameters()))

    def test_mutating_checkpoint_and_nonfinite_logits_are_rejected(self):
        item = {'id': 'crop', 'original': np.zeros((64, 64), dtype=bool), 'transform': self.transform()}
        with self.assertRaises(ValueError): infer_crop_masks(PixelDetector(mutate=True), [item], progress=False)
        class BadDetector(PixelDetector):
            def detect(self, x): return torch.full((1, 1, 256, 256), float('nan'))
        with self.assertRaises(ValueError): infer_crop_masks(BadDetector(), [item], progress=False)


if __name__ == '__main__': unittest.main()
