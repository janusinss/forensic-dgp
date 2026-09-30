"""Glare measurements exclude the already annotated mouth mask."""
import unittest
import numpy as np
from scripts.audit_face_occlusion_glare_pixels import added_region_counts


class GlarePixelTests(unittest.TestCase):
    def targets(self):
        old = np.array([[False, False, False], [True, True, False]])
        current = old.copy(); current[0, :2] = True
        return current, old

    def test_accurate_mouth_mask_cannot_claim_glare_recovery(self):
        current, old = self.targets()
        result = added_region_counts(old, current, old)
        self.assertEqual(result['added_glare_pixels'], 2)
        self.assertEqual(result['recovered_added_glare_pixels'], 0)
        self.assertEqual(result['glare_recall'], 0.)
        self.assertEqual(result['original_mask_recovered_pixels'], 2)

    def test_glare_recall_and_visible_false_pixels_are_reported_separately(self):
        current, old = self.targets()
        prediction = old.copy(); prediction[0, 0] = prediction[0, 2] = True
        result = added_region_counts(prediction, current, old)
        self.assertEqual(result['glare_recall'], .5)
        self.assertEqual(result['missed_added_glare_pixels'], 1)
        self.assertEqual(result['whole_mask_false_positive_pixels'], 1)

    def test_removed_parent_pixels_empty_delta_and_wrong_shapes_are_rejected(self):
        current, old = self.targets()
        changed = current.copy(); changed[1, 0] = False
        for prediction, target, parent in ((old, changed, old), (old, old, old),
                                           (old[:, :2], current, old), (old.astype(float), current, old)):
            with self.assertRaises(ValueError): added_region_counts(prediction, target, parent)


if __name__ == '__main__': unittest.main()
