"""Saved-mask counterexamples: crop changes cannot escape the declared ROI."""
import unittest
import numpy as np
from scripts.audit_face_crop_diagnostic import check_composition


class CropReturnAuditTests(unittest.TestCase):
    def test_changed_outside_roi_pixel_is_rejected_even_with_good_total_iou(self):
        original = np.zeros((64, 64), dtype=bool); predicted = original.copy()
        case = {'cropped': True, 'bounds': [12, 12, 52, 52], 'source_shape': [64, 64]}
        predicted[20, 20] = True
        self.assertEqual(check_composition(predicted, original, case)['outside_roi_differences'], 0)
        predicted[2, 2] = True
        with self.assertRaises(ValueError): check_composition(predicted, original, case)

    def test_fallback_changes_and_nonbinary_predictions_are_rejected(self):
        original = np.zeros((64, 64), dtype=bool)
        case = {'cropped': False, 'bounds': [0, 0, 64, 64], 'source_shape': [64, 64]}
        changed = original.copy(); changed[20, 20] = True
        for predicted in (changed, original.astype(float)):
            with self.assertRaises(ValueError): check_composition(predicted, original, case)


if __name__ == '__main__': unittest.main()
