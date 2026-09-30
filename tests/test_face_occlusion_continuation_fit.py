"""Area diagnostics distinguish small covered targets from clear negatives."""
import copy
import unittest
import numpy as np

from scripts.compare_replay_fit import counts
from scripts.diagnose_face_occlusion_continuation_fit import area_summary


class ContinuationFitTests(unittest.TestCase):
    def record(self, case, target_pixels, predicted_pixels, total=100):
        truth = np.arange(total) < target_pixels
        prediction = np.arange(total) < predicted_pixels
        return {'case': case, 'counts': counts(prediction, truth)}

    def test_small_targets_and_clear_negatives_remain_separate_with_pixel_aggregation(self):
        result = area_summary([self.record(0, 0, 1), self.record(1, 1, 1), self.record(2, 10, 2)])
        self.assertAlmostEqual(result['all']['iou'], 3 / 12)
        self.assertEqual(result['by_area']['clear']['negative_false_positive_cases'], 1)
        self.assertEqual(result['by_area']['covered_le_1pct']['iou'], 1.)
        self.assertEqual(result['by_area']['covered_gt_1pct']['iou'], .2)

    def test_one_percent_boundary_uses_target_pixels_not_predictions(self):
        result = area_summary([self.record(0, 10, 0, 1000), self.record(1, 11, 1, 1000)])
        self.assertEqual(result['by_area']['covered_le_1pct']['cases'], 1)
        self.assertEqual(result['by_area']['covered_gt_1pct']['cases'], 1)
        self.assertEqual(result['by_area']['covered_le_1pct']['empty_mask_cases'], 1)

    def test_duplicates_and_invalid_target_confusion_counts_are_rejected(self):
        rows = [self.record(0, 1, 1)]
        with self.assertRaises(ValueError): area_summary(rows + rows)
        for field, value in [('tp', -1), ('fn', float('nan')), ('positive', False), ('fp', 100)]:
            changed = copy.deepcopy(rows); changed[0]['counts'][field] = value
            with self.assertRaises(ValueError): area_summary(changed)


if __name__ == '__main__': unittest.main()
