"""Preserve the previously declared visual cohort, including both special cases."""
import copy
import unittest
from scripts.review_face_occlusion_focus_results import fixed_preview


class FocusReviewTests(unittest.TestCase):
    def test_declared_cohort_is_reused_without_error_ranking(self):
        rows = [dict(image=f'images/{i}.png') for i in range(25)]
        rows[8]['image'] = 'images/new_covered_40.png'
        rows[23]['glare_stratum'] = 'strong_lens_reflection'
        report = {'preview_validation_indices':[0,1,5,6,2,3,4,17,8,23]}
        before = copy.deepcopy(report)
        self.assertEqual(fixed_preview(report,rows),report['preview_validation_indices'])
        self.assertEqual(report,before)
        for cohort in ([0,1,2,3,4,5,6,7,22,23],[0,1,2,3,4,5,6,7,8,22],
                       [0,1,2,3,4,5,6,7,8,8],[0,1,2,3,4,5,6,7,8,25]):
            with self.assertRaises(ValueError): fixed_preview({'preview_validation_indices':cohort},rows)


if __name__ == '__main__': unittest.main()
