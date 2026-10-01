"""Training-only membership and diagnostic advancement safeguards."""
import copy
import unittest
from scripts.run_face_crop_diagnostic import training_members, diagnostic_signal


class FaceCropProtocolTests(unittest.TestCase):
    def rows(self):
        return [{'split': 'train' if i < 73 else 'validation' if i < 98 else 'test',
                 'kind': 'covered' if i < 47 else 'uncovered', 'image_sha256': f'{i:064x}'} for i in range(105)]

    def scores(self):
        return {'training_real': {'iou': .9, 'visible_false_positive': .001, 'negative_false_positive_cases': 0},
                'training_reflection': {'iou': .8, 'negative_false_positive_cases': 0},
                'lens_recovered_pixels': 100}

    def test_all_and_only_the_fixed_training_rows_are_admitted(self):
        rows = self.rows(); selected = training_members(rows)
        self.assertEqual(len(selected), 73)
        self.assertEqual({r['image_sha256'] for r in selected}, {f'{i:064x}' for i in range(73)})

    def test_changed_split_duplicate_source_or_missing_photo_is_rejected(self):
        rows = self.rows(); changed = copy.deepcopy(rows); changed[-1]['split'] = 'train'
        duplicate = copy.deepcopy(rows); duplicate[-1]['image_sha256'] = duplicate[0]['image_sha256']
        for invalid in (changed, duplicate, rows[:-1]):
            with self.assertRaises(ValueError): training_members(invalid)

    def test_lens_gain_cannot_override_whole_mask_clear_or_fixture_regression(self):
        original = self.scores(); good = self.scores(); good['lens_recovered_pixels'] += 1
        self.assertTrue(diagnostic_signal(original, good)['training_signal_passes'])
        variants = []
        for domain, metric, value in (('training_real', 'iou', .89),
                                     ('training_real', 'visible_false_positive', .002),
                                     ('training_real', 'negative_false_positive_cases', 1),
                                     ('training_reflection', 'iou', .79),
                                     ('training_reflection', 'negative_false_positive_cases', 1)):
            changed = copy.deepcopy(good); changed[domain][metric] = value; variants.append(changed)
        unchanged = self.scores(); variants.append(unchanged)
        for candidate in variants:
            self.assertFalse(diagnostic_signal(original, candidate)['training_signal_passes'])


if __name__ == '__main__': unittest.main()
