"""Counterexamples for read-only reflection/retention diagnostics."""
import unittest
import numpy as np

from scripts.analyze_reflection_coverage_results import transition_counts, aggregate_counts


class RetentionTransitionTests(unittest.TestCase):
    def test_parent_recovery_and_candidate_losses_remain_separate(self):
        truth = np.array([[True, True, True, False, False, False]])
        parent = np.array([[True, True, False, True, False, False]])
        candidate = np.array([[True, False, True, False, True, False]])
        row = transition_counts(parent, candidate, truth)
        self.assertEqual(row['parent_tp'], 2)
        self.assertEqual(row['candidate_tp'], 2)
        self.assertEqual(row['lost_parent_tp'], 1)
        self.assertEqual(row['recovered_parent_fn'], 1)
        self.assertEqual(row['removed_parent_fp'], 1)
        self.assertEqual(row['added_visible_fp'], 1)

    def test_clear_case_false_positive_cannot_disappear_in_pixel_average(self):
        clear = np.zeros((2, 2), dtype=bool)
        candidate = clear.copy(); candidate[0, 0] = True
        a = transition_counts(clear, candidate, clear)
        b = transition_counts(clear, clear, clear)
        result = aggregate_counts([a, b])
        self.assertEqual(result['negative_cases'], 2)
        self.assertEqual(result['candidate_negative_false_positive_cases'], 1)
        self.assertEqual(result['candidate_visible_false_positive'], 1/8)

    def test_partition_uses_micro_counts_not_mean_case_iou(self):
        a = np.array([[True, False, False, False]])
        b = np.ones((1, 4), dtype=bool)
        partial = np.array([[True, True, False, False]])
        rows = [transition_counts(a, a, a), transition_counts(b, partial, b)]
        result = aggregate_counts(rows)
        self.assertEqual(result['parent_iou'], 1.)
        self.assertEqual(result['candidate_iou'], 3/5)
        self.assertEqual(result['candidate_missed_fraction'], 2/5)

    def test_nonbinary_shapes_or_empty_partitions_are_rejected(self):
        mask = np.zeros((2, 2), dtype=bool)
        for a, b, t in ((mask.astype(float), mask, mask),
                         (mask[:1], mask, mask), (mask.ravel(), mask, mask)):
            with self.assertRaises(ValueError): transition_counts(a, b, t)
        with self.assertRaises(ValueError): aggregate_counts([])


if __name__ == '__main__': unittest.main()
