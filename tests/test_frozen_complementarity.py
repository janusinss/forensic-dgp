"""Frozen-head bounds cannot become a target-informed production router."""
import unittest
import numpy as np
import torch
from frozen_complementarity import (
    counts, aggregate, fixed_compositions, diagnostic_oracles, conflicting_inputs, frozen_predictions,
)


class Echo(torch.nn.Module):
    def __init__(self, mutate=False):
        super().__init__()
        self.register_buffer('counter', torch.zeros(()))
        self.mutate = mutate

    def detect(self, image):
        if self.mutate:
            self.counter.add_(1)
        return image[:, :1]*2-1


class FrozenComplementarityTests(unittest.TestCase):
    def masks(self):
        target = np.zeros((4, 4), bool); target[0, :3] = True
        valid = np.ones_like(target)
        parent = np.zeros_like(target); parent[0, :2] = True
        candidate = target.copy()
        return parent, candidate, target, valid

    def test_case_oracle_refuses_more_recall_with_more_false_positives(self):
        parent, candidate, target, valid = self.masks()
        candidate[2, 2] = True
        oracle = diagnostic_oracles(parent, candidate, target, valid, 'real')
        self.assertFalse(oracle['candidate_dominates'])
        self.assertTrue(np.array_equal(oracle['dominance_oracle'], parent))
        self.assertTrue(np.array_equal(oracle['pixel_oracle'], target))
        candidate[2, 2] = False
        self.assertTrue(diagnostic_oracles(parent, candidate, target, valid, 'real')['candidate_dominates'])

    def test_unknown_predictions_do_not_affect_counts_or_case_choice(self):
        parent, candidate, target, valid = self.masks(); valid[3] = False
        changed = candidate.copy(); changed[3] = True
        a = counts(candidate, target, valid); b = counts(changed, target, valid)
        self.assertEqual({k:v for k,v in a.items() if k!='ignored_positive_pixels'},
                         {k:v for k,v in b.items() if k!='ignored_positive_pixels'})
        self.assertEqual(b['ignored_positive_pixels'], 4)
        self.assertTrue(diagnostic_oracles(parent, changed, target, valid, 'real')['candidate_dominates'])
        target[3, 0] = True
        with self.assertRaises(ValueError):
            counts(changed, target, valid)

    def test_fixed_merge_needs_no_target_and_pixel_oracle_cannot_invent_shared_misses(self):
        parent, candidate, target, valid = self.masks()
        parent[1, 1] = True; candidate[1, 1] = True; candidate[0, 2] = False
        fixed = fixed_compositions(parent, candidate)
        self.assertTrue(np.array_equal(fixed['union'], parent | candidate))
        self.assertTrue(np.array_equal(fixed['intersection'], parent & candidate))
        bound = diagnostic_oracles(parent, candidate, target, valid, 'replay')
        self.assertTrue(np.array_equal(bound['pool_oracle'], parent))
        self.assertEqual(counts(bound['pixel_oracle'], target, valid)['fn'], 1)
        self.assertEqual(counts(bound['pixel_oracle'], target, valid)['fp'], 1)
        with self.assertRaises(ValueError):
            diagnostic_oracles(parent, candidate, target, valid, 'validation')

    def test_aggregation_preserves_clear_errors_and_full_pixel_denominators(self):
        parent, candidate, target, valid = self.masks()
        clear = np.zeros_like(target); false = clear.copy(); false[1, 1] = True
        report = aggregate([counts(candidate, target, valid), counts(false, clear, valid)])
        self.assertEqual((report['tp'], report['fp'], report['fn'], report['visible']), (3, 1, 0, 29))
        self.assertEqual(report['negative_false_positive_cases'], 1)
        self.assertAlmostEqual(report['iou'], 3/4)
        self.assertAlmostEqual(report['visible_false_positive'], 1/29)
        with self.assertRaises(ValueError):
            counts(parent.astype(float), target, valid)

    def test_identical_rgb_conflicting_supervision_is_reported_only_on_common_support(self):
        _, _, target, valid = self.masks(); rgb = np.zeros((4,4,3), np.uint8)
        rows = [{'id': 'a', 'rgb': rgb, 'target': target, 'valid': valid},
                {'id': 'b', 'rgb': rgb.copy(), 'target': np.zeros_like(target), 'valid': valid.copy()}]
        conflict = conflicting_inputs(rows)
        self.assertEqual(conflict[0]['conflicting_pixels'], 3)
        rows[1]['valid'][0, :3] = False
        self.assertEqual(conflicting_inputs(rows), [])

    def test_cpu_inference_keeps_state_and_rejects_mutating_model(self):
        rgb = np.zeros((4,4,3), np.uint8); rgb[0] = 255
        items = [{'id': 'a', 'rgb': rgb}, {'id': 'b', 'rgb': rgb.copy()}]
        model = Echo(); predictions, report = frozen_predictions(model, items, batch_size=2, progress=False)
        self.assertEqual(report['forward_images'], 2)
        self.assertTrue(report['model_state_unchanged'])
        self.assertFalse(report['optimizer_constructed'])
        self.assertTrue(predictions['a'][0].all()); self.assertFalse(predictions['a'][1:].any())
        with self.assertRaises(ValueError):
            frozen_predictions(Echo(mutate=True), items, batch_size=2, progress=False)


if __name__ == '__main__':
    unittest.main()
