"""Counterexamples for native occlusion annotation and unsupported regions."""
import unittest
import numpy as np
from scripts.prepare_real_reflection_proposals import prepare_target, proposal_status


class RealReflectionProposalTests(unittest.TestCase):
    def test_pending_photo_never_becomes_automatic_empty_target(self):
        self.assertEqual(proposal_status(171), 'proposed_covering')
        self.assertEqual(proposal_status(207), 'proposed_clear')
        self.assertEqual(proposal_status(218), 'pending_native_review')
        self.assertEqual(proposal_status(119), 'pending_native_review')
        with self.assertRaises(ValueError):
            proposal_status(999)

    def test_uniform_geometry_preserves_shape_and_excludes_unknown_region(self):
        rgb = np.full((120, 80, 3), 130, np.uint8)
        target = [[(25, 25), (45, 25), (45, 45), (25, 45)]]
        unknown = [[(0, 90), (79, 90), (79, 119), (0, 119)]]
        result = prepare_target(rgb, target, unknown, size=256)
        self.assertEqual(result['affine'][0, 0], result['affine'][1, 1])
        self.assertEqual(result['affine'][0, 1], 0)
        self.assertEqual(result['affine'][1, 0], 0)
        self.assertFalse(result['native_valid'][100].any())
        self.assertFalse((result['mask'] & ~result['valid']).any())
        self.assertTrue(np.all(result['rgb'][~result['source_valid']] == 96))
        self.assertEqual(int(result['rgb'][220, 128, 0]), 130)
        y, x = np.where(result['mask'])
        self.assertLessEqual(abs((x.max()-x.min())-(y.max()-y.min())), 1)
        self.assertTrue(result['native_mask'][35, 35])

    def test_outside_nonfinite_degenerate_or_unknown_target_is_rejected(self):
        rgb = np.zeros((80, 80, 3), np.uint8)
        bad = [[(-1, 3), (10, 3), (10, 20)],
               [(2, 2), (81, 2), (10, 10)],
               [(1, 1), (5, 5), (10, 10)],
               [(2, 2), (float('nan'), 10), (10, 10)]]
        for polygon in bad:
            with self.assertRaises(ValueError):
                prepare_target(rgb, [polygon], [], size=256)
        region = [[(20, 20), (40, 20), (40, 40), (20, 40)]]
        with self.assertRaises(ValueError):
            prepare_target(rgb, region, region, size=256)


if __name__ == '__main__':
    unittest.main()
