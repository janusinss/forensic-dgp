import unittest
import numpy as np
from scripts.evaluate_coverage_results import recount


class CoverageRecountTests(unittest.TestCase):
    def test_counts_positive_empty_and_clear_false_positive(self):
        targets=[np.array([[1,0],[0,0]],dtype=bool),np.zeros((2,2),bool)]
        predictions=[np.zeros((2,2),bool),np.ones((2,2),bool)]
        m=recount(predictions,targets)
        self.assertEqual(m['iou'],0)
        self.assertEqual(m['missed_fraction'],1)
        self.assertEqual(m['visible_false_positive'],4/7)
        self.assertEqual(m['empty_mask_cases'],1)
        self.assertEqual(m['negative_false_positive_cases'],1)

    def test_perfect_masks_and_missing_or_wrong_size_rejected(self):
        t=[np.eye(2,dtype=bool),np.zeros((2,2),bool)]
        self.assertEqual(recount(t,t)['iou'],1)
        with self.assertRaises(ValueError):recount(t[:1],t)
        with self.assertRaises(ValueError):recount([np.ones((3,3),bool)],t[:1])
        with self.assertRaises(ValueError):recount([],[])
