import unittest
from scripts.evaluate_presence_comparison import gated_counts

class GatedCountsTests(unittest.TestCase):
    def test_rejection_moves_true_positives_into_misses(self):
        raw=dict(tp=8,fp=2,fn=3,visible=89,empty=False,negative_fp=False)
        self.assertEqual(gated_counts(raw,.49),dict(tp=0,fp=0,fn=11,visible=89,empty=True,negative_fp=False))
        self.assertEqual(gated_counts(raw,.5),raw)

    def test_clear_rejection_is_not_empty_covered(self):
        raw=dict(tp=0,fp=2,fn=0,visible=100,empty=False,negative_fp=True)
        result=gated_counts(raw,.1)
        self.assertFalse(result['empty'])
        self.assertFalse(result['negative_fp'])
