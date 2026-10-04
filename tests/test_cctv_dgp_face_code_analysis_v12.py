"""Numeric interpretation boundaries; no networks or backward calls."""
import numpy as np
import unittest

from scripts.analyze_cctv_dgp_face_code_fit_v12 import code_statistics


class CodeAnalysisContract(unittest.TestCase):
    def test_lower_cross_entropy_can_coexist_with_lower_code_accuracy(self):
        truth = np.array([[0, 0]], dtype=np.int64)
        support = np.ones_like(truth, dtype=bool)
        before = np.array([[[1.0, 0.0, -1.0], [-9.0, 9.0, 0.0]]])
        after = np.array([[[0.0, 0.1, 0.0], [0.0, 0.1, 0.0]]])
        old, _, _ = code_statistics(before, truth, support)
        new, _, _ = code_statistics(after, truth, support)
        self.assertLess(new['code_ce'], old['code_ce'])
        self.assertLess(new['accuracy'], old['accuracy'])


    def test_unobserved_logits_do_not_affect_statistics(self):
        truth = np.array([[0, 1]], dtype=np.int64)
        support = np.array([[True, False]])
        logits = np.array([[[2.0, 0.0], [-2.0, 3.0]]])
        old, _, _ = code_statistics(logits, truth, support)
        logits[0, 1] = [90.0, -90.0]
        new, _, _ = code_statistics(logits, truth, support)
        self.assertEqual(old, new)
        with self.assertRaisesRegex(ValueError, 'Invalid code probe'):
            code_statistics(logits, truth, np.zeros_like(support))


if __name__ == '__main__':
    unittest.main()
