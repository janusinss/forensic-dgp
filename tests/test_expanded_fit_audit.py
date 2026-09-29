import unittest
from unittest.mock import patch
from scripts.audit_expanded_fit_vm import summarize,main


class FitAuditTests(unittest.TestCase):
    def test_counts_and_denominators(self):
        counts=dict(tp=3,fp=1,fn=2,visible=10,empty=0,negative_fp=0,positive=1,negative=0)
        result=summarize([{'raw':counts,'gated':counts}]*2)
        self.assertEqual(result['cases'],2)
        self.assertEqual(result['raw']['counts']['tp'],6)
        self.assertEqual(result['raw']['iou'],.5)
        self.assertEqual(result['raw']['missed_fraction'],.4)
        self.assertEqual(result['raw']['visible_false_positive'],.1)
        with self.assertRaises(ValueError):summarize([])

    def test_cpu_guard_precedes_workspace_access(self):
        with patch('torch.cuda.is_available',return_value=False),patch('pathlib.Path.resolve',side_effect=AssertionError('access')):
            with self.assertRaisesRegex(RuntimeError,'VM GPU'):main('.')
