import unittest
from unittest.mock import patch
import numpy as np
from scripts.audit_expanded_regions_vm import region_counts, main


class RegionAuditTests(unittest.TestCase):
    def test_disjoint_regions_account_for_errors(self):
        core = np.array([1, 1, 0, 0, 0, 0], dtype=bool)
        target = np.array([1, 1, 1, 1, 0, 0], dtype=bool)
        pred = np.array([1, 0, 1, 0, 1, 0], dtype=bool)
        self.assertEqual(region_counts(pred, core, target), dict(
            core_pixels=2, core_missed=1, border_pixels=2, border_missed=1,
            outside_pixels=2, outside_fp=1))
        with self.assertRaises(ValueError):
            region_counts(pred, ~target, target)

    def test_cpu_guard(self):
        with patch('torch.cuda.is_available', return_value=False), patch('pathlib.Path.resolve', side_effect=AssertionError('access')):
            with self.assertRaisesRegex(RuntimeError, 'VM GPU'):
                main('.')
