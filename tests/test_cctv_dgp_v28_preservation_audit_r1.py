"""Cross-platform arithmetic tolerance cannot accept meaningful or malformed changes."""
import importlib.util
from pathlib import Path
import unittest

import numpy as np

PATH = Path(__file__).resolve().parents[1] / 'scripts/audit_cctv_dgp_v28_preservation_diagnostic_v1_return_r1.py'
SPEC = importlib.util.spec_from_file_location('v28_repair', PATH)
MODULE = importlib.util.module_from_spec(SPEC)
SPEC.loader.exec_module(MODULE)


class ArithmeticRepair(unittest.TestCase):
    def test_single_ulp_cross_platform_rounding(self):
        values = np.array([.18788578305524228, .4024753498422732])
        self.assertTrue(MODULE.reduction_equal(values, [.1878857830552423, .4024753498422731]))

    def test_meaningful_change_is_rejected(self):
        self.assertFalse(MODULE.reduction_equal([.4], [.400000001]))
        self.assertFalse(MODULE.reduction_equal([1e-10], [0]))

    def test_nonfinite_or_different_shape_is_rejected(self):
        self.assertFalse(MODULE.reduction_equal([np.nan], [np.nan]))
        self.assertFalse(MODULE.reduction_equal([np.inf], [np.inf]))
        self.assertFalse(MODULE.reduction_equal([[1]], [1]))

    def test_saved_arrays_still_require_exact_values(self):
        source = PATH.read_text(encoding='utf-8')
        self.assertIn('assert np.array_equal(total,derived)', source)
        self.assertIn('assert np.array_equal(displacement,np.load(', source)
        self.assertIn('assert np.array_equal(png,prior_png)', source)
        self.assertIn("assert sha(target)==receipt['files_sha256'][name]", source)


if __name__ == '__main__':
    unittest.main()
