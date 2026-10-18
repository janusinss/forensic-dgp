import unittest
import numpy as np
from research.codeformer_adain_gradient_isolation_v1.return_error_decomposition import split_error


class ErrorDecompositionTests(unittest.TestCase):
    def test_constant_colour_offset_has_no_residual(self):
        target = np.full((2, 2, 3), .25, np.float64)
        delta = np.array([.1, -.2, .3])
        result = split_error(target + delta, target)
        self.assertAlmostEqual(result['mse'], (.01+.04+.09)/3)
        self.assertAlmostEqual(result['global_channel_bias_MSE_component'], result['mse'])
        self.assertLess(result['residual_MSE_component'], 1e-30)

    def test_alternating_errors_have_no_global_bias(self):
        target = np.zeros((2, 2, 3))
        error = np.array([[1, -1], [-1, 1]])[..., None]*np.ones((1, 1, 3))*.125
        result = split_error(error, target)
        self.assertEqual(result['global_channel_bias_MSE_component'], 0)
        self.assertEqual(result['mse'], .015625)
        self.assertEqual(result['residual_MSE_component'], .015625)

    def test_nonfinite_and_unaligned_inputs_are_rejected(self):
        with self.assertRaises(ValueError):split_error(np.zeros((2,2,3)), np.zeros((3,2,3)))
        with self.assertRaises(ValueError):split_error(np.full((2,2,3), np.nan), np.zeros((2,2,3)))

    def test_padding_is_excluded_from_bias_and_residual(self):
        target = np.zeros((2,2,3))
        value = target.copy();value[0] = 100;value[1] = .25
        mask = np.array([[False,False],[True,True]])
        result = split_error(value,target,mask)
        self.assertEqual(result['observed_pixels'],2)
        self.assertEqual(result['mse'],.0625)
        self.assertEqual(result['residual_MSE_component'],0)


if __name__ == '__main__':unittest.main()
