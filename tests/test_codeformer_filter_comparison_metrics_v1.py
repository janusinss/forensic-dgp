"""Known-answer and adversarial saved-pixel diagnostic fixtures; no models."""
import importlib.util
import io
import math
from pathlib import Path
import unittest

import numpy as np

ROOT=Path(__file__).resolve().parents[1]
SPEC=importlib.util.spec_from_file_location('filter_comparison_audit',ROOT/'scripts/audit_codeformer_filter_comparison_v1.py')
a=importlib.util.module_from_spec(SPEC);SPEC.loader.exec_module(a)


class IndependentComparisonMetrics(unittest.TestCase):
    def test_identical_zero_error_has_null_offset_share_and_infinite_psnr_null(self):
        image=np.full((9,9,3),.4,np.float64);m=a.residual_metrics(image,image,np.ones((9,9),bool))
        self.assertEqual(m['mae'],0);self.assertEqual(m['mse'],0);self.assertEqual(m['ssim'],1)
        self.assertIsNone(m['global_RGB_offset_energy_share']);self.assertIsNone(m['psnr'])
        self.assertEqual(m['adjacent_RGB_residual_gradient_mae'],0)

    def test_constant_colour_offset_removes_centered_and_adjacent_error(self):
        truth=np.full((9,9,3),.25,np.float64);pred=truth+np.array([.125,.25,.375]);m=a.residual_metrics(pred,truth,np.ones((9,9),bool))
        self.assertAlmostEqual(m['global_RGB_offset_energy_share'],1)
        self.assertEqual(m['mean_centered_residual_mae'],0)
        self.assertEqual(m['adjacent_RGB_residual_gradient_mae'],0)
        self.assertAlmostEqual(m['MSE_decomposition_closure_residual'],0,places=15)

    def test_mean_centered_mae_can_increase_not_l1_error_explained(self):
        truth=np.zeros((1,3,3),np.float64);pred=truth.copy();pred[:,2]=.9
        m=a.residual_metrics(pred,truth,np.ones((1,3),bool))
        self.assertAlmostEqual(m['mae'],.3);self.assertAlmostEqual(m['mean_centered_residual_mae'],.4)
        self.assertAlmostEqual(m['global_RGB_offset_energy_share'],1/3)
        self.assertAlmostEqual(m['MSE_decomposition_closure_residual'],0,places=15)

    def test_zero_mean_spatial_residual_has_zero_offset_energy(self):
        truth=np.full((2,2,3),.5,np.float64);pred=truth+np.array([[.25,-.25],[-.25,.25]])[...,None]
        m=a.residual_metrics(pred,truth,np.ones((2,2),bool))
        self.assertEqual(m['global_RGB_offset_energy_share'],0)
        self.assertEqual(m['mean_centered_residual_mse'],m['mse'])
        self.assertEqual(m['adjacent_RGB_residual_gradient_mae'],.5)

    def test_pair_weighting_and_support_holes_no_cross_padding(self):
        truth=np.zeros((2,3,3),np.float64);pred=truth.copy();pred[0,:,0]=[0,.2,.4];pred[1,:,0]=[1,1,1]
        support=np.array([[1,1,1],[0,1,0]],bool)
        m=a.residual_metrics(pred,truth,support)
        self.assertEqual(m['adjacent_x_pairs'],2);self.assertEqual(m['adjacent_y_pairs'],1)
        self.assertAlmostEqual(m['adjacent_RGB_residual_gradient_mae'],1.2/9)
        self.assertAlmostEqual(m['adjacent_RGB_residual_gradient_x_mae'],.4/6)
        self.assertAlmostEqual(m['adjacent_RGB_residual_gradient_y_mae'],.8/3)
        self.assertIsNone(m['ssim'])

    def test_single_supported_pixel_no_pairs_is_null_not_zero(self):
        clean=np.zeros((2,2,3),np.float64);mask=np.eye(2,dtype=bool);m=a.residual_metrics(clean,clean,mask)
        self.assertEqual(m['observed_pixels'],2);self.assertIsNone(m['adjacent_RGB_residual_gradient_mae'])

    def test_nonfinite_invalid_dtype_empty_support_and_geometry_refused(self):
        good=np.zeros((8,8,3),np.float64);mask=np.ones((8,8),bool)
        bad=good.copy();bad[0,0,0]=np.nan
        for pred,truth,support in [(bad,good,mask),(good.astype(np.uint8),good,mask),(good,good,~mask),(good[:7],good,mask)]:
            with self.assertRaises(ValueError):a.residual_metrics(pred,truth,support)

    def test_psnr_known_mse(self):
        clean=np.zeros((8,8,3),np.float64);pred=np.full_like(clean,.1)
        self.assertAlmostEqual(a.residual_metrics(pred,clean,np.ones((8,8),bool))['psnr'],20.,places=12)

    def test_SSIM_constant_known_luminance_term(self):
        clean=np.full((9,9,3),.2,np.float64);pred=np.full_like(clean,.3)
        expected=(2*.2*.3+.01**2)/(.2**2+.3**2+.01**2)
        self.assertAlmostEqual(a.residual_metrics(pred,clean,np.ones((9,9),bool))['ssim'],expected,places=10)

    def test_RAW_geometry_nonfinite_and_trailing_payload_refused(self):
        good=np.zeros((256,256,3),np.float32);stream=io.BytesIO();np.save(stream,good,allow_pickle=False)
        self.assertTrue(np.array_equal(a.raw_rgb(stream.getvalue()),good))
        with self.assertRaisesRegex(ValueError,'Trailing'):a.raw_rgb(stream.getvalue()+b'x')
        good[0,0,0]=np.inf;stream=io.BytesIO();np.save(stream,good,allow_pickle=False)
        with self.assertRaises(ValueError):a.raw_rgb(stream.getvalue())

    def test_appearance_flag_always_a_separate_stratum(self):
        self.assertEqual(a.source_stratum('va_ffhq_08025'),'appearance_flag_va_ffhq_08025')
        self.assertNotEqual(a.source_stratum('tr_ffhq_00084'),a.source_stratum('va_ffhq_08025'))

    def test_floor_conversion_rejects_rounding_and_single_pixel_mutation(self):
        raw=np.zeros((256,256,3),np.float32);raw[0,0]=[1,.5,np.float32(1)/np.float32(255)]
        correct=np.floor(raw*np.float32(255)).astype(np.uint8)
        a.check_floor(raw,correct)
        wrong=correct.copy();wrong[0,0,1]=128
        with self.assertRaisesRegex(ValueError,'floor'):a.check_floor(raw,wrong)

    def test_noncanonical_output_namespace_is_rejected(self):
        for name in ['../bad','/bad','C:bad','a\\bad','a//bad','./bad']:
            with self.subTest(name=name),self.assertRaises(ValueError):a.canonical_relative(name)

    def test_offset_energy_is_colour_not_just_luminance(self):
        truth=np.full((8,8,3),.5,np.float64)
        # R and G offsets cancel exactly in luminance up to finite arithmetic.
        offset=np.array([.1,-.1*.2126/.7152,0.])
        m=a.residual_metrics(truth+offset,truth,np.ones((8,8),bool))
        self.assertAlmostEqual(m['signed_luminance_bias'],0,places=14)
        self.assertGreater(m['global_RGB_offset_energy'],0)
        self.assertAlmostEqual(m['global_RGB_offset_energy_share'],1,places=14)


if __name__=='__main__':unittest.main()
