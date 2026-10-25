"""Pure saved-pixel arithmetic/parser/classification fixtures, no model call."""
import ast
import hashlib
import io
from pathlib import Path
import subprocess
import sys
import tempfile
import unittest

import numpy as np

ROOT=Path(__file__).resolve().parents[1]
sys.path.insert(0,str(ROOT/'scripts'))
import audit_u96_smoothanchors_lr_fixed_DEV_saved_v1 as A


class DeliveryFixtures(unittest.TestCase):
    def test_bilinear_separable_FP32_order_differs_from_four_flat_terms(self):
        rng=np.random.default_rng(151);native=rng.normal(size=(1,3,512,512)).astype(np.float32)
        actual=A.down2(native)
        a=native[0,:,0::2,0::2];b=native[0,:,0::2,1::2];c=native[0,:,1::2,0::2];d=native[0,:,1::2,1::2]
        expected=((a*.5+b*.5)*.5+(c*.5+d*.5)*.5).transpose(1,2,0)
        np.testing.assert_array_equal(actual,expected)
        flat=(((a*.25+b*.25)+c*.25)+d*.25).transpose(1,2,0)
        self.assertGreater(np.count_nonzero(actual!=flat),0)

    def test_clipping_happens_at_native512_before_downsample(self):
        native=np.zeros((1,3,512,512),np.float32);native[:,:,:,0::2]=-3
        canvas=np.zeros((256,256,3),np.uint8);observed=np.ones((256,256),bool)
        unclipped,raw,png=A.delivery(native,canvas,observed)
        self.assertTrue(np.all(unclipped==-.25));self.assertTrue(np.all(raw==.25))
        self.assertTrue(np.all(png==63));self.assertFalse(np.array_equal(raw,np.clip(unclipped,0,1)))

    def test_exact_padding_preservation_and_float32_floor(self):
        native=np.zeros((1,3,512,512),np.float32);canvas=np.full((256,256,3),128,np.uint8)
        observed=np.ones((256,256),bool);observed[:,:31]=False
        _,raw,png=A.delivery(native,canvas,observed)
        np.testing.assert_array_equal(raw[~observed],canvas.astype(np.float32)[~observed]/255)
        np.testing.assert_array_equal(png[~observed],canvas[~observed])
        self.assertTrue(np.all(png[observed]==127))

    def test_original_geometry_non_square_odd_padding(self):
        original=np.arange(17*12*3,dtype=np.uint8).reshape(17,12,3)
        canvas,support,geometry=A.prepare(original)
        self.assertEqual(geometry['native_size'],[12,17]);self.assertEqual(geometry['native_center_offset'],[2,0])
        self.assertEqual(canvas.shape,(256,256,3));self.assertEqual(support.dtype,np.bool_)
        self.assertEqual(geometry['observed_sha256'],hashlib.sha256(support.tobytes()).hexdigest())


class MetricFixtures(unittest.TestCase):
    def test_whole_HVmeans_vs_masked_joint_edge_count(self):
        y,x=np.mgrid[:3,:4];value=np.repeat((x+4*y)[:,:,None],3,axis=2).astype(np.float64);target=np.zeros_like(value);mask=np.ones((3,4),bool)
        self.assertEqual(A.regional(value,target,mask,True)['adjacent_residual_RGB_error'],2.5)
        self.assertEqual(A.regional(value,target,mask,False)['adjacent_residual_RGB_error'],41/17)

    def test_outside_does_not_count_boundary_crossing_edges(self):
        target=np.zeros((256,256,3),np.float64);value=target.copy();value[64:192,64:192]=1
        regions=A.all_regions(value,target,np.ones((256,256),bool))
        self.assertEqual(regions['outside_central128']['MAE'],0.)
        self.assertEqual(regions['outside_central128']['adjacent_residual_RGB_error'],0.)
        self.assertEqual(regions['central128']['MAE'],1.)
        self.assertGreater(regions['whole']['adjacent_residual_RGB_error'],0.)

    def test_unsupported_masked_region_reports_null_notNaN(self):
        value=np.zeros((3,3,3),np.float64);result=A.regional(value,value,np.zeros((3,3),bool))
        self.assertIsNone(result['MAE']);self.assertIsNone(result['adjacent_residual_RGB_error'])

    def test_paired_PSNR_SSIM_identity_and_known_constant_error(self):
        x=np.full((256,256,3),.4,np.float64);mask=np.ones((256,256),bool)
        exact=A.paired(x,x,mask)
        self.assertIsNone(exact['PSNR_dB']);self.assertTrue(exact['PSNR_infinite_exact_match'])
        self.assertEqual(exact['SSIM_RGB_valid11_gaussian1p5_population'],1.)
        changed=A.paired(x+.1,x,mask)
        self.assertAlmostEqual(changed['PSNR_dB'],20.,places=12)
        self.assertAlmostEqual(changed['SSIM_RGB_valid11_gaussian1p5_population'],(2*.4*.5+.01**2)/(.4**2+.5**2+.01**2),places=11)
        mask[:,0]=False
        with self.assertRaises(ValueError):A.paired(x,x,mask)

    def test_native_mask_ignores_padding_in_drift(self):
        target=np.zeros((256,256,3),np.float64);value=target.copy();value[:,:20]=1
        observed=np.ones((256,256),bool);observed[:,:20]=False
        self.assertEqual(A.all_regions(value,target,observed)['whole']['MAE'],0.)


class ParserAndClassificationFixtures(unittest.TestCase):
    def test_NPY_schema_and_extent_before_allocation(self):
        with tempfile.TemporaryDirectory(dir=ROOT/'scratch')as directory:
            path=Path(directory)/'synthetic.npy';np.save(path,np.zeros((256,256,3),np.float32),allow_pickle=False)
            self.assertEqual(A.array(path,(256,256,3)).shape,(256,256,3))
            raw=path.read_bytes();path.write_bytes(raw+b'extra')
            with self.assertRaises(ValueError):A.array(path,(256,256,3))
            stream=io.BytesIO();np.lib.format.write_array_header_1_0(stream,{'descr':'<f4','fortran_order':False,'shape':(2**34,)})
            path.write_bytes(stream.getvalue())
            with self.assertRaises(ValueError):A.array(path,(256,256,3))

    def test_partial_and_reserved_return_never_fabricate136_success(self):
        supervisor={'worker_exit':2,'timeout':False}
        result=A.classify(None,supervisor,None,None,True,0)
        self.assertFalse(result['full_saved_pixel_phase_supported']);self.assertTrue(result['forward_outcome_unknown'])
        counters={'full_forward_attempts':7,'full_forward_returns':6,'successful_saved_outputs':6,'completed_cases':3}
        status={'counters':counters,'forward_outcome_unknown':True}
        self.assertFalse(A.classify(status,supervisor,None,{},False,6)['full_saved_pixel_phase_supported'])
        self.assertEqual(A.classify(status,supervisor,None,{},False,6)['forward_attempts'],7)

    def test_complete_requires_terminal_counters_record_count_and_no_failure(self):
        counters={'full_forward_attempts':136,'full_forward_returns':136,'successful_saved_outputs':136,'completed_cases':68}
        status={'counters':counters,'forward_outcome_unknown':False};supervisor={'worker_exit':0,'timeout':False}
        self.assertTrue(A.classify(status,supervisor,{'complete':True},None,False,136)['full_saved_pixel_phase_supported'])
        for count,failure,pending in[(135,None,False),(136,{'status':'STOP'},False),(136,None,True)]:
            self.assertFalse(A.classify(status,supervisor,{'complete':True},failure,pending,count)['full_saved_pixel_phase_supported'])

    def test_JSON_duplicate_nonfinite_and3_10_AST(self):
        for raw in [b'{"a":1,"a":2}',b'{"a":NaN}',b'{"a":1e999}']:
            with self.assertRaises(ValueError):A.strict_json(raw)
        ast.parse((ROOT/'scripts/audit_u96_smoothanchors_lr_fixed_DEV_saved_v1.py').read_text(),feature_version=(3,10))
        self.assertNotIn('torch',sys.modules)


if __name__=='__main__':unittest.main()
