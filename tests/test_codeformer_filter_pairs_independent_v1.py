"""Pure CPU sanity/adversarial fixtures for the independently authored pixel oracle."""
import hashlib
import importlib.util
from pathlib import Path
import tempfile
import unittest

import numpy as np

ROOT = Path(__file__).resolve().parents[1]
SPEC = importlib.util.spec_from_file_location('independent_filter_auditor',
    ROOT / 'scripts/audit_codeformer_filter_pairs_v1_independent.py')
a = importlib.util.module_from_spec(SPEC)
SPEC.loader.exec_module(a)


class IndependentFilterOracleFixtures(unittest.TestCase):
    def test_profiles_are_locked_twelve_with_five_strong_controls(self):
        self.assertEqual(len(a.PROFILES), 12)
        self.assertEqual(sum(p[3] for p in a.PROFILES), 5)
        self.assertEqual(a.RECT, (64,64,192,192))

    def test_reflected_edges_are_half_sample_not_whole_sample(self):
        values = np.array([1.,2.,3.]).reshape(3,1,1)
        actual = a.correlate_axis(values,np.array([.25,.5,.25]),0)
        self.assertTrue(np.array_equal(actual[:,0,0],[1.25,2.,2.75]))

    def test_gaussian_kernel_radii_symmetry_and_constant_colour(self):
        for sigma,radius in [(0.,0),(.7,2),(2.4,7),(5.,15),(10.,30)]:
            kernel=a.gaussian_kernel(sigma)
            self.assertEqual(len(kernel),2*radius+1)
            self.assertTrue(np.array_equal(kernel,kernel[::-1]))
            self.assertAlmostEqual(float(kernel.sum()),1.,places=14)
        image=np.zeros((256,256,3),np.uint8);image[:]=[17,128,231]
        for p in a.PROFILES:
            pixels,_=a.reconstruct(image,p)
            self.assertTrue(np.array_equal(pixels,image),p[0])

    def test_half_up_not_bankers_rounding_and_nonfinite_rejected(self):
        self.assertTrue(np.array_equal(a.half_up(np.array([.5,1.5,2.5,254.5])),[1,2,3,255]))
        with self.assertRaisesRegex(ValueError,'Finite'):
            a.half_up(np.array([np.nan]))

    def test_row_blend_and_mosaic_hand_results(self):
        y,x=np.indices((256,256),dtype=np.uint8)
        image=np.stack([x,y,np.zeros_like(x)],axis=-1)
        mosaic,_=a.reconstruct(image,a.PROFILES[6])
        self.assertTrue(np.all(mosaic[:8,:8]==[4,4,0]))
        self.assertTrue(np.all(mosaic[:8,8:16]==[12,4,0]))
        repeat,_=a.reconstruct(image,a.PROFILES[5])
        self.assertEqual(repeat[:8,0,1].tolist(),[0,0,1,1,4,4,5,5])

    def test_local_footprint_preserves_exact_outside_and_caller(self):
        y,x=np.indices((256,256))
        v=((x*19+y*31+x*y)%256).astype(np.uint8)
        image=np.stack([v,v^127,255-v],axis=-1);before=image.copy()
        expected=np.zeros((256,256),np.uint8);expected[64:192,64:192]=255
        for p in a.PROFILES[8:]:
            result,foot=a.reconstruct(image,p)
            self.assertTrue(np.array_equal(foot,expected))
            self.assertTrue(np.array_equal(result[foot==0],image[foot==0]))
        self.assertTrue(np.array_equal(image,before))

    def test_identity_exact_separate_storage_zero_footprint(self):
        image=np.arange(256*256*3,dtype=np.uint8).reshape(256,256,3)
        result,foot=a.reconstruct(image,a.PROFILES[0])
        self.assertTrue(np.array_equal(result,image))
        self.assertFalse(np.shares_memory(result,image));self.assertFalse(foot.any())

    def test_fingerprint_rejects_same_size_tamper(self):
        with tempfile.TemporaryDirectory(prefix='independent-filter-audit-',dir=ROOT/'scratch') as folder:
            path=Path(folder)/'test.bin';path.write_bytes(b'first')
            pin=hashlib.sha256(b'first').hexdigest()
            self.assertEqual(a.bound(path,pin,5),b'first')
            path.write_bytes(b'other')
            with self.assertRaisesRegex(ValueError,'fingerprint'):
                a.bound(path,pin,5)

    def test_noncanonical_or_escaping_names_rejected(self):
        for name in ['../bad','a/../bad','a\\bad','/absolute','C:bad','a//bad','./bad']:
            with self.subTest(name=name),self.assertRaises(ValueError):a.relative_name(name)

    def test_duplicate_metadata_and_nonfinite_rejected(self):
        for data in [b'{"a":1,"a":2}',b'{"a":NaN}']:
            with self.assertRaises(ValueError):a.strict_json(data)


if __name__=='__main__':
    unittest.main()
