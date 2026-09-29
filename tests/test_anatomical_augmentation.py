import unittest
import inspect
import numpy as np
from anatomical_augmentation import anatomical_covering

class AnatomicalTests(unittest.TestCase):
    def setUp(self):
        self.rgb=np.full((256,256,3),125,np.uint8)
        self.points=np.array([[85,90],[170,100],[128,135],[103,170],[151,175]],float)

    def test_eye_and_mouth_anchors_covered_and_outside_unchanged(self):
        for kind,ids in [('eyes',(0,1)),('lower',(3,4))]:
            out,mask,event=anatomical_covering(self.rgb,self.points,kind,42)
            self.assertEqual(event['status'],'generated')
            self.assertTrue(np.isin(mask,[0,1]).all())
            for i in ids:
                x,y=np.rint(self.points[i]).astype(int);self.assertEqual(mask[y,x],1)
            np.testing.assert_array_equal(out[mask==0],self.rgb[mask==0])
            other=anatomical_covering(self.rgb,self.points,kind,42)
            np.testing.assert_array_equal(out,other[0])

    def test_invalid_landmarks_return_explicit_rejection(self):
        for points in (None,np.zeros((5,2)),np.full((5,2),np.nan),self.points-300):
            out,mask,event=anatomical_covering(self.rgb,points,'eyes',42)
            self.assertIsNone(out);self.assertIsNone(mask)
            self.assertEqual(event['status'],'rejected');self.assertTrue(event['reason'])

    def test_input_not_mutated(self):
        copy=self.points.copy();anatomical_covering(self.rgb,self.points,'lower',2)
        np.testing.assert_array_equal(copy,self.points)

    def test_unsupported_kind_is_error(self):
        with self.assertRaises(ValueError):anatomical_covering(self.rgb,self.points,'unknown',42)

    def test_explicit_clipping_matches_crop_of_same_polygon(self):
        self.assertIn('boundary_policy',inspect.signature(anatomical_covering).parameters)
        points=self.points+np.array([0,60])
        self.assertEqual(anatomical_covering(self.rgb,points,'lower',42)[2]['reason'],'polygon_outside_image')
        out,mask,event=anatomical_covering(self.rgb,points,'lower',42,boundary_policy='clip')
        large=np.full((320,256,3),125,np.uint8)
        expected,full_mask,_=anatomical_covering(large,points,'lower',42)
        np.testing.assert_array_equal(mask,full_mask[:256])
        np.testing.assert_array_equal(out,expected[:256])
        self.assertTrue(event['border_clipped'])
        self.assertEqual(event['version'],'anatomical-covering-v2-clip')
        self.assertTrue(mask[-1].any())
        np.testing.assert_array_equal(out[mask==0],self.rgb[mask==0])

    def test_clipping_keeps_geometry_rejections_and_interior_pixels(self):
        self.assertIn('boundary_policy',inspect.signature(anatomical_covering).parameters)
        for kind in ('eyes','lower'):
            original=anatomical_covering(self.rgb,self.points,kind,42)
            clipped=anatomical_covering(self.rgb,self.points,kind,42,boundary_policy='clip')
            np.testing.assert_array_equal(original[0],clipped[0])
            np.testing.assert_array_equal(original[1],clipped[1])
            self.assertFalse(clipped[2]['border_clipped'])
        for points in (None,self.points-300,np.zeros((5,2))):
            out,mask,event=anatomical_covering(self.rgb,points,'lower',42,boundary_policy='clip')
            self.assertIsNone(out);self.assertIsNone(mask)
            self.assertEqual(event['status'],'rejected')
        with self.assertRaises(ValueError):
            anatomical_covering(self.rgb,self.points,'lower',42,boundary_policy='unknown')
