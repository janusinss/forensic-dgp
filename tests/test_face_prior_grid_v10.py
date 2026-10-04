"""Regression for the expensive post-inference NumPy/Pillow paste failure."""
import unittest
import numpy as np
from face_prior_grid_v10 import render_grid


class GridTests(unittest.TestCase):
    def test_every_original_cell_survives_without_resize_or_color_changes(self):
        colors=[(255,0,13),(0,181,255),(91,12,0),(120,121,122)]
        images=[np.full((256,256,3),c,np.uint8) for c in colors]
        sheet=np.asarray(render_grid(['a','b'],[{'id':'first','images':images[:2]},{'id':'second','images':images[2:]}]))
        self.assertEqual(sheet.shape,(600,520,3))
        for i in range(2):
            for j in range(2):np.testing.assert_array_equal(sheet[52+i*288:308+i*288,2+j*260:258+j*260],images[i*2+j])

    def test_wrong_shape_or_float_image_is_rejected(self):
        for image in [np.zeros((128,128,3),np.uint8),np.zeros((256,256,3),np.float32)]:
            with self.assertRaises(ValueError):render_grid(['a'],[{'id':'x','images':[image]}])

    def test_column_count_mismatch_is_rejected(self):
        with self.assertRaises(ValueError):render_grid(['a','b'],[{'id':'x','images':[np.zeros((256,256,3),np.uint8)]}])


if __name__=='__main__':unittest.main()
