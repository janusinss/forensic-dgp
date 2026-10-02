import unittest
import numpy as np

from face_color_policy import preserve_input_palette


class PaletteTests(unittest.TestCase):
    def test_black_white_input_has_gray_generated_regions_and_exact_observed_pixels(self):
        rgb = np.full((256,256,3),100,np.uint8)
        mask = np.zeros((256,256),np.uint8)
        mask[120:175,155:200] = 1
        output = rgb.copy()
        output[mask==1] = [90,140,210]
        fixed,signal = preserve_input_palette(rgb,mask,output)
        self.assertTrue(signal['grayscale_input'])
        self.assertTrue(np.array_equal(fixed[mask==0],rgb[mask==0]))
        self.assertTrue(np.array_equal(fixed[...,0],fixed[...,1]))
        self.assertTrue(np.array_equal(fixed[...,1],fixed[...,2]))

    def test_colored_covering_does_not_change_palette_selection(self):
        rgb = np.full((256,256,3),100,np.uint8)
        mask = np.zeros((256,256),np.uint8)
        mask[120:175,155:200] = 1
        rgb[mask==1] = [255,40,0]
        fixed,signal = preserve_input_palette(rgb,mask,np.full_like(rgb,[40,80,120]))
        self.assertTrue(signal['grayscale_input'])
        self.assertTrue(np.array_equal(fixed[mask==0],np.full_like(rgb,[40,80,120])[mask==0]))

    def test_color_input_bypasses_and_insufficient_support_does_not_guess(self):
        rgb = np.full((256,256,3),[90,130,170],np.uint8)
        mask = np.zeros((256,256),np.uint8)
        fixed,signal = preserve_input_palette(rgb,mask,rgb)
        self.assertFalse(signal['grayscale_input'])
        self.assertTrue(np.array_equal(fixed,rgb))
        _,signal = preserve_input_palette(rgb,np.ones_like(mask),rgb)
        self.assertFalse(signal['grayscale_input'])

    def test_restored_gray_result_has_no_false_color_and_same_generated_pixels(self):
        original = np.full((256,256,3),100,np.uint8)
        mask = np.zeros((256,256),np.uint8)
        mask[120:175,155:200] = 1
        completed = original.copy();completed[mask==1] = [60,90,150]
        restored = np.full_like(original,[120,130,140]);restored[mask==1] = completed[mask==1]
        off,_ = preserve_input_palette(original,mask,completed,False)
        on,_ = preserve_input_palette(original,mask,restored,True)
        self.assertTrue(np.array_equal(off[mask==1],on[mask==1]))
        self.assertTrue(np.array_equal(on[...,0],on[...,2]))


if __name__=='__main__':
    unittest.main()
