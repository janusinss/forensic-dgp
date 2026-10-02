import unittest
import numpy as np
import torch

from face_workflow_palette import PaletteFaceWorkflow


class PaletteWorkflowTests(unittest.TestCase):
    def test_gray_estimate_preserves_observed_pixels_and_post_restoration_region(self):
        engine=PaletteFaceWorkflow(device='cpu')
        def completion(image,mask):
            color=torch.tensor([.2,.4,.8],device=image.device).reshape(1,3,1,1)
            return color.expand_as(image).clone()
        engine.generator=completion
        engine.restorer=lambda image,fidelity:torch.full_like(image,.5)
        original=np.full((256,256,3),100,np.uint8)
        mask=np.zeros((256,256),np.uint8);mask[150:190,100:160]=1
        off,meta=engine.generate(original,mask,'off')
        on,restored=engine.generate(original,mask,'on')
        self.assertEqual(meta['policy'],'reviewed-face-workflow-v2')
        self.assertTrue(meta['color_policy']['applied'])
        self.assertTrue(np.array_equal(off[mask==0],original[mask==0]))
        self.assertTrue(np.array_equal(off[...,0],off[...,2]))
        self.assertTrue(np.array_equal(off[mask==1],on[mask==1]))
        self.assertTrue(restored['restoration_applied'])

    def test_empty_color_crop_off_still_bypasses_both_models(self):
        engine=PaletteFaceWorkflow(device='cpu',completion_path='missing',restoration_path='missing')
        original=np.full((256,256,3),[90,130,170],np.uint8)
        output,meta=engine.generate(original,np.zeros((256,256),np.uint8),'off')
        self.assertTrue(np.array_equal(output,original))
        self.assertFalse(meta['color_policy']['applied'])
        self.assertIsNone(engine.generator)
        self.assertIsNone(engine.restorer)


if __name__=='__main__':
    unittest.main()
