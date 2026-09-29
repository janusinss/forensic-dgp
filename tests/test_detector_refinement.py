import unittest
import numpy as np
from detector_refinement import detector_prompt, refine_mask


class FakePredictor:
    def __init__(self,masks,scores):self.masks,self.scores=masks,scores;self.calls=0
    def set_image(self,image):self.calls+=1
    def predict(self,**kwargs):
        self.kwargs=kwargs
        return self.masks,self.scores,None


class RefinementTests(unittest.TestCase):
    def test_empty_abstains_without_loading_image(self):
        p=np.zeros((32,32),np.float32);predictor=FakePredictor(None,None)
        result,meta=refine_mask(np.zeros((32,32,3),np.uint8),p,predictor)
        self.assertFalse(result.any());self.assertEqual(predictor.calls,0)
        self.assertEqual(meta['status'],'empty_detector')

    def test_prompt_uses_largest_component_and_xy_coordinates(self):
        p=np.zeros((32,40),np.float32);p[12:22,24:34]=.8;p[1,1]=1
        prompt=detector_prompt(p)
        x,y=prompt['point_coords'][0]
        self.assertTrue(24<=x<34 and 12<=y<22)
        self.assertTrue(np.array_equal(prompt['box'],[23,11,34,22]))

    def test_selects_predicted_quality_and_preserves_size(self):
        p=np.zeros((32,32),np.float32);p[10:20,10:20]=.8
        a=np.zeros_like(p,bool);a[8:22,8:22]=True
        predictor=FakePredictor(np.stack([p>.5,a]),np.array([.2,.9]))
        result,meta=refine_mask(np.zeros((32,32,3),np.uint8),p,predictor)
        self.assertTrue(np.array_equal(result,a));self.assertEqual(meta['candidate'],1)
        self.assertNotIn('mask_input',predictor.kwargs)

    def test_invalid_proposal_falls_back_and_records_reason(self):
        p=np.zeros((32,32),np.float32);p[10:20,10:20]=.8
        predictor=FakePredictor(np.ones((1,32,32),bool),np.array([.9]))
        result,meta=refine_mask(np.zeros((32,32,3),np.uint8),p,predictor)
        self.assertTrue(np.array_equal(result,p>=.5));self.assertEqual(meta['status'],'rejected_coverage')

    def test_bad_probabilities_rejected(self):
        for p in [np.ones((4,4))*2,np.ones((4,4))*np.nan,np.ones((4,4,1))]:
            with self.assertRaises(ValueError):detector_prompt(p)
