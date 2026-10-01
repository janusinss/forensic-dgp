import importlib.util
import unittest

import numpy as np


class ProposalTests(unittest.TestCase):
    def setUp(self):
        self.assertIsNotNone(importlib.util.find_spec('scripts.prepare_reflection_source_cohort'),
                             'Training-only native eye proposal builder missing')

    def detector(self,boxes,points):
        class Detector:
            def detect(self,image,max_num=0):
                self.image=image.copy()
                return np.asarray(boxes),np.asarray(points) if points is not None else None
        return Detector()

    def test_keypoints_invert_uniform_native_transform(self):
        from scripts.prepare_reflection_source_cohort import propose_eyes
        detector=self.detector([[0,0,255,255,.9]],[[[80,90],[175,90],[128,120],[90,170],[166,170]]])
        rgb=np.full((201,101,3),(10,20,30),np.uint8)
        record=propose_eyes(rgb,detector)
        self.assertEqual(record['decision'],'proposed')
        points=np.array(record['eyes_native']);affine=np.array(record['affine'])
        np.testing.assert_allclose(points@affine[:,:2].T+affine[:,2],[[80,90],[175,90]],atol=1e-10)
        self.assertEqual(detector.image.shape,(256,256,3))
        self.assertTrue(np.array_equal(detector.image[100,128],[30,20,10]))
        self.assertFalse(record['eyes_reviewed'])
        self.assertFalse(record['training_enabled'])

    def test_multiple_low_confidence_missing_or_padded_eyes_are_review_failures(self):
        from scripts.prepare_reflection_source_cohort import propose_eyes
        rgb=np.full((201,101,3),100,np.uint8)
        good=[[[80,90],[175,90],[128,120],[90,170],[166,170]]]
        for boxes,points in [([],None),([[0,0,255,255,.5]],good),
                             ([[0,0,255,255,.9]]*2,good*2),
                             ([[0,0,255,255,.9]],[[[10,90],[175,90],[128,120],[90,170],[166,170]]])]:
            record=propose_eyes(rgb,self.detector(boxes,points))
            self.assertEqual(record['decision'],'needs_review')
            self.assertIsNone(record['eyes_native'])


if __name__=='__main__':unittest.main()
