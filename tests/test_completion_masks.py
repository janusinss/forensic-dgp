import unittest
import torch
from completion_masks import completion_mask


class MaskTests(unittest.TestCase):
    def test_baseline_unchanged(self):
        p=torch.tensor([[[[.1,.35,.5,.9]]]])
        self.assertTrue(torch.equal(completion_mask(p), (p>=.5).float()))

    def test_growth_requires_seed_proximity_and_confidence(self):
        p=torch.zeros(1,1,7,7);p[:,:,3,3]=.9
        p[:,:,3,4]=.4;p[:,:,3,5]=.4;p[:,:,2,3]=.1;p[:,:,0,0]=.4
        m=completion_mask(p,'boundary035')
        self.assertEqual(m.sum(),2)
        self.assertEqual(m[0,0,3,4],1)
        self.assertEqual(m[0,0,3,5],0)

    def test_empty_seed_stays_empty(self):
        self.assertEqual(completion_mask(torch.full((1,1,8,8),.4),'boundary035').sum(),0)

    def test_reject_invalid_probabilities(self):
        for p in [torch.full((1,1,8,8),float('nan')),torch.ones(1,2,8,8),torch.full((1,1,8,8),1.1)]:
            with self.assertRaises(ValueError):completion_mask(p)
