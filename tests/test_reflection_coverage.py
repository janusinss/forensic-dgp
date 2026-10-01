import importlib.util
import unittest

import numpy as np
import torch


class CoverageContracts(unittest.TestCase):
    def setUp(self):
        self.assertIsNotNone(importlib.util.find_spec('reflection_coverage'),
                             'Reflection coverage/support module missing')

    def test_supported_loss_matches_existing_loss_with_full_support(self):
        from reflection_coverage import supported_segmentation_loss
        from detector_training import segmentation_loss
        torch.manual_seed(3)
        logits=torch.randn(2,1,16,16);mask=torch.zeros_like(logits);mask[0,:,5:10,5:10]=1
        actual=supported_segmentation_loss(logits,mask,torch.ones_like(mask))
        self.assertTrue(torch.allclose(actual,segmentation_loss(logits,mask,.25,.1),atol=1e-7,rtol=0))

    def test_padding_logits_neither_change_loss_nor_receive_gradients(self):
        from reflection_coverage import supported_segmentation_loss
        logits=torch.zeros(2,1,16,16,requires_grad=True);mask=torch.zeros_like(logits)
        mask.data[0,:,6:10,6:10]=1
        valid=torch.ones_like(logits);valid[:,:,:,:4]=0
        loss=supported_segmentation_loss(logits,mask,valid)
        other=logits.detach().clone();other[:,:,:,:4]=100
        self.assertTrue(torch.equal(loss,supported_segmentation_loss(other,mask,valid)))
        loss.backward();self.assertEqual(logits.grad[:,:,:,:4].abs().sum(),0)
        self.assertGreater(logits.grad[:,:,:,4:].abs().sum(),0)

    def test_no_supported_pixels_or_hole_in_padding_fails(self):
        from reflection_coverage import supported_segmentation_loss
        logits=torch.zeros(2,1,16,16);mask=torch.zeros_like(logits);valid=torch.ones_like(logits)
        valid[1]=0
        with self.assertRaises(ValueError):supported_segmentation_loss(logits,mask,valid)
        valid[1]=1;valid[0,0,0,0]=0;mask[0,0,0,0]=1
        with self.assertRaises(ValueError):supported_segmentation_loss(logits,mask,valid)

    def test_camera_is_shared_reproducible_and_never_labels_padding(self):
        from reflection_coverage import camera_fixture
        base=np.full((64,64,3),120,np.uint8);covered=base.copy();covered[28:36,25:39]=240
        hole=np.zeros((64,64),np.uint8);hole[28:36,25:39]=1
        valid=np.ones((64,64),np.uint8);valid[:,:6]=0;base[:,:6]=96;covered[:,:6]=96
        positive=camera_fixture(covered,hole,valid,seed=6,degraded=True)
        repeated=camera_fixture(covered,hole,valid,seed=6,degraded=True)
        negative=camera_fixture(base,np.zeros_like(hole),valid,seed=6,degraded=True)
        self.assertTrue(np.array_equal(positive['input'],repeated['input']))
        self.assertTrue(np.all(positive['mask']>=hole))
        self.assertFalse(positive['mask'][:,0:6].any())
        self.assertTrue(np.all(positive['input'][:,0:6]==96))
        self.assertFalse(negative['mask'].any())
        self.assertEqual(positive['camera'],negative['camera'])

    def test_schedule_keeps_paired_inputs_and_replay_membership_fixed(self):
        from reflection_coverage import supplemental_schedule, sanitize_schedule
        schedule=supplemental_schedule(28,252,42)
        self.assertEqual(len(schedule),252)
        types={};ids=set()
        for positive,clear in schedule:
            self.assertNotEqual(positive%10//2,0)
            self.assertEqual(clear%10//2,0)
            group=(positive%10//2,positive%2)
            types[group]=types.get(group,0)+1;ids.add(positive)
        self.assertEqual(len(ids),224)
        self.assertLessEqual(max(types.values())-min(types.values()),1)
        batches=[[[0,1,73,74,78,79,83,84]]]
        # Case0 is clear,1 covered,5 clear,6 covered. Quarantine source0.
        available={0,1,5,6,10,11,15,16,20,21,25,26,30,31,35,36}
        result=sanitize_schedule(batches,available,[0],real_count=73,seed=42)
        self.assertEqual(result['batches'][0][0][:2],[0,1])
        self.assertEqual(len(result['batches'][0][0]),8)
        self.assertTrue(all((i-73)//10!=0 for i in result['batches'][0][0][2:]))
        for before,after in zip(batches[0][0][2:],result['batches'][0][0][2:]):
            self.assertEqual((before-73)%5==0,(after-73)%5==0)
            self.assertEqual((before-73)%10>=5,(after-73)%10>=5)
        self.assertEqual(batches[0][0],[0,1,73,74,78,79,83,84])


if __name__=='__main__':unittest.main()
