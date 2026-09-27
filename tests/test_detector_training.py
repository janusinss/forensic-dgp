import unittest
import torch
from detector_training import validate_records,detector_optimizer,segmentation_loss,BalancedMaskBatches,visible_penalty
from completion import CompletionNet


class DetectorTrainingTests(unittest.TestCase):
    def test_balanced_batches_cover_majority_and_are_reproducible(self):
        rows=[{'kind':'covered'}]*7+[{'kind':'uncovered'}]*3
        a=BalancedMaskBatches(rows,4,12);b=BalancedMaskBatches(rows,4,12)
        batches=list(a);self.assertEqual(batches,list(b))
        self.assertEqual(len(batches),4)
        for batch in batches:self.assertEqual(sum(rows[i]['kind']=='covered' for i in batch),2)
        self.assertEqual({i for batch in batches for i in batch if i<7},set(range(7)))
        a.set_epoch(1);self.assertNotEqual(list(a),batches)

    def test_invalid_balancing_configuration(self):
        for rows,size in [([{'kind':'covered'}],4),(self.records(),3)]:
            with self.assertRaises(ValueError):BalancedMaskBatches(rows,size,42)

    def test_penalty_targets_visible_errors_only(self):
        logits=torch.tensor([[[[-3.,3.],[1.,2.]]]],requires_grad=True)
        mask=torch.tensor([[[[0.,0.],[1.,1.]]]])
        loss=visible_penalty(logits,mask,hard_fraction=.5)
        loss.backward()
        self.assertEqual(logits.grad[0,0,1].abs().sum(),0)
        self.assertGreater(logits.grad[0,0,0,1],0)
        self.assertEqual(logits.grad[0,0,0,0],0)
        self.assertEqual(visible_penalty(logits,torch.ones_like(mask)),0)

    def test_zero_penalty_preserves_legacy_loss(self):
        x=torch.randn(2,1,4,4);m=torch.zeros_like(x)
        self.assertTrue(torch.equal(segmentation_loss(x,m),segmentation_loss(x,m,background_weight=0)))
        self.assertGreater(segmentation_loss(x,m,background_weight=1),segmentation_loss(x,m))

    def records(self):
        return [dict(reviewed=True,split=s,group=f'{s}-{kind}',image_sha256=f'{s}-{kind}',kind=kind)
                for s in ['train','validation','test'] for kind in ['covered','uncovered']]

    def test_pending_labels_rejected(self):
        rows=self.records();rows[0]['reviewed']=False
        with self.assertRaises(ValueError):validate_records(rows)

    def test_cross_split_group_and_duplicate_rejected(self):
        for key in ['group','image_sha256']:
            rows=self.records();rows[2][key]=rows[0][key]
            with self.assertRaises(ValueError):validate_records(rows)

    def test_split_requires_negative_and_positive_controls(self):
        rows=self.records();validate_records(rows)
        with self.assertRaises(ValueError):validate_records(rows[:-1])

    def test_step_changes_only_detector(self):
        torch.set_num_threads(2);model=CompletionNet(4)
        before={k:v.clone() for k,v in model.state_dict().items()}
        optimizer=detector_optimizer(model,1e-3)
        x=torch.rand(2,3,32,32);m=torch.zeros(2,1,32,32);m[0,:,12:24,8:24]=1
        loss=segmentation_loss(model.detect(x),m);loss.backward();optimizer.step()
        for k,v in model.generator.state_dict().items():self.assertTrue(torch.equal(v,before['generator.'+k]))
        self.assertTrue(any(not torch.equal(v,before['segmenter.'+k]) for k,v in model.segmenter.state_dict().items()))
