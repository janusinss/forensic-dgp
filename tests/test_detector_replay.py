import unittest
import torch
from detector_replay import MixedBatches, select_sources, retention_passes, replay_consistency, TaggedReplay


class ReplayTests(unittest.TestCase):
    def test_consistency_only_updates_synthetic_student(self):
        student=torch.zeros(4,1,2,2,requires_grad=True)
        teacher=torch.full_like(student,2.,requires_grad=True)
        loss=replay_consistency(student,teacher,torch.tensor([False,False,True,True]))
        loss.backward()
        self.assertEqual(student.grad[:2].abs().sum(),0)
        self.assertGreater(student.grad[2:].abs().sum(),0)
        self.assertIsNone(teacher.grad)

    def test_consistency_is_zero_when_identical_or_no_replay(self):
        x=torch.randn(4,1,2,2,requires_grad=True)
        self.assertAlmostEqual(float(replay_consistency(x,x,torch.ones(4,dtype=torch.bool)).detach()),0.,places=6)
        loss=replay_consistency(x,x,torch.zeros(4,dtype=torch.bool))
        loss.backward();self.assertEqual(x.grad.abs().sum(),0)

    def test_tagged_replay_preserves_source_boundary(self):
        real=[(torch.tensor(1),torch.tensor(0))]*2
        synthetic=[(torch.tensor(2),torch.tensor(1))]*3
        data=TaggedReplay(real,synthetic)
        self.assertEqual(len(data),5)
        self.assertFalse(data[1][2]);self.assertTrue(data[2][2])

    def test_every_batch_contains_four_groups(self):
        groups=[[0,1,2],[3],[4,5,6],[7,8]]
        sampler=MixedBatches(groups,4,21,42)
        batches=list(sampler)
        self.assertEqual(len(batches),21)
        for batch in batches:
            self.assertEqual([sum(i in g for i in batch) for g in groups],[1]*4)
        self.assertEqual(batches,list(sampler))
        sampler.epoch=1
        self.assertNotEqual(batches,list(sampler))

    def test_invalid_groups(self):
        for groups,size in [([[0],[1],[2],[]],4),([[0],[1],[2],[3]],6)]:
            with self.assertRaises(ValueError):MixedBatches(groups,size,2,42)

    def test_sources_exclude_validation_benchmark_and_duplicates(self):
        rows=[('a','A'),('b','B'),('duplicate','A'),('c','C'),('d','D')]
        self.assertEqual(select_sources(rows,{'B','C'},2,42),select_sources(rows,{'B','C'},2,42))
        self.assertEqual({h for _,h in select_sources(rows,{'B','C'},2,42)},{'A','D'})
        with self.assertRaises(ValueError):select_sources(rows,{'B','C'},3,42)

    def test_any_retention_regression_blocks_selection(self):
        base=dict(iou=.97,missed_fraction=.02,visible_false_positive=.001,
                  empty_mask_cases=1,negative_false_positive_cases=0)
        self.assertTrue(retention_passes(base,base))
        for key in base:
            worse=dict(base);worse[key]+=(-.01 if key=='iou' else .01)
            self.assertFalse(retention_passes(worse,base),key)
