"""CPU prediction reproduction contracts; no training or optimizer."""
import copy
import importlib.util
from pathlib import Path
import tempfile
import unittest

import numpy as np
from PIL import Image
import torch


class PixelDetector(torch.nn.Module):
    def __init__(self, mutate=False, nonfinite=False):
        super().__init__()
        self.scale=torch.nn.Parameter(torch.tensor(1.))
        self.register_buffer('calls',torch.tensor(0))
        self.mutate=mutate;self.nonfinite=nonfinite

    def detect(self,x):
        if self.mutate:self.calls.add_(1)
        return torch.full_like(x[:,:1],float('nan')) if self.nonfinite else (x[:,:1]-.5)*self.scale


class ReproductionContracts(unittest.TestCase):
    def setUp(self):
        self.assertIsNotNone(importlib.util.find_spec('scripts.reproduce_reflection_coverage_results'),
                             'CPU reflection checkpoint reproduction missing')

    def pixels(self):
        rgb=torch.zeros(3,4,4,dtype=torch.uint8);rgb[:,1,1]=255;rgb[:,0,0]=255
        target=torch.zeros(1,4,4,dtype=torch.uint8);target[:,1,1]=1
        valid=torch.zeros_like(target);valid[:,1:3,1:3]=1
        return {9:{'input':rgb,'mask':target,'valid':valid}}

    def saved(self,folder,positive_padding=True):
        mask=np.zeros((4,4),np.uint8);mask[1,1]=255
        if positive_padding:mask[0,0]=255
        Image.fromarray(mask).save(folder/'0009.png')

    def test_compares_raw_padding_pixels_without_changing_supported_scores(self):
        from scripts.reproduce_reflection_coverage_results import infer_fixture_cpu
        root=Path(__file__).resolve().parents[1]
        with tempfile.TemporaryDirectory(dir=root/'outputs',prefix='reflection_cpu_fixture_') as temp:
            folder=Path(temp);self.saved(folder,False)
            model=PixelDetector()
            report=infer_fixture_cpu(model,self.pixels(),folder,progress=False)
            self.assertEqual(report['scores']['iou'],1.)
            self.assertEqual(report['scores']['ignored_positive_pixels'],1)
            self.assertEqual(report['mask_reproduction']['cases'],1)
            self.assertEqual(report['mask_reproduction']['different_pixels'],1)
            self.assertFalse(report['mask_reproduction']['all_exact'])
            self.assertTrue(report['model_state_unchanged'])
            self.assertIsNone(model.scale.grad)
            self.assertEqual(model.calls.item(),0)

    def test_rejects_changed_model_state(self):
        from scripts.reproduce_reflection_coverage_results import infer_fixture_cpu
        root=Path(__file__).resolve().parents[1]
        with tempfile.TemporaryDirectory(dir=root/'outputs',prefix='reflection_cpu_fixture_') as temp:
            folder=Path(temp);self.saved(folder)
            with self.assertRaisesRegex(ValueError,'state'):
                infer_fixture_cpu(PixelDetector(mutate=True),self.pixels(),folder,progress=False)

    def test_rejects_nonfinite_predictions_and_extra_saved_masks(self):
        from scripts.reproduce_reflection_coverage_results import infer_fixture_cpu
        root=Path(__file__).resolve().parents[1]
        with tempfile.TemporaryDirectory(dir=root/'outputs',prefix='reflection_cpu_fixture_') as temp:
            folder=Path(temp);self.saved(folder)
            with self.assertRaisesRegex(ValueError,'logits'):
                infer_fixture_cpu(PixelDetector(nonfinite=True),self.pixels(),folder,progress=False)
            Image.fromarray(np.zeros((4,4),np.uint8)).save(folder/'0010.png')
            with self.assertRaisesRegex(ValueError,'membership'):
                infer_fixture_cpu(PixelDetector(),self.pixels(),folder,progress=False)

    def test_independent_cpu_selection_requires_retention_before_advancing_best(self):
        from scripts.reproduce_reflection_coverage_results import cpu_selection
        from tests.test_reflection_coverage_results import ReturnContracts
        _,histories,_,baseline,_=ReturnContracts().history()
        reproductions={'parent':{'scores':baseline}}
        for arm in ('control','reflective'):
            for row in histories[arm]:
                reproductions[f'{arm}/{row["epoch"]}']={'scores':copy.deepcopy({k:row[k] for k in baseline})}
        choices,agree=cpu_selection(reproductions,histories)
        self.assertTrue(agree)
        self.assertTrue(all(r['selected'] for arm in choices.values() for r in arm))
        reproductions['control/36']['scores']['synthetic']['missed_fraction']=.2
        reproductions['control/42']['scores']['real']['iou']=.19
        choices,agree=cpu_selection(reproductions,histories)
        self.assertFalse(agree)
        self.assertFalse(choices['control'][0]['selected'])
        self.assertTrue(choices['control'][1]['selected'])

    def test_requires_complete_hash_bound_return_audit_before_cpu_inference(self):
        from scripts.reproduce_reflection_coverage_results import verify_audit_binding
        good={'archive_sha256':'archive','script_sha256':'checker','checksum_verified':True,
              'saved_masks_recounted':4315,'log_audit':{'total_experiment_updates':504},'promoted':False}
        verify_audit_binding(good,'archive','checker')
        for key,value in (('archive_sha256','other'),('script_sha256','other'),('checksum_verified',False),
                          ('saved_masks_recounted',2915),('log_audit',{'total_experiment_updates':420})):
            with self.assertRaises(ValueError):verify_audit_binding({**good,key:value},'archive','checker')


if __name__=='__main__':unittest.main()
