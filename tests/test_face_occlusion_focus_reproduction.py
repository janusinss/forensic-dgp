"""CPU reproduction/selection fixtures; no optimization of a detector."""
import copy
from pathlib import Path
import tempfile
import unittest
import numpy as np
from PIL import Image
import torch

from scripts.reproduce_face_occlusion_focus_results import cpu_selection, infer_cpu
from scripts.audit_face_occlusion_focus_results import ROOT
from tests import test_face_occlusion_focus_results as fixtures


class FocusReproductionTests(unittest.TestCase):
    def test_cpu_selection_keeps_arm_best_values_independent(self):
        _,histories,_,base,_ = fixtures.FocusResultTests().history_fixture()
        results = {'parent':{'scores':base}}
        for arm,history in histories.items():
            for row in history: results[f'{arm}/{row["epoch"]}'] = {'scores':copy.deepcopy(row)}
        selected,agrees = cpu_selection(results,histories)
        self.assertTrue(agrees)
        self.assertTrue(all(r['selected'] for rows in selected.values() for r in rows))
        results['component_focus/35']['scores']['synthetic']['missed_fraction'] = .2
        selected,agrees = cpu_selection(results,histories)
        self.assertFalse(agrees)
        self.assertFalse(selected['component_focus'][0]['selected'])
        self.assertTrue(selected['control'][0]['selected'])

    def infer_fixture(self, root, mutate=False):
        class Model(torch.nn.Module):
            def __init__(self):
                super().__init__(); self.bias = torch.nn.Parameter(torch.tensor(-10.)); self.register_buffer('running',torch.zeros(1))
            def detect(self,x):
                if mutate: self.running += 1
                return self.bias.expand(len(x),1,*x.shape[2:])
        data = {d:[(torch.zeros(3,2,2),torch.zeros(1,2,2)) for _ in range(2)]
                for d in ('real','synthetic','training_real')}
        rows = [dict(image='face.png',glare_stratum='strong_lens_reflection'),dict(image='new_covered_40.png')]
        score = dict(iou=0.,missed_fraction=0.,visible_false_positive=0.,empty_mask_cases=0,
                     covered_cases=0,negative_false_positive_cases=0,negative_cases=2)
        logged = {d:score.copy() for d in data}
        for d in ('human_real','mannequin','glare'): logged[d] = {**score,'negative_cases':1}
        for d in data:
            (root/d).mkdir()
            for i in range(2): Image.fromarray(np.zeros((2,2),dtype='uint8')).save(root/d/f'{i:04}.png')
        return Model(),data,rows,root,logged,tuple(data),'fixture'

    def test_all_saved_pixels_and_metrics_are_recomputed_and_differences_reported(self):
        with tempfile.TemporaryDirectory(dir=ROOT/'outputs',prefix='focus_reproduction_fixture_') as task_tmp:
            root = Path(task_tmp).resolve(); self.assertTrue(root.is_relative_to(ROOT/'outputs'))
            args = self.infer_fixture(root)
            mask = np.zeros((2,2),dtype='uint8'); mask[0,0] = 255; Image.fromarray(mask).save(root/'synthetic/0001.png')
            result = infer_cpu(*args,progress=False)
            self.assertTrue(result['model_state_unchanged'])
            self.assertEqual(result['mask_reproduction']['synthetic']['different_pixels'],1)
            self.assertEqual(result['mask_reproduction']['synthetic']['nonexact_cases'],[{'index':1,'different_pixels':1}])
            self.assertTrue(all(v == 0 for d in result['metric_deltas_cpu_minus_logged'].values() for v in d.values()))

    def test_read_only_inference_rejects_mutated_model_state(self):
        with tempfile.TemporaryDirectory(dir=ROOT/'outputs',prefix='focus_reproduction_fixture_') as task_tmp:
            root = Path(task_tmp).resolve(); self.assertTrue(root.is_relative_to(ROOT/'outputs'))
            with self.assertRaisesRegex(ValueError,'changed model'):
                infer_cpu(*self.infer_fixture(root,mutate=True),progress=False)


if __name__ == '__main__': unittest.main()
