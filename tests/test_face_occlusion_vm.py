"""VM guard, fixed sampler and common-baseline selection contracts."""
import copy
import json
from pathlib import Path
from types import SimpleNamespace
import unittest
from unittest.mock import patch

import torch

from scripts.train_face_occlusion_vm import main, validate_schedule, selection, dependency_versions, DEPENDENCIES


class FaceOcclusionVMTests(unittest.TestCase):
    def test_pinned_runtime_rejects_version_drift_and_inherited_modules(self):
        target = Path('outputs/face_occlusion_dependencies').resolve()
        versions = {name:version for name,(_,version) in DEPENDENCIES.items()}
        def distribution(name):return SimpleNamespace(version=versions[name])
        def module(name):return SimpleNamespace(__file__=str(target/name/'__init__.py'))
        with patch('scripts.train_face_occlusion_vm.metadata.distribution',side_effect=distribution), \
             patch('scripts.train_face_occlusion_vm.importlib.import_module',side_effect=module):
            report=dependency_versions(target)
            self.assertEqual({k:v['version'] for k,v in report.items()},versions)
            versions['timm']='1.0.16'
            with self.assertRaisesRegex(ValueError,'timm'):
                dependency_versions(target)
            versions['timm']=DEPENDENCIES['timm'][1]
        with patch('scripts.train_face_occlusion_vm.metadata.distribution',side_effect=distribution), \
             patch('scripts.train_face_occlusion_vm.importlib.import_module',
                   return_value=SimpleNamespace(__file__=str(target.parent/'inherited.py'))):
            with self.assertRaisesRegex(ValueError,'Pinned dependency'):
                dependency_versions(target)

    def test_linux_cuda_guard_precedes_files_model_and_optimizer(self):
        with patch('scripts.train_face_occlusion_vm.require_vm_gpu', side_effect=RuntimeError('VM required')), \
             patch('scripts.train_face_occlusion_vm.load_adapter') as load:
            with self.assertRaisesRegex(RuntimeError, 'VM required'):
                main(SimpleNamespace())
            load.assert_not_called()

    def frozen_inputs(self):
        protocol = json.loads(Path('outputs/coverage_protocol_v1/protocol.json').read_text())
        rows = [r for r in json.loads(Path('dataset/detector_training_extension_v2/manifest.json').read_text())['records'] if r['split']=='train']
        ids = {i-73 for e in protocol['schedules']['extended']['batches'] for b in e for i in b if i>=73}
        cache = {i:(None,torch.tensor([int(i%5!=0)])) for i in ids}
        return protocol,rows,cache

    def test_exact_mixed_schedule_and_replay_membership(self):
        args = self.frozen_inputs()
        schedule = validate_schedule(*args)
        self.assertEqual(sum(len(e) for e in schedule), 210)
        bad = copy.deepcopy(args[0]);bad['schedules']['extended']['batches'][0][0][0] = 100000
        with self.assertRaises(ValueError):
            validate_schedule(bad,*args[1:])
        bad = copy.deepcopy(args[0]);batch=bad['schedules']['extended']['batches'][0][0]
        real_clear=next(i for i,r in enumerate(args[1]) if r['kind']=='uncovered')
        batch[:] = [real_clear]*4 + batch[4:]
        with self.assertRaises(ValueError):
            validate_schedule(bad,*args[1:])

    def test_selection_uses_common_parent_and_all_retention_metrics(self):
        baseline={'real':dict(iou=.1,visible_false_positive=.01,empty_mask_cases=3,negative_false_positive_cases=1),
                  'synthetic':dict(iou=.97,missed_fraction=.02,visible_false_positive=.001,empty_mask_cases=1,negative_false_positive_cases=0)}
        scores=copy.deepcopy(baseline);scores['real']['iou']=.2
        self.assertEqual(selection(scores,baseline,.1),(True,True,True))
        scores['synthetic']['missed_fraction']=.021
        self.assertEqual(selection(scores,baseline,.1),(True,False,False))
        scores['synthetic']['missed_fraction']=.02;scores['real']['iou']=.05
        self.assertEqual(selection(scores,baseline,.1),(False,True,False))


if __name__=='__main__':unittest.main()
