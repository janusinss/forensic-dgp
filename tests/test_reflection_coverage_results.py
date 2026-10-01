"""Return-audit counterexamples; no optimizer or model training."""
import copy
import hashlib
import importlib.util
import io
from pathlib import Path
import tarfile
import tempfile
import unittest
from unittest.mock import patch

import numpy as np
import torch


class ReturnContracts(unittest.TestCase):
    def setUp(self):
        self.assertIsNotNone(importlib.util.find_spec('scripts.audit_reflection_coverage_results'),
                             'Independent reflection return checker missing')

    def history(self):
        batches = [[[0, 1, 2, 3, 73, 74, 75, 76] for _ in range(21)] for _ in range(12)]
        data = {'core_schedule': {'batches': batches}, 'supplemental_schedule': [[2, 10] for _ in range(252)]}
        real = dict(iou=.1, missed_fraction=.8, visible_false_positive=.02, empty_mask_cases=1,
                    covered_cases=15, negative_false_positive_cases=1, negative_cases=10)
        synthetic = dict(iou=.9, missed_fraction=.1, visible_false_positive=.01, empty_mask_cases=0,
                         covered_cases=320, negative_false_positive_cases=0, negative_cases=80)
        baseline = {k: copy.deepcopy(real) for k in ('real', 'training_real', 'human_real', 'mannequin', 'glare')}
        baseline['synthetic'] = synthetic
        terms = dict(supervised=.5, teacher=.2, supplemental=.4, loss=.8)
        steps, histories, arms = {}, {}, {}
        for arm in ('control', 'reflective'):
            steps[arm] = [dict(experiment_epoch=(n-1)//21+1, step=(n-1)%21+1,
                experiment_updates=n, cumulative_model_updates=630+n, optimizer_state_step=420+n,
                indices=batch, supplemental_indices=[0 if arm=='control' else 2, 10],
                supplemental_weight=.25, pre_clip_norm=.5, **terms)
                for n, batch in enumerate([b for epoch in batches for b in epoch], 1)]
            histories[arm] = []
            for e, iou in ((6, .2), (12, .3)):
                scores = copy.deepcopy(baseline); scores['real']['iou'] = iou
                histories[arm].append(dict(epoch=30+e, optimizer_updates=630+21*e,
                    fresh_optimizer_updates=420+21*e, additional_epoch=20+e, experiment_epoch=e,
                    experiment_updates=21*e, arm=arm, mean_epoch_terms=terms.copy(),
                    real_gate=True, synthetic_gate=True, selected=True, **scores))
            arms[arm] = dict(experiment_updates=252, cumulative_model_updates=882,
                            optimizer_state_step=672, selected_epochs=[36, 42])
        complete = dict(complete=True, parent_unchanged=True, arms=arms, total_experiment_updates=504)
        return steps, histories, data, baseline, complete

    def test_matched_budget_and_gates(self):
        from scripts.audit_reflection_coverage_results import audit_history
        report = audit_history(*self.history())
        self.assertEqual(report['total_experiment_updates'], 504)
        self.assertEqual(report['arms']['reflective']['selected_epochs'], [36, 42])

    def test_rejects_changed_camera_pair_exposures_or_counter_origin(self):
        from scripts.audit_reflection_coverage_results import audit_history
        for mistake in ('short', 'camera', 'arm', 'core', 'moment', 'model', 'complete'):
            args = self.history()
            if mistake=='short': args[0]['control'].pop()
            if mistake=='camera': args[0]['control'][0]['supplemental_indices']=[1, 10]
            if mistake=='arm': args[0]['reflective'][0]['supplemental_indices']=[0, 10]
            if mistake=='core': args[0]['control'][0]['indices']=[0]*8
            if mistake=='moment': args[0]['control'][0]['optimizer_state_step']=631
            if mistake=='model': args[0]['control'][0]['cumulative_model_updates']=421
            if mistake=='complete': args[4]['arms']['reflective']['experiment_updates']=882
            with self.assertRaises(ValueError): audit_history(*args)

    def test_rejects_false_selection_or_loss_reconstruction(self):
        from scripts.audit_reflection_coverage_results import audit_history
        for mistake in ('teacher', 'weight', 'loss', 'mean', 'retention', 'best'):
            args = self.history()
            if mistake=='teacher': args[0]['reflective'][0]['teacher']=float('nan')
            if mistake=='weight': args[0]['reflective'][0]['supplemental_weight']=.5
            if mistake=='loss': args[0]['control'][0]['loss']=.9
            if mistake=='mean': args[1]['control'][0]['mean_epoch_terms']['loss']=.9
            if mistake=='retention': args[1]['reflective'][0]['synthetic']['missed_fraction']=.2
            if mistake=='best': args[1]['control'][1]['real']['iou']=.19
            with self.assertRaises(ValueError): audit_history(*args)

    def test_support_recount_excludes_padding_without_hiding_padding_predictions(self):
        from scripts.audit_reflection_coverage_results import recount_supported
        valid=np.zeros((4,4),bool); valid[1:3,1:3]=True
        target=np.zeros_like(valid); target[1,1]=True
        prediction=target.copy(); prediction[0,0]=True; prediction[1,2]=True
        cases={0: {'mask':target[None].astype('uint8'), 'valid':valid[None].astype('uint8')}}
        scores=recount_supported({0:prediction},cases)
        self.assertEqual(scores['iou'], .5)
        self.assertEqual(scores['visible_false_positive'], 1/3)
        self.assertEqual(scores['ignored_positive_pixels'], 1)
        self.assertEqual(scores['records'][0]['tp'], 1)
        self.assertEqual(scores['records'][0]['fp'], 1)
        with self.assertRaises(ValueError):recount_supported({1:prediction},cases)

    def test_members_include_every_raw_mask_and_optional_best(self):
        from scripts.audit_reflection_coverage_results import expected_files, PREFIX
        plain=expected_files({}, {'control':[], 'reflective':[]})
        self.assertEqual(sum(p.endswith('.png') for p in plain), 4315)
        extra=expected_files({}, {'control':[36], 'reflective':[42]})-plain
        self.assertEqual(extra,{f'{PREFIX}/{a}/best_detector.pth' for a in ('control','reflective')})

    def test_final_moments_require672_steps_and_new_model_binding(self):
        from scripts.audit_reflection_coverage_results import audit_final_optimizer, MODEL_SHA
        from tests.test_face_occlusion_focus import FocusLossTests
        source, groups=FocusLossTests().optimizer_fixture();source['model_sha256']=MODEL_SHA
        final=dict(state=copy.deepcopy(source['state']),model_sha256='b'*64,
                   experiment_updates=252,optimizer_state_step=672,cumulative_model_updates=882)
        for s in final['state']['state'].values():s['step']+=252;s['exp_avg']+=1
        self.assertEqual(audit_final_optimizer(final,groups,'b'*64,source)['step_per_parameter'],672)
        for mistake in ('binding','step','count','rate','shape'):
            bad=copy.deepcopy(final)
            if mistake=='binding':bad['model_sha256']='c'*64
            if mistake=='step':bad['state']['state'][0]['step']=torch.tensor(630.)
            if mistake=='count':bad['experiment_updates']=882
            if mistake=='rate':bad['state']['param_groups'][0]['lr']=.1
            if mistake=='shape':bad['state']['state'][0]['exp_avg']=torch.zeros(99)
            with self.assertRaises(ValueError):audit_final_optimizer(bad,groups,'b'*64,source)

    def test_safe_extraction_rejects_escaping_links_duplicates_and_changed_provenance(self):
        from scripts.audit_reflection_coverage_results import extract_return, ROOT, PREFIX
        cases=[('../escape','file','canonical'),(f'{PREFIX}/initial.pth','link','unsupported'),
               ('new.py','file','provenance'),(f'{PREFIX}/unknown.json','file','Unexpected'),
               (f'{PREFIX}/initial.pth','duplicate','Duplicate')]
        with tempfile.TemporaryDirectory(dir=ROOT/'outputs',prefix='reflection_return_fixture_') as temp:
            folder=Path(temp).resolve();self.assertTrue(folder.is_relative_to(ROOT/'outputs'))
            reference=folder/'new.py';reference.write_bytes(b'verified')
            reference_name=reference.relative_to(ROOT).as_posix()
            for i,(name,kind,error) in enumerate(cases):
                if name=='new.py':name=reference_name
                archive=folder/f'return{i}.tar.gz'
                with tarfile.open(archive,'w:gz') as tar:
                    for _ in range(2 if kind=='duplicate' else 1):
                        item=tarfile.TarInfo(name)
                        if kind=='link':item.type=tarfile.SYMTYPE;item.linkname='escape';tar.addfile(item)
                        else:item.size=1;tar.addfile(item,io.BytesIO(b'x'))
                with self.assertRaisesRegex(ValueError,error):
                    extract_return(archive,folder/f'extracted{i}',{reference_name:hashlib.sha256(b'verified').hexdigest()})
            self.assertFalse((folder/'escape').exists())

    def test_extraction_preserves_existing_output_and_verifies_registered_bytes(self):
        from scripts.audit_reflection_coverage_results import extract_return, ROOT, PREFIX, INVENTORY_PATH
        with tempfile.TemporaryDirectory(dir=ROOT/'outputs',prefix='reflection_return_fixture_') as temp:
            folder=Path(temp).resolve();self.assertTrue(folder.is_relative_to(ROOT/'outputs'))
            source=folder/'new.py';source.write_bytes(b'verified');name=source.relative_to(ROOT).as_posix()
            entries=[(name,b'verified'),(INVENTORY_PATH,b'{}'),(f'{PREFIX}/initial_optimizer.pth',b'moments')]
            archive=folder/'return.tar.gz'
            with tarfile.open(archive,'w:gz') as tar:
                for member,payload in entries:
                    info=tarfile.TarInfo(member);info.size=len(payload);tar.addfile(info,io.BytesIO(payload))
            output=folder/'extracted'
            with patch('scripts.audit_reflection_coverage_results.INVENTORY_SHA',hashlib.sha256(b'{}').hexdigest()):
                hashes=extract_return(archive,output,{name:hashlib.sha256(b'verified').hexdigest()})
                self.assertEqual(hashes[f'{PREFIX}/initial_optimizer.pth'],hashlib.sha256(b'moments').hexdigest())
                with self.assertRaisesRegex(ValueError,'Preserve'):
                    extract_return(archive,output,{name:hashlib.sha256(b'verified').hexdigest()})

    def test_checkpoint_preserves_prior_metadata_and_frozen_tensors(self):
        from scripts.audit_reflection_coverage_results import checkpoint_audit
        initial=dict(format='fixture',initialization='pretrained',epoch=30,optimizer_updates=630,
            additional_epoch=20,fresh_optimizer_updates=420,prior_metadata={'unchanged':True},
            model={'network.encoder.weight':torch.zeros(2),'network.encoder.running_mean':torch.zeros(2),
                   'network.decoder.weight':torch.zeros(2),'network.segmentation_head.weight':torch.zeros(2),
                   'reference_visible_head.weight':torch.zeros(2)},selection={'prior':True})
        run={'fixed':True};row=self.history()[1]['control'][0]
        candidate=copy.deepcopy(initial)
        candidate.update({k:row[k] for k in ('epoch','optimizer_updates','additional_epoch','fresh_optimizer_updates',
                                           'experiment_epoch','experiment_updates')})
        candidate.update(reflection_metadata=run,reflection_arm='control',selection=row)
        for key in ('network.encoder.weight','network.decoder.weight','network.segmentation_head.weight'):
            candidate['model'][key]+=1
        self.assertEqual(checkpoint_audit(candidate,initial,run,row)['frozen_tensors_verified'],2)
        for mistake in ('counter','prior','frozen','head','nan'):
            bad=copy.deepcopy(candidate)
            if mistake=='counter':bad['optimizer_updates']=126
            if mistake=='prior':bad['prior_metadata']['unchanged']=False
            if mistake=='frozen':bad['model']['network.encoder.running_mean'][0]=1
            if mistake=='head':bad['model']['reference_visible_head.weight'][0]=1
            if mistake=='nan':bad['model']['network.decoder.weight'][0]=float('nan')
            with self.assertRaises(ValueError):checkpoint_audit(bad,initial,run,row)


if __name__=='__main__':unittest.main()
