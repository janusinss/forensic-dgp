"""Independent matched-return evidence fixtures; no model fitting."""
import copy
import hashlib
import io
from pathlib import Path
import tarfile
import tempfile
import unittest
from unittest.mock import patch
import torch

from scripts.audit_face_occlusion_focus_results import (
    audit_history, checkpoint_audit, audit_final_optimizer, expected_files,
    member_limit, extract_return, ROOT, PREFIX, MODEL_SHA, OPTIMIZER_SHA, INVENTORY_PATH,
)
from tests import test_face_occlusion_focus as loss_fixtures


class FocusResultTests(unittest.TestCase):
    def history_fixture(self):
        schedule = [[[e, s, 2, 3, 73, 74, 75, 76] for s in range(21)] for e in range(10)]
        real = dict(iou=.1, missed_fraction=.8, visible_false_positive=.02, empty_mask_cases=1,
                    covered_cases=15, negative_false_positive_cases=1, negative_cases=10)
        synthetic = dict(iou=.9, missed_fraction=.1, visible_false_positive=.01, empty_mask_cases=0,
                         covered_cases=320, negative_false_positive_cases=0, negative_cases=80)
        baseline = {'real': real, 'synthetic': synthetic}
        for domain in ('training_real', 'human_real', 'mannequin', 'glare'): baseline[domain] = copy.deepcopy(real)
        steps = {}; histories = {}; arms = {}
        for arm, weight in (('control', 0.), ('component_focus', .25)):
            terms = dict(supervised=.5, teacher=.2, focus=.4, loss=.7 + weight * .4)
            steps[arm] = [dict(experiment_epoch=e+1, global_epoch=e+31, step=s+1,
                experiment_updates=e*21+s+1, cumulative_model_updates=630+e*21+s+1,
                optimizer_state_step=420+e*21+s+1, indices=batch, focus_weight=weight,
                pre_clip_norm=.5, **terms) for e, batches in enumerate(schedule) for s, batch in enumerate(batches)]
            histories[arm] = []
            for e, iou in ((5, .2), (10, .3)):
                scores = copy.deepcopy(baseline); scores['real']['iou'] = iou
                histories[arm].append(dict(epoch=30+e, optimizer_updates=630+21*e,
                    additional_epoch=20+e, fresh_optimizer_updates=420+21*e, experiment_epoch=e,
                    experiment_updates=21*e, arm=arm, mean_epoch_terms=terms.copy(),
                    real_gate=True, synthetic_gate=True, selected=True, **scores))
            arms[arm] = dict(experiment_updates=210, cumulative_model_updates=840,
                            optimizer_state_step=630, selected_epochs=[35,40])
        complete = dict(complete=True, parent_unchanged=True, arms=arms, total_experiment_updates=420,
                        source_model_sha256=MODEL_SHA, source_optimizer_sha256=OPTIMIZER_SHA)
        return steps, histories, schedule, baseline, complete

    def test_exact_arms_steps_means_and_unchanged_selection(self):
        result = audit_history(*self.history_fixture())
        self.assertEqual(result['total_experiment_updates'], 420)
        self.assertEqual(result['arms']['component_focus']['selected_epochs'], [35,40])

    def test_partial_changed_order_and_exchanged_counter_origins_fail(self):
        for mistake in ('partial', 'order', 'model_counter', 'optimizer_counter', 'completion'):
            data = self.history_fixture()
            if mistake == 'partial': data[0]['control'].pop()
            if mistake == 'order': data[0]['component_focus'][100]['indices'] = [999]*8
            if mistake == 'model_counter': data[0]['control'][0]['cumulative_model_updates'] = 421
            if mistake == 'optimizer_counter': data[0]['control'][0]['optimizer_state_step'] = 631
            if mistake == 'completion': data[4]['arms']['control']['experiment_updates'] = 840
            with self.assertRaises(ValueError): audit_history(*data)

    def test_changed_weight_nan_and_inconsistent_total_or_mean_fail(self):
        for mistake in ('weight', 'nan', 'total', 'mean'):
            data = self.history_fixture()
            if mistake == 'weight': data[0]['component_focus'][0]['focus_weight'] = .5
            if mistake == 'nan': data[0]['control'][0]['teacher'] = float('nan')
            if mistake == 'total': data[0]['component_focus'][0]['loss'] += .05
            if mistake == 'mean': data[1]['control'][0]['mean_epoch_terms']['supervised'] = 9.
            with self.assertRaises(ValueError): audit_history(*data)

    def test_weakened_retention_later_real_selection_or_missing_arm_fail(self):
        for mistake in ('retention', 'later_real', 'arm'):
            data = self.history_fixture()
            if mistake == 'retention': data[1]['component_focus'][0]['synthetic']['missed_fraction'] = .2
            if mistake == 'later_real': data[1]['control'][1]['real']['iou'] = .19
            if mistake == 'arm': data[0].pop('control')
            with self.assertRaises(ValueError): audit_history(*data)

    def checkpoint_fixture(self):
        source = dict(format='fixture', initialization='pretrained', epoch=30, optimizer_updates=630,
            additional_epoch=20, fresh_optimizer_updates=420, continuation_metadata={'immutable': True},
            model={'network.encoder.weight': torch.zeros(2), 'network.encoder.running_mean': torch.zeros(2),
                   'network.decoder.weight': torch.zeros(2), 'network.segmentation_head.weight': torch.zeros(2),
                   'reference_visible_head.weight': torch.zeros(2)}, selection={'previous': True})
        run = {'fixed': True}; row = self.history_fixture()[1]['control'][0]
        payload = copy.deepcopy(source)
        payload.update({key: row[key] for key in ('epoch','optimizer_updates','additional_epoch',
                       'fresh_optimizer_updates','experiment_epoch','experiment_updates')})
        payload.update(focus_metadata=run, focus_arm='control', selection=row)
        for key in ('network.encoder.weight','network.decoder.weight','network.segmentation_head.weight'):
            payload['model'][key] += 1
        return payload, source, run, row

    def test_checkpoint_preserves_original_metadata_and_frozen_state(self):
        result = checkpoint_audit(*self.checkpoint_fixture())
        self.assertEqual(result['frozen_tensors_verified'], 2)
        for mistake in ('counter', 'prior_metadata', 'batchnorm', 'head', 'nan', 'unchanged'):
            payload, source, run, row = self.checkpoint_fixture()
            if mistake == 'counter': payload['optimizer_updates'] = 105
            if mistake == 'prior_metadata': payload['continuation_metadata']['immutable'] = False
            if mistake == 'batchnorm': payload['model']['network.encoder.running_mean'][0] = 1
            if mistake == 'head': payload['model']['reference_visible_head.weight'][0] = 1
            if mistake == 'nan': payload['model']['network.decoder.weight'][0] = float('nan')
            if mistake == 'unchanged': payload['model']['network.decoder.weight'].zero_()
            with self.assertRaises(ValueError): checkpoint_audit(payload, source, run, row)

    def optimizer_fixture(self):
        source, groups = loss_fixtures.FocusLossTests().optimizer_fixture(); source['model_sha256'] = MODEL_SHA
        final = dict(state=copy.deepcopy(source['state']), model_sha256='b'*64, experiment_updates=210,
                     optimizer_state_step=630, cumulative_model_updates=840)
        for state in final['state']['state'].values(): state['step'] += 210; state['exp_avg'] += 1
        return final, groups, 'b'*64, source

    def test_optimizer_restored_source_and_final_binding_are_distinct(self):
        report = audit_final_optimizer(*self.optimizer_fixture())
        self.assertEqual(report['step_per_parameter'], 630)
        self.assertEqual(report['parameter_states_verified'], 2)
        for mistake in ('source', 'binding', 'step', 'rate', 'moment', 'counter', 'ids'):
            final, groups, digest, source = self.optimizer_fixture()
            if mistake == 'source': source['model_sha256'] = 'c'*64
            if mistake == 'binding': final['model_sha256'] = 'a'*64
            if mistake == 'step': final['state']['state'][0]['step'] = torch.tensor(420.)
            if mistake == 'rate': final['state']['param_groups'][0]['lr'] = .1
            if mistake == 'moment': final['state']['state'][0]['exp_avg_sq'][0] = float('nan')
            if mistake == 'counter': final['experiment_updates'] = 210.
            if mistake == 'ids': final['state']['param_groups'][1]['params'] = [0]
            with self.assertRaises(ValueError): audit_final_optimizer(final, groups, digest, source)

    def test_fixed_members_cover_2915_masks_eight_weight_files_and_optional_bests(self):
        files = expected_files({}, selected={'control': [], 'component_focus': []})
        self.assertEqual(sum(p.endswith('.png') for p in files), 2915)
        self.assertEqual(sum(p.endswith('.pth') for p in files), 8)
        extra = expected_files({}, selected={'control':[35], 'component_focus':[40]}) - files
        self.assertEqual(extra, {f'{PREFIX}/{a}/best_detector.pth' for a in ('control','component_focus')})
        self.assertEqual(member_limit(f'{PREFIX}/initial_optimizer.pth'), 128*1024*1024)
        self.assertEqual(member_limit(f'{PREFIX}/control/final_optimizer.pth'), 128*1024*1024)
        self.assertEqual(member_limit(f'{PREFIX}/control/epoch_40.pth'), 64*1024*1024)

    def archive(self, path, entries):
        with tarfile.open(path,'w:gz') as tar:
            for name, payload, kind in entries:
                item = tarfile.TarInfo(name)
                if kind == 'link': item.type = tarfile.SYMTYPE; item.linkname = 'escape'
                else: item.size = len(payload)
                tar.addfile(item, io.BytesIO(payload) if kind == 'file' else None)

    def test_extraction_accepts_bound_source_optimizer_and_preserves_evidence(self):
        with tempfile.TemporaryDirectory(dir=ROOT/'outputs',prefix='focus_audit_fixture_') as task_tmp:
            root = Path(task_tmp).resolve(); self.assertTrue(root.is_relative_to(ROOT/'outputs'))
            code = b'verified'; inventory = {'new.py': hashlib.sha256(code).hexdigest()}
            registry = b'{}'; digest = hashlib.sha256(registry).hexdigest(); archive = root/'return.tar.gz'
            entries = [('new.py',code,'file'),(INVENTORY_PATH,registry,'file'),
                       (f'{PREFIX}/initial_optimizer.pth',b'snapshot','file')]
            self.archive(archive, entries)
            with patch('scripts.audit_face_occlusion_focus_results.INVENTORY_SHA',digest):
                hashes = extract_return(archive,root/'extracted',inventory)
                self.assertEqual(hashes[f'{PREFIX}/initial_optimizer.pth'], hashlib.sha256(b'snapshot').hexdigest())
                with self.assertRaisesRegex(ValueError,'Preserve'): extract_return(archive,root/'extracted',inventory)

    def test_extraction_rejects_unsafe_members_and_changed_provenance(self):
        cases = [([(f'{PREFIX}/initial.pth',b'','link')],'unsupported'),
                 ([(f'{PREFIX}/initial.pth',b'x','file')]*2,'Duplicate'),
                 ([('../escape',b'x','file')],'canonical'),
                 ([(f'{PREFIX}/unknown.json',b'x','file')],'Unexpected'),
                 ([('new.py',b'wrong','file')],'provenance')]
        with tempfile.TemporaryDirectory(dir=ROOT/'outputs',prefix='focus_audit_fixture_') as task_tmp:
            root = Path(task_tmp).resolve(); self.assertTrue(root.is_relative_to(ROOT/'outputs'))
            for i,(entries,error) in enumerate(cases):
                archive = root/f'return{i}.tar.gz'; self.archive(archive,entries)
                with self.assertRaisesRegex(ValueError,error): extract_return(archive,root/f'extracted{i}',{'new.py':'a'*64})
            self.assertFalse((root/'escape').exists())


if __name__ == '__main__': unittest.main()
