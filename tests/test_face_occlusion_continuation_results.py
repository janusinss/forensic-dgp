"""Continuation evidence checks; fixtures perform no optimization or inference."""
import copy
import hashlib
import io
from pathlib import Path
import tarfile
import tempfile
import unittest

import torch

from scripts.audit_face_occlusion_continuation_results import (
    audit_history, audit_optimizer, checkpoint_audit, member_limit,
    expected_files, extract_return, INVENTORY_PATH, read_json, ROOT, START_SHA, PREFIX,
)


class ContinuationResultTests(unittest.TestCase):
    def history_fixture(self):
        # Distinct batches expose changes to either replay cycle or its order.
        schedule = [[[e, s, 2, 3, 73, 74, 75, 76] for s in range(21)] for e in range(10)]
        real = dict(iou=.1, missed_fraction=.8, visible_false_positive=.02,
                    empty_mask_cases=1, covered_cases=15,
                    negative_false_positive_cases=1, negative_cases=10)
        synthetic = dict(iou=.9, missed_fraction=.1, visible_false_positive=.01,
                         empty_mask_cases=0, covered_cases=320,
                         negative_false_positive_cases=0, negative_cases=80)
        baseline = {'real': real, 'synthetic': synthetic}
        steps = [dict(additional_epoch=e + 1, global_epoch=e + 11, step=s + 1,
                      fresh_optimizer_updates=e * 21 + s + 1,
                      cumulative_optimizer_updates=210 + e * 21 + s + 1,
                      indices=batch, loss=1., pre_clip_norm=.5)
                 for e, batches in enumerate(schedule + schedule)
                 for s, batch in enumerate(batches)]
        history = []
        for n, e in enumerate((1, 5, 10, 20)):
            metrics = copy.deepcopy(baseline)
            metrics['real']['iou'] = .2 + n * .1
            history.append(dict(epoch=e + 10, additional_epoch=e,
                                fresh_optimizer_updates=e * 21,
                                optimizer_updates=210 + e * 21,
                                mean_epoch_loss=1., real_gate=True,
                                synthetic_gate=True, selected=True, **metrics))
        complete = dict(complete=True, parent_unchanged=True,
                        fresh_optimizer_updates=420, cumulative_model_updates=630,
                        source_sha256=START_SHA, selected_epochs=[11, 15, 20, 30])
        return steps, history, schedule, baseline, complete

    def test_exact_two_cycles_and_distinct_update_counters(self):
        result = audit_history(*self.history_fixture())
        self.assertEqual(result['fresh_optimizer_updates'], 420)
        self.assertEqual(result['cumulative_model_updates'], 630)
        self.assertEqual(result['selected_epochs'], [11, 15, 20, 30])

    def test_changed_second_cycle_or_partial_run_fails(self):
        data = self.history_fixture()
        data[0][250]['indices'] = [999] * 8
        with self.assertRaisesRegex(ValueError, 'schedule'):
            audit_history(*data)
        data = self.history_fixture()
        data[0].pop()
        with self.assertRaisesRegex(ValueError, '420'):
            audit_history(*data)

    def test_cumulative_counter_cannot_replace_fresh_counter(self):
        data = self.history_fixture()
        data[0][0]['fresh_optimizer_updates'] = 211
        with self.assertRaisesRegex(ValueError, 'counter'):
            audit_history(*data)
        data = self.history_fixture()
        data[4]['fresh_optimizer_updates'] = 630
        with self.assertRaisesRegex(ValueError, 'counter'):
            audit_history(*data)

    def test_nan_loss_and_changed_mean_fail(self):
        data = self.history_fixture()
        data[0][0]['loss'] = float('nan')
        with self.assertRaisesRegex(ValueError, 'finite'):
            audit_history(*data)
        data = self.history_fixture()
        data[1][0]['mean_epoch_loss'] = 2.
        with self.assertRaisesRegex(ValueError, 'loss'):
            audit_history(*data)

    def test_false_retention_and_weaker_later_real_scores_cannot_select(self):
        data = self.history_fixture()
        data[1][0]['synthetic']['missed_fraction'] = .11
        with self.assertRaisesRegex(ValueError, 'selection'):
            audit_history(*data)
        data = self.history_fixture()
        data[1][1]['real']['iou'] = .19
        with self.assertRaisesRegex(ValueError, 'selection'):
            audit_history(*data)

    def test_expected_members_include_all_2915_masks_and_final_optimizer(self):
        files = expected_files({'scripts/train.py': 'a' * 64}, selected=False)
        self.assertEqual(sum(name.endswith('.png') for name in files), 2915)
        self.assertEqual(sum(name.endswith('.pth') for name in files), 6)
        self.assertIn(f'{PREFIX}/final_optimizer.pth', files)
        self.assertNotIn(f'{PREFIX}/best_detector.pth', files)
        self.assertEqual(expected_files({'scripts/train.py': 'a' * 64}, selected=True) - files,
                         {f'{PREFIX}/best_detector.pth'})

    def test_optimizer_archive_limit_differs_from_model_and_rejects_escape(self):
        self.assertEqual(member_limit(f'{PREFIX}/final_optimizer.pth'), 128 * 1024 * 1024)
        self.assertEqual(member_limit(f'{PREFIX}/epoch_30.pth'), 64 * 1024 * 1024)
        for path in ('../escape', 'C:/escape', f'{PREFIX}/../escape', f'{PREFIX}//alias'):
            with self.assertRaises(ValueError):
                member_limit(path)

    def optimizer_fixture(self):
        parameters = [[torch.zeros(2)], [torch.zeros(1, 2)]]
        groups = [{'lr': 1e-5, 'params': parameters[0]}, {'lr': 1e-4, 'params': parameters[1]}]
        states = {i: {'step': torch.tensor(420.), 'exp_avg': torch.zeros_like(p),
                      'exp_avg_sq': torch.ones_like(p)}
                  for i, p in enumerate(parameters[0] + parameters[1])}
        stored_groups = [dict(params=[i], lr=g['lr'], weight_decay=1e-4,
                              betas=(.9, .999), eps=1e-8, amsgrad=False,
                              maximize=False, capturable=False, differentiable=False,
                              foreach=None, fused=None)
                         for i, g in enumerate(groups)]
        payload = dict(state={'state': states, 'param_groups': stored_groups},
                       model_sha256='a' * 64, fresh_optimizer_updates=420,
                       cumulative_model_updates=630)
        return payload, groups, 'a' * 64

    def test_optimizer_binding_moments_and_fresh_step_are_verified(self):
        result = audit_optimizer(*self.optimizer_fixture())
        self.assertEqual(result['parameter_states_verified'], 2)
        self.assertEqual(result['step_per_parameter'], 420)
        self.assertEqual(result['parameter_elements'], 4)
        data = self.optimizer_fixture()
        for group in data[0]['state']['param_groups']:
            group['decoupled_weight_decay'] = True
        self.assertEqual(audit_optimizer(*data)['step_per_parameter'], 420)

    def test_optimizer_wrong_binding_rate_or_step_fail(self):
        for mutation in ('binding', 'rate', 'step'):
            data = self.optimizer_fixture()
            if mutation == 'binding': data[0]['model_sha256'] = 'b' * 64
            if mutation == 'rate': data[0]['state']['param_groups'][0]['lr'] = 1e-4
            if mutation == 'step': data[0]['state']['state'][0]['step'] = torch.tensor(630.)
            with self.assertRaises(ValueError): audit_optimizer(*data)

    def test_optimizer_duplicate_ids_shape_nonfinite_and_negative_variance_fail(self):
        for mutation in ('duplicate', 'shape', 'nan', 'negative'):
            data = self.optimizer_fixture()
            if mutation == 'duplicate': data[0]['state']['param_groups'][1]['params'] = [0]
            if mutation == 'shape': data[0]['state']['state'][0]['exp_avg'] = torch.zeros(3)
            if mutation == 'nan': data[0]['state']['state'][0]['exp_avg'][0] = float('nan')
            if mutation == 'negative': data[0]['state']['state'][0]['exp_avg_sq'][0] = -1.
            with self.assertRaises(ValueError): audit_optimizer(*data)

    def test_optimizer_update_counters_must_be_integers(self):
        for field in ('fresh_optimizer_updates', 'cumulative_model_updates'):
            data = self.optimizer_fixture()
            data[0][field] = float(data[0][field])
            with self.assertRaises(ValueError): audit_optimizer(*data)

    def test_optimizer_cannot_switch_to_coupled_decay_or_unknown_settings(self):
        for field, value in (('decoupled_weight_decay', False), ('unreviewed_flag', True)):
            data = self.optimizer_fixture()
            data[0]['state']['param_groups'][0][field] = value
            with self.assertRaises(ValueError): audit_optimizer(*data)

    def checkpoint_fixture(self):
        initial = dict(initialization='pretrained', epoch=10, optimizer_updates=210,
                       pilot_metadata={'previous': True}, source_sha256='f' * 64,
                       model={name: torch.zeros(1) for name in (
                           'network.encoder.weight', 'network.decoder.weight',
                           'network.segmentation_head.weight', 'reference_visible_head.weight',
                           'network.encoder.bn.running_mean')})
        report = dict(epoch=11, optimizer_updates=231, additional_epoch=1,
                      fresh_optimizer_updates=21, selected=False)
        run = {'optimizer_reset': True}
        payload = copy.deepcopy(initial)
        payload.update({k: report[k] for k in ('epoch', 'optimizer_updates',
                                             'additional_epoch', 'fresh_optimizer_updates')})
        payload.update(continuation_metadata=run, selection=report)
        for name in ('network.encoder.weight', 'network.decoder.weight',
                     'network.segmentation_head.weight'):
            payload['model'][name].fill_(1.)
        return payload, initial, run, report

    def test_checkpoint_preserves_original_metadata_and_frozen_source_state(self):
        result = checkpoint_audit(*self.checkpoint_fixture())
        self.assertEqual(result['frozen_tensors_verified'], 2)
        self.assertEqual(result['changed_tensors']['network.encoder.'], 1)

    def test_checkpoint_wrong_metadata_frozen_buffer_or_extra_generator_fail(self):
        for mutation in ('metadata', 'buffer', 'generator'):
            data = self.checkpoint_fixture()
            if mutation == 'metadata': data[0]['pilot_metadata'] = data[2]
            if mutation == 'buffer': data[0]['model']['network.encoder.bn.running_mean'].fill_(1.)
            if mutation == 'generator': data[0]['model']['generator.weight'] = torch.zeros(1)
            with self.assertRaises(ValueError): checkpoint_audit(*data)

    def make_archive(self, path, files):
        with tarfile.open(path, 'w:gz') as tar:
            for name, payload, kind in files:
                info = tarfile.TarInfo(name)
                if kind == 'link':
                    info.type = tarfile.SYMTYPE; info.linkname = '../escape'
                    tar.addfile(info)
                else:
                    info.size = len(payload); tar.addfile(info, io.BytesIO(payload))

    def test_real_package_provenance_extraction_hashes_each_small_member(self):
        inventory = read_json(ROOT / INVENTORY_PATH)
        files = [(p, (ROOT / p).read_bytes(), 'file') for p in (*inventory, INVENTORY_PATH)]
        with tempfile.TemporaryDirectory(dir=ROOT / 'outputs', prefix='continuation_audit_fixture_') as task_tmp:
            folder = Path(task_tmp).resolve()
            self.assertTrue(folder.is_relative_to(ROOT / 'outputs'))
            archive = folder / 'return.tar.gz'; self.make_archive(archive, files)
            hashes = extract_return(archive, folder / 'extracted', inventory)
            self.assertEqual(set(hashes), {*inventory, INVENTORY_PATH})
            for name, payload, _ in files:
                self.assertEqual(hashes[name], hashlib.sha256(payload).hexdigest())

    def test_extraction_rejects_links_duplicates_traversal_and_unknown_files(self):
        member = f'{PREFIX}/epoch_30.pth'
        scenarios = [([(member, b'', 'link')], 'unsupported'),
                     ([(member, b'x', 'file'), (member, b'y', 'file')], 'Duplicate'),
                     ([('../escape', b'x', 'file')], 'canonical'),
                     ([(f'{PREFIX}/unknown.json', b'x', 'file')], 'Unexpected')]
        with tempfile.TemporaryDirectory(dir=ROOT / 'outputs', prefix='continuation_audit_fixture_') as task_tmp:
            folder = Path(task_tmp).resolve()
            self.assertTrue(folder.is_relative_to(ROOT / 'outputs'))
            for i, (files, error) in enumerate(scenarios):
                archive = folder / f'return{i}.tar.gz'; self.make_archive(archive, files)
                with self.assertRaisesRegex(ValueError, error):
                    extract_return(archive, folder / f'extracted{i}', {})
            self.assertFalse((folder / 'escape').exists())

    def test_extraction_preserves_existing_directory_and_rejects_changed_code(self):
        with tempfile.TemporaryDirectory(dir=ROOT / 'outputs', prefix='continuation_audit_fixture_') as task_tmp:
            folder = Path(task_tmp).resolve()
            self.assertTrue(folder.is_relative_to(ROOT / 'outputs'))
            archive = folder / 'return.tar.gz'; output = folder / 'existing'; output.mkdir()
            (output / 'evidence.txt').write_text('preserve')
            self.make_archive(archive, [])
            with self.assertRaisesRegex(ValueError, 'Preserve'):
                extract_return(archive, output, {})
            self.assertEqual((output / 'evidence.txt').read_text(), 'preserve')
            self.make_archive(archive, [('scripts/changed.py', b'changed', 'file')])
            with self.assertRaisesRegex(ValueError, 'provenance'):
                extract_return(archive, folder / 'changed', {'scripts/changed.py': 'a' * 64})


if __name__ == '__main__': unittest.main()
