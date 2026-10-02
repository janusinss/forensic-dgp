"""Native-expert contracts use fixtures; no actual model training/optimizer."""
import copy
import unittest
from unittest.mock import patch

import torch


def rows():
    return [{'split': 'train', 'kind': 'covered' if i < 51 else 'uncovered'}
            for i in range(83)]


def fixtures():
    return [{'case_id': i, 'style': 'clear' if i % 10 < 2 else 'reflection'}
            for i in range(280)]


def moment_fixture():
    parameters = [torch.zeros(2), torch.zeros(3)]
    groups = [{'params': [parameters[0]], 'lr': 1e-5},
              {'params': [parameters[1]], 'lr': 1e-4}]
    common = {'betas': (.9, .999), 'eps': 1e-8, 'weight_decay': 1e-4,
              'amsgrad': False, 'maximize': False, 'foreach': None,
              'capturable': False, 'differentiable': False, 'fused': None,
              'decoupled_weight_decay': True}
    payload = {'model_sha256': 'a' * 64, 'experiment_updates': 252,
               'optimizer_state_step': 672, 'cumulative_model_updates': 882,
               'state': {'param_groups': [{**common, 'lr': g['lr'], 'params': [i]}
                                          for i, g in enumerate(groups)],
                         'state': {i: {'step': torch.tensor(672.),
                                       'exp_avg': torch.zeros_like(p),
                                       'exp_avg_sq': torch.ones_like(p)}
                                   for i, p in enumerate(parameters)}}}
    return groups, payload


def score(tp=20, fp=1, fn=2, empty=0, clear_errors=0):
    return {'tp': tp, 'fp': fp, 'fn': fn, 'iou': tp / (tp + fp + fn),
            'missed_fraction': fn / (tp + fn), 'visible_false_positive': fp / 100,
            'empty_mask_cases': empty, 'negative_false_positive_cases': clear_errors}


def baseline_fixture():
    return {'original_real': score(), 'native_real': score(), 'reflection': score(),
            'all_real': score(), 'lens': {'old': 10, 'new': 20},
            'native_cases': {str(i): {'tp': 5, 'fp': 0, 'fn': 5}
                             for i in (171, 216, 348, 374)}}


class NativeExpertContracts(unittest.TestCase):
    def test_schedule_is_bounded_training_only_and_covers_each_fixture_each_epoch(self):
        from native_expert import fixed_schedule
        schedule = fixed_schedule(rows(), fixtures())
        self.assertEqual(schedule, fixed_schedule(rows(), fixtures()))
        self.assertEqual(len(schedule), 6)
        for epoch in schedule:
            self.assertEqual(len(epoch), 56)
            real_ids, fixture_ids = [], []
            for batch in epoch:
                self.assertEqual(len(batch['real']), 3)
                self.assertEqual(sum(rows()[i]['kind'] == 'covered' for i in batch['real']), 2)
                self.assertEqual(len(batch['fixture']), 5)
                self.assertEqual(sum(i % 10 >= 2 for i in batch['fixture']), 4)
                self.assertEqual(len(set(batch['real'])), 3)
                real_ids.extend(batch['real']); fixture_ids.extend(batch['fixture'])
            self.assertEqual(set(real_ids), set(range(83)))
            self.assertEqual(sorted(fixture_ids), list(range(280)))
        bad = rows(); bad[0]['split'] = 'validation'
        with self.assertRaises(ValueError): fixed_schedule(bad, fixtures())

    def test_source42_moments_require_exact_counters_shapes_and_finite_values(self):
        from native_expert import validate_source_optimizer
        groups, payload = moment_fixture()
        summary = validate_source_optimizer(payload, groups, 'a' * 64)
        self.assertEqual(summary['restored_step'], 672)
        for change in ('step', 'counter', 'shape', 'finite', 'negative', 'lr'):
            bad = copy.deepcopy(payload)
            if change == 'step': bad['state']['state'][0]['step'] = torch.tensor(420.)
            if change == 'counter': bad['cumulative_model_updates'] = 630
            if change == 'shape': bad['state']['state'][0]['exp_avg'] = torch.zeros(3)
            if change == 'finite': bad['state']['state'][0]['exp_avg'][0] = float('nan')
            if change == 'negative': bad['state']['state'][0]['exp_avg_sq'][0] = -1
            if change == 'lr': bad['state']['param_groups'][0]['lr'] = 1e-4
            with self.subTest(change=change), self.assertRaises(ValueError):
                validate_source_optimizer(bad, groups, 'a' * 64)

    def test_counters_do_not_confuse_new_updates_with_model_or_moment_lifetime(self):
        from native_expert import counters
        self.assertEqual(counters(6), {'epoch': 48, 'optimizer_updates': 1218,
                         'additional_epoch': 38, 'fresh_optimizer_updates': 1008,
                         'experiment_epoch': 6, 'experiment_updates': 336})
        for invalid in (0, 7, True):
            with self.assertRaises(ValueError): counters(invalid)

    def test_ignored_fixture_logits_never_change_either_domain_loss_or_gradient(self):
        from native_expert import expert_loss
        real = torch.zeros(3, 1, 4, 4, requires_grad=True)
        fixture = torch.zeros(5, 1, 4, 4, requires_grad=True)
        real_target = torch.zeros_like(real); real_target[:, :, 0, 0] = 1
        target = torch.zeros_like(fixture); target[:, :, 0, 0] = 1
        valid = torch.ones_like(fixture); valid[:, :, 3] = 0
        actual, _, _ = expert_loss(real, real_target, torch.ones_like(real), fixture, target, valid)
        altered = fixture.detach().clone(); altered[:, :, 3] = 100
        expected, _, _ = expert_loss(real, real_target, torch.ones_like(real), altered, target, valid)
        self.assertTrue(torch.equal(actual, expected))
        actual.backward()
        self.assertEqual(int(torch.count_nonzero(fixture.grad[:, :, 3])), 0)
        self.assertGreater(int(torch.count_nonzero(fixture.grad[:, :, :3])), 0)

    def test_native_gain_cannot_hide_clear_errors_or_old_expert_regression(self):
        from native_expert import fit_decision
        before = baseline_fixture(); after = copy.deepcopy(before)
        after['native_real'] = score(24, 1, 1)
        after['lens']['new'] = 24
        for value in after['native_cases'].values(): value['tp'] += 1
        self.assertTrue(fit_decision(before, after)['passes'])
        for change in ('clear', 'old_iou', 'fixture_fp', 'one_lens', 'old_lens'):
            bad = copy.deepcopy(after)
            if change == 'clear': bad['all_real']['negative_false_positive_cases'] = 1
            if change == 'old_iou': bad['original_real'] = score(10, 1, 12)
            if change == 'fixture_fp': bad['reflection'] = score(24, 2, 1)
            if change == 'one_lens': bad['native_cases']['171']['tp'] = 5
            if change == 'old_lens': bad['lens']['old'] = 9
            with self.subTest(change=change): self.assertFalse(fit_decision(before, bad)['passes'])

    def test_vm_guard_rejects_windows_even_with_gpu_and_linux_without_cuda(self):
        from native_expert import require_vm_gpu
        for platform, available in (('win32', True), ('linux', False)):
            with self.subTest(platform=platform), patch('native_expert.sys.platform', platform), \
                    patch('native_expert.torch.cuda.is_available', return_value=available):
                with self.assertRaises(RuntimeError): require_vm_gpu()


if __name__ == '__main__': unittest.main()
