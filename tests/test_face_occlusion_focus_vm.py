"""Budget/source/optimizer guards; no real detector training."""
import copy
from types import SimpleNamespace
import unittest
from unittest.mock import patch
import torch

from scripts.train_face_occlusion_focus_vm import main, fixed_schedule, counters, restore_optimizer, priority_masks
from tests import test_face_occlusion_focus as fixture_tests


class FocusVMTests(unittest.TestCase):
    def test_linux_cuda_guard_precedes_files_model_and_optimizer(self):
        with patch('scripts.train_face_occlusion_focus_vm.require_vm_gpu', side_effect=RuntimeError('VM only')), \
             patch('scripts.train_face_occlusion_focus_vm.load_adapter') as model:
            with self.assertRaisesRegex(RuntimeError, 'VM only'): main(SimpleNamespace())
            model.assert_not_called()

    def test_matched_schedule_is_one_exact_independent_original_cycle(self):
        protocol = {'schedules': {'extended': {'batches': [[[e, s] for s in range(21)] for e in range(10)]}}}
        before = copy.deepcopy(protocol); first = fixed_schedule(protocol); second = fixed_schedule(protocol)
        self.assertEqual(first, second); self.assertEqual(sum(map(len, first)), 210)
        first[0][0][0] = -1
        self.assertEqual(second, before['schedules']['extended']['batches'])
        self.assertEqual(protocol, before)
        bad = copy.deepcopy(protocol); bad['schedules']['extended']['batches'].pop()
        with self.assertRaises(ValueError): fixed_schedule(bad)

    def test_model_updates_and_optimizer_lifetime_steps_have_distinct_origins(self):
        self.assertEqual(counters(5), {'epoch': 35, 'optimizer_updates': 735,
            'additional_epoch': 25, 'fresh_optimizer_updates': 525,
            'experiment_epoch': 5, 'experiment_updates': 105})
        self.assertEqual(counters(10)['optimizer_updates'], 840)
        self.assertEqual(counters(10)['fresh_optimizer_updates'], 630)
        for e in (0, 11, True, 1.5):
            with self.assertRaises(ValueError): counters(e)

    def test_optimizer_restore_copies_exact_moments_without_alias_or_updates(self):
        payload, fixture_groups = fixture_tests.FocusLossTests().optimizer_fixture()
        groups = [dict(lr=g['lr'], params=[torch.nn.Parameter(p.clone()) for p in g['params']]) for g in fixture_groups]
        optimizer = torch.optim.AdamW(groups, weight_decay=1e-4)
        before = copy.deepcopy(payload)
        summary = restore_optimizer(optimizer, payload, groups, 'a' * 64)
        self.assertEqual(summary['restored_step'], 420)
        restored = optimizer.state_dict()
        for key, state in restored['state'].items():
            for name, tensor in state.items(): self.assertTrue(torch.equal(tensor.cpu(), payload['state']['state'][key][name]))
        optimizer.state[groups[0]['params'][0]]['exp_avg_sq'].zero_()
        self.assertTrue(torch.equal(payload['state']['state'][0]['exp_avg_sq'], before['state']['state'][0]['exp_avg_sq']))

    def priority_fixture(self):
        rows = [dict(split='train', image_sha256=str(i), mask_sha256=str(i)) for i in range(73)]
        for i in (15, 46): rows[i]['glare_stratum'] = 'strong_lens_reflection'
        registry = {'records': [dict(batch_index=i, shape=[256,256], flat_indices=[1, 2],
                                     image_sha256=str(i), mask_sha256=str(i)) for i in (15,46)]}
        return registry, rows

    def test_reflection_maps_accept_only_the_two_original_training_cases(self):
        registry, rows = self.priority_fixture(); result = priority_masks(registry, rows)
        self.assertEqual(set(result), {15,46})
        self.assertEqual(int(result[15].sum()), 2)
        for mistake in ('validation', 'source', 'duplicate', 'out_of_range'):
            registry, rows = self.priority_fixture()
            if mistake == 'validation': rows[15]['split'] = 'validation'
            if mistake == 'source': registry['records'][0]['image_sha256'] = 'heldout'
            if mistake == 'duplicate': registry['records'][0]['flat_indices'] = [1,1]
            if mistake == 'out_of_range': registry['records'][0]['flat_indices'] = [65536]
            with self.assertRaises(ValueError): priority_masks(registry, rows)


if __name__ == '__main__': unittest.main()
