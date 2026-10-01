"""Small deterministic loss fixtures; no model fitting or optimizer steps."""
import copy
import math
import unittest

import numpy as np
import torch

from face_occlusion_focus import component_weights, focus_loss, validate_source, validate_optimizer


class FocusLossTests(unittest.TestCase):
    def target(self):
        target = np.zeros((24, 24), dtype=bool)
        target[2, 2] = True
        target[12:16, 12:16] = True
        return target

    def test_small_and_large_components_each_have_equal_mass(self):
        target = self.target(); before = target.copy()
        weights, info = component_weights(target)
        self.assertAlmostEqual(float(weights.sum()), 1., places=6)
        self.assertAlmostEqual(float(weights[2, 2]), .25)
        self.assertAlmostEqual(float(weights[12:16, 12:16].sum()), .25)
        self.assertAlmostEqual(float(weights[~target].sum()), .5)
        self.assertEqual(info['component_pixels'], [1, 16])
        self.assertEqual(info['components'], 2)
        self.assertTrue(np.array_equal(target, before))

    def test_ring_excludes_other_occlusions_and_weights_are_deterministic(self):
        target = np.zeros((10, 10), dtype=bool); target[2, 2] = target[2, 4] = True
        first, _ = component_weights(target)
        again, _ = component_weights(target)
        self.assertTrue(np.array_equal(first, again))
        self.assertAlmostEqual(float(first[target].sum()), .5)
        self.assertEqual(float(first[9, 9]), 0.)
        diagonal = np.eye(3, dtype=bool)
        self.assertEqual(component_weights(diagonal)[1]['components'], 1)

    def test_reviewed_glare_touching_large_mask_gets_its_own_region_mass(self):
        target = np.zeros((12, 12), dtype=bool); target[4:8, 4:8] = True; target[3, 4] = True
        priority = np.zeros_like(target); priority[3, 4] = True
        regular, info = component_weights(target)
        separated, separated_info = component_weights(target, priority)
        self.assertEqual(info['components'], 1)
        self.assertEqual(separated_info['components'], 2)
        self.assertAlmostEqual(float(separated[priority].sum()), .25)
        self.assertAlmostEqual(float(separated[target & ~priority].sum()), .25)
        self.assertGreater(float(separated[priority].sum()), float(regular[priority].sum()))
        outside = priority.copy(); outside[0, 0] = True
        with self.assertRaises(ValueError): component_weights(target, outside)

    def test_all_covered_and_clear_targets_are_finite(self):
        full = np.ones((4, 4), dtype=bool); empty = np.zeros((4, 4), dtype=bool)
        full_weights, full_info = component_weights(full)
        empty_weights, empty_info = component_weights(empty)
        self.assertAlmostEqual(float(full_weights.sum()), 1.)
        self.assertEqual(full_info['ring_pixels'], [0])
        self.assertEqual(empty_info['components'], 0)
        self.assertEqual(float(empty_weights.sum()), 0.)
        logits = torch.zeros(2, 1, 4, 4)
        masks = torch.from_numpy(np.stack([full, empty])[:, None].astype('float32'))
        weights = torch.from_numpy(np.stack([full_weights, empty_weights])[:, None])
        self.assertAlmostEqual(float(focus_loss(logits, masks, weights)), math.log(2), places=6)

    def test_loss_penalizes_missing_small_region_and_visible_ring_spill(self):
        target = self.target(); weights, _ = component_weights(target)
        masks = torch.from_numpy(target.astype('float32'))[None, None]
        w = torch.from_numpy(weights)[None, None]
        perfect = torch.where(masks.bool(), 10., -10.)
        miss = perfect.clone(); miss[0, 0, 2, 2] = -10.
        spill = perfect.clone(); spill[0, 0, 1:4, 1:4] = 10.
        self.assertGreater(float(focus_loss(miss, masks, w)), float(focus_loss(perfect, masks, w)) + 2.)
        self.assertGreater(float(focus_loss(spill, masks, w)), float(focus_loss(perfect, masks, w)))

    def test_clear_controls_use_hardest_ten_percent_and_distant_visible_pixels_are_zero_weighted(self):
        target = torch.zeros(1, 1, 10, 10); weights = torch.zeros_like(target)
        logits = torch.full_like(target, -10.); logits[0, 0, 0, 0] = 10.
        expected = (torch.nn.functional.softplus(torch.tensor(10.)) + 9 *
                    torch.nn.functional.softplus(torch.tensor(-10.))) / 10
        self.assertAlmostEqual(float(focus_loss(logits, target, weights)), float(expected), places=6)
        mask = self.target(); w, _ = component_weights(mask)
        m = torch.from_numpy(mask.astype('float32'))[None, None]
        base = torch.zeros_like(m); changed = base.clone(); changed[0, 0, 23, 23] = 10.
        self.assertEqual(float(focus_loss(base, m, torch.from_numpy(w)[None, None])),
                         float(focus_loss(changed, m, torch.from_numpy(w)[None, None])))

    def test_invalid_masks_maps_and_nonfinite_logits_fail(self):
        for mask in (np.zeros((2, 2), dtype=float), np.zeros((2, 2, 1), dtype=bool)):
            with self.assertRaises(ValueError): component_weights(mask)
        z = torch.zeros(1, 1, 2, 2); m = torch.ones_like(z); w = torch.full_like(z, .25)
        for logits, masks, weights in ((z + float('nan'), m, w), (z, m * .5, w),
                                      (z, m, -w), (z, m, w * 2), (z, m[:, :, :, :1], w)):
            with self.assertRaises(ValueError): focus_loss(logits, masks, weights)

    def optimizer_fixture(self):
        groups = [{'params': [torch.zeros(2)], 'lr': 1e-5}, {'params': [torch.zeros(1)], 'lr': 1e-4}]
        stored = {'param_groups': [dict(params=[i], lr=g['lr'], weight_decay=1e-4, betas=(.9, .999),
                                         eps=1e-8, amsgrad=False, maximize=False, capturable=False,
                                         differentiable=False, foreach=None, fused=None)
                                   for i, g in enumerate(groups)],
                  'state': {i: dict(step=torch.tensor(420.), exp_avg=torch.zeros_like(g['params'][0]),
                                    exp_avg_sq=torch.ones_like(g['params'][0])) for i, g in enumerate(groups)}}
        return {'state': stored, 'model_sha256': 'a' * 64, 'fresh_optimizer_updates': 420,
                'cumulative_model_updates': 630}, groups

    def test_only_verified_epoch30_source_and_correct_optimizer_moments_are_allowed(self):
        payload = dict(initialization='pretrained', epoch=30, optimizer_updates=630,
                       additional_epoch=20, fresh_optimizer_updates=420)
        validate_source(payload)
        for key, value in [('epoch', 10), ('initialization', 'random'), ('optimizer_updates', 420)]:
            with self.assertRaises(ValueError): validate_source({**payload, key: value})
        optimizer, groups = self.optimizer_fixture()
        before = copy.deepcopy(optimizer)
        self.assertEqual(validate_optimizer(optimizer, groups, 'a' * 64)['parameter_states'], 2)
        for i in optimizer['state']['state']:
            self.assertTrue(torch.equal(optimizer['state']['state'][i]['exp_avg_sq'],
                                        before['state']['state'][i]['exp_avg_sq']))

    def test_misbound_wrong_step_nonfinite_or_duplicated_optimizer_fails(self):
        for mistake in ('binding', 'step', 'nonfinite', 'ids'):
            payload, groups = self.optimizer_fixture()
            if mistake == 'binding': payload['model_sha256'] = 'b' * 64
            if mistake == 'step': payload['state']['state'][0]['step'] = torch.tensor(630.)
            if mistake == 'nonfinite': payload['state']['state'][0]['exp_avg'][0] = float('nan')
            if mistake == 'ids': payload['state']['param_groups'][1]['params'] = [0]
            with self.assertRaises(ValueError): validate_optimizer(payload, groups, 'a' * 64)


if __name__ == '__main__': unittest.main()
