"""CPU array and pre-neural boundary regressions; no model construction."""
import copy
import importlib
from pathlib import Path
import sys
import unittest
from unittest.mock import patch

import numpy as np


class ContractTests(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        cls.c = importlib.import_module('cctv_dgp_spatial_readout_v1_contract')

    def test_center_embedding_keeps_pointwise_operator_and_norm(self):
        w = np.arange(12, dtype=np.float32).reshape(3, 4, 1, 1) / 13
        k = self.c.center_kernel(w)
        self.assertEqual(k.shape, (3, 4, 3, 3))
        self.assertTrue(np.array_equal(k[:, :, 1:2, 1:2], w))
        x = np.arange(100, dtype=np.float32).reshape(4, 5, 5) / 17
        y = np.einsum('oc,chw->ohw', w[:, :, 0, 0], x)
        padded = np.pad(x, ((0, 0), (1, 1), (1, 1)))
        actual = sum(np.einsum('oc,chw->ohw', k[:, :, i, j], padded[:, i:i+5, j:j+5])
                     for i in range(3) for j in range(3))
        self.assertTrue(np.array_equal(actual, y))
        self.assertEqual(np.linalg.norm(k.astype(np.float64)), np.linalg.norm(w.astype(np.float64)))

    def test_projection_is_bound_to_original_not_previous_step(self):
        initial = np.array([1., 2., 3.], dtype=np.float32)
        proposal = initial + np.array([.3, .1, -.2], dtype=np.float32)
        old = initial.copy()
        after, receipt = self.c.project_original_ball(initial, proposal, .03)
        self.assertTrue(np.array_equal(initial, old))
        self.assertLessEqual(np.linalg.norm(after.astype(np.float64)-initial), .03)
        self.assertLess(receipt['scale'], 1.)
        again, _ = self.c.project_original_ball(initial, after + .1, .03)
        self.assertLessEqual(np.linalg.norm(again.astype(np.float64)-initial), .03)

    def test_inside_ball_is_not_changed(self):
        initial = np.array([1., 2.], np.float32)
        proposed = initial + np.array([.001, -.001], np.float32)
        value, receipt = self.c.project_original_ball(initial, proposed, .1)
        self.assertTrue(np.array_equal(value, proposed))
        self.assertEqual(receipt['scale'], 1.)

    def test_extra_parameters_do_not_inflate_common_radius(self):
        self.assertEqual(self.c.common_radius([12., 99., 8.]), .008)
        for bad in [[], [0., 1.], [np.nan, 1.], [-1., 2.]]:
            with self.assertRaises((AssertionError, ValueError)):
                self.c.common_radius(bad)

    def test_nonfinite_or_wrong_dtype_proposal_rejected(self):
        initial = np.array([1., 2.], np.float32)
        for bad in [np.array([np.nan, 2.], np.float32), initial.astype(np.float64), initial[:1]]:
            with self.assertRaises((AssertionError, ValueError)):
                self.c.project_original_ball(initial, bad, .01)

    def fixture_groups(self):
        sources = ['photo/a', 'photo/b']
        keys = {'all', 'clear', 'degraded'} | {s+'/'+p for s in sources
                for p in ['all', 'clear', 'degraded', 'blur_lr24', 'lowlight_lr32', 'motion_lr48', 'compound_lr24']}
        return {k: dict(cases=5, MSE=.1, SSIM=.8, ArcFace_observed_fixed=.7,
                       landmark_high_frequency_MSE=.01, constant_mean_shift_only_MSE=.1) for k in keys}

    def test_raw_pass_does_not_hide_delivered_preservation_failure(self):
        old = self.fixture_groups()
        raw = copy.deepcopy(old)
        for key in raw:
            raw[key]['landmark_high_frequency_MSE'] *= .98
            raw[key]['MSE'] *= .99
        png = copy.deepcopy(raw)
        png['photo/a/clear']['SSIM'] = old['photo/a/clear']['SSIM']-1.1e-6
        self.assertTrue(self.c.stage_gate(old, raw)['pass'])
        self.assertFalse(self.c.stage_gate(old, png)['pass'])

    def test_each_source_must_reach_one_percent(self):
        old = self.fixture_groups()
        new = copy.deepcopy(old)
        for key in new:
            new[key]['landmark_high_frequency_MSE'] *= .98
        new['photo/b/degraded']['landmark_high_frequency_MSE'] = .00995
        self.assertFalse(self.c.stage_gate(old, new)['pass'])

    def test_brightness_and_all_groups_are_retained(self):
        old = self.fixture_groups()
        new = copy.deepcopy(old)
        for key in new:
            new[key]['landmark_high_frequency_MSE'] *= .98
            new[key]['MSE'] = .09
            new[key]['constant_mean_shift_only_MSE'] = .097
        self.assertFalse(self.c.stage_gate(old, new)['pass'])
        del new['photo/a/motion_lr48']
        with self.assertRaises(AssertionError):
            self.c.stage_gate(old, new)

    def test_windows_guard_precedes_any_torch_import(self):
        before = set(sys.modules)
        with patch.object(self.c.platform, 'system', return_value='Windows'):
            with self.assertRaisesRegex(AssertionError, 'manual Linux'):
                self.c.require_manual(Path.cwd())
        self.assertNotIn('torch', set(sys.modules)-before)

    def test_model_factory_cannot_construct_a_local_candidate(self):
        module = importlib.import_module('cctv_dgp_spatial_readout_v1_model')
        before = set(sys.modules)
        with patch.object(self.c.platform, 'system', return_value='Windows'):
            with self.assertRaisesRegex(AssertionError, 'manual Linux'):
                module.build_model(Path.cwd(), Path.cwd(), None, None, 'B')
        self.assertNotIn('torch', set(sys.modules)-before)

    def test_vm_worker_is_importable_for_hash_checks_on_windows(self):
        before = set(sys.modules)
        module = importlib.import_module('cctv_dgp_spatial_readout_v1_vm')
        self.assertTrue(callable(module.machine_guard))
        self.assertNotIn('torch', set(sys.modules)-before)

    def test_training_cli_rejects_local_execution_before_file_lookup(self):
        worker = importlib.import_module('cctv_dgp_spatial_readout_v1_vm')
        before = set(sys.modules)
        argv = ['worker', '--root', str(Path.cwd()/'nonexistent_manual_packet'), '--protocol-sha', '0'*64, '--train']
        with patch.object(sys, 'argv', argv), patch.object(self.c.platform, 'system', return_value='Windows'):
            with self.assertRaisesRegex(AssertionError, 'manual Linux'):
                worker.main()
        self.assertNotIn('torch', set(sys.modules)-before)

    def test_independent_auditor_rejects_a_single_PNG_regression(self):
        audit = importlib.import_module('audit_cctv_dgp_spatial_readout_v1_return')
        before = self.fixture_groups()
        after = copy.deepcopy(before)
        for key in after:
            after[key]['landmark_high_frequency_MSE'] *= .98
        after['photo/b/clear']['ArcFace_observed_fixed'] -= 1.01e-6
        self.assertFalse(audit.independent_gate(before, after)['pass'])

    def test_return_namespace_rejects_traversal_and_link_members(self):
        audit = importlib.import_module('audit_cctv_dgp_spatial_readout_v1_return')
        import tarfile
        for name in ['../escape', '/absolute', 'C:/windows', '..\\escape']:
            member = tarfile.TarInfo(name)
            with self.assertRaises(AssertionError):
                audit.safe_member(member)
        member = tarfile.TarInfo('symlink')
        member.type = tarfile.SYMTYPE
        with self.assertRaises(AssertionError):
            audit.safe_member(member)


if __name__ == '__main__':
    unittest.main(verbosity=2)
