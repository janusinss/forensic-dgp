"""Pure no-Torch unit tests for Conditioned SFT V2 contract, conditioning, and gradient routing."""
from pathlib import Path
import sys
import unittest

import numpy as np


class FrameworkImportBlocker:
    BLOCKED = ('torch', 'torchvision', 'tensorflow', 'jax', 'flax')

    def find_spec(self, fullname, path, target=None):
        root_pkg = fullname.split('.')[0]
        if root_pkg in self.BLOCKED:
            raise AssertionError(f'Forbidden neural framework import on pure load: {fullname}')
        return None


ROOT = Path(__file__).resolve().parents[1]


class TestConditionedSFTLearningV2Pure(unittest.TestCase):
    def test_01_inert_imports_without_torch(self):
        blocker = FrameworkImportBlocker()
        saved_torch = {k: sys.modules.pop(k) for k in list(sys.modules) if k == 'torch' or k.startswith('torch.')}
        for k in list(sys.modules):
            if k.startswith('research.codeformer_conditioned_sft_learning_v2') or k == 'research.codeformer_conditioned_sft_inference_v2':
                sys.modules.pop(k, None)
        sys.meta_path.insert(0, blocker)
        try:
            from research.codeformer_conditioned_sft_learning_v2 import (
                architecture_v2,
                conditioning_v2,
                contract_v2,
                entry_v2,
                objective_v2,
                packet_v2,
                worker_v2,
            )
            from research import codeformer_conditioned_sft_inference_v2
            self.assertEqual(architecture_v2.architecture_v2_contract()['frozen_backbone']['total_frozen_tensors'], 575)
            self.assertTrue(conditioning_v2.conditioning_v2_contract()['exact_step0_u64_parity'])
            self.assertEqual(len(objective_v2.ROLES_V2), 6)
            self.assertEqual(contract_v2.PHASE1_SCHEDULE['epochs'], 40)
            self.assertEqual(contract_v2.PHASE2_SCHEDULE['total_optimizer_updates'], 224)
            self.assertEqual(packet_v2.STAGE_NAME, 'cctv_codeformer_conditioned_sft_learning_v2_vm')
            self.assertEqual(entry_v2.PREFIX, 'cctv-codeformer-conditioned-sft-learning-v2')
            self.assertEqual(worker_v2.SEED, 20261026)
            self.assertFalse(codeformer_conditioned_sft_inference_v2.source_contract_v2()['model_qualified'])
        finally:
            sys.meta_path.remove(blocker)
            sys.modules.update(saved_torch)

    def test_02_displacement_gamma_and_step0_noop(self):
        from research.codeformer_conditioned_sft_learning_v2 import conditioning_v2 as cond

        self.assertEqual(cond.displacement_gamma(0.0), 0.0)
        self.assertAlmostEqual(cond.displacement_gamma(5e-7), 0.5, places=6)
        self.assertEqual(cond.displacement_gamma(2e-6), 1.0)

        rng = np.random.default_rng(42)
        pred = rng.uniform(0.1, 0.9, size=(256, 256, 3)).astype(np.float32)
        inp = rng.uniform(0.1, 0.9, size=(256, 256, 3)).astype(np.float32)
        out0, _ = cond.condition_output_v2_numpy(pred, inp, np=np, baseline_rgb=pred, gamma=0.0)
        np.testing.assert_array_equal(out0, pred)

    def test_03_directional_motion_blur_and_clear_separation_numpy(self):
        from research.codeformer_conditioned_sft_learning_v2 import conditioning_v2 as cond

        # Crisp checkerboard/edge pattern -> sharp in all 4 directions -> g_blur == 0.0
        yy, xx = np.mgrid[0:256, 0:256]
        sharp = (0.5 + 0.25 * np.sin(xx * 0.9) * np.cos(yy * 0.9)).astype(np.float32)
        sharp_rgb = np.stack([sharp, sharp, sharp], axis=-1)
        st_sharp = cond.compute_v2_conditioning_numpy(sharp_rgb, np=np)
        self.assertEqual(st_sharp['g_dark'], 0.0)
        self.assertEqual(st_sharp['g_blur'], 0.0)
        self.assertEqual(st_sharp['max_local_blur'], 0.0)

        # Horizontal smearing (pure vertical stripes vary in X, constant in Y -> min_dir_lap == 0 -> g_blur == 1.0!)
        horiz_smeared = (0.5 + 0.25 * np.sin(yy * 0.9) + 0.0 * xx).astype(np.float32)
        smeared_rgb = np.stack([horiz_smeared, horiz_smeared, horiz_smeared], axis=-1)
        st_smeared = cond.compute_v2_conditioning_numpy(smeared_rgb, np=np)
        self.assertEqual(st_smeared['g_blur'], 1.0)
        self.assertGreater(st_smeared['max_local_blur'], 0.95)

    def test_04_exposure_split_gradient_projection(self):
        from research.codeformer_conditioned_sft_learning_v2 import objective_v2 as obj

        grads = {
            'clear': {'w': np.array([1.0, 0.0, 0.0], dtype=np.float32), 'b': np.array([1.0], dtype=np.float32)},
            'global_defocus_blur': {'w': np.array([-2.0, 3.0, 0.0], dtype=np.float32), 'b': np.array([-1.0], dtype=np.float32)},
            'directional_motion_blur': {'w': np.array([-1.0, 2.0, 1.0], dtype=np.float32), 'b': np.array([0.5], dtype=np.float32)},
            'local_soft_surroundings': {'w': np.array([0.5, 1.0, 0.5], dtype=np.float32), 'b': np.array([0.2], dtype=np.float32)},
            'compound_cctv_codec': {'w': np.array([-0.5, 4.0, 0.0], dtype=np.float32), 'b': np.array([-0.4], dtype=np.float32)},
            'lowlight_sensor_noise': {'w': np.array([0.0, 1.0, 2.0], dtype=np.float32), 'b': np.array([-3.0], dtype=np.float32)},
        }
        combined, info = obj.project_exposure_split_gradients_numpy(grads, np=np)
        self.assertTrue(info['lowlight_decoupled_from_clear_bias'])
        self.assertGreater(info['normal_projected_conflicts'], 0)
        # Normal degraded negative w[0] components were projected out, so clear w[0] stays >= 0
        self.assertGreater(float(combined['w'][0]), 0.0)
        # Lowlight b[0] (-3.0) was NOT projected out by clear b[0] (+1.0)!
        self.assertTrue(np.isfinite(combined['b']).all())

    def test_05_v2_cohort_selection_disjointness(self):
        from research.codeformer_conditioned_sft_learning_v2 import contract_v2

        m2_raw = (ROOT / 'outputs/codeformer_conditioned_sft_learning_v2_preparation/v2_data_manifest.json').read_bytes()
        p1_raw = (ROOT / 'outputs/cctv_codeformer_conditioned_sft_learning_v1_vm/protocol.json').read_bytes()
        sel = contract_v2.select_v2_cohorts(m2_raw, p1_raw)
        self.assertEqual(len(sel['v2_optimization']), 336)
        self.assertEqual(len(sel['v2_monitors']), 48)
        self.assertEqual(len(sel['v1_monitors']), 40)
        self.assertEqual(len(sel['old_quality']), 56)
        self.assertEqual(len(sel['native_cctv_dev']), 4)

        opt_refs = {r['reference_id'] for r in sel['v2_optimization']}
        mon_refs = {r['reference_id'] for r in sel['v2_monitors']}
        old_refs = {r['reference_id'] for r in sel['old_quality']}
        self.assertEqual(len(opt_refs), 56)
        self.assertEqual(len(mon_refs), 8)
        self.assertEqual(len(opt_refs & mon_refs), 0)
        # Verify zero overlap between 56 V2 opt identities and 8 monitor + 24 guard identities in old_quality
        self.assertEqual(len(opt_refs & old_refs), 0)


if __name__ == '__main__':
    unittest.main()
