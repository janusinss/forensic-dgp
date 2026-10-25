"""Unit tests for Stage 1 Conditioned SFT V1 conditioning & saved-array feasibility."""
import json
from pathlib import Path
import sys
import unittest

import numpy as np
import torch

from research.codeformer_conditioned_sft_learning_v1 import conditioning


ROOT = Path(__file__).resolve().parents[1]
RECEIPT_PATH = (
    ROOT
    / 'outputs'
    / 'codeformer_conditioned_sft_learning_v1_feasibility'
    / 'feasibility_receipt_v1.json'
)


class TestConditioningAndFeasibilityV1(unittest.TestCase):
    def test_pure_import_loads_no_worker_or_model(self):
        contract = conditioning.conditioning_contract()
        self.assertEqual(
            contract['status'],
            'PURE_INPUT_ONLY_EXPOSURE_DEGRADATION_CONDITIONING_V1',
        )
        self.assertFalse(contract['target_or_role_label_used_in_gates'])
        self.assertTrue(contract['zero_update_exact_u64_parity_preserved'])
        self.assertNotIn(
            'research.codeformer_conditioned_sft_learning_v1.worker',
            sys.modules,
        )

    def test_block16_mean_idempotence_and_orthogonality(self):
        rng = np.random.default_rng(42)
        arr = rng.uniform(0.0, 1.0, size=(256, 256, 3)).astype(np.float32)
        b1 = conditioning.block16_mean_numpy(arr, np=np)
        b2 = conditioning.block16_mean_numpy(b1, np=np)
        self.assertTrue(np.allclose(b1, b2, atol=1e-12))
        detail = arr.astype(np.float64) - b1
        b_detail = conditioning.block16_mean_numpy(detail, np=np)
        self.assertTrue(np.allclose(b_detail, 0.0, atol=1e-12))

    def test_numpy_and_torch_conditioning_parity(self):
        rng = np.random.default_rng(7)
        inp = rng.uniform(0.15, 0.95, size=(256, 256, 3)).astype(np.float32)
        pred = np.clip(
            inp + rng.normal(0.0, 0.03, size=(256, 256, 3)), 0.0, 1.0
        ).astype(np.float32)

        cond_np, gates_np = conditioning.condition_output_numpy(
            pred, inp, np=np, baseline_rgb=None, beta=0.005, gamma=1.0
        )
        inp_t = torch.from_numpy(inp.transpose(2, 0, 1)[None]).to(torch.float32)
        pred_t = torch.from_numpy(pred.transpose(2, 0, 1)[None]).to(
            torch.float32
        ).requires_grad_(True)
        cond_t, gates_t = conditioning.condition_output_torch(
            pred_t, inp_t, torch=torch, beta=0.005, gamma=1.0
        )
        cond_t_hwc = np.clip(
            cond_t.detach().numpy()[0].transpose(1, 2, 0), 0.0, 1.0
        )
        self.assertAlmostEqual(gates_np['g_dark'], gates_t['g_dark'], places=10)
        self.assertAlmostEqual(gates_np['g_deg'], gates_t['g_deg'], places=10)
        self.assertTrue(np.allclose(cond_np, cond_t_hwc, atol=1e-6))

        loss = cond_t.mean()
        loss.backward()
        self.assertIsNotNone(pred_t.grad)
        self.assertTrue(torch.isfinite(pred_t.grad).all().item())

    def test_zero_update_gamma_zero_preserves_exact_prediction(self):
        rng = np.random.default_rng(99)
        inp = rng.uniform(0.0, 1.0, size=(256, 256, 3)).astype(np.float32)
        pred = rng.uniform(0.0, 1.0, size=(256, 256, 3)).astype(np.float32)
        cond_np, _ = conditioning.condition_output_numpy(
            pred, inp, np=np, baseline_rgb=None, beta=0.005, gamma=0.0
        )
        self.assertTrue(np.array_equal(cond_np, pred))

    def test_saved_array_feasibility_receipt_verifies_gates(self):
        self.assertTrue(RECEIPT_PATH.is_file(), f'Missing receipt: {RECEIPT_PATH}')
        receipt = json.loads(RECEIPT_PATH.read_text(encoding='utf-8'))
        self.assertEqual(
            receipt['status'],
            'SAVED_ARRAY_PHOTOMETRIC_AND_GATE_FEASIBILITY_VERIFIED_V1',
        )
        self.assertTrue(receipt['normal_lit_vs_lowlight_zero_overlap'])
        unmod = receipt['variants']['unmodified_B8']
        self.assertFalse(unmod['inh8_pass'])
        self.assertFalse(unmod['sup8_pass'])
        self.assertEqual(unmod['monitor_clear_total_delta_rgb8_codes'], 1126)

        for key in (
            'global_mean_gated_B8',
            'block16_drift_removed_B8',
            'single_pass_input_anchored_B8_beta0p005',
            'drift_removed_plus_input_anchored_B8_beta0p005',
        ):
            v = receipt['variants'][key]
            self.assertTrue(v['inh8_pass'], f'{key} failed inh8')
            self.assertTrue(v['sup8_pass'], f'{key} failed sup8')
            self.assertLess(v['monitor_clear_total_delta_rgb8_codes'], 0)

        for key in (
            'single_pass_input_anchored_B8_beta0p005',
            'drift_removed_plus_input_anchored_B8_beta0p005',
        ):
            v = receipt['variants'][key]
            self.assertTrue(
                v['all_gates_pass_u8_and_u48'],
                f'{key} must pass all 56+40 gates at both u8 and u48',
            )


if __name__ == '__main__':
    unittest.main()
