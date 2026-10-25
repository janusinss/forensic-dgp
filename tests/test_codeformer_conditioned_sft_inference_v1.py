"""Unit tests for Conditioned SFT V1 frozen CPU inference adapter."""
import io
from pathlib import Path
import unittest

import numpy as np

from research import codeformer_conditioned_sft_inference_v1 as inf


ROOT = Path(__file__).resolve().parents[1]


class TestConditionedSftInferenceV1(unittest.TestCase):
    def test_source_contract_and_parameter_schemas(self):
        contract = inf.source_contract()
        self.assertEqual(contract['format'], 'conditioned-sft-v1-frozen-CPU-inference-source-v1')
        self.assertEqual(contract['frozen_tensors'], 575)
        self.assertEqual(contract['arm_A_trainable_parameters'], 168960)
        self.assertEqual(contract['arm_B_trainable_parameters'], 220288)
        self.assertEqual(len(inf.adapter_shapes()), 24)
        self.assertEqual(len(inf.head_shapes()), 36)
        self.assertEqual(len(inf.trainable_60_shapes()), 60)

    def test_inspect_trainable_npz_accepts_36_and_60_tensor_bundles(self):
        rng = np.random.default_rng(7)
        for schema in (inf.head_shapes(), inf.trainable_60_shapes()):
            buf = io.BytesIO()
            arrays = {k: rng.standard_normal(s, dtype=np.float32) for k, s in schema.items()}
            np.savez(buf, **arrays)
            parsed = inf.inspect_trainable_npz(buf.getvalue())
            self.assertEqual(set(parsed), set(schema))
            for k in schema:
                np.testing.assert_array_equal(parsed[k], arrays[k])

    def test_apply_conditioned_postprocessing_numpy_zero_and_active_gamma(self):
        inp = np.full((256, 256, 3), 0.75, dtype=np.float32)
        pred = np.full((256, 256, 3), 0.70, dtype=np.float32)
        raw0, rgb8_0, gates0 = inf.apply_conditioned_postprocessing_numpy(
            pred, inp, np=np, gamma=0.0
        )
        np.testing.assert_array_equal(raw0, pred)
        self.assertEqual(gates0['g_dark'], 0.0)

        raw1, rgb8_1, gates1 = inf.apply_conditioned_postprocessing_numpy(
            pred, inp, np=np, baseline_rgb=pred, beta=0.005, gamma=1.0
        )
        # Should shift slightly toward inp (0.75) from pred (0.70)
        self.assertGreater(float(raw1.mean()), float(pred.mean()))
        self.assertEqual(rgb8_1.dtype, np.uint8)
        self.assertEqual(rgb8_1.shape, (256, 256, 3))

        # Dark exposure coarse-drift sign alignment preserves fine detail while flipping 16x16 drift
        dark_inp = np.full((256, 256, 3), 0.25, dtype=np.float32)
        base_dark = np.full((256, 256, 3), 0.50, dtype=np.float32)
        pred_dark = np.full((256, 256, 3), 0.49, dtype=np.float32)
        raw_dark, _, gates_dark = inf.apply_conditioned_postprocessing_numpy(
            pred_dark, dark_inp, np=np, baseline_rgb=base_dark, dark_drift_align=True
        )
        self.assertEqual(gates_dark['g_dark'], 1.0)
        self.assertAlmostEqual(float(raw_dark.mean()), 0.51, places=5)


if __name__ == '__main__':
    unittest.main()
