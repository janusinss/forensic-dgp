"""Forward-only regressions for the corrected candidate inference loader."""
from pathlib import Path
import sys
import unittest
from unittest.mock import patch

import torch
from torch import nn

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
import dgp_frozen_inference_v2 as adapter
from dgp_face_restoration import DGPFaceRestoration


class FrozenCandidateInference(unittest.TestCase):
    def mock_load(self, count=5):
        net = nn.Sequential(*[
            nn.InstanceNorm2d(3, affine=True, track_running_stats=True)
            for _ in range(count)
        ])
        return DGPFaceRestoration(net), {
            "policy": "dgp256-observed-canvas-v1",
            "automatic_app_promotion": False,
            "weights_sha256": "a" * 64,
        }

    def test_kernel_writeback_cannot_change_state_across_tail_batches(self):
        loaded = self.mock_load()
        with patch.object(adapter, "load_dgp_restorer", return_value=loaded):
            model, provenance = adapter.load_frozen_dgp_restorer("audited.pth", expected_sha256="a" * 64)
        before = {key: value.clone() for key, value in model.state_dict().items()}
        parameter_ids = [id(value) for value in model.parameters()]
        calls = []

        def writeback(value, mean, variance, *args):
            # Emulate an eval kernel writing to the supplied statistics.
            calls.append(value.shape[0])
            mean.add_(1)
            variance.add_(1)
            return value

        with patch("torch.nn.functional.instance_norm", side_effect=writeback):
            for size in (6, 1, 6):
                output = model(torch.zeros(size, 3, 256, 256))
                self.assertEqual(tuple(output.shape), (size, 3, 256, 256))
                self.assertFalse(output.requires_grad)
        self.assertEqual(calls, [6] * 5 + [1] * 5 + [6] * 5)
        self.assertEqual(parameter_ids, [id(value) for value in model.parameters()])
        for key, value in model.state_dict().items():
            torch.testing.assert_close(value, before[key], rtol=0, atol=0)
        self.assertEqual(provenance["normalization_layers"], 5)
        self.assertEqual(provenance["policy"], adapter.POLICY)
        self.assertFalse(provenance["automatic_app_promotion"])
        self.assertTrue(all(not value.requires_grad for value in model.parameters()))

    def test_wrong_normalization_schema_is_refused(self):
        with patch.object(adapter, "load_dgp_restorer", return_value=self.mock_load(4)):
            with self.assertRaisesRegex(ValueError, "exactly five"):
                adapter.load_frozen_dgp_restorer("audited.pth", expected_sha256="a" * 64)

    def test_changed_dependency_stops_before_checkpoint_loading(self):
        with patch.object(adapter, "sha", return_value="b" * 64), patch.object(adapter, "load_dgp_restorer") as load:
            with self.assertRaisesRegex(ValueError, "Changed frozen inference dependency"):
                adapter.load_frozen_dgp_restorer("audited.pth", expected_sha256="a" * 64)
            load.assert_not_called()


if __name__ == "__main__":
    torch.set_num_threads(2)
    unittest.main()
