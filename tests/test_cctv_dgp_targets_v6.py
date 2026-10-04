"""Offline controls and safety tests; no CUDA/backward/optimizer calls."""
import copy
from pathlib import Path
import unittest
from unittest import mock

import numpy as np
from PIL import Image

from cctv_dgp_targets_v6 import ARMS, arm_protocol, reduced_target, require_vm, safe_path
from cctv_dgp_pilot import qualifies


class TargetControlTests(unittest.TestCase):
    def test_targets_change_without_changing_inputs_identity_or_validation(self):
        p = {"references": [{"id": "train", "role": "train", "target": "hq.png",
                             "reduced_target": "reduced.png", "identity_target": "hq.png"},
                            {"id": "val", "role": "validation", "target": "validation.png"}],
             "training_epochs": {"1": [{"input": "one_shared_input.png"}]},
             "validation_cases": [{"input": "common_eval.png"}]}
        original = copy.deepcopy(p)
        control, hq = [arm_protocol(p, arm) for arm in ARMS]
        self.assertEqual(p, original)
        self.assertNotEqual(control["references"][0]["target"], hq["references"][0]["target"])
        self.assertEqual(control["references"][0]["identity_target"], hq["references"][0]["identity_target"])
        self.assertEqual(control["references"][1], hq["references"][1])
        self.assertEqual(control["training_epochs"], hq["training_epochs"])
        self.assertEqual(control["validation_cases"], hq["validation_cases"])

    def test_control_removes_high_frequency_without_changing_canvas_or_color(self):
        y, x = np.indices((256, 256))
        rgb = np.stack([((x + y) % 2) * 255, np.full_like(x, 80), np.full_like(x, 120)], -1).astype(np.uint8)
        result = np.asarray(reduced_target(Image.fromarray(rgb)))
        self.assertEqual(result.shape, rgb.shape)
        self.assertLess(result[:, :, 0].std(), rgb[:, :, 0].std() / 10)
        np.testing.assert_array_equal(result[:, :, 1:], rgb[:, :, 1:])

    def test_unsafe_asset_names_are_rejected(self):
        directory = Path.cwd() / "scratch"
        for value in ("../weights.pth", "/tmp/weights", "C:/weights", "a\\b", ""):
            with self.assertRaises(ValueError):
                safe_path(directory, value)
        self.assertEqual(safe_path(directory, "weights/start.pth"), directory / "weights/start.pth")

    def test_local_training_is_rejected_before_importing_cuda(self):
        with mock.patch("cctv_dgp_targets_v6.sys.platform", "win32"):
            with self.assertRaisesRegex(RuntimeError, "only on"):
                require_vm(Path.cwd())
        with mock.patch("cctv_dgp_targets_v6.sys.platform", "linux"), mock.patch("cctv_dgp_targets_v6.platform.node", return_value="other-host"):
            with self.assertRaisesRegex(RuntimeError, "only on"):
                require_vm(Path.cwd())

    def test_aggregate_gain_cannot_hide_source_profile_regression(self):
        base = {key: {"cases": 10, "identity_pairs": 10, "MSE": .1, "SSIM": .6,
                      "ArcFace_observed_fixed": .5} for key in ("degraded", "source/blur")}
        candidate = copy.deepcopy(base)
        candidate["degraded"]["MSE"] = .08
        candidate["source/blur"]["MSE"] = .1001
        self.assertFalse(qualifies(candidate, base, base))
        candidate["source/blur"]["MSE"] = .09
        self.assertTrue(qualifies(candidate, base, base))
        candidate["source/blur"]["ArcFace_observed_fixed"] = .49
        self.assertFalse(qualifies(candidate, base, base))


if __name__ == "__main__":
    unittest.main()
