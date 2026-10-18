"""Inference-only regression checks; no backward, optimizer or restoration training."""
from __future__ import annotations

import json
from pathlib import Path
import unittest
import uuid

import torch

from research.codeformer_adaptation_v1.perceptual import load_perceptual


ROOT = Path(__file__).resolve().parents[1]
ASSETS = ROOT / "outputs/codeformer_perceptual_assets_v1"


class PerceptualIntegrityTests(unittest.TestCase):
    def test_hash_mismatch_stops_before_deserialization(self):
        scratch = (ROOT / "scratch").resolve()
        scratch.mkdir(exist_ok=True)
        # Python 3.13 TemporaryDirectory's 0700 ACL blocks this Windows sandbox.
        # A normal project directory avoids that ACL; remove only our two files.
        directory = scratch / f"perceptual-integrity-{uuid.uuid4().hex}"
        directory.mkdir()
        self.assertEqual(directory.resolve().parent, scratch)
        try:
            (directory / "vgg16_features.pth").write_bytes(b"not a checkpoint")
            manifest = {"schema": "codeformer-lpips-vgg16-assets-v1",
                        "features": {"path": "vgg16_features.pth", "bytes": 16, "sha256": "0" * 64}}
            (directory / "manifest.json").write_text(json.dumps(manifest), encoding="utf-8")
            with self.assertRaisesRegex(ValueError, "fingerprint differs"):
                load_perceptual(directory)
        finally:
            for name in ("manifest.json", "vgg16_features.pth"):
                (directory / name).unlink(missing_ok=True)
            directory.rmdir()


@unittest.skipUnless((ASSETS / "manifest.json").exists(), "Verified perceptual assets not prepared")
class PerceptualInferenceTests(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        torch.set_num_threads(4)
        cls.model = load_perceptual(ASSETS)

    def test_parent_training_mode_keeps_fixed_loss_in_eval(self):
        parent = torch.nn.Module()
        parent.add_module("fixed_loss", self.model)
        parent.train()
        self.assertTrue(parent.training)
        self.assertFalse(any(module.training for module in self.model.modules()))
        self.assertFalse(any(parameter.requires_grad for parameter in self.model.parameters()))

    def test_identity_and_nonidentity_are_scalar_finite_forward_only(self):
        image = torch.linspace(-1, 1, 3 * 32 * 32).reshape(1, 3, 32, 32)
        with torch.inference_mode():
            identity = self.model(image, image)
            changed = self.model(torch.flip(image, dims=(-1,)), image)
        self.assertEqual(identity.ndim, 0)
        self.assertEqual(identity.item(), 0.0)
        self.assertTrue(torch.isfinite(changed).item())
        self.assertGreater(changed.item(), 0.0)

    def test_invalid_shapes_ranges_and_nonfinite_are_rejected(self):
        image = torch.zeros(1, 3, 32, 32)
        with self.assertRaisesRegex(ValueError, "matching NCHW"):
            self.model(image, image[:, :, :-1])
        with self.assertRaisesRegex(ValueError, "finite"):
            self.model(image + float("nan"), image)
        with self.assertRaisesRegex(ValueError, "target must use RGB"):
            self.model(image, image + 2)


if __name__ == "__main__":
    unittest.main()
