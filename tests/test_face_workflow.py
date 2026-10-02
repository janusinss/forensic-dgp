import io
import json
import zipfile

import cv2
import numpy as np
import unittest
import torch
from PIL import Image

from face_workflow import (FaceWorkflow, decode_image, decode_removal_mask,
                           make_review_bundle, quality_signals, visibility_check)


def sample():
    yy, xx = np.mgrid[:256, :256]
    return np.stack((xx, yy, (xx+yy)//2), -1).astype(np.uint8)


class Completion:
    def __call__(self, image, mask):
        return torch.full_like(image, .8)


class Restoration:
    def __call__(self, image, fidelity):
        assert fidelity == 1
        return torch.zeros_like(image)


def engine():
    workflow = FaceWorkflow(device="cpu")
    workflow.generator = Completion()
    workflow.restorer = Restoration()
    return workflow



class FaceWorkflowTests(unittest.TestCase):
    def test_reviewed_mask_support_and_post_restoration(self):
        original = sample()
        mask = np.zeros((256, 256), np.uint8)
        mask[164:200, 95:160] = 1
        workflow = engine()
        off, metadata = workflow.generate(original, mask, "off")
        on, restored = workflow.generate(original, mask, "on")
        assert np.array_equal(off[mask == 0], original[mask == 0])
        assert np.array_equal(off[mask == 1], on[mask == 1])
        assert np.any(on[mask == 0] != original[mask == 0])
        assert metadata["additional_mask_expansion"] == 0
        assert restored["visible_restoration_blend"] == .5


    def test_nonsquare_empty_bypass_has_no_model_dependency(self):
        original = sample()[20:236]
        workflow = FaceWorkflow(device="cpu", completion_path="missing", restoration_path="missing")
        output, metadata = workflow.generate(original, np.zeros(original.shape[:2], np.uint8), "off")
        assert np.array_equal(output, original)
        assert workflow.generator is None and workflow.restorer is None
        assert (metadata["height"], metadata["width"]) == original.shape[:2]


    def test_nearly_hidden_face_rejected_before_model_load_even_with_background(self):
        mask = np.zeros((256, 256), np.uint8)
        mask[55:235, 42:214] = 1
        assert mask.mean() < .5
        assert visibility_check(mask)["rejected"]
        workflow = FaceWorkflow(device="cpu", completion_path="missing")
        with self.assertRaisesRegex(ValueError, "less-covered"):
            workflow.generate(sample(), mask, "off")
        assert workflow.generator is None
        lower = np.zeros_like(mask)
        lower[145:230, 45:214] = 1
        assert not visibility_check(lower)["rejected"]


    def test_invalid_mask_and_output_fail_without_silent_fallback(self):
        workflow = engine()
        with self.assertRaisesRegex(ValueError, "binary"):
            workflow.generate(sample(), np.full((256, 256), .5), "off")
        workflow.generator = lambda x, m: torch.full_like(x, float("nan"))
        mask = np.zeros((256, 256), np.uint8)
        mask[160:200, 95:160] = 1
        with self.assertRaises(FloatingPointError):
            workflow.generate(sample(), mask, "off")


    def test_input_only_quality_signals_distinguish_detail_blur_and_noise(self):
        yy, xx = np.mgrid[:256, :256]
        gray = (110 + 20*np.sin(xx*.8) + 20*np.sin(yy*.8)).clip(0,255).astype(np.uint8)
        detail = np.repeat(gray[..., None], 3, -1)
        mask = np.zeros((256, 256), np.uint8)
        assert not quality_signals(detail, mask)["suggest_restoration"]
        blurred = cv2.GaussianBlur(detail, (13, 13), 3)
        assert quality_signals(blurred, mask)["blur_variance"] < 24
        noisy = np.clip(detail.astype(float)+np.random.default_rng(7).normal(0, 20, detail.shape),0,255).round().astype(np.uint8)
        signals = quality_signals(noisy, mask)
        assert signals["noise_sigma_255"] >= 8 and signals["suggest_restoration"]


    def test_decode_orientation_mask_dimensions_and_bundle_contents(self):
        original = Image.fromarray(sample()[:128])
        exif = Image.Exif()
        exif[274] = 6
        stream = io.BytesIO()
        original.save(stream, format="JPEG", exif=exif)
        decoded = decode_image(stream.getvalue())
        assert decoded.shape == (256, 128, 3)
        mask_stream = io.BytesIO()
        Image.new("L", (128, 256), 0).save(mask_stream, format="PNG")
        mask = decode_removal_mask(mask_stream.getvalue(), decoded.shape[:2])
        with self.assertRaisesRegex(ValueError, "dimensions"):
            decode_removal_mask(mask_stream.getvalue(), (128, 256))
        bundle = make_review_bundle(decoded, mask, decoded, {"meaning": "estimate"})
        with zipfile.ZipFile(io.BytesIO(bundle)) as archive:
            assert set(archive.namelist()) == {"original.png", "removal-mask.png", "estimate.png", "processing.json", "README.txt"}
            assert json.loads(archive.read("processing.json"))["meaning"] == "estimate"

if __name__ == "__main__":
    unittest.main()
