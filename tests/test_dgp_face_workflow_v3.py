import base64
import asyncio
import io
import json
import unittest
import zipfile
from unittest.mock import patch

import numpy as np
import torch
from fastapi import HTTPException, UploadFile

import face_workflow_web as web
from dgp_face_restoration import prepare_crop
from dgp_face_workflow_v3 import (DGPFaceWorkflow, array_sha, canonical_tensor,
                                  decode_crop, make_dgp_bundle)
from face_workflow import FaceWorkflow, png_bytes


class Completion(torch.nn.Module):
    def __init__(self):
        super().__init__()
        self.calls = 0

    def forward(self, image, mask):
        self.calls += 1
        return torch.full_like(image, .7)


class Restorer(torch.nn.Module):
    def __init__(self):
        super().__init__()
        self.calls = 0
        self.input = None

    def forward(self, image):
        self.calls += 1
        self.input = image.clone()
        return image * .5


class WorkflowTests(unittest.TestCase):
    def setUp(self):
        torch.set_num_threads(1)
        self.engine = DGPFaceWorkflow(device="cpu")
        self.engine.generator = Completion()
        self.engine.restorer = Restorer()
        self.engine.generator_provenance = {"backend": "test-completion"}
        self.engine.restorer_provenance = {"backend": "test-DGP"}
        self.rgb = np.random.default_rng(2).integers(0, 256, (48, 64, 3), dtype=np.uint8)
        self.mask = np.zeros((48, 64), np.uint8)
        self.mask[15:22, 25:31] = 1

    def generate(self, **kwargs):
        return self.engine.generate(self.rgb, self.mask, input_review="usable", mask_reviewed=True, **kwargs)

    def test_review_rejections_happen_before_neural_calls(self):
        for decision, message in (("", "Review the input"), ("needs_clearer", "clearer"), ("out_of_scope", "frontal")):
            with self.subTest(decision=decision), self.assertRaisesRegex(ValueError, message):
                self.engine.generate(self.rgb, self.mask, "on", decision, True)
        with self.assertRaisesRegex(ValueError, "confirm"):
            self.engine.generate(self.rgb, self.mask, "on", "usable", False)
        self.assertEqual(self.engine.restorer.calls + self.engine.generator.calls, 0)

    def test_tiny_native_decoder_and_off_canvas(self):
        tiny = self.rgb[:16, :20].copy()
        np.testing.assert_array_equal(decode_crop(png_bytes(tiny)), tiny)
        expected, _, _, _ = prepare_crop(tiny)
        result = self.engine.generate(tiny, np.zeros(tiny.shape[:2], np.uint8), "off", "usable", True)
        np.testing.assert_array_equal(result["output"], expected)
        self.assertEqual(result["output"].shape, (256, 256, 3))
        self.assertEqual(self.engine.restorer.calls + self.engine.generator.calls, 0)

    def test_blank_and_nearly_covered_inputs_fail_before_models(self):
        with self.assertRaisesRegex(ValueError, "clearer"):
            self.engine.generate(np.full_like(self.rgb, 90), self.mask, "on", "usable", True)
        mask = np.ones_like(self.mask)
        mask[:4] = 0
        with self.assertRaisesRegex(ValueError, "less-covered"):
            self.engine.generate(self.rgb, mask, "on", "usable", True)
        self.assertEqual(self.engine.restorer.calls + self.engine.generator.calls, 0)

    def test_off_exact_and_on_uses_only_dgp_for_visible_pixels(self):
        off = self.generate(restoration="off")
        on = self.generate(restoration="on")
        canvas, support, mask, _ = prepare_crop(self.rgb, self.mask)
        np.testing.assert_array_equal(off["output"][~mask], canvas[~mask])
        np.testing.assert_array_equal(off["output"][mask], on["output"][mask])
        np.testing.assert_array_equal(on["output"][~support], canvas[~support])
        visible = support & ~mask
        expected = np.floor(on["raw"] * np.float32(255)).astype(np.uint8)
        np.testing.assert_array_equal(on["output"][visible], expected[visible])
        np.testing.assert_array_equal(self.engine.restorer.input.numpy(), canonical_tensor(canvas, "cpu").numpy())
        self.assertEqual(on["metadata"]["visible_restoration"]["backend"], "test-DGP")
        self.assertEqual(on["metadata"]["additional_mask_expansion"], 0)

    def test_auto_and_overrides_use_observed_input_only(self):
        auto = self.generate(restoration="auto")
        expected = auto["metadata"]["quality_signals"]["suggest_restoration"]
        self.assertEqual(auto["metadata"]["restoration_applied"], expected)
        self.assertFalse(auto["metadata"]["quality_signals"]["structure_qualification_established"])
        self.assertTrue(self.generate(restoration="on")["metadata"]["restoration_applied"])
        self.assertFalse(self.generate(restoration="off")["metadata"]["restoration_applied"])

    def test_grayscale_presentation_is_separate_from_raw(self):
        values = self.rgb[..., :1]
        gray = np.repeat(values, 3, axis=-1)
        result = self.engine.generate(gray, self.mask, "on", "usable", True)
        np.testing.assert_array_equal(result["output"][..., 0], result["output"][..., 1])
        self.assertTrue(result["metadata"]["display_processing"]["colour_policy"]["applied"])
        self.assertEqual(result["metadata"]["raw_dgp"]["sha256"], array_sha(result["raw"]))

    def test_proposal_binding_and_small_margin(self):
        fake = {"original": self.rgb, "raw_mask": self.mask, "mask": self.mask,
                "proposal_margin_pixels": 1}
        with patch.object(FaceWorkflow, "review_mask", return_value=fake):
            proposed = self.engine.review_mask(self.rgb)
        self.assertEqual(proposed["proposal_margin_pixels"], 0)
        self.assertEqual(self.generate(restoration="off")["metadata"]["mask_source"], "automatic_reviewed")
        self.mask[0, 0] = 1
        self.assertEqual(self.generate(restoration="off")["metadata"]["mask_source"], "assisted_reviewed")

    def test_invalid_models_never_produce_a_result(self):
        class Bad(torch.nn.Module):
            def forward(self, image):
                return torch.full_like(image, float("nan"))
        self.engine.restorer = Bad()
        with self.assertRaisesRegex(FloatingPointError, "stopped"):
            self.generate(restoration="on")
        self.engine.restorer = None
        self.engine.restoration_path = self.engine.restoration_path.with_name("missing.pth")
        with self.assertRaises(FileNotFoundError):
            self.generate(restoration="on")

    def test_bundle_preserves_native_input_geometry_and_raw(self):
        result = self.generate(restoration="on")
        bundle = make_dgp_bundle(self.rgb, self.mask, result["original"], result["mask"], result["output"], result["metadata"], result["raw"])
        with zipfile.ZipFile(io.BytesIO(bundle)) as archive:
            np.testing.assert_array_equal(decode_crop(archive.read("original.png")), self.rgb)
            raw = np.load(io.BytesIO(archive.read("dgp-raw-float32.npy")), allow_pickle=False)
            np.testing.assert_array_equal(raw, result["raw"])
            self.assertEqual(json.loads(archive.read("processing.json"))["optimizer_updates"], 0)

    def test_live_api_requires_both_reviews_and_returns_256(self):
        def call(review, confirmed):
            crop = UploadFile(file=io.BytesIO(png_bytes(self.rgb)), filename="crop.png")
            mask = UploadFile(file=io.BytesIO(png_bytes(self.mask * 255)), filename="mask.png")
            return asyncio.run(web.generate_endpoint(crop, mask, "off", True, review, confirmed))
        with patch.object(web, "engine", self.engine):
            with self.assertRaises(HTTPException) as failure:
                call("", False)
            self.assertEqual(failure.exception.status_code, 422)
            self.assertIn("Review the input", failure.exception.detail)
            result = call("usable", True)
            output = decode_crop(base64.b64decode(result["output"].split(",", 1)[1]))
            self.assertEqual(output.shape, (256, 256, 3))
            self.assertEqual(result["metadata"]["mask_source"], "assisted_reviewed")


if __name__ == "__main__":
    unittest.main()
