"""Completion contracts use plain arrays/toy tensors; no trained model is run."""
import hashlib
import json
from pathlib import Path
import subprocess
import sys
import tempfile
import unittest
from unittest.mock import patch

import numpy as np
from PIL import Image

from implementation import completion_assets as assets
from implementation import completion_runtime as runtime
from implementation import completion_worker as worker

ROOT = Path(__file__).resolve().parents[1]


class CompletionRuntimeTests(unittest.TestCase):
    def setUp(self):
        self.rgb = np.arange(256 * 256 * 3, dtype=np.uint32).reshape(256, 256, 3).astype(np.uint8)
        self.mask = np.zeros((256, 256), np.uint8)
        self.mask[170:195, 110:146] = 1

    def test_import_and_empty_completion_need_no_torch_or_checkpoint(self):
        code = ("import sys,numpy as np;from implementation import completion_runtime as r;"
                "import implementation.completion_worker;import implementation.completion_assets;"
                "x=np.full((256,256,3),17,np.uint8);m=np.zeros((256,256),np.uint8);"
                "a=r.complete(x,m);assert a['metadata']['neural_calls']==0;"
                "assert np.array_equal(a['output_rgb'],x);assert 'torch' not in sys.modules")
        result = subprocess.run([sys.executable, "-B", "-c", code], cwd=ROOT, capture_output=True, text=True)
        self.assertEqual(result.returncode, 0, result.stderr)

    def test_architecture_pins_and_only_line_endings_may_change(self):
        runtime.verify_architecture_sources()
        with tempfile.TemporaryDirectory() as folder:
            path = Path(folder) / "toy.py"
            path.write_bytes(b"value=1\n")
            expected = runtime.architecture_file_sha256(path)
            path.write_bytes(b"value=1\r\n")
            self.assertEqual(runtime.architecture_file_sha256(path), expected)
            path.write_bytes(b"value=2\n")
            self.assertNotEqual(runtime.architecture_file_sha256(path), expected)
            path.write_bytes(b"value=1\r")
            with self.assertRaises(ValueError):
                runtime.architecture_file_sha256(path)

    def test_manifest_separates_completion_from_restoration_and_rejects_drift(self):
        body = assets.load_manifest()
        self.assertEqual(body["component"], "reviewed-r2-face-completion")
        self.assertEqual(body["default_model"], "completion_r2")
        self.assertEqual(set(body["assets"]), {"completion_r2", "covering_detector"})
        self.assertEqual(body["assets"]["completion_r2"]["size"], 370773572)
        self.assertEqual(body["assets"]["completion_r2"]["filename"], "codeformer_completion_r2.pth")
        self.assertEqual(body["assets"]["completion_r2"]["sha256"],
                         "ed9f92baf06a73090343ad00cdc23b8493c9dde738a90cde9bdb650e69fee07c")
        self.assertEqual(body["assets"]["covering_detector"]["sha256"],
                         "ccc22d524073368bc4f5d6b6d2136c90d0e2a117083e4c75fa98d42d7bfff6cf")
        self.assertIsNone(body["assets"]["completion_r2"]["url"])
        with tempfile.TemporaryDirectory() as folder:
            path = Path(folder) / "manifest.json"
            body["assets"]["completion_r2"]["sha256"] = "0" * 64
            path.write_text(json.dumps(body), encoding="utf-8")
            with self.assertRaises(assets.CompletionAssetError):
                assets.load_manifest(path)

    def test_contract_and_empty_bypass_bind_r2_provenance(self):
        contract = runtime.model_contract()
        self.assertEqual(contract["model_id"], "completion_r2")
        self.assertEqual(contract["model_label"], "Trained CodeFormer completion R2")
        self.assertEqual(contract["backend"], "codeformer-inpainting-finetune-r2")
        self.assertEqual(contract["weights_sha256"],
                         "ed9f92baf06a73090343ad00cdc23b8493c9dde738a90cde9bdb650e69fee07c")
        self.assertEqual(contract["logical_state_sha256"],
                         "300b13b8f01db17aa95ee95ca41dbbdfbd4e5e3257dd16657fcc7950d3351ea0")
        self.assertEqual(contract["training_protocol_sha256"],
                         "463c1977e9cd74f0a4fd0240df7a784f48b613043ef667714f1ef94adb4cd60a")
        self.assertEqual(contract["device"], "cpu")
        self.assertFalse(contract["model_qualified"])
        self.assertFalse(contract["hidden_identity_verified"])
        with patch.object(runtime, "load_frozen_cpu_completion", side_effect=AssertionError("must bypass")):
            result = runtime.complete(self.rgb, np.zeros_like(self.mask))
        for key, value in contract.items():
            self.assertEqual(result["metadata"][key], value)
        self.assertEqual(result["metadata"]["neural_calls"], 0)
        self.assertTrue(np.array_equal(result["output_rgb"], self.rgb))

    def test_r1_fingerprint_cannot_be_relabelled_as_r2(self):
        body = assets.load_manifest()
        self.assertIn("completion_r2", body["assets"])
        with tempfile.TemporaryDirectory() as folder:
            path = Path(folder) / "manifest.json"
            body["assets"]["completion_r2"]["sha256"] = (
                "7ae2486739de0f3efa99cf721a0fcb755f8f56cf69e295412f74190d2fda37e0")
            path.write_text(json.dumps(body), encoding="utf-8")
            with self.assertRaises(assets.CompletionAssetError):
                assets.load_manifest(path)

    def test_missing_r2_never_falls_back_to_retained_r1(self):
        with tempfile.TemporaryDirectory() as folder:
            models = Path(folder)
            old = models / "codeformer_completion_r1.pth"
            old.write_bytes(b"retain previous R1 model")
            with self.assertRaisesRegex(assets.CompletionAssetError, "codeformer_completion_r2.pth"):
                assets.asset_path("completion_r2", model_dir=models)
            self.assertEqual(old.read_bytes(), b"retain previous R1 model")

    def test_empty_mask_bypass_cannot_construct_a_model(self):
        with patch.object(runtime, "load_frozen_cpu_completion", side_effect=AssertionError("model must not load")):
            result = runtime.complete(self.rgb, np.zeros_like(self.mask))
        self.assertTrue(result["metadata"]["bypassed"])
        self.assertFalse(result["metadata"]["checkpoint_loaded"])
        self.assertTrue(np.array_equal(result["output_rgb"], self.rgb))

    def test_binary_mask_input_and_canvas_are_validated(self):
        for value in (2, 127):
            invalid = self.mask.copy(); invalid[0, 0] = value
            with self.assertRaises(ValueError):
                runtime.validate_arrays(self.rgb, invalid)
        mixed = self.mask.copy(); mixed[0, 0] = 255
        with self.assertRaises(ValueError):
            runtime.validate_arrays(self.rgb, mixed)
        for invalid in (self.rgb.astype(np.float32), self.rgb[:128], self.rgb[..., 0]):
            with self.assertRaises(ValueError):
                runtime.validate_arrays(invalid, self.mask)

    def test_padding_is_unavailable_and_cannot_be_removed(self):
        removal, bounds, review = runtime.validate_arrays(self.rgb, self.mask, [10, 0, 246, 256])
        self.assertFalse(review["rejected"])
        self.assertTrue(np.array_equal(removal, self.mask))
        mask = self.mask.copy(); mask[20, 0] = 1
        with self.assertRaises(ValueError):
            runtime.validate_arrays(self.rgb, mask, bounds)
        with self.assertRaises(runtime.CompletionRefused):
            runtime.validate_arrays(self.rgb, np.zeros_like(mask), [120, 0, 136, 256])

    def test_visibility_check_letterbox_observed_ratio_and_combo_mask(self):
        # Letterboxed crop (valid observed face >= 45%) with large lower mask should pass.
        observed = np.zeros((256, 256), dtype=bool)
        observed[:, 28:228] = True
        removal = np.zeros((256, 256), dtype=np.uint8)
        removal[130:225, 65:191] = 1
        unavailable = np.where(observed, removal * 255, 255).astype(np.uint8)
        vis = runtime.visibility_check256(unavailable, observed=observed)
        self.assertFalse(vis["rejected"])
        self.assertGreaterEqual(vis["observed_face_fraction"], 0.45)

        # Sunglasses + surgical mask with visible forehead/cheeks (<65% total face covered) passes.
        full_obs = np.ones((256, 256), dtype=bool)
        combo = np.zeros((256, 256), dtype=np.uint8)
        combo[84:133, 60:196] = 255  # eyes band
        combo[119:208, 82:174] = 255  # nose + mouth band
        vis_combo = runtime.visibility_check256(combo, observed=full_obs)
        self.assertLess(vis_combo["face_covered_fraction"], 0.65)
        self.assertFalse(vis_combo["rejected"])

    def test_blend_inside_mask_preserves_outside_bytes_and_matches_seam_tone(self):
        # Clean skin canvas with slightly darker generated interior.
        rgb = np.full((256, 256, 3), (185, 145, 125), dtype=np.uint8)
        removal = np.zeros((256, 256), dtype=np.uint8)
        removal[100:180, 90:170] = 1
        observed = np.ones((256, 256), dtype=bool)
        generated = rgb.astype(np.float32) / np.float32(255.0)
        generated[removal != 0] = np.array([150, 115, 100], dtype=np.float32) / np.float32(255.0)

        blended = runtime._blend_inside_mask(rgb, generated, removal, observed)
        self.assertEqual(blended.dtype, np.float32)
        self.assertTrue(np.isfinite(blended).all())
        self.assertGreaterEqual(float(blended.min()), 0.0)
        self.assertLessEqual(float(blended.max()), 1.0)
        # Bit-exact outside mask preservation in both float32 and uint8 encoding.
        expected_out = rgb.astype(np.float32) / np.float32(255.0)
        self.assertTrue(np.array_equal(blended[removal == 0], expected_out[removal == 0]))
        encoded = runtime.encode_output(rgb, removal, blended)
        self.assertTrue(np.array_equal(encoded[removal == 0], rgb[removal == 0]))
        # Interior seam shifts closer to surrounding skin luminance.
        seam_l_before = float(generated[105:110, 120:140].mean())
        seam_l_after = float(blended[105:110, 120:140].mean())
        self.assertGreater(seam_l_after, seam_l_before)

        # Covering-bleed guard: bright blue covering pixel right inside mask edge must NOT bleed into seam or L1/L2/G3.
        rgb_cov = rgb.copy()
        rgb_cov[removal != 0] = (30, 110, 235)
        blended_cov = runtime._blend_inside_mask(rgb_cov, generated, removal, observed)
        # Outermost pixels (d=1..8) inside mask must stay near skin, zero blue covering bleed (B=235/255=0.92).
        self.assertLess(float(blended_cov[100:108, 120:140, 2].max()), 0.62)
        self.assertTrue(np.array_equal(blended_cov[removal == 0], expected_out[removal == 0]))

    def test_blend_harmonic_side_lighting_multiband_and_grain_matching(self):
        import cv2
        # 1. Strong left-to-right directional side lighting with flat generated interior.
        xx = np.linspace(0.0, 1.0, 256, dtype=np.float32)[None, :]
        grad_r = np.rint(215.0 - 95.0 * xx).clip(0, 255)
        grad_g = np.rint(175.0 - 80.0 * xx).clip(0, 255)
        grad_b = np.rint(155.0 - 70.0 * xx).clip(0, 255)
        rgb = np.repeat(np.stack([grad_r, grad_g, grad_b], axis=2), 256, axis=0).astype(np.uint8)
        removal = np.zeros((256, 256), dtype=np.uint8)
        removal[90:190, 76:180] = 1
        observed = np.ones((256, 256), dtype=bool)
        generated = rgb.astype(np.float32) / np.float32(255.0)
        generated[removal != 0] = np.array([155, 125, 110], dtype=np.float32) / np.float32(255.0)

        blended = runtime._blend_inside_mask(rgb, generated, removal, observed)
        expected_out = rgb.astype(np.float32) / np.float32(255.0)
        self.assertTrue(np.array_equal(blended[removal == 0], expected_out[removal == 0]))
        encoded = runtime.encode_output(rgb, removal, blended)
        self.assertTrue(np.array_equal(encoded[removal == 0], rgb[removal == 0]))

        # Harmonic field must track directional illumination: left interior brighter than right interior
        # (a global scalar Lab shift produces ~0.0; harmonic interpolation tracks >55% of the ~0.097 outer delta).
        left_in = float(blended[120:160, 82:96].mean())
        right_in = float(blended[120:160, 160:174].mean())
        self.assertGreater(left_in - right_in, 0.055)
        # Seam jump on both left and right edges decreases significantly vs raw generated.
        jump_before = abs(float(generated[120:160, 76:80].mean() - expected_out[120:160, 72:76].mean()))
        jump_after = abs(float(blended[120:160, 76:80].mean() - expected_out[120:160, 72:76].mean()))
        self.assertLess(jump_after, jump_before * 0.55)

        # 2. Sensor grain matching on grainy outer skin vs airbrushed interior.
        rng = np.random.default_rng(42)
        base_skin = np.full((256, 256, 3), (175, 140, 120), dtype=np.float32)
        noisy = np.clip(base_skin + rng.normal(0.0, 9.0, size=(256, 256, 3)), 0, 255).astype(np.uint8)
        gen_smooth = noisy.astype(np.float32) / np.float32(255.0)
        gen_smooth[removal != 0] = np.array([175, 140, 120], dtype=np.float32) / np.float32(255.0)
        blended_grain = runtime._blend_inside_mask(noisy, gen_smooth, removal, observed)
        self.assertTrue(np.array_equal(blended_grain[removal == 0], (noisy.astype(np.float32) / np.float32(255.0))[removal == 0]))
        lap_before = np.abs(cv2.Laplacian(cv2.cvtColor(gen_smooth, cv2.COLOR_RGB2GRAY), cv2.CV_32F)[110:170, 96:160]).mean()
        lap_after = np.abs(cv2.Laplacian(cv2.cvtColor(blended_grain, cv2.COLOR_RGB2GRAY), cv2.CV_32F)[110:170, 96:160]).mean()
        self.assertGreater(lap_after, lap_before + 0.01)

    def test_ffhq_umeyama_alignment_bypasses_canonical_and_aligns_rolled_face(self):
        import cv2
        import torch
        from implementation import app as restoration

        face_path = ROOT.parent / "test_images" / "face 1 - default.jpg"
        if not face_path.is_file():
            self.skipTest("face 1 - default.jpg fixture not present")
        with Image.open(face_path) as im:
            native = np.array(im.convert("RGB"))
        rgb, observed, _, geometry = restoration.prepare_crop(native)
        bounds = geometry["observed_bounds_256"]
        removal = np.zeros((256, 256), dtype=np.uint8)
        removal[148:204, 88:168] = 1

        # Already-canonical frontal portrait should bypass alignment cleanly.
        canonical_info = runtime._estimate_ffhq_alignment(rgb, removal, observed)
        self.assertFalse(canonical_info["applied"])

        # Synthetic 16-degree head roll must trigger Umeyama alignment and stay bit-exact outside mask.
        rot_mat = cv2.getRotationMatrix2D((127.5, 127.5), 16.0, 1.0)
        rolled = cv2.warpAffine(rgb, rot_mat, (256, 256), flags=cv2.INTER_LINEAR, borderMode=cv2.BORDER_REPLICATE)
        rolled_info = runtime._estimate_ffhq_alignment(rolled, removal, observed)
        self.assertTrue(rolled_info["applied"])
        self.assertGreaterEqual(abs(rolled_info["roll_deg"]), 10.0)
        self.assertGreaterEqual(rolled_info["anchors_used"], 2)

        class IdentityToy:
            def __init__(self): self.calls = 0
            def eval(self): return self
            def requires_grad_(self, _v): return self
            def __call__(self, x, *, w, adain):
                self.calls += 1
                return (torch.zeros_like(x),)

        toy_adapter = runtime.CompletionAdapter(IdentityToy())
        with patch.object(runtime, "load_frozen_cpu_completion", return_value=toy_adapter):
            res = runtime.complete(rolled, removal, prepared_bounds=bounds)
        self.assertEqual(res["metadata"]["neural_calls"], 1)
        self.assertTrue(res["metadata"]["ffhq_alignment"]["applied"])
        self.assertTrue(np.array_equal(res["output_rgb"][removal == 0], rolled[removal == 0]))
        expected_f = rolled.astype(np.float32) / np.float32(255.0)
        self.assertTrue(np.array_equal(res["completed_rgb"][removal == 0], expected_f[removal == 0]))

    def test_yunet_loader_rejects_sha256_mismatch_missing_and_cv2_error_fallback(self):
        import cv2
        import tempfile
        import torch

        old_cache = runtime._YUNET_BYTES_CACHE
        try:
            # 1) Missing YuNet file -> graceful None fallback.
            runtime._YUNET_BYTES_CACHE = None
            with patch.object(runtime.assets, "ordinary_file", side_effect=FileNotFoundError("missing")):
                self.assertIsNone(runtime._load_yunet_bytes())

            # 2) Linked / reparse-point file refused by ordinary_file -> graceful None fallback.
            runtime._YUNET_BYTES_CACHE = None
            with patch.object(
                runtime.assets,
                "ordinary_file",
                side_effect=runtime.assets.CompletionAssetError("reparse point refused"),
            ):
                self.assertIsNone(runtime._load_yunet_bytes())

            # 3) SHA-256 mismatch -> graceful None fallback.
            with tempfile.TemporaryDirectory() as tmpdir:
                tampered = Path(tmpdir) / "face_detection_yunet_2023mar.onnx"
                tampered.write_bytes(b"tampered-onnx-bytes")
                runtime._YUNET_BYTES_CACHE = None
                with patch.object(runtime.assets, "ordinary_file", return_value=tampered):
                    self.assertIsNone(runtime._load_yunet_bytes())

            # 4) Real cv2.error during YuNet inference -> clean None & completion bypasses alignment.
            runtime._YUNET_BYTES_CACHE = b"dummy-valid-bytes"
            with patch.object(cv2.FaceDetectorYN, "create", side_effect=cv2.error("simulated onnx fail")):
                self.assertIsNone(runtime._detect_yunet_landmarks_256(self.rgb))
                align_info = runtime._estimate_ffhq_alignment(self.rgb, self.mask, np.ones((256, 256), dtype=bool))
                self.assertFalse(align_info["applied"])

                class ZeroToy:
                    def eval(self): return self
                    def requires_grad_(self, _v): return self
                    def __call__(self, x, *, w, adain): return (torch.zeros_like(x),)

                toy = runtime.CompletionAdapter(ZeroToy())
                with patch.object(runtime, "load_frozen_cpu_completion", return_value=toy):
                    out = runtime.complete(self.rgb, self.mask)
                self.assertFalse(out["metadata"]["ffhq_alignment"]["applied"])
                self.assertTrue(np.array_equal(out["output_rgb"][self.mask == 0], self.rgb[self.mask == 0]))
        finally:
            runtime._YUNET_BYTES_CACHE = old_cache

    def test_nearly_hidden_face_refuses_before_loading(self):
        mask = np.ones_like(self.mask)
        with patch.object(runtime, "load_frozen_cpu_completion", side_effect=AssertionError("must not load")):
            with self.assertRaises(runtime.CompletionRefused):
                runtime.complete(self.rgb, mask)

    def test_png_rounding_and_outside_bytes_are_explicit(self):
        completed = np.full_like(self.rgb, .5, dtype=np.float32)
        final = runtime.encode_output(self.rgb, self.mask, completed)
        self.assertTrue(np.all(final[self.mask != 0] == 128))
        self.assertTrue(np.array_equal(final[self.mask == 0], self.rgb[self.mask == 0]))
        completed[0, 0, 0] = np.nan
        with self.assertRaises(ValueError):
            runtime.encode_output(self.rgb, self.mask, completed)

    def test_adapter_matches_retained_math_and_uses_only_active_batch_members(self):
        import torch
        from torch.nn import functional as F

        class PlainToy:
            def __init__(self): self.calls = 0
            def eval(self): return self
            def requires_grad_(self, _value): return self
            def __call__(self, x, *, w, adain):
                self.calls += 1
                self.seen = x.clone()
                self.w, self.adain = w, adain
                return (x * .25,)

        images = torch.from_numpy(np.stack([self.rgb, self.rgb])).permute(0, 3, 1, 2).float() / 255
        masks = torch.from_numpy(np.stack([np.zeros_like(self.mask), self.mask]))[:, None].float()
        net = PlainToy(); adapter = runtime.CompletionAdapter(net)
        events = []
        actual = adapter(images, masks, forward_event=events.append)
        active = masks.sum((1, 2, 3)) > 0
        visible = 1 - masks[active]
        support = F.interpolate(visible, size=(512, 512), mode="bilinear", align_corners=False)
        x = F.interpolate(images[active] * visible, size=(512, 512), mode="bilinear", align_corners=False)
        x = x / support.clamp_min(1e-8)
        m = F.interpolate(masks[active], size=(512, 512), mode="nearest")
        x = (x * (1 - m) + m) * 2 - 1
        generated = ((x * .25 + 1) / 2).clamp(0, 1)
        generated = F.interpolate(generated, size=(256, 256), mode="bilinear", align_corners=False)
        expected = images.clone()
        expected[active] = images[active] * (1 - masks[active]) + generated * masks[active]
        self.assertTrue(torch.equal(actual, expected))
        self.assertTrue(torch.equal(net.seen, x))
        self.assertEqual(net.calls, 1)
        self.assertEqual((net.w, net.adain), (1, False))
        self.assertEqual(events, ["entered", "returned"])
        self.assertTrue(torch.equal(actual[0], images[0]))

    def test_covered_rgb_cannot_leak_into_visible_interpolation_context(self):
        import torch
        class Capture:
            def eval(self): return self
            def requires_grad_(self, _value): return self
            def __call__(self, x, **_kwargs): self.input = x.clone(); return (torch.zeros_like(x),)
        first = self.rgb.copy(); second = self.rgb.copy()
        first[self.mask != 0] = 0; second[self.mask != 0] = 255
        seen = []
        for rgb in (first, second):
            net = Capture(); adapter = runtime.CompletionAdapter(net)
            adapter(torch.from_numpy(rgb).permute(2, 0, 1)[None].float() / 255,
                    torch.from_numpy(self.mask)[None, None].float())
            seen.append(net.input)
        self.assertTrue(torch.equal(*seen))

    def test_local_staging_preserves_wrong_existing_files(self):
        raw = b"toy safe asset"
        entry = {"filename": "toy.pth", "size": len(raw), "sha256": hashlib.sha256(raw).hexdigest()}
        with tempfile.TemporaryDirectory() as folder:
            base = Path(folder); source = base / "source"; source.write_bytes(raw)
            models = base / "models"
            with patch.object(assets, "load_manifest", return_value={"assets": {"toy": entry}}):
                installed = assets.stage_local("toy", source, model_dir=models)
                self.assertEqual(installed.read_bytes(), raw)
                installed.write_bytes(b"keep this failed asset")
                with self.assertRaises(assets.CompletionAssetError):
                    assets.stage_local("toy", source, model_dir=models)
                self.assertEqual(installed.read_bytes(), b"keep this failed asset")


class CompletionWorkerTests(unittest.TestCase):
    def prepare(self, base, full=False):
        rgb = np.full((256, 256, 3), [17, 99, 205], np.uint8)
        mask = np.full((256, 256), 255 if full else 0, np.uint8)
        Image.fromarray(rgb).save(base / "input.png")
        Image.fromarray(mask).save(base / "mask.png")
        request = {"input_path": "input.png", "mask_path": "mask.png", "output_dir": "output",
                   "prepared_bounds": [0, 0, 256, 256]}
        return request, rgb

    def test_empty_request_replays_four_fields_without_model(self):
        with tempfile.TemporaryDirectory() as folder:
            base = Path(folder); request, rgb = self.prepare(base)
            result = worker.run_request(request, request_root=base)
            self.assertTrue(result["complete"])
            self.assertEqual(result["metadata"]["model_id"], "completion_r2")
            self.assertEqual(result["metadata"]["neural_attempts"], 0)
            self.assertEqual(result["metadata"]["neural_returns"], 0)
            with Image.open(result["png_path"]) as image:
                self.assertTrue(np.array_equal(np.array(image), rgb))
            with np.load(result["stages_path"], allow_pickle=False) as data:
                self.assertEqual(set(data.files), {"input_rgb", "removal_mask", "completed_rgb", "output_rgb"})
                self.assertTrue(np.array_equal(data["output_rgb"], rgb))
            status = json.loads((base / "output/status.json").read_text())
            self.assertEqual(status["status"], "complete")
            self.assertEqual(status["component"], "completion_r2")
            with self.assertRaises(ValueError):
                worker.run_request(request, request_root=base)

    def test_cli_refusal_is_typed_separately_from_runtime_failure(self):
        with tempfile.TemporaryDirectory() as folder:
            base = Path(folder); request, _ = self.prepare(base, full=True)
            (base / "request.json").write_text(json.dumps(request), encoding="utf-8")
            result = subprocess.run([sys.executable, "-B", "-m", "implementation.completion_worker",
                                     "--request", str(base / "request.json"), "--result", str(base / "result.json")],
                                    cwd=ROOT, capture_output=True, text=True)
            self.assertEqual(result.returncode, 2, result.stderr)
            self.assertEqual(json.loads((base / "result.json").read_text())["status"], "refused")
            self.assertFalse((base / "output/completion.png").exists())
            self.assertEqual(json.loads((base / "output/status.json").read_text())["neural_attempts"], 0)

    def test_paths_cannot_escape_request_and_wrong_png_modes_refuse(self):
        with tempfile.TemporaryDirectory() as folder:
            base = Path(folder); request, rgb = self.prepare(base)
            request["input_path"] = "../outside.png"
            with self.assertRaises(ValueError):
                worker.run_request(request, request_root=base)
        with tempfile.TemporaryDirectory() as folder:
            base = Path(folder); request, rgb = self.prepare(base)
            Image.fromarray(rgb).save(base / "mask.png")
            with self.assertRaises(ValueError):
                worker.run_request(request, request_root=base)


class Phase2Landmarks106AlignmentTests(unittest.TestCase):
    def _load_face1(self):
        from implementation import app as restoration
        face_path = ROOT.parent / "test_images" / "face 1 - default.jpg"
        if not face_path.is_file():
            self.skipTest("face 1 - default.jpg fixture not present")
        with Image.open(face_path) as im:
            native = np.array(im.convert("RGB"))
        return restoration.prepare_crop(native)

    def test_import_and_empty_completion_load_no_torch_or_onnxruntime(self):
        code = (
            "import sys, numpy as np; from implementation import completion_runtime as r;"
            "x = np.full((256, 256, 3), 23, np.uint8); m = np.zeros((256, 256), np.uint8);"
            "out = r.complete(x, m); assert out['metadata']['neural_calls'] == 0;"
            "assert np.array_equal(out['output_rgb'], x);"
            "assert 'torch' not in sys.modules; assert 'onnxruntime' not in sys.modules"
        )
        result = subprocess.run([sys.executable, "-B", "-c", code], cwd=ROOT, capture_output=True, text=True)
        self.assertEqual(result.returncode, 0, result.stderr)

    def test_2d106det_5pt_alignment_without_yunet_and_bit_exact_outside_mask(self):
        import cv2
        import torch
        from implementation import automatic_covering as ac

        rgb, observed, _, geometry = self._load_face1()
        bounds = geometry["observed_bounds_256"]
        removal = np.zeros((256, 256), dtype=np.uint8)
        removal[148:204, 88:168] = 1

        pts106 = ac._run_landmarks_106(rgb)
        self.assertIsNotNone(pts106)
        self.assertEqual(pts106.shape, (106, 2))
        pts5 = ac.landmarks_5pt_from_106(pts106)
        self.assertIsNotNone(pts5)
        self.assertEqual(pts5.shape, (5, 2))
        self.assertEqual(pts5.dtype, np.float32)

        rot_mat = cv2.getRotationMatrix2D((127.5, 127.5), 16.0, 1.0)
        rolled = cv2.warpAffine(rgb, rot_mat, (256, 256), flags=cv2.INTER_LINEAR, borderMode=cv2.BORDER_REPLICATE)

        with patch.object(runtime, "_detect_yunet_landmarks_256", side_effect=AssertionError("YuNet must not run when 2d106det succeeds")):
            canonical_info = runtime._estimate_ffhq_alignment(rgb, removal, observed)
            self.assertFalse(canonical_info["applied"])

            rolled_info = runtime._estimate_ffhq_alignment(rolled, removal, observed)
            self.assertTrue(rolled_info["applied"])
            self.assertGreaterEqual(abs(rolled_info["roll_deg"]), 10.0)
            self.assertGreaterEqual(rolled_info["anchors_used"], 2)

            class ZeroToy:
                def eval(self): return self
                def requires_grad_(self, _v): return self
                def __call__(self, x, *, w, adain): return (torch.zeros_like(x),)

            toy = runtime.CompletionAdapter(ZeroToy())
            with patch.object(runtime, "load_frozen_cpu_completion", return_value=toy):
                res = runtime.complete(rolled, removal, prepared_bounds=bounds)
        self.assertEqual(res["metadata"]["neural_calls"], 1)
        self.assertTrue(res["metadata"]["ffhq_alignment"]["applied"])
        self.assertTrue(np.array_equal(res["output_rgb"][removal == 0], rolled[removal == 0]))
        expected_f = rolled.astype(np.float32) / np.float32(255.0)
        self.assertTrue(np.array_equal(res["completed_rgb"][removal == 0], expected_f[removal == 0]))

    def test_alignment_falls_back_cleanly_to_yunet_when_2d106det_unavailable(self):
        import cv2
        import torch
        from implementation import automatic_covering as ac

        rgb, observed, _, geometry = self._load_face1()
        bounds = geometry["observed_bounds_256"]
        removal = np.zeros((256, 256), dtype=np.uint8)
        removal[148:204, 88:168] = 1
        rot_mat = cv2.getRotationMatrix2D((127.5, 127.5), 16.0, 1.0)
        rolled = cv2.warpAffine(rgb, rot_mat, (256, 256), flags=cv2.INTER_LINEAR, borderMode=cv2.BORDER_REPLICATE)

        class ZeroToy:
            def eval(self): return self
            def requires_grad_(self, _v): return self
            def __call__(self, x, *, w, adain): return (torch.zeros_like(x),)

        toy = runtime.CompletionAdapter(ZeroToy())
        with patch.object(ac, "_run_landmarks_106", return_value=None), \
             patch.object(runtime, "_detect_yunet_landmarks_256", wraps=runtime._detect_yunet_landmarks_256) as yunet_spy, \
             patch.object(runtime, "load_frozen_cpu_completion", return_value=toy):
            info = runtime._estimate_ffhq_alignment(rolled, removal, observed)
            self.assertTrue(info["applied"])
            self.assertGreaterEqual(abs(info["roll_deg"]), 10.0)
            self.assertGreaterEqual(yunet_spy.call_count, 1)

            res = runtime.complete(rolled, removal, prepared_bounds=bounds)
        self.assertTrue(res["metadata"]["ffhq_alignment"]["applied"])
        self.assertTrue(np.array_equal(res["output_rgb"][removal == 0], rolled[removal == 0]))
        expected_f = rolled.astype(np.float32) / np.float32(255.0)
        self.assertTrue(np.array_equal(res["completed_rgb"][removal == 0], expected_f[removal == 0]))


class Phase3AdaptiveCompletionTests(unittest.TestCase):
    def setUp(self):
        self.rgb = np.full((256, 256, 3), (185, 145, 125), dtype=np.uint8)
        self.mask = np.zeros((256, 256), dtype=np.uint8)
        self.mask[140:200, 96:160] = 1
        self.observed = np.ones((256, 256), dtype=bool)

    def test_monochrome_guard_removes_tint_on_bw_and_preserves_color_and_outside_bytes(self):
        import cv2

        # 1. Grayscale / B&W portrait with strong pink/orange hallucinated chroma inside mask.
        bw_rgb = np.full((256, 256, 3), (128, 128, 128), dtype=np.uint8)
        bw_rgb[:, :128] = (110, 110, 110)
        bw_gen = bw_rgb.astype(np.float32) / np.float32(255.0)
        bw_gen[self.mask != 0] = np.array([215, 125, 115], dtype=np.float32) / np.float32(255.0)

        guarded = runtime._apply_monochrome_guard(bw_rgb, bw_gen, self.mask, self.observed)
        self.assertEqual(guarded.dtype, np.float32)
        orig_bw_f = bw_rgb.astype(np.float32) / np.float32(255.0)
        self.assertTrue(np.array_equal(guarded[self.mask == 0], orig_bw_f[self.mask == 0]))
        encoded = runtime.encode_output(bw_rgb, self.mask, guarded)
        self.assertTrue(np.array_equal(encoded[self.mask == 0], bw_rgb[self.mask == 0]))
        replay = np.rint(guarded * np.float32(255.0)).clip(0, 255).astype(np.uint8)
        replay[self.mask == 0] = bw_rgb[self.mask == 0]
        self.assertTrue(np.array_equal(encoded, replay))

        lab_before = cv2.cvtColor(
            np.rint(bw_gen * np.float32(255.0)).clip(0, 255).astype(np.uint8), cv2.COLOR_RGB2LAB
        ).astype(np.float32)
        lab_after = cv2.cvtColor(encoded, cv2.COLOR_RGB2LAB).astype(np.float32)
        chroma_before = float(np.hypot(lab_before[..., 1][self.mask != 0] - 128.0, lab_before[..., 2][self.mask != 0] - 128.0).mean())
        chroma_after = float(np.hypot(lab_after[..., 1][self.mask != 0] - 128.0, lab_after[..., 2][self.mask != 0] - 128.0).mean())
        self.assertGreater(chroma_before, 25.0)
        self.assertLess(chroma_after, 6.0)

        # Full _blend_inside_mask also invokes monochrome guard and stays bit-exact outside mask.
        blended_bw = runtime._blend_inside_mask(bw_rgb, bw_gen, self.mask, self.observed)
        self.assertTrue(np.array_equal(blended_bw[self.mask == 0], orig_bw_f[self.mask == 0]))
        blended_bw_u8 = runtime.encode_output(bw_rgb, self.mask, blended_bw)
        lab_blended = cv2.cvtColor(blended_bw_u8, cv2.COLOR_RGB2LAB).astype(np.float32)
        chroma_blended = float(np.hypot(lab_blended[..., 1][self.mask != 0] - 128.0, lab_blended[..., 2][self.mask != 0] - 128.0).mean())
        self.assertLess(chroma_blended, 6.0)

        # 2. Color portrait: monochrome guard must be a strict no-op!
        color_gen = self.rgb.astype(np.float32) / np.float32(255.0)
        color_gen[self.mask != 0] = np.array([200, 130, 110], dtype=np.float32) / np.float32(255.0)
        color_after = runtime._apply_monochrome_guard(self.rgb, color_gen, self.mask, self.observed)
        self.assertTrue(np.array_equal(color_after, color_gen))

    def test_max_passes_3_early_stops_at_q1_threshold_and_on_missing_evaluator(self):
        import torch

        class CountingToy:
            def __init__(self):
                self.calls = 0
            def eval(self): return self
            def requires_grad_(self, _v): return self
            def __call__(self, x, *, w, adain):
                self.calls += 1
                return (torch.zeros_like(x),)

        net = CountingToy()
        adapter = runtime.CompletionAdapter(net)
        events = []
        with patch.object(runtime, "load_frozen_cpu_completion", return_value=adapter), \
             patch.object(runtime, "_estimate_ffhq_alignment", return_value={"applied": False}), \
             patch.object(
                 runtime,
                 "_score_completion_candidate",
                 return_value=(82, {"score_0_100": 82, "factors": {"occluder_clear": 90}}, 0.71),
             ):
            res = runtime.complete(self.rgb, self.mask, forward_event=events.append, max_passes=3, q_threshold=78)

        self.assertEqual(net.calls, 1)
        self.assertEqual(res["metadata"]["neural_calls"], 1)
        self.assertEqual(events, ["entered", "returned"])
        rerank = res["metadata"]["adaptive_rerank"]
        self.assertTrue(rerank["enabled"])
        self.assertTrue(rerank["early_stopped"])
        self.assertEqual(rerank["passes"], 1)
        self.assertEqual(rerank["chosen"], "base_w1")
        self.assertEqual(rerank["pass1_Q"], 82)
        self.assertEqual(rerank["best_Q"], 82)
        self.assertTrue(np.array_equal(res["output_rgb"][self.mask == 0], self.rgb[self.mask == 0]))

        # Missing / failing evaluator degrades cleanly after Pass 1 (Q_1 is None -> early exit).
        net2 = CountingToy()
        adapter2 = runtime.CompletionAdapter(net2)
        with patch.object(runtime, "load_frozen_cpu_completion", return_value=adapter2), \
             patch.object(runtime, "_estimate_ffhq_alignment", return_value={"applied": False}), \
             patch("implementation.automatic_covering.evaluate_completion_quality", side_effect=RuntimeError("ort missing")):
            res2 = runtime.complete(self.rgb, self.mask, max_passes=3, q_threshold=78)
        self.assertEqual(net2.calls, 1)
        self.assertEqual(res2["metadata"]["neural_calls"], 1)
        self.assertTrue(res2["metadata"]["adaptive_rerank"]["early_stopped"])
        self.assertIsNone(res2["metadata"]["adaptive_rerank"]["pass1_Q"])

    def test_max_passes_3_reranks_candidates_selects_best_and_returns_winner_artifacts(self):
        import torch

        class MultiPassToy:
            def __init__(self):
                self.calls = 0
                self.values = [-0.6, 0.5, 0.0]
            def eval(self): return self
            def requires_grad_(self, _v): return self
            def __call__(self, x, *, w, adain):
                val = self.values[min(self.calls, len(self.values) - 1)]
                self.calls += 1
                return (torch.full_like(x, val),)

        # Small/medium mask (<15%) triggers Pass 1 base_w1 -> Pass 2 d3_w1 -> Pass 3 mild_sw07_d3.
        net = MultiPassToy()
        adapter = runtime.CompletionAdapter(net)
        scores = [
            (61, {"score_0_100": 61, "factors": {"occluder_clear": 70}}, 0.50),
            (75, {"score_0_100": 75, "factors": {"occluder_clear": 80}}, 0.65),
            (68, {"score_0_100": 68, "factors": {"occluder_clear": 75}}, 0.60),
        ]
        events = []
        with patch.object(runtime, "load_frozen_cpu_completion", return_value=adapter), \
             patch.object(runtime, "_estimate_ffhq_alignment", return_value={"applied": False}), \
             patch.object(runtime, "_score_completion_candidate", side_effect=scores):
            res = runtime.complete(self.rgb, self.mask, forward_event=events.append, max_passes=3, q_threshold=78)

        self.assertEqual(net.calls, 3)
        self.assertEqual(res["metadata"]["neural_calls"], 3)
        self.assertEqual(events, ["entered", "returned"] * 3)
        rerank = res["metadata"]["adaptive_rerank"]
        self.assertFalse(rerank["early_stopped"])
        self.assertEqual(rerank["passes"], 3)
        self.assertEqual(rerank["chosen"], "d3_w1")
        self.assertEqual(rerank["best_Q"], 75)
        self.assertEqual([c["config"] for c in rerank["candidates"]], ["base_w1", "d3_w1", "mild_sw07_d3"])
        # Winner raw512 must match Pass 2's output (0.5), NOT Pass 3's (0.0)!
        self.assertTrue(np.allclose(res["raw512"], 0.5, atol=1e-6))
        self.assertEqual(res["input512"].shape, (1, 3, 512, 512))
        # Outside-mask bit-exact in both uint8 & float32, and PNG replay exact.
        orig_f = self.rgb.astype(np.float32) / np.float32(255.0)
        self.assertTrue(np.array_equal(res["output_rgb"][self.mask == 0], self.rgb[self.mask == 0]))
        self.assertTrue(np.array_equal(res["completed_rgb"][self.mask == 0], orig_f[self.mask == 0]))
        self.assertTrue(np.array_equal(runtime.encode_output(self.rgb, self.mask, res["completed_rgb"]), res["output_rgb"]))

        # Large mask (>=15%) + occluder_clear < 65 triggers prog_from_c2, plus FIQA tie-break within 1 Q pt.
        big_mask = np.zeros((256, 256), dtype=np.uint8)
        big_mask[115:215, 70:186] = 1  # ~17.7%
        net_prog = MultiPassToy()
        adapter_prog = runtime.CompletionAdapter(net_prog)
        prog_scores = [
            (58, {"score_0_100": 58, "factors": {"occluder_clear": 50}}, 0.45),
            (72, {"score_0_100": 72, "factors": {"occluder_clear": 55}}, 0.40),
            (71, {"score_0_100": 71, "factors": {"occluder_clear": 80}}, 0.90),  # 71 + 5*0.9 = 75.5 > 72 + 5*0.4 = 74.0
        ]
        with patch.object(runtime, "load_frozen_cpu_completion", return_value=adapter_prog), \
             patch.object(runtime, "_estimate_ffhq_alignment", return_value={"applied": False}), \
             patch.object(runtime, "_score_completion_candidate", side_effect=prog_scores):
            res_prog = runtime.complete(self.rgb, big_mask, max_passes=3, q_threshold=78)
        self.assertEqual([c["config"] for c in res_prog["metadata"]["adaptive_rerank"]["candidates"]],
                         ["base_w1", "d3_w1", "prog_from_c2"])
        self.assertEqual(res_prog["metadata"]["adaptive_rerank"]["chosen"], "prog_from_c2")
        self.assertTrue(np.allclose(res_prog["raw512"], 0.0, atol=1e-6))
        self.assertTrue(np.array_equal(res_prog["output_rgb"][big_mask == 0], self.rgb[big_mask == 0]))

        # Large mask (>=18%) with high occluder_clear (>=65) triggers d6_w1.
        huge_mask = np.zeros((256, 256), dtype=np.uint8)
        huge_mask[105:220, 68:188] = 1  # ~21.0%
        net_d6 = MultiPassToy()
        adapter_d6 = runtime.CompletionAdapter(net_d6)
        d6_scores = [
            (60, {"score_0_100": 60, "factors": {"occluder_clear": 80}}, 0.50),
            (64, {"score_0_100": 64, "factors": {"occluder_clear": 82}}, 0.52),
            (76, {"score_0_100": 76, "factors": {"occluder_clear": 85}}, 0.60),
        ]
        with patch.object(runtime, "load_frozen_cpu_completion", return_value=adapter_d6), \
             patch.object(runtime, "_estimate_ffhq_alignment", return_value={"applied": False}), \
             patch.object(runtime, "_score_completion_candidate", side_effect=d6_scores):
            res_d6 = runtime.complete(self.rgb, huge_mask, max_passes=3, q_threshold=78)
        self.assertEqual([c["config"] for c in res_d6["metadata"]["adaptive_rerank"]["candidates"]],
                         ["base_w1", "d3_w1", "d6_w1"])
        self.assertEqual(res_d6["metadata"]["adaptive_rerank"]["chosen"], "d6_w1")

    def test_attach_r2_hooks_modifies_adain_and_sft_and_cleans_up_on_error(self):
        import torch
        from torch import nn

        class FuseToy(nn.Module):
            def forward(self, enc_feat, dec_feat):
                return dec_feat + enc_feat

        class HookableNet(nn.Module):
            def __init__(self):
                super().__init__()
                self.encoder = nn.Module()
                self.encoder.blocks = nn.ModuleList([nn.Identity()])
                self.generator = nn.Module()
                self.generator.blocks = nn.ModuleList([nn.Identity()])
                self.fuse_convs_dict = nn.ModuleDict({"32": FuseToy(), "64": FuseToy(), "128": FuseToy()})
                self.last_gen_in = None
                self.should_fail = False

            def forward(self, lq, quant, enc32, dec32):
                _ = self.encoder.blocks[-1](lq)
                g_out = self.generator.blocks[0](quant)
                self.last_gen_in = g_out.clone()
                if self.should_fail:
                    raise RuntimeError("simulated forward boom")
                fused = self.fuse_convs_dict["32"](enc32, dec32)
                return g_out, fused

        net = HookableNet().eval()
        mask512 = torch.zeros((1, 1, 512, 512), dtype=torch.float32)
        mask512[:, :, 256:448, 128:384] = 1.0  # masked region in lower half

        lq = torch.full((1, 4, 16, 16), 2.0)
        lq[:, :, :, ::2] = 4.0
        quant = torch.full((1, 4, 16, 16), -1.0)
        quant[:, :, :, ::2] = -0.5
        enc32 = torch.ones((1, 4, 32, 32))
        dec32 = torch.zeros((1, 4, 32, 32))

        base_g, base_f = net(lq, quant, enc32, dec32)
        cfg = runtime.HookConfig("test_hooks", w_in_32=0.50, w_in_64=0.70, w_in_128=0.55, adain_alpha=0.5)
        with runtime.attach_r2_hooks(net, mask512, cfg):
            hooked_g, hooked_f = net(lq, quant, enc32, dec32)

        # AdaIN moved generator input toward lq visible statistics.
        self.assertFalse(torch.allclose(base_g, hooked_g))
        self.assertGreater(float(hooked_g.mean()), float(base_g.mean()))
        # SFT attenuated residual inside mask (down near 0.5) while keeping unmasked region near 1.0.
        self.assertAlmostEqual(float(hooked_f[0, 0, 2, 2].item()), 1.0, places=3)
        self.assertAlmostEqual(float(hooked_f[0, 0, 22, 16].item()), 0.5, places=3)
        # All hooks removed after clean exit.
        self.assertEqual(len(net.encoder.blocks[-1]._forward_hooks), 0)
        self.assertEqual(len(net.generator.blocks[0]._forward_pre_hooks), 0)
        for mod in net.fuse_convs_dict.values():
            self.assertEqual(len(mod._forward_hooks), 0)

        # Hooks MUST be removed even when forward raises RuntimeError!
        net.should_fail = True
        with self.assertRaisesRegex(RuntimeError, "simulated forward boom"):
            with runtime.attach_r2_hooks(net, mask512, cfg):
                net(lq, quant, enc32, dec32)
        self.assertEqual(len(net.encoder.blocks[-1]._forward_hooks), 0)
        self.assertEqual(len(net.generator.blocks[0]._forward_pre_hooks), 0)
        for mod in net.fuse_convs_dict.values():
            self.assertEqual(len(mod._forward_hooks), 0)

        # Bare toy module without encoder/generator/fuse_convs_dict degrades cleanly.
        with runtime.attach_r2_hooks(nn.Identity(), mask512, cfg):
            pass

    def test_max_passes_validation_rejects_invalid_values(self):
        for bad in (0, 4, -1, True, False, "3", 1.5, None):
            with self.subTest(bad=bad):
                with self.assertRaises(ValueError):
                    runtime.complete(self.rgb, np.zeros_like(self.mask), max_passes=bad)

    def test_completion_worker_enforces_max_passes_bounds_and_multipass(self):
        with tempfile.TemporaryDirectory() as folder:
            base = Path(folder)
            Image.fromarray(self.rgb).save(base / "input.png")
            Image.fromarray((self.mask * 255).astype(np.uint8)).save(base / "mask.png")

            for bad in (0, 4, True, "3"):
                req_bad = {
                    "input_path": "input.png",
                    "mask_path": "mask.png",
                    "output_dir": f"out_bad_{type(bad).__name__}_{bad}",
                    "prepared_bounds": [0, 0, 256, 256],
                    "max_passes": bad,
                }
                with self.assertRaises(ValueError):
                    worker.run_request(req_bad, request_root=base)

            def fake_complete_two(rgb, mask, bounds, *, forward_event=None, max_passes=1):
                for _ in range(2):
                    forward_event("entered")
                    forward_event("returned")
                return {
                    "output_rgb": rgb.copy(),
                    "completed_rgb": rgb.astype(np.float32) / np.float32(255.0),
                    "raw512": None,
                    "input512": None,
                    "metadata": {**runtime.model_contract(), "neural_calls": 2, "prepared_bounds": bounds},
                }

            req_ok = {
                "input_path": "input.png",
                "mask_path": "mask.png",
                "output_dir": "out_ok",
                "prepared_bounds": [0, 0, 256, 256],
                "max_passes": 2,
            }
            with patch.object(runtime, "complete", side_effect=fake_complete_two):
                res = worker.run_request(req_ok, request_root=base)
            self.assertTrue(res["complete"])
            self.assertEqual(res["metadata"]["neural_attempts"], 2)
            self.assertEqual(res["metadata"]["neural_returns"], 2)
            self.assertEqual(res["metadata"]["neural_calls"], 2)

            def fake_complete_overflow(rgb, mask, bounds, *, forward_event=None, max_passes=1):
                for _ in range(3):
                    forward_event("entered")
                    forward_event("returned")
                return {}

            req_over = {
                "input_path": "input.png",
                "mask_path": "mask.png",
                "output_dir": "out_overflow",
                "prepared_bounds": [0, 0, 256, 256],
                "max_passes": 2,
            }
            with patch.object(runtime, "complete", side_effect=fake_complete_overflow):
                with self.assertRaisesRegex(ValueError, "2 R2 forwards maximum"):
                    worker.run_request(req_over, request_root=base)

    def test_import_and_empty_completion_with_max_passes_3_load_no_torch_or_onnxruntime(self):
        code = (
            "import sys, numpy as np; from implementation import completion_runtime as r;"
            "import implementation.completion_worker;"
            "x = np.full((256, 256, 3), 29, np.uint8); m = np.zeros((256, 256), np.uint8);"
            "out = r.complete(x, m, max_passes=3); assert out['metadata']['neural_calls'] == 0;"
            "assert np.array_equal(out['output_rgb'], x);"
            "assert 'torch' not in sys.modules; assert 'onnxruntime' not in sys.modules"
        )
        result = subprocess.run([sys.executable, "-B", "-c", code], cwd=ROOT, capture_output=True, text=True)
        self.assertEqual(result.returncode, 0, result.stderr)


if __name__ == "__main__":
    unittest.main()
