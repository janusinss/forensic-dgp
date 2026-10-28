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
        self.assertEqual(set(body["assets"]), {"completion_r1", "covering_detector"})
        self.assertEqual(body["assets"]["completion_r1"]["size"], 370764381)
        self.assertIsNone(body["assets"]["completion_r1"]["url"])
        with tempfile.TemporaryDirectory() as folder:
            path = Path(folder) / "manifest.json"
            body["assets"]["completion_r1"]["sha256"] = "0" * 64
            path.write_text(json.dumps(body), encoding="utf-8")
            with self.assertRaises(assets.CompletionAssetError):
                assets.load_manifest(path)

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
            self.assertEqual(result["metadata"]["neural_attempts"], 0)
            self.assertEqual(result["metadata"]["neural_returns"], 0)
            with Image.open(result["png_path"]) as image:
                self.assertTrue(np.array_equal(np.array(image), rgb))
            with np.load(result["stages_path"], allow_pickle=False) as data:
                self.assertEqual(set(data.files), {"input_rgb", "removal_mask", "completed_rgb", "output_rgb"})
                self.assertTrue(np.array_equal(data["output_rgb"], rgb))
            self.assertEqual(json.loads((base / "output/status.json").read_text())["status"], "complete")
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


if __name__ == "__main__":
    unittest.main()
