"""Portable input, V3.4 schema/parity, tamper rejection, and delivery contract tests."""
import io
import json
import math
from pathlib import Path
import shutil
import subprocess
import sys
import tempfile
import unittest
import zipfile

import numpy as np
from PIL import Image

from implementation import runtime


class PortableRuntimeTests(unittest.TestCase):
    def test_architecture_source_hash_allows_only_line_ending_conversion(self):
        source = b"class FrozenModel:\n    pass\n"
        fingerprint = runtime.architecture_source_sha256(source)
        self.assertEqual(fingerprint, runtime.architecture_source_sha256(source.replace(b"\n", b"\r\n")))
        self.assertNotEqual(fingerprint, runtime.architecture_source_sha256(source.replace(b"pass", b"return")))
        self.assertNotEqual(fingerprint, runtime.architecture_source_sha256(source + b"\n"))
        with self.assertRaises(ValueError):
            runtime.architecture_source_sha256(b"class FrozenModel:\r    pass\n")
        self.assertIn("implementation.architecture_v34", runtime.ARCHITECTURE_SOURCES)
        for binding in runtime.ARCHITECTURE_SOURCES.values():
            self.assertEqual(runtime.architecture_file_sha256(runtime.ROOT / binding["path"]), binding["sha256"])

    def test_import_constructs_no_model_or_torch(self):
        root = Path(__file__).resolve().parents[1]
        result = subprocess.run(
            [sys.executable, "-B", "-c", "import sys; import implementation.runtime; import implementation.model_worker; assert 'torch' not in sys.modules"],
            cwd=root, capture_output=True, text=True, check=False,
        )
        self.assertEqual(result.returncode, 0, result.stderr)

    def test_exact_model_schema_and_selection(self):
        shapes = runtime.v34_shapes()
        self.assertEqual(len(shapes), 344)
        self.assertEqual(sum(math.prod(shape) for shape in shapes.values()), 2241395)
        contract = runtime.model_contract("v3")
        self.assertEqual(contract["frozen_tensors"], 859)
        self.assertEqual(contract["arch_variant"], "v34")
        self.assertEqual(contract["v34_state_sha"], runtime.V34_STATE)
        self.assertEqual(set(contract["assets"]), {"codeformer.pth", "v34_precond_modules.npz"})
        for invalid in ("v2", "u96", "u64", "successor"):
            with self.assertRaises(ValueError):
                runtime.model_contract(invalid)

    def test_bounded_input_decoding(self):
        stream = io.BytesIO()
        Image.new("RGB", (25, 35), (29, 133, 225)).save(stream, format="PNG")
        decoded = runtime.decode_crop(stream.getvalue())
        self.assertEqual(decoded.shape, (35, 25, 3))
        self.assertEqual(decoded.dtype, np.uint8)
        for contents in (b"", b"bad image", b"0" * (runtime.MAX_IMAGE_BYTES + 1)):
            with self.assertRaises(ValueError):
                runtime.decode_crop(contents)
        for image, format_name in ((Image.new("RGB", (1601, 1)), "PNG"), (Image.new("RGB", (1, 1)), "GIF")):
            stream = io.BytesIO()
            image.save(stream, format=format_name)
            with self.assertRaises(ValueError):
                runtime.decode_crop(stream.getvalue())

    def test_neutral_pad_geometry_and_exact_support_delivery(self):
        rgb = np.full((6, 2, 3), [7, 73, 231], dtype=np.uint8)
        canvas, observed, removal, geometry = runtime.prepare_crop(rgb)
        self.assertEqual(geometry["native_size"], [2, 6])
        self.assertEqual(geometry["native_center_offset"], [2, 0])
        self.assertEqual(canvas.shape, (256, 256, 3))
        self.assertFalse(removal.any())
        self.assertTrue((canvas[:, :40] == 128).all())
        proposed = np.full((256, 256, 3), np.float32(0.501), dtype=np.float32)
        raw, pixels = runtime.delivered_views(proposed, canvas, observed)
        np.testing.assert_array_equal(raw[~observed], canvas[~observed].astype(np.float32) / np.float32(255))
        np.testing.assert_array_equal(pixels[~observed], canvas[~observed])
        np.testing.assert_array_equal(pixels[observed], np.floor(proposed[observed] * np.float32(255)).astype(np.uint8))
        self.assertTrue((proposed == np.float32(0.501)).all())

    def test_invalid_delivery_refused(self):
        canvas, observed, _, _ = runtime.prepare_crop(np.zeros((3, 3, 3), dtype=np.uint8))
        for raw in (np.zeros((256, 256, 3), dtype=np.float64), np.full((256, 256, 3), np.nan, dtype=np.float32),
                    np.full((256, 256, 3), 1.1, dtype=np.float32)):
            with self.assertRaises(ValueError):
                runtime.delivered_views(raw, canvas, observed)

    def test_v34_npz_requires_exact_unique_finite_arrays(self):
        arrays = {name: np.zeros(shape, np.float32) for name, shape in runtime.v34_shapes().items()}
        stream = io.BytesIO()
        np.savez(stream, **arrays)
        parsed = runtime.inspect_v34_npz(stream.getvalue())
        self.assertEqual(set(parsed), set(arrays))
        self.assertEqual(runtime.numpy_state_sha(parsed), runtime.numpy_state_sha(arrays))
        arrays[next(iter(arrays))].flat[0] = np.nan
        stream = io.BytesIO()
        np.savez(stream, **arrays)
        with self.assertRaises(ValueError):
            runtime.inspect_v34_npz(stream.getvalue())
        stream = io.BytesIO()
        with zipfile.ZipFile(stream, "w") as archive:
            archive.writestr("../array.npy", b"bad")
        with self.assertRaises(ValueError):
            runtime.inspect_v34_npz(stream.getvalue())

    def test_bad_image_failure_records_zero_forward_attempts(self):
        from implementation.model_worker import run_request
        with tempfile.TemporaryDirectory() as temporary:
            folder = Path(temporary)
            image = folder / "crop.png"
            image.write_bytes(b"invalid image")
            output = folder / "result"
            with self.assertRaises(ValueError):
                run_request({"input_path": str(image), "model": "v3", "output_dir": str(output)})
            status = json.loads((output / "status.json").read_text(encoding="utf-8"))
            self.assertEqual(status["phase"], "loading")
            self.assertEqual(status["forward_attempts"], 0)
            self.assertEqual(status["forward_returns"], 0)
            self.assertTrue(status["failed"])
            self.assertFalse((output / "raw.npy").exists())
            self.assertFalse((output / "restoration.png").exists())

    def test_v34_real_cpu_forward_tamper_rejection_and_observed_preservation(self):
        cf_path = runtime.MODEL_DIR / "codeformer.pth"
        v34_path = runtime.MODEL_DIR / "v34_precond_modules.npz"
        if not (cf_path.is_file() and v34_path.is_file()):
            self.skipTest("V3.4 model weights not staged in MODEL_DIR.")
        import torch
        from implementation.architecture_v34 import PreConditionedCodeFormerV34, load_v34_modules_npz, v34_shapes

        self.assertEqual(runtime.v34_shapes(), v34_shapes())
        net = runtime.load_frozen_cpu_restorer("v3")
        self.assertEqual(net.validate_invariants(), {"pass": True, "frozen_tensors": 859, "model": "v3"})

        # Tamper rejection: flipping one byte inside v34_precond_modules.npz must fail load.
        with tempfile.TemporaryDirectory(dir=runtime.ROOT / "scratch") as tmp_dir:
            tamper_dir = Path(tmp_dir)
            shutil.copyfile(cf_path, tamper_dir / "codeformer.pth")
            raw_v34 = bytearray(v34_path.read_bytes())
            raw_v34[-1] ^= 0xFF
            (tamper_dir / "v34_precond_modules.npz").write_bytes(bytes(raw_v34))
            with self.assertRaises(ValueError):
                runtime.load_frozen_cpu_restorer("v3", model_dir=tamper_dir)

        # Real non-square crop CPU forward + exact ~observed neutral-128 preservation + parity
        rng = np.random.default_rng(2026)
        crop = rng.integers(20, 235, size=(120, 80, 3), dtype=np.uint8)
        canvas, observed, _, _ = runtime.prepare_crop(crop)
        self.assertFalse(observed.all())
        with torch.no_grad():
            raw = net.restore_canvas(canvas, observed=observed)
            raw_delivered, pixels_delivered = runtime.delivered_views(raw, canvas, observed)
            # Independent PreConditionedCodeFormerV34 parity check
            ref = PreConditionedCodeFormerV34().eval()
            ref.base.load_state_dict(torch.load(cf_path, map_location="cpu", weights_only=True)["params_ema"], strict=True)
            load_v34_modules_npz(v34_path, ref)
            inp_t = torch.from_numpy(canvas.transpose(2, 0, 1).copy()).to(dtype=torch.float32)[None] / 255.0
            obs_t = torch.from_numpy(observed.astype(np.float32))[None, None]
            ref_raw = ref(inp_t, obs_t)["y_256"][0].detach().cpu().numpy().transpose(1, 2, 0).astype(np.float32)

        self.assertEqual(net.last_base_calls, 1)
        self.assertEqual(net.last_telemetry["arch_variant"], "v34")
        self.assertEqual(net.last_telemetry["v34_state_sha"], runtime.V34_STATE)
        for key in ("w_pred", "g_photo", "g_pre", "alpha_clean"):
            self.assertTrue(math.isfinite(net.last_telemetry[key]))
        np.testing.assert_array_equal(raw_delivered[~observed], canvas[~observed].astype(np.float32) / np.float32(255.0))
        np.testing.assert_array_equal(pixels_delivered[~observed], canvas[~observed])
        self.assertLessEqual(float(np.max(np.abs(raw - ref_raw))), 1e-6)


if __name__ == "__main__":
    unittest.main()
