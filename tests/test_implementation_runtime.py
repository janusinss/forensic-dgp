"""Portable input and delivery contract tests; no Torch/model assets required."""
import io
import json
import math
from pathlib import Path
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
        self.assertEqual(len(runtime.adapter_shapes()), 24)
        self.assertEqual(sum(math.prod(shape) for shape in runtime.adapter_shapes().values()), 51328)
        self.assertEqual(len(runtime.head_shapes()), 36)
        self.assertEqual(sum(math.prod(shape) for shape in runtime.head_shapes().values()), 168960)
        self.assertEqual(len(runtime.v2_shapes()), 212)
        self.assertEqual(sum(math.prod(shape) for shape in runtime.v2_shapes().values()), 942272)
        self.assertEqual(runtime.model_contract("v2")["frozen_tensors"], 787)
        self.assertIn("v2_p2_u0224_heads.npz", runtime.model_contract("v2")["assets"])
        for invalid in ("u96", "u64", "successor"):
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

    def test_head_npz_requires_exact_unique_finite_arrays(self):
        arrays = {name: np.zeros(shape, np.float32) for name, shape in runtime.head_shapes().items()}
        stream = io.BytesIO()
        np.savez(stream, **arrays)
        parsed = runtime.inspect_head_npz(stream.getvalue())
        self.assertEqual(set(parsed), set(arrays))
        self.assertEqual(runtime.numpy_state_sha(parsed), runtime.numpy_state_sha(arrays))
        arrays[next(iter(arrays))].flat[0] = np.nan
        stream = io.BytesIO()
        np.savez(stream, **arrays)
        with self.assertRaises(ValueError):
            runtime.inspect_head_npz(stream.getvalue())
        stream = io.BytesIO()
        with zipfile.ZipFile(stream, "w") as archive:
            archive.writestr("../array.npy", b"bad")
        with self.assertRaises(ValueError):
            runtime.inspect_head_npz(stream.getvalue())

    def test_bad_image_failure_records_zero_forward_attempts(self):
        from implementation.model_worker import run_request
        with tempfile.TemporaryDirectory() as temporary:
            folder = Path(temporary)
            image = folder / "crop.png"
            image.write_bytes(b"invalid image")
            output = folder / "result"
            with self.assertRaises(ValueError):
                run_request({"input_path": str(image), "model": "v2", "output_dir": str(output)})
            status = json.loads((output / "status.json").read_text(encoding="utf-8"))
            self.assertEqual(status["phase"], "loading")
            self.assertEqual(status["forward_attempts"], 0)
            self.assertEqual(status["forward_returns"], 0)
            self.assertTrue(status["failed"])
            self.assertFalse((output / "raw.npy").exists())
            self.assertFalse((output / "restoration.png").exists())


if __name__ == "__main__":
    unittest.main()
