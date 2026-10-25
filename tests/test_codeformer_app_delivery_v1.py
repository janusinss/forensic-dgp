"""Saved real-worker transport replay and pure delivery regressions; zero models."""
import base64
import copy
import hashlib
import io
import json
import os
from pathlib import Path
import subprocess
import sys
import unittest

import numpy as np
from PIL import Image

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
import codeformer_app_delivery_v1 as D

OLD = "run-1bc53c024a854d5280cfec1ac675c638"
NEW = "run-5377326d7ac440ec86a3a1bfe06037eb"
REPLAYS = [
    (OLD, 0, "5f3a1dffa9c9fbbb99acfcd0c26c097b20e2f8a0d7d3f96a18d57f7c410afa2e",
     "15ec9bf8361a3ccb77f461269871841c1e2e50842a84963b1cde67c530374bdc"),
    (OLD, 1, "eab86676e2ac348670df5ee2e238442675effabcec1d08fc508c7b81dc94dd69",
     "1835357f75ce31224cadb49c4c1c3f11079ebeb5078f0b19e5c7850b67280e59"),
    (NEW, 0, "5f3a1dffa9c9fbbb99acfcd0c26c097b20e2f8a0d7d3f96a18d57f7c410afa2e",
     "18db87531bf2b25cbe4cf264f7a13eec1dea6e09562ba442a892c37a94e0f4db"),
]


def png(array):
    stream = io.BytesIO()
    Image.fromarray(array).save(stream, format="PNG")
    return stream.getvalue()


def npy(array):
    stream = io.BytesIO()
    np.save(stream, array, allow_pickle=False)
    return stream.getvalue()


def replay(index=0):
    token, slot, input_sha, result_sha = REPLAYS[index]
    folder = ROOT / "scratch/protect-apply-process-v2" / token
    contents = (folder / f"input-{slot}.bin").read_bytes()
    encoded = (folder / f"result-{slot}.json").read_bytes()
    assert D.sha(contents) == input_sha and D.sha(encoded) == result_sha
    message = json.loads(encoded)
    assert message["ok"] is True
    value = {k: base64.b64decode(v, validate=True) if k != "metadata" else v
             for k, v in message["value"].items()}
    return contents, value, message["selection"], copy.deepcopy(value["metadata"]["models"])


def synthetic(shape=(64, 1600)):
    h, w = shape
    yy, xx = np.indices((h, w))
    rgb = np.stack(((xx * 7) % 256, (yy * 11) % 256, (xx + yy) % 256), axis=-1).astype(np.uint8)
    contents = png(rgb)
    canvas, support, geometry = D.prepare_native(contents)
    raw = np.full((256, 256, 3), np.float32(.414), dtype=np.float32)
    raw[~support] = canvas[~support].astype(np.float32) / np.float32(255)
    pixels = canvas.copy()
    pixels[support] = np.floor(raw[support] * np.float32(255)).astype(np.uint8)
    _, value, arm, models = replay()
    meta = value["metadata"]
    meta.update(geometry=geometry, uploaded_bytes_sha256=D.sha(contents),
                raw_RGB_sha256=D.sha(raw.tobytes()), PNG_RGB_sha256=D.sha(pixels.tobytes()))
    value.update(PNG=png(pixels), raw=npy(raw), prepared=png(canvas), support=png(support.astype(np.uint8) * 255))
    return contents, value, arm, models


class DeliveryTests(unittest.TestCase):
    def test_three_actual_saved_worker_deliveries_exact(self):
        for index in range(3):
            with self.subTest(index=index):
                args = replay(index)
                result = D.decode_worker_delivery(args[1], args[0], args[2], args[3])
                self.assertEqual(result.png, args[1]["PNG"])
                self.assertEqual(result.upload_sha256, D.sha(args[0]))
                self.assertTrue(np.array_equal(D.compose_visible(result, np.zeros((256, 256), np.uint8)), result.pixels))
                self.assertFalse(result.raw.flags.writeable)
                self.assertFalse(result.prepared.flags.writeable)

    def test_non_square_padding_and_visible_mask_composition(self):
        for shape in [(64, 1600), (1600, 64), (17, 23), (1, 1)]:
            with self.subTest(shape=shape):
                contents, value, arm, models = synthetic(shape)
                result = D.decode_worker_delivery(value, contents, arm, models)
                mask = np.zeros((256, 256), np.uint8)
                mask[100:160, 100:160] = 1
                composed = D.compose_visible(result, mask)
                ignored = ~result.observed | (mask > 0)
                self.assertTrue(np.array_equal(composed[ignored], result.prepared[ignored]))
                self.assertTrue(np.array_equal(composed[~ignored], result.pixels[~ignored]))

    def test_selection_input_and_model_binding_drift_refused(self):
        contents, value, arm, models = replay()
        for bad_value, bad_contents, bad_arm, bad_models in [
            (value, contents, D.ARMS[1], models),
            (value, contents + b"\n", arm, models),
            (value, contents, "historical_DGP", models),
            (value, contents, arm, {**models, "heads": {**models["heads"], "sha256": "0" * 64}}),
        ]:
            with self.assertRaises(ValueError):
                D.decode_worker_delivery(bad_value, bad_contents, bad_arm, bad_models)

    def test_display_or_qualification_parameter_drift_refused(self):
        contents, original, arm, models = replay()
        for key, bad in [("display_processing", "grayscale"), ("w", 0.5), ("w", 1),
                         ("model_qualified", True), ("AdaIN", 1), ("CPU_FP32", False)]:
            value = copy.deepcopy(original)
            value["metadata"][key] = bad
            with self.subTest(key=key, bad=bad), self.assertRaises(ValueError):
                D.decode_worker_delivery(value, contents, arm, models)

    def test_wrong_png_quantization_and_geometry_refused(self):
        contents, original, arm, models = replay()
        pixels = D.png_array(original["PNG"], "RGB", (256, 256, 3))
        pixels[50, 50, 0] ^= 1
        bad = copy.deepcopy(original)
        bad["PNG"] = png(pixels)
        bad["metadata"]["PNG_RGB_sha256"] = D.sha(pixels.tobytes())
        with self.assertRaisesRegex(ValueError, "floor PNG"):
            D.decode_worker_delivery(bad, contents, arm, models)
        bad = copy.deepcopy(original)
        bad["metadata"]["geometry"]["native_center_offset"] = [1, 0]
        with self.assertRaisesRegex(ValueError, "geometry"):
            D.decode_worker_delivery(bad, contents, arm, models)

    def test_raw_nan_wrong_dtype_trailing_or_large_header_refused(self):
        contents, original, arm, models = replay()
        raw = np.load(io.BytesIO(original["raw"]), allow_pickle=False)
        bad_nan = raw.copy()
        bad_nan[0, 0, 0] = np.nan
        huge = io.BytesIO()
        np.lib.format.write_array_header_1_0(huge, {"shape": (2 ** 40, 3), "fortran_order": False, "descr": "<f4"})
        for bad_bytes in [npy(bad_nan), npy(raw.astype(np.float64)), original["raw"] + b"x", huge.getvalue()]:
            value = copy.deepcopy(original)
            value["raw"] = bad_bytes
            with self.assertRaises(ValueError):
                D.decode_worker_delivery(value, contents, arm, models)

    def test_changed_padding_refused_even_with_matching_hashes(self):
        contents, value, arm, models = synthetic()
        raw = np.load(io.BytesIO(value["raw"]), allow_pickle=False)
        raw[0, 0] = 0
        value["raw"] = npy(raw)
        value["metadata"]["raw_RGB_sha256"] = D.sha(raw.tobytes())
        with self.assertRaisesRegex(ValueError, "Padding"):
            D.decode_worker_delivery(value, contents, arm, models)

    def test_invalid_completion_mask_refused(self):
        contents, value, arm, models = replay()
        result = D.decode_worker_delivery(value, contents, arm, models)
        for mask in [np.full((256, 256), 255, np.uint8), np.zeros((10, 10), np.uint8),
                     np.zeros((256, 256), np.float32)]:
            with self.assertRaises(ValueError):
                D.compose_visible(result, mask)

    def test_no_torch_import_or_app_default_mutation(self):
        environment = dict(os.environ)
        executable = sys.executable
        if os.name == "nt":
            executable = sys._base_executable
            environment["__PYVENV_LAUNCHER__"] = sys.executable
        result = subprocess.run([executable, "-B", "-c",
            "import sys; import codeformer_app_delivery_v1; "
            "assert not {'torch', 'face_workflow_web', 'app'} & set(sys.modules)"],
            cwd=ROOT, env=environment, capture_output=True, timeout=10, text=True)
        self.assertEqual(result.returncode, 0, result.stderr)


if __name__ == "__main__":
    unittest.main(verbosity=2)
