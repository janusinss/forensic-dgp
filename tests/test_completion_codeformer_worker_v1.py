"""Request/PNG preservation checks with no neural invocation or checkpoint load."""
import importlib.util
import json
from pathlib import Path
import subprocess
import sys
import tempfile
from types import SimpleNamespace
import unittest
from unittest.mock import patch

import numpy as np
from PIL import Image

WORKER = Path(__file__).resolve().parents[1] / "scripts/completion_codeformer_worker_v1.py"
SPEC = importlib.util.spec_from_file_location("completion_worker_contract", WORKER)
MODULE = importlib.util.module_from_spec(SPEC)
SPEC.loader.exec_module(MODULE)
TEMP_ROOT = Path(r"C:\xampp\htdocs\YEAR 4\Testing\outputs")


def prepared(path, mask=None):
    yy, xx = np.mgrid[:256, :256]
    rgb = np.stack([xx, yy, (xx + yy) // 2], axis=-1).astype(np.uint8)
    Image.fromarray(rgb).save(path / "input.png")
    Image.fromarray(np.zeros((256, 256), np.uint8) if mask is None else mask).save(path / "mask.png")
    return rgb


class CompletionWorkerContract(unittest.TestCase):
    def test_empty_bypass_has_no_model_or_checkpoint_dependency(self):
        with tempfile.TemporaryDirectory(dir=TEMP_ROOT) as temporary:
            root = Path(temporary)
            rgb = prepared(root)
            with patch.object(MODULE, "complete_nonempty", side_effect=AssertionError("No neural call permitted")):
                receipt = MODULE.run_request(root, root / "nonexistent.pth")
            self.assertTrue(receipt["complete"])
            self.assertEqual(receipt["neural_calls"], 0)
            self.assertIsNone(receipt["runtime"]["checkpoint_sha256"])
            np.testing.assert_array_equal(np.array(Image.open(root / "result.png")), rgb)
            with np.load(root / "stages.npz", allow_pickle=False) as stages:
                self.assertEqual(set(stages.files), {"input_rgb", "removal_mask", "completed_rgb", "output_rgb"})
                np.testing.assert_array_equal(stages["output_rgb"], rgb)
            self.assertEqual(receipt["outside_source_bytes"], 256 * 256 * 3)
            self.assertEqual(receipt["outside_source_rgb_sha256"], receipt["outside_output_rgb_sha256"])
            self.assertEqual(receipt["outputs_sha256"]["result.png"], MODULE.sha(root / "result.png"))

    def test_cli_empty_request_does_not_import_torch(self):
        with tempfile.TemporaryDirectory(dir=TEMP_ROOT) as temporary:
            root = Path(temporary)
            prepared(root)
            code = "import importlib.util,sys; from pathlib import Path; s=importlib.util.spec_from_file_location('w',sys.argv[1]); m=importlib.util.module_from_spec(s); s.loader.exec_module(m); m.run_request(Path(sys.argv[2]),Path(sys.argv[2])/'missing'); assert 'torch' not in sys.modules; print('NO_TORCH_PASS')"
            result = subprocess.run([sys.executable, "-B", "-c", code, str(WORKER), str(root)], capture_output=True, text=True, timeout=20)
            self.assertEqual(result.returncode, 0, result.stderr)
            self.assertIn("NO_TORCH_PASS", result.stdout)

    def test_nonbinary_wrong_size_and_excessive_masks_rejected(self):
        with tempfile.TemporaryDirectory(dir=TEMP_ROOT) as temporary:
            root = Path(temporary)
            mask = np.zeros((256, 256), np.uint8)
            mask[0, 0] = 127
            prepared(root, mask)
            with self.assertRaisesRegex(ValueError, "binary"):
                MODULE.load_request(root)
            Image.new("L", (128, 128)).save(root / "mask.png")
            with self.assertRaisesRegex(ValueError, "prepared"):
                MODULE.load_request(root)
            Image.new("L", (256, 256), 255).save(root / "mask.png")
            with self.assertRaisesRegex(ValueError, "visible"):
                MODULE.load_request(root)

    def test_rounding_and_source_copy_match_existing_policy(self):
        rgb = np.full((256, 256, 3), 17, np.uint8)
        mask = np.zeros((256, 256), np.uint8)
        mask[0, 0] = 1
        completed = np.full((256, 256, 3), .5, np.float32)
        final = MODULE.encode_output(rgb, mask, completed)
        np.testing.assert_array_equal(final[0, 0], [128, 128, 128])
        np.testing.assert_array_equal(final[mask == 0], rgb[mask == 0])
        completed[0, 0, 0] = np.nan
        with self.assertRaises(FloatingPointError):
            MODULE.encode_output(rgb, mask, completed)

    def test_fresh_request_required(self):
        with tempfile.TemporaryDirectory(dir=TEMP_ROOT) as temporary:
            root = Path(temporary)
            prepared(root)
            MODULE.run_request(root, root / "missing")
            before = (root / "receipt.json").read_bytes()
            with self.assertRaisesRegex(ValueError, "fresh"):
                MODULE.run_request(root, root / "missing")
            self.assertEqual((root / "receipt.json").read_bytes(), before)

    def test_padding_is_unavailable_and_never_removable(self):
        with tempfile.TemporaryDirectory(dir=TEMP_ROOT) as temporary:
            root = Path(temporary)
            prepared(root)
            (root / "request.json").write_text(json.dumps({"prepared_bounds": [0, 125, 256, 131]}))
            with self.assertRaisesRegex(ValueError, "hidden"):
                MODULE.load_request(root)
            (root / "request.json").write_text(json.dumps({"prepared_bounds": [16, 0, 240, 256]}))
            mask = np.zeros((256, 256), np.uint8)
            mask[12, 12] = 255
            Image.fromarray(mask).save(root / "mask.png")
            with self.assertRaisesRegex(ValueError, "padding"):
                MODULE.load_request(root)

    def test_wrong_preloaded_dependency_rejected_without_model_import(self):
        with patch.dict(sys.modules, {"completion": SimpleNamespace(__file__="C:/wrong-checkout/completion.py")}):
            with self.assertRaisesRegex(RuntimeError, "noncanonical"):
                MODULE.verify_loaded_sources()


if __name__ == "__main__":
    unittest.main()
