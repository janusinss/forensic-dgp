"""Preserve a confirmed zero-update failure; refuse other histories."""
import json
from pathlib import Path
import sys
import shutil
import unittest
import uuid
from unittest.mock import patch

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
sys.path.insert(0, str(ROOT / "scripts"))
from cctv_dgp_pilot import sha
from run_cctv_dgp_normfix_vm import archive_zero_update_failure
import audit_cctv_dgp_normfix_results as correction_audit


class FailurePreservation(unittest.TestCase):
    def setUp(self):
        (ROOT / "scratch").mkdir(exist_ok=True)
        self.root = ROOT / "scratch" / ("normfix-preservation-" + uuid.uuid4().hex)
        self.root.mkdir()
        self.addCleanup(self.clean_fixture)
        self.out = self.root / "outputs/cctv_dgp_pilot"
        self.out.mkdir(parents=True)
        (self.out / "partial_state.pth").write_bytes(b"fixture-zero-update-state")
        self.digest = sha(self.out / "partial_state.pth")
        self.failure = {
            "error": "Matched branch did not start at identical baseline",
            "optimizer_updates_recorded": 0, "partial_state_sha256": self.digest,
        }
        self.save_failure()
        (self.root / "pilot.log").write_text("original CUDA failure\n")

    def clean_fixture(self):
        resolved = self.root.resolve()
        if not resolved.is_relative_to((ROOT / "scratch").resolve()):
            raise ValueError("Fixture cleanup escaped scratch")
        shutil.rmtree(resolved)

    def save_failure(self):
        (self.out / "failure.json").write_text(json.dumps(self.failure))

    def archive(self):
        with patch("run_cctv_dgp_normfix_vm.FAILED_PARTIAL", self.digest):
            return archive_zero_update_failure(self.root)

    def test_preserves_exact_failure_and_original_log(self):
        before = {path.name: path.read_bytes() for path in self.out.iterdir()}
        archived = self.archive()
        self.assertFalse(self.out.exists())
        for name, contents in before.items():
            self.assertEqual((archived / name).read_bytes(), contents)
        self.assertEqual((archived / "original_pilot.log").read_bytes(),
                         (self.root / "pilot.log").read_bytes())

    def test_trained_or_other_failure_is_not_moved(self):
        for updates, error in [(1, self.failure["error"]), (0, "different failure")]:
            self.failure.update(optimizer_updates_recorded=updates, error=error)
            self.save_failure()
            with self.assertRaisesRegex(ValueError, "confirmed zero-update"):
                self.archive()
            self.assertTrue(self.out.exists())
            self.assertFalse((self.out / "original_pilot.log").exists())

    def test_recorded_training_trace_is_not_moved(self):
        (self.out / "updates.jsonl").write_text("{}\n")
        with self.assertRaisesRegex(ValueError, "confirmed zero-update"):
            self.archive()
        self.assertTrue(self.out.exists())

    def test_existing_archive_is_not_overwritten(self):
        archived = self.root / "outputs/cctv_dgp_pilot_failed_v1"
        archived.mkdir()
        (archived / "keep.txt").write_text("retain")
        with self.assertRaisesRegex(ValueError, "no existing failure archive"):
            self.archive()
        self.assertEqual((archived / "keep.txt").read_text(), "retain")
        self.assertTrue(self.out.exists())

    def test_local_audit_preserves_received_vm_receipt(self):
        received = self.out / "independent_audit.json"
        received.write_bytes(b"received VM audit evidence")
        receipt = {"complete": True, "recognizer_preview_forwards": 60}
        local = self.root / "local_independent_audit.json"
        correction_audit.save_receipt(self.out, receipt, local)
        self.assertEqual(received.read_bytes(), b"received VM audit evidence")
        self.assertEqual(json.loads(local.read_text()), receipt)

    def test_audit_refuses_overwrite_or_extra_returned_inventory(self):
        existing = self.out / "independent_audit.json"
        existing.write_bytes(b"retain")
        with self.assertRaisesRegex(ValueError, "existing audit receipt"):
            correction_audit.save_receipt(self.out, {"complete": True})
        with self.assertRaisesRegex(ValueError, "outside the returned output"):
            correction_audit.save_receipt(self.out, {"complete": True},
                                          self.out / "another_audit.json")
        self.assertEqual(existing.read_bytes(), b"retain")


if __name__ == "__main__":
    unittest.main()
