"""Unit tests for pinned offline ONNX asset manifest validation and staging."""
from __future__ import annotations

import contextlib
import hashlib
import io
import json
import os
from pathlib import Path
import tempfile
import unittest
from unittest.mock import patch

from implementation import onnx_install


def _sha256(data: bytes) -> str:
    return hashlib.sha256(data).hexdigest()


class _FakePayload:
    def __init__(self, sources: dict[tuple[str, str], Path]):
        self.models = {key: {"group": key[0], "id": key[1]} for key in sources}
        self._sources = sources

    def model_source(self, group: str, identifier: str, asset: dict) -> Path:
        path = self._sources[(group, identifier)]
        if path.stat().st_size != asset["size"] or _sha256(path.read_bytes()) != asset["sha256"]:
            raise RuntimeError("Fake payload hash mismatch")
        return path


class OnnxInstallTests(unittest.TestCase):
    def setUp(self):
        self.temp = tempfile.TemporaryDirectory()
        self.addCleanup(self.temp.cleanup)
        self.root = Path(self.temp.name)
        self.src_dir = self.root / "payload_models"
        self.src_dir.mkdir()
        self.models_root = self.root / "models"
        self.manifest_path = self.root / "onnx_assets.json"
        self.assets = {}
        self.sources = {}
        specs = [
            ("scrfd_2_5g", "detection", "det_2.5g.onnx", b"scrfd-model-bytes"),
            ("ediffiqa_tiny", "detection", "ediffiqa_tiny_jun2024.onnx", b"ediffiqa-model-bytes"),
            ("bisenet_r18_qdq", "occlusion", "bisenet_resnet18_qdq_int8.onnx", b"bisenet-model-bytes"),
            ("ofiq_occlusion_qdq", "occlusion", "ofiq_face_occlusion_qdq_int8.onnx", b"ofiq-model-bytes"),
            ("lms_2d106det", "occlusion", "2d106det.onnx", b"lms-2d106-model-bytes"),
        ]
        for key, subdir, filename, data in specs:
            src = self.src_dir / filename
            src.write_bytes(data)
            self.assets[key] = {
                "filename": filename,
                "subdir": subdir,
                "size": len(data),
                "sha256": _sha256(data),
            }
            self.sources[("onnx", key)] = src
        self.manifest_path.write_text(
            json.dumps(
                {
                    "schema_version": 1,
                    "component": "auto-mode-onnx-assets",
                    "assets": self.assets,
                },
                indent=2,
            )
            + "\n",
            encoding="utf-8",
        )
        self.payload = _FakePayload(self.sources)

    def test_real_manifest_schema_and_five_pinned_models(self):
        manifest = onnx_install.load_manifest()
        self.assertEqual(manifest["schema_version"], 1)
        self.assertEqual(manifest["component"], "auto-mode-onnx-assets")
        self.assertEqual(
            set(manifest["assets"]),
            {"scrfd_2_5g", "ediffiqa_tiny", "bisenet_r18_qdq", "ofiq_occlusion_qdq", "lms_2d106det"},
        )

    def test_manifest_rejects_bad_subdir_and_traversal_filename(self):
        bad = json.loads(self.manifest_path.read_text(encoding="utf-8"))
        bad["assets"]["scrfd_2_5g"]["subdir"] = "implementation"
        bad_path = self.root / "bad_subdir.json"
        bad_path.write_text(json.dumps(bad), encoding="utf-8")
        with self.assertRaises(onnx_install.OnnxInstallError):
            onnx_install.load_manifest(bad_path)

        bad2 = json.loads(self.manifest_path.read_text(encoding="utf-8"))
        bad2["assets"]["scrfd_2_5g"]["filename"] = "../det_2.5g.onnx"
        bad2_path = self.root / "bad_name.json"
        bad2_path.write_text(json.dumps(bad2), encoding="utf-8")
        with self.assertRaises(onnx_install.OnnxInstallError):
            onnx_install.load_manifest(bad2_path)

    def test_offline_staging_and_preservation_of_valid_files(self):
        with patch.object(onnx_install, "load_payload", return_value=self.payload), \
                contextlib.redirect_stdout(io.StringIO()):
            result = onnx_install.install_onnx_assets(
                manifest_path=self.manifest_path,
                models_root=self.models_root,
            )
        self.assertTrue(result["complete"])
        self.assertFalse(result["skipped"])
        self.assertTrue(all(item["acquisition"] == "verified_local_copy" for item in result["assets"]))
        for asset in self.assets.values():
            dest = self.models_root / asset["subdir"] / asset["filename"]
            self.assertTrue(dest.is_file())
            self.assertEqual(_sha256(dest.read_bytes()), asset["sha256"])

        # Re-running preserves already-valid files without recopying
        with patch.object(onnx_install, "load_payload", return_value=self.payload), \
                contextlib.redirect_stdout(io.StringIO()):
            second = onnx_install.install_onnx_assets(
                manifest_path=self.manifest_path,
                models_root=self.models_root,
            )
        self.assertTrue(all(item["acquisition"] == "existing_verified" for item in second["assets"]))

        # --verify-only passes when all 5 files match
        with patch.object(onnx_install, "load_payload", return_value=None):
            verified = onnx_install.install_onnx_assets(
                verify_only=True,
                manifest_path=self.manifest_path,
                models_root=self.models_root,
            )
        self.assertTrue(verified["verified_only"])
        self.assertTrue(all(item["acquisition"] == "existing_verified" for item in verified["assets"]))

    def test_corrupted_file_repaired_with_backup_and_rejected_in_verify_only(self):
        with patch.object(onnx_install, "load_payload", return_value=self.payload), \
                contextlib.redirect_stdout(io.StringIO()):
            onnx_install.install_onnx_assets(
                manifest_path=self.manifest_path,
                models_root=self.models_root,
            )
        target = self.models_root / "detection" / "det_2.5g.onnx"
        target.write_bytes(b"corrupted-bytes")

        # --verify-only strictly rejects corrupted file
        with patch.object(onnx_install, "load_payload", return_value=None):
            with self.assertRaises(onnx_install.OnnxInstallError):
                onnx_install.install_onnx_assets(
                    verify_only=True,
                    manifest_path=self.manifest_path,
                    models_root=self.models_root,
                )

        # Offline staging backs up corrupted file to .model-*.previous and replaces atomically
        with patch.object(onnx_install, "load_payload", return_value=self.payload), \
                contextlib.redirect_stdout(io.StringIO()):
            repaired = onnx_install.install_onnx_assets(
                manifest_path=self.manifest_path,
                models_root=self.models_root,
            )
        by_key = {item["key"]: item for item in repaired["assets"]}
        self.assertEqual(by_key["scrfd_2_5g"]["acquisition"], "verified_local_repair")
        self.assertEqual(target.read_bytes(), b"scrfd-model-bytes")
        backups = list((self.models_root / "detection").glob(".model-*.previous"))
        self.assertEqual(len(backups), 1)
        self.assertEqual(backups[0].read_bytes(), b"corrupted-bytes")

    def test_verify_only_rejects_missing_when_managed_and_skips_when_unbundled_dev(self):
        # Missing files with explicit models_root fail --verify-only
        with patch.object(onnx_install, "load_payload", return_value=None):
            with self.assertRaises(onnx_install.OnnxInstallError):
                onnx_install.install_onnx_assets(
                    verify_only=True,
                    manifest_path=self.manifest_path,
                    models_root=self.models_root,
                )

        # Unbundled dev checkout without payload or managed_root skips missing files cleanly
        dev_models = self.root / "dev_empty_models"
        with patch.object(onnx_install, "load_payload", return_value=None), \
                patch.object(onnx_install, "managed_root", return_value=None), \
                patch.object(onnx_install, "data_path", side_effect=lambda _root, *parts: dev_models.joinpath(*parts[1:])):
            res = onnx_install.install_onnx_assets(verify_only=True, manifest_path=self.manifest_path)
        self.assertTrue(res["skipped"])

        # Unbundled dev checkout still strictly fails if a present file is corrupted
        (dev_models / "detection").mkdir(parents=True)
        (dev_models / "detection" / "det_2.5g.onnx").write_bytes(b"bad-bytes")
        with patch.object(onnx_install, "load_payload", return_value=None), \
                patch.object(onnx_install, "managed_root", return_value=None), \
                patch.object(onnx_install, "data_path", side_effect=lambda _root, *parts: dev_models.joinpath(*parts[1:])):
            with self.assertRaises(onnx_install.OnnxInstallError):
                onnx_install.install_onnx_assets(verify_only=True, manifest_path=self.manifest_path)

    def test_online_managed_without_marker_skips_and_with_marker_enforces_strict_presence(self):
        managed_models = self.root / "managed_models"
        managed_dir = self.root / "managed_root"
        managed_dir.mkdir()
        # Online managed install (no offline payload, no marker, no ONNX files) skips cleanly
        with patch.object(onnx_install, "load_payload", return_value=None), \
                patch.object(onnx_install, "managed_root", return_value=managed_dir), \
                patch.object(onnx_install, "data_path", side_effect=lambda _root, *parts: managed_models.joinpath(*parts[1:])):
            res = onnx_install.install_onnx_assets(verify_only=True, manifest_path=self.manifest_path)
        self.assertTrue(res["skipped"])

        # Offline staging writes the bundle marker; post-install deletion is then caught strictly
        with patch.object(onnx_install, "load_payload", return_value=self.payload), \
                patch.object(onnx_install, "managed_root", return_value=managed_dir), \
                patch.object(onnx_install, "data_path", side_effect=lambda _root, *parts: managed_models.joinpath(*parts[1:])), \
                contextlib.redirect_stdout(io.StringIO()):
            onnx_install.install_onnx_assets(manifest_path=self.manifest_path)
        self.assertTrue((managed_models / "detection" / onnx_install.MARKER_NAME).is_file())

        # Deleting one ONNX file after offline install must fail --verify-only even without payload
        (managed_models / "occlusion" / "2d106det.onnx").unlink()
        with patch.object(onnx_install, "load_payload", return_value=None), \
                patch.object(onnx_install, "managed_root", return_value=managed_dir), \
                patch.object(onnx_install, "data_path", side_effect=lambda _root, *parts: managed_models.joinpath(*parts[1:])):
            with self.assertRaises(onnx_install.OnnxInstallError):
                onnx_install.install_onnx_assets(verify_only=True, manifest_path=self.manifest_path)

    def test_symlink_destination_is_refused(self):
        (self.models_root / "detection").mkdir(parents=True)
        outside = self.root / "outside.onnx"
        outside.write_bytes(b"scrfd-model-bytes")
        link_path = self.models_root / "detection" / "det_2.5g.onnx"
        try:
            link_path.symlink_to(outside)
        except OSError as exc:
            self.skipTest(f"symlink creation unavailable: {exc}")
        with patch.object(onnx_install, "load_payload", return_value=self.payload):
            with self.assertRaises(onnx_install.OnnxInstallError):
                onnx_install.install_onnx_assets(
                    manifest_path=self.manifest_path,
                    models_root=self.models_root,
                )


if __name__ == "__main__":
    unittest.main()
