"""Unit and ASGI integration tests for the in-place application self-updater and UI chrome."""
from __future__ import annotations

import asyncio
import hashlib
from http.server import BaseHTTPRequestHandler, ThreadingHTTPServer
import json
import os
from pathlib import Path
import re
import subprocess
import tempfile
import threading
import unittest
from unittest.mock import patch
import zipfile

from implementation import app as controller
from implementation import updater
from tests.test_capture_api import asgi_request

APP_ROOT = Path(__file__).resolve().parents[1]


def _sha256_bytes(data: bytes) -> str:
    return hashlib.sha256(data).hexdigest()


class AppUpdaterTests(unittest.TestCase):
    def setUp(self):
        fixture_root = Path(os.environ.get("IMPLEMENTATION_TEST_SCRATCH", "scratch/implementation_app_tests")).resolve()
        fixture_root.mkdir(parents=True, exist_ok=True)
        self.temporary = tempfile.TemporaryDirectory(dir=fixture_root)
        self.root = Path(self.temporary.name).resolve()
        self.saved_env = {
            key: os.environ.get(key)
            for key in (
                "DGP_APP_VERSION",
                "DGP_VERSION_FILE",
                "DGP_INSTALL_ROOT",
                "DGP_UPDATE_MANIFEST_URL",
                "DGP_UPDATE_MANIFEST_FILE",
                "DGP_UPDATE_MANIFEST_file",
                "DGP_INPLACE_UPDATE_MODE",
                "DGP_UPDATE_TARGET_ROOT",
                "DGP_UPDATE_ALLOWED_HOSTS",
            )
        }
        for key in self.saved_env:
            os.environ.pop(key, None)

    def tearDown(self):
        for key, value in self.saved_env.items():
            if value is None:
                os.environ.pop(key, None)
            else:
                os.environ[key] = value
        self.temporary.cleanup()

    def test_get_app_version_precedence_and_endpoint(self):
        status, body = asyncio.run(asgi_request(controller.app, "GET", "/app/version"))
        self.assertEqual(status, 200)
        self.assertEqual(json.loads(body), {"version": updater.APP_VERSION})

        install_dir = self.root / "install"
        install_dir.mkdir()
        (install_dir / "release.json").write_text(json.dumps({"version": "0.2.13"}), encoding="utf-8")
        os.environ["DGP_INSTALL_ROOT"] = str(install_dir)
        self.assertEqual(updater.get_current_version(), "0.2.13")

        version_file = self.root / "version.json"
        version_file.write_text(json.dumps({"version": "0.2.14"}), encoding="utf-8")
        os.environ["DGP_VERSION_FILE"] = str(version_file)
        self.assertEqual(updater.get_current_version(), "0.2.14")

        os.environ["DGP_APP_VERSION"] = "0.2.15"
        status, body = asyncio.run(asgi_request(controller.app, "GET", "/app/version"))
        self.assertEqual(status, 200)
        self.assertEqual(json.loads(body), {"version": "0.2.15"})

    def test_check_update_up_to_date_and_available_via_file_manifest(self):
        package_bytes = b"dummy-installer-payload"
        package_path = self.root / "ForensicDGP_Setup-0.2.13.exe"
        package_path.write_bytes(package_bytes)

        manifest_path = self.root / "release.json"
        manifest_path.write_text(
            json.dumps({
                "schema": 1,
                "version": "0.2.12",
                "bytes": len(package_bytes),
                "sha256": _sha256_bytes(package_bytes),
                "downloadUrl": package_path.as_uri(),
            }),
            encoding="utf-8",
        )
        os.environ["DGP_UPDATE_MANIFEST_URL"] = manifest_path.as_uri()

        status, body = asyncio.run(asgi_request(controller.app, "POST", "/app/update/check"))
        self.assertEqual(status, 200)
        data = json.loads(body)
        self.assertEqual(data["status"], "up_to_date")
        self.assertFalse(data["update_available"])
        self.assertEqual(data["current_version"], "0.2.12")
        self.assertEqual(data["latest_version"], "0.2.12")
        self.assertEqual(data["message"], "Up to date")

        manifest_path.write_text(
            json.dumps({
                "schema": 1,
                "version": "0.2.13",
                "bytes": len(package_bytes),
                "sha256": _sha256_bytes(package_bytes),
                "downloadUrl": package_path.as_uri(),
            }),
            encoding="utf-8",
        )
        status, body = asyncio.run(asgi_request(controller.app, "POST", "/app/update/check"))
        self.assertEqual(status, 200)
        data = json.loads(body)
        self.assertEqual(data["status"], "update_available")
        self.assertTrue(data["update_available"])
        self.assertEqual(data["latest_version"], "0.2.13")
        self.assertEqual(data["sha256"], _sha256_bytes(package_bytes))
        self.assertEqual(data["bytes"], len(package_bytes))
        self.assertEqual(data["message"], "Update 0.2.13 available")

    def test_check_update_rejects_untrusted_download_host(self):
        manifest_path = self.root / "release.json"
        manifest_path.write_text(
            json.dumps({
                "schema": 1,
                "version": "0.2.13",
                "bytes": 100,
                "sha256": "a" * 64,
                "downloadUrl": "https://evil.example.com/ForensicDGP_Setup-0.2.13.exe",
            }),
            encoding="utf-8",
        )
        os.environ["DGP_UPDATE_MANIFEST_URL"] = manifest_path.as_uri()

        status, body = asyncio.run(asgi_request(controller.app, "POST", "/app/update/check"))
        self.assertEqual(status, 400)
        self.assertIn("Untrusted", json.loads(body)["detail"])

    def test_install_update_rejects_sha256_mismatch(self):
        package_path = self.root / "ForensicDGP_Setup-0.2.13.exe"
        package_path.write_bytes(b"tampered-payload")
        manifest_path = self.root / "release.json"
        manifest_path.write_text(
            json.dumps({
                "schema": 1,
                "version": "0.2.13",
                "bytes": len(b"tampered-payload"),
                "sha256": "0" * 64,
                "downloadUrl": package_path.as_uri(),
            }),
            encoding="utf-8",
        )
        os.environ["DGP_UPDATE_MANIFEST_URL"] = manifest_path.as_uri()

        status, body = asyncio.run(asgi_request(controller.app, "POST", "/app/update/install"))
        self.assertEqual(status, 400)
        self.assertIn("SHA-256", json.loads(body)["detail"])

    def test_install_update_reload_mode_via_loopback_http_server(self):
        zip_path = self.root / "update-0.2.13.zip"
        with zipfile.ZipFile(zip_path, "w") as zf:
            zf.writestr("VERSION.txt", "0.2.13\n")
        zip_bytes = zip_path.read_bytes()
        zip_sha = _sha256_bytes(zip_bytes)

        manifest_holder = {}

        class Handler(BaseHTTPRequestHandler):
            def log_message(self, format, *args):
                return

            def do_GET(self):
                if self.path == "/release.json":
                    payload = json.dumps(manifest_holder["manifest"]).encode("utf-8")
                    self.send_response(200)
                    self.send_header("Content-Type", "application/json")
                    self.end_headers()
                    self.wfile.write(payload)
                elif self.path == "/update-0.2.13.zip":
                    self.send_response(200)
                    self.send_header("Content-Type", "application/zip")
                    self.end_headers()
                    self.wfile.write(zip_bytes)
                else:
                    self.send_response(404)
                    self.end_headers()

        server = ThreadingHTTPServer(("127.0.0.1", 0), Handler)
        port = server.server_address[1]
        manifest_holder["manifest"] = {
            "schema": 1,
            "version": "0.2.13",
            "bytes": len(zip_bytes),
            "sha256": zip_sha,
            "downloadUrl": f"http://127.0.0.1:{port}/update-0.2.13.zip",
        }
        thread = threading.Thread(target=server.serve_forever, daemon=True)
        thread.start()
        try:
            version_file = self.root / "active_release.json"
            version_file.write_text(json.dumps({"version": "0.2.12"}), encoding="utf-8")
            target_root = self.root / "extracted"
            os.environ["DGP_VERSION_FILE"] = str(version_file)
            os.environ["DGP_UPDATE_MANIFEST_URL"] = f"http://127.0.0.1:{port}/release.json"
            os.environ["DGP_INPLACE_UPDATE_MODE"] = "reload"
            os.environ["DGP_UPDATE_TARGET_ROOT"] = str(target_root)

            with patch.object(updater, "schedule_process_restart", return_value=True) as mock_restart:
                status, body = asyncio.run(asgi_request(controller.app, "POST", "/app/update/install"))
            self.assertEqual(status, 200, body)
            data = json.loads(body)
            self.assertEqual(data["status"], "updating")
            self.assertEqual(data["target_version"], "0.2.13")
            self.assertTrue(data["restart"])
            self.assertTrue(data["restart_scheduled"])
            mock_restart.assert_called_once()
            self.assertEqual((target_root / "VERSION.txt").read_text(encoding="utf-8"), "0.2.13\n")

            status, version_body = asyncio.run(asgi_request(controller.app, "GET", "/app/version"))
            self.assertEqual(status, 200)
            self.assertEqual(json.loads(version_body), {"version": "0.2.13"})
        finally:
            server.shutdown()
            server.server_close()

    def test_install_update_desktop_mode_launches_detached_powershell_helper(self):
        install_root = self.root / "Forensic DGP"
        install_root.mkdir()
        (install_root / "ForensicDGP.exe").write_bytes(b"MZ-stub")
        (install_root / "release.json").write_text(json.dumps({"version": "0.2.12"}), encoding="utf-8")

        installer_bytes = b"MZ-new-installer-0.2.13"
        installer_src = self.root / "ForensicDGP_Setup-0.2.13.exe"
        installer_src.write_bytes(installer_bytes)

        manifest_path = self.root / "manifest.json"
        manifest_path.write_text(
            json.dumps({
                "schema": 1,
                "version": "0.2.13",
                "filename": "ForensicDGP_Setup-0.2.13.exe",
                "bytes": len(installer_bytes),
                "sha256": _sha256_bytes(installer_bytes),
                "downloadUrl": installer_src.as_uri(),
            }),
            encoding="utf-8",
        )
        os.environ["DGP_INSTALL_ROOT"] = str(install_root)
        os.environ["DGP_UPDATE_MANIFEST_FILE"] = str(manifest_path)

        captured = {}

        def fake_popen(cmd, **kwargs):
            captured["cmd"] = cmd
            captured["kwargs"] = kwargs
            script_path = Path(cmd[-1])
            captured["script_text"] = script_path.read_text(encoding="utf-8")
            script_path.unlink(missing_ok=True)
            return None

        with patch.object(updater.subprocess, "Popen", side_effect=fake_popen):
            status, body = asyncio.run(asgi_request(controller.app, "POST", "/app/update/install"))
        self.assertEqual(status, 200, body)
        data = json.loads(body)
        self.assertEqual(data["status"], "updating")
        self.assertEqual(data["target_version"], "0.2.13")
        self.assertTrue(data["restart"])
        self.assertEqual(data["mode"], "desktop_installer")

        script_text = captured["script_text"]
        self.assertIn("/VERYSILENT", script_text)
        self.assertIn("/SUPPRESSMSGBOXES", script_text)
        self.assertIn("/NORESTART", script_text)
        self.assertIn("/CLOSEAPPLICATIONS", script_text)
        self.assertIn("ForensicDGP.exe", script_text)
        self.assertIn("release.json", script_text)

        # Clean up staged installer temp file captured in script
        match = re.search(r"\$installer = '([^']+)'", script_text)
        if match:
            Path(match.group(1).replace("''", "'")).unlink(missing_ok=True)

    def test_ui_chrome_markup_styles_and_cache_versions(self):
        html = (APP_ROOT / "templates" / "implementation.html").read_text(encoding="utf-8")
        for element_id in (
            'id="app-update-prompt"',
            'id="app-update-prompt-text"',
            'id="app-install-update"',
            'id="app-menu-container"',
            'id="app-menu-button"',
            'id="app-menu-dropdown"',
            'id="app-check-update"',
            'id="app-update-status"',
        ):
            self.assertIn(element_id, html)
        self.assertIn('/static/implementation.js?v=40', html)
        self.assertIn('/static/viewport-workbench.js?v=22', html)
        css_ver = re.search(r'/static/viewport-workbench\.css\?v=(\d+)', html)
        self.assertIsNotNone(css_ver)
        self.assertGreaterEqual(int(css_ver.group(1)), 29)

        css = (APP_ROOT / "static" / "viewport-workbench.css").read_text(encoding="utf-8")
        self.assertIn(".app-update-prompt.hidden { display: none !important; }", css)
        self.assertIn(".app-menu-dropdown.hidden { display: none !important; }", css)
        self.assertIn("z-index: 1200;", css)

        js = (APP_ROOT / "static" / "implementation.js").read_text(encoding="utf-8")
        self.assertIn("/app/update/check", js)
        self.assertIn("/app/update/install", js)
        self.assertIn("/app/version", js)


if __name__ == "__main__":
    unittest.main()
