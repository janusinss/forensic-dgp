"""Bounded completion distribution tests using tiny files and fake HTTPS only."""
import contextlib
import hashlib
import io
import json
import os
from pathlib import Path
import subprocess
import sys
import tempfile
import unittest
from unittest.mock import patch

from implementation import completion_install as install


class Response(io.BytesIO):
    def __init__(self, body, *, declared=None, url="https://release-assets.githubusercontent.com/fixture", status=200, encoding=None):
        super().__init__(body)
        self.headers = {}
        if declared is not None:
            self.headers["Content-Length"] = str(declared)
        if encoding is not None:
            self.headers["Content-Encoding"] = encoding
        self.status, self.url = status, url

    def geturl(self):
        return self.url


class Opener:
    def __init__(self, responses):
        self.responses, self.calls = list(responses), []

    def open(self, request, timeout):
        self.calls.append((request.full_url, timeout))
        response = self.responses.pop(0)
        if isinstance(response, Exception):
            raise response
        return response


class CompletionInstallTests(unittest.TestCase):
    def setUp(self):
        self.temporary = tempfile.TemporaryDirectory()
        self.addCleanup(self.temporary.cleanup)
        self.root = Path(self.temporary.name)
        self.destination = self.root / "models"
        self.payloads = {"completion_r1": b"fixed R1 fixture weights", "covering_detector": b"fixed detector fixture weights"}
        self.manifest = {"assets": {key: {"filename": filename, "size": len(self.payloads[key]),
                          "sha256": hashlib.sha256(self.payloads[key]).hexdigest()}
                          for key, filename in (("completion_r1", "codeformer_completion_r1.pth"),
                                                ("covering_detector", "covering_detector_segmenter_v1.pth"))}}
        self.manifest_path = self.root / "manifest.json"
        self.manifest_path.write_text(json.dumps(self.manifest), encoding="utf-8")
        self.load_patch = patch.object(install.assets, "load_manifest", return_value=self.manifest)
        self.load_patch.start()
        self.addCleanup(self.load_patch.stop)
        self.path_patch = patch.object(install.assets, "MANIFEST_PATH", self.manifest_path)
        self.path_patch.start()
        self.addCleanup(self.path_patch.stop)

    def files(self):
        folder = self.root / "local"
        folder.mkdir()
        result = {}
        for key, body in self.payloads.items():
            path = folder / self.manifest["assets"][key]["filename"]
            path.write_bytes(body)
            result[key] = path
        return result

    def installed(self):
        self.destination.mkdir()
        for key, asset in self.manifest["assets"].items():
            (self.destination / asset["filename"]).write_bytes(self.payloads[key])

    def setup(self, **kwargs):
        with contextlib.redirect_stdout(io.StringIO()):
            return install.install_assets(model_dir=self.destination, **kwargs)

    def test_pure_standard_library_import(self):
        code = "import sys; import implementation.completion_install; assert 'torch' not in sys.modules; assert 'numpy' not in sys.modules; assert 'PIL' not in sys.modules"
        result = subprocess.run([sys.executable, "-B", "-c", code], cwd=Path(__file__).resolve().parents[1], capture_output=True, text=True)
        self.assertEqual(result.returncode, 0, result.stderr)

    def test_fixed_release_names_and_immutable_runtime_manifest_urls(self):
        self.assertEqual(install.RELEASE_BASE, "https://github.com/janusinss/forensic-dgp/releases/download/completion-r1-models-v1/")
        real = json.loads((Path(__file__).resolve().parents[1] / "implementation/completion_assets.json").read_text())
        self.assertTrue(all(row["url"] is None for row in real["assets"].values()))

    def test_verified_offline_copies_are_independent_and_record_distribution(self):
        sources = self.files()
        result = self.setup(local_r1=sources["completion_r1"], local_detector=sources["covering_detector"], opener=Opener([]))
        self.assertTrue(result["complete"])
        self.assertEqual(result["inference_calls"], 0)
        self.assertFalse(result["restoration_files_modified"])
        record = json.loads((self.destination / result["installation_receipt"]).read_text())
        self.assertEqual(record["completion_manifest_sha256"], hashlib.sha256(self.manifest_path.read_bytes()).hexdigest())
        for key, asset in self.manifest["assets"].items():
            path = self.destination / asset["filename"]
            self.assertEqual(path.read_bytes(), self.payloads[key])
            self.assertEqual(path.stat().st_nlink, 1)
            self.assertFalse(path.samefile(sources[key]))
        self.assertFalse(list(self.destination.glob("*.partial")))

    def test_verify_only_never_writes_or_downloads(self):
        self.installed()
        before = {p.name: (p.read_bytes(), p.stat().st_mtime_ns) for p in self.destination.iterdir()}
        opener = Opener([])
        result = self.setup(verify_only=True, opener=opener)
        self.assertTrue(result["verified_only"])
        self.assertNotIn("installation_receipt", result)
        self.assertEqual(opener.calls, [])
        self.assertEqual(before, {p.name: (p.read_bytes(), p.stat().st_mtime_ns) for p in self.destination.iterdir()})

    def test_verify_missing_creates_no_directory(self):
        with self.assertRaises(install.CompletionInstallError):
            self.setup(verify_only=True)
        self.assertFalse(self.destination.exists())

    def test_existing_mismatch_refuses_before_any_other_download(self):
        self.destination.mkdir()
        bad = self.destination / self.manifest["assets"]["covering_detector"]["filename"]
        bad.write_bytes(b"keep original mismatched detector")
        opener = Opener([])
        with self.assertRaises(install.CompletionInstallError):
            self.setup(opener=opener)
        self.assertEqual(bad.read_bytes(), b"keep original mismatched detector")
        self.assertEqual(opener.calls, [])
        self.assertEqual(len(list(self.destination.iterdir())), 1)

    def test_invalid_local_source_refuses_before_any_copy(self):
        sources = self.files()
        sources["covering_detector"].write_bytes(b"wrong")
        with self.assertRaises(install.CompletionInstallError):
            self.setup(local_r1=sources["completion_r1"], local_detector=sources["covering_detector"])
        self.assertFalse(self.destination.exists())

    def test_local_mode_does_not_silently_download_missing_peer(self):
        sources = self.files()
        opener = Opener([])
        with self.assertRaises(install.CompletionInstallError):
            self.setup(local_r1=sources["completion_r1"], opener=opener)
        self.assertEqual(opener.calls, [])
        self.assertFalse(self.destination.exists())

    def test_valid_https_downloads_are_exact_and_no_retries(self):
        opener = Opener([Response(self.payloads[key], declared=len(self.payloads[key])) for key in self.payloads])
        result = self.setup(opener=opener)
        self.assertEqual(len(opener.calls), 2)
        self.assertTrue(all(url.startswith(install.RELEASE_BASE) and timeout == 60 for url, timeout in opener.calls))
        self.assertTrue(all(row["acquisition"] == "verified_https_download" for row in result["assets"]))
        for key, asset in self.manifest["assets"].items():
            self.assertEqual((self.destination / asset["filename"]).read_bytes(), self.payloads[key])

    def test_hash_mismatch_keeps_bad_partial_and_no_final_model(self):
        wrong = b"x" * len(self.payloads["completion_r1"])
        opener = Opener([Response(wrong)])
        with self.assertRaises(install.CompletionInstallError):
            self.setup(opener=opener)
        partials = list(self.destination.glob(".*.partial"))
        self.assertEqual(len(partials), 1)
        self.assertEqual(partials[0].read_bytes(), wrong)
        self.assertFalse((self.destination / "codeformer_completion_r1.pth").exists())
        self.assertEqual(len(opener.calls), 1)

    def test_download_overrun_is_bounded_and_partial_retained(self):
        opener = Opener([Response(self.payloads["completion_r1"] + b"extra")])
        with self.assertRaises(install.CompletionInstallError):
            self.setup(opener=opener)
        partial = next(self.destination.glob(".*.partial"))
        self.assertLessEqual(partial.stat().st_size, len(self.payloads["completion_r1"]))
        self.assertEqual(len(opener.calls), 1)

    def test_truncated_download_keeps_partial(self):
        opener = Opener([Response(self.payloads["completion_r1"][:-1])])
        with self.assertRaises(install.CompletionInstallError):
            self.setup(opener=opener)
        self.assertEqual(next(self.destination.glob(".*.partial")).read_bytes(), self.payloads["completion_r1"][:-1])

    def test_invalid_headers_and_non_https_final_response_fail_closed(self):
        for response in (Response(b"", declared=1), Response(b"", declared="bad"),
                         Response(b"", url="http://example.test/unsafe"), Response(b"", status=206),
                         Response(b"", encoding="gzip")):
            with self.subTest(response=response.headers, url=response.url):
                folder = self.root / uuid_name()
                with self.assertRaises(install.CompletionInstallError):
                    install.install_assets(model_dir=folder, opener=Opener([response]))
                self.assertEqual(len(list(folder.glob(".*.partial"))), 1)

    def test_redirect_handler_rejects_protocol_downgrade(self):
        handler = install.HTTPSRedirectHandler()
        request = install.urllib.request.Request(install.RELEASE_BASE + "fixture.pth")
        with self.assertRaises(install.CompletionInstallError):
            handler.redirect_request(request, None, 302, "Found", {}, "http://example.test/model")
        with self.assertRaises(install.CompletionInstallError):
            install._https("https://user:secret@example.test/model")

    def test_download_total_time_stops_before_read_and_retains_partial(self):
        ticks = iter((0, 1801))
        with self.assertRaises(install.CompletionInstallError):
            self.setup(opener=Opener([Response(self.payloads["completion_r1"])]), clock=lambda: next(ticks))
        self.assertEqual(next(self.destination.glob(".*.partial")).stat().st_size, 0)

    def test_atomic_publication_never_replaces_racing_existing_file(self):
        self.destination.mkdir()
        asset = self.manifest["assets"]["completion_r1"]
        partial = self.destination / ".prepared.partial"
        partial.write_bytes(self.payloads["completion_r1"])
        target = self.destination / asset["filename"]
        target.write_bytes(b"racing existing model must remain")
        with self.assertRaises(FileExistsError):
            install._publish(partial, target, asset)
        self.assertEqual(target.read_bytes(), b"racing existing model must remain")
        self.assertEqual(partial.read_bytes(), self.payloads["completion_r1"])

    def test_hardlinked_source_or_installed_model_is_refused(self):
        sources = self.files()
        alias = self.root / "aliased.pth"
        try:
            os.link(sources["completion_r1"], alias)
        except OSError as exc:
            self.skipTest(str(exc))
        with self.assertRaises(install.CompletionInstallError):
            self.setup(local_r1=sources["completion_r1"], local_detector=sources["covering_detector"])
        self.assertFalse(self.destination.exists())

    def test_linked_destination_directory_is_refused(self):
        real = self.root / "real-models"
        real.mkdir()
        try:
            self.destination.symlink_to(real, target_is_directory=True)
        except OSError as exc:
            self.skipTest(str(exc))
        with self.assertRaises(install.CompletionInstallError):
            self.setup(opener=Opener([]))
        self.assertEqual(list(real.iterdir()), [])

    def test_verify_only_cannot_stage_local_files(self):
        with self.assertRaises(install.CompletionInstallError):
            self.setup(verify_only=True, local_r1=self.root / "unused")

    def test_default_cli_bootstraps_only_validated_configured_python(self):
        configured = Path(sys.executable)
        with patch("implementation.environment.configured_python", return_value=configured) as config, \
                patch("implementation.environment.check_python") as check, \
                patch("implementation.environment.run_foreground", return_value=0) as run, \
                patch.object(install, "install_assets") as acquire:
            self.assertEqual(install.main([]), 0)
        config.assert_called_once_with()
        check.assert_called_once_with(configured)
        self.assertEqual(run.call_args.args[0], [str(configured), "-B", "-m", "implementation.completion_install", "--configured-child"])
        acquire.assert_not_called()

    def test_verify_cli_has_no_environment_bootstrap_or_install_changes(self):
        with patch("implementation.environment.configured_python", side_effect=AssertionError("no environment bootstrap")), \
                patch.object(install, "install_assets", return_value={"complete": True, "verified_only": True}) as verify, \
                contextlib.redirect_stdout(io.StringIO()):
            self.assertEqual(install.main(["--verify-only"]), 0)
        verify.assert_called_once_with(local_r1=None, local_detector=None, verify_only=True)


def uuid_name():
    import uuid
    return uuid.uuid4().hex


if __name__ == "__main__":
    unittest.main()
