"""Installer concurrency and package-source regressions; no network or real venv changes."""
import contextlib
import io
import os
from pathlib import Path
import subprocess
import sys
import tempfile
import threading
import unittest
from unittest.mock import patch

from implementation import completion_assets, completion_install, environment


class DependencyInstallationTests(unittest.TestCase):
    python = Path("C:/DGP/test-environment/Scripts/python.exe")

    def probe(self, code):
        return patch.object(environment.subprocess, "run", return_value=subprocess.CompletedProcess([], code))

    def assert_uv_install(self, command):
        self.assertEqual(command[0], str(self.python))
        self.assertIn("uv", command)
        self.assertIn("install", command)
        for flag, value in (("--torch-backend", "cpu"), ("--python", str(self.python)),
                            ("--default-index", "https://pypi.org/simple"),
                            ("--only-binary", ":all:")):
            self.assertIn(flag, command)
            self.assertEqual(command[command.index(flag) + 1], value)
        for flag in ("--strict", "--no-config", "--no-python-downloads"):
            self.assertIn(flag, command)
        self.assertEqual(Path(command[command.index("-r") + 1]), environment.ROOT / "requirements.txt")
        self.assertNotIn("--upgrade", command)
        if "--link-mode" in command:
            self.assertNotEqual(command[command.index("--link-mode") + 1], "copy",
                                "allow uv to reuse wheel files instead of forcing full copies")

    def assert_uv_check(self, command):
        self.assertEqual(command[0], str(self.python))
        self.assertIn("uv", command)
        self.assertIn("check", command)
        self.assertNotIn("install", command)
        self.assertEqual(command[command.index("--python") + 1], str(self.python))

    def test_existing_pinned_uv_rerun_uses_one_install_and_check_without_pip_bootstrap(self):
        with self.probe(0) as probe, patch.object(environment, "run_foreground", return_value=0) as run:
            self.assertEqual(environment.install_dependencies(self.python), 0)
        self.assertEqual(run.call_count, 2)
        self.assert_uv_install(run.call_args_list[0].args[0])
        self.assert_uv_check(run.call_args_list[1].args[0])
        command = probe.call_args.args[0]
        self.assertEqual(command[0], str(self.python))
        self.assertIn("-I", command)
        self.assertTrue(probe.call_args.kwargs.get("timeout"))

    def test_missing_or_changed_uv_bootstraps_pinned_tool_once(self):
        with self.probe(1), patch.object(environment, "run_foreground", return_value=0) as run:
            self.assertEqual(environment.install_dependencies(self.python), 0)
        self.assertEqual(run.call_count, 3)
        bootstrap, installation, check = [call.args[0] for call in run.call_args_list]
        self.assertEqual(bootstrap[0], str(self.python))
        self.assertIn("pip", bootstrap)
        self.assertIn("uv==0.13.1", bootstrap)
        self.assertNotIn("--upgrade", bootstrap)
        self.assertIn("https://pypi.org/simple", bootstrap)
        self.assert_uv_install(installation)
        self.assert_uv_check(check)

    def test_bootstrap_failure_stops_before_dependency_install(self):
        with self.probe(1), patch.object(environment, "run_foreground", return_value=17) as run:
            self.assertEqual(environment.install_dependencies(self.python), 17)
        self.assertEqual(run.call_count, 1)

    def test_dependency_failure_is_not_hidden_by_fallback(self):
        with self.probe(0), patch.object(environment, "run_foreground", return_value=23) as run:
            self.assertEqual(environment.install_dependencies(self.python), 23)
        self.assertEqual(run.call_count, 1)

    def test_dependency_integrity_check_failure_is_not_reported_as_success(self):
        with self.probe(0), patch.object(environment, "run_foreground", side_effect=(0, 29)) as run:
            self.assertEqual(environment.install_dependencies(self.python), 29)
        self.assertEqual(run.call_count, 2)
        self.assert_uv_check(run.call_args_list[1].args[0])

    def test_package_commands_ignore_inherited_index_and_insecure_settings(self):
        poisoned = {
            "UV_INDEX": "https://untrusted.invalid/simple", "UV_EXTRA_INDEX_URL": "https://untrusted.invalid/extra",
            "UV_FIND_LINKS": "C:/untrusted/wheels", "UV_INSECURE_HOST": "untrusted.invalid",
            "UV_TORCH_BACKEND": "cu130", "UV_NO_DEPS": "true",
            "PIP_EXTRA_INDEX_URL": "https://untrusted.invalid/extra", "PIP_INDEX_URL": "https://untrusted.invalid/simple",
            "PIP_CONFIG_FILE": "C:/untrusted/pip.ini", "UV_CACHE_DIR": "C:/trusted/uv-cache",
            "UV_SYSTEM_CERTS": "true", "HTTPS_PROXY": "http://proxy.example.test:8080",
            "SSL_CERT_FILE": "C:/trusted/certificate.pem",
        }
        with patch.dict(environment.os.environ, poisoned), self.probe(1), \
                patch.object(environment, "run_foreground", return_value=0) as run:
            self.assertEqual(environment.install_dependencies(self.python), 0)
        self.assertEqual(run.call_count, 3)
        for call in run.call_args_list:
            with self.subTest(command=call.args[0]):
                cleaned = call.kwargs["env"]
                for key in cleaned:
                    if key.startswith("UV_"):
                        self.assertIn(key, {"UV_CACHE_DIR", "UV_SYSTEM_CERTS"})
                    if key.startswith("PIP_"):
                        self.assertEqual(key, "PIP_CONFIG_FILE")
                self.assertEqual(cleaned["PIP_CONFIG_FILE"], os.devnull)
                for key in ("UV_CACHE_DIR", "UV_SYSTEM_CERTS", "HTTPS_PROXY", "SSL_CERT_FILE"):
                    self.assertEqual(cleaned[key], poisoned[key])

    def test_probe_timeout_surfaces_without_installing(self):
        with patch.object(environment.subprocess, "run", side_effect=subprocess.TimeoutExpired("python", 15)), \
                patch.object(environment, "run_foreground") as run:
            with self.assertRaises((environment.EnvironmentError, subprocess.TimeoutExpired)):
                environment.install_dependencies(self.python)
        run.assert_not_called()

    def test_interrupted_probe_preserves_exit_status_without_bootstrap(self):
        for status in (130, -1073741510):
            with self.subTest(status=status), self.probe(status), \
                    patch.object(environment, "run_foreground") as run:
                self.assertEqual(environment.install_dependencies(self.python), status)
            run.assert_not_called()


class ConcurrentModelInstallationTests(unittest.TestCase):
    def test_model_families_start_without_waiting_for_other_downloads(self):
        restoration_started = threading.Event()
        completion_started = threading.Event()

        def run(command):
            self.assertEqual(command[0], sys.executable)
            self.assertIn("-B", command)
            if "implementation.setup_models" in command:
                restoration_started.set()
                self.assertTrue(completion_started.wait(3), "completion must start before restoration finishes")
            else:
                self.assertIn("implementation.environment", command)
                self.assertEqual(command[-1], "completion-assets")
                completion_started.set()
                self.assertTrue(restoration_started.wait(3), "restoration must start before completion finishes")
            return 0

        with patch.object(environment, "run_foreground", side_effect=run), \
                patch.object(environment, "configured_python", side_effect=AssertionError("no configured env needed")), \
                contextlib.redirect_stdout(io.StringIO()):
            self.assertEqual(environment.install_models(), 0)

    def test_model_failure_waits_for_peer_without_reporting_success(self):
        completion_finished = threading.Event()

        def run(command):
            if "implementation.setup_models" in command:
                return 19
            completion_finished.set()
            return 0

        with patch.object(environment, "run_foreground", side_effect=run), \
                contextlib.redirect_stdout(io.StringIO()):
            try:
                result = environment.install_models()
            except environment.EnvironmentError:
                result = 1
        self.assertNotEqual(result, 0)
        self.assertTrue(completion_finished.is_set(), "model peer must finish before installer returns")

    def test_completion_exception_surfaces_after_restoration_finishes(self):
        restoration_finished = threading.Event()

        def run(command):
            if "implementation.setup_models" in command:
                restoration_finished.set()
                return 0
            raise OSError("fixture failure")

        with patch.object(environment, "run_foreground", side_effect=run), \
                contextlib.redirect_stdout(io.StringIO()), contextlib.redirect_stderr(io.StringIO()):
            try:
                result = environment.install_models()
            except (environment.EnvironmentError, OSError):
                result = 1
        self.assertNotEqual(result, 0)
        self.assertTrue(restoration_finished.is_set())


class SeededCacheSelectionTests(unittest.TestCase):
    def setUp(self):
        self.temporary = tempfile.TemporaryDirectory()
        self.addCleanup(self.temporary.cleanup)
        self.root = Path(self.temporary.name)
        self.seeded = self.root / "implementation" / "uv_cache"

    def package_environment(self, inherited):
        with patch.object(environment, "ROOT", self.root), \
                patch.dict(environment.os.environ, inherited, clear=True), \
                patch.object(environment.subprocess, "run", return_value=subprocess.CompletedProcess([], 0)), \
                patch.object(environment, "run_foreground", return_value=0) as run:
            self.assertEqual(environment.install_dependencies(Path(sys.executable)), 0)
        self.assertEqual(run.call_count, 2)
        self.assertIn("install", run.call_args_list[0].args[0])
        self.assertIn("check", run.call_args_list[1].args[0])
        return [call.kwargs["env"] for call in run.call_args_list]

    def test_bundled_cache_is_selected_when_present_and_no_explicit_override(self):
        self.seeded.mkdir(parents=True)
        for selected in self.package_environment({}):
            self.assertEqual(Path(selected["UV_CACHE_DIR"]), self.seeded)

    def test_explicit_user_cache_takes_priority_over_bundled_cache(self):
        self.seeded.mkdir(parents=True)
        chosen = str(self.root / "user wheel cache")
        for selected in self.package_environment({"UV_CACHE_DIR": chosen}):
            self.assertEqual(selected["UV_CACHE_DIR"], chosen)

    def test_missing_bundled_cache_preserves_uv_default_cache_selection(self):
        for selected in self.package_environment({}):
            self.assertNotIn("UV_CACHE_DIR", selected)

    def test_bundled_cache_file_is_preserved_and_refused_before_install(self):
        self.seeded.parent.mkdir()
        original = b"not a wheel cache directory"
        self.seeded.write_bytes(original)
        with patch.object(environment, "ROOT", self.root), \
                patch.dict(environment.os.environ, {}, clear=True), \
                patch.object(environment.subprocess, "run", return_value=subprocess.CompletedProcess([], 0)), \
                patch.object(environment, "run_foreground") as run:
            with self.assertRaises(environment.EnvironmentError):
                environment.install_dependencies(Path(sys.executable))
        run.assert_not_called()
        self.assertEqual(self.seeded.read_bytes(), original)

    def test_linked_bundled_cache_is_refused_before_install(self):
        self.seeded.parent.mkdir()
        real_cache = self.root / "real-cache"
        real_cache.mkdir()
        try:
            self.seeded.symlink_to(real_cache, target_is_directory=True)
        except OSError as exc:
            self.skipTest(f"local symlink creation unavailable: {exc}")
        with patch.object(environment, "ROOT", self.root), \
                patch.dict(environment.os.environ, {}, clear=True), \
                patch.object(environment.subprocess, "run", return_value=subprocess.CompletedProcess([], 0)), \
                patch.object(environment, "run_foreground") as run:
            with self.assertRaises(environment.EnvironmentError):
                environment.install_dependencies(Path(sys.executable))
        run.assert_not_called()
        self.assertTrue(self.seeded.is_symlink())


class WholeInstallationTests(unittest.TestCase):
    def setUp(self):
        self.python = Path(sys.executable)
        self.record = {"environment_root": str(self.python.parent), "python": str(self.python)}

    def install_patches(self):
        stack = contextlib.ExitStack()
        self.addCleanup(stack.close)
        stack.enter_context(patch.object(environment.sys, "version_info", (3, 13, 0)))
        stack.enter_context(patch.object(environment.struct, "calcsize", return_value=8))
        stack.enter_context(patch.object(environment, "expected_record", return_value=self.record))
        stack.enter_context(patch.object(environment, "check_python"))
        stack.enter_context(contextlib.redirect_stdout(io.StringIO()))
        return stack

    def test_dependencies_overlap_model_acquisition_before_config_publication(self):
        self.install_patches()
        dependencies_started = threading.Event()
        models_started = threading.Event()
        dependencies_done = threading.Event()
        models_done = threading.Event()

        def dependencies(python):
            self.assertEqual(python, self.python)
            dependencies_started.set()
            self.assertTrue(models_started.wait(3), "model acquisition must overlap package installation")
            dependencies_done.set()
            return 0

        def models():
            models_started.set()
            self.assertTrue(dependencies_started.wait(3), "packages must start before model acquisition finishes")
            models_done.set()
            return 0

        def record(value):
            self.assertTrue(dependencies_done.is_set())
            self.assertTrue(models_done.is_set())

        with patch.object(environment, "install_dependencies", side_effect=dependencies), \
                patch.object(environment, "install_models", side_effect=models), \
                patch.object(environment, "write_record", side_effect=record) as write:
            self.assertEqual(environment.install(), 0)
        write.assert_called_once_with(self.record)

    def test_each_failure_prevents_success_config_and_awaits_other_work(self):
        self.install_patches()
        for dependency_result, model_result in ((31, 0), (0, 41)):
            with self.subTest(dependencies=dependency_result, models=model_result), \
                    patch.object(environment, "install_dependencies", return_value=dependency_result) as dependencies, \
                    patch.object(environment, "install_models", return_value=model_result) as models, \
                    patch.object(environment, "write_record") as write:
                self.assertNotEqual(environment.install(), 0)
                dependencies.assert_called_once_with(self.python)
                models.assert_called_once_with()
                write.assert_not_called()

    def test_keyboard_interrupt_does_not_publish_configuration(self):
        self.install_patches()
        with patch.object(environment, "install_dependencies", side_effect=KeyboardInterrupt), \
                patch.object(environment, "install_models", return_value=0) as models, \
                patch.object(environment, "write_record") as write:
            self.assertEqual(environment.main(["install"]), 130)
        models.assert_called_once_with()
        write.assert_not_called()


class CompletionAssetCommandTests(unittest.TestCase):
    def test_missing_or_invalid_completion_asset_returns_concise_error_without_traceback(self):
        for error in (completion_install.CompletionInstallError("fixture missing completion model"),
                      completion_assets.CompletionAssetError("fixture invalid completion manifest")):
            with self.subTest(error=type(error).__name__), \
                    patch.object(completion_install, "install_assets", side_effect=error), \
                    contextlib.redirect_stdout(io.StringIO()) as output, \
                    contextlib.redirect_stderr(io.StringIO()) as errors:
                self.assertEqual(environment.main(["completion-assets"]), 1)
            self.assertIn("[ERROR]", errors.getvalue())
            self.assertIn(str(error), errors.getvalue())
            self.assertNotIn("Traceback", errors.getvalue())
            self.assertNotIn("passed", output.getvalue())


if __name__ == "__main__":
    unittest.main()
