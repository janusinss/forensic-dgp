"""Windows batch selection tests using real local Python and a harmless app stub."""
import json
import os
from pathlib import Path
import shutil
import subprocess
import sys
import tempfile
import unittest
import venv


ROOT = Path(__file__).resolve().parents[1]


@unittest.skipUnless(os.name == "nt", "Windows installer batch behavior")
class WindowsBootstrapTests(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        launcher = shutil.which("py.exe")
        if not launcher:
            raise unittest.SkipTest("a system Python launcher is required for fallback fixtures")
        checked = subprocess.run([launcher, "-3.13", "-c", "import struct,sys;sys.exit(struct.calcsize('P') != 8)"],
                                 capture_output=True, timeout=15, check=False)
        if checked.returncode:
            raise unittest.SkipTest("system Python 3.13 64-bit is required for fallback fixtures")

    def setUp(self):
        self.temporary = tempfile.TemporaryDirectory(prefix="installer fixture ")
        self.addCleanup(self.temporary.cleanup)
        self.root = Path(self.temporary.name)
        (self.root / "implementation").mkdir()
        self.installer = self.root / "implementation" / "install_source.bat"
        shutil.copyfile(ROOT / "implementation" / "install_source.bat", self.installer)
        implementation = self.root / "implementation"
        implementation.mkdir(exist_ok=True)
        (implementation / "__init__.py").write_text("", encoding="utf-8")
        (implementation / "environment.py").write_text(
            "import json,os,sys\n"
            "from pathlib import Path\n"
            "with Path(os.environ['FIXTURE_SELECTION_LOG']).open('a', encoding='utf-8') as stream:\n"
            "    stream.write(json.dumps({'python': sys.executable, 'args': sys.argv[1:]}) + '\\n')\n"
            "raise SystemExit(int(os.environ.get('FIXTURE_INSTALL_EXIT', '0')))\n", encoding="utf-8")
        self.local = self.root / "Local App Data with spaces"
        self.private = self.local / "DGP" / "python-3.13.9-amd64"
        self.python = self.private / "python.exe"
        self.log = self.root / "selection.jsonl"

    def valid_private_python(self):
        # The redirector and pyvenv.cfg reuse installed local stdlib files;
        # this fixture does not install pip, dependencies, models, or Python.
        venv.EnvBuilder(with_pip=False).create(self.private)
        shutil.copyfile(self.private / "Scripts" / "python.exe", self.python)
        checked = subprocess.run([str(self.python), "-I", "-c", "import struct,sys;sys.exit(struct.calcsize('P') != 8)"],
                                 capture_output=True, timeout=15, check=False)
        self.assertEqual(checked.returncode, 0, checked.stderr.decode(errors="replace"))

    def run_installer(self, *, install_exit=0):
        fixture_env = os.environ.copy()
        fixture_env.update({"LOCALAPPDATA": str(self.local), "IMPLEMENTATION_NO_PAUSE": "1",
                            "FIXTURE_SELECTION_LOG": str(self.log), "FIXTURE_INSTALL_EXIT": str(install_exit)})
        result = subprocess.run([os.environ.get("COMSPEC", "cmd.exe"), "/d", "/c", "call", str(self.installer), "--no-pause"],
                                cwd=self.root, env=fixture_env, capture_output=True, text=True,
                                timeout=30, check=False)
        rows = [json.loads(line) for line in self.log.read_text(encoding="utf-8").splitlines()] if self.log.exists() else []
        return result, rows

    def test_valid_private_runtime_is_preferred_even_with_supported_system_python(self):
        self.valid_private_python()
        result, rows = self.run_installer()
        self.assertEqual(result.returncode, 0, result.stdout + result.stderr)
        self.assertEqual(len(rows), 1)
        self.assertEqual(Path(rows[0]["python"]), self.python)
        self.assertEqual(rows[0]["args"], ["install"])

    def test_invalid_private_runtime_is_preserved_during_system_fallback(self):
        self.private.mkdir(parents=True)
        original = b"invalid private Python fixture must remain unchanged"
        self.python.write_bytes(original)
        result, rows = self.run_installer()
        self.assertEqual(result.returncode, 0, result.stdout + result.stderr)
        self.assertEqual(self.python.read_bytes(), original)
        self.assertEqual(len(rows), 1)
        self.assertNotEqual(Path(rows[0]["python"]), self.python)
        self.assertEqual(rows[0]["args"], ["install"])

    def test_actual_install_failure_does_not_retry_using_another_python(self):
        self.valid_private_python()
        result, rows = self.run_installer(install_exit=23)
        self.assertNotEqual(result.returncode, 0)
        self.assertEqual(len(rows), 1, "a dependency/model failure must stop without a different-runtime retry")
        self.assertEqual(Path(rows[0]["python"]), self.python)
        self.assertIn("Installation stopped", result.stdout)


if __name__ == "__main__":
    unittest.main()
