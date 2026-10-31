"""Release metadata must identify exact source and never include untracked files."""
import importlib.util
import io
import os
import subprocess
import tempfile
import zipfile
import unittest
from unittest import mock
from pathlib import Path

spec = importlib.util.spec_from_file_location("dgp_release_builder", Path(__file__).resolve().parents[1] / "packaging/build_release.py")
builder = importlib.util.module_from_spec(spec)
spec.loader.exec_module(builder)


class ReleaseManifestTests(unittest.TestCase):
    def test_compiler_version_uses_engine_version_command(self):
        with mock.patch.object(builder.subprocess, 'run', return_value=mock.Mock(stdout=b'7.1.1\r\n')) as run:
            builder.check_inno_version(Path('ISCC.exe'))
        run.assert_called_once_with(['ISCC.exe', '--version'], stdout=builder.subprocess.PIPE,
                                    stderr=builder.subprocess.STDOUT, check=True)

    def test_compiler_version_requires_exact_pinned_engine(self):
        for output in (b'7.1.10\r\n', b'Inno Setup 7 Command-Line Compiler\r\n', b'6.7.1.1'):
            with self.subTest(output=output), mock.patch.object(builder.subprocess, 'run', return_value=mock.Mock(stdout=output)):
                with self.assertRaises(RuntimeError):
                    builder.check_inno_version(Path('ISCC.exe'))

    def test_archive_uses_canonical_unspecified_text_despite_autocrlf(self):
        with tempfile.TemporaryDirectory() as temporary:
            root = Path(temporary)
            config = root / 'global-config'
            config.write_text('[core]\n    autocrlf = true\n', encoding='utf8')
            env = dict(os.environ, GIT_CONFIG_GLOBAL=str(config))
            repo = root / 'repo'
            repo.mkdir()
            def git(*args, input=None):
                return subprocess.run(['git', *args], cwd=repo, env=env, input=input,
                                      check=True, stdout=subprocess.PIPE, stderr=subprocess.PIPE).stdout
            git('init', '-q')
            (repo / '.gitattributes').write_bytes(b'*.bat text eol=crlf\n')
            (repo / 'NOTICE').write_bytes(b'first\nsecond\n')
            (repo / 'start.bat').write_bytes(b'@echo off\nexit /b 0\n')
            git('-c', 'core.autocrlf=false', 'add', '.')
            git('-c', 'user.name=Fixture', '-c', 'user.email=fixture@example.invalid',
                '-c', 'commit.gpgsign=false', 'commit', '--no-verify', '-qm', 'fixture')
            sha = git('rev-parse', 'HEAD').decode().strip()
            # Establish the exact failure: ambient Windows autocrlf changes a
            # non-batch archive member even though its canonical Git blob is LF.
            with zipfile.ZipFile(io.BytesIO(git('archive', '--format=zip', sha))) as broken:
                self.assertEqual(broken.read('NOTICE'), b'first\r\nsecond\r\n')
            with mock.patch.object(builder, 'ROOT', repo), mock.patch.dict(os.environ, env, clear=True):
                archive = builder.source_archive(sha)
            with zipfile.ZipFile(io.BytesIO(archive)) as fixed:
                prefix = 'forensic-dgp-' + sha + '/'
                self.assertEqual(fixed.read(prefix + 'NOTICE'), git('show', sha + ':NOTICE'))
                self.assertEqual(fixed.read(prefix + '.gitattributes'), git('show', sha + ':.gitattributes'))
                self.assertEqual(fixed.read(prefix + 'start.bat'), b'@echo off\r\nexit /b 0\r\n')

    def test_manifest_binds_payload_and_tree(self):
        sha = "a" * 40
        value = builder.source_manifest(sha, b"zip", b"100644 blob " + b"b" * 40 + b"\tapp.py\0", (sha + "\n").encode())
        self.assertEqual(value["files"], {"app.py": "b" * 40})
        self.assertEqual(value["source_zip_size"], 3)
        self.assertEqual(len(value["source_zip_sha256"]), 64)

    def test_links_and_traversal_refused(self):
        sha = "a" * 40
        for mode, name in ((b"120000", b"link"), (b"100644", b"../outside"), (b"100644", b"bad\\path")):
            with self.subTest(name=name), self.assertRaises(ValueError):
                builder.source_manifest(sha, b"zip", mode + b" blob " + b"b" * 40 + b"\t" + name + b"\0", (sha + "\n").encode())

    def test_version_and_history_validation(self):
        for value in ("1.2", "1.2.3;evil", "01.2.3", "10000.0.0"):
            with self.subTest(value=value), self.assertRaises(ValueError):
                builder.checked_version(value)
        self.assertEqual(builder.checked_version("0.1.0"), "0.1.0")
        with self.assertRaises(ValueError):
            builder.source_manifest("a" * 40, b"zip", b"100644 blob " + b"b" * 40 + b"\tapp.py\0", b"bad\n")


if __name__ == "__main__":
    unittest.main()
