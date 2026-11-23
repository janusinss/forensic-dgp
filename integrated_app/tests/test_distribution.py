"""Offline checks of the managed distribution's pure PowerShell helpers."""
import hashlib
import base64
import json
import os
from pathlib import Path
import shutil
import subprocess
import tempfile
import unittest
import zipfile

SCRIPT = Path(__file__).resolve().parents[1] / 'implementation' / 'distribution.ps1'
SHELL = shutil.which('pwsh') or shutil.which('powershell.exe')


@unittest.skipUnless(SHELL, 'PowerShell is required')
class DistributionSafety(unittest.TestCase):
    def setUp(self):
        self.temp = tempfile.TemporaryDirectory(dir=Path(__file__).resolve().parents[1])
        self.addCleanup(self.temp.cleanup)
        self.root = Path(self.temp.name)
        # Load only helper definitions: do not run the installer or any network call.
        source = SCRIPT.read_text(encoding='utf8')
        self.helpers = source[source.index('function Check-Path'):source.index('\nCheck-Path $root\n')]

    def run_ps(self, body):
        script = "$ErrorActionPreference='Stop'\n" + self.helpers + '\n' + body
        encoded = base64.b64encode(script.encode('utf-16le')).decode('ascii')
        env = dict(os.environ, DISTRIBUTION_TEST_ROOT=str(self.root))
        return subprocess.run([SHELL, '-NoProfile', '-NonInteractive', '-EncodedCommand', encoded],
                              capture_output=True, text=True, encoding='utf8', errors='replace', env=env, timeout=15)

    def test_windows_path_validation(self):
        result = self.run_ps(r'''
$bad = @('../evil','a/../b','a/./b','/absolute','a//b','C:/evil','a\b','a/NUL.txt','a/COM1','a/trail.','a/trail ','a/file:stream','a/evil?.txt')
foreach ($p in $bad) { if (Valid-Relative $p) { throw "Accepted unsafe path: $p" } }
foreach ($p in @('README.MD','static/capture.js','implementation/assets/file.npz')) { if (!(Valid-Relative $p)) { throw "Rejected valid path: $p" } }
''')
        self.assertEqual(result.returncode, 0, result.stderr)

    def test_git_blob_hash_matches_git_object_algorithm(self):
        payload = b'fixture\0contents\n'
        file = self.root / 'blob'
        file.write_bytes(payload)
        expected = hashlib.sha1(b'blob ' + str(len(payload)).encode() + b'\0' + payload).hexdigest()
        result = self.run_ps("Git-Blob (Join-Path $env:DISTRIBUTION_TEST_ROOT 'blob')")
        self.assertEqual(result.returncode, 0, result.stderr)
        self.assertEqual(result.stdout.strip(), expected)

    def test_archive_batch_crlf_must_match_canonical_git_blob(self):
        canonical = b'@echo off\necho fixture\n'
        expected = hashlib.sha1(b'blob ' + str(len(canonical)).encode() + b'\0' + canonical).hexdigest()
        path = self.root / 'fixture.bat'
        path.write_bytes(canonical.replace(b'\n', b'\r\n'))
        body = "Verify-SourceBlob (Join-Path $env:DISTRIBUTION_TEST_ROOT 'fixture.bat') 'fixture.bat' '" + expected + "'"
        result = self.run_ps(body)
        self.assertEqual(result.returncode, 0, result.stderr)
        self.assertEqual(path.read_bytes(), canonical)
        tampered = b'@echo off\r\necho changed\r\n'
        path.write_bytes(tampered)
        result = self.run_ps(body)
        self.assertNotEqual(result.returncode, 0)
        self.assertEqual(path.read_bytes(), tampered)
        path.write_bytes(canonical.replace(b'\n', b'\r\n'))
        non_batch = body.replace("'fixture.bat' '", "'fixture.txt' '")
        self.assertNotEqual(self.run_ps(non_batch).returncode, 0)

    def test_check_path_rejects_reparse_parent_even_when_leaf_exists(self):
        actual = self.root / 'actual'
        actual.mkdir()
        (actual / 'file.txt').write_text('fixture', encoding='utf8')
        alias = self.root / 'alias'
        try:
            os.symlink(actual, alias, target_is_directory=True)
        except OSError as exc:
            self.skipTest(f'Directory symlink creation unavailable: {exc}')
        result = self.run_ps("Check-Path (Join-Path $env:DISTRIBUTION_TEST_ROOT 'alias\\file.txt')")
        self.assertNotEqual(result.returncode, 0)
        self.assertEqual((actual / 'file.txt').read_text(encoding='utf8'), 'fixture')

    def test_edited_or_extra_file_refuses_without_mutation(self):
        revision = 'a' * 40
        version = self.root / 'versions' / revision
        version.mkdir(parents=True)
        receipt_dir = self.root / 'receipts'
        receipt_dir.mkdir()
        tracked = version / 'app.txt'
        tracked.write_bytes(b'original')
        receipt = {'schema': 1, 'commit': revision, 'files': {'app.txt': hashlib.sha256(b'original').hexdigest()}}
        (receipt_dir / (revision + '.json')).write_text(json.dumps(receipt), encoding='utf8')
        pointer = self.root / 'current.json'
        pointer.write_bytes(b'prior pointer')
        body = "$root=$env:DISTRIBUTION_TEST_ROOT\nVerify-Version '" + revision + "'"
        good = self.run_ps(body)
        self.assertEqual(good.returncode, 0, good.stderr)
        tracked.write_bytes(b'user edit')
        bad = self.run_ps(body)
        self.assertNotEqual(bad.returncode, 0)
        self.assertEqual(tracked.read_bytes(), b'user edit')
        self.assertEqual(pointer.read_bytes(), b'prior pointer')
        tracked.write_bytes(b'original')
        (version / 'extra.txt').write_bytes(b'user file')
        self.assertNotEqual(self.run_ps(body).returncode, 0)
        self.assertEqual(pointer.read_bytes(), b'prior pointer')

    def test_archive_preflight_refuses_unsafe_incomplete_or_oversized_entries(self):
        revision = 'a' * 40
        prefix = 'forensic-dgp-' + revision + '/'
        archive = self.root / 'input.zip'
        pointer = self.root / 'current.json'
        pointer.write_bytes(b'prior pointer')
        command = r'''Add-Type -AssemblyName System.IO.Compression.FileSystem
$archive=[IO.Compression.ZipFile]::OpenRead((Join-Path $env:DISTRIBUTION_TEST_ROOT 'input.zip'))
try { Test-Archive $archive @{'app.txt'='unused'} ('a' * 40) } finally { $archive.Dispose() }
'''
        def write(entries):
            with zipfile.ZipFile(archive, 'w') as z:
                for entry, contents in entries:
                    z.writestr(entry, contents)
        write([(prefix + 'app.txt', b'valid')])
        result = self.run_ps(command)
        self.assertEqual(result.returncode, 0, result.stderr)
        link = zipfile.ZipInfo(prefix + 'app.txt')
        link.create_system = 3
        link.external_attr = 0xA1FF << 16
        cases = [
            [(prefix + '../app.txt', b'bad')],
            [(prefix + 'NUL.txt', b'bad')],
            [(prefix + 'other.txt', b'bad')],
            [(prefix + 'app.txt', b'a'), (prefix + 'app.txt', b'b')],
            [(link, b'target')],
        ]
        for entries in cases:
            with self.subTest(entries=entries):
                write(entries)
                result = self.run_ps(command)
                self.assertNotEqual(result.returncode, 0, result.stdout)
                self.assertEqual(pointer.read_bytes(), b'prior pointer')
        oversized = r'''$entry=[pscustomobject]@{Length=31457281;ExternalAttributes=0;FullName=('forensic-dgp-'+('a'*40)+'/app.txt')}
Test-Archive ([pscustomobject]@{Entries=@($entry)}) @{'app.txt'='unused'} ('a'*40)'''
        self.assertNotEqual(self.run_ps(oversized).returncode, 0)


if __name__ == '__main__':
    unittest.main()
