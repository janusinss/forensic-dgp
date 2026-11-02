"""Offline managed-install flow with real PowerShell and harmless batch fixtures."""
import hashlib
import json
import os
from pathlib import Path
import subprocess
import tempfile
import unittest
import zipfile

BASE = Path(__file__).resolve().parents[1]
HELPER = BASE / 'implementation' / 'distribution.ps1'
PS = Path(os.environ.get('SystemRoot', r'C:\Windows')) / 'System32/WindowsPowerShell/v1.0/powershell.exe'
COMMIT = 'a' * 40


def blob(data):
    return hashlib.sha1(b'blob ' + str(len(data)).encode() + b'\0' + data).hexdigest()


@unittest.skipUnless(PS.is_file(), 'Windows PowerShell 5 is required')
class ManagedFlow(unittest.TestCase):
    def setUp(self):
        self.temp = tempfile.TemporaryDirectory(dir=BASE)
        self.addCleanup(self.temp.cleanup)
        self.local = Path(self.temp.name)
        self.root = self.local / 'DGP' / 'app'
        self.archive = self.local / 'source.zip'
        files = {
            'start.bat': (BASE / 'start.bat').read_bytes(),
            'implementation/distribution.ps1': HELPER.read_bytes(),
            'implementation/start_source.bat': b'@echo off\r\nif /i "%~1"=="--verify-only" exit /b 0\r\necho %* > "%DGP_TEST_ARGS%"\r\necho %IMPLEMENTATION_NO_PAUSE% > "%DGP_TEST_NO_PAUSE%"\r\nexit /b 0\r\n',
            'implementation/install_source.bat': b'@echo off\r\nif defined DGP_TEST_FAIL exit /b 1\r\nexit /b 0\r\n',
            'implementation/file.txt': b'fixture\n',
        }
        with zipfile.ZipFile(self.archive, 'w') as z:
            for name, data in files.items():
                z.writestr('forensic-dgp-' + COMMIT + '/' + name, data)
        self.tree = self.local / 'tree.json'
        self.tree.write_text(json.dumps({'truncated': False, 'tree': [
            {'type': 'blob', 'mode': '100644', 'path': name, 'sha': blob(data)} for name, data in files.items()
        ]}), encoding='utf8')
        source = HELPER.read_text(encoding='utf8')
        original = '  $tree = Api "https://api.github.com/repos/$repo/git/trees/$($Commit)?recursive=1"'
        replacement = "  $tree = Get-Content -LiteralPath $env:DGP_TEST_TREE -Raw | ConvertFrom-Json"
        self.assertEqual(source.count(original), 1)
        source = source.replace(original, replacement)
        original = '    Download-Limited "https://github.com/$repo/archive/$Commit.zip" $zip 100MB'
        self.assertEqual(source.count(original), 1)
        source = source.replace(original, '    Copy-Item -LiteralPath $env:DGP_TEST_ZIP -Destination $zip')
        self.fixture = self.local / 'fixture.ps1'
        self.fixture.write_text(source, encoding='utf8')
        self.env = dict(os.environ, LOCALAPPDATA=str(self.local), DGP_TEST_TREE=str(self.tree), DGP_TEST_ZIP=str(self.archive),
                        DGP_TEST_ARGS=str(self.local / 'launch-args.txt'), DGP_TEST_NO_PAUSE=str(self.local / 'launch-no-pause.txt'))

    def install(self, fail=False):
        env = dict(self.env)
        if fail:
            env['DGP_TEST_FAIL'] = '1'
        return subprocess.run([str(PS), '-NoProfile', '-NonInteractive', '-ExecutionPolicy', 'Bypass', '-File', str(self.fixture),
                               '-Action', 'Install', '-ManagedRoot', str(self.root), '-Commit', COMMIT],
                              capture_output=True, text=True, env=env, timeout=30)

    def test_install_repeat_and_dirty_refusal(self):
        result = self.install()
        self.assertEqual(result.returncode, 0, result.stdout + result.stderr)
        pointer = self.root / 'current.json'
        self.assertEqual(json.loads(pointer.read_text(encoding='utf-8-sig'))['commit'], COMMIT)
        launched = subprocess.run([str(self.root / 'start.bat'), '--no-pause', '--port', '4123', '--no-browser'],
                                  capture_output=True, text=True, env=self.env, timeout=30)
        self.assertEqual(launched.returncode, 0, launched.stdout + launched.stderr)
        self.assertEqual((self.local / 'launch-args.txt').read_text().strip(), '--port 4123 --no-browser --no-pause')
        self.assertEqual((self.local / 'launch-no-pause.txt').read_text().strip(), '1')
        self.archive.unlink()
        result = self.install()
        self.assertEqual(result.returncode, 0, result.stdout + result.stderr)
        tracked = self.root / 'versions' / COMMIT / 'implementation' / 'file.txt'
        tracked.write_bytes(b'user edit')
        original_pointer = pointer.read_bytes()
        result = self.install()
        self.assertNotEqual(result.returncode, 0)
        self.assertEqual(tracked.read_bytes(), b'user edit')
        self.assertEqual(pointer.read_bytes(), original_pointer)

    def test_failed_setup_leaves_prior_pointer(self):
        self.root.mkdir(parents=True)
        (self.root / 'owner.json').write_text('{"schema_version":1,"repository":"janusinss/forensic-dgp"}', encoding='utf8')
        pointer = self.root / 'current.json'
        pointer.write_bytes(b'{"schema":1,"commit":"old"}')
        prior = pointer.read_bytes()
        result = self.install(fail=True)
        self.assertNotEqual(result.returncode, 0)
        self.assertEqual(pointer.read_bytes(), prior)

    def test_held_mutex_blocks_install(self):
        root_string = str(self.root.resolve()).lower()
        ident = hashlib.sha256(root_string.encode()).hexdigest()[:24].upper()
        mutex_name = 'Local\\DGP-app-' + ident
        script = "$m=New-Object Threading.Mutex($false,$env:DGP_TEST_MUTEX);$m.WaitOne() | Out-Null;Write-Output 'held';[Console]::Out.Flush();[Console]::In.ReadLine() | Out-Null;$m.ReleaseMutex();$m.Dispose()"
        holder = subprocess.Popen([str(PS), '-NoProfile', '-NonInteractive', '-Command', script],
                                  stdin=subprocess.PIPE, stdout=subprocess.PIPE, stderr=subprocess.PIPE, text=True,
                                  env=dict(self.env, DGP_TEST_MUTEX=mutex_name))
        try:
            self.assertEqual(holder.stdout.readline().strip(), 'held')
            result = self.install()
            self.assertNotEqual(result.returncode, 0, result.stdout + result.stderr)
            self.assertFalse((self.root / 'current.json').exists())
        finally:
            holder.stdin.write('\n'); holder.stdin.flush()
            holder.communicate(timeout=10)


if __name__ == '__main__':
    unittest.main()
