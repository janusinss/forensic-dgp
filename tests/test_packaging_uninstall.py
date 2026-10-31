"""Real PowerShell tests for non-destructive shared-install uninstall checks."""
import hashlib
import json
import os
from pathlib import Path
import subprocess
import tempfile
import unittest

BASE = Path(__file__).resolve().parents[1]
HELPER = BASE / 'packaging' / 'uninstall.ps1'
PS = Path(os.environ.get('SystemRoot', r'C:\Windows')) / 'System32/WindowsPowerShell/v1.0/powershell.exe'
MARKER = {'schema_version': 1, 'repository': 'janusinss/forensic-dgp'}


@unittest.skipUnless(PS.is_file(), 'Windows PowerShell is required')
class SharedInstallationUninstall(unittest.TestCase):
    def setUp(self):
        (BASE / 'scratch').mkdir(exist_ok=True)
        self.temp = tempfile.TemporaryDirectory(prefix='uninstall-', dir=BASE / 'scratch')
        self.addCleanup(self.temp.cleanup)
        self.local = Path(self.temp.name)
        self.root = self.local / 'DGP' / 'app'
        self.root.mkdir(parents=True)
        (self.root / 'owner.json').write_text(json.dumps(MARKER), encoding='utf8')
        for name in ('data/scratch/capture_requests/original.png',
                     'data/scratch/completion_requests/mask.png',
                     'data/scratch/implementation_requests/result.png',
                     'data/models/implementation/model.pth',
                     'environments/shared/python.exe', 'versions/shared/code.py'):
            path = self.root / name
            path.parent.mkdir(parents=True, exist_ok=True)
            path.write_bytes(('sentinel ' + name).encode())
        private = self.local / 'DGP' / 'python-private' / 'python.exe'
        private.parent.mkdir()
        private.write_bytes(b'shared python sentinel')
        self.env = dict(os.environ, LOCALAPPDATA=str(self.local))

    def inventory(self):
        return {str(p.relative_to(self.local)): p.read_bytes() for p in self.local.rglob('*') if p.is_file()}

    def invoke(self, action='Check', root=None, env=None):
        return subprocess.run([str(PS), '-NoProfile', '-NonInteractive', '-ExecutionPolicy', 'Bypass',
                               '-File', str(HELPER), '-Action', action, '-ManagedRoot', str(root or self.root)],
                              capture_output=True, text=True, env=env or self.env, timeout=15)

    def test_check_emits_expected_lock_without_changing_any_file(self):
        before = self.inventory()
        result = self.invoke()
        self.assertEqual(result.returncode, 0, result.stdout + result.stderr)
        body = json.loads(result.stdout)
        ident = hashlib.sha256(str(self.root.resolve()).lower().encode()).hexdigest()[:24].upper()
        self.assertEqual(body['mutex_name'], 'Local\\DGP-app-' + ident)
        self.assertEqual(Path(body['retained_data']), self.root / 'data')
        self.assertEqual(body['action'], 'Check')
        self.assertEqual(self.inventory(), before)

    def test_uninstall_retains_evidence_models_and_shared_environments(self):
        before = self.inventory()
        result = self.invoke('Uninstall')
        self.assertEqual(result.returncode, 0, result.stdout + result.stderr)
        body = json.loads(result.stdout)
        self.assertEqual(body['policy'], 'retain_shared_backend')
        self.assertEqual(self.inventory(), before)

    def test_missing_backend_is_allowed_without_creating_it(self):
        root = self.local / 'empty-local' / 'DGP' / 'app'
        result = self.invoke(root=root, env=dict(self.env, LOCALAPPDATA=str(self.local / 'empty-local')))
        self.assertEqual(result.returncode, 0, result.stdout + result.stderr)
        self.assertFalse(root.exists())
        self.assertFalse(json.loads(result.stdout)['managed_exists'])

    def test_missing_marker_refuses_existing_root_without_deletion(self):
        (self.root / 'owner.json').unlink()
        before = self.inventory()
        result = self.invoke('Uninstall')
        self.assertNotEqual(result.returncode, 0)
        self.assertIn('marker', result.stderr.lower())
        self.assertEqual(self.inventory(), before)

    def test_wrong_or_extended_marker_refuses_without_deletion(self):
        for marker in ({**MARKER, 'extra': True}, {**MARKER, 'schema_version': True},
                       {**MARKER, 'repository': 'another/repository'}):
            with self.subTest(marker=marker):
                (self.root / 'owner.json').write_text(json.dumps(marker), encoding='utf8')
                before = self.inventory()
                result = self.invoke('Uninstall')
                self.assertNotEqual(result.returncode, 0)
                self.assertIn('marker', result.stderr.lower())
                self.assertEqual(self.inventory(), before)

    def test_outside_expected_root_is_refused(self):
        result = self.invoke(root=self.local / 'DGP')
        self.assertNotEqual(result.returncode, 0)
        self.assertIn('expected', result.stderr.lower())
        self.assertTrue((self.root / 'owner.json').is_file())

    def test_relative_local_appdata_is_refused(self):
        result = self.invoke(env=dict(self.env, LOCALAPPDATA='relative'))
        self.assertNotEqual(result.returncode, 0)
        self.assertIn('absolute', result.stderr.lower())

    def test_held_mutex_blocks_check_and_uninstall(self):
        ident = hashlib.sha256(str(self.root.resolve()).lower().encode()).hexdigest()[:24].upper()
        script = "$m=New-Object Threading.Mutex($false,$env:DGP_TEST_MUTEX);$m.WaitOne()|Out-Null;Write-Output 'held';[Console]::Out.Flush();[Console]::In.ReadLine()|Out-Null;$m.ReleaseMutex();$m.Dispose()"
        holder = subprocess.Popen([str(PS), '-NoProfile', '-NonInteractive', '-Command', script],
                                  stdin=subprocess.PIPE, stdout=subprocess.PIPE, stderr=subprocess.PIPE, text=True,
                                  env=dict(self.env, DGP_TEST_MUTEX='Local\\DGP-app-' + ident))
        try:
            self.assertEqual(holder.stdout.readline().strip(), 'held')
            before = self.inventory()
            for action in ('Check', 'Uninstall'):
                with self.subTest(action=action):
                    result = self.invoke(action)
                    self.assertNotEqual(result.returncode, 0)
                    self.assertIn('running', result.stderr.lower())
                    self.assertEqual(self.inventory(), before)
        finally:
            holder.communicate('\n', timeout=10)

    def test_junction_root_is_refused(self):
        link_local = self.local / 'linked-local'
        link_local.mkdir()
        (link_local / 'DGP').mkdir()
        link_root = link_local / 'DGP' / 'app'
        created = subprocess.run([str(PS), '-NoProfile', '-NonInteractive', '-Command',
                                  "New-Item -ItemType Junction -Path $env:DGP_TEST_LINK -Target $env:DGP_TEST_TARGET | Out-Null"],
                                 capture_output=True, text=True,
                                 env=dict(self.env, DGP_TEST_LINK=str(link_root), DGP_TEST_TARGET=str(self.root)))
        if created.returncode:
            self.skipTest('Cannot create a Windows junction: ' + created.stderr.strip())
        try:
            before = self.inventory()
            result = self.invoke(root=link_root, env=dict(self.env, LOCALAPPDATA=str(link_local)))
            self.assertNotEqual(result.returncode, 0)
            self.assertIn('reparse', result.stderr.lower())
            self.assertEqual(self.inventory(), before)
        finally:
            # Remove only the verified junction itself; never recurse into its target.
            link_root.rmdir()


if __name__ == '__main__':
    unittest.main()