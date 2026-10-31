"""Custom installation boundary checks using temporary folders and inert launch resolution."""
import json
import hashlib
import os
from pathlib import Path
import subprocess
import tempfile
import unittest

BASE = Path(__file__).resolve().parents[1]
PS = Path(os.environ.get('SystemRoot', r'C:\Windows')) / 'System32/WindowsPowerShell/v1.0/powershell.exe'
CSC = Path(os.environ.get('SystemRoot', r'C:\Windows')) / 'Microsoft.NET/Framework64/v4.0.30319/csc.exe'
LOCATION = BASE / 'packaging/install_location.ps1'
UNINSTALL = BASE / 'packaging/uninstall.ps1'
SHELL_MARKER = {'schema': 1, 'application': 'Forensic DGP'}


@unittest.skipUnless(PS.is_file(), 'Windows PowerShell is required')
class InstallLocationTests(unittest.TestCase):
    def setUp(self):
        (BASE / 'scratch').mkdir(exist_ok=True)
        self.temp = tempfile.TemporaryDirectory(prefix='location-', dir=BASE / 'scratch')
        self.addCleanup(self.temp.cleanup)
        self.local = Path(self.temp.name)
        self.shell = self.local / 'Chosen É location' / 'Forensic DGP'
        self.env = dict(os.environ, LOCALAPPDATA=str(self.local / 'user-local'))

    def invoke(self, action='Validate', shell=None):
        return subprocess.run([str(PS), '-NoProfile', '-NonInteractive', '-ExecutionPolicy', 'Bypass', '-File',
                               str(LOCATION), '-Action', action, '-InstallRoot', str(shell or self.shell)],
                              capture_output=True, text=True, encoding='utf8', env=self.env, timeout=15)

    def mark_shell(self):
        self.shell.mkdir(parents=True)
        (self.shell / 'shell-owner.json').write_text(json.dumps(SHELL_MARKER), encoding='utf8')

    def test_validate_is_read_only_and_returns_bound_runtime(self):
        result = self.invoke()
        self.assertEqual(result.returncode, 0, result.stdout + result.stderr)
        body = json.loads(result.stdout)
        self.assertEqual(body['schema'], 1)
        self.assertEqual(Path(body['install_root']), self.shell)
        self.assertEqual(Path(body['managed_root']), self.shell / 'runtime')
        self.assertFalse(self.shell.exists())

    def test_prepare_creates_only_owned_shell_marker(self):
        result = self.invoke('Prepare')
        self.assertEqual(result.returncode, 0, result.stdout + result.stderr)
        self.assertEqual(json.loads((self.shell / 'shell-owner.json').read_text(encoding='utf-8-sig')), SHELL_MARKER)
        self.assertFalse((self.shell / 'install-location.json').exists())
        self.assertFalse((self.shell / 'runtime').exists())
        again = self.invoke('Prepare')
        self.assertEqual(again.returncode, 0, again.stdout + again.stderr)

    def test_retained_custom_binding_is_accepted_on_reinstall(self):
        self.mark_shell()
        root = self.shell / 'runtime'; root.mkdir()
        (root / 'owner.json').write_text(json.dumps({'schema_version': 2, 'repository': 'janusinss/forensic-dgp',
            'layout': 'desktop-runtime-v1', 'install_root': str(self.shell)}), encoding='utf8')
        config = {'schema': 1, 'install_root': str(self.shell), 'managed_root': str(root)}
        (self.shell / 'install-location.json').write_text(json.dumps(config, ensure_ascii=False), encoding='utf8')
        result = self.invoke('Prepare')
        self.assertEqual(result.returncode, 0, result.stdout + result.stderr)
        self.assertEqual(json.loads((self.shell / 'install-location.json').read_text(encoding='utf8')), config)
        bad = dict(config, managed_root=str(self.local / 'unrelated'))
        (self.shell / 'install-location.json').write_text(json.dumps(bad), encoding='utf8')
        result = self.invoke('Prepare')
        self.assertNotEqual(result.returncode, 0)
        self.assertEqual(json.loads((self.shell / 'install-location.json').read_text()), bad)

    def test_existing_unrecognized_nonempty_folder_is_preserved(self):
        self.shell.mkdir(parents=True)
        sentinel = self.shell / 'unrelated.txt'
        sentinel.write_bytes(b'preserve')
        for action in ('Validate', 'Prepare'):
            result = self.invoke(action)
            self.assertNotEqual(result.returncode, 0)
            self.assertEqual(sentinel.read_bytes(), b'preserve')
            self.assertFalse((self.shell / 'shell-owner.json').exists())

    def test_owner_marker_with_extra_fields_is_refused(self):
        self.mark_shell()
        marker = self.shell / 'shell-owner.json'
        marker.write_text(json.dumps({**SHELL_MARKER, 'extra': True}), encoding='utf8')
        result = self.invoke()
        self.assertNotEqual(result.returncode, 0)
        self.assertTrue(marker.is_file())

    def test_system_bases_network_and_ambiguous_paths_are_refused(self):
        values = [str(Path(os.environ['SystemRoot']).anchor), os.environ['SystemRoot'],
                  str(Path(os.environ['SystemRoot']) / 'Forensic DGP'),
                  os.environ['USERPROFILE'], r'\\server\share\Forensic DGP',
                  r'\\?\C:\Forensic DGP', 'C:Forensic DGP', 'relative',
                  str(self.local / 'bad.'), str(self.local / 'CON'), str(self.local / ('x' * 121))]
        if os.environ.get('ProgramFiles'):
            values.append(str(Path(os.environ['ProgramFiles']) / 'Forensic DGP'))
        for value in values:
            with self.subTest(path=value):
                result = self.invoke(shell=value)
                self.assertNotEqual(result.returncode, 0, result.stdout + result.stderr)

    def test_junction_ancestor_is_refused(self):
        target = self.local / 'target'; target.mkdir()
        junction = self.local / 'junction'
        result = subprocess.run([str(PS), '-NoProfile', '-NonInteractive', '-Command',
                                 "New-Item -ItemType Junction -Path $env:DGP_TEST_LINK -Target $env:DGP_TEST_TARGET | Out-Null"],
                                capture_output=True, text=True,
                                env=dict(self.env, DGP_TEST_LINK=str(junction), DGP_TEST_TARGET=str(target)))
        if result.returncode:
            self.skipTest('Junction fixture unavailable')
        try:
            refused = self.invoke(shell=junction / 'Forensic DGP')
            self.assertNotEqual(refused.returncode, 0)
            self.assertFalse((target / 'Forensic DGP').exists())
        finally:
            junction.rmdir()

    def test_custom_uninstall_retains_every_backend_file(self):
        self.mark_shell()
        root = self.shell / 'runtime'; root.mkdir()
        marker = {'schema_version': 2, 'repository': 'janusinss/forensic-dgp',
                  'layout': 'desktop-runtime-v1', 'install_root': str(self.shell)}
        (root / 'owner.json').write_text(json.dumps(marker), encoding='utf8')
        saved = root / 'data/scratch/capture_requests/original.png'
        saved.parent.mkdir(parents=True); saved.write_bytes(b'original evidence')
        result = subprocess.run([str(PS), '-NoProfile', '-NonInteractive', '-ExecutionPolicy', 'Bypass', '-File',
                                 str(UNINSTALL), '-Action', 'Uninstall', '-ManagedRoot', str(root), '-InstallRoot', str(self.shell)],
                                capture_output=True, text=True, encoding='utf8', env=self.env, timeout=15)
        self.assertEqual(result.returncode, 0, result.stdout + result.stderr)
        self.assertEqual(Path(json.loads(result.stdout)['managed_root']), root)
        self.assertEqual(saved.read_bytes(), b'original evidence')
        self.assertEqual(json.loads((root / 'owner.json').read_text()), marker)

    def test_custom_uninstall_refuses_unbound_backend_marker(self):
        self.mark_shell()
        root = self.shell / 'runtime'; root.mkdir()
        marker = {'schema_version': 2, 'repository': 'janusinss/forensic-dgp',
                  'layout': 'desktop-runtime-v1', 'install_root': str(self.local / 'other')}
        (root / 'owner.json').write_text(json.dumps(marker), encoding='utf8')
        result = subprocess.run([str(PS), '-NoProfile', '-NonInteractive', '-ExecutionPolicy', 'Bypass', '-File',
                                 str(UNINSTALL), '-Action', 'Check', '-ManagedRoot', str(root), '-InstallRoot', str(self.shell)],
                                capture_output=True, text=True, encoding='utf8', env=self.env, timeout=15)
        self.assertNotEqual(result.returncode, 0)
        self.assertEqual(json.loads((root / 'owner.json').read_text()), marker)


@unittest.skipUnless(CSC.is_file(), 'Windows C# compiler is required')
class LauncherInstallLocationTests(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        (BASE / 'scratch').mkdir(exist_ok=True)
        cls.temp = tempfile.TemporaryDirectory(prefix='launcher-location-', dir=BASE / 'scratch')
        cls.folder = Path(cls.temp.name)
        fixture = cls.folder / 'LocationTests.cs'
        fixture.write_text('''using System;
internal static class LocationFixture {
 public static int Main(string[] args) {
  try { Console.OutputEncoding = new System.Text.UTF8Encoding(false); InstallLocation value = InstallLocation.Resolve(args[0], args[1]); Console.WriteLine(value.ManagedRoot); Console.WriteLine(value.InstallRoot ?? "legacy"); return 0; }
  catch(Exception error) { Console.Error.WriteLine(error.Message); return 1; }
 }
}''', encoding='utf8')
        cls.exe = cls.folder / 'LocationTests.exe'
        cls.compiled = subprocess.run([str(CSC), '/nologo', '/target:exe', '/main:LocationFixture',
                                       '/reference:System.Windows.Forms.dll', '/reference:System.Web.Extensions.dll',
                                       '/reference:System.Drawing.dll', '/out:' + str(cls.exe),
                                       str(BASE / 'packaging/launcher/Program.cs'), str(fixture)],
                                      capture_output=True, text=True, timeout=20)

    @classmethod
    def tearDownClass(cls):
        cls.temp.cleanup()

    def setUp(self):
        self.assertEqual(self.compiled.returncode, 0, self.compiled.stdout + self.compiled.stderr)
        self.local = self.folder / hashlib.sha1(self._testMethodName.encode()).hexdigest()[:6]
        self.local.mkdir()
        self.shell = self.local / 'chosen' / 'Forensic DGP'
        self.shell.mkdir(parents=True)
        (self.shell / 'shell-owner.json').write_text(json.dumps(SHELL_MARKER), encoding='utf8')

    def resolve(self, shell=None):
        return subprocess.run([str(self.exe), str(shell or self.shell), str(self.local)], capture_output=True, text=True, encoding='utf8', timeout=10)

    def config(self, **changes):
        value = {'schema': 1, 'install_root': str(self.shell), 'managed_root': str(self.shell / 'runtime')}
        value.update(changes)
        (self.shell / 'install-location.json').write_text(json.dumps(value), encoding='utf8')

    def test_custom_config_binds_to_own_executable_directory(self):
        self.config()
        result = self.resolve()
        self.assertEqual(result.returncode, 0, result.stderr)
        self.assertEqual(Path(result.stdout.splitlines()[0]), self.shell / 'runtime')
        self.assertEqual(Path(result.stdout.splitlines()[1]), self.shell)

    def test_custom_launcher_cannot_fallback_to_unrelated_legacy_install(self):
        result = self.resolve()
        self.assertNotEqual(result.returncode, 0)

    def test_legacy_missing_config_allowed_only_in_exact_default_shell(self):
        shell = self.local / 'Programs' / 'Forensic DGP'
        shell.mkdir(parents=True)
        result = self.resolve(shell)
        self.assertEqual(result.returncode, 0, result.stderr)
        self.assertEqual(Path(result.stdout.splitlines()[0]), self.local / 'DGP' / 'app')
        self.assertEqual(result.stdout.splitlines()[1], 'legacy')

    def test_config_cannot_redirect_runtime_outside_own_shell(self):
        for key, value in [('install_root', str(self.local / 'other')),
                           ('managed_root', str(self.local / 'DGP/app')), ('schema', True), ('extra', True)]:
            self.config(**{key: value})
            result = self.resolve()
            self.assertNotEqual(result.returncode, 0, result.stdout)


if __name__ == '__main__':
    unittest.main()