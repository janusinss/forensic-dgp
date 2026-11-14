"""Exercise the real Inno uninstaller against a disposable Windows app registration."""

import ctypes
from ctypes import wintypes
import hashlib
import json
import os
from pathlib import Path
import shutil
import subprocess
import tempfile
import time
import unittest
import uuid

if os.name == 'nt':
    import winreg


BASE = Path(__file__).resolve().parents[1]
COMPILER = Path(os.environ.get('DGP_INNO_COMPILER', BASE / 'scratch/tools/inno-7.1.1/ISCC.exe'))
PRODUCTION_APP_ID = 'FEE649E9-195D-4576-8F30-DAE06A5099C8'
UNINSTALL_KEY = r'Software\Microsoft\Windows\CurrentVersion\Uninstall'
SILENT_ARGS = ('/VERYSILENT', '/SUPPRESSMSGBOXES', '/NORESTART')


@unittest.skipUnless(os.name == 'nt' and COMPILER.is_file(), 'Verified Windows Inno compiler required')
class NativeUninstallTests(unittest.TestCase):
    def setUp(self):
        scratch = BASE / 'scratch'
        scratch.mkdir(exist_ok=True)
        temporary = tempfile.TemporaryDirectory(prefix='native-uninstall-', dir=scratch)
        self.addCleanup(temporary.cleanup)
        self.area = Path(temporary.name)
        self.shell = self.area / 'fixture-app'
        self.runtime = self.shell / 'runtime'
        self.tag = 'DGP uninstall fixture ' + uuid.uuid4().hex
        self.app_id = str(uuid.uuid4()).upper()
        self.registry_key = UNINSTALL_KEY + '\\{' + self.app_id + '}_is1'
        self.shortcut = (Path(os.environ['APPDATA']) / 'Microsoft/Windows/Start Menu/Programs'
                         / (self.tag + '.lnk'))
        self.addCleanup(self.clear_fixture_registration)
        self.addCleanup(self.shortcut.unlink, missing_ok=True)

        # Production's backend creates the selected folder before Inno installs its shell.
        self.shell.mkdir()
        self.compile_fixture()
        result, log = self.run_inno(self.area / 'fixture-setup.exe', 'setup', '/DIR=' + str(self.shell))
        self.assertEqual(result.returncode, 0, log)
        self.assert_registered()
        self.assertTrue(self.shortcut.is_file(), log)
        self.uninstaller = self.shell / 'unins000.exe'
        self.assertTrue(self.uninstaller.is_file())

        self.write(self.runtime / 'owner.json', {
            'schema_version': 2, 'repository': 'janusinss/forensic-dgp',
            'layout': 'desktop-runtime-v1', 'install_root': str(self.shell),
        })
        self.binding = {
            'schema': 1, 'install_root': str(self.shell), 'managed_root': str(self.runtime),
        }
        self.write(self.shell / 'install-location.json', self.binding)
        self.write(self.shell / ('install-location.json.previous.' + '1' * 32), self.binding)
        for path in ('data/models/model.pth', 'data/records/saved.json', 'python/python.exe',
                     'environments/capture/library.dll', 'versions/release/backend.py'):
            self.write(self.runtime / path, path.encode('ascii'))
        self.sibling = self.area / 'unrelated' / 'keep.txt'
        self.write(self.sibling, b'outside the installation')

    def compile_fixture(self):
        packaging = BASE / 'packaging'
        script = (packaging / 'installer.iss').read_text(encoding='utf-8')
        self.assertIn('AppId={{' + PRODUCTION_APP_ID + '}', script)
        script = script.replace(PRODUCTION_APP_ID, self.app_id)
        script = script.replace('AppName=Forensic DGP', 'AppName=' + self.tag)
        script = script.replace('AppVerName=Forensic DGP ', 'AppVerName=' + self.tag + ' ')
        default_directory = r'DefaultDirName={localappdata}\Programs\Forensic DGP'
        self.assertIn(default_directory, script)
        script = script.replace(default_directory, r'DefaultDirName={#StageDir}\fixture-app')
        script = script.replace('DefaultGroupName=Forensic DGP', 'DefaultGroupName=' + self.tag)
        script = script.replace('OutputBaseFilename=ForensicDGP_Setup', 'OutputBaseFilename=fixture-setup')
        script = script.replace(r'Name: "{userprograms}\Forensic DGP"', 'Name: "{userprograms}\\' + self.tag + '"')
        script = script.replace(r'Name: "{userdesktop}\Forensic DGP"', 'Name: "{userdesktop}\\' + self.tag + '"')
        for function in ('NextButtonClick', 'PrepareToInstall'):
            event = 'function ' + function + '('
            self.assertEqual(script.count(event), 1)
            script = script.replace(event, 'function Fixture' + function + '(')
        fixture = self.area / 'fixture.iss'
        fixture.write_text(script, encoding='utf-8-sig')

        helper = self.area / 'uninstall.ps1'
        shutil.copyfile(packaging / helper.name, helper)
        shutil.copyfile(BASE / 'static/branding/favicon.ico', self.area / 'app.ico')
        self.write(self.area / 'shell-owner.json', {'schema': 1, 'application': 'Forensic DGP'})
        self.write(self.area / 'release.json', {})
        for file in ('ForensicDGP.exe', 'gui_setup.ps1', 'distribution.ps1', 'app-source.zip',
                     'app-source.json', 'install_location.ps1', 'install_state.ps1', 'LICENSE.txt'):
            self.write(self.area / file, b'Fixture for uninstall testing. This file is never executed.')
        self.write(self.area / 'release.inc', (
            '#define ReleaseVersion "0.0.1"\n'
            '#define ReleaseCommit "' + '0' * 40 + '"\n'
            '#define ReleaseManifestSHA256 "' + '0' * 64 + '"\n'
            '#define UninstallSHA256 "' + hashlib.sha256(helper.read_bytes()).hexdigest() + '"\n'
        ).encode('ascii'))
        result = subprocess.run([str(COMPILER), '/DStageDir=' + str(self.area),
                                 '/DOutputDir=' + str(self.area), str(fixture)],
                                capture_output=True, text=True, timeout=45)
        self.assertEqual(result.returncode, 0, result.stdout + result.stderr)

    @staticmethod
    def write(path, body):
        path.parent.mkdir(parents=True, exist_ok=True)
        path.write_bytes(json.dumps(body).encode('utf-8') if isinstance(body, dict) else body)

    def clear_fixture_registration(self):
        try:
            winreg.DeleteKeyEx(winreg.HKEY_CURRENT_USER, self.registry_key, winreg.KEY_WOW64_64KEY)
        except FileNotFoundError:
            pass

    def assert_registered(self):
        with winreg.OpenKey(winreg.HKEY_CURRENT_USER, self.registry_key, 0,
                            winreg.KEY_READ | winreg.KEY_WOW64_64KEY) as key:
            path, _ = winreg.QueryValueEx(key, 'InstallLocation')
        self.assertEqual(Path(path), self.shell)

    def run_inno(self, executable, action, *args):
        path = self.area / (action + '.log')
        result = subprocess.run([str(executable), *SILENT_ARGS, '/LOG=' + str(path), *args],
                                cwd=self.area, capture_output=True, text=True, timeout=60)
        # Inno can return from its launcher while the temporary uninstall process
        # still owns the log and is finishing cleanup.
        log = ''
        deadline = time.monotonic() + 10
        while time.monotonic() < deadline:
            if path.exists():
                log = path.read_text(encoding='utf-8-sig', errors='replace')
                if 'Log closed.' in log:
                    break
            time.sleep(0.1)
        self.assertIn('Log closed.', log, result.stdout + result.stderr)
        return result, result.stdout + result.stderr + '\n' + log

    def assert_windows_integration_removed(self, log):
        with self.assertRaises(FileNotFoundError):
            self.assert_registered()
        self.assertFalse(self.shortcut.exists(), log)
        self.assertEqual(self.sibling.read_bytes(), b'outside the installation')

    def assert_removed(self, log):
        for _ in range(20):
            if not self.shell.exists():
                break
            time.sleep(0.1)
        self.assertFalse(self.shell.exists(), log + '\n' + repr(list(self.shell.glob('*'))))
        self.assert_windows_integration_removed(log)

    def test_uninstall_removes_installed_folder_without_following_junction(self):
        junction = self.runtime / 'data' / 'linked-outside'
        powershell = Path(os.environ['SystemRoot']) / 'System32/WindowsPowerShell/v1.0/powershell.exe'
        created = subprocess.run([str(powershell), '-NoProfile', '-NonInteractive', '-Command',
            "$ErrorActionPreference='Stop'; New-Item -ItemType Junction -Path $env:DGP_TEST_LINK "
            '-Target $env:DGP_TEST_OUTSIDE | Out-Null'], capture_output=True, text=True, timeout=15,
            env=dict(os.environ, DGP_TEST_LINK=str(junction), DGP_TEST_OUTSIDE=str(self.sibling.parent)))
        self.assertEqual(created.returncode, 0, created.stdout + created.stderr)
        try:
            result, log = self.run_inno(self.uninstaller, 'uninstall')
            self.assertEqual(result.returncode, 0, log)
            self.assert_removed(log)
        finally:
            if junction.is_junction():
                junction.rmdir()

    def test_uninstall_preserves_an_unrelated_file_in_the_selected_folder(self):
        personal = self.shell / 'personal-notes.txt'
        self.write(personal, b'not installed by DGP')
        result, log = self.run_inno(self.uninstaller, 'uninstall')
        self.assertEqual(result.returncode, 0, log)
        for _ in range(20):
            if list(self.shell.iterdir()) == [personal]:
                break
            time.sleep(0.1)
        self.assertEqual(list(self.shell.iterdir()), [personal], log)
        self.assertEqual(personal.read_bytes(), b'not installed by DGP')
        self.assert_windows_integration_removed(log)

    def test_locked_record_preserves_the_uninstaller_and_can_be_retried(self):
        kernel = ctypes.WinDLL('kernel32', use_last_error=True)
        open_file = kernel.CreateFileW
        open_file.argtypes = (wintypes.LPCWSTR, wintypes.DWORD, wintypes.DWORD, wintypes.LPVOID,
                              wintypes.DWORD, wintypes.DWORD, wintypes.HANDLE)
        open_file.restype = wintypes.HANDLE
        close_file = kernel.CloseHandle
        close_file.argtypes = (wintypes.HANDLE,)
        close_file.restype = wintypes.BOOL
        record = self.runtime / 'data/records/saved.json'
        # Allow other readers but deny delete sharing until the first attempt finishes.
        handle = open_file(str(record), 0x80000000, 1, None, 3, 0, None)
        if handle in (None, ctypes.c_void_p(-1).value):
            raise ctypes.WinError(ctypes.get_last_error())
        try:
            result, log = self.run_inno(self.uninstaller, 'locked')
            self.assertNotEqual(result.returncode, 0, log)
            self.assertEqual(record.read_bytes(), b'data/records/saved.json')
            self.assertTrue((self.runtime / 'owner.json').is_file(), log)
            self.assertTrue(self.uninstaller.is_file(), log)
            self.assert_registered()
        finally:
            close_file(handle)
        result, log = self.run_inno(self.uninstaller, 'retry')
        self.assertEqual(result.returncode, 0, log)
        self.assert_removed(log)

    def test_uninstall_removes_long_paths_and_readonly_files(self):
        deep_dir = self.runtime / 'environments' / ('a' * 60) / ('b' * 60) / ('c' * 60) / 'licenses'
        deep_file = deep_dir / ('long_license_notice_' + 'd' * 40 + '.txt')
        self.assertGreater(len(str(deep_file)), 280)
        long_dir = Path('\\\\?\\' + str(deep_dir))
        long_file = Path('\\\\?\\' + str(deep_file))
        long_dir.mkdir(parents=True, exist_ok=True)
        long_file.write_bytes(b'long path license content')
        readonly_file = self.runtime / 'environments' / 'capture' / 'readonly.txt'
        self.write(readonly_file, b'readonly content')
        os.chmod(readonly_file, 0o444)
        os.chmod(str(long_file), 0o444)
        result, log = self.run_inno(self.uninstaller, 'uninstall')
        self.assertEqual(result.returncode, 0, log)
        self.assert_removed(log)

    def test_uninstall_stops_running_app_processes_and_removes_folder(self):
        powershell = Path(os.environ['SystemRoot']) / 'System32/WindowsPowerShell/v1.0/powershell.exe'
        ident = hashlib.sha256(str(self.runtime.resolve()).lower().encode()).hexdigest()[:24].upper()
        mutex_name = 'Local\\DGP-app-' + ident
        work_dir = self.runtime / 'environments' / 'capture'
        script = (
            "$m = New-Object Threading.Mutex($false, $env:DGP_TEST_MUTEX); "
            "$m.WaitOne() | Out-Null; "
            "Write-Output 'ready'; [Console]::Out.Flush(); "
            "Start-Sleep -Seconds 60"
        )
        proc = subprocess.Popen(
            [str(powershell), '-NoProfile', '-NonInteractive', '-Command',
             f"# {work_dir}\n{script}"],
            cwd=work_dir,
            stdout=subprocess.PIPE,
            stderr=subprocess.PIPE,
            text=True,
            env=dict(os.environ, DGP_TEST_MUTEX=mutex_name),
        )
        try:
            self.assertEqual(proc.stdout.readline().strip(), 'ready')
            result, log = self.run_inno(self.uninstaller, 'uninstall')
            self.assertEqual(result.returncode, 0, log)
            self.assertIsNotNone(proc.wait(timeout=10))
            self.assert_removed(log)
        finally:
            if proc.poll() is None:
                proc.kill()
                proc.wait(timeout=5)
            if proc.stdout:
                proc.stdout.close()
            if proc.stderr:
                proc.stderr.close()


if __name__ == '__main__':
    unittest.main()
