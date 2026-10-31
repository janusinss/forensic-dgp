"""Wizard safety contracts and real PowerShell wrapper execution with inert payloads.

Pascal checks protect reviewed wiring. They do not replace compilation or tests
of the built wizard; the wrapper tests execute the actual shipped PowerShell.
"""
import hashlib
import json
import os
from pathlib import Path
import re
import subprocess
import tempfile
import unittest

BASE = Path(__file__).resolve().parents[1]
WIZARD = BASE / 'packaging' / 'installer.iss'
WRAPPER = BASE / 'packaging' / 'gui_setup.ps1'
PS = Path(os.environ.get('SystemRoot', r'C:\Windows')) / 'System32/WindowsPowerShell/v1.0/powershell.exe'
COMMIT = 'a' * 40


def sha(value):
    return hashlib.sha256(value).hexdigest()


class WizardSafetyContracts(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        cls.code = WIZARD.read_text(encoding='utf8')

    def section(self, start, end):
        return self.code.split(start, 1)[1].split(end, 1)[0]

    def test_setup_waits_for_actual_exit_code_without_assumed_pid(self):
        prepare = self.section('function PrepareToInstall', '\nvar\n  CheckOutput:')
        self.assertIn('ExecAndLogOutput(PS, Params', prepare)
        self.assertIn('ewWaitUntilTerminated, ResultCode, @SetupOutput)', prepare)
        self.assertNotIn('ewNoWait', prepare)
        self.assertNotIn('OpenProcess', prepare)
        callback = self.section('procedure SetupOutput', 'procedure InitializeWizard')
        self.assertIn('Log(S)', callback)
        self.assertIn('SaveStringToFile(SetupLog, S + #13#10, True)', callback)

    def test_zero_exit_is_authoritative_over_late_cancel_request(self):
        prepare = self.section('function PrepareToInstall', '\nvar\n  CheckOutput:')
        tail = prepare.split("UpdateProgress('');", 1)[1]
        self.assertRegex(tail, r'if ResultCode = 0 then BackendReady := True\s+else Result :=')
        self.assertNotIn('if CancelRequested', tail)
        self.assertNotIn('prior version and saved records remain', tail)

    def test_cancel_stops_at_activation_boundary(self):
        callback = self.section('procedure CancelButtonClick', 'function PrepareToInstall')
        guard = "if (LastStage = 'activation') or (LastStage = 'complete') then Exit;"
        self.assertIn(guard, callback)
        self.assertLess(callback.index(guard), callback.index('SaveStringToFile(CancelFile'))
        progress = self.section('procedure UpdateProgress', 'procedure SetupOutput')
        self.assertIn("(LastStage = 'activation') or (LastStage = 'complete')", progress)
        self.assertIn('WizardForm.CancelButton.Enabled := False', progress)

    def test_shell_preflight_covers_ancestors_and_write_boundaries(self):
        ordinary = self.section('function OrdinaryShellPath', 'function Q(')
        self.assertIn('GetFileAttributes(Path)', ordinary)
        self.assertIn('(Attr and $400)', ordinary)  # FILE_ATTRIBUTE_REPARSE_POINT
        self.assertIn('(Attr and $10)', ordinary)   # FILE_ATTRIBUTE_DIRECTORY
        self.assertIn('Parent := ExtractFileDir(Path)', ordinary)
        prepare = self.section('function PrepareToInstall', '\nvar\n  CheckOutput:')
        self.assertLess(prepare.index('if not OrdinaryShellPath'), prepare.index('ExtractTemporaryFile'))
        uninstall = self.section('function InitializeUninstall', 'procedure CurStepChanged')
        self.assertLess(uninstall.index('if not OrdinaryShellPath'), uninstall.index('ExecAndLogOutput'))
        final_write = self.section('procedure CurStepChanged', 'procedure DeinitializeUninstall')
        self.assertIn('(CurStep = ssInstall) and not OrdinaryShellPath', final_write)
        self.assertIn('RaiseException(', final_write)

    def test_selected_user_path_and_trusted_uninstall_helper(self):
        self.assertIn('PrivilegesRequired=lowest', self.code)
        self.assertIn('DisableDirPage=no', self.code)
        self.assertIn('UsePreviousAppDir=yes', self.code)
        prepare = self.section('function PrepareToInstall', '\nvar\n  CheckOutput:')
        self.assertIn("' -InstallRoot ' + Q(ExpandConstant('{app}'))", prepare)
        self.assertIn("ExtractTemporaryFile('install_location.ps1')", prepare)
        self.assertIn("PreviousInstallRoot := RegisteredInstallRoot", self.code)
        self.assertIn("PreviousInstallRoot) <> 0", prepare)
        uninstall = self.section('function InitializeUninstall', 'procedure CurStepChanged')
        self.assertIn("GetSHA256OfFile(ExpandConstant('{app}\\uninstall.ps1'))", uninstall)
        self.assertLess(uninstall.index('GetSHA256OfFile'), uninstall.index('ExecAndLogOutput'))
        self.assertIn("'{#UninstallSHA256}'", uninstall)
        self.assertIn('WaitForSingleObject(UninstallMutex, 0)', uninstall)


@unittest.skipUnless(PS.is_file(), 'Windows PowerShell is required')
class GuiSetupWrapper(unittest.TestCase):
    def setUp(self):
        (BASE / 'scratch').mkdir(exist_ok=True)
        self.temp = tempfile.TemporaryDirectory(prefix='wizard-', dir=BASE / 'scratch')
        self.addCleanup(self.temp.cleanup)
        self.stage = Path(self.temp.name)
        self.install_root = self.stage / 'Selected App'
        self.called = self.stage / 'backend-called.json'
        self.events = self.stage / 'events.jsonl'
        self.cancel = self.stage / 'cancel-request'
        self.log = self.stage / 'setup.log'
        self.env = dict(os.environ, LOCALAPPDATA=str(self.stage / 'user'),
                        DGP_TEST_CALLED=str(self.called), DGP_BOOTSTRAP_FILE='must be cleared')
        # Mandatory ManagedRoot catches a missing wrapper argument without any
        # access to the real installation. This stub never installs anything.
        script = r'''param(
  [string]$Action,
  [Parameter(Mandatory=$true)][string]$ManagedRoot,
  [string]$Commit,
  [string]$BundlePath,
  [string]$BundleManifest
)
$record = @{action=$Action;root=$ManagedRoot;commit=$Commit;bundle=$BundlePath;manifest=$BundleManifest;
            events=$env:DGP_SETUP_EVENTS;cancel=$env:DGP_SETUP_CANCEL_FILE;
            no_pause=$env:IMPLEMENTATION_NO_PAUSE;bootstrap_present=(Test-Path Env:\DGP_BOOTSTRAP_FILE);
            install_root=$env:DGP_INSTALL_ROOT;config_present_at_backend=(Test-Path -LiteralPath (Join-Path $env:DGP_INSTALL_ROOT 'install-location.json'))}
[IO.File]::WriteAllText($env:DGP_TEST_CALLED,($record|ConvertTo-Json -Compress))
Write-Output 'inert backend output'
if ($env:DGP_TEST_NATIVE_STDERR -eq '1') { & $env:ComSpec /d /c 'echo harmless native diagnostic 1>&2' }
if ($env:DGP_TEST_LATE_CANCEL -eq '1') {
  [IO.File]::WriteAllText($env:DGP_SETUP_CANCEL_FILE,'late cancellation')
  [IO.File]::WriteAllText($env:DGP_SETUP_EVENTS,'{"stage":"complete","status":"complete"}')
}
if ($env:DGP_TEST_FAIL -eq '1') { exit 17 }
exit 0
'''
        # Trusted inert helper maps the selected folder to its private runtime.
        # It never invokes the production location validator or touches user paths.
        location_script = r'''param([string]$Action,[Parameter(Mandatory=$true)][string]$InstallRoot)
if ($Action -notin @('Prepare','Validate')) { throw 'Unexpected location action' }
if ($Action -eq 'Validate' -and $env:DGP_TEST_LOCATION_CHANGE -eq '1') { throw 'Location changed at final validation' }
[IO.Directory]::CreateDirectory($InstallRoot) | Out-Null
@{schema=1;install_root=$InstallRoot;managed_root=(Join-Path $InstallRoot 'runtime')} | ConvertTo-Json -Compress | Write-Output
exit 0
'''
        self.inputs = {
            'app-source.zip': b'inert zip fixture; never extracted',
            'app-source.json': b'{"inert":true}',
            'distribution.ps1': script.encode('utf8'),
            'uninstall.ps1': b'# inert fixture',
            'gui_setup.ps1': WRAPPER.read_bytes(),
            'ForensicDGP.exe': b'inert executable; never started',
            'install_location.ps1': location_script.encode('utf8'),
        }
        for name, value in self.inputs.items():
            (self.stage / name).write_bytes(value)
        self.manifest = {'schema': 1, 'commit': COMMIT,
                         'files': {name: sha(value) for name, value in self.inputs.items()}}
        self.save_manifest()

    def save_manifest(self):
        path = self.stage / 'release.json'
        path.write_text(json.dumps(self.manifest), encoding='utf8')
        self.expected = sha(path.read_bytes())

    def invoke(self, env=None, expected=None):
        return subprocess.run([str(PS), '-NoProfile', '-NonInteractive', '-ExecutionPolicy', 'Bypass',
                               '-File', str(self.stage / 'gui_setup.ps1'), '-Stage', str(self.stage),
                               '-InstallRoot', str(self.install_root),
                               '-EventsFile', str(self.events), '-CancelFile', str(self.cancel),
                               '-LogPath', str(self.log), '-ExpectedManifestHash', expected or self.expected],
                              capture_output=True, text=True, timeout=15, env=env or self.env)

    def test_verified_inputs_run_stub_with_pinned_source_and_event_contract(self):
        result = self.invoke()
        self.assertEqual(result.returncode, 0, result.stdout + result.stderr)
        called = json.loads(self.called.read_text(encoding='utf8'))
        self.assertEqual(called['action'], 'Install')
        self.assertEqual(Path(called['root']), self.install_root / 'runtime')
        self.assertEqual(Path(called['install_root']), self.install_root)
        self.assertEqual(called['commit'], COMMIT)
        self.assertEqual(Path(called['bundle']), self.stage / 'app-source.zip')
        self.assertEqual(Path(called['manifest']), self.stage / 'app-source.json')
        self.assertEqual(Path(called['events']), self.events)
        self.assertEqual(Path(called['cancel']), self.cancel)
        self.assertEqual(called['no_pause'], '1')
        self.assertFalse(called['bootstrap_present'])
        self.assertIn('inert backend output', result.stdout)
        # Inno's SetupOutput captures stdout/stderr and owns the persistent log.
        # The wrapper must forward output without PS stream-redirection coercion.
        self.assertNotIn('Tee-Object', (self.stage / 'gui_setup.ps1').read_text(encoding='utf8'))

    def test_bound_configuration_is_written_only_after_backend_success(self):
        result = self.invoke()
        self.assertEqual(result.returncode, 0, result.stdout + result.stderr)
        configuration = json.loads((self.install_root / 'install-location.json').read_text(encoding='utf8'))
        self.assertEqual(configuration, {'schema': 1, 'install_root': str(self.install_root),
                                        'managed_root': str(self.install_root / 'runtime')})
        called = json.loads(self.called.read_text(encoding='utf8'))
        self.assertFalse(called['config_present_at_backend'])

    def test_failed_backend_does_not_publish_location_configuration(self):
        result = self.invoke(env=dict(self.env, DGP_TEST_FAIL='1'))
        self.assertNotEqual(result.returncode, 0)
        self.assertTrue(self.called.is_file(), result.stdout + result.stderr)
        self.assertFalse((self.install_root / 'install-location.json').exists())

    def test_changed_location_after_backend_does_not_publish_configuration(self):
        result = self.invoke(env=dict(self.env, DGP_TEST_LOCATION_CHANGE='1'))
        self.assertNotEqual(result.returncode, 0)
        self.assertTrue(self.called.is_file())
        self.assertFalse((self.install_root / 'install-location.json').exists())
        self.assertIn('Location changed at final validation', ' '.join((result.stdout + result.stderr).split()))

    def test_backend_failure_does_not_become_wrapper_success(self):
        result = self.invoke(env=dict(self.env, DGP_TEST_FAIL='1'))
        self.assertNotEqual(result.returncode, 0, result.stdout + result.stderr)
        self.assertTrue(self.called.is_file())
        self.assertIn('Application preparation failed', ' '.join(result.stderr.split()))
        self.assertIn('Application preparation failed', self.log.read_text(encoding='utf-8-sig'))

    def test_native_stderr_with_zero_exit_is_not_a_setup_failure(self):
        result = self.invoke(env=dict(self.env, DGP_TEST_NATIVE_STDERR='1'))
        self.assertEqual(result.returncode, 0, result.stdout + result.stderr)
        self.assertTrue(self.called.is_file())
        self.assertIn('harmless native diagnostic', result.stdout + result.stderr)

    def test_late_cancellation_cannot_override_successful_backend_exit(self):
        result = self.invoke(env=dict(self.env, DGP_TEST_LATE_CANCEL='1'))
        self.assertEqual(result.returncode, 0, result.stdout + result.stderr)
        self.assertTrue(self.cancel.is_file())
        self.assertEqual(json.loads(self.events.read_text())['stage'], 'complete')

    def test_untrusted_release_manifest_prevents_backend_execution(self):
        result = self.invoke(expected='0' * 64)
        self.assertNotEqual(result.returncode, 0)
        self.assertFalse(self.called.exists())
        self.assertIn('Embedded release manifest differs', ' '.join(result.stderr.split()))

    def test_changed_embedded_backend_is_refused_before_execution(self):
        (self.stage / 'distribution.ps1').write_bytes(self.inputs['distribution.ps1'] + b'\n# changed')
        result = self.invoke()
        self.assertNotEqual(result.returncode, 0)
        self.assertFalse(self.called.exists())
        self.assertIn('Embedded release input differs: distribution.ps1', ' '.join(result.stderr.split()))

    def test_incomplete_manifest_prevents_backend_execution(self):
        del self.manifest['files']['uninstall.ps1']
        self.save_manifest()
        result = self.invoke()
        self.assertNotEqual(result.returncode, 0)
        self.assertFalse(self.called.exists())
        self.assertIn('Incomplete release inputs', ' '.join(result.stderr.split()))

    def test_unexpected_manifest_input_prevents_backend_execution(self):
        self.manifest['files']['../outside.ps1'] = '0' * 64
        self.save_manifest()
        result = self.invoke()
        self.assertNotEqual(result.returncode, 0)
        self.assertFalse(self.called.exists())
        self.assertIn('Unexpected release input', ' '.join(result.stderr.split()))

    def test_invalid_source_identity_prevents_backend_execution(self):
        self.manifest['commit'] = 'IMPLEMENTATION'
        self.save_manifest()
        result = self.invoke()
        self.assertNotEqual(result.returncode, 0)
        self.assertFalse(self.called.exists())
        self.assertIn('Invalid embedded release identity', ' '.join(result.stderr.split()))


if __name__ == '__main__':
    unittest.main()