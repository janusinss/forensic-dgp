"""Online EXE backend boundary tests using inert local payloads."""
import contextlib
import io
import hashlib
import json
import os
from pathlib import Path
import subprocess
import tempfile
import unittest
import zipfile
from unittest.mock import patch
from implementation import environment

BASE = Path(__file__).resolve().parents[1]
PS = Path(os.environ.get('SystemRoot', r'C:\Windows')) / 'System32/WindowsPowerShell/v1.0/powershell.exe'
COMMIT = 'a' * 40
PRIOR = 'b' * 40


def blob(data):
    return hashlib.sha1(b'blob ' + str(len(data)).encode() + b'\0' + data).hexdigest()


@unittest.skipUnless(PS.is_file(), 'PowerShell 5 required')
class InnoBundleTests(unittest.TestCase):
    def setUp(self):
        self.temp = tempfile.TemporaryDirectory(dir=BASE)
        self.addCleanup(self.temp.cleanup)
        self.local = Path(self.temp.name)
        self.root = self.local / 'DGP/app'
        self.zip = self.local / 'source.zip'
        self.manifest = self.local / 'manifest.json'
        self.events = self.local / 'events.jsonl'
        self.cancel = self.local / 'cancel'
        files = {
            'start.bat': b'@echo off\r\nexit /b 0\r\n',
            'implementation/install_source.bat': b'@echo off\r\nif defined DGP_TEST_CANCEL type nul > "%DGP_SETUP_CANCEL_FILE%"\r\nif defined DGP_TEST_FAIL exit /b 17\r\nexit /b 0\r\n',
            'data.txt': b'fixture source\n',
        }
        with zipfile.ZipFile(self.zip, 'w') as z:
            for name, data in files.items():
                z.writestr('forensic-dgp-' + COMMIT + '/' + name, data)
        self.doc = {'schema': 1, 'commit': COMMIT,
                    'files': {name: blob(data) for name, data in files.items()},
                    'source_zip_sha256': hashlib.sha256(self.zip.read_bytes()).hexdigest(),
                    'source_zip_size': self.zip.stat().st_size, 'ancestors': [COMMIT, PRIOR]}
        self.manifest.write_text(json.dumps(self.doc), encoding='utf8')
        self.env = dict(os.environ, LOCALAPPDATA=str(self.local),
                        DGP_SETUP_EVENTS=str(self.events), DGP_SETUP_CANCEL_FILE=str(self.cancel))

    def run_install(self, **overrides):
        return subprocess.run([str(PS), '-NoProfile', '-NonInteractive', '-ExecutionPolicy', 'Bypass',
            '-File', str(BASE / 'implementation/distribution.ps1'), '-Action', 'Install',
            '-ManagedRoot', str(self.root), '-Commit', COMMIT,
            '-BundlePath', str(self.zip), '-BundleManifest', str(self.manifest)],
            env=dict(self.env, **overrides), capture_output=True, text=True, timeout=30)

    def prior(self, commit=PRIOR):
        self.root.mkdir(parents=True)
        (self.root/'owner.json').write_text('{"schema_version":1,"repository":"janusinss/forensic-dgp"}')
        pointer=self.root/'current.json'
        pointer.write_text(json.dumps({'schema':1, 'commit':commit}))
        return pointer, pointer.read_bytes()

    def test_local_bundle_installs_without_github_and_reports_activation(self):
        result=self.run_install()
        self.assertEqual(result.returncode, 0, result.stdout+result.stderr)
        self.assertIn('DGP_EVENT ', result.stdout)
        self.assertEqual(json.loads((self.root/'current.json').read_text(encoding='utf-8-sig'))['commit'], COMMIT)
        events=[json.loads(line) for line in self.events.read_text(encoding='utf-8-sig').splitlines()]
        self.assertEqual(events[-1]['stage'], 'complete')
        self.assertTrue(any(event['stage']=='verification' for event in events))

    def test_tampered_zip_never_activates(self):
        pointer, prior=self.prior()
        with self.zip.open('ab') as stream: stream.write(b'bad')
        result=self.run_install()
        self.assertNotEqual(result.returncode, 0)
        self.assertEqual(pointer.read_bytes(), prior)
        self.assertFalse((self.root/'versions'/COMMIT).exists())

    def test_cancel_during_setup_preserves_active_pointer(self):
        pointer, prior=self.prior()
        result=self.run_install(DGP_TEST_CANCEL='1')
        self.assertNotEqual(result.returncode, 0)
        self.assertEqual(pointer.read_bytes(), prior)
        self.assertNotIn('complete', [json.loads(line)['stage'] for line in self.events.read_text(encoding='utf-8-sig').splitlines()])

    def test_unrelated_newer_version_is_refused(self):
        pointer, prior=self.prior('c'*40)
        result=self.run_install()
        self.assertNotEqual(result.returncode, 0)
        self.assertEqual(pointer.read_bytes(), prior)
        self.assertFalse((self.root/'versions'/COMMIT).exists())

    def test_preexisting_cancel_never_activates(self):
        self.cancel.write_bytes(b'cancel')
        result=self.run_install()
        self.assertNotEqual(result.returncode, 0)
        self.assertFalse((self.root/'current.json').exists())

    def test_unrecognized_directory_is_untouched(self):
        self.root.mkdir(parents=True)
        saved=self.root/'personal.txt'; saved.write_bytes(b'user data')
        result=self.run_install()
        self.assertNotEqual(result.returncode, 0)
        self.assertEqual(saved.read_bytes(), b'user data')
        self.assertFalse((self.root/'owner.json').exists())


class SharedEnvironmentTests(unittest.TestCase):
    def test_existing_managed_runtime_is_verified_without_package_mutation(self):
        with tempfile.TemporaryDirectory() as temp:
            root=Path(temp); runtime=root/'runtime'; (runtime/'Scripts').mkdir(parents=True)
            python=runtime/'Scripts/python.exe'; python.write_bytes(b'fixture')
            record={'environment_root':str(runtime),'python':str(python)}
            with patch.object(environment,'managed_root',return_value=root), \
                 patch.object(environment,'expected_record',return_value=record), \
                 patch.object(environment,'verify_runtime',return_value=0) as verify, \
                 patch.object(environment,'install_dependencies',side_effect=AssertionError('must not mutate shared packages')), \
                 patch.object(environment,'install_capture',side_effect=AssertionError('must not mutate shared capture')), \
                 patch.object(environment,'install_models',return_value=0), \
                 patch.object(environment,'write_record') as publish:
                self.assertEqual(environment.install(),0)
            verify.assert_called_once_with(python, include_models=False)
            publish.assert_called_once_with(record)

    def test_failed_shared_verification_refuses_repair_and_record_publication(self):
        with tempfile.TemporaryDirectory() as temp:
            root=Path(temp); runtime=root/'runtime'; runtime.mkdir()
            python=runtime/'python.exe'; python.write_bytes(b'unchanged')
            with patch.object(environment,'managed_root',return_value=root), \
                 patch.object(environment,'expected_record',return_value={'environment_root':str(runtime),'python':str(python)}), \
                 patch.object(environment,'verify_runtime',return_value=19), \
                 patch.object(environment,'install_dependencies') as install, \
                 patch.object(environment,'write_record') as publish:
                with self.assertRaises(environment.EnvironmentError): environment.install()
            install.assert_not_called(); publish.assert_not_called()
            self.assertEqual(python.read_bytes(),b'unchanged')


class SetupControlTests(unittest.TestCase):
    def test_cancellation_after_shared_model_task_keeps_runtime_and_record(self):
        with tempfile.TemporaryDirectory() as temp:
            root=Path(temp); runtime=root/'runtime'; runtime.mkdir()
            python=runtime/'python.exe'; python.write_bytes(b'old working runtime')
            cancel=root/'cancel'; record=root/'record.json'; record.write_bytes(b'old record')
            def models():
                cancel.write_bytes(b'cancel')
                return 0
            with patch.object(environment,'managed_root',return_value=root), \
                 patch.object(environment,'expected_record',return_value={'environment_root':str(runtime),'python':str(python)}), \
                 patch.object(environment,'verify_runtime',return_value=0), \
                 patch.object(environment,'install_models',side_effect=models), \
                 patch.object(environment,'install_dependencies',side_effect=AssertionError('shared runtime mutation')), \
                 patch.object(environment,'write_record') as publish, \
                 patch.dict(os.environ,{'DGP_SETUP_CANCEL_FILE':str(cancel)}):
                self.assertNotEqual(environment.install(),0)
            publish.assert_not_called()
            self.assertEqual(python.read_bytes(),b'old working runtime')
            self.assertEqual(record.read_bytes(),b'old record')

    def test_relative_event_paths_are_refused(self):
        from implementation import setup_events
        with patch.dict(os.environ,{'DGP_SETUP_EVENTS':'relative.jsonl'}):
            with self.assertRaises(RuntimeError): setup_events.event('fixture','running','fixture')

    def test_progress_events_are_utf8_json_lines(self):
        from implementation import setup_events
        with tempfile.TemporaryDirectory() as temp:
            events=Path(temp)/'events.jsonl'
            output=io.StringIO()
            with patch.dict(os.environ,{'DGP_SETUP_EVENTS':str(events)}), contextlib.redirect_stdout(output):
                setup_events.event('models','progress','fixture',current=25,total=100)
            self.assertTrue(output.getvalue().startswith('DGP_EVENT '))
            self.assertEqual(json.loads(output.getvalue()[len('DGP_EVENT '):]), json.loads(events.read_text(encoding='utf8')))
            record=json.loads(events.read_text(encoding='utf8').strip())
            self.assertEqual((record['current'],record['total']),(25,100))
            self.assertEqual(record['schema'],1)


@unittest.skipUnless(os.name == 'nt', 'Windows sharing modes required')
class WindowsProgressSharingTests(unittest.TestCase):
    def deny_writer(self, path):
        import ctypes
        from ctypes import wintypes
        kernel=ctypes.WinDLL('kernel32',use_last_error=True)
        kernel.CreateFileW.argtypes=[wintypes.LPCWSTR,wintypes.DWORD,wintypes.DWORD,ctypes.c_void_p,wintypes.DWORD,wintypes.DWORD,wintypes.HANDLE]
        kernel.CreateFileW.restype=wintypes.HANDLE
        kernel.CloseHandle.argtypes=[wintypes.HANDLE]; kernel.CloseHandle.restype=wintypes.BOOL
        handle=kernel.CreateFileW(str(path),0x80000000,1,None,3,128,None)
        self.assertNotEqual(handle,ctypes.c_void_p(-1).value)
        return kernel,handle

    def test_transient_reader_deny_write_retries_until_reader_closes(self):
        from implementation import setup_events
        import threading
        import time
        with tempfile.TemporaryDirectory() as temp:
            path=Path(temp)/'events.jsonl'; path.write_bytes(b'')
            kernel,handle=self.deny_writer(path)
            timer=threading.Timer(0.2,lambda: kernel.CloseHandle(handle)); timer.start()
            try:
                start=time.monotonic()
                with patch.dict(os.environ,{'DGP_SETUP_EVENTS':str(path)}), contextlib.redirect_stdout(io.StringIO()):
                    setup_events.event('fixture','running','after reader')
                self.assertGreaterEqual(time.monotonic()-start,0.15)
                self.assertLess(time.monotonic()-start,2)
            finally:
                timer.join()
            self.assertEqual(json.loads(path.read_text())['message'],'after reader')

    def test_permanent_reader_deny_write_fails_after_bounded_retry(self):
        from implementation import setup_events
        import time
        with tempfile.TemporaryDirectory() as temp:
            path=Path(temp)/'events.jsonl'; path.write_bytes(b'')
            kernel,handle=self.deny_writer(path)
            try:
                start=time.monotonic()
                with patch.dict(os.environ,{'DGP_SETUP_EVENTS':str(path)}), contextlib.redirect_stdout(io.StringIO()):
                    with self.assertRaises(OSError): setup_events.event('fixture','running','must fail')
                self.assertLess(time.monotonic()-start,2.8)
                self.assertGreaterEqual(time.monotonic()-start,1.8)
            finally:
                kernel.CloseHandle(handle)
            self.assertEqual(path.read_bytes(),b'')

    def test_readonly_file_access_denial_is_not_hidden_or_retried(self):
        from implementation import setup_events
        import stat
        import time
        with tempfile.TemporaryDirectory() as temp:
            path=Path(temp)/'events.jsonl'; path.write_bytes(b'')
            os.chmod(path,stat.S_IREAD)
            try:
                start=time.monotonic()
                with patch.dict(os.environ,{'DGP_SETUP_EVENTS':str(path)}), contextlib.redirect_stdout(io.StringIO()):
                    with self.assertRaises(PermissionError): setup_events.event('fixture','running','must fail')
                self.assertLess(time.monotonic()-start,0.5)
            finally:
                os.chmod(path,stat.S_IREAD|stat.S_IWRITE)
