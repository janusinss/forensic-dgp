"""Contract checks for the owned desktop launch handshake."""
import json
import os
from pathlib import Path
import tempfile
import unittest
from unittest import mock

from implementation.desktop import DesktopError, DesktopSession


class DesktopContractTests(unittest.TestCase):
    def setUp(self):
        self.token = 'a' * 32
        self.folder = Path(tempfile.gettempdir()) / ('ForensicDGP-' + self.token)
        self.folder.mkdir(exist_ok=False)
        self.ready = self.folder / 'ready.json'
        self.env = {'DGP_DESKTOP_READY_FILE': str(self.ready),
                    'DGP_DESKTOP_STOP_EVENT': 'Local\\ForensicDGP-stop-' + self.token}

    def tearDown(self):
        self.ready.unlink(missing_ok=True)
        self.folder.rmdir()

    def session(self, env=None):
        return DesktopSession.from_environment(self.env if env is None else env, open_event=lambda _: 123)

    def test_no_desktop_environment_is_not_a_desktop_session(self):
        self.assertIsNone(DesktopSession.from_environment({}, open_event=lambda _: 123))

    def test_partial_contract_refused(self):
        for key in self.env:
            with self.assertRaises(DesktopError):
                self.session({key: self.env[key]})

    def test_event_and_folder_must_have_same_unguessable_token(self):
        altered = dict(self.env, DGP_DESKTOP_STOP_EVENT='Local\\ForensicDGP-stop-' + 'b'*32)
        with self.assertRaises(DesktopError):
            self.session(altered)

    def test_ready_file_cannot_overwrite_an_existing_record(self):
        self.ready.write_text('preserve', encoding='utf8')
        with self.assertRaises(DesktopError):
            self.session()
        self.assertEqual(self.ready.read_text(), 'preserve')

    def test_path_outside_private_handshake_directory_refused(self):
        with self.assertRaises(DesktopError):
            self.session(dict(self.env, DGP_DESKTOP_READY_FILE=str(self.folder / '..' / 'ready.json')))

    def test_ready_record_contains_actual_loopback_url_and_token(self):
        session = self.session()
        session.publish('http://127.0.0.1:53719', 1234)
        record = json.loads(self.ready.read_text(encoding='utf8'))
        self.assertEqual(record, {'schema':1, 'url':'http://127.0.0.1:53719', 'pid':1234, 'token':self.token})

    def test_non_loopback_or_invalid_port_refused(self):
        for url in ('http://evil.test:8000', 'http://127.0.0.1:0', 'http://127.0.0.1:8000/path', 'https://127.0.0.1:8000'):
            with self.subTest(url=url), self.assertRaises(DesktopError):
                self.session().publish(url, 123)

    def test_stop_event_requests_graceful_server_exit(self):
        session = self.session()
        server = mock.Mock(should_exit=False)
        session.watch(server, wait_event=lambda handle, millis: 0, close_event=lambda handle: None)
        session.thread.join(timeout=1)
        self.assertTrue(server.should_exit)
        session.close()

    @unittest.skipUnless(os.name == 'nt', 'Windows named-event integration')
    def test_real_windows_event_reaches_graceful_shutdown(self):
        import ctypes
        from ctypes import wintypes
        kernel = ctypes.WinDLL('kernel32', use_last_error=True)
        kernel.CreateEventW.argtypes = [ctypes.c_void_p, wintypes.BOOL, wintypes.BOOL, wintypes.LPCWSTR]
        kernel.CreateEventW.restype = wintypes.HANDLE
        kernel.SetEvent.argtypes = [wintypes.HANDLE]
        kernel.CloseHandle.argtypes = [wintypes.HANDLE]
        host = kernel.CreateEventW(None, True, False, self.env['DGP_DESKTOP_STOP_EVENT'])
        self.assertTrue(host)
        try:
            session = DesktopSession.from_environment(self.env)
            server = mock.Mock(should_exit=False)
            session.watch(server)
            kernel.SetEvent(host)
            session.thread.join(timeout=2)
            self.assertTrue(server.should_exit)
            self.assertFalse(session.failed)
            session.close()
        finally:
            kernel.CloseHandle(host)

    def test_launch_failure_closes_desktop_session(self):
        from implementation import launch
        session = mock.Mock()
        with mock.patch.object(launch.DesktopSession, 'from_environment', return_value=session), \
             mock.patch.object(launch, 'verify_models', side_effect=launch.ModelSetupError('fixture missing model')), \
             mock.patch('sys.stderr'):
            self.assertEqual(launch.main(['--desktop']), 1)
        session.close.assert_called_once()

    def test_wait_failure_is_reported_and_requests_exit(self):
        session = self.session()
        server = mock.Mock(should_exit=False)
        with mock.patch('sys.stderr') as error:
            session.watch(server, wait_event=lambda handle, millis: 0xffffffff, close_event=lambda handle: None)
            session.thread.join(timeout=1)
            self.assertTrue(server.should_exit)
            self.assertTrue(session.failed)
            self.assertIn('Desktop stop event failed', str(error.write.call_args_list))
        session.close()


if __name__ == '__main__':
    unittest.main()
