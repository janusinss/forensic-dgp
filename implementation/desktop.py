"""Private Windows desktop-host handshake; no public HTTP shutdown endpoint."""
from __future__ import annotations

import os
from pathlib import Path
import json
import re
import sys
import tempfile
import threading
from urllib.parse import urlsplit


class DesktopError(RuntimeError):
    """Desktop host readiness or process-control contract is invalid."""


def _ordinary(path: Path, directory: bool) -> None:
    import stat
    info = path.lstat()
    if (stat.S_ISLNK(info.st_mode) or getattr(info, 'st_file_attributes', 0) & 1024
            or (not stat.S_ISDIR(info.st_mode) if directory else not stat.S_ISREG(info.st_mode))):
        raise DesktopError('Desktop handshake cannot contain linked or unexpected paths.')


def _windows_event(name):
    if sys.platform != 'win32':
        raise DesktopError('Desktop mode requires Windows.')
    import ctypes
    from ctypes import wintypes
    kernel = ctypes.WinDLL('kernel32', use_last_error=True)
    kernel.OpenEventW.argtypes = [wintypes.DWORD, wintypes.BOOL, wintypes.LPCWSTR]
    kernel.OpenEventW.restype = wintypes.HANDLE
    handle = kernel.OpenEventW(0x00100000, False, name)
    if not handle:
        raise DesktopError('Desktop host stop event is unavailable.')
    return handle


def _wait_event(handle, milliseconds):
    import ctypes
    from ctypes import wintypes
    kernel = ctypes.WinDLL('kernel32', use_last_error=True)
    kernel.WaitForSingleObject.argtypes = [wintypes.HANDLE, wintypes.DWORD]
    kernel.WaitForSingleObject.restype = wintypes.DWORD
    return kernel.WaitForSingleObject(handle, milliseconds)


def _close_event(handle):
    import ctypes
    from ctypes import wintypes
    kernel = ctypes.WinDLL('kernel32', use_last_error=True)
    kernel.CloseHandle.argtypes = [wintypes.HANDLE]
    kernel.CloseHandle.restype = wintypes.BOOL
    if not kernel.CloseHandle(handle):
        print('[ERROR] Desktop stop event handle could not be closed.', file=sys.stderr, flush=True)


class DesktopSession:
    def __init__(self, ready, token, handle):
        self.ready = ready
        self.token = token
        self.handle = handle
        self.finished = threading.Event()
        self.thread = None
        self.failed = False

    @classmethod
    def from_environment(cls, environ=None, *, open_event=_windows_event):
        env = os.environ if environ is None else environ
        ready, event = env.get('DGP_DESKTOP_READY_FILE'), env.get('DGP_DESKTOP_STOP_EVENT')
        if ready is None and event is None:
            return None
        if not ready or not event:
            raise DesktopError('Desktop host requires both readiness and stop-event values.')
        match = re.fullmatch(r'Local\\ForensicDGP-stop-([0-9a-f]{32})', event)
        if not match:
            raise DesktopError('Desktop host event name is invalid.')
        token = match.group(1)
        expected = Path(tempfile.gettempdir()) / ('ForensicDGP-' + token) / 'ready.json'
        path = Path(ready)
        if (not path.is_absolute() or '..' in path.parts
                or os.path.normcase(path.name) != os.path.normcase(expected.name)
                or os.path.normcase(path.parent.name) != os.path.normcase(expected.parent.name)):
            raise DesktopError('Desktop readiness path does not match the host session.')
        try:
            for parent in dict.fromkeys((path.parent, *path.parent.parents, expected.parent, *expected.parent.parents)):
                _ordinary(parent, True)
            # Windows can expose the same temporary directory with either an
            # 8.3 alias or its long name. Compare filesystem identity after
            # checking both paths for links and reparse points.
            if path.parent != expected.parent and not path.parent.samefile(expected.parent):
                raise DesktopError('Desktop readiness path does not match the host session.')
            if path.exists() or path.is_symlink():
                raise DesktopError('Existing desktop readiness record was preserved.')
            handle = open_event(event)
        except (OSError, ValueError) as exc:
            raise DesktopError('Desktop readiness directory or event is unavailable.') from exc
        return cls(path, token, handle)

    def publish(self, url, pid):
        parsed = urlsplit(url)
        try:
            valid = (parsed.scheme == 'http' and parsed.hostname == '127.0.0.1'
                     and parsed.port is not None and 1 <= parsed.port <= 65535
                     and not parsed.path and not parsed.query and not parsed.fragment
                     and not parsed.username and not parsed.password
                     and type(pid) is int and pid > 0)
        except ValueError:
            valid = False
        if not valid:
            raise DesktopError('Desktop readiness URL or process identity is invalid.')
        try:
            _ordinary(self.ready.parent, True)
            with self.ready.open('x', encoding='utf-8') as stream:
                json.dump({'schema': 1, 'url': url, 'pid': pid, 'token': self.token}, stream)
        except OSError as exc:
            raise DesktopError('Desktop readiness record could not be created.') from exc

    def watch(self, server, *, wait_event=_wait_event, close_event=_close_event):
        def monitor():
            try:
                while not self.finished.is_set():
                    status = wait_event(self.handle, 200)
                    if status == 0:
                        server.should_exit = True
                        return
                    if status != 258:
                        self.failed = True
                        server.should_exit = True
                        print('[ERROR] Desktop stop event failed; shutting down owned server.', file=sys.stderr, flush=True)
                        return
            finally:
                close_event(self.handle)
        self.thread = threading.Thread(target=monitor, daemon=True, name='desktop-stop-event')
        self.thread.start()

    def close(self):
        self.finished.set()
        if self.thread:
            self.thread.join(timeout=1)
        elif self.handle:
            _close_event(self.handle)
        self.handle = None
