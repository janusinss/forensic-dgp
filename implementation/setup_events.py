"""Small, standard-library progress channel for the graphical installer."""
from __future__ import annotations

import json
import os
import sys
from pathlib import Path
import threading
import time

_lock = threading.Lock()


class SetupCancelled(RuntimeError):
    """A cancellation request reached an installation boundary."""


def _control_path(name):
    selected = os.environ.get(name)
    if not selected:
        return None
    path = Path(selected)
    if not path.is_absolute():
        raise RuntimeError(f'{name} must be absolute')
    for part in (path, *path.parents):
        if part.is_symlink() or (hasattr(part, 'is_junction') and part.is_junction()):
            raise RuntimeError('Installer control paths cannot contain links')
    return path


def check_cancel():
    path = _control_path('DGP_SETUP_CANCEL_FILE')
    if path is not None and path.exists():
        raise SetupCancelled('Installation cancelled at a safe boundary; previous active version retained.')


def _append_windows(path, data):
    """Distinguish transient sharing conflicts from real access denials.

    The CRT behind os.open reports both as EACCES. CreateFile preserves the
    Windows error code, so only sharing/lock violations receive a bounded retry.
    FILE_APPEND_DATA keeps each record at EOF even with another setup process.
    """
    import ctypes
    from ctypes import wintypes
    kernel = ctypes.WinDLL('kernel32', use_last_error=True)
    kernel.CreateFileW.argtypes = [wintypes.LPCWSTR, wintypes.DWORD, wintypes.DWORD,
                                   ctypes.c_void_p, wintypes.DWORD, wintypes.DWORD, wintypes.HANDLE]
    kernel.CreateFileW.restype = wintypes.HANDLE
    kernel.WriteFile.argtypes = [wintypes.HANDLE, ctypes.c_void_p, wintypes.DWORD,
                                ctypes.POINTER(wintypes.DWORD), ctypes.c_void_p]
    kernel.WriteFile.restype = wintypes.BOOL
    kernel.CloseHandle.argtypes = [wintypes.HANDLE]
    kernel.CloseHandle.restype = wintypes.BOOL
    deadline = time.monotonic() + 2.0
    invalid = ctypes.c_void_p(-1).value
    while True:
        handle = kernel.CreateFileW(str(path), 4, 3, None, 4, 128, None)
        if handle != invalid:
            break
        code = ctypes.get_last_error()
        if code not in (32, 33) or time.monotonic() >= deadline:
            raise ctypes.WinError(code)
        time.sleep(min(0.025, max(0, deadline-time.monotonic())))
    try:
        written = wintypes.DWORD()
        buffer = ctypes.create_string_buffer(data)
        if not kernel.WriteFile(handle, buffer, len(data), ctypes.byref(written), None):
            raise ctypes.WinError(ctypes.get_last_error())
        if written.value != len(data):
            raise OSError('Installer progress record was not written completely')
    finally:
        if not kernel.CloseHandle(handle):
            raise ctypes.WinError(ctypes.get_last_error())


def _append(path, data):
    if os.name == 'nt':
        _append_windows(path, data)
        return
    descriptor = os.open(path, os.O_WRONLY | os.O_CREAT | os.O_APPEND, 0o600)
    try:
        if os.write(descriptor, data) != len(data):
            raise OSError('Installer progress record was not written completely')
    finally:
        os.close(descriptor)


def event(stage, status, message, *, current=None, total=None):
    path = _control_path('DGP_SETUP_EVENTS')
    if path is None:
        return
    record = {'schema': 1, 'stage': stage, 'status': status, 'message': message}
    if current is not None:
        record['current'] = current
    if total is not None and total > 0:
        record['total'] = total
    encoded = json.dumps(record, ensure_ascii=True, separators=(',', ':'))
    line = (encoded + '\n').encode('utf8')
    with _lock:
        print('DGP_EVENT ' + encoded, file=sys.stdout, flush=True)
        _append(path, line)
