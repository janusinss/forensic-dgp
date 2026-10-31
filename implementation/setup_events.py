"""Small, standard-library progress channel for the graphical installer."""
from __future__ import annotations

import json
import os
import sys
from pathlib import Path
import threading

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
        descriptor = os.open(path, os.O_WRONLY | os.O_CREAT | os.O_APPEND | getattr(os, 'O_BINARY', 0), 0o600)
        try:
            os.write(descriptor, line)
        finally:
            os.close(descriptor)
