"""Resolve the optional, explicitly marked managed installation's data root."""
from __future__ import annotations

import json
import os
from pathlib import Path
import stat

MARKER = {'schema_version': 1, 'repository': 'janusinss/forensic-dgp'}


class ManagedPathError(RuntimeError):
    """The opted-in managed installation is not an ordinary owned directory."""


def _ordinary_existing(path: Path, *, directory: bool) -> None:
    info = path.lstat()
    if (stat.S_ISLNK(info.st_mode) or getattr(info, 'st_file_attributes', 0) & 1024
            or not (stat.S_ISDIR(info.st_mode) if directory else stat.S_ISREG(info.st_mode))):
        raise ManagedPathError(f'Expected an ordinary managed path: {path}')


def data_root(code_root: Path) -> Path:
    """Use checkout-local storage unless a validated managed root was selected."""
    selected = os.environ.get('DGP_MANAGED_ROOT')
    if selected is None:
        return Path(code_root)
    local = os.environ.get('LOCALAPPDATA', '')
    if not local or not Path(local).is_absolute() or not selected or not Path(selected).is_absolute():
        raise ManagedPathError('Managed installation needs absolute LOCALAPPDATA and DGP_MANAGED_ROOT.')
    expected = Path(os.path.abspath(os.path.join(local, 'DGP', 'app')))
    supplied = Path(os.path.abspath(selected))
    if os.path.normcase(str(supplied)) != os.path.normcase(str(expected)):
        raise ManagedPathError('Managed installation root does not match this user.')
    try:
        for parent in (expected, *expected.parents):
            _ordinary_existing(parent, directory=True)
        marker = expected / 'owner.json'
        _ordinary_existing(marker, directory=False)
        body = json.loads(marker.read_text(encoding='utf-8')) if marker.stat().st_size <= 8192 else None
        if type(body) is not dict or body != MARKER or type(body.get('schema_version')) is not int:
            raise ManagedPathError('Managed installation ownership marker is invalid.')
        data = expected / 'data'
        if data.exists() or data.is_symlink():
            _ordinary_existing(data, directory=True)
        return data
    except (OSError, ValueError, TypeError) as exc:
        raise ManagedPathError('Managed installation ownership or directories could not be verified.') from exc


def data_path(code_root: Path, *parts: str) -> Path:
    """Reject existing linked components before returning a managed data path."""
    root = data_root(code_root)
    if 'DGP_MANAGED_ROOT' not in os.environ:
        return root.joinpath(*parts)
    path = root
    for part in parts:
        if part in ('', '.', '..') or Path(part).name != part or '/' in part or '\\' in part:
            raise ManagedPathError('Invalid managed data path component.')
        path /= part
        try:
            if path.exists() or path.is_symlink():
                _ordinary_existing(path, directory=True)
        except OSError as exc:
            raise ManagedPathError('Managed data path could not be verified.') from exc
    return path


def managed_root() -> Path | None:
    """Return the validated managed root, or None for a checkout installation."""
    if 'DGP_MANAGED_ROOT' not in os.environ:
        return None
    return data_root(Path(__file__).resolve().parents[1]).parent
