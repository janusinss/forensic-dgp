"""Resolve the optional, explicitly marked managed installation's data root."""
from __future__ import annotations

import json
import os
import re
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


def _same(left: Path, right: Path) -> bool:
    return os.path.normcase(str(left)) == os.path.normcase(str(right))


def _absolute(selected: str, label: str) -> Path:
    if not selected or not Path(selected).is_absolute():
        raise ManagedPathError(f'{label} must be an absolute path.')
    if os.name == 'nt':
        if not re.match(r'^[A-Za-z]:[\\/]', selected):
            raise ManagedPathError('A fully qualified local drive path is required.')
        for part in re.split(r'[\\/]', selected[3:]):
            if (not part or part in ('.', '..') or part.endswith(('.', ' '))
                    or re.search(r'[\x00-\x1f<>\"|?*:]', part)
                    or re.match(r'(?i)^(CON|PRN|AUX|NUL|COM[1-9]|LPT[1-9])(?:\.|$)', part)):
                raise ManagedPathError('Installation path contains an unsafe Windows component.')
    path = Path(os.path.abspath(selected))
    if str(path).startswith(('\\\\', '//')):
        raise ManagedPathError('Network installation locations are not supported.')
    return path


def _validate_custom_shell(shell: Path) -> None:
    if os.name != 'nt':
        raise ManagedPathError('Custom installation locations require Windows.')
    import ctypes
    from ctypes import wintypes
    get_drive_type = ctypes.WinDLL('kernel32', use_last_error=True).GetDriveTypeW
    get_drive_type.argtypes = [wintypes.LPCWSTR]
    get_drive_type.restype = wintypes.UINT
    if get_drive_type(shell.anchor) != 3:
        raise ManagedPathError('Choose a directory on a fixed local drive.')
    if _same(shell, Path(shell.anchor)) or len(str(shell)) > 120:
        raise ManagedPathError('Choose a dedicated installation folder with a path of at most 120 characters.')
    forbidden = [Path(os.path.abspath(value)) for key in ('SystemRoot', 'ProgramFiles', 'ProgramFiles(x86)', 'ProgramData')
                 if (value := os.environ.get(key))]
    for folder in forbidden:
        if _same(shell, folder) or folder in shell.parents:
            raise ManagedPathError('System and shared program directories cannot host a per-user installation.')
    bases = [Path(os.path.abspath(value)) for key in ('USERPROFILE', 'LOCALAPPDATA', 'APPDATA', 'TEMP', 'TMP')
             if (value := os.environ.get(key))]
    profile = os.environ.get('USERPROFILE')
    if profile:
        bases.extend(Path(profile) / name for name in ('Desktop', 'Documents', 'Downloads', 'Pictures', 'Videos', 'Music'))
    if any(_same(shell, folder) for folder in bases):
        raise ManagedPathError('Choose a dedicated child folder instead of a profile or data root.')
    for parent in (shell, *shell.parents):
        _ordinary_existing(parent, directory=True)
    marker = shell / 'shell-owner.json'
    _ordinary_existing(marker, directory=False)
    if marker.stat().st_size > 8192:
        raise ManagedPathError('Desktop shell marker is oversized.')
    expected = {'schema': 1, 'application': 'Forensic DGP'}
    body = json.loads(marker.read_text(encoding='utf-8-sig'))
    if type(body) is not dict or body != expected or type(body.get('schema')) is not int:
        raise ManagedPathError('Desktop shell ownership marker differs.')


def custom_install_root(code_root: Path | None = None) -> Path | None:
    selected = os.environ.get('DGP_INSTALL_ROOT')
    if selected is None:
        return None
    shell = _absolute(selected, 'DGP_INSTALL_ROOT')
    try:
        _validate_custom_shell(shell)
        supplied = _absolute(os.environ.get('DGP_MANAGED_ROOT', ''), 'DGP_MANAGED_ROOT')
        expected = shell / 'runtime'
        if not _same(supplied, expected):
            raise ManagedPathError('Custom managed root must be the chosen folder runtime child.')
        code = Path(os.path.abspath(code_root or Path(__file__).resolve().parents[1]))
        if not re.fullmatch('[0-9a-f]{40}', code.name):
            raise ManagedPathError('Custom runtime code must belong to its selected version directory.')
        version_parent = expected / 'versions'
        if not _same(code.parent, version_parent):
            # Windows packaged processes can resolve a LocalAppData path to its
            # package-backed physical spelling. Require the very same directory
            # identity rather than accepting an unrelated lexical source path.
            if not os.path.samefile(code.parent, version_parent):
                raise ManagedPathError('Custom runtime code must belong to its selected version directory.')
        return shell
    except (OSError, ValueError, TypeError) as exc:
        raise ManagedPathError('Custom installation ownership or directories could not be verified.') from exc


def data_root(code_root: Path) -> Path:
    """Use checkout-local storage unless a validated managed root was selected."""
    selected = os.environ.get('DGP_MANAGED_ROOT')
    if selected is None:
        if 'DGP_INSTALL_ROOT' in os.environ:
            raise ManagedPathError('A custom installation requires its managed root.')
        return Path(code_root)
    supplied = _absolute(selected, 'DGP_MANAGED_ROOT')
    custom = custom_install_root(code_root)
    if custom:
        expected = custom / 'runtime'
        owner = {'schema_version': 2, 'repository': MARKER['repository'],
                 'layout': 'desktop-runtime-v1', 'install_root': str(custom)}
    else:
        local = _absolute(os.environ.get('LOCALAPPDATA', ''), 'LOCALAPPDATA')
        expected = local / 'DGP' / 'app'
        owner = MARKER
    if not _same(supplied, expected):
        raise ManagedPathError('Managed installation root does not match this installation.')
    try:
        for parent in (expected, *expected.parents):
            _ordinary_existing(parent, directory=True)
        marker = expected / 'owner.json'
        _ordinary_existing(marker, directory=False)
        body = json.loads(marker.read_text(encoding='utf-8-sig')) if marker.stat().st_size <= 8192 else None
        if type(body) is not dict or body != owner or type(body.get('schema_version')) is not int:
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



def managed_cache(name: str) -> Path:
    """Return a bounded ordinary cache path for the chosen desktop installation."""
    if name not in ('pip', 'uv') or custom_install_root() is None:
        raise ManagedPathError('A chosen desktop installation and known cache name are required.')
    path = managed_root()
    for component in ('cache', name):
        path /= component
        if path.exists() or path.is_symlink():
            _ordinary_existing(path, directory=True)
    return path
