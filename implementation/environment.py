"""Standard-library bootstrap for a short, checkout-specific Windows environment.

Keeping dependency paths outside a deeply nested clone avoids Windows path
length failures while installing PyTorch. No registry change or administrator
permission is needed, and existing project venv directories are preserved.
"""
from __future__ import annotations

import argparse
import hashlib
import json
import os
from pathlib import Path
import struct
import subprocess
import sys
import venv

ROOT = Path(__file__).resolve().parents[1]
CONFIG_PATH = ROOT / '.implementation-env.json'
SUPPORTED = {(3, 12), (3, 13)}
VERSION_CHECK = "import struct,sys; sys.exit(0 if sys.version_info[:2] in ((3,12),(3,13)) and struct.calcsize('P') == 8 else 1)"


class EnvironmentError(RuntimeError):
    """The installation's expected local Python environment is unavailable."""


def expected_environment(root: Path = ROOT, local_app_data: str | None = None) -> Path:
    local = local_app_data if local_app_data is not None else os.environ.get('LOCALAPPDATA', '')
    if not local or not Path(local).is_absolute():
        raise EnvironmentError('LOCALAPPDATA must name an absolute user-local directory.')
    identity = os.path.normcase(str(root.resolve())).encode('utf-8')
    checkout_id = hashlib.sha256(identity).hexdigest()[:12]
    return Path(local) / 'DGP' / checkout_id


def expected_record(root: Path = ROOT, local_app_data: str | None = None) -> dict:
    environment = expected_environment(root, local_app_data)
    return {'schema_version': 1, 'checkout_root': str(root.resolve()),
            'environment_root': str(environment),
            'python': str(environment / 'Scripts' / 'python.exe')}


def configured_python(config_path: Path = CONFIG_PATH, root: Path = ROOT,
                      local_app_data: str | None = None) -> Path:
    """Never execute a path supplied by configuration; match its computed value."""
    expected = expected_record(root, local_app_data)
    try:
        if not config_path.is_file() or config_path.is_symlink() or config_path.stat().st_size > 8192:
            raise ValueError('missing or invalid local environment configuration')
        configured = json.loads(config_path.read_text(encoding='utf-8'))
        if configured != expected:
            raise ValueError('local environment configuration does not match this checkout and user')
    except (OSError, ValueError, TypeError) as exc:
        raise EnvironmentError(f'{exc}. Run install.bat for this checkout.') from exc
    python = Path(expected['python'])
    if not python.is_file():
        raise EnvironmentError(f'The expected environment Python is missing: {python}. Run install.bat.')
    return python


def check_python(python: Path) -> None:
    try:
        checked = subprocess.run([str(python), '-I', '-c', VERSION_CHECK],
                                 cwd=ROOT, stdout=subprocess.DEVNULL, stderr=subprocess.DEVNULL,
                                 timeout=15, check=False)
    except (OSError, subprocess.TimeoutExpired) as exc:
        raise EnvironmentError(f'Cannot validate the environment Python: {python}: {exc}') from exc
    if checked.returncode:
        raise EnvironmentError(f'The environment must use Python 3.12 or 3.13 64-bit: {python}. '
                               'Move this environment directory aside and run install.bat again.')


def run_foreground(command: list[str]) -> int:
    """Keep ownership of the exact child until it exits, including on Ctrl+C."""
    process = subprocess.Popen(command, cwd=ROOT)
    interrupted = False
    while True:
        try:
            returncode = process.wait()
            return 130 if interrupted and returncode == 0 else returncode
        except KeyboardInterrupt:
            # Windows delivers console Ctrl+C to both processes. The application
            # must finish its own worker cleanup; do not abandon it or kill ports.
            interrupted = True
            if process.poll() is None:
                print('Waiting for the current process to finish cleanup...', flush=True)


def write_record(record: dict, config_path: Path = CONFIG_PATH) -> None:
    temporary = config_path.with_name(config_path.name + f'.{os.getpid()}.tmp')
    try:
        temporary.write_text(json.dumps(record, indent=2) + '\n', encoding='utf-8')
        temporary.replace(config_path)
    finally:
        temporary.unlink(missing_ok=True)


def install() -> int:
    if sys.version_info[:2] not in SUPPORTED or struct.calcsize('P') != 8:
        raise EnvironmentError('Setup requires Python 3.12 or 3.13 64-bit.')
    record = expected_record()
    environment = Path(record['environment_root'])
    python = Path(record['python'])
    print(f'[1/4] Python environment: {environment}', flush=True)
    if not python.is_file():
        venv.EnvBuilder(with_pip=True, clear=False, symlinks=False).create(environment)
    check_python(python)
    print('[2/4] Installing CPU PyTorch and application dependencies...', flush=True)
    commands = [
        [str(python), '-m', 'pip', 'install', '--disable-pip-version-check', '--upgrade', 'pip'],
        [str(python), '-m', 'pip', 'install', '--disable-pip-version-check',
         '--index-url', 'https://download.pytorch.org/whl/cpu', 'torch==2.13.0'],
        [str(python), '-m', 'pip', 'install', '--disable-pip-version-check',
         '--index-url', 'https://pypi.org/simple', '-r', str(ROOT / 'requirements.txt')],
    ]
    for command in commands:
        result = run_foreground(command)
        if result:
            return result
    print('[3/4] Downloading or verifying the frozen model files...', flush=True)
    result = run_foreground([str(python), '-B', '-m', 'implementation.setup_models'])
    if result:
        return result
    write_record(record)
    print('[4/4] Restoration setup complete. Continuing to completion setup...', flush=True)
    return 0


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('command', choices=('install', 'launch', 'python', 'path'))
    args, launch_args = parser.parse_known_args(argv)
    if args.command != 'launch' and launch_args:
        parser.error('extra arguments are accepted only for launch')
    try:
        if args.command == 'path':
            print(expected_environment())
            return 0
        if args.command == 'install':
            return install()
        python = configured_python()
        check_python(python)
        if args.command == 'python':
            print(python)
            return 0
        return run_foreground([str(python), '-B', '-m', 'implementation.launch', *launch_args])
    except (EnvironmentError, OSError, subprocess.SubprocessError) as exc:
        print(f'[ERROR] {exc}', file=sys.stderr)
        return 1
    except KeyboardInterrupt:
        return 130


if __name__ == '__main__':
    raise SystemExit(main())
