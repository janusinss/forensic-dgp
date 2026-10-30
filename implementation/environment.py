"""Standard-library bootstrap for a short, checkout-specific Windows environment.

Keeping dependency paths outside a deeply nested clone avoids Windows path
length failures while installing PyTorch. No registry change or administrator
permission is needed, and existing project venv directories are preserved.
"""
from __future__ import annotations

import argparse
from concurrent.futures import ThreadPoolExecutor
import hashlib
import json
import os
from pathlib import Path
import struct
import subprocess
import sys
import time
import venv

ROOT = Path(__file__).resolve().parents[1]
CONFIG_PATH = ROOT / '.implementation-env.json'
SUPPORTED = {(3, 12), (3, 13)}
UV_VERSION = '0.13.1'
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


def run_foreground(command: list[str], *, env: dict | None = None) -> int:
    """Keep ownership of the exact child until it exits, including on Ctrl+C."""
    process = subprocess.Popen(command, cwd=ROOT, env=env)
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


def install_dependencies(python: Path) -> int:
    """Resolve once with CPU-only Torch; satisfied reruns use uv's fast audit."""
    probe = (
        "import importlib.metadata as m,sys; "
        f"sys.exit(0 if m.version('uv') == {UV_VERSION!r} else 1)"
    )
    checked = subprocess.run([str(python), '-I', '-c', probe], cwd=ROOT,
                             stdout=subprocess.DEVNULL, stderr=subprocess.DEVNULL,
                             timeout=15, check=False)
    if checked.returncode not in (0, 1):
        return checked.returncode
    # --no-config excludes files, but uv also accepts environment overrides for
    # indexes, constraints and insecure hosts. Keep only its cache/TLS settings;
    # standard HTTPS_PROXY and SSL_CERT_FILE settings remain available.
    package_env = {key: value for key, value in os.environ.items()
                   if not key.upper().startswith(('UV_', 'PIP_'))
                   or key.upper() in ('UV_CACHE_DIR', 'UV_SYSTEM_CERTS')}
    package_env['PIP_CONFIG_FILE'] = os.devnull
    bundled_cache = ROOT / 'implementation' / 'uv_cache'
    if 'UV_CACHE_DIR' not in package_env and (bundled_cache.exists() or bundled_cache.is_symlink()):
        if not bundled_cache.is_dir() or bundled_cache.is_symlink() or bundled_cache.is_junction():
            raise EnvironmentError('The bundled uv cache must be an ordinary directory.')
        package_env['UV_CACHE_DIR'] = str(bundled_cache)
        print(f'[packages] Reusing bundled uv cache: {bundled_cache}', flush=True)
    if checked.returncode == 1:
        print(f'[packages] Installing uv {UV_VERSION} once into this environment...', flush=True)
        result = run_foreground([
            str(python), '-I', '-m', 'pip', '--isolated', 'install',
            '--disable-pip-version-check', '--only-binary', ':all:',
            '--index-url', 'https://pypi.org/simple', f'uv=={UV_VERSION}',
        ], env=package_env)
        if result:
            return result
    result = run_foreground([
        str(python), '-I', '-m', 'uv', '--no-config', '--no-python-downloads',
        'pip', 'install', '--python', str(python), '--torch-backend', 'cpu',
        '--default-index', 'https://pypi.org/simple', '--only-binary', ':all:',
        '--strict', '-r', str(ROOT / 'requirements.txt'),
    ], env=package_env)
    if result:
        return result
    # --strict prints diagnostics; pip check turns inconsistent dependency
    # metadata into a nonzero result before installation can report success.
    return run_foreground([
        str(python), '-I', '-m', 'uv', '--no-config', '--no-python-downloads',
        'pip', 'check', '--python', str(python),
    ], env=package_env)


def run_install_tasks(tasks: dict) -> int:
    """Join every owned task, including failed peers, before returning to cmd."""
    result = 0
    with ThreadPoolExecutor(max_workers=len(tasks)) as executor:
        futures = [(label, executor.submit(task)) for label, task in tasks.items()]
        for label, future in futures:
            while True:
                try:
                    code = future.result()
                    if code:
                        print(f'[ERROR] {label} failed (exit {code}).', file=sys.stderr, flush=True)
                        result = result or code
                    break
                except KeyboardInterrupt:
                    result = 130
                    print('Installation interrupted; waiting for owned processes to finish cleanup...', flush=True)
                    # A task can itself raise KeyboardInterrupt during a console
                    # interrupt. A completed future must not be awaited forever.
                    if future.done():
                        break
                except Exception as exc:
                    print(f'[ERROR] {label}: {exc}', file=sys.stderr, flush=True)
                    result = result or 1
                    break
    return result


def install_models() -> int:
    # Both downloaders use only the standard library. Separate child processes
    # receive Windows console Ctrl+C, unlike network calls inside worker threads.
    return run_install_tasks({
        'Restoration models': lambda: run_foreground([
            sys.executable, '-B', '-m', 'implementation.setup_models']),
        'Completion models': lambda: run_foreground([
            sys.executable, '-B', '-m', 'implementation.environment', 'completion-assets']),
    })


def install() -> int:
    if sys.version_info[:2] not in SUPPORTED or struct.calcsize('P') != 8:
        raise EnvironmentError('Setup requires Python 3.12 or 3.13 64-bit.')
    started = time.monotonic()
    record = expected_record()
    environment = Path(record['environment_root'])
    python = Path(record['python'])
    print(f'[1/3] Python environment: {environment}', flush=True)
    if not python.is_file():
        venv.EnvBuilder(with_pip=True, clear=False, symlinks=False).create(environment)
    check_python(python)
    print('[2/3] Installing dependencies and verifying both model families in parallel...', flush=True)
    result = run_install_tasks({
        'Application dependencies': lambda: install_dependencies(python),
        'Model assets': install_models,
    })
    if result:
        return result
    write_record(record)
    print(f'[3/3] Installation complete in {time.monotonic() - started:.1f}s. '
          'Run start.bat to open the application.', flush=True)
    return 0


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('command', choices=('install', 'launch', 'python', 'path', 'completion-assets'))
    args, launch_args = parser.parse_known_args(argv)
    if args.command != 'launch' and launch_args:
        parser.error('extra arguments are accepted only for launch')
    try:
        if args.command == 'path':
            print(expected_environment())
            return 0
        if args.command == 'install':
            return install()
        if args.command == 'completion-assets':
            # Trusted installer orchestration only; no model execution or
            # environment configuration is needed to acquire these exact bytes.
            from .completion_install import install_assets, CompletionInstallError
            from .completion_assets import CompletionAssetError
            try:
                install_assets()
            except (CompletionInstallError, CompletionAssetError) as exc:
                raise EnvironmentError(str(exc)) from exc
            print('Both completion files passed size and SHA-256 verification.', flush=True)
            return 0
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
