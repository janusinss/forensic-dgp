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
import re
import stat
import uuid
import struct
import subprocess
import sys
import time
import venv

ROOT = Path(__file__).resolve().parents[1]
from .managed_paths import managed_root, custom_install_root, managed_cache
from .setup_events import SetupCancelled, check_cancel, event
from .offline_payload import OfflinePayloadError, load_payload, package_environment, committed_versions

_MANAGED_ROOT = managed_root()
CONFIG_PATH = ((_MANAGED_ROOT / 'data' / 'installations' / (ROOT.name + '.json'))
               if _MANAGED_ROOT else ROOT / '.implementation-env.json')
SUPPORTED = {(3, 12), (3, 13)}
UV_VERSION = '0.13.4'
VERSION_CHECK = "import struct,sys; sys.exit(0 if sys.version_info[:2] in ((3,12),(3,13)) and struct.calcsize('P') == 8 else 1)"


class EnvironmentError(RuntimeError):
    """The installation's expected local Python environment is unavailable."""


def dependency_identity(root: Path | None = None) -> str:
    """UI changes reuse the runtime; pinned dependency or Python changes isolate it."""
    root = ROOT if root is None else root
    digest = hashlib.sha256()
    for name in ('requirements.txt', 'implementation/capture_install.py'):
        digest.update(name.encode() + b'\0' + (root / name).read_bytes().replace(b'\r\n', b'\n'))
    for name in ('implementation/locks/windows-cp313-application.txt','implementation/locks/windows-cp313-capture.txt'):
        path=root/name
        if path.exists():
            digest.update(name.encode()+b'\0'+path.read_bytes().replace(b'\r\n',b'\n'))
    digest.update(f'{sys.version_info[:2]}:{UV_VERSION}:64'.encode())
    return digest.hexdigest()[:16]


def _base_environment(root: Path = ROOT, local_app_data: str | None = None) -> Path:
    local = local_app_data if local_app_data is not None else os.environ.get('LOCALAPPDATA', '')
    if not local or not Path(local).is_absolute():
        raise EnvironmentError('LOCALAPPDATA must name an absolute user-local directory.')
    managed = managed_root()
    if managed:
        return managed / 'environments' / dependency_identity(root)
    identity = os.path.normcase(str(root.resolve())).encode('utf-8')
    checkout_id = hashlib.sha256(identity).hexdigest()[:12]
    return Path(local) / 'DGP' / checkout_id


def _generation_path(root: Path, local_app_data: str | None, generation: str) -> Path:
    if not isinstance(generation,str) or not re.fullmatch('[0-9a-f]{32}',generation):
        raise EnvironmentError('Invalid managed environment generation.')
    if managed_root() is None:
        raise EnvironmentError('Environment generations require a managed installation.')
    base = _base_environment(root,local_app_data)
    return base.with_name(base.name + '-g' + generation)


def _ordinary_runtime(path: Path, directory: bool = False) -> None:
    info = path.lstat()
    if (stat.S_ISLNK(info.st_mode) or getattr(info,'st_file_attributes',0) & 1024
            or not (stat.S_ISDIR(info.st_mode) if directory else stat.S_ISREG(info.st_mode))
            or (not directory and info.st_nlink != 1)):
        raise EnvironmentError('Managed runtime paths cannot contain links or unexpected file types.')


def _configuration(path: Path) -> dict:
    _ordinary_runtime(path)
    if path.stat().st_size > 8192:
        raise EnvironmentError('Managed environment configuration is oversized.')
    result = json.loads(path.read_text(encoding='utf-8'))
    if type(result) is not dict:
        raise EnvironmentError('Invalid managed environment configuration.')
    return result


def _record_for_path(root: Path, environment: Path, generation: str | None = None) -> dict:
    result = {'schema_version': 2 if generation else 1,'checkout_root':str(root.resolve()),
              'environment_root':str(environment),'python':str(environment/'Scripts'/'python.exe')}
    if generation:
        result['generation'] = generation
    return result


def expected_environment(root: Path = ROOT, local_app_data: str | None = None) -> Path:
    generation = os.environ.get('DGP_SETUP_GENERATION')
    if generation is not None:
        return _generation_path(root,local_app_data,generation)
    managed = managed_root()
    if managed:
        configuration_path = managed/'data'/'installations'/(root.name + '.json')
        if configuration_path.exists() or configuration_path.is_symlink():
            configuration = _configuration(configuration_path)
            if configuration.get('schema_version') == 2:
                generation = configuration.get('generation')
                path = _generation_path(root,local_app_data,generation)
                if configuration != _record_for_path(root,path,generation):
                    raise EnvironmentError('Managed environment generation binding differs.')
                _verify_generation_marker(path,root,generation)
                return path
    return _base_environment(root,local_app_data)


def expected_record(root: Path = ROOT, local_app_data: str | None = None, *, generation: str | None = None) -> dict:
    environment = (_generation_path(root,local_app_data,generation) if generation is not None
                   else expected_environment(root,local_app_data))
    if generation is None:
        match = re.fullmatch('[0-9a-f]{16}-g([0-9a-f]{32})',environment.name)
        generation = match.group(1) if match else None
    return _record_for_path(root,environment,generation)


def _verify_generation_marker(environment: Path, root: Path, generation: str) -> None:
    for directory in (environment,*environment.parents):
        _ordinary_runtime(directory,directory=True)
    marker = _configuration(environment/'.dgp-environment.json')
    expected = {'schema':1,'dependency_identity':dependency_identity(root),'generation':generation,'state':'ready'}
    if marker != expected or type(marker.get('schema')) is not int:
        raise EnvironmentError('Managed environment generation is not verified and ready.')


def configured_python(config_path: Path = CONFIG_PATH, root: Path = ROOT,
                      local_app_data: str | None = None) -> Path:
    """Never execute an arbitrary configuration path; recompute all bindings."""
    try:
        configured = _configuration(config_path)
        generation = configured.get('generation') if configured.get('schema_version') == 2 else None
        expected = (expected_record(root,local_app_data,generation=generation) if generation is not None
                    else _record_for_path(root,_base_environment(root,local_app_data)))
        if configured != expected or type(configured.get('schema_version')) is not int:
            raise ValueError('local environment configuration does not match this checkout and user')
        if generation is not None:
            _verify_generation_marker(Path(expected['environment_root']),root,generation)
    except (OSError,ValueError,TypeError) as exc:
        raise EnvironmentError(f'{exc}. Repair using the installer for this checkout.') from exc
    python = Path(expected['python'])
    if not python.is_file():
        raise EnvironmentError(f'The expected environment Python is missing: {python}. Repair using the installer.')
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
    check_cancel()
    process = subprocess.Popen(command, cwd=ROOT, env=env)
    interrupted = False
    while True:
        try:
            returncode = process.wait()
            check_cancel()
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
        config_path.parent.mkdir(parents=True, exist_ok=True)
        temporary.write_text(json.dumps(record, indent=2) + '\n', encoding='utf-8')
        temporary.replace(config_path)
    finally:
        temporary.unlink(missing_ok=True)


def install_dependencies(python: Path) -> int:
    """Install a validated local wheel set, or preserve the developer online path."""
    payload = load_payload()
    if payload is not None:
        event('packages','running','Installing verified offline application packages')
        result = run_foreground(payload.package_command(python,'application'),env=package_environment())
        if result:
            return result
        return run_foreground([str(python),'-I','-m','pip','--isolated','check'],env=package_environment())
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
    custom = custom_install_root()
    if custom:
        package_env['UV_CACHE_DIR'] = str(managed_cache('uv'))
        package_env['PIP_CACHE_DIR'] = str(managed_cache('pip'))
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
            '--index-url', 'https://pypi.org/simple',
            *(['--cache-dir', str(managed_cache('pip'))] if custom else []), f'uv=={UV_VERSION}',
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
    def execute(label, task):
        check_cancel()
        event(label, 'running', label)
        code = task()
        check_cancel()
        event(label, 'complete' if code == 0 else 'error', label)
        return code
    with ThreadPoolExecutor(max_workers=len(tasks)) as executor:
        futures = [(label, executor.submit(execute, label, task)) for label, task in tasks.items()]
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


def install_capture() -> int:
    arguments=[sys.executable,'-B','-m','implementation.capture_install']
    if os.environ.get('DGP_OFFLINE_PAYLOAD') and os.environ.get('DGP_SETUP_GENERATION'):
        arguments.append('--defer-verification')
    return run_foreground(arguments)


def verify_runtime(python: Path, *, include_models: bool = True) -> int:
    """Verify an existing runtime without changing its installed packages."""
    check_python(python)
    requirements = []
    for line in (ROOT / 'requirements.txt').read_text(encoding='utf8').splitlines():
        line = line.strip()
        if line and not line.startswith('#'):
            if line.count('==') != 1:
                raise EnvironmentError('Runtime verification needs exact direct dependency pins.')
            requirements.append(line.split('=='))
    probe = ("import importlib.metadata as m; "
             "expected=" + repr(requirements + [['uv', UV_VERSION]]) + "; "
             "bad=[name for name,version in expected if m.version(name)!=version]; "
             "assert not bad, 'Installed package versions differ: '+repr(bad)")
    payload = load_payload()
    locked = (payload.locked_versions('application') if payload is not None else committed_versions('application',ROOT))
    if locked is not None:
        expected_locks = repr(locked)
        probe += "; locked=" + expected_locks + "; assert all(m.version(name)==version for name,version in locked), 'Offline package lock differs'"
    checks = [
        [str(python), '-I', '-B', '-c', probe],
        [str(python), '-I', '-B', '-m', 'pip', '--isolated', 'check'],
        [str(python), '-B', '-m', 'implementation.capture_install', '--verify-only'],
        [str(python), '-B', '-c', 'import app,torch,cv2,numpy; print("Application imports verified.")'],
    ]
    if include_models:
        checks[0:0] = [
            [str(python), '-B', '-m', 'implementation.setup_models', '--verify-only'],
            [str(python), '-B', '-m', 'implementation.completion_install', '--verify-only'],
        ]
    for command in checks:
        result = run_foreground(command)
        if result:
            return result
    return 0


def verify_installation() -> int:
    return verify_runtime(configured_python())


def _write_generation_state(folder: Path, generation: str, state: str) -> None:
    for parent in (folder,*folder.parents):
        _ordinary_runtime(parent,directory=True)
    value = {'schema':1,'dependency_identity':dependency_identity(ROOT),'generation':generation,'state':state}
    target = folder/'.dgp-environment.json'
    if target.exists():
        _ordinary_runtime(target)
    temporary = folder/('.dgp-environment.' + uuid.uuid4().hex + '.tmp')
    with temporary.open('x',encoding='utf-8',newline='\n') as writer:
        writer.write(json.dumps(value) + '\n')
        writer.flush()
        os.fsync(writer.fileno())
    temporary.replace(target)


def _install_offline(payload) -> int:
    """Preserve failed/active generations and publish only a fully verified candidate."""
    candidate = None
    try:
        candidate = expected_record()
        python = Path(candidate['python'])
        if python.is_file() and verify_runtime(python,include_models=False) == 0:
            event('environment','running','Reusing verified offline environment')
            result = install_models()
            if result:
                return result
            result = verify_runtime(python)
            if result == 0:
                write_record(candidate)
            return result
    except (EnvironmentError,OSError,ValueError,subprocess.SubprocessError) as exc:
        print('[environment] Existing runtime was preserved; preparing a separate repair generation: ' + str(exc),flush=True)
    generation = uuid.uuid4().hex
    candidate = expected_record(generation=generation)
    folder = Path(candidate['environment_root'])
    if folder.exists() or folder.is_symlink():
        raise EnvironmentError('New environment generation already exists; existing files were preserved.')
    for parent in folder.parents:
        if parent.exists():
            _ordinary_runtime(parent,directory=True)
    folder.mkdir(parents=True,exist_ok=False)
    _write_generation_state(folder,generation,'preparing')
    previous_generation = os.environ.get('DGP_SETUP_GENERATION')
    os.environ['DGP_SETUP_GENERATION'] = generation
    try:
        python = Path(candidate['python'])
        event('environment','running','Preparing isolated offline environment')
        venv.EnvBuilder(with_pip=True,clear=False,symlinks=False).create(folder)
        check_python(python)
        result = run_install_tasks({
            'Application dependencies': lambda: install_dependencies(python),
            'Model assets': install_models,
            'Capture runtime': install_capture,
        })
        if result:
            return result
        check_cancel()
        event('verification','running','Verifying offline packages, models and application')
        result = verify_runtime(python)
        if result:
            return result
        check_cancel()
        _write_generation_state(folder,generation,'ready')
        write_record(candidate)
        event('environment','complete','Offline environment verified and ready')
        return 0
    finally:
        if previous_generation is None:
            os.environ.pop('DGP_SETUP_GENERATION',None)
        else:
            os.environ['DGP_SETUP_GENERATION'] = previous_generation


def install() -> int:
    if sys.version_info[:2] not in SUPPORTED or struct.calcsize('P') != 8:
        raise EnvironmentError('Setup requires Python 3.12 or 3.13 64-bit.')
    check_cancel()
    payload = load_payload()
    if payload is not None and managed_root() is not None:
        return _install_offline(payload)
    started = time.monotonic()
    record = expected_record()
    environment = Path(record['environment_root'])
    python = Path(record['python'])
    print(f'[1/3] Python environment: {environment}', flush=True)
    reuse_managed = managed_root() is not None and environment.exists()
    if reuse_managed:
        for component in (environment, *environment.parents, python):
            if component.is_symlink() or (hasattr(component, 'is_junction') and component.is_junction()):
                raise EnvironmentError('Shared runtime paths cannot contain links; existing files were preserved.')
        if not python.is_file():
            raise EnvironmentError('An incomplete shared environment was preserved; repair requires a separate environment generation.')
        event('environment', 'running', 'Verifying shared runtime without package changes')
        if verify_runtime(python, include_models=False):
            raise EnvironmentError('Shared environment verification failed. Existing packages were preserved; repair requires a separate environment generation.')
        tasks = {'Model assets': install_models}
    else:
        if not python.is_file():
            venv.EnvBuilder(with_pip=True, clear=False, symlinks=False).create(environment)
        check_python(python)
        tasks = {
            'Application dependencies': lambda: install_dependencies(python),
            'Model assets': install_models,
            'Capture runtime': install_capture,
        }
    print('[2/3] Preparing dependencies and verifying model families...', flush=True)
    result = run_install_tasks(tasks)
    if result:
        return result
    check_cancel()
    write_record(record)
    print(f'[3/3] Installation complete in {time.monotonic() - started:.1f}s. '
          'Run start.bat to open the application.', flush=True)
    return 0


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('command', choices=('install', 'launch', 'python', 'path', 'completion-assets', 'verify'))
    args, launch_args = parser.parse_known_args(argv)
    if args.command != 'launch' and launch_args:
        parser.error('extra arguments are accepted only for launch')
    try:
        if args.command == 'path':
            print(expected_environment())
            return 0
        if args.command == 'install':
            return install()
        if args.command == 'verify':
            return verify_installation()
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
    except SetupCancelled as exc:
        print(f'[CANCELLED] {exc}', file=sys.stderr)
        return 130
    except (EnvironmentError, OfflinePayloadError, OSError, subprocess.SubprocessError) as exc:
        print(f'[ERROR] {exc}', file=sys.stderr)
        return 1
    except KeyboardInterrupt:
        return 130


if __name__ == '__main__':
    raise SystemExit(main())
