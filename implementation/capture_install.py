"""Install and verify the isolated capture detector; reuse a matching runtime."""
import argparse
import json
import os
import re
from pathlib import Path
import subprocess
import sys
import venv

from .capture_worker import checked_model
from .environment import expected_environment, dependency_identity, _ordinary_runtime
from .managed_paths import custom_install_root, managed_cache
from .setup_events import SetupCancelled, check_cancel, event
from .offline_payload import load_payload, package_environment, committed_versions

PROBE = ("import importlib.metadata as m, numpy, runpy, sys; "
         "assert m.version('opencv-python-headless')=='4.13.0.92'; "
         "assert m.version('numpy')=='2.5.2'; "
         "scan=runpy.run_path(sys.argv[1])['scan_faces']; "
         "assert scan(numpy.zeros((64,64,3),dtype=numpy.uint8)) == []")


def capture_python():
    return expected_environment() / 'capture' / ('Scripts/python.exe' if sys.platform == 'win32' else 'bin/python')


def verified(python, *, timeout=30):
    if not python.is_file():
        return False
    payload=load_payload()
    probe=PROBE
    locked=(payload.locked_versions('capture') if payload is not None else committed_versions('capture'))
    if locked is not None:
        probe += '; locked='+repr(locked)+"; assert all(m.version(name)==version for name,version in locked), 'Offline capture package lock differs'"
    check_cancel()
    worker = Path(__file__).with_name('capture_worker.py')
    try:
        result=subprocess.run([str(python), '-I', '-B', '-c', probe, str(worker)], stdout=subprocess.DEVNULL,
                              stderr=subprocess.DEVNULL, timeout=timeout, check=False)
    except subprocess.TimeoutExpired as problem:
        check_cancel()
        raise RuntimeError(f'Capture verification exceeded {timeout} seconds while loading its libraries. Existing files were preserved; close other heavy work and retry setup.') from problem
    check_cancel()
    return result.returncode == 0


def _deferred_setup_allowed(python, payload):
    generation=os.environ.get('DGP_SETUP_GENERATION','')
    if payload is None or not re.fullmatch('[0-9a-f]{32}',generation):
        return False
    folder=expected_environment()
    if python.parent.parent != folder/'capture':
        return False
    marker=folder/'.dgp-environment.json'
    for parent in (folder,*folder.parents):
        _ordinary_runtime(parent,directory=True)
    _ordinary_runtime(marker)
    if marker.stat().st_size>8192:
        return False
    body=json.loads(marker.read_text(encoding='utf-8'))
    return body=={'schema':1,'dependency_identity':dependency_identity(),'generation':generation,'state':'preparing'}


def main(argv=None):
    parser=argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--verify-only',action='store_true')
    parser.add_argument('--defer-verification',action='store_true',help=argparse.SUPPRESS)
    args=parser.parse_args(argv)
    try:
        check_cancel()
        event('capture-runtime', 'running', 'Verifying the capture runtime')
        checked_model()
        python=capture_python()
        payload=load_payload()
        deferred=args.defer_verification
        if deferred and (args.verify_only or not _deferred_setup_allowed(python,payload)):
            raise RuntimeError('Deferred capture verification requires an owned, preparing offline generation.')
        probe_timeout=120 if not args.verify_only or os.environ.get('DGP_SETUP_GENERATION') else 30
        if not deferred and verified(python,timeout=probe_timeout):
            event('capture-runtime', 'complete', 'Capture runtime verified; cached packages reused')
            print('Capture detector runtime already verified; no package download needed.')
            return 0
        if args.verify_only:
            raise RuntimeError('Capture detector runtime is missing or differs. Run install.bat.')
        event('capture-runtime', 'running', 'Preparing the isolated capture runtime')
        if not python.is_file():
            venv.EnvBuilder(with_pip=True,clear=False).create(python.parent.parent)
        check_cancel()
        event('capture-runtime', 'running', 'Installing capture packages')
        payload = load_payload()
        cache_args = ['--cache-dir', str(managed_cache('pip'))] if custom_install_root() else []
        if payload is not None:
            subprocess.run(payload.package_command(python,'capture'),env=package_environment(),check=True)
        else:
            subprocess.run([str(python), '-I', '-m', 'pip', '--isolated', 'install', '--disable-pip-version-check',
                        '--index-url', 'https://pypi.org/simple', '--only-binary=:all:', *cache_args,
                        'opencv-python-headless==4.13.0.92', 'numpy==2.5.2'], check=True)
        check_cancel()
        if deferred:
            event('capture-runtime','running','Capture packages prepared; verification follows after bulk file installation')
            return 0
        event('capture-runtime', 'running', 'Checking the capture detector')
        if not verified(python,timeout=probe_timeout):
            raise RuntimeError('Capture detector runtime did not pass verification.')
        event('capture-runtime', 'complete', 'Capture runtime installed and verified')
        return 0
    except SetupCancelled as problem:
        event('capture-runtime', 'cancelled', str(problem))
        print(f'[CANCELLED] Capture setup: {problem}',file=sys.stderr)
        return 130
    except (OSError,RuntimeError,subprocess.SubprocessError) as problem:
        event('capture-runtime', 'error', str(problem))
        print(f'[ERROR] Capture setup: {problem}',file=sys.stderr)
        return 1


if __name__=='__main__':
    raise SystemExit(main())
