"""Install and verify the isolated capture detector; reuse a matching runtime."""
import argparse
from pathlib import Path
import subprocess
import sys
import venv

from .environment import expected_environment
from .managed_paths import custom_install_root, managed_cache
from .setup_events import SetupCancelled, check_cancel, event

PROBE = ("import importlib.metadata as m,cv2,numpy; "
         "assert m.version('opencv-python-headless')=='4.13.0.92'; "
         "assert m.version('numpy')=='2.5.2'; "
         "assert not cv2.CascadeClassifier(cv2.data.haarcascades+'haarcascade_frontalface_default.xml').empty()")


def capture_python():
    return expected_environment() / 'capture' / ('Scripts/python.exe' if sys.platform == 'win32' else 'bin/python')


def verified(python):
    if not python.is_file():
        return False
    return subprocess.run([str(python), '-I', '-B', '-c', PROBE], stdout=subprocess.DEVNULL,
                          stderr=subprocess.DEVNULL, timeout=30, check=False).returncode == 0


def main(argv=None):
    parser=argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--verify-only',action='store_true')
    args=parser.parse_args(argv)
    try:
        check_cancel()
        event('capture-runtime', 'running', 'Verifying the capture runtime')
        python=capture_python()
        if verified(python):
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
        cache_args = ['--cache-dir', str(managed_cache('pip'))] if custom_install_root() else []
        subprocess.run([str(python), '-I', '-m', 'pip', '--isolated', 'install', '--disable-pip-version-check',
                        '--index-url', 'https://pypi.org/simple', '--only-binary=:all:', *cache_args,
                        'opencv-python-headless==4.13.0.92', 'numpy==2.5.2'], check=True)
        check_cancel()
        event('capture-runtime', 'running', 'Checking the capture detector')
        if not verified(python):
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
