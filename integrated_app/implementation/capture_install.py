"""Install and verify the isolated capture detector; reuse a matching runtime."""
import argparse
from pathlib import Path
import subprocess
import sys
import venv

from .environment import expected_environment
from .capture_worker import checked_model

PROBE = ("import importlib.metadata as m, numpy, runpy, sys; "
         "assert m.version('opencv-python-headless')=='4.13.0.92'; "
         "assert m.version('numpy')=='2.5.2'; "
         "scan=runpy.run_path(sys.argv[1])['scan_faces']; "
         "assert scan(numpy.zeros((64,64,3),dtype=numpy.uint8)) == []")


def capture_python():
    return expected_environment() / 'capture' / ('Scripts/python.exe' if sys.platform == 'win32' else 'bin/python')


def verified(python):
    if not python.is_file():
        return False
    worker = Path(__file__).with_name('capture_worker.py')
    return subprocess.run([str(python), '-I', '-B', '-c', PROBE, str(worker)], stdout=subprocess.DEVNULL,
                          stderr=subprocess.DEVNULL, timeout=30, check=False).returncode == 0


def main(argv=None):
    parser=argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--verify-only',action='store_true')
    args=parser.parse_args(argv)
    try:
        checked_model()
        python=capture_python()
        if verified(python):
            print('Capture detector runtime already verified; no package download needed.')
            return 0
        if args.verify_only:
            raise RuntimeError('Capture detector runtime is missing or differs. Run install.bat.')
        if not python.is_file():
            venv.EnvBuilder(with_pip=True,clear=False).create(python.parent.parent)
        subprocess.run([str(python), '-I', '-m', 'pip', '--isolated', 'install', '--disable-pip-version-check',
                        '--index-url', 'https://pypi.org/simple', '--only-binary=:all:',
                        'opencv-python-headless==4.13.0.92', 'numpy==2.5.2'], check=True)
        if not verified(python):
            raise RuntimeError('Capture detector runtime did not pass verification.')
        return 0
    except (OSError,RuntimeError,subprocess.SubprocessError) as problem:
        print(f'[ERROR] Capture setup: {problem}',file=sys.stderr)
        return 1


if __name__=='__main__':
    raise SystemExit(main())
