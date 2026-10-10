"""Bound the single prepared frozen CPU development inference run."""
import hashlib
import json
import os
from pathlib import Path
import subprocess
import sys
import time

ROOT = Path(__file__).resolve().parents[1]
OUT = ROOT / 'outputs/cctv_dgp_v38_quarter_paired_development_run_v1'
DATA = ROOT / 'outputs/cctv_dgp_v38_quarter_paired_development_v1'
WORKER = ROOT / 'scripts/run_cctv_dgp_v38_quarter_paired_development_v1.py'
PLAN_SHA = '4bb1649b87978509eb037a9e669edcee6086a5e6db52877c1837deae247543a3'
CAP = 1290


def sha(path):
    with Path(path).open('rb') as stream:
        return hashlib.file_digest(stream, 'sha256').hexdigest()


def write(path, value):
    with Path(path).open('x', encoding='utf-8', newline='\n') as stream:
        stream.write(json.dumps(value, indent=2, allow_nan=False) + '\n')


def main():
    assert not OUT.exists(), 'Preserve prior attempts; no automatic retry'
    assert Path(sys.executable).resolve() == (ROOT / 'venv/Scripts/python.exe').resolve()
    assert sha(DATA / 'plan.json') == PLAN_SHA
    plan = json.loads((DATA / 'plan.json').read_text(encoding='utf-8'))
    assert sha(WORKER) == plan['sources_sha256'][WORKER.relative_to(ROOT).as_posix()]
    assert not (DATA / 'execution.json').exists() and not (DATA / 'results.json').exists()
    OUT.mkdir()
    write(OUT / 'execution.json', {'plan_sha256': PLAN_SHA, 'worker_sha256': sha(WORKER),
          'supervisor_sha256': sha(Path(__file__)), 'device': 'cpu', 'internal_seconds': 1200,
          'external_seconds': CAP, 'threads': 4, 'VM_calls': 0, 'local_autograd_calls': 0,
          'optimizer_updates': 0, 'frozen_inference_only': True, 'automatic_repeat_permitted': False})
    env = os.environ.copy()
    env.update({'OPENBLAS_NUM_THREADS': '4', 'OMP_NUM_THREADS': '4', 'MKL_NUM_THREADS': '4'})
    start = time.monotonic(); timed_out = False
    with (OUT / 'inference.log').open('xb') as log:
        process = subprocess.Popen([sys.executable, '-B', '-u', str(WORKER)], cwd=ROOT,
                                   env=env, stdout=log, stderr=subprocess.STDOUT)
        write(OUT / 'process.json', {'pid': process.pid, 'external_stop_seconds': CAP})
        print(json.dumps({'inference_pid': process.pid, 'external_cap_seconds': CAP,
                          'log': str(OUT / 'inference.log')}), flush=True)
        try:
            code = process.wait(timeout=CAP)
        except subprocess.TimeoutExpired:
            timed_out = True; process.kill(); code = process.wait(timeout=10)
    write(OUT / 'external_receipt.json', {'complete': code == 0 and not timed_out,
          'worker_exit_code': code, 'timeout': timed_out, 'seconds': time.monotonic() - start,
          'external_cap_seconds': CAP, 'log_sha256': sha(OUT / 'inference.log'),
          'worker_sha256': sha(WORKER), 'supervisor_sha256': sha(Path(__file__)),
          'VM_calls': 0, 'local_autograd_calls': 0, 'optimizer_updates': 0,
          'automatic_repeat_permitted': False})
    print(json.dumps({'complete': code == 0 and not timed_out, 'exit_code': code,
                      'seconds': time.monotonic() - start}), flush=True)
    if code:
        raise SystemExit(code)


if __name__ == '__main__':
    main()
