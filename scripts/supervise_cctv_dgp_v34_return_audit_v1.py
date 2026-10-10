"""Bound the prospective V34 return audit; never repeat or launch VM work."""
import hashlib
import json
from pathlib import Path
import subprocess
import sys
import time

ROOT = Path(__file__).resolve().parents[1]
OUT = ROOT / 'outputs/cctv_dgp_group_guard_grad_v34_audit_run_v1'
CHECKER = ROOT / 'scripts/audit_cctv_dgp_group_guard_grad_v34_return.py'
ARCHIVE_SHA = '0a9a6a2b1dffcedf623c47e6e7d30c711785b28714a591a6d48c4c9e22d968c7'
CHECKER_SHA = '8126d7bb59bb8c6800f19128ccf45d35de08e0e41eaaf96883467ed1e4eae809'
ARCHIVE_BYTES = 1017922963
CAP = 1230


def sha(path):
    with Path(path).open('rb') as stream:
        return hashlib.file_digest(stream, 'sha256').hexdigest()


def write(path, value):
    with Path(path).open('x', encoding='utf-8', newline='\n') as stream:
        stream.write(json.dumps(value, indent=2, allow_nan=False) + '\n')


def main():
    assert not OUT.exists(), 'Preserve prior attempts'
    assert Path(sys.executable).resolve() == (ROOT / 'venv/Scripts/python.exe').resolve()
    assert sha(CHECKER) == CHECKER_SHA
    assert not (ROOT / 'outputs/cctv_dgp_group_guard_grad_v34_independent_audit.json').exists()
    OUT.mkdir()
    write(OUT / 'execution.json', {
        'checker_sha256': CHECKER_SHA, 'supervisor_sha256': sha(Path(__file__)),
        'protocol_sha256': sha(ROOT / 'outputs/cctv_dgp_group_guard_grad_v34_vm/protocol.json'),
        'archive_sha256': ARCHIVE_SHA, 'archive_bytes': ARCHIVE_BYTES,
        'device': 'cpu', 'internal_seconds': 1200, 'external_seconds': CAP,
        'matching_audit_processes_before_launch': 0,
        'process_inventory_tool_chunk': 'dbda89',
        'local_autograd_calls': 0, 'local_optimizer_updates': 0, 'VM_calls': 0,
        'frozen_model_replays_only': True, 'automatic_repeat_permitted': False})
    started = time.monotonic()
    timed_out = False
    with (OUT / 'audit.log').open('xb') as log:
        process = subprocess.Popen([sys.executable, '-B', '-u', str(CHECKER),
            '--expected-sha', ARCHIVE_SHA, '--expected-bytes', str(ARCHIVE_BYTES)],
            cwd=ROOT, stdout=log, stderr=subprocess.STDOUT)
        write(OUT / 'process.json', {'pid': process.pid,
            'checker_sha256': CHECKER_SHA, 'external_stop_seconds': CAP})
        print(json.dumps({'audit_pid': process.pid, 'external_cap_seconds': CAP,
            'log': str(OUT / 'audit.log')}), flush=True)
        try:
            code = process.wait(timeout=CAP)
        except subprocess.TimeoutExpired:
            timed_out = True
            process.kill()
            code = process.wait(timeout=10)
    receipt = {'complete': code == 0 and not timed_out, 'worker_exit_code': code,
        'timeout': timed_out, 'seconds': time.monotonic()-started,
        'external_cap_seconds': CAP, 'log_sha256': sha(OUT / 'audit.log'),
        'checker_sha256': sha(CHECKER), 'supervisor_sha256': sha(Path(__file__)),
        'VM_calls': 0, 'local_autograd_calls': 0, 'local_optimizer_updates': 0,
        'automatic_repeat_permitted': False}
    write(OUT / 'external_receipt.json', receipt)
    print(json.dumps({'complete': receipt['complete'], 'exit_code': code,
        'seconds': receipt['seconds']}), flush=True)
    if code != 0:
        raise SystemExit(code)


if __name__ == '__main__':
    main()
