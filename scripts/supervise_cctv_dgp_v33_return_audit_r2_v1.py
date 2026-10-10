"""Bound an independent return audit, reusing preserved imported files.

The checker constructs frozen CPU models for predetermined replay only. It
does not call backward, construct an optimizer, or connect to the VM.
"""
import hashlib
import json
from pathlib import Path
import subprocess
import sys
import time

ROOT = Path(__file__).resolve().parents[1]
OUT = ROOT / 'outputs/cctv_dgp_loss_cone_probe_v33_audit_r2_run_v1'
CHECKER = ROOT / 'scripts/audit_cctv_dgp_loss_cone_probe_v33_return_r2.py'
ARCHIVE_SHA = '20a196d6515bc46d2e53c74d1ae4182c28f39812649e32bf44a2c684b56b7c40'


def sha(path):
    with Path(path).open('rb') as stream:
        return hashlib.file_digest(stream, 'sha256').hexdigest()


def write(path, value):
    with path.open('x', encoding='utf-8', newline='\n') as stream:
        stream.write(json.dumps(value, indent=2, allow_nan=False) + '\n')


def main():
    assert not OUT.exists(), 'Preserve prior attempts and receipts'
    assert Path(sys.executable).resolve() == (ROOT / 'venv/Scripts/python.exe').resolve()
    assert not (ROOT / 'outputs/cctv_dgp_loss_cone_probe_v33_independent_audit_r2.json').exists()
    p = json.loads((ROOT / 'outputs/cctv_dgp_loss_cone_probe_v33_vm/protocol.json').read_text())
    assert p['budgets']['local_audit_seconds'] == 1800
    OUT.mkdir()
    write(OUT / 'execution.json', {
        'checker_sha256': sha(CHECKER), 'supervisor_sha256': sha(Path(__file__)),
        'protocol_sha256': sha(ROOT / 'outputs/cctv_dgp_loss_cone_probe_v33_vm/protocol.json'),
        'archive_sha256': ARCHIVE_SHA, 'archive_bytes': 1166123827,
        'device': 'cpu', 'internal_seconds': 1800, 'external_seconds': 1830,
        'manual_VM_training_boundary_retained': True,
        'before_interrupted_attempt_process_inventory_tool_chunk': '8a18cc',
        'matching_audit_processes_before_launch': 0,
        'prior_interrupted_R2_terminal_result': 'Unobserved; no surviving process or completed audit receipt',
        'model_replays_only': True, 'local_optimizer_updates': 0, 'VM_calls': 0})
    started = time.monotonic(); timed_out = False
    with (OUT / 'audit.log').open('xb') as log:
        process = subprocess.Popen([sys.executable, '-B', '-u', str(CHECKER),
            '--expected-sha', ARCHIVE_SHA, '--expected-bytes', '1166123827'],
            cwd=ROOT, stdout=log, stderr=subprocess.STDOUT)
        write(OUT / 'process.json', {'pid': process.pid, 'checker_sha256': sha(CHECKER),
            'external_stop_seconds': 1830, 'model_or_optimizer_calls_in_supervisor': 0})
        print(json.dumps({'audit_pid': process.pid, 'external_cap_seconds': 1830,
            'log': str(OUT / 'audit.log')}), flush=True)
        try:
            code = process.wait(timeout=1830)
        except subprocess.TimeoutExpired:
            timed_out = True; process.kill(); code = process.wait(timeout=10)
    write(OUT / 'external_receipt.json', {'complete': code == 0 and not timed_out,
        'worker_exit_code': code, 'timeout': timed_out,
        'seconds': time.monotonic() - started, 'external_cap_seconds': 1830,
        'log_sha256': sha(OUT / 'audit.log'), 'checker_sha256': sha(CHECKER),
        'supervisor_sha256': sha(Path(__file__)), 'VM_calls': 0,
        'local_optimizer_updates': 0, 'automatic_repeat_permitted': False})
    print(json.dumps({'complete': code == 0 and not timed_out,
        'exit_code': code, 'seconds': time.monotonic() - started}), flush=True)
    if code != 0: raise SystemExit(code)


if __name__ == '__main__': main()
