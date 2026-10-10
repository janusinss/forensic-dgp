"""Bound the frozen V36 return audit; do not execute returned code or VM work."""
import hashlib
import json
from pathlib import Path
import subprocess
import sys
import time

ROOT = Path(__file__).resolve().parents[1]
OUT = ROOT / 'outputs/cctv_dgp_finite_clearance_probe_v36_audit_run_v1'
CHECKER = ROOT / 'scripts/audit_cctv_dgp_finite_clearance_probe_v36_return.py'
ARCHIVE_SHA = 'a84cc39b8caa0bb3b6aa29859cb5aff4b7aa3c43ed311002c6fbc52fa174064a'
CHECKER_SHA = '66cfe49db0c0aac15a5fa89d0a18ab31da2021b5fe0511849d5d1318d7fc625e'
PROTOCOL_SHA = '9af4cbf10d7c141e2cbef2248bc282c135a9cc6117f37b47feaef76d71dadd38'
ARCHIVE_BYTES = 388522811
CAP = 1230


def sha(path):
    with Path(path).open('rb') as stream:
        return hashlib.file_digest(stream, 'sha256').hexdigest()


def write(path, value):
    with Path(path).open('x', encoding='utf-8', newline='\n') as stream:
        stream.write(json.dumps(value, indent=2, allow_nan=False) + '\n')


def main():
    assert not OUT.exists(), 'Preserve all preceding attempts'
    assert Path(sys.executable).resolve() == (ROOT / 'venv/Scripts/python.exe').resolve()
    assert sha(CHECKER) == CHECKER_SHA
    protocol = ROOT / 'outputs/cctv_dgp_finite_clearance_probe_v36_vm/protocol.json'
    assert sha(protocol) == PROTOCOL_SHA
    assert not (ROOT / 'outputs/cctv_dgp_finite_clearance_probe_v36_return').exists()
    assert not (ROOT / 'outputs/cctv_dgp_finite_clearance_probe_v36_independent_audit.json').exists()
    archive = ROOT / 'outputs/cctv-dgp-finite-clearance-probe-v36-results.tar.gz'
    sidecar = Path(str(archive) + '.sha256')
    export = ROOT / 'outputs/cctv-dgp-finite-clearance-probe-v36-export.json'
    assert archive.stat().st_size == ARCHIVE_BYTES and sha(archive) == ARCHIVE_SHA
    assert sidecar.read_text().strip().split() == [ARCHIVE_SHA, archive.name]
    receipt = json.loads(export.read_text(encoding='utf-8'))
    assert receipt['complete'] and receipt['archive_sha256'] == ARCHIVE_SHA
    assert receipt['bytes'] == ARCHIVE_BYTES and receipt['optimizer_updates'] == 0
    assert receipt['candidate_displacement_trials'] == 4 and receipt['training_success_not_implied']
    OUT.mkdir()
    write(OUT / 'execution.json', {
        'checker_sha256': CHECKER_SHA, 'supervisor_sha256': sha(Path(__file__)),
        'protocol_sha256': PROTOCOL_SHA, 'archive_sha256': ARCHIVE_SHA,
        'archive_bytes': ARCHIVE_BYTES, 'sidecar_sha256': sha(sidecar),
        'export_receipt_sha256': sha(export), 'device': 'cpu',
        'internal_seconds': 1200, 'external_seconds': CAP,
        'matching_local_workload_processes_before_launch': 0,
        'process_inventory_tool_chunk': '79404e',
        'process_inventory_UTC': '2026-10-08T07:19:26.9339818Z',
        'local_autograd_calls': 0, 'local_optimizer_updates': 0, 'VM_calls': 0,
        'frozen_model_replays_only': True, 'automatic_repeat_permitted': False})
    started = time.monotonic()
    timed_out = False
    with (OUT / 'audit.log').open('xb') as log:
        process = subprocess.Popen([sys.executable, '-B', '-u', str(CHECKER),
            '--expected-sha', ARCHIVE_SHA, '--expected-bytes', str(ARCHIVE_BYTES)],
            cwd=ROOT, stdout=log, stderr=subprocess.STDOUT)
        write(OUT / 'process.json', {'pid': process.pid, 'checker_sha256': CHECKER_SHA,
            'external_stop_seconds': CAP})
        print(json.dumps({'audit_pid': process.pid, 'external_cap_seconds': CAP,
            'log': str(OUT / 'audit.log')}), flush=True)
        try:
            code = process.wait(timeout=CAP)
        except subprocess.TimeoutExpired:
            timed_out = True
            process.kill()
            code = process.wait(timeout=10)
    receipt = {'complete': code == 0 and not timed_out, 'worker_exit_code': code,
        'timeout': timed_out, 'seconds': time.monotonic() - started,
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
