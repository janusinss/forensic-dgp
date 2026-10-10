"""Bound the frozen prospective V38 checker; no returned code or VM execution."""
import hashlib
import json
import os
from pathlib import Path
import subprocess
import sys
import time

ROOT = Path(__file__).resolve().parents[1]
OUT = ROOT / 'outputs/cctv_dgp_delivered_margin_probe_v38_audit_run_v1'
CHECKER = ROOT / 'scripts/audit_cctv_dgp_delivered_margin_probe_v38_return.py'
CHECKER_SHA = 'b92a975c39c10126c6c12fb632f8656d50c46563959107939e14b7d2a1df5899'
PROTOCOL_SHA = 'caea39469005b7066d7cdbd74e073ba674b9b9b6fb9324f3110fe827a06948ab'
ARCHIVE_SHA = 'c1403d67ba0a66a1200285bcab291cd87dc9d6d7bf0518534f3b07a12ebff876'
ARCHIVE_BYTES = 388426632
CAP = 1230


def sha(path):
    with Path(path).open('rb') as stream: return hashlib.file_digest(stream, 'sha256').hexdigest()


def write(path, value):
    with Path(path).open('x', encoding='utf-8', newline='\n') as stream: stream.write(json.dumps(value, indent=2, allow_nan=False) + '\n')


def main():
    assert not OUT.exists(), 'Preserve prior attempts; no automatic retry'
    assert Path(sys.executable).resolve() == (ROOT / 'venv/Scripts/python.exe').resolve()
    assert sha(CHECKER) == CHECKER_SHA
    assert sha(ROOT / 'outputs/cctv_dgp_delivered_margin_probe_v38_vm/protocol.json') == PROTOCOL_SHA
    assert not (ROOT / 'outputs/cctv_dgp_delivered_margin_probe_v38_return').exists()
    assert not (ROOT / 'outputs/cctv_dgp_delivered_margin_probe_v38_independent_audit.json').exists()
    archive = ROOT / 'outputs/cctv-dgp-delivered-margin-probe-v38-results.tar.gz'
    sidecar = Path(str(archive) + '.sha256'); export = ROOT / 'outputs/cctv-dgp-delivered-margin-probe-v38-export.json'
    assert archive.stat().st_size == ARCHIVE_BYTES and sha(archive) == ARCHIVE_SHA
    assert sidecar.read_text().strip().split() == [ARCHIVE_SHA, archive.name]
    receipt = json.loads(export.read_text())
    assert receipt['complete'] and receipt['archive_sha256'] == ARCHIVE_SHA and receipt['bytes'] == ARCHIVE_BYTES
    assert receipt['optimizer_updates'] == receipt['committed_trajectory_updates'] == 0 and receipt['candidate_displacement_trials'] == 4
    assert receipt['training_success_not_implied'] and receipt['run_results_present'] and not receipt['failure_present']
    OUT.mkdir()
    write(OUT / 'execution.json', {'checker_sha256':CHECKER_SHA, 'supervisor_sha256':sha(Path(__file__)),
          'protocol_sha256':PROTOCOL_SHA, 'archive_sha256':ARCHIVE_SHA, 'archive_bytes':ARCHIVE_BYTES,
          'sidecar_sha256':sha(sidecar), 'export_receipt_sha256':sha(export), 'device':'cpu',
          'internal_seconds':1200, 'external_seconds':CAP, 'threads':4,
          'matching_local_workload_processes_before_launch':0, 'process_inventory_tool_chunk':'da1c7b',
          'process_inventory_UTC':'2026-10-08T09:22:27.3326737Z',
          'local_autograd_calls':0, 'local_optimizer_updates':0, 'VM_calls':0,
          'frozen_model_replays_only':True, 'automatic_repeat_permitted':False})
    started = time.monotonic(); timed_out = False; env = os.environ.copy()
    env.update({'OPENBLAS_NUM_THREADS':'4', 'OMP_NUM_THREADS':'4', 'MKL_NUM_THREADS':'4'})
    with (OUT / 'audit.log').open('xb') as log:
        process = subprocess.Popen([sys.executable, '-B', '-u', str(CHECKER), '--expected-sha', ARCHIVE_SHA,
                                   '--expected-bytes', str(ARCHIVE_BYTES)], cwd=ROOT, env=env, stdout=log, stderr=subprocess.STDOUT)
        write(OUT / 'process.json', {'pid':process.pid, 'checker_sha256':CHECKER_SHA, 'external_stop_seconds':CAP})
        print(json.dumps({'audit_pid':process.pid, 'external_cap_seconds':CAP, 'log':str(OUT / 'audit.log')}), flush=True)
        try: code = process.wait(timeout=CAP)
        except subprocess.TimeoutExpired:
            timed_out = True; process.kill(); code = process.wait(timeout=10)
    write(OUT / 'external_receipt.json', {'complete':code == 0 and not timed_out, 'worker_exit_code':code,
          'timeout':timed_out, 'seconds':time.monotonic() - started, 'external_cap_seconds':CAP,
          'log_sha256':sha(OUT / 'audit.log'), 'checker_sha256':sha(CHECKER), 'supervisor_sha256':sha(Path(__file__)),
          'VM_calls':0, 'local_autograd_calls':0, 'local_optimizer_updates':0, 'automatic_repeat_permitted':False})
    print(json.dumps({'complete':code == 0 and not timed_out, 'exit_code':code, 'seconds':time.monotonic() - started}), flush=True)
    if code: raise SystemExit(code)


if __name__ == '__main__': main()
