"""Bound the frozen V37 return checker; zero local gradients or VM calls."""
import hashlib
import json
from pathlib import Path
import subprocess
import sys
import time

ROOT = Path(__file__).resolve().parents[1]
OUT = ROOT / 'outputs/cctv_dgp_delivered_guard_grad_v37_audit_run_v1'
CHECKER = ROOT / 'scripts/audit_cctv_dgp_delivered_guard_grad_v37_return.py'
CHECKER_SHA = 'cc75fcdfa50aa4faae3dec00a1bb5a97d03ad4c36a721d5738d1d11b444dffcf'
PROTOCOL_SHA = '7e3c9ae20fe2f4faa2d0e83ba628294c5b91a8d0f43093ea53bffcc6320d0090'
ARCHIVE_SHA = 'b2b069232d8e899147cf1229a7f9b3d24240509cb2899a7cba43b782d2909c9c'
ARCHIVE_BYTES = 1018359217
CAP = 1230


def sha(path):
    with Path(path).open('rb') as stream: return hashlib.file_digest(stream, 'sha256').hexdigest()


def write(path, value):
    with Path(path).open('x', encoding='utf-8', newline='\n') as stream:
        stream.write(json.dumps(value, indent=2, allow_nan=False) + '\n')


def main():
    assert not OUT.exists(), 'Retain preceding attempts; no automatic retry'
    assert Path(sys.executable).resolve() == (ROOT / 'venv/Scripts/python.exe').resolve()
    assert sha(CHECKER) == CHECKER_SHA
    assert sha(ROOT / 'outputs/cctv_dgp_delivered_guard_grad_v37_vm/protocol.json') == PROTOCOL_SHA
    assert not (ROOT / 'outputs/cctv_dgp_delivered_guard_grad_v37_return').exists()
    assert not (ROOT / 'outputs/cctv_dgp_delivered_guard_grad_v37_independent_audit.json').exists()
    archive = ROOT / 'outputs/cctv-dgp-delivered-guard-grad-v37-results.tar.gz'
    sidecar = Path(str(archive) + '.sha256'); exported = ROOT / 'outputs/cctv-dgp-delivered-guard-grad-v37-export.json'
    assert archive.stat().st_size == ARCHIVE_BYTES and sha(archive) == ARCHIVE_SHA
    assert sidecar.read_text().strip().split() == [ARCHIVE_SHA, archive.name]
    result = json.loads(exported.read_text(encoding='utf-8'))
    assert result['complete'] and result['archive_sha256'] == ARCHIVE_SHA and result['bytes'] == ARCHIVE_BYTES
    assert result['optimizer_updates'] == 0 and result['run_results_present'] and not result['failure_present']
    assert result['training_success_not_implied']
    OUT.mkdir()
    write(OUT / 'execution.json', {'checker_sha256': CHECKER_SHA, 'supervisor_sha256': sha(Path(__file__)),
          'protocol_sha256': PROTOCOL_SHA, 'archive_sha256': ARCHIVE_SHA, 'archive_bytes': ARCHIVE_BYTES,
          'sidecar_sha256': sha(sidecar), 'export_receipt_sha256': sha(exported), 'device': 'cpu',
          'internal_seconds': 1200, 'external_seconds': CAP,
          'matching_local_workload_processes_before_launch': 0, 'process_inventory_tool_chunk': '3c6e01',
          'process_inventory_UTC': '2026-10-08T08:27:02.6371000Z',
          'local_autograd_calls': 0, 'local_optimizer_updates': 0, 'VM_calls': 0,
          'frozen_model_replays_only': True, 'automatic_repeat_permitted': False})
    started = time.monotonic(); timed_out = False
    with (OUT / 'audit.log').open('xb') as log:
        process = subprocess.Popen([sys.executable, '-B', '-u', str(CHECKER), '--expected-sha', ARCHIVE_SHA,
                                   '--expected-bytes', str(ARCHIVE_BYTES)], cwd=ROOT, stdout=log, stderr=subprocess.STDOUT)
        write(OUT / 'process.json', {'pid': process.pid, 'checker_sha256': CHECKER_SHA, 'external_stop_seconds': CAP})
        print(json.dumps({'audit_pid': process.pid, 'external_cap_seconds': CAP, 'log': str(OUT / 'audit.log')}), flush=True)
        try: code = process.wait(timeout=CAP)
        except subprocess.TimeoutExpired:
            timed_out = True; process.kill(); code = process.wait(timeout=10)
    receipt = {'complete': code == 0 and not timed_out, 'worker_exit_code': code, 'timeout': timed_out,
               'seconds': time.monotonic() - started, 'external_cap_seconds': CAP,
               'log_sha256': sha(OUT / 'audit.log'), 'checker_sha256': sha(CHECKER),
               'supervisor_sha256': sha(Path(__file__)), 'VM_calls': 0, 'local_autograd_calls': 0,
               'local_optimizer_updates': 0, 'automatic_repeat_permitted': False}
    write(OUT / 'external_receipt.json', receipt)
    print(json.dumps({'complete': receipt['complete'], 'exit_code': code, 'seconds': receipt['seconds']}), flush=True)
    if code: raise SystemExit(code)


if __name__ == '__main__': main()
