"""Run the previously frozen return checker once under a finite local observer."""
from datetime import datetime, timezone
import hashlib
import json
from pathlib import Path
import subprocess
import sys
import time

ROOT = Path(__file__).resolve().parents[1]
OUT = ROOT / 'outputs/cctv_dgp_v32_loss_gradient_v1_audit_execution'
STATUS = ROOT / 'outputs/cctv_dgp_v32_loss_gradient_status_v1/status_receipt.json'
CHECKER = ROOT / 'scripts/audit_cctv_dgp_v32_loss_gradient_v1_return.py'
CHECKER_SHA = 'aed9fb7c4c7f24647e5dceb5718f5b28436821ff310b7deb7d6cfb227e22e496'
STEM = 'cctv-dgp-v32-loss-gradient-v1'


def sha(path):
    with Path(path).open('rb') as stream:
        return hashlib.file_digest(stream, 'sha256').hexdigest()


def read(path):
    return json.loads(Path(path).read_text(encoding='utf-8-sig'))


def write(path, value):
    with path.open('x', encoding='utf-8', newline='\n') as stream:
        stream.write(json.dumps(value, indent=2, allow_nan=False) + '\n')


def main():
    status = read(STATUS)
    assert status['complete'] and status['remote']['VM_writes'] == 0
    assert status['remote']['diagnostic_processes'] == []
    exported = status['remote']['export_receipt']
    assert exported['complete'] and exported['optimizer_updates'] == 0
    assert exported['run_results_present'] and not exported['failure_present']
    assert sha(CHECKER) == CHECKER_SHA, 'Previously prospective auditor must remain exact'
    files = [ROOT / 'outputs' / (STEM + suffix) for suffix in
             ['-results.tar.gz', '-results.tar.gz.sha256', '-export.json']]
    assert all(p.is_file() and not p.is_symlink() for p in files), 'Await completed human download and both sidecars'
    assert files[0].stat().st_size == exported['bytes']
    assert not OUT.exists(), 'Retain every earlier observer or audit attempt'
    OUT.mkdir()
    started = time.monotonic()
    plan = {'frozen_UTC': datetime.now(timezone.utc).isoformat(), 'reader_sha256': sha(Path(__file__)),
            'status_receipt_sha256': sha(STATUS), 'checker_sha256': CHECKER_SHA,
            'archive_sha256': exported['archive_sha256'], 'archive_bytes': exported['bytes'],
            'source_files_sha256': {p.relative_to(ROOT).as_posix(): sha(p) for p in files[1:]},
            'CPU_audit_cap_seconds': 1800, 'external_cap_seconds': 1860,
            'gradient_calls': 0, 'backward_calls': 0, 'optimizer_updates': 0,
            'returned_code_executed': False, 'VM_writes': 0, 'VM_run_started': False,
            'scope': 'Saved gradient arithmetic and forward-only replay of all200 returned TRAIN outputs; no quality qualification',
            'goal_complete': False}
    write(OUT / 'plan.json', plan)
    assert sha(files[0]) == exported['archive_sha256'], 'Do not audit a partial or different transfer'
    assert files[1].read_text(encoding='ascii').strip().split() == [exported['archive_sha256'], files[0].name]
    assert read(files[2]) == exported
    args = [sys.executable, '-X', 'utf8', '-B', str(CHECKER),
            '--expected-sha', exported['archive_sha256'], '--expected-bytes', str(exported['bytes'])]
    timeout = False
    try:
        with (OUT / 'stdout.log').open('xb') as out, (OUT / 'stderr.log').open('xb') as err:
            result = subprocess.run(args, cwd=ROOT, stdout=out, stderr=err, timeout=1860)
        code = result.returncode
    except subprocess.TimeoutExpired:
        timeout = True
        code = None
    receipt = {'complete': code == 0 and not timeout, 'plan_sha256': sha(OUT / 'plan.json'),
               'checker_sha256': sha(CHECKER), 'exit_code': code, 'timeout': timeout,
               'seconds': time.monotonic() - started,
               'stdout_sha256': sha(OUT / 'stdout.log'), 'stderr_sha256': sha(OUT / 'stderr.log'),
               'all_failure_logs_preserved': True, 'VM_writes': 0,
               'training_started_by_agent': False, 'local_gradient_calls': 0,
               'local_backward_calls': 0, 'local_optimizer_updates': 0, 'goal_complete': False}
    write(OUT / 'execution.json', receipt)
    print(json.dumps(receipt, indent=2), flush=True)
    if not receipt['complete']:
        print((OUT / 'stderr.log').read_text(encoding='utf-8', errors='replace')[-4000:], flush=True)
        raise RuntimeError('Return audit did not pass; preserve source, receipts and partial import')
    audit = read(ROOT / 'outputs/cctv_dgp_v32_loss_gradient_v1_independent_audit.json')
    assert audit['complete'] and audit['diagnostic_complete']
    print(json.dumps({k: audit[k] for k in ['members_verified', 'optimizer_updates', 'seconds', 'CPU_replay']}, indent=2))


if __name__ == '__main__':
    main()
