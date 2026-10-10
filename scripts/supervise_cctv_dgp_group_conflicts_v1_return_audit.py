"""Bound the frozen prospective return audit; retain every exit and timeout."""
import argparse
import hashlib
import json
from pathlib import Path
import subprocess
import sys
import time

ROOT = Path(__file__).resolve().parents[1]
OUT = ROOT / 'outputs/cctv_dgp_group_conflicts_v1_audit_supervision'
CHECKER = ROOT / 'scripts/audit_cctv_dgp_group_conflicts_v1_return.py'
PROTOCOL = ROOT / 'outputs/cctv_dgp_group_conflicts_v1_vm/protocol.json'


def sha(path):
    with path.open('rb') as stream:
        return hashlib.file_digest(stream, 'sha256').hexdigest()


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--expected-sha', required=True)
    args = parser.parse_args()
    assert len(args.expected_sha) == 64 and all(c in '0123456789abcdef' for c in args.expected_sha)
    assert Path(sys.executable).resolve() == (ROOT / 'venv/Scripts/python.exe').resolve()
    assert not OUT.exists(), 'Retain the prior audit supervision evidence'
    protocol = json.loads(PROTOCOL.read_text(encoding='utf-8'))
    assert sha(CHECKER) == protocol['local_sources_sha256'][CHECKER.relative_to(ROOT).as_posix()]
    OUT.mkdir()
    start = time.monotonic()
    command = [sys.executable, '-B', '-u', str(CHECKER), '--expected-sha', args.expected_sha]
    timed_out = False
    with (OUT / 'audit.log').open('xb') as log:
        child = subprocess.Popen(command, cwd=ROOT, stdout=log, stderr=subprocess.STDOUT)
        print({'child_pid': child.pid, 'log': str((OUT / 'audit.log').relative_to(ROOT)),
            'external_cap_seconds': 1530}, flush=True)
        try:
            code = child.wait(timeout=1530)
        except subprocess.TimeoutExpired:
            timed_out = True
            child.terminate()
            try:
                code = child.wait(timeout=20)
            except subprocess.TimeoutExpired:
                child.kill()
                code = child.wait(timeout=20)
    receipt = {'complete': True, 'worker_exit_code': code, 'timeout': timed_out,
        'cap_seconds': 1530, 'termination_allowance_seconds': 20,
        'seconds': time.monotonic() - start, 'archive_sha256': args.expected_sha,
        'checker_sha256': sha(CHECKER), 'protocol_sha256': sha(PROTOCOL),
        'log_sha256': sha(OUT / 'audit.log'), 'supervisor_sha256': sha(Path(__file__)),
        'VM_connections': 0, 'local_training_launched': False,
        'scientific_thresholds_changed': False, 'model_qualification': False,
        'goal_complete': False}
    with (OUT / 'receipt.json').open('x', encoding='utf-8', newline='\n') as stream:
        json.dump(receipt, stream, indent=2, allow_nan=False)
        stream.write('\n')
    print({'worker_exit_code': code, 'timeout': timed_out, 'seconds': receipt['seconds']}, flush=True)
    if code or timed_out:
        raise RuntimeError('Independent return audit stopped; preserve its import, log and receipt')


if __name__ == '__main__':
    main()
