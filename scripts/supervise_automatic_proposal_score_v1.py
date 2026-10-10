"""Separate finite worker/audit supervision with persistent logs and receipts."""
import argparse
import subprocess
import sys
import time
from automatic_proposal_score_v1_common import ROOT, OUT, BUDGETS, sha, read, write, bindings


def main():
    parser = argparse.ArgumentParser(); parser.add_argument('--mode', choices=['run', 'audit'], required=True)
    parser.add_argument('--protocol-sha', required=True); args = parser.parse_args()
    assert sha(OUT/'protocol.json') == args.protocol_sha
    p = read(OUT/'protocol.json'); bindings(p)
    worker = ROOT/'scripts'/(('run' if args.mode == 'run' else 'audit')+'_automatic_proposal_score_v1.py')
    assert sha(worker) == p['sources_sha256'][worker.relative_to(ROOT).as_posix()]
    log = OUT/(args.mode+'.log'); receipt = OUT/(args.mode+'_external_receipt.json')
    assert not log.exists() and not receipt.exists()
    start = time.monotonic(); timeout = False
    with log.open('xb') as stream:
        child = subprocess.Popen([sys.executable, '-B', '-u', str(worker), '--protocol-sha', args.protocol_sha],
            cwd=ROOT, stdout=stream, stderr=subprocess.STDOUT)
        print({'mode': args.mode, 'child_pid': child.pid, 'external_cap_seconds': 210}, flush=True)
        try: code = child.wait(timeout=210)
        except subprocess.TimeoutExpired:
            timeout = True; child.terminate()
            try: code = child.wait(timeout=20)
            except subprocess.TimeoutExpired: child.kill(); code = child.wait(timeout=20)
    write(receipt, {'complete': True, 'mode': args.mode, 'exit_code': code, 'timeout': timeout,
        'cap_seconds': 210, 'termination_allowance_seconds': 20, 'seconds': time.monotonic()-start,
        'protocol_sha256': args.protocol_sha, 'worker_sha256': sha(worker), 'log_sha256': sha(log),
        'VM_connections': 0, 'actual_training_launched': False})
    if code or timeout: raise RuntimeError('Detector trace/audit stopped; preserve evidence and do not rerun unchanged')
    print({'mode': args.mode, 'exit_code': code, 'timeout': timeout, 'seconds': time.monotonic()-start}, flush=True)


if __name__ == '__main__': main()
