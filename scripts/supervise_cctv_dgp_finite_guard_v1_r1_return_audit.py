"""Bound the frozen independent CPU audit; retain its timeout and exit evidence."""
import argparse
import hashlib
import json
from pathlib import Path
import subprocess
import sys
import time

ROOT = Path(__file__).resolve().parents[1]
OUT = ROOT/'outputs/cctv_dgp_finite_guard_v1_r1_audit_supervision'
CHECKER = ROOT/'scripts/audit_cctv_dgp_finite_guard_v1_r1_return.py'
PROTOCOL = ROOT/'outputs/cctv_dgp_finite_guard_v1_r1_vm/protocol.json'


def sha(path):
    with path.open('rb') as stream: return hashlib.file_digest(stream,'sha256').hexdigest()


def main():
    ap = argparse.ArgumentParser(); ap.add_argument('--expected-sha',required=True); a = ap.parse_args()
    assert len(a.expected_sha) == 64 and set(a.expected_sha) <= set('0123456789abcdef')
    assert Path(sys.executable).resolve() == (ROOT/'venv/Scripts/python.exe').resolve()
    assert not OUT.exists(), 'Preserve any preceding audit supervision'
    p = json.loads(PROTOCOL.read_text(encoding='utf-8'))
    assert sha(CHECKER) == p['local_sources_sha256'][CHECKER.relative_to(ROOT).as_posix()]
    assert p['local_audit_maximum_seconds'] == 2400
    OUT.mkdir(); start = time.monotonic(); timed_out = False
    with (OUT/'audit.log').open('xb') as log:
        child = subprocess.Popen([sys.executable,'-B','-u',str(CHECKER),'--expected-sha',a.expected_sha],
            cwd=ROOT,stdout=log,stderr=subprocess.STDOUT)
        print({'child_pid':child.pid,'log':str((OUT/'audit.log').relative_to(ROOT)),'external_cap_seconds':2430},flush=True)
        try: code = child.wait(timeout=2430)
        except subprocess.TimeoutExpired:
            timed_out = True; child.terminate()
            try: code = child.wait(timeout=20)
            except subprocess.TimeoutExpired: child.kill(); code = child.wait(timeout=20)
    receipt = {'complete':True,'worker_exit_code':code,'timeout':timed_out,'cap_seconds':2430,
        'termination_allowance_seconds':20,'seconds':time.monotonic()-start,'archive_sha256':a.expected_sha,
        'checker_sha256':sha(CHECKER),'protocol_sha256':sha(PROTOCOL),'log_sha256':sha(OUT/'audit.log'),
        'supervisor_sha256':sha(Path(__file__)),'VM_connections':0,'local_training_launched':False,
        'scientific_thresholds_changed':False,'model_qualification':False,'goal_complete':False}
    with (OUT/'receipt.json').open('x',encoding='utf-8',newline='\n') as stream:
        json.dump(receipt,stream,indent=2,allow_nan=False); stream.write('\n')
    print({'worker_exit_code':code,'timeout':timed_out,'seconds':receipt['seconds']},flush=True)
    if code or timed_out: raise RuntimeError('Audit stopped; retain import, log and receipt')


if __name__ == '__main__': main()
