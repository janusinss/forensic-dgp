"""Bound one local inference/audit child; never connects to or launches a VM."""
import argparse
from datetime import datetime,timezone
import hashlib
import json
from pathlib import Path
import subprocess
import sys
import time

ROOT = Path(__file__).resolve().parents[1]
OUT = ROOT/'outputs/cctv_dgp_finite_guard_quantization_v1'
LOG = ROOT/'outputs/cctv_dgp_finite_guard_quantization_v1_supervision'


def main():
    parser = argparse.ArgumentParser(); parser.add_argument('--phase',choices=['worker','audit'],required=True)
    parser.add_argument('--protocol-sha',required=True); a = parser.parse_args()
    path = OUT/'protocol.json'
    assert hashlib.sha256(path.read_bytes()).hexdigest() == a.protocol_sha
    q = json.loads(path.read_text(encoding='utf-8'))
    name = 'diagnose_cctv_dgp_finite_guard_quantization_v1.py' if a.phase == 'worker' else 'audit_cctv_dgp_finite_guard_quantization_v1.py'
    relative = 'scripts/'+name
    assert hashlib.sha256((ROOT/relative).read_bytes()).hexdigest() == q['source_bindings'][relative]
    cap = q['external_worker_seconds'] if a.phase == 'worker' else q['external_audit_seconds']
    LOG.mkdir(exist_ok=True); log = LOG/(a.phase+'.log'); receipt = LOG/(a.phase+'_receipt.json')
    assert not log.exists() and not receipt.exists(), 'Retain every prior execution'
    args = [sys.executable,'-B','-u',str(ROOT/relative),'--protocol-sha',a.protocol_sha]
    started = time.monotonic(); timeout = False
    with log.open('x',encoding='utf-8') as stream:
        child = subprocess.Popen(args,cwd=ROOT,stdout=stream,stderr=subprocess.STDOUT)
        print({'child_pid':child.pid,'phase':a.phase,'external_cap_seconds':cap,'log':str(log)},flush=True)
        try: code = child.wait(timeout=cap)
        except subprocess.TimeoutExpired:
            timeout = True; child.kill(); child.wait(timeout=20); code = child.returncode
    value = {'complete':True,'phase':a.phase,'protocol_sha256':a.protocol_sha,'child_pid':child.pid,
        'external_cap_seconds':cap,'timeout':timeout,'exit_code':code,
        'seconds':time.monotonic()-started,'ended_UTC':datetime.now(timezone.utc).isoformat(),
        'VM_started':False,'automatic_training':False,'gradient_queries':0,'parameter_updates':0}
    with receipt.open('x',encoding='utf-8',newline='\n') as stream: stream.write(json.dumps(value,indent=2)+'\n')
    print(value,flush=True)
    if timeout or code != 0: raise SystemExit(1)


if __name__ == '__main__': main()
