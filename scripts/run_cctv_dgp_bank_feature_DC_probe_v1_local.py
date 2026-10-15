"""External 930-second cap for the finite local CPU-only inference diagnostic."""
from datetime import datetime,timezone
import hashlib
import json
from pathlib import Path
import subprocess
import sys
import time

ROOT=Path(__file__).resolve().parents[1]
OUT=ROOT/'outputs/cctv_dgp_bank_feature_DC_probe_v1_execution'
WORKER=ROOT/'scripts/probe_cctv_dgp_bank_feature_DC_v1.py'


def sha(path):
    return hashlib.sha256(Path(path).read_bytes()).hexdigest()


def write(path,value):
    with path.open('x',encoding='utf-8',newline='\n') as stream:
        stream.write(json.dumps(value,indent=2,allow_nan=False)+'\n')


def main():
    assert not OUT.exists() and not (ROOT/'outputs/cctv_dgp_bank_feature_DC_probe_v1').exists()
    OUT.mkdir()
    before=sha(WORKER); start=time.monotonic()
    command=[sys.executable,'-B','-u',str(WORKER)]
    cap=930
    write(OUT/'dispatch.json',dict(UTC=datetime.now(timezone.utc).isoformat(),
        command=command,worker_sha256=before,external_cap_seconds=cap,
        local_CPU_inference_only=True,actual_training_authorized=False))
    with (OUT/'probe.log').open('x',encoding='utf-8') as stream:
        child=subprocess.Popen(command,cwd=ROOT,stdout=stream,stderr=subprocess.STDOUT)
        print(dict(child_pid=child.pid,log=str(OUT/'probe.log'),external_cap_seconds=cap),flush=True)
        timeout=False
        try:
            code=child.wait(timeout=cap)
        except subprocess.TimeoutExpired:
            timeout=True; child.kill();child.wait(timeout=30);code=124
    assert sha(WORKER)==before
    write(OUT/'supervisor_receipt.json',dict(complete=True,worker_sha256=before,
        driver_sha256=sha(__file__),child_pid=child.pid,exit_code=code,timeout=timeout,
        external_cap_seconds=cap,seconds=time.monotonic()-start,
        local_CPU_inference_only=True,optimizer_updates=0,gradient_queries=0,
        actual_training_or_qualification_not_implied=True))
    print(dict(exit_code=code,timeout=timeout,seconds=time.monotonic()-start),flush=True)
    sys.exit(code)


if __name__=='__main__':main()
