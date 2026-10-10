"""Bounded local CPU inference supervisor; no model imports or VM work."""
import hashlib
import json
from pathlib import Path
import subprocess
import sys
import time

ROOT=Path(__file__).resolve().parents[1]
OUT=ROOT/'outputs/completion_input_footprints_comparison_v1_r1'


def main():
    assert not (OUT/'external_receipt.json').exists() and not (OUT/'inference.log').exists()
    protocol=(OUT/'protocol.json').read_bytes();p=json.loads(protocol)
    assert p['external_timeout_seconds']==630 and p['cap_seconds']==600
    runner=(ROOT/'venv/Scripts/python.exe').resolve()
    assert Path(sys.executable).resolve()==runner,'Use existing project CPU inference Python'
    runtime=json.loads((ROOT/'outputs/completion_input_footprints_runtime_v1/runtime_preflight.json').read_text())
    assert runtime['complete'] and Path(runtime['executable']).resolve()==runner and runtime['torch']=='2.13.0+cpu'
    started=time.monotonic();timed_out=False
    with (OUT/'inference.log').open('xb') as log:
        try:
            result=subprocess.run([sys.executable,'-B','-u',str(ROOT/'scripts/run_completion_input_footprints_comparison_v1_r1.py')],
                cwd=ROOT,stdout=log,stderr=subprocess.STDOUT,timeout=630,check=False)
            code=result.returncode
        except subprocess.TimeoutExpired:
            code=None;timed_out=True
    receipt={'complete':code==0,'worker_exit_code':code,'timeout':timed_out,'external_seconds':time.monotonic()-started,
        'external_cap_seconds':630,'protocol_sha256':hashlib.sha256(protocol).hexdigest(),
        'supervisor_sha256':hashlib.sha256(Path(__file__).read_bytes()).hexdigest(),
        'log_sha256':hashlib.sha256((OUT/'inference.log').read_bytes()).hexdigest(),
        'training':False,'VM_writes':0,'automatic_retry_permitted':False}
    with (OUT/'external_receipt.json').open('x',encoding='utf-8',newline='\n') as stream:
        stream.write(json.dumps(receipt,indent=2)+'\n')
    print(json.dumps(receipt),flush=True)
    if code!=0:raise SystemExit(1)


if __name__=='__main__':main()
