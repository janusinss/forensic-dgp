"""Bound one frozen local CPU inference process; never launch VM training."""
from datetime import datetime,timezone
from pathlib import Path
import subprocess
import sys
import time

from audit_dgp_detector_confidence_v1_r1 import ROOT,OUT,read,sha,write


def main():
    assert sys.prefix==str(ROOT/'venv'),'Use the project virtual environment'
    assert (OUT/'binding.json').is_file() and not (OUT/'supervisor.json').exists() and not (OUT/'execution.json').exists()
    assert not (OUT/'results.json').exists() and not (OUT/'failure.json').exists()
    p=read(OUT/'protocol.json');assert sha(OUT/'protocol.json')==read(OUT/'binding.json')['protocol_sha256']
    assert not p['actual_training_permitted'] and all(sha(ROOT/n)==h for n,h in p['assets_sha256'].items())
    command=[sys.executable,'-B','-u',str(ROOT/'scripts/record_dgp_detector_confidence_v1_r1.py')]
    write(OUT/'execution.json',dict(protocol_sha256=sha(OUT/'protocol.json'),command=command,
        external_wall_cap_seconds=210,kill_grace_seconds=5,local_CPU_inference_only=True,actual_training_permitted=False))
    started=time.monotonic();timeout=False
    with (OUT/'run.log').open('xb') as log:
        child=subprocess.Popen(command,cwd=ROOT,stdout=log,stderr=subprocess.STDOUT)
        print(dict(child_pid=child.pid,external_wall_cap_seconds=210,log=str(OUT/'run.log')),flush=True)
        try:code=child.wait(timeout=210)
        except subprocess.TimeoutExpired:
            timeout=True;child.terminate()
            try:code=child.wait(timeout=5)
            except subprocess.TimeoutExpired:child.kill();code=child.wait(timeout=5)
    write(OUT/'supervisor.json',dict(complete=True,UTC=datetime.now(timezone.utc).isoformat(),script_sha256=sha(__file__),
        child_pid=child.pid,child_exit_code=code,timeout=timeout,seconds=time.monotonic()-started,external_wall_cap_seconds=210,
        neural_training_or_VM_calls=0,inference_results_present=(OUT/'results.json').exists()))
    print((OUT/'run.log').read_text(encoding='utf-8'),end='',flush=True)
    print(dict(child_exit_code=code,timeout=timeout,inference_results_present=(OUT/'results.json').exists()),flush=True)
    if code!=0:raise SystemExit(code if code>0 else 1)


if __name__=='__main__':main()
