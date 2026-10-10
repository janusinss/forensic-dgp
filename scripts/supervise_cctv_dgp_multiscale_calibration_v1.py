"""Manual Linux/tmux supervisor with external worker/export deadlines."""
import argparse
import os
from pathlib import Path
import platform
import signal
import subprocess
import sys
import time

sys.path.insert(0,str(Path(__file__).resolve().parents[1]))
from cctv_dgp_multiscale_calibration_contract_v1 import NAME,BUDGETS,verify,read,write


def child(root,pin,mode,log_name,seconds):
    stamp=time.monotonic();offset=0;timeout=False
    worker=root/'scripts/cctv_dgp_multiscale_calibration_vm_v1.py';path=root/log_name
    with path.open('xb') as log:
        proc=subprocess.Popen([sys.executable,'-B','-u',str(worker),'--root',str(root),'--protocol-sha',pin,'--'+mode],
            stdout=log,stderr=subprocess.STDOUT,start_new_session=True)
        print({'child_pid':proc.pid,'mode':mode,'log':str(path),'external_cap_seconds':seconds},flush=True)
        try:
            while proc.poll() is None:
                if time.monotonic()-stamp>=seconds:
                    timeout=True;os.killpg(proc.pid,signal.SIGTERM)
                    try:proc.wait(timeout=BUDGETS['kill_grace_seconds'])
                    except subprocess.TimeoutExpired:os.killpg(proc.pid,signal.SIGKILL);proc.wait(timeout=5)
                    break
                time.sleep(1)
                with path.open('rb') as incoming:
                    incoming.seek(offset);chunk=incoming.read();offset=incoming.tell()
                if chunk:print(chunk.decode('utf-8',errors='replace'),end='',flush=True)
        except BaseException:
            if proc.poll() is None:
                os.killpg(proc.pid,signal.SIGTERM)
                try:proc.wait(timeout=30)
                except subprocess.TimeoutExpired:os.killpg(proc.pid,signal.SIGKILL);proc.wait(timeout=5)
            raise
    with path.open('rb') as incoming:
        incoming.seek(offset);chunk=incoming.read()
    if chunk:print(chunk.decode('utf-8',errors='replace'),end='',flush=True)
    code=124 if timeout else proc.returncode
    print({'mode':mode,'exit_code':code,'timeout':timeout,'seconds':time.monotonic()-stamp},flush=True)
    return code,time.monotonic()-stamp


def main():
    parser=argparse.ArgumentParser();parser.add_argument('--root',type=Path,required=True);parser.add_argument('--protocol-sha',required=True)
    a=parser.parse_args();root=a.root.resolve()
    assert sys.platform=='linux' and platform.node().split('.')[0]=='forensic-dgp-thesis'
    assert root==(Path.home()/'forensic-dgp'/NAME).resolve() and os.environ.get('TMUX')
    assert Path(sys.executable).absolute()==Path.home()/'forensic-dgp/cctv_dgp_vm_bundle/.venv/bin/python','Activate retained VM environment'
    for name in ['outputs','trainer.log','trainer_exit_code.txt','supervisor_receipt.json','export_manifest.json','export.log','export_exit_code.txt']:
        assert not (root/name).exists(),'Preserve prior/partial run; no repeat: '+name
    verify(root,a.protocol_sha)
    code,elapsed=child(root,a.protocol_sha,'run','trainer.log',1830)
    with (root/'trainer_exit_code.txt').open('x',encoding='ascii') as stream:stream.write(str(code)+'\n')
    evidence=root/'outputs/results.json'
    failure=root/'outputs/failure.json'
    if not evidence.exists() and not failure.exists():
        failure.parent.mkdir(exist_ok=True)
        write(failure,{'complete':False,'supervisor_retained_after_child_exit':True,
            'trainer_exit_code':code,'exact_completed_updates_unknown':True,
            'progress':{'optimizer_updates':None},'training_success_not_implied':True})
    updates=read(evidence)['optimizer_updates'] if evidence.exists() else (read(failure)['progress']['optimizer_updates'] if failure.exists() else None)
    write(root/'supervisor_receipt.json',{'complete':True,'protocol_sha256':a.protocol_sha,'seconds':elapsed,
        'external_cap_seconds':1830,'kill_grace_seconds':30,'within_external_bound':elapsed<=1865,
        'trainer_exit_code':code,'optimizer_updates':updates})
    export_code,_=child(root,a.protocol_sha,'export','export.log',630)
    with (root/'export_exit_code.txt').open('x',encoding='ascii') as stream:stream.write(str(export_code)+'\n')
    raise SystemExit(code if code else export_code)


if __name__=='__main__':main()
