"""Owned-process finite supervisor for the HQ TRAIN completion calibration."""
import argparse
import hashlib
import json
import os
from pathlib import Path
import subprocess
import sys
import time
import psutil

ROOT=Path(__file__).resolve().parents[1]
WORKER=ROOT/'scripts/run_completion_codeformer_hq_train_probe_v1.py'
OUT=ROOT/'outputs/completion_codeformer_hq_train_probe_v1'

def save(path,data):
    path.write_text(json.dumps(data,indent=2,allow_nan=False)+'\n',encoding='utf-8')

def main():
    parser=argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--protocol-sha',required=True)
    parser.add_argument('--worker-sha',required=True)
    a=parser.parse_args()
    assert hashlib.sha256(WORKER.read_bytes()).hexdigest()==a.worker_sha, 'Worker source changed'
    assert hashlib.sha256((OUT/'protocol.json').read_bytes()).hexdigest()==a.protocol_sha, 'Protocol changed'
    input_review_path=OUT/'root_input_review.json'
    input_review=json.loads(input_review_path.read_text(encoding='utf-8-sig'))
    assert input_review['complete'] and input_review['protocol_sha256']==a.protocol_sha and input_review['local_neural_forwards_authorized']==24, 'Root input review required'
    receipt=OUT/'supervisor.json'
    assert not receipt.exists(), 'Do not relaunch a retained experiment automatically'
    live=[]
    for process in psutil.process_iter(['pid','cmdline']):
        if process.pid==os.getpid():continue
        try:
            command=process.info['cmdline'] or []
            if any(Path(item).name in ('completion_codeformer_worker_v1.py','run_completion_codeformer_hq_train_probe_v1.py') for item in command):
                live.append(process.pid)
        except (psutil.AccessDenied,psutil.NoSuchProcess):pass
    assert not live, 'An existing completion request is active; do not compete'
    started=time.monotonic(); child=None; samples=[]; peak_rss=peak_private=peak_wset=0
    record={'complete':False,'worker_sha256':a.worker_sha,'protocol_sha256':a.protocol_sha,'external_cap_seconds':630,'per_case_cap_seconds':120,'root_input_review_sha256':hashlib.sha256(input_review_path.read_bytes()).hexdigest(),'training_calls':0,'hardware':{'logical_cpus':psutil.cpu_count(),'physical_cpus':psutil.cpu_count(logical=False),'RAM_bytes':psutil.virtual_memory().total,'available_RAM_at_start':psutil.virtual_memory().available},'observations':samples}
    try:
        with (OUT/'worker.log').open('xb') as log:
            child=subprocess.Popen([sys.executable,'-B',str(WORKER),'--run','--protocol-sha',a.protocol_sha],cwd=ROOT,stdout=log,stderr=subprocess.STDOUT,creationflags=getattr(subprocess,'CREATE_NO_WINDOW',0),env={**os.environ,'PYTHONDONTWRITEBYTECODE':'1','OMP_NUM_THREADS':'4','MKL_NUM_THREADS':'4'})
            record['child_pid']=child.pid; save(receipt,record)
            measured=psutil.Process(child.pid); next_print=0
            while child.poll() is None:
                elapsed=time.monotonic()-started
                if elapsed>=630:
                    child.kill();child.wait(timeout=15);record['timed_out']=True;break
                try:
                    progress_path=OUT/'progress.json'
                    if progress_path.exists():
                        state=json.loads(progress_path.read_text(encoding='utf-8-sig'))
                        if state.get('phase') in ('case','forward','saving') and state.get('active_started_unix') is not None and time.time()-state['active_started_unix']>120:
                            child.kill();child.wait(timeout=15);record.update({'timed_out':True,'case_timeout':state.get('active_case')});break
                    memory=measured.memory_info()
                    peak_rss=max(peak_rss,memory.rss)
                    peak_private=max(peak_private,getattr(memory,'private',0))
                    peak_wset=max(peak_wset,getattr(memory,'peak_wset',0))
                    if elapsed>=next_print:
                        samples.append({'seconds':round(elapsed,3),'rss_bytes':memory.rss,'private_bytes':getattr(memory,'private',None),'available_RAM_bytes':psutil.virtual_memory().available})
                        print(json.dumps({'child_pid':child.pid,'seconds':round(elapsed,1),'peak_RSS_MiB':round(peak_rss/1024**2,1)}),flush=True)
                        save(receipt,record);next_print=elapsed+30
                except psutil.NoSuchProcess:pass
                time.sleep(.25)
        record.update({'complete':child.returncode==0,'returncode':child.returncode,'seconds':time.monotonic()-started,'peak_observed_RSS_bytes':peak_rss,'peak_observed_private_bytes':peak_private,'peak_reported_working_set_bytes':peak_wset,'minimum_RAM_certified':False,'runtime_is_local_CPU_calibration_not_training':True})
        save(receipt,record)
        print(json.dumps({k:v for k,v in record.items() if k!='observations'}),flush=True)
        return 0 if record['complete'] else 2
    finally:
        if child is not None and child.poll() is None:
            child.kill();child.wait(timeout=15)
            record.update({'complete':False,'interrupted':True,'returncode':child.returncode,'seconds':time.monotonic()-started})
            save(receipt,record)

if __name__=='__main__':raise SystemExit(main())
