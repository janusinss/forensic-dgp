"""Finite manual inference supervision and failure export; never starts training."""
import argparse
from pathlib import Path
import os
import signal
import subprocess
import sys
import time

ROOT=Path(__file__).resolve().parents[1]
sys.path.insert(0,str(ROOT))
from cctv_dgp_actual_step_review_v1_contract import BUDGETS,read,write,vm_scope,verify


def stop_owned(child):
    if child.poll() is not None:return
    os.killpg(child.pid,signal.SIGTERM)
    try:child.wait(timeout=30)
    except subprocess.TimeoutExpired:
        os.killpg(child.pid,signal.SIGKILL);child.wait(timeout=10)


def supervise(root,pin):
    vm_scope(root);verify(root,pin)
    assert not (root/'outputs').exists() and not (root/'supervisor_receipt.json').exists(), 'No resume or automatic repetition'
    started=time.monotonic();calls=[]
    def call(flag,log_name,cap):
        with (root/log_name).open('x',encoding='utf-8',newline='\n') as stream:
            args=[sys.executable,'-B','-u',str(root/'scripts/review_cctv_dgp_actual_steps_v1_vm.py'),
                  '--root',str(root),'--protocol-sha',pin,flag]
            child=subprocess.Popen(args,cwd=root,stdout=stream,stderr=subprocess.STDOUT,start_new_session=True)
            print({'child_pid':child.pid,'log':log_name,'optimizer_updates':0},flush=True)
            timed_out=False;stamp=time.monotonic()
            try:code=child.wait(timeout=cap)
            except subprocess.TimeoutExpired:
                timed_out=True;stop_owned(child);code=child.returncode
            except BaseException:
                stop_owned(child);raise
        calls.append({'flag':flag,'exit_code':code,'timeout':timed_out,'seconds':time.monotonic()-stamp,'external_cap_seconds':cap})
        return code
    review_code=call('--review','review.log',BUDGETS['worker_seconds']+30)
    with (root/'review_exit_code.txt').open('x',encoding='utf-8',newline='\n') as stream:stream.write(str(review_code)+'\n')
    out=root/'outputs';out.mkdir(exist_ok=True)
    if not (out/'results.json').exists() and not (out/'failure.json').exists():
        write(out/'failure.json',{'complete':False,'error':'Supervisor observed terminal review without a scientific receipt',
             'review_exit_code':review_code,'calls':calls,'optimizer_updates':0,'gradient_queries':0,'new_trained_checkpoint':False,
             'app_promotion':False,'partial_outputs_retained':True})
    write(root/'supervisor_receipt.json',{'complete':review_code==0,'review_exit_code':review_code,'calls':calls.copy(),
          'seconds':time.monotonic()-started,'optimizer_updates':0,'gradient_queries':0,'new_trained_checkpoint':False,
          'automatic_training_follow_on':False,'app_promotion':False})
    export_code=call('--export','export.log',BUDGETS['export_seconds']+30)
    print({'review_exit_code':review_code,'export_exit_code':export_code,'optimizer_updates':0},flush=True)
    return review_code if review_code else export_code


if __name__=='__main__':
    parser=argparse.ArgumentParser();parser.add_argument('--root',type=Path,required=True);parser.add_argument('--protocol-sha',required=True)
    args=parser.parse_args();sys.exit(supervise(args.root.resolve(),args.protocol_sha))
