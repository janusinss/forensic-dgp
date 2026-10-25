"""Human-invoked finite tmux supervisor; never imports Torch or runs automatically."""
import argparse
import os
from pathlib import Path
import platform
import selectors
import shlex
import shutil
import signal
import subprocess
import sys
import time

sys.dont_write_bytecode=True
ROOT=Path(__file__).resolve().parents[1];sys.path.insert(0,str(ROOT))
from research.codeformer_spatial_objectives_v1 import packet as p
ENTRY='scripts/run_codeformer_spatial_objectives_v1.py'


class WholeDeadline(TimeoutError):
    pass


def inventory():
    gpu=subprocess.run(['nvidia-smi','--query-compute-apps=pid,process_name','--format=csv,noheader,nounits'],
                       capture_output=True,text=True,check=True,timeout=20)
    p.require(not gpu.stdout.strip(),'GPU workload active; preserve it and stop')
    rows=subprocess.run(['ps','-eo','pid=,args='],capture_output=True,text=True,check=True,timeout=20)
    own={os.getpid(),os.getppid()};items=[]
    for line in rows.stdout.splitlines():
        fields=line.strip().split(None,1)
        p.require(len(fields)==2 and fields[0].isdigit(),'Unclassifiable process inventory')
        pid=int(fields[0]);args=fields[1]
        if pid in own:continue
        try:words=shlex.split(args)
        except ValueError:raise ValueError('Unclassifiable process arguments; stop before training')
        leaves=[Path(w).name.lower() for w in words]
        python=any(w.startswith('python') for w in leaves[:2])
        activity=any(w in words for w in ('--run','--train','--fit','--epochs','--max-updates'))
        readonly=any(w in words for w in ('--verify-transfer','--verify','--export','--audit','--help')) and not activity
        trainer=any(w.endswith('.py') and (w.startswith(('train_','supervise_')) or 'training' in w) for w in leaves)
        inline=python and '-c' in words and any(s in args for s in ('.backward(','.step(', 'optimizer.step'))
        shell=bool(leaves) and leaves[0] in ('bash','sh','zsh') and any(w.startswith('run_') and w.endswith('.sh') for w in leaves[1:])
        competing=(python and not readonly and (activity or trainer or inline)) or shell
        p.require(not competing,'Competing trainer/supervisor; preserve active process '+str(pid))
        items.append({'pid':pid,'arguments':args,'classification':'disjoint'})
    return {'complete':True,'GPU_processes':[],'processes':items,'nothing_killed':True}


def stop_owned(child):
    # Only the start_new_session child created by this invocation is signaled.
    try:os.killpg(child.pid,signal.SIGTERM)
    except ProcessLookupError:pass
    try:child.wait(timeout=15)
    except subprocess.TimeoutExpired:
        try:os.killpg(child.pid,signal.SIGKILL)
        except ProcessLookupError:pass
        child.wait(timeout=10)
    until=time.monotonic()+5
    while group_alive(child.pid) and time.monotonic()<until:time.sleep(.05)
    if group_alive(child.pid):
        os.killpg(child.pid,signal.SIGKILL)
        until=time.monotonic()+5
        while group_alive(child.pid) and time.monotonic()<until:time.sleep(.05)
    p.require(not group_alive(child.pid),'Owned process group remains; export forbidden')


def group_alive(pid):
    try:os.killpg(pid,0);return True
    except ProcessLookupError:return False


def supervised_main():
    parser=argparse.ArgumentParser(description=__doc__);parser.add_argument('--root',type=Path,default=ROOT)
    parser.add_argument('--protocol-sha',required=True);a=parser.parse_args();root=p.directory(a.root)
    p.require(platform.system()=='Linux' and bool(os.environ.get('TMUX')) and
              root==p.directory(Path.home()/'forensic-dgp'/p.VM_NAME),'Exact manual Linux tmux root required')
    protocol=p.verify(root,a.protocol_sha)
    p.require(not any((root/n).exists() for n in ('launch.json','supervisor.json','trainer.log','outputs','cache')),
              'Preserve prior/partial run; no retry or automatic resume')
    p.require(shutil.disk_usage(root).free>=p.LIMITS['initial_free_bytes'],'At least6GiB free required before launch')
    idle=inventory();writer=p.Writer(root,protocol)
    writer.json('launch.json',{'complete':True,'protocol_sha256':a.protocol_sha,'manual_tmux':True,
                             'inventory':idle,'maximum_worker_seconds':p.LIMITS['supervisor_seconds'],
                             'training_success_not_implied':True})
    environment=dict(os.environ,PYTHONDONTWRITEBYTECODE='1',PYTHONUNBUFFERED='1',CUBLAS_WORKSPACE_CONFIG=':4096:8')
    command=[sys.executable,'-B','-u',ENTRY,'--root',str(root),'--protocol-sha',a.protocol_sha,'--run']
    start=time.monotonic();timeout=False;capped=False;exception=None;written=0
    with (root/'trainer.log').open('xb') as log:
        child=subprocess.Popen(command,cwd=root,env=environment,stdout=subprocess.PIPE,stderr=subprocess.STDOUT,start_new_session=True)
        print({'child_pid':child.pid,'log':'trainer.log','external_cap_seconds':p.LIMITS['supervisor_seconds']},flush=True)
        try:
            with selectors.DefaultSelector() as selector:
                selector.register(child.stdout,selectors.EVENT_READ)
                while selector.get_map():
                    if time.monotonic()-start>p.LIMITS['supervisor_seconds']:
                        timeout=True;stop_owned(child);break
                    for key,_ in selector.select(.25):
                        b=os.read(key.fileobj.fileno(),65536)
                        if not b:selector.unregister(key.fileobj);continue
                        room=max(0,2*p.MiB-written)
                        p.require(shutil.disk_usage(root).free>=p.LIMITS['disk_reserve_bytes']+min(room,len(b)),
                                  'Supervisor log would cross4GiB reserve')
                        log.write(b[:room]);written+=min(room,len(b));sys.stdout.buffer.write(b[:room]);sys.stdout.flush()
                        if len(b)>room:capped=True;stop_owned(child);break
                    if capped:break
            code=child.wait(timeout=10)
        except BaseException as error:
            stop_owned(child);exception={'type':type(error).__name__,'message':str(error)[:2000]};code=125
        log.flush();os.fsync(log.fileno())
    if timeout:code=124
    if capped:code=125
    if group_alive(child.pid):stop_owned(child)
    p.require(child.poll() is not None,'Child not terminal; export forbidden')
    # Recount physical worker outputs after child completion before the writer
    # reserves its supervisor metadata; separate processes never share counters.
    writer=p.Writer(root,protocol)
    writer.json('supervisor.json',{'complete':True,'protocol_sha256':a.protocol_sha,'worker_exit':code,
        'worker_terminal':True,'owned_child_group_terminal':True,'child_pid':child.pid,'timeout':timeout,'log_capped':capped,'log_bytes':written,
        'seconds':time.monotonic()-start,'exception':exception,'training_success_not_implied':True,'model_qualified':False})
    if exception and exception['type']=='WholeDeadline':
        print({'whole_deadline_exceeded':True,'worker_terminal':True,
               'automatic_export_skipped':True,'manual_export_required':True},flush=True)
        return 124
    export_command=[sys.executable,'-B','-u',ENTRY,'--root',str(root),'--protocol-sha',a.protocol_sha,'--export']
    try:exported=subprocess.run(export_command,cwd=root,env=environment,timeout=315,check=False).returncode
    except subprocess.TimeoutExpired:exported=124
    print({'worker_exit':code,'export_exit':exported,'training_success_not_implied':True},flush=True)
    return 0 if code==0 and exported==0 else 1


def main():
    p.require(platform.system()=='Linux','Manual Linux supervisor required')
    def cleanup_expired(sig,frame):raise SystemExit('Bounded whole-deadline cleanup expired; inspect owned process state before export')
    def expired(sig,frame):
        signal.signal(signal.SIGALRM,cleanup_expired)
        signal.alarm(p.LIMITS['whole_deadline_cleanup_seconds'])
        raise WholeDeadline('Whole supervisor3120s ceiling exceeded; cleanup allowance40s')
    old=signal.signal(signal.SIGALRM,expired);signal.alarm(p.LIMITS['whole_supervisor_seconds'])
    try:return supervised_main()
    finally:signal.alarm(0);signal.signal(signal.SIGALRM,old)


if __name__=='__main__':raise SystemExit(main())
