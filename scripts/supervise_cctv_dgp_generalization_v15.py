"""Finite L4 inference/audit/export supervisor; no training or dependency changes."""
import argparse
import os
from pathlib import Path
import signal
import subprocess
import sys
import tarfile
import time


def supervise(root,parent,capacity,pin):
    sys.path.insert(0,str(parent));sys.path.insert(0,str(root))
    from cctv_dgp_targets_v6 import require_vm
    require_vm(root)
    import cctv_dgp_generalization_v15 as v
    import torch,torchvision
    p=v.verify(root,parent,pin)
    v.require((root.parent/'cctv_dgp_vm_bundle/cuda_runtime_before.txt').read_text().splitlines()==
              [torch.__version__,torchvision.__version__],'Existing CUDA runtime changed')
    start=time.monotonic();deadline=start+1800
    v.write(root/'supervisor_execution.json',{'complete':True,'protocol_sha256':pin,'pid':os.getpid(),
        'torch':torch.__version__,'torchvision':torchvision.__version__,'started_unix':time.time(),
        'budget_seconds':1800,'optimizer_updates':0,'backward_calls':0})
    def call(script,extra,name,cap):
        args=[sys.executable,'-u',str(root/'scripts'/script),'--root',str(root),'--parent',str(parent),
              '--capacity',str(capacity),'--expected-sha',pin]+extra
        with (root/name).open('x') as log:
            child=subprocess.Popen(args,cwd=root,stdout=log,stderr=subprocess.STDOUT,start_new_session=True)
            print({'child_pid':child.pid,'log':name},flush=True)
            try:code=child.wait(timeout=min(cap,max(1,deadline-time.monotonic())))
            except subprocess.TimeoutExpired:
                os.killpg(child.pid,signal.SIGINT)
                try:child.wait(timeout=15)
                except subprocess.TimeoutExpired:os.killpg(child.pid,signal.SIGKILL);child.wait()
                raise TimeoutError('Finite V15 child cap exceeded')
        print((root/name).read_text()[-1800:],flush=True)
        if code:raise subprocess.CalledProcessError(code,args)
    def export(success):
        archive=root/('cctv-dgp-generalization-v15-results.tar.gz' if success else 'cctv-dgp-generalization-v15-failure.tar.gz')
        names=[name for name in p['assets_sha256'] if name.endswith('.py') or name.startswith('lineage/')]
        names += [v.PLAN,'protocol.sha256','outputs/generalization_v15','runner.log','audit.log',
                  'supervisor_execution.json','supervisor_launch.json','supervisor_failure.json']
        with tarfile.open(archive,'x:gz',compresslevel=3) as stream:
            for name in names:
                if success:v.require(time.monotonic()<=deadline,'V15 export cap exceeded')
                path=root/name
                if path.exists():stream.add(path,arcname=name,recursive=True)
        Path(str(archive)+'.sha256').write_text(v.sha(archive)+'  '+archive.name+'\n',encoding='ascii',newline='\n')
        return archive
    try:
        call('run_cctv_dgp_generalization_v15.py',[],'runner.log',1230)
        out=root/'outputs/generalization_v15'
        call('audit_cctv_dgp_generalization_v15.py',['--results',str(out),'--receipt',str(out/'independent_audit_vm.json')],'audit.log',180)
        r=v.read(out/'results.json');archive=export(True)
        v.require(time.monotonic()<=deadline,'V15 supervisor cap exceeded')
        v.write(root/'supervisor_completion.json',{'complete':True,'protocol_sha256':pin,
            'results_sha256':v.sha(out/'results.json'),'archive_sha256':v.sha(archive),'bytes':archive.stat().st_size,
            'seconds':time.monotonic()-start,'optimizer_updates':0,'backward_calls':0,'production_promoted':False})
        print((root/'supervisor_completion.json').read_text(),flush=True)
    except BaseException as error:
        v.write(root/'supervisor_failure.json',{'complete':False,'protocol_sha256':pin,'error_type':type(error).__name__,
            'error':str(error),'seconds':time.monotonic()-start,'resume_permitted':False,'optimizer_updates':0,'backward_calls':0})
        archive=export(False);v.write(root/'failure_export.json',{'complete':True,'archive_sha256':v.sha(archive),'bytes':archive.stat().st_size})
        raise


if __name__=='__main__':
    p=argparse.ArgumentParser(description=__doc__)
    for name in ['root','parent','capacity']:p.add_argument('--'+name,type=Path,required=True)
    p.add_argument('--expected-sha',required=True);a=p.parse_args()
    supervise(a.root.resolve(),a.parent.resolve(),a.capacity.resolve(),a.expected_sha)
