"""L4-only finite fitting, independent audit and preserved success/failure export."""
import argparse
import os
from pathlib import Path
import signal
import subprocess
import sys
import tarfile
import time


def run(root, expected_sha):
    sys.path.insert(0,str(root))
    from cctv_dgp_face_code_fit_v12 import require_vm,verify,read,write,sha,require
    require_vm(root);p=verify(root,expected_sha)
    import torch,torchvision
    require((root.parent/'cctv_dgp_vm_bundle/cuda_runtime_before.txt').read_text().splitlines()==[torch.__version__,torchvision.__version__],'Preserve existing CUDA runtime')
    require(not (root/'supervisor_execution_v12.json').exists(),'Preserve previous supervisor execution')
    started=time.monotonic();deadline=started+900
    write(root/'supervisor_execution_v12.json',{'protocol_sha256':expected_sha,'pid':os.getpid(),
        'source_sha256':sha(Path(__file__)),'torch':torch.__version__,'torchvision':torchvision.__version__,
        'started_unix_seconds':time.time(),'trainer_cap_seconds':600,'supervisor_cap_seconds':900,
        'optimizer_updates_budget':300,'production_promoted':False})
    def call(args,name):
        with (root/name).open('x',encoding='utf-8') as log:
            child=subprocess.Popen(args,cwd=root,stdout=log,stderr=subprocess.STDOUT,start_new_session=True)
            print({'child_pid':child.pid,'log':name},flush=True)
            try:code=child.wait(timeout=max(1,deadline-time.monotonic()))
            except subprocess.TimeoutExpired:
                os.killpg(child.pid,signal.SIGINT)
                try:child.wait(timeout=15)
                except subprocess.TimeoutExpired:os.killpg(child.pid,signal.SIGKILL);child.wait()
                raise TimeoutError('V12 supervisor exceeded900 seconds')
        print((root/name).read_text()[-1800:],flush=True)
        if code:raise subprocess.CalledProcessError(code,args)
    def export(success):
        archive=root/('cctv-dgp-face-code-v12-r2-results.tar.gz' if success else 'cctv-dgp-face-code-v12-r2-failure.tar.gz')
        names=['face_code_fit_protocol_v12.json','face_code_fit_protocol_v12.sha256','schedule_v12.json',
            'cctv_dgp_face_code_fit_v12.py','dgp_face_code_conditioner_v11.py','scripts','models','third_party',
            'outputs/cctv_dgp_face_code_fit_v12','training_v12.log','audit_v12.log',
            'supervisor_execution_v12.json','supervisor_failure_v12.json','training_audit_completion_v12.json',
            'supervisor_launch_v12.json','weight_binding_v12.json']
        with tarfile.open(archive,'x:gz',compresslevel=3) as stream:
            for name in names:
                if success and time.monotonic()>deadline:raise TimeoutError('V12 export exceeded900 seconds')
                path=root/name
                if path.exists():stream.add(path,arcname=name)
        Path(str(archive)+'.sha256').write_text(sha(archive)+'  '+archive.name+'\n',encoding='ascii',newline='\n')
        return archive
    try:
        call([sys.executable,'-u',str(root/'scripts/train_cctv_dgp_face_code_fit_v12_r2.py'),'--root',str(root),'--expected-protocol-sha',expected_sha],'training_v12.log')
        out=root/'outputs/cctv_dgp_face_code_fit_v12'
        call([sys.executable,'-u',str(root/'scripts/audit_cctv_dgp_face_code_fit_v12.py'),'--root',str(root),'--expected-protocol-sha',expected_sha,
            '--results',str(out),'--receipt',str(out/'independent_audit_vm.json')],'audit_v12.log')
        result=read(out/'results.json');write(root/'training_audit_completion_v12.json',{'complete':True,
            'protocol_sha256':expected_sha,'optimizer_updates':result['optimizer_updates'],'seconds':time.monotonic()-started,'production_promoted':False})
        archive=export(True);require(time.monotonic()<=deadline,'Export budget exceeded')
        write(root/'supervisor_completion_v12.json',{'complete':True,'protocol_sha256':expected_sha,
            'archive_sha256':sha(archive),'bytes':archive.stat().st_size,'seconds':time.monotonic()-started,
            'optimizer_updates':300,'production_promoted':False})
        print((root/'supervisor_completion_v12.json').read_text(),flush=True)
    except Exception as error:
        write(root/'supervisor_failure_v12.json',{'complete':False,'error_type':type(error).__name__,'error':str(error),
            'seconds':time.monotonic()-started,'resume_permitted':False})
        try:
            archive=export(False);write(root/'failure_export_v12.json',{'complete':True,'archive_sha256':sha(archive),'bytes':archive.stat().st_size})
        except Exception as exporting:write(root/'failure_export_error_v12.json',{'complete':False,'error':str(exporting)})
        raise


if __name__=='__main__':
    parser=argparse.ArgumentParser(description=__doc__);parser.add_argument('--root',type=Path,required=True);parser.add_argument('--expected-protocol-sha',required=True)
    args=parser.parse_args();run(args.root.resolve(),args.expected_protocol_sha)
