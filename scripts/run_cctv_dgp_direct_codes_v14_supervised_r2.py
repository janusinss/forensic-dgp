"""Existing L4 only: finite training, arithmetic audit and success/failure export."""
import argparse
import os
from pathlib import Path
import signal
import subprocess
import sys
import tarfile
import time


def run(root, parent, expected_sha):
    sys.path.insert(0,str(parent)); sys.path.insert(0,str(root))
    from cctv_dgp_direct_codes_v14 import require_vm, verify, require, read, write, sha, PLAN
    require_vm(root); p = verify(root, parent, expected_sha)
    import torch, torchvision
    require((root.parent/'cctv_dgp_vm_bundle/cuda_runtime_before.txt').read_text().splitlines()
            == [torch.__version__,torchvision.__version__], 'Preserve existing CUDA runtime')
    started = time.monotonic(); deadline = started+p['design']['supervisor_cap_seconds']
    write(root/'supervisor_execution_v14.json', {'complete':True,'protocol_sha256':expected_sha,
          'pid':os.getpid(),'torch':torch.__version__,'torchvision':torchvision.__version__,
          'started_unix_seconds':time.time(),'budget_seconds':900,'updates_budget':1000,
          'source_sha256':sha(Path(__file__))})

    def call(args, name, timeout):
        with (root/name).open('x',encoding='utf-8') as log:
            child = subprocess.Popen(args,cwd=root,stdout=log,stderr=subprocess.STDOUT,start_new_session=True)
            print({'child_pid':child.pid,'log':name},flush=True)
            try: code = child.wait(timeout=min(timeout,max(1,deadline-time.monotonic())))
            except subprocess.TimeoutExpired:
                os.killpg(child.pid,signal.SIGINT)
                try:child.wait(timeout=15)
                except subprocess.TimeoutExpired:os.killpg(child.pid,signal.SIGKILL);child.wait()
                raise TimeoutError('Finite V14 subprocess budget exceeded')
        print((root/name).read_text()[-2400:],flush=True)
        if code: raise subprocess.CalledProcessError(code,args)

    def export(success):
        archive = root/('cctv-dgp-direct-codes-v14-r2-results.tar.gz' if success else 'cctv-dgp-direct-codes-v14-r2-failure.tar.gz')
        names = list(p['sources_sha256']) + [PLAN,'direct_code_protocol_v14.sha256','schedule_v14.json',
            'outputs/cctv_dgp_direct_codes_v14_r2','training_v14.log','audit_v14.log',
            'supervisor_execution_v14.json','supervisor_launch_v14.json','supervisor_failure_v14.json']
        with tarfile.open(archive,'x:gz',compresslevel=3) as stream:
            for name in names:
                if success:require(time.monotonic()<=deadline,'V14 export budget exceeded')
                path=root/name
                if path.exists():stream.add(path,arcname=name,recursive=True)
        Path(str(archive)+'.sha256').write_text(sha(archive)+'  '+archive.name+'\n',encoding='ascii',newline='\n')
        return archive

    try:
        call([sys.executable,'-u',str(root/'scripts/train_cctv_dgp_direct_codes_v14_r2.py'),
              '--root',str(root),'--parent-bundle',str(parent),'--expected-protocol-sha',expected_sha], 'training_v14.log',620)
        out=root/'outputs/cctv_dgp_direct_codes_v14_r2'
        call([sys.executable,'-u',str(root/'scripts/audit_cctv_dgp_direct_codes_v14_r2.py'),
              '--root',str(root),'--parent-bundle',str(parent),'--expected-protocol-sha',expected_sha,
              '--results',str(out),'--receipt',str(out/'independent_audit_vm.json')], 'audit_v14.log',120)
        result=read(out/'results.json'); archive=export(True)
        require(time.monotonic()<=deadline,'Total V14 budget exceeded')
        write(root/'supervisor_completion_v14.json', {'complete':True,'protocol_sha256':expected_sha,
              'results_sha256':sha(out/'results.json'),'archive_sha256':sha(archive),'bytes':archive.stat().st_size,
              'seconds':time.monotonic()-started,'optimizer_updates':result['optimizer_updates'],
              'backward_calls':result['backward_calls'],'production_promoted':False})
        print((root/'supervisor_completion_v14.json').read_text(),flush=True)
    except BaseException as error:
        write(root/'supervisor_failure_v14.json', {'complete':False,'protocol_sha256':expected_sha,
              'error_type':type(error).__name__,'error':str(error),'seconds':time.monotonic()-started,
              'resume_permitted':False})
        try:
            archive=export(False)
            write(root/'failure_export_v14.json', {'complete':True,'archive_sha256':sha(archive),'bytes':archive.stat().st_size})
        except BaseException as exporting:
            write(root/'failure_export_error_v14.json', {'complete':False,'error':str(exporting)})
        raise


if __name__ == '__main__':
    parser=argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--root',type=Path,required=True)
    parser.add_argument('--parent-bundle',type=Path,required=True)
    parser.add_argument('--expected-protocol-sha',required=True)
    args=parser.parse_args()
    run(args.root.resolve(),args.parent_bundle.resolve(),args.expected_protocol_sha)
