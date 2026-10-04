"""Verified fresh extraction and one-time finite launch on the configured L4."""
import argparse
import hashlib
import json
import os
from pathlib import Path, PurePosixPath
import shlex
import shutil
import subprocess
import sys
import tarfile
import time


def sha(path):
    h=hashlib.sha256()
    with Path(path).open('rb') as f:
        for block in iter(lambda:f.read(1048576),b''):h.update(block)
    return h.hexdigest()


def launch(protocol_sha,archive_sha):
    assert sys.platform=='linux' and os.uname().nodename.split('.')[0]=='forensic-dgp-thesis'
    repo=(Path.home()/'forensic-dgp').resolve();root=repo/'cctv_dgp_direct_codes_vm_v14_r2'
    parent=repo/'cctv_dgp_face_code_fit_vm_v12_r2'
    assert not root.exists() and root.resolve().is_relative_to(repo), 'Preserve previous/partial V14 run'
    assert shutil.disk_usage(repo).free>5*1024**3,'Need5GiB free'
    apps=subprocess.run(['nvidia-smi','--query-compute-apps=pid','--format=csv,noheader'],capture_output=True,text=True,check=True)
    tmux=subprocess.run(['tmux','list-sessions'],capture_output=True,text=True)
    assert not apps.stdout.strip() and tmux.returncode==1 and not tmux.stdout.strip(),'Competing GPU/tmux task'
    archive=Path.home()/'cctv-dgp-direct-codes-v14-r2-execution.tar.gz'
    assert sha(archive)==archive_sha and Path(str(archive)+'.sha256').read_text().split()==[archive_sha,archive.name]
    with tarfile.open(archive,'r:gz') as stream:
        members=stream.getmembers();assert len({m.name for m in members})==len(members) and sum(m.size for m in members)<2*1024**2
        for m in members:
            p=PurePosixPath(m.name)
            assert m.isfile() and p.parts and not p.is_absolute() and '..' not in p.parts and ':' not in m.name and '\\' not in m.name
            assert (root/m.name).resolve().is_relative_to(root)
        root.mkdir();stream.extractall(root,members=members)
    assert sha(root/'direct_code_protocol_v14.json')==protocol_sha
    python=repo/'cctv_dgp_vm_bundle/.venv/bin/python';assert python.is_file()
    code='import sys;sys.path.insert(0,'+repr(str(parent))+');sys.path.insert(0,'+repr(str(root))+');from cctv_dgp_direct_codes_v14 import verify,require_vm;require_vm('+repr(str(root))+');verify('+repr(str(root))+','+repr(str(parent))+','+repr(protocol_sha)+');print("V14 full asset/CUDA verification passed")'
    verified=subprocess.run([str(python),'-c',code],capture_output=True,text=True,check=True,timeout=90)
    print(verified.stdout,flush=True)
    command='exec '+shlex.join([str(python),'-u',str(root/'scripts/run_cctv_dgp_direct_codes_v14_supervised_r2.py'),
        '--root',str(root),'--parent-bundle',str(parent),'--expected-protocol-sha',protocol_sha])+' > '+shlex.quote(str(root/'supervisor_v14.log'))+' 2>&1'
    apps=subprocess.run(['nvidia-smi','--query-compute-apps=pid','--format=csv,noheader'],capture_output=True,text=True,check=True)
    assert not apps.stdout.strip(),'GPU became busy'
    subprocess.run(['tmux','new-session','-d','-s','dgp_direct_codes_v14_r2','-c',str(root),command],check=True)
    receipt={'complete':True,'protocol_sha256':protocol_sha,'archive_sha256':archive_sha,
        'launcher_sha256':sha(Path(__file__)),'session':'dgp_direct_codes_v14_r2','python':str(python),
        'root':str(root),'started_unix_seconds':time.time(),'production_promoted':False}
    with (root/'supervisor_launch_v14.json').open('x') as f:json.dump(receipt,f,indent=2)
    print(json.dumps(receipt),flush=True)


if __name__=='__main__':
    parser=argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--protocol-sha',required=True);parser.add_argument('--archive-sha',required=True)
    args=parser.parse_args();launch(args.protocol_sha,args.archive_sha)
