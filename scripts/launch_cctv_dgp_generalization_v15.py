"""Verified one-time extraction/launch on the user's existing idle L4 VM."""
import argparse
import hashlib
import json
import os
from pathlib import Path,PurePosixPath
import shlex
import shutil
import subprocess
import sys
import tarfile
import time


def sha(path):
    h=hashlib.sha256()
    with path.open('rb') as f:
        for b in iter(lambda:f.read(1048576),b''):h.update(b)
    return h.hexdigest()


def launch(pin,archive_sha):
    assert sys.platform=='linux' and os.uname().nodename.split('.')[0]=='forensic-dgp-thesis'
    repo=(Path.home()/'forensic-dgp').resolve();root=repo/'cctv_dgp_generalization_vm_v15'
    parent=repo/'cctv_dgp_face_code_fit_vm_v12_r2'
    capacity=repo/'cctv_dgp_direct_codes_vm_v14_r2/outputs/cctv_dgp_direct_codes_v14_r2'
    assert not root.exists() and root.resolve().is_relative_to(repo),'Preserve existing V15'
    assert shutil.disk_usage(repo).free>5*1024**3,'Need5GiB free'
    def idle():
        apps=subprocess.run(['nvidia-smi','--query-compute-apps=pid','--format=csv,noheader'],capture_output=True,text=True,check=True)
        tmux=subprocess.run(['tmux','list-sessions'],capture_output=True,text=True)
        assert not apps.stdout.strip() and tmux.returncode==1 and not tmux.stdout.strip(),'Competing GPU/tmux task'
    idle();archive=Path.home()/'cctv-dgp-generalization-v15-execution.tar.gz'
    assert sha(archive)==archive_sha and Path(str(archive)+'.sha256').read_text().split()==[archive_sha,archive.name]
    with tarfile.open(archive,'r:gz') as t:
        members=t.getmembers();assert len({m.name for m in members})==len(members) and sum(m.size for m in members)<512*1024**2
        for m in members:
            p=PurePosixPath(m.name)
            assert m.isfile() and p.parts and not p.is_absolute() and '..' not in p.parts and ':' not in m.name and '\\' not in m.name
            assert (root/m.name).resolve().is_relative_to(root)
        root.mkdir();t.extractall(root,members=members)
    assert sha(root/'generalization_protocol_v15.json')==pin
    plan=json.loads((root/'generalization_protocol_v15.json').read_text())
    assert sha(Path(__file__))==plan['assets_sha256']['scripts/launch_cctv_dgp_generalization_v15.py']
    python=repo/'cctv_dgp_vm_bundle/.venv/bin/python';assert python.is_file()
    code='import sys;sys.path.insert(0,'+repr(str(parent))+');sys.path.insert(0,'+repr(str(root))+');from cctv_dgp_generalization_v15 import verify;from cctv_dgp_targets_v6 import require_vm;require_vm('+repr(str(root))+');verify('+repr(str(root))+','+repr(str(parent))+','+repr(pin)+');print("V15 full asset/CUDA preflight passed")'
    result=subprocess.run([str(python),'-c',code],capture_output=True,text=True,check=True,timeout=90)
    print(result.stdout,flush=True);idle()
    command='exec '+shlex.join([str(python),'-u',str(root/'scripts/supervise_cctv_dgp_generalization_v15.py'),
        '--root',str(root),'--parent',str(parent),'--capacity',str(capacity),'--expected-sha',pin])+' > '+shlex.quote(str(root/'supervisor.log'))+' 2>&1'
    subprocess.run(['tmux','new-session','-d','-s','dgp_generalization_v15','-c',str(root),command],check=True)
    receipt={'complete':True,'protocol_sha256':pin,'archive_sha256':archive_sha,'launcher_sha256':sha(Path(__file__)),
        'session':'dgp_generalization_v15','root':str(root),'started_unix':time.time(),'optimizer_updates':0,'production_promoted':False}
    with (root/'supervisor_launch.json').open('x') as f:json.dump(receipt,f,indent=2)
    print(json.dumps(receipt),flush=True)


if __name__=='__main__':
    p=argparse.ArgumentParser(description=__doc__);p.add_argument('--protocol-sha',required=True);p.add_argument('--archive-sha',required=True)
    a=p.parse_args();launch(a.protocol_sha,a.archive_sha)
