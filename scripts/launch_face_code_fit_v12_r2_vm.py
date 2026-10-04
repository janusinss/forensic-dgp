"""Verified one-time V12 launch on the configured existing VM; no installer."""
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

PROTOCOL='06f056ea4b1b04e6d8c631e5e78df129345c78279230c7946f6b5deb2e226846'
ARCHIVE_SHA='12d8295f7385bebec2a5449c06ab9519781ac5c4be2161e5a5bf673308e0210b'


def sha(path):
    h=hashlib.sha256()
    with Path(path).open('rb') as f:
        for b in iter(lambda:f.read(1048576),b''):h.update(b)
    return h.hexdigest()


def main():
    assert sys.platform=='linux' and os.uname().nodename.split('.')[0]=='forensic-dgp-thesis','Existing Linux VM only'
    repo=(Path.home()/'forensic-dgp').resolve();root=repo/'cctv_dgp_face_code_fit_vm_v12_r2'
    assert root.resolve().is_relative_to(repo) and not root.exists(),'Preserve previous/partial pilot'
    assert shutil.disk_usage(repo).free>5*1024**3,'Need5GiB free'
    apps=subprocess.run(['nvidia-smi','--query-compute-apps=pid,process_name,used_gpu_memory','--format=csv,noheader'],capture_output=True,text=True,check=True)
    sessions=subprocess.run(['tmux','list-sessions'],capture_output=True,text=True)
    assert not apps.stdout.strip() and sessions.returncode==1 and not sessions.stdout.strip(),'Competing GPU/tmux job; stop'
    archive=Path.home()/'cctv-dgp-face-code-v12-r2-execution.tar.gz'
    assert sha(archive)==ARCHIVE_SHA,'Upload fingerprint differs'
    with tarfile.open(archive,'r:gz') as stream:
        members=stream.getmembers();assert len(members)==114
        for m in members:
            parts=PurePosixPath(m.name).parts
            assert m.isfile() and parts[0]==root.name and '..' not in parts and not PurePosixPath(m.name).is_absolute(),'Unsafe member'
            assert (repo/m.name).resolve().is_relative_to(root),'Member escapes fresh package'
        stream.extractall(repo,members=members)
    protocol=json.loads((root/'face_code_fit_protocol_v12.json').read_text());assert sha(root/'face_code_fit_protocol_v12.json')==PROTOCOL
    sources={
        'weights/dgp_v2.pth':repo/'cctv_dgp_mixed_vm_v9_r2/outputs/cctv_dgp_mixed_v9/checkpoints/baseline.pth',
        'weights/w600k_r50.onnx':repo/'cctv_dgp_vm_bundle/weights/w600k_r50.onnx',
        'weights/codeformer.pth':repo/'pretrained_face_prior_v11/codeformer.pth',
        'weights/vqgan_code1024.pth':repo/'pretrained_face_prior_v11/vqgan_code1024.pth'}
    bound=[]
    for name,source in sources.items():
        expected=protocol['external_weights'][name]
        assert source.resolve().is_relative_to(repo) and source.stat().st_size==expected['bytes'] and sha(source)==expected['sha256'],'Missing/changed weight'
        destination=root/name;assert not destination.exists();destination.parent.mkdir(exist_ok=True)
        os.link(source,destination)
        assert sha(destination)==expected['sha256']
        bound.append({'source':str(source),'destination':name,'bytes':expected['bytes'],'sha256':expected['sha256']})
    with (root/'weight_binding_v12.json').open('x') as f:json.dump({'complete':True,'weights':bound},f,indent=2)
    python=repo/'cctv_dgp_vm_bundle/.venv/bin/python';assert python.is_file()
    code='import sys;sys.path.insert(0,'+repr(str(root))+');from cctv_dgp_face_code_fit_v12 import verify,require_vm;require_vm('+repr(str(root))+');verify('+repr(str(root))+','+repr(PROTOCOL)+');print("V12 full asset/CUDA verification passed")'
    verified=subprocess.run([str(python),'-c',code],capture_output=True,text=True,check=True,timeout=90)
    print(verified.stdout,flush=True)
    command='exec '+shlex.join([str(python),'-u',str(root/'scripts/run_cctv_dgp_face_code_fit_v12_r2_supervised.py'),
        '--root',str(root),'--expected-protocol-sha',PROTOCOL])+' > '+shlex.quote(str(root/'supervisor_v12.log'))+' 2>&1'
    # Recheck after transfers/asset verification before creating the only job.
    apps=subprocess.run(['nvidia-smi','--query-compute-apps=pid','--format=csv,noheader'],capture_output=True,text=True,check=True)
    assert not apps.stdout.strip(),'GPU became busy; stop'
    subprocess.run(['tmux','new-session','-d','-s','dgp_face_code_v12_r2','-c',str(root),command],check=True)
    receipt={'complete':True,'protocol_sha256':PROTOCOL,'archive_sha256':ARCHIVE_SHA,
        'launcher_sha256':sha(Path(__file__)),'session':'dgp_face_code_v12_r2','python':str(python),
        'root':str(root),'started_unix_seconds':time.time(),'external_weights_verified':4,'archive_members':114,'production_promoted':False}
    with (root/'supervisor_launch_v12.json').open('x') as f:json.dump(receipt,f,indent=2)
    print(json.dumps(receipt),flush=True)


if __name__=='__main__':main()
