"""Fresh, evidence-bound VM maintenance transport; never launch model work."""
import argparse
import base64
from datetime import datetime, timezone
import gzip
import hashlib
import json
import os
from pathlib import Path
import shlex
import subprocess
import time

ROOT = Path(__file__).resolve().parents[1]
OUT = ROOT / 'outputs/cctv_dgp_multiscale_archive_cleanup_v1'
HOSTKEY = 'SHA256:pGOZAAcM7szdr2S1Ty+8UORXEPPAhxMDw7jDfzZ4I0E'
INSTANCE_ID = '4410777042005672095'
REMOTE_OUT = '/home/janusdominic0/dgp_multiscale_archive_cleanup_v1_receipts'
CLOUD = Path(os.environ['LOCALAPPDATA']) / 'Google/Cloud SDK/google-cloud-sdk/bin/gcloud.cmd'
BASE = ['--project=forensic-dgp-thesis', '--zone=us-central1-a', '--quiet']
FLAGS = ['-batch', '-hostkey', HOSTKEY]

INVENTORY_SOURCE = '''import json,os,shutil,socket,stat,subprocess,urllib.request
from datetime import datetime,timezone
from pathlib import Path
home=Path('/home/janusdominic0');root=home/'forensic-dgp'
assert Path.home().resolve()==home and socket.gethostname().split('.')[0]=='forensic-dgp-thesis' and os.getuid()==1001 and root.is_dir()
def meta(key):
 return urllib.request.urlopen(urllib.request.Request('http://metadata.google.internal/computeMetadata/v1/instance/'+key,headers={'Metadata-Flavor':'Google'}),timeout=5).read().decode()
assert meta('id')=='4410777042005672095' and meta('machine-type').endswith('/g2-standard-4')
def call(args,cap=20):
 r=subprocess.run(args,capture_output=True,text=True,timeout=cap)
 return dict(exit_code=r.returncode,stdout=r.stdout,stderr=r.stderr)
archives=[];other=[]
for path in sorted(home.iterdir()):
 s=path.lstat()
 row=dict(path=str(path),regular_file=stat.S_ISREG(s.st_mode),symlink=path.is_symlink(),bytes=s.st_size,allocated_bytes=s.st_blocks*512,uid=s.st_uid,nlink=s.st_nlink,inode=s.st_ino,mtime_ns=s.st_mtime_ns)
 if path.name.endswith('.tar.gz'):archives.append(row)
 elif stat.S_ISREG(s.st_mode) and s.st_size>100*1024**2:other.append(row)
disk=shutil.disk_usage(root)
print(json.dumps(dict(complete=True,UTC=datetime.now(timezone.utc).isoformat(),instance_id=meta('id'),hostname=socket.gethostname(),home=str(home),root=str(root),uid=os.getuid(),machine_type=meta('machine-type'),disk=dict(total_bytes=disk.total,used_bytes=disk.used,free_bytes=disk.free),gpu=call(['nvidia-smi','--query-gpu=name,memory.used,memory.total','--format=csv,noheader']),gpu_processes=call(['nvidia-smi','--query-compute-apps=pid,process_name,used_gpu_memory','--format=csv,noheader']),tmux=call(['tmux','list-panes','-a','-F','#S|#P|#{pane_current_command}|#{pane_pid}|#{pane_current_path}|#{pane_dead}']),processes=call(['ps','-eo','pid,ppid,comm,args']),df=call(['df','-B1',str(root)]),home_usage=call(['du','-x','-B1','--max-depth=1',str(home)],120),archives=archives,other_large_home_files=other,venv_python_present=(root/'cctv_dgp_vm_bundle/.venv/bin/python').is_file(),current_multiscale_installed=(root/'cctv_dgp_multiscale_calibration_vm_v1').exists(),files_removed=0,model_gradient_or_training_calls=0,VM_started=False),allow_nan=False))
'''


def sha(path):
    with Path(path).open('rb') as stream:
        return hashlib.file_digest(stream, 'sha256').hexdigest()


def read(path):
    return json.loads(Path(path).read_text(encoding='utf-8'))


def write(path, value):
    with Path(path).open('x', encoding='utf-8', newline='\n') as stream:
        stream.write(json.dumps(value, indent=2, allow_nan=False) + '\n')


def invoke(args, stem, cap, read_only):
    transport = OUT / 'transport'
    assert transport.is_dir()
    env = dict(os.environ)
    env['CLOUDSDK_CORE_CUSTOM_CA_CERTS_FILE'] = str(ROOT / 'scratch/gcloud-windows-roots-v1.pem')
    started = time.monotonic()
    expired = False
    with (transport / (stem + '_stdout.log')).open('xb') as stdout, (transport / (stem + '_stderr.log')).open('xb') as stderr:
        try:
            result = subprocess.run(args, env=env, stdout=stdout, stderr=stderr, timeout=cap)
            code = result.returncode
        except subprocess.TimeoutExpired:
            expired = True
            code = None
    receipt = dict(complete=code == 0, exit_code=code, timeout=expired, cap_seconds=cap,
                   seconds=time.monotonic()-started, read_only=read_only, TLS_validation_enabled=True,
                   hostkey_pinned=HOSTKEY, model_gradient_or_training_calls=0,
                   stdout_sha256=sha(transport / (stem + '_stdout.log')),
                   stderr_sha256=sha(transport / (stem + '_stderr.log')))
    write(transport / (stem + '_transport.json'), receipt)
    assert receipt['complete'], 'Retain failed transport evidence: ' + stem
    return transport / (stem + '_stdout.log')


def ssh_source(source, stem, cap=240):
    encoded = base64.b64encode(source.encode()).decode()
    command = 'python3 -B -c ' + shlex.quote('import base64;exec(base64.b64decode(' + repr(encoded) + '))')
    args = [str(CLOUD), 'compute', 'ssh', 'janusdominic0@forensic-dgp-thesis'] + BASE
    args += ['--ssh-flag=' + flag for flag in FLAGS] + ['--command=' + command]
    assert len(subprocess.list2cmdline(args)) < 8000
    return invoke(args, stem, cap, True)


def api(stem):
    output = invoke([str(CLOUD), 'compute', 'instances', 'describe', 'forensic-dgp-thesis'] + BASE +
                    ['--format=json(id,name,status,machineType,zone,guestAccelerators)'], stem, 60, True)
    value = read(output)
    assert value['id'] == INSTANCE_ID and value['name'] == 'forensic-dgp-thesis'
    assert value['machineType'].endswith('/g2-standard-4') and value['zone'].endswith('/us-central1-a')
    assert value['status'] == 'RUNNING' and value['guestAccelerators'][0]['acceleratorType'].endswith('/nvidia-l4')
    return value


def inventory():
    assert not OUT.exists(), 'Preserve previous maintenance attempt'
    OUT.mkdir()
    (OUT / 'transport').mkdir()
    trust = ROOT / 'outputs/cctv_dgp_vm_storage_cleanup_20261007_v2/hostkey_verification.json'
    old = read(trust)
    assert old['complete'] and old['known_historical_hostkey'] == HOSTKEY and old['current_offered_key_exact_match']
    write(OUT / 'instance_before.json', api('api_before'))
    (OUT / 'inventory_source.py').write_text(INVENTORY_SOURCE, encoding='utf-8', newline='\n')
    guest = read(ssh_source(INVENTORY_SOURCE, 'inventory_before'))
    assert guest['complete'] and guest['gpu']['exit_code'] == guest['gpu_processes']['exit_code'] == 0
    assert 'NVIDIA L4' in guest['gpu']['stdout']
    write(OUT / 'inventory_before.json', guest)
    write(OUT / 'inventory_transport_binding.json', dict(complete=True, source_sha256=sha(OUT / 'inventory_source.py'),
          local_driver_sha256=sha(Path(__file__)), trust_receipt_sha256=sha(trust), hostkey_pinned=HOSTKEY,
          VM_started=False, files_removed=0, model_gradient_or_training_calls=0))
    print(json.dumps(dict(complete=True, free_GiB=guest['disk']['free_bytes']/1024**3,
          GPU_compute_idle=not guest['gpu_processes']['stdout'].strip(), home_archives=len(guest['archives']),
          current_multiscale_installed=guest['current_multiscale_installed'], files_removed=0)), flush=True)


def main():
    p = argparse.ArgumentParser()
    p.add_argument('--phase', choices=['inventory'], required=True)
    a = p.parse_args()
    if a.phase == 'inventory':
        inventory()


if __name__ == '__main__':
    main()
