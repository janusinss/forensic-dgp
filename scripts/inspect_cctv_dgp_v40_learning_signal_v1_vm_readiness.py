"""One bounded read-only maintenance inventory; never launch learning or deletion."""
import base64
from datetime import datetime, timezone
import hashlib
import json
import os
from pathlib import Path
import shlex
import subprocess
import time

ROOT = Path(__file__).resolve().parents[1]
OUT = ROOT/'outputs/cctv_dgp_v40_learning_signal_v1_vm_readiness'
HOSTKEY = 'SHA256:pGOZAAcM7szdr2S1Ty+8UORXEPPAhxMDw7jDfzZ4I0E'
SOURCE = '''import json,os,shutil,socket,subprocess,urllib.request
from datetime import datetime,timezone
from pathlib import Path
assert socket.gethostname().split('.')[0]=='forensic-dgp-thesis'
assert str(Path.home())=='/home/janusdominic0'
root=Path.home()/'forensic-dgp'
assert root.is_dir()
def command(args):
 r=subprocess.run(args,capture_output=True,text=True,timeout=15)
 return {'args':args,'exit_code':r.returncode,'stdout':r.stdout,'stderr':r.stderr}
machine=urllib.request.urlopen(urllib.request.Request('http://metadata.google.internal/computeMetadata/v1/instance/machine-type',headers={'Metadata-Flavor':'Google'}),timeout=5).read().decode()
gpu=command(['nvidia-smi','--query-gpu=name,memory.used,memory.total','--format=csv,noheader'])
processes=command(['nvidia-smi','--query-compute-apps=pid,process_name,used_gpu_memory','--format=csv,noheader'])
tmux=command(['tmux','list-panes','-a','-F','#S|#P|#{pane_current_command}|#{pane_pid}'])
disk=shutil.disk_usage(root)
venv=(root/'cctv_dgp_vm_bundle/.venv/bin/python').is_file()
print(json.dumps({'complete':True,'UTC':datetime.now(timezone.utc).isoformat(),'hostname':socket.gethostname(),'home':str(Path.home()),'root':str(root),'machine_type':machine,'disk':{'total_bytes':disk.total,'used_bytes':disk.used,'free_bytes':disk.free},'gpu':gpu,'gpu_processes':processes,'tmux':tmux,'venv_python_present':venv,'files_removed':0,'model_or_gradient_calls':0,'optimizer_updates':0,'diagnostic_or_training_launched':False},allow_nan=False))
'''


def sha(path):
    with Path(path).open('rb') as stream: return hashlib.file_digest(stream, 'sha256').hexdigest()


def read(path): return json.loads(Path(path).read_text(encoding='utf-8'))


def write(path, value):
    with Path(path).open('x', encoding='utf-8', newline='\n') as stream: stream.write(json.dumps(value, indent=2, allow_nan=False)+'\n')


def main():
    assert not OUT.exists(), 'Preserve any prior/partial inventory; no automatic repeat'
    trust = ROOT/'outputs/cctv_dgp_vm_storage_cleanup_20261007_v2/hostkey_verification.json'
    historical = read(trust)
    assert historical['complete'] and historical['known_historical_hostkey'] == HOSTKEY and historical['current_offered_key_exact_match']
    ca = ROOT/'scratch/gcloud_windows_trust.pem'; assert ca.is_file()
    ppath = ROOT/'outputs/cctv_dgp_v40_learning_signal_v1_vm/protocol.json'; protocol = read(ppath)
    assert protocol['budgets']['minimum_free_disk_bytes'] == 2*1024**3
    closed = ROOT/'outputs/cctv_dgp_v40_learning_signal_v1_milestone/independent_closure_audit.json'; assert read(closed)['complete']
    OUT.mkdir(); (OUT/'remote_source.py').write_text(SOURCE, encoding='utf-8')
    environment = dict(os.environ); environment['CLOUDSDK_CORE_CUSTOM_CA_CERTS_FILE'] = str(ca)
    # Use the same verified Windows trust file as historical maintenance. TLS
    # validation stays enabled; no account/SDK configuration is changed.
    cloud = Path(os.environ['LOCALAPPDATA'])/'Google/Cloud SDK/google-cloud-sdk/bin/gcloud.cmd'
    encoded = base64.b64encode(SOURCE.encode()).decode()
    command = 'python3 -B -c '+shlex.quote('import base64;exec(base64.b64decode('+repr(encoded)+'))')
    args = [str(cloud),'compute','ssh','janusdominic0@forensic-dgp-thesis','--project=forensic-dgp-thesis','--zone=us-central1-a',
            '--quiet','--ssh-flag=-batch','--ssh-flag=-hostkey','--ssh-flag='+HOSTKEY,'--command='+command]
    start = time.monotonic(); print({'inventory_started':True,'timeout_seconds':120,'read_only':True}, flush=True)
    with (OUT/'stdout.log').open('xb') as stdout, (OUT/'stderr.log').open('xb') as stderr:
        result = subprocess.run(args, env=environment, stdout=stdout, stderr=stderr, timeout=120)
    receipt = {'complete':result.returncode == 0,'UTC':datetime.now(timezone.utc).isoformat(),'exit_code':result.returncode,
        'seconds':time.monotonic()-start,'cap_seconds':120,'source_sha256':sha(OUT/'remote_source.py'),
        'local_inspector_sha256':sha(Path(__file__)),'trust_bundle_sha256':sha(ca),'historical_hostkey_receipt_sha256':sha(trust),
        'protocol_sha256':sha(ppath),'closure_audit_sha256':sha(closed),'stdout_sha256':sha(OUT/'stdout.log'),
        'stderr_sha256':sha(OUT/'stderr.log'),'TLS_validation_enabled':True,'hostkey_pinned':HOSTKEY,
        'VM_maintenance_connection_attempts_here':1,'files_removed':0,'model_or_gradient_calls':0,'optimizer_updates':0,
        'diagnostic_or_training_launched':False,'goal_complete':False}
    write(OUT/'transport.json',receipt)
    assert result.returncode == 0, (OUT/'stderr.log').read_text(encoding='utf-8',errors='replace')[-1800:]
    value = read(OUT/'stdout.log'); assert value['complete'] and value['machine_type'].endswith('/g2-standard-4')
    assert value['hostname'].split('.')[0] == 'forensic-dgp-thesis' and value['root'] == '/home/janusdominic0/forensic-dgp'
    assert value['gpu']['exit_code'] == value['gpu_processes']['exit_code'] == 0 and 'NVIDIA L4' in value['gpu']['stdout']
    free = value['disk']['free_bytes']; idle = not value['gpu_processes']['stdout'].strip()
    write(OUT/'readiness.json',{'complete':True,'inventory_sha256':sha(OUT/'stdout.log'),'transport_sha256':sha(OUT/'transport.json'),
        'UTC':value['UTC'],'free_bytes':free,'free_GiB':free/1024**3,'required_free_bytes':2*1024**3,'space_requirement_met':free>=2*1024**3,
        'GPU_compute_idle_at_snapshot':idle,'existing_venv_python_present':value['venv_python_present'],
        'snapshot_not_future_workload_guarantee':True,'training_and_diagnostic_manual_unrun':True,'files_removed':0,'goal_complete':False})
    print({'complete':True,'free_GiB':free/1024**3,'required_GiB':2,'GPU_compute_idle':idle,'files_removed':0,'learning_launched':False},flush=True)


if __name__ == '__main__': main()
