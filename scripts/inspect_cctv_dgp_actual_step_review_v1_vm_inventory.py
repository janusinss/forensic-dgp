"""One bounded authorized maintenance inventory; no deletion or VM job launch."""
import base64
from datetime import datetime,timezone
import hashlib
import json
import os
from pathlib import Path
import shlex
import subprocess
import time

ROOT=Path(__file__).resolve().parents[1]
OUT=ROOT/'outputs/cctv_dgp_actual_step_review_v1_vm_inventory'
HOSTKEY='SHA256:pGOZAAcM7szdr2S1Ty+8UORXEPPAhxMDw7jDfzZ4I0E'
SOURCE='''import hashlib,json,os,shutil,socket,stat,subprocess,time,urllib.request
from datetime import datetime,timezone
from pathlib import Path
home=Path('/home/janusdominic0');root=home/'forensic-dgp'
assert socket.gethostname().split('.')[0]=='forensic-dgp-thesis' and Path.home().resolve()==home and root.is_dir()
def call(args):
 r=subprocess.run(args,capture_output=True,text=True,timeout=15)
 return {'exit_code':r.returncode,'stdout':r.stdout,'stderr':r.stderr}
machine=urllib.request.urlopen(urllib.request.Request('http://metadata.google.internal/computeMetadata/v1/instance/machine-type',headers={'Metadata-Flavor':'Google'}),timeout=5).read().decode()
assert machine.rstrip().endswith('/g2-standard-4')
archives=[]
for path in home.iterdir():
 if path.name.startswith('cctv-dgp-') and path.name.endswith('.tar.gz'):
  s=path.lstat()
  archives.append({'path':str(path),'regular_file':stat.S_ISREG(s.st_mode),'symlink':path.is_symlink(),'bytes':s.st_size,'allocated_bytes':s.st_blocks*512,'uid':s.st_uid,'nlink':s.st_nlink,'inode':s.st_ino,'mtime_ns':s.st_mtime_ns})
disk=shutil.disk_usage(root);pilot=root/'cctv_dgp_actual_step_review_v1_vm';protocol=pilot/'protocol.json'
current={'root_present':pilot.is_dir(),'protocol_sha256':hashlib.sha256(protocol.read_bytes()).hexdigest() if protocol.is_file() else None,'review_log_present':(pilot/'review.log').is_file(),'results_present':(pilot/'outputs/results.json').is_file(),'failure_present':(pilot/'outputs/failure.json').is_file()}
print(json.dumps({'complete':True,'UTC':datetime.now(timezone.utc).isoformat(),'hostname':socket.gethostname(),'home':str(home),'root':str(root),'current_uid':os.getuid(),'machine_type':machine,'disk':{'total_bytes':disk.total,'used_bytes':disk.used,'free_bytes':disk.free},'gpu':call(['nvidia-smi','--query-gpu=name,memory.used,memory.total','--format=csv,noheader']),'gpu_processes':call(['nvidia-smi','--query-compute-apps=pid,process_name,used_gpu_memory','--format=csv,noheader']),'tmux':call(['tmux','list-panes','-a','-F','#S|#P|#{pane_current_command}|#{pane_pid}']),'processes':call(['ps','-eo','pid,ppid,comm,args']),'venv_python_present':(root/'cctv_dgp_vm_bundle/.venv/bin/python').is_file(),'home_transfer_archive_metadata_only':archives,'current_diagnostic_metadata':current,'files_removed':0,'VM_started':False,'diagnostic_or_training_launched':False,'model_or_gradient_calls':0},allow_nan=False))
'''


def sha(path):
    with Path(path).open('rb') as stream:return hashlib.file_digest(stream,'sha256').hexdigest()


def read(path):return json.loads(Path(path).read_text(encoding='utf-8'))


def write(path,value):
    with Path(path).open('x',encoding='utf-8',newline='\n') as stream:stream.write(json.dumps(value,indent=2,allow_nan=False)+'\n')


def invoke(args,environment,stem,cap):
    started=time.monotonic();expired=False
    with (OUT/(stem+'_stdout.log')).open('xb') as stdout,(OUT/(stem+'_stderr.log')).open('xb') as stderr:
        try:result=subprocess.run(args,env=environment,stdout=stdout,stderr=stderr,timeout=cap);code=result.returncode
        except subprocess.TimeoutExpired:expired=True;code=None
    value={'complete':code==0,'exit_code':code,'observation_timeout':expired,'seconds':time.monotonic()-started,'cap_seconds':cap,
       'stdout_sha256':sha(OUT/(stem+'_stdout.log')),'stderr_sha256':sha(OUT/(stem+'_stderr.log')),'read_only':True,
       'TLS_validation_enabled':True,'files_removed':0,'VM_started':False,'diagnostic_or_training_launched':False}
    write(OUT/(stem+'_transport.json'),value);return value


def main():
    assert not OUT.exists(),'Retain every previous inventory attempt'
    trust=ROOT/'outputs/cctv_dgp_vm_storage_cleanup_20261007_v2/hostkey_verification.json'
    recorded=read(trust);assert recorded['complete'] and recorded['known_historical_hostkey']==HOSTKEY and recorded['current_offered_key_exact_match']
    ca=ROOT/'scratch/gcloud_windows_trust.pem';assert ca.is_file()
    prepared=ROOT/'outputs/cctv_dgp_actual_step_review_v1_prepared_milestone'
    assert read(prepared/'independent_closure_audit_r1.json')['complete']
    cloud=Path(os.environ['LOCALAPPDATA'])/'Google/Cloud SDK/google-cloud-sdk/bin/gcloud.cmd';assert cloud.is_file()
    environment=dict(os.environ);environment['CLOUDSDK_CORE_CUSTOM_CA_CERTS_FILE']=str(ca)
    OUT.mkdir();(OUT/'remote_inventory_source.py').write_text(SOURCE,encoding='utf-8',newline='\n')
    started=time.monotonic();print({'API_inventory_started':True,'timeout_seconds':60,'read_only':True},flush=True)
    args=[str(cloud),'compute','instances','describe','forensic-dgp-thesis','--project=forensic-dgp-thesis','--zone=us-central1-a',
          '--format=json(name,id,status,machineType,zone,disks.deviceName,disks.source)']
    api=invoke(args,environment,'api',60)
    if not api['complete']:
        print({'API_complete':False,'observation_timeout':api['observation_timeout'],'VM_started':False});return
    instance=read(OUT/'api_stdout.log');assert instance['name']=='forensic-dgp-thesis' and instance['id']=='4410777042005672095'
    assert instance['machineType'].endswith('/g2-standard-4') and instance['zone'].endswith('/us-central1-a')
    write(OUT/'instance_status.json',instance)
    basis={'complete':True,'UTC':datetime.now(timezone.utc).isoformat(),'instance_status':instance['status'],
           'hostkey_pinned':HOSTKEY,'trust_receipt_sha256':sha(trust),'trust_bundle_sha256':sha(ca),'inspector_sha256':sha(Path(__file__)),
           'previous_prepared_milestone_sha256':sha(prepared/'milestone.json'),'previous_prepared_closure_sha256':sha(prepared/'independent_closure_audit_r1.json'),
           'previous_goal_turn_classification':'progress: frozen packet, prospective audit and handoff closure completed',
           'files_removed':0,'VM_started':False,'diagnostic_or_training_launched':False,'model_or_gradient_calls':0,'goal_complete':False}
    if instance['status']!='RUNNING':
        write(OUT/'availability.json',{**basis,'fresh_guest_inventory_available':False,'seconds':time.monotonic()-started})
        print({'complete':True,'VM_status':instance['status'],'guest_inventory_available':False,'VM_started':False},flush=True);return
    encoded=base64.b64encode(SOURCE.encode()).decode()
    remote='python3 -B -c '+shlex.quote('import base64;exec(base64.b64decode('+repr(encoded)+'))')
    args=[str(cloud),'compute','ssh','janusdominic0@forensic-dgp-thesis','--project=forensic-dgp-thesis','--zone=us-central1-a',
          '--quiet','--ssh-flag=-batch','--ssh-flag=-hostkey','--ssh-flag='+HOSTKEY,'--command='+remote]
    assert len(subprocess.list2cmdline(args))<8000
    print({'guest_inventory_started':True,'timeout_seconds':120,'read_only':True},flush=True)
    transport=invoke(args,environment,'guest',120)
    if not transport['complete']:
        write(OUT/'availability.json',{**basis,'fresh_guest_inventory_available':False,'guest_observation_timeout':transport['observation_timeout'],'seconds':time.monotonic()-started})
        print({'guest_inventory_complete':False,'observation_timeout':transport['observation_timeout'],'VM_started':False},flush=True);return
    guest=read(OUT/'guest_stdout.log');assert guest['complete'] and guest['gpu']['exit_code']==guest['gpu_processes']['exit_code']==0
    assert 'NVIDIA L4' in guest['gpu']['stdout'] and guest['files_removed']==0 and not guest['diagnostic_or_training_launched']
    write(OUT/'guest_inventory.json',guest);free=guest['disk']['free_bytes']
    write(OUT/'availability.json',{**basis,'fresh_guest_inventory_available':True,'free_bytes':free,'free_GiB':free/1024**3,
          'packet_free_requirement_bytes':6*1024**3,'meets_requirement_before_install':free>=6*1024**3,
          'GPU_compute_idle_at_snapshot':not guest['gpu_processes']['stdout'].strip(),'snapshot_not_future_workload_guarantee':True,
          'guest_inventory_sha256':sha(OUT/'guest_inventory.json'),'seconds':time.monotonic()-started})
    print({'complete':True,'free_GiB':free/1024**3,'GPU_compute_idle':not guest['gpu_processes']['stdout'].strip(),
           'home_transfer_archives':len(guest['home_transfer_archive_metadata_only']),'files_removed':0,'learning_launched':False},flush=True)


if __name__=='__main__':main()
