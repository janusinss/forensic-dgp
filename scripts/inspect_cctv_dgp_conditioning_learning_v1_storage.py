"""Fresh read-only VM maintenance inventory; no model or training execution."""
import base64
from datetime import datetime,timezone
import os
from pathlib import Path
import shlex
import subprocess
import inspect_cctv_dgp_actual_step_review_v1_vm_inventory as transport

ROOT=Path(__file__).resolve().parents[1]
DEST=ROOT/'outputs/cctv_dgp_conditioning_learning_v1_storage/inventory'

SOURCE='''import json,os,stat,subprocess,shutil,time
from pathlib import Path
root=Path.home()/'forensic-dgp'
def call(args):
 p=subprocess.run(args,capture_output=True,text=True,timeout=10)
 return dict(exit_code=p.returncode,stdout=p.stdout,stderr=p.stderr)
gpu=call(['nvidia-smi','--query-compute-apps=pid','--format=csv,noheader'])
hardware=call(['nvidia-smi','--query-gpu=name','--format=csv,noheader'])
ps=call(['ps','-eo','pid=,args='])
work=[s.strip() for s in ps['stdout'].splitlines() if ('cctv_dgp_' in s or 'forensic-dgp/' in s) and not s.strip().startswith(str(os.getpid())+' ')]
panes=call(['tmux','list-panes','-a','-F','#{session_name}\\t#{pane_pid}\\t#{pane_current_command}\\t#{pane_current_path}'])
archives=[]
for p in sorted(Path.home().glob('cctv-dgp-*-results.tar.gz')):
 s=p.lstat()
 archives.append(dict(path=str(p),bytes=s.st_size,uid=s.st_uid,inode=s.st_ino,nlink=s.st_nlink,mtime_ns=s.st_mtime_ns,regular=stat.S_ISREG(s.st_mode)))
d=shutil.disk_usage(root)
print(json.dumps(dict(complete=True,UTC=time.strftime('%Y-%m-%dT%H:%M:%SZ',time.gmtime()),root=str(root),uid=os.getuid(),disk=dict(free_bytes=d.free,total_bytes=d.total,used_bytes=d.used),gpu_processes=gpu,hardware=hardware,workload_candidates=work,tmux_panes=panes,archives=archives,files_removed=0,neural_or_training_launched=False)))
'''


def main():
    assert not DEST.exists();DEST.mkdir(parents=True)
    transport.OUT=DEST
    cloud=Path(os.environ['LOCALAPPDATA'])/'Google/Cloud SDK/google-cloud-sdk/bin/gcloud.cmd'
    ca=ROOT/'scratch/gcloud_windows_trust.pem'
    assert cloud.is_file() and ca.is_file()
    environment=dict(os.environ);environment['CLOUDSDK_CORE_CUSTOM_CA_CERTS_FILE']=str(ca)
    trust=ROOT/'outputs/cctv_dgp_vm_storage_cleanup_20261007_v2/hostkey_verification.json'
    record=transport.read(trust)
    assert record['complete'] and record['current_offered_key_exact_match'] and record['known_historical_hostkey']==transport.HOSTKEY
    base=['--project=forensic-dgp-thesis','--zone=us-central1-a']
    args=[str(cloud),'compute','instances','describe','forensic-dgp-thesis']+base+['--format=json(name,id,status,machineType,zone)']
    result=transport.invoke(args,environment,'api',60)
    assert result['complete'],'Read the retained transport failure before retry'
    instance=transport.read(DEST/'api_stdout.log')
    assert instance['name']=='forensic-dgp-thesis' and instance['id']=='4410777042005672095'
    assert instance['machineType'].endswith('/g2-standard-4') and instance['zone'].endswith('/us-central1-a')
    transport.write(DEST/'instance.json',instance)
    if instance['status']!='RUNNING':
        transport.write(DEST/'unavailable.json',dict(complete=True,status=instance['status'],VM_started=False,files_removed=0))
        print(dict(VM_status=instance['status'],VM_started=False));return
    (DEST/'remote_source.py').write_text(SOURCE,encoding='utf-8',newline='\n')
    code='import base64;exec(base64.b64decode('+repr(base64.b64encode(SOURCE.encode()).decode())+'))'
    args=[str(cloud),'compute','ssh','janusdominic0@forensic-dgp-thesis']+base+['--quiet','--ssh-flag=-batch',
        '--ssh-flag=-hostkey','--ssh-flag='+transport.HOSTKEY,'--command=python3 -B -c '+shlex.quote(code)]
    assert len(subprocess.list2cmdline(args))<8000
    result=transport.invoke(args,environment,'guest',60)
    assert result['complete'],'Read the retained guest transport failure before retry'
    guest=transport.read(DEST/'guest_stdout.log')
    assert guest['complete'] and guest['uid']==1001 and guest['root']=='/home/janusdominic0/forensic-dgp'
    assert guest['gpu_processes']['exit_code']==guest['hardware']['exit_code']==0 and 'NVIDIA L4' in guest['hardware']['stdout']
    transport.write(DEST/'guest_inventory.json',guest)
    transport.write(DEST/'inventory_receipt.json',dict(complete=True,UTC=datetime.now(timezone.utc).isoformat(),
        inspector_sha256=transport.sha(__file__),remote_source_sha256=transport.sha(DEST/'remote_source.py'),
        inventory_sha256=transport.sha(DEST/'guest_inventory.json'),trust_receipt_sha256=transport.sha(trust),
        hostkey_pinned=True,TLS_validation_enabled=True,files_removed=0,VM_started=False,training_launched=False))
    print(dict(complete=True,free_GiB=guest['disk']['free_bytes']/1024**3,archives=len(guest['archives']),
               GPU_idle=not guest['gpu_processes']['stdout'].strip(),files_removed=0))


if __name__=='__main__':main()
