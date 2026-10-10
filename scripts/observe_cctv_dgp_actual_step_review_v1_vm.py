"""Read only the confirmed live manual review handles and bounded log tails."""
import argparse
import base64
from datetime import datetime,timezone
import json
import os
from pathlib import Path
import re
import shlex
import subprocess
import time
from cctv_dgp_actual_step_review_v1_contract import NAME,STEM,read,write,sha

ROOT=Path(__file__).resolve().parents[1]
OUT=ROOT/'outputs/cctv_dgp_actual_step_review_v1_vm_observations'
HOSTKEY='SHA256:pGOZAAcM7szdr2S1Ty+8UORXEPPAhxMDw7jDfzZ4I0E'
PIN='339c20425b6fa6b4141001aae4ff3010bf9d438acc32eb71ec08f04f001476f5'
SOURCE='''import hashlib,json,os,socket,time
from datetime import datetime,timezone
from pathlib import Path
home=Path('/home/janusdominic0');root=home/'forensic-dgp/cctv_dgp_actual_step_review_v1_vm';pin='339c20425b6fa6b4141001aae4ff3010bf9d438acc32eb71ec08f04f001476f5'
assert socket.gethostname().split('.')[0]=='forensic-dgp-thesis' and Path.home().resolve()==home
assert hashlib.sha256((root/'protocol.json').read_bytes()).hexdigest()==pin
def process(pid,script,mode):
 path=Path('/proc')/str(pid)
 try:
  args=path.joinpath('cmdline').read_bytes().decode().rstrip('\\0').split('\\0');status=path.joinpath('status').read_text()
  ticks=int(path.joinpath('stat').read_text().rsplit(') ',1)[1].split()[19]);cwd=path.joinpath('cwd').resolve(strict=True)
  owned=int(next(x for x in status.splitlines() if x.startswith('Uid:')).split()[1])==os.getuid()
  match=owned and cwd==root and pin in args and str(root) in args and any(a==script or a==str(root/script) for a in args) and (mode is None or mode in args)
  return {'pid':pid,'present':True,'expected_handle_live':match,'start_ticks':ticks,'elapsed_seconds':float(Path('/proc/uptime').read_text().split()[0])-ticks/os.sysconf('SC_CLK_TCK'),'args':args if match else None}
 except (FileNotFoundError,ProcessLookupError):return {'pid':pid,'present':False,'expected_handle_live':False}
def small(path):
 if not path.is_file():return None
 assert not path.is_symlink() and path.stat().st_size<1024**2
 return json.loads(path.read_text())
def tail(path):
 if not path.is_file():return None
 with path.open('rb') as stream:
  stream.seek(0,2);stream.seek(max(0,stream.tell()-12000));text=stream.read().decode(errors='replace')
 return text.splitlines()[-12:]
parent=process(1301,'scripts/supervise_cctv_dgp_actual_steps_v1.py',None);review=process(1305,'scripts/review_cctv_dgp_actual_steps_v1_vm.py','--review')
children=[]
if parent['expected_handle_live']:
 try:ids=Path('/proc/1301/task/1301/children').read_text().split()
 except FileNotFoundError:ids=[]
 for pid in ids:children.append(process(int(pid),'scripts/review_cctv_dgp_actual_steps_v1_vm.py','--export'))
failure=small(root/'outputs/failure.json');results=small(root/'outputs/results.json')
print(json.dumps({'complete':True,'UTC':datetime.now(timezone.utc).isoformat(),'protocol_sha256':pin,'review_handle':review,'supervisor_handle':parent,'export_handles':children,'review_log_tail':tail(root/'review.log'),'export_log_tail':tail(root/'export.log'),'preflight':small(root/'outputs/preflight.json'),'cache_receipt':small(root/'outputs/cache_receipt.json'),'timing_projection':small(root/'outputs/timing_projection.json'),'failure':failure,'results':results,'supervisor_receipt':small(root/'supervisor_receipt.json'),'export_receipt':small(home/'cctv-dgp-actual-step-review-v1-export.json'),'review_exit_code':(root/'review_exit_code.txt').read_text().strip() if (root/'review_exit_code.txt').is_file() else None,'files_removed':0,'model_or_gradient_calls':0,'diagnostic_or_training_launched':False},allow_nan=False))
'''


def main(sample):
    assert re.fullmatch('[a-z0-9_]{1,40}',sample)
    inventory=read(ROOT/'outputs/cctv_dgp_actual_step_review_v1_vm_inventory/guest_inventory.json')
    assert inventory['complete'] and inventory['current_diagnostic_metadata']['protocol_sha256']==PIN
    assert '1305,' in inventory['gpu_processes']['stdout'] and 'supervise_cctv_dgp_actual_steps_v1.py' in inventory['processes']['stdout']
    OUT.mkdir(exist_ok=True);folder=OUT/sample;assert not folder.exists();folder.mkdir()
    (folder/'remote_source.py').write_text(SOURCE,encoding='utf-8',newline='\n')
    ca=ROOT/'scratch/gcloud_windows_trust.pem';assert ca.is_file()
    environment=dict(os.environ);environment['CLOUDSDK_CORE_CUSTOM_CA_CERTS_FILE']=str(ca)
    cloud=Path(os.environ['LOCALAPPDATA'])/'Google/Cloud SDK/google-cloud-sdk/bin/gcloud.cmd'
    encoded=base64.b64encode(SOURCE.encode()).decode();remote='python3 -B -c '+shlex.quote('import base64;exec(base64.b64decode('+repr(encoded)+'))')
    args=[str(cloud),'compute','ssh','janusdominic0@forensic-dgp-thesis','--project=forensic-dgp-thesis','--zone=us-central1-a',
          '--quiet','--ssh-flag=-batch','--ssh-flag=-hostkey','--ssh-flag='+HOSTKEY,'--command='+remote]
    assert len(subprocess.list2cmdline(args))<8000
    started=time.monotonic();expired=False
    print({'observation_started':True,'confirmed_review_pid':1305,'timeout_seconds':60},flush=True)
    with (folder/'stdout.log').open('xb') as stdout,(folder/'stderr.log').open('xb') as stderr:
        try:result=subprocess.run(args,env=environment,stdout=stdout,stderr=stderr,timeout=60);code=result.returncode
        except subprocess.TimeoutExpired:expired=True;code=None
    write(folder/'transport.json',{'complete':code==0,'exit_code':code,'observation_timeout':expired,'seconds':time.monotonic()-started,'cap_seconds':60,
          'stdout_sha256':sha(folder/'stdout.log'),'stderr_sha256':sha(folder/'stderr.log'),'source_sha256':sha(folder/'remote_source.py'),
          'local_inspector_sha256':sha(Path(__file__)),'TLS_validation_enabled':True,'hostkey_pinned':HOSTKEY,'read_only':True,
          'files_removed':0,'VM_started':False,'diagnostic_or_training_launched':False})
    if code!=0:print({'observation_complete':False,'observation_timeout':expired,'handle_terminal_not_inferred':True});return
    value=read(folder/'stdout.log');assert value['complete'] and value['protocol_sha256']==PIN and value['files_removed']==0
    write(folder/'observation.json',value)
    live=value['review_handle']['expected_handle_live'] or value['supervisor_handle']['expected_handle_live'] or any(p['expected_handle_live'] for p in value['export_handles'])
    print({'complete':True,'live_expected_handle':live,'results_present':value['results'] is not None,'failure_present':value['failure'] is not None,
           'export_complete':bool(value['export_receipt'] and value['export_receipt']['complete']),'seconds':time.monotonic()-started},flush=True)
    print('\n'.join(value['review_log_tail'][-4:] if value['review_log_tail'] else []),flush=True)


if __name__=='__main__':
    parser=argparse.ArgumentParser();parser.add_argument('--sample',required=True);a=parser.parse_args();main(a.sample)
