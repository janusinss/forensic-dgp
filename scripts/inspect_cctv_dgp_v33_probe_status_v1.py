"""Read-only check of the manual V33 packet, process and export handles."""
import base64
import hashlib
import json
from pathlib import Path
import shlex
import storage_gcloud_20261006_v1 as transport

ROOT = Path(__file__).resolve().parents[1]
OUT = ROOT / 'outputs/cctv_dgp_v33_probe_status_v1'
KEY = 'SHA256:pGOZAAcM7szdr2S1Ty+8UORXEPPAhxMDw7jDfzZ4I0E'
REMOTE = r'''
from datetime import datetime, timezone
import hashlib,json,os,shutil,subprocess
from pathlib import Path
assert os.uname().nodename.split('.')[0]=='forensic-dgp-thesis'
assert str(Path.home())=='/home/janusdominic0'
root=Path.home()/'forensic-dgp/cctv_dgp_loss_cone_probe_v33_vm'
expected='ba8d1f87cae38cac8e1b3b893378a185c6126c25151dadbcfa0ebb373a51adee'
def command(args):
 result=subprocess.run(args,capture_output=True,text=True,timeout=10)
 return {'exit_code':result.returncode,'stdout':result.stdout[-12000:],'stderr':result.stderr[-2000:]}
def tail(path):
 if not path.is_file():return {'exists':False}
 with path.open('rb') as f:
  f.seek(max(0,path.stat().st_size-16000));data=f.read().decode('utf-8',errors='replace')
 return {'exists':True,'bytes':path.stat().st_size,'last_lines':data.splitlines()[-20:]}
processes=[]
for process in Path('/proc').iterdir():
 if not process.name.isdigit():continue
 try:
  args=(process/'cmdline').read_bytes().decode(errors='replace').split('\0')
  if any(a.endswith('cctv_dgp_loss_cone_probe_v33_vm.py') or a.endswith('scripts/run_v33_probe.sh') for a in args):
   processes.append({'pid':int(process.name),'args':[a for a in args if a],'status':(process/'status').read_text().splitlines()[:3]})
 except (FileNotFoundError,PermissionError,ProcessLookupError):pass
protocol=root/'protocol.json';digest=hashlib.sha256(protocol.read_bytes()).hexdigest() if protocol.is_file() else None
receipt={'complete':True,'snapshot_utc':datetime.now(timezone.utc).isoformat(),'read_only':True,'VM_writes':0,
 'training_started_by_agent':False,'root_exists':root.is_dir(),'protocol_sha256':digest,
 'protocol_matches_if_present':digest==expected if digest else None,'probe_processes':processes,
 'tmux_panes':command(['tmux','list-panes','-t','dgp_cone_v33','-F','#{session_name} #{pane_id} #{pane_pid} #{pane_current_command}']),
 'tmux_last_lines':command(['tmux','capture-pane','-p','-t','dgp_cone_v33','-S','-18']),
 'probe_log':tail(root/'probe.log'),'outputs_present':sorted(p.name for p in (root/'outputs').iterdir())[:30] if (root/'outputs').is_dir() else [],
 'GPU':command(['nvidia-smi','--query-compute-apps=pid,process_name,used_gpu_memory','--format=csv,noheader']),
 'free_bytes':shutil.disk_usage(Path.home()/'forensic-dgp').free}
export=Path.home()/'cctv-dgp-loss-cone-probe-v33-export.json'
if export.is_file() and export.stat().st_size<16000:receipt['export_receipt']=json.loads(export.read_text())
print(json.dumps(receipt,indent=2))
'''


def main():
    OUT.mkdir(exist_ok=False); transport.OUT = OUT
    source = 'import base64;exec(base64.b64decode(' + repr(base64.b64encode(REMOTE.encode()).decode()) + '))'
    transport.run(['compute', 'ssh', 'janusdominic0@forensic-dgp-thesis'] + transport.BASE +
                  ['--ssh-flag=-batch', '--ssh-flag=-hostkey', '--ssh-flag=' + KEY,
                   '--command=python3 -B -c ' + shlex.quote(source)], 'status', 60)
    remote = json.loads((OUT / 'status.log').read_text(encoding='utf-8-sig'))
    assert remote['complete'] and remote['read_only'] and remote['VM_writes'] == 0
    assert remote['protocol_matches_if_present'] in (True, None)
    local = {'complete': True, 'reader_sha256': transport.sha(Path(__file__)),
             'remote_source_sha256': hashlib.sha256(REMOTE.encode()).hexdigest(), 'remote_status_sha256': transport.sha(OUT / 'status.log'),
             'remote': remote, 'local_model_or_gradient_calls': 0, 'training_started_by_agent': False, 'files_deleted': 0}
    with (OUT / 'status_receipt.json').open('x', encoding='utf-8') as stream: stream.write(json.dumps(local, indent=2) + '\n')
    print(json.dumps({k: remote[k] for k in ['snapshot_utc', 'root_exists', 'probe_processes', 'free_bytes', 'GPU']}, indent=2))


if __name__ == '__main__': main()
