"""One refresh of the inventoried live PID and storage receipts; no VM mutation."""
import json
from pathlib import Path

import storage_gcloud_20261009_v1 as transport

ROOT = Path(__file__).resolve().parents[1]
OUT = ROOT/'outputs/cctv_dgp_v42_vm_storage_preflight_v1'

SOURCE = '''import json,shutil,socket,subprocess,time
from pathlib import Path
home=Path('/home/janusdominic0');root=home/'forensic-dgp';pilot=root/'cctv_dgp_residual_epochs_vm_v42'
assert Path.home().resolve()==home and socket.gethostname().split('.')[0]=='forensic-dgp-thesis'
proc=Path('/proc/1324');cmd=(proc/'cmdline').read_bytes().replace(b'\\0',b' ').decode(errors='replace') if (proc/'cmdline').is_file() else None
live=bool(cmd and str(pilot/'scripts/cctv_dgp_residual_epochs_v42_vm.py') in cmd and '--run' in cmd)
def brief(path,keys):
 if not path.is_file():return None
 try:value=json.loads(path.read_text())
 except json.JSONDecodeError:return {'partial_json_read':True}
 return {k:value.get(k) for k in keys}
receipts={}
for name in ['results.json','failure.json']:
 receipts[name]=brief(pilot/'outputs'/name,['complete','protocol_sha256','optimizer_updates','backwards','gradient_queries','seconds','error','completed_epochs','complete_snapshots','necessary_capacity_pass'])
for name in ['cache_timing.json','timing_update20.json','storage_projection.json']:
 receipts[name]=brief(pilot/'outputs'/name,['projected_seconds','cap_seconds','projected_uncompressed_bytes','cap_bytes'])
log=pilot/'trainer.log';tail=''
if log.is_file():
 with log.open('rb') as f:
  f.seek(max(0,log.stat().st_size-16384));tail=f.read().decode(errors='replace')
 tail='\\n'.join(tail.splitlines()[-30:])
gpu=subprocess.run(['nvidia-smi','--query-compute-apps=pid,process_name,used_gpu_memory','--format=csv,noheader'],capture_output=True,text=True,timeout=15)
export=brief(home/'cctv-dgp-residual-epochs-v42-export.json',['complete','archive_sha256','bytes','optimizer_updates','run_results_present','failure_present','training_success_not_implied'])
print(json.dumps({'complete':True,'observed_unix':time.time(),'inventoried_worker_PID':1324,'same_expected_worker_live':live,'worker_cmdline':cmd,
 'GPU_processes':{'exit_code':gpu.returncode,'stdout':gpu.stdout,'stderr':gpu.stderr},'free_bytes':shutil.disk_usage(root).free,
 'terminal_receipts':receipts,'trainer_log_tail':tail,'export_receipt':export,
 'files_removed':0,'processes_started_or_stopped':0,'local_or_remote_neural_calls_by_observer':0},allow_nan=False))
'''


def main():
    assert transport.transport.read(OUT/'independent_audit.json')['complete']
    transport.OUT = OUT; transport.LOGS = OUT/'transport'
    assert not (OUT/'runtime_refresh.json').exists()
    (OUT/'runtime_refresh_source.py').write_text(SOURCE, encoding='utf-8', newline='\n')
    transport.ssh(SOURCE, 'runtime_refresh', 90)
    value = transport.transport.read(transport.LOGS/'runtime_refresh_stdout.log')
    assert value['complete'] and value['files_removed'] == value['processes_started_or_stopped'] == 0
    transport.transport.write(OUT/'runtime_refresh.json', value)
    print(value, flush=True)


if __name__ == '__main__': main()
