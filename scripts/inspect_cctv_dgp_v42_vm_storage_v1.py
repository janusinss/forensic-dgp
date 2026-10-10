"""Fresh read-only maintenance inventory for V42; never start a VM or training."""
import json
from pathlib import Path
import time

import storage_gcloud_20261009_v1 as transport

ROOT = Path(__file__).resolve().parents[1]
OUT = ROOT/'outputs/cctv_dgp_v42_vm_storage_preflight_v1'

SOURCE = '''import hashlib,json,os,shutil,socket,subprocess,time,urllib.request
from pathlib import Path
home=Path('/home/janusdominic0');root=home/'forensic-dgp'
assert Path.home().resolve()==home and socket.gethostname().split('.')[0]=='forensic-dgp-thesis' and root.is_dir()
machine=urllib.request.urlopen(urllib.request.Request('http://metadata.google.internal/computeMetadata/v1/instance/machine-type',headers={'Metadata-Flavor':'Google'}),timeout=5).read().decode().strip()
assert machine.endswith('/g2-standard-4')
def call(args):
 r=subprocess.run(args,capture_output=True,text=True,timeout=20)
 return {'exit_code':r.returncode,'stdout':r.stdout,'stderr':r.stderr}
disk=shutil.disk_usage(root);pilot=root/'cctv_dgp_residual_epochs_vm_v42'
python=root/'cctv_dgp_vm_bundle/.venv/bin/python'
paths=['protocol.json','outputs/results.json','outputs/failure.json','trainer.log','supervisor_receipt.json']
gpu=call(['nvidia-smi','--query-compute-apps=pid,process_name,used_gpu_memory','--format=csv,noheader'])
tmux=call(['tmux','list-panes','-a','-F','#S|#P|#{pane_current_command}|#{pane_pid}'])
processes=call(['ps','-eo','pid,ppid,comm,args'])
selected=[line for line in processes['stdout'].splitlines() if any(word in line for word in ['train_cctv','run_v42.sh','cctv_dgp_residual_epochs_v42_vm.py','cctv_dgp_actual_step_tail_v1_vm.py'])]
print(json.dumps({'complete':True,'observed_unix':time.time(),'home':str(home),'hostname':socket.gethostname(),
 'machine_type':machine,'free_bytes':disk.free,'free_GiB':disk.free/1024**3,
 'required_after_install_bytes':8*1024**3,'free_after_install_requirement_met_now':disk.free>=8*1024**3,
 'df_h':call(['df','-h',str(root)]),'GPU':call(['nvidia-smi','--query-gpu=name,memory.used,memory.total','--format=csv,noheader']),
 'GPU_processes':gpu,'GPU_idle':gpu['exit_code']==0 and not gpu['stdout'].strip(),
 'tmux':tmux,'training_or_diagnostic_process_lines':selected,
 'shared_python_present':python.is_file(),'shared_python_version':call([str(python),'--version']) if python.is_file() else None,
 'V42_root_present':pilot.is_dir(),'V42_paths_present':{name:(pilot/name).is_file() for name in paths},
 'V42_protocol_sha256':hashlib.sha256((pilot/'protocol.json').read_bytes()).hexdigest() if (pilot/'protocol.json').is_file() else None,
 'files_removed':0,'VM_started':False,'training_or_diagnostic_launched':False,'model_or_gradient_calls':0},allow_nan=False))
'''


def main():
    assert not OUT.exists(), 'Preserve every inventory attempt'
    prepared = ROOT/'outputs/cctv_dgp_residual_epochs_v42_milestone/independent_readback.json'
    assert transport.transport.read(prepared)['complete']
    OUT.mkdir(); transport.OUT = OUT; transport.LOGS = OUT/'transport'
    (OUT/'remote_inventory_source.py').write_text(SOURCE, encoding='utf-8', newline='\n')
    start = time.monotonic()
    args = ['compute', 'instances', 'describe', 'forensic-dgp-thesis']+transport.BASE+[
        '--format=json(name,id,status,machineType,zone)']
    transport.invoke(args, 'instance_status', 60)
    instance = transport.transport.read(transport.LOGS/'instance_status_stdout.log')
    assert instance['name']=='forensic-dgp-thesis' and instance['id']=='4410777042005672095'
    assert instance['machineType'].endswith('/g2-standard-4') and instance['zone'].endswith('/us-central1-a')
    transport.transport.write(OUT/'instance_status.json', instance)
    if instance['status'] != 'RUNNING':
        transport.transport.write(OUT/'availability.json', {'complete': True,
            'instance_status': instance['status'], 'VM_started': False,
            'guest_inventory_available': False, 'files_removed': 0,
            'training_or_diagnostic_launched': False, 'goal_complete': False})
        print({'VM_status': instance['status'], 'VM_started': False}, flush=True); return
    transport.ssh(SOURCE, 'guest_inventory', 120)
    guest = transport.transport.read(transport.LOGS/'guest_inventory_stdout.log')
    assert guest['complete'] and guest['files_removed']==0 and guest['VM_started'] is False
    assert guest['GPU']['exit_code']==0 and 'NVIDIA L4' in guest['GPU']['stdout']
    assert guest['df_h']['exit_code']==0
    transport.transport.write(OUT/'guest_inventory.json', guest)
    transport.transport.write(OUT/'availability.json', {'complete': True,
        'instance_status': instance['status'], 'guest_inventory_available': True,
        'free_GiB': guest['free_GiB'], 'free_after_install_requirement_met_now': guest['free_after_install_requirement_met_now'],
        'GPU_idle': guest['GPU_idle'], 'V42_root_present': guest['V42_root_present'],
        'fresh_storage_measurement_only': True, 'future_free_space_not_guaranteed': True,
        'inspector_sha256': transport.transport.sha(Path(__file__)),
        'remote_inventory_source_sha256': transport.transport.sha(OUT/'remote_inventory_source.py'),
        'transport_source_sha256': transport.transport.sha(ROOT/'scripts/storage_gcloud_20261009_v1.py'),
        'prepared_milestone_sha256': transport.transport.sha(prepared),
        'previous_goal_turn_classification': 'progress: new V42 packet, independent audit and handoff complete',
        'files_removed': 0, 'VM_started': False, 'training_or_diagnostic_launched': False,
        'model_or_gradient_calls': 0, 'goal_complete': False, 'seconds': time.monotonic()-start})
    print({'complete': True, 'free_GiB': guest['free_GiB'], 'GPU_idle': guest['GPU_idle'],
           'V42_root_present': guest['V42_root_present'], 'files_removed': 0}, flush=True)


if __name__ == '__main__': main()
