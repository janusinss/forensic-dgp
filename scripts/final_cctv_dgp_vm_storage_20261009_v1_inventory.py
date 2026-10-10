"""Final bounded disk/runtime inventory after independent cleanup verification."""
import json
import storage_gcloud_20261009_v1 as transport


def main():
    assert transport.transport.read(transport.OUT / 'independent_live_audit.json')['complete']
    source = transport.remote_check() + '''import shutil,time
def call(args):
 result=subprocess.run(args,capture_output=True,text=True,timeout=20)
 return {'exit_code':result.returncode,'stdout':result.stdout,'stderr':result.stderr}
root=Path('/home/janusdominic0/forensic-dgp')
parent=root/'cctv_dgp_actual_step_review_v1_vm'
tail=root/'cctv_dgp_actual_step_tail_v1_vm'
disk=shutil.disk_usage(root)
gpu=call(['nvidia-smi','--query-compute-apps=pid','--format=csv,noheader'])
assert gpu['exit_code']==0 and not gpu['stdout'].strip()
runtime=call([str(root/'cctv_dgp_vm_bundle/.venv/bin/python'),'--version'])
assert runtime['exit_code']==0
protocol=hashlib.sha256((parent/'protocol.json').read_bytes()).hexdigest()
assert protocol=='339c20425b6fa6b4141001aae4ff3010bf9d438acc32eb71ec08f04f001476f5'
failure=hashlib.sha256((parent/'outputs/failure.json').read_bytes()).hexdigest()
assert failure=='fae5988d4470e3f3037b9e1e7e269339878718ec5b47abddba217ca3d42ba4b7'
df=call(['df','-h','/']);assert df['exit_code']==0
tmux=call(['tmux','list-panes','-a','-F','#S|#P|#{pane_current_command}|#{pane_pid}'])
assert tmux['exit_code'] in (0,1)
print(json.dumps({'complete':True,'observed_unix':time.time(),'free_bytes':disk.free,
 'free_GiB':disk.free/1024**3,'df_h_root':df,'GPU_idle':True,'GPU_processes':gpu,'tmux':tmux,
 'shared_VM_Python':runtime,'parent_protocol_sha256':protocol,'original_parent_failure_sha256':failure,
 'tail_root_present':tail.is_dir(),'tail_protocol_sha256':hashlib.sha256((tail/'protocol.json').read_bytes()).hexdigest() if (tail/'protocol.json').is_file() else None,
 'manual_tail_free_requirement_met':disk.free>=2*1024**3,'files_removed_by_this_inventory':0,
 'training_or_diagnostic_launched':False,'model_or_gradient_calls':0},allow_nan=False))
'''
    transport.ssh(source, 'final_disk_runtime', 120)
    value = json.loads((transport.LOGS / 'final_disk_runtime_stdout.log').read_text(encoding='utf-8'))
    assert value['complete'] and value['manual_tail_free_requirement_met'] and value['GPU_idle']
    transport.transport.write(transport.OUT / 'final_disk_runtime_inventory.json', value)
    print({'complete': True, 'free_GiB': value['free_GiB'], 'GPU_idle': True,
           'tail_root_present': value['tail_root_present']}, flush=True)


if __name__ == '__main__':
    main()
