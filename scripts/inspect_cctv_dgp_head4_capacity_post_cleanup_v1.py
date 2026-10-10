"""Fresh read-only post-cleanup inventory via the existing safe encoded transport."""
import base64
import hashlib
import os
from pathlib import Path
import shlex
import inspect_cctv_dgp_actual_step_review_v1_vm_inventory as t

ROOT=Path(__file__).resolve().parents[1];OUT=ROOT/'outputs/cctv_dgp_head4_archive_cleanup_v1'
SOURCE='''import json,shutil,socket,subprocess,os,time
from pathlib import Path
home=Path.home();root=home/'forensic-dgp'
assert str(home)=='/home/janusdominic0' and socket.gethostname().split('.')[0]=='forensic-dgp-thesis' and os.getuid()==1001
plan=json.loads((home/'dgp_head4_archive_cleanup_v1_plan.json').read_text())
remaining=[r['path'] for r in plan['candidates'] if Path(r['path']).exists()]
gpu=subprocess.run(['nvidia-smi','--query-compute-apps=pid','--format=csv,noheader,nounits'],capture_output=True,text=True,timeout=15)
assert gpu.returncode==0
print(json.dumps({'complete':True,'UTC':time.strftime('%Y-%m-%dT%H:%M:%SZ',time.gmtime()),'hostname':socket.gethostname(),
 'free_bytes':shutil.disk_usage(root).free,'GPU_compute_idle':not gpu.stdout.strip(),'target_archives_remaining':len(remaining),
 'remaining_archives':remaining,'new_capacity_packet_present':(root/'cctv_dgp_head4_capacity_vm_v1').exists(),
 'training_launched':False,'VM_started':False,'post_cleanup_receipt_present':(home/'dgp_head4_archive_cleanup_v1_receipts/cleanup_receipt.json').is_file()},indent=2))
'''


def main():
    assert not (OUT/'post_apply_inventory.json').exists();t.OUT=OUT/'transport'
    assert t.read(OUT/'independent_audit.json')['complete']
    failure=t.read(t.OUT/'post_apply_inventory_transport.json');assert failure['exit_code']==2 and not failure['complete']
    env=dict(os.environ);env['CLOUDSDK_CORE_CUSTOM_CA_CERTS_FILE']=str(ROOT/'scratch/gcloud-windows-roots-v1.pem')
    cloud=Path(os.environ['LOCALAPPDATA'])/'Google/Cloud SDK/google-cloud-sdk/bin/gcloud.cmd'
    encoded=base64.b64encode(SOURCE.encode()).decode()
    command='python3 -B -c '+shlex.quote('import base64;exec(base64.b64decode('+repr(encoded)+'))')
    result=t.invoke([str(cloud),'compute','ssh','janusdominic0@forensic-dgp-thesis','--project=forensic-dgp-thesis','--zone=us-central1-a',
        '--quiet','--ssh-flag=-batch','--ssh-flag=-hostkey','--ssh-flag='+t.HOSTKEY,'--command='+command],env,'post_apply_inventory_r1',90)
    assert result['complete'],'Preserve observation failure; do not repeat cleanup'
    x=t.read(t.OUT/'post_apply_inventory_r1_stdout.log')
    assert x['complete'] and x['post_cleanup_receipt_present'] and x['target_archives_remaining']==0 and x['GPU_compute_idle']
    packet=ROOT/'outputs/cctv_dgp_head4_capacity_vm_v1';p=t.read(packet/'protocol.json')
    install=sum((packet/n).stat().st_size for n in p['assets_sha256'])+(packet/'protocol.json').stat().st_size+(ROOT/'outputs/cctv-dgp-head4-capacity-v1-execution.tar.gz').stat().st_size+32*1024**2
    x.update({'hostkey_pinned':t.HOSTKEY,'TLS_validation_enabled':True,'free_GiB':x['free_bytes']/1024**3,
        'projection_after_upload_install_GiB':(x['free_bytes']-install)/1024**3,'installation_projection_includes_32MiB_overhead':True,
        'transport_receipt_sha256':t.sha(t.OUT/'post_apply_inventory_r1_transport.json'),'readonly_source_sha256':hashlib.sha256(SOURCE.encode()).hexdigest(),
        'prior_shell_quoting_failure_preserved':True,'snapshot_not_future_launch_guarantee':True,'inspector_sha256':t.sha(Path(__file__))})
    assert x['free_bytes']-install>=14*1024**3,'More maintenance needed; retain14GiB bound'
    t.write(OUT/'post_apply_inventory.json',x)
    print({'complete':True,'live_free_GiB':x['free_GiB'],'conservative_after_install_GiB':x['projection_after_upload_install_GiB'],
        'GPU_idle':True,'training_launched':False},flush=True)


if __name__=='__main__':main()
