"""Bounded existing-VM maintenance inventory with the retained verified host key."""
import base64
from datetime import datetime,timezone
import os
from pathlib import Path
import shlex
import subprocess
import time
import inspect_cctv_dgp_actual_step_review_v1_vm_inventory as transport

ROOT=Path(__file__).resolve().parents[1];OUT=ROOT/'outputs/cctv_dgp_head4_capacity_v1_maintenance'


def main():
    assert not OUT.exists(),'Retain every maintenance attempt'
    trust=ROOT/'outputs/cctv_dgp_vm_storage_cleanup_20261007_v2/hostkey_verification.json'
    t=transport.read(trust)
    assert t['complete'] and t['known_historical_hostkey']==transport.HOSTKEY and t['current_offered_key_exact_match']
    assert transport.sha(ROOT/t['previous_trust_record'])==t['previous_trust_record_sha256']
    assert t['current_VM_ID']=='4410777042005672095'
    assert transport.read(ROOT/'outputs/cctv_dgp_head4_capacity_v1_preparation/independent_packet_audit.json')['complete']
    ca=ROOT/'scratch/gcloud-windows-roots-v1.pem';cloud=Path(os.environ['LOCALAPPDATA'])/'Google/Cloud SDK/google-cloud-sdk/bin/gcloud.cmd'
    assert ca.is_file() and cloud.is_file();env=dict(os.environ);env['CLOUDSDK_CORE_CUSTOM_CA_CERTS_FILE']=str(ca)
    OUT.mkdir();transport.OUT=OUT;start=time.monotonic()
    source=transport.SOURCE.replace('cctv_dgp_actual_step_review_v1_vm','cctv_dgp_head4_capacity_vm_v1')
    (OUT/'remote_inventory_source.py').write_text(source,encoding='utf-8',newline='\n')
    transport.write(OUT/'prior_connection_failure.json',{'cause':'The new external IP has no PuTTY cache entry in batch mode',
        'offered_hostkey':transport.HOSTKEY,'same_key_as_retained_verified_VM':True,
        'guest_attribute_hostkeys_unavailable':True,'current_serial_tail_had_no_fingerprint_lines':True,
        'blind_key_acceptance':False,'global_host_key_cache_modified':False,'files_removed':0,'training_launched':False})
    base=['--project=forensic-dgp-thesis','--zone=us-central1-a']
    print({'API_identity_check':True,'cap_seconds':60},flush=True)
    a=transport.invoke([str(cloud),'compute','instances','describe','forensic-dgp-thesis']+base+
        ['--format=json(name,id,status,machineType,zone)'],env,'api',60)
    assert a['complete'],'Preserve transport failure; no VM start'
    api=transport.read(OUT/'api_stdout.log')
    assert api['name']=='forensic-dgp-thesis' and api['id']==t['current_VM_ID']
    assert api['machineType'].endswith('/g2-standard-4') and api['zone'].endswith('/us-central1-a')
    transport.write(OUT/'instance_status.json',api)
    basis={'complete':True,'UTC':datetime.now(timezone.utc).isoformat(),'instance_id':api['id'],'VM_status':api['status'],
        'hostkey_pinned':transport.HOSTKEY,'TLS_validation_enabled':True,'trust_receipt_sha256':transport.sha(trust),
        'inspector_sha256':transport.sha(Path(__file__)),'files_removed':0,'VM_started':False,'training_launched':False}
    if api['status']!='RUNNING':
        transport.write(OUT/'availability.json',{**basis,'guest_inventory_available':False});print(basis,flush=True);return
    encoded=base64.b64encode(source.encode()).decode();remote='python3 -B -c '+shlex.quote('import base64;exec(base64.b64decode('+repr(encoded)+'))')
    args=[str(cloud),'compute','ssh','janusdominic0@forensic-dgp-thesis']+base+[
        '--quiet','--ssh-flag=-batch','--ssh-flag=-hostkey','--ssh-flag='+transport.HOSTKEY,'--command='+remote]
    assert len(subprocess.list2cmdline(args))<8000
    print({'guest_inventory':True,'cap_seconds':120,'pinned_previous_verified_host_key':True},flush=True)
    result=transport.invoke(args,env,'guest',120);assert result['complete'],'Preserve failed guest receipt; no automatic retry'
    guest=transport.read(OUT/'guest_stdout.log')
    assert guest['complete'] and guest['files_removed']==0 and not guest['diagnostic_or_training_launched']
    assert guest['gpu']['exit_code']==guest['gpu_processes']['exit_code']==0 and 'NVIDIA L4' in guest['gpu']['stdout']
    transport.write(OUT/'guest_inventory.json',guest);free=guest['disk']['free_bytes']
    transport.write(OUT/'availability.json',{**basis,'guest_inventory_available':True,'free_bytes':free,'free_GiB':free/1024**3,
        'minimum_free_after_install_bytes':14*1024**3,'GPU_compute_idle_at_snapshot':not guest['gpu_processes']['stdout'].strip(),
        'snapshot_not_future_launch_guarantee':True,'guest_inventory_sha256':transport.sha(OUT/'guest_inventory.json'),
        'seconds':time.monotonic()-start})
    print({'complete':True,'free_GiB':free/1024**3,'GPU_compute_idle':not guest['gpu_processes']['stdout'].strip(),
        'files_removed':0,'training_launched':False},flush=True)


if __name__=='__main__':main()
