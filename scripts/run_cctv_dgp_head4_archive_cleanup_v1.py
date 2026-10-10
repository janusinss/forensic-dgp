"""Authorized, exact backup-bound maintenance; never run model or training code."""
import base64
from pathlib import Path
import os
import shlex
import sys
import time
import inspect_cctv_dgp_actual_step_review_v1_vm_inventory as t

ROOT=Path(__file__).resolve().parents[1];OUT=ROOT/'outputs/cctv_dgp_head4_archive_cleanup_v1';RPC=OUT/'transport'


def main():
    assert not RPC.exists();RPC.mkdir();t.OUT=RPC;start=time.monotonic()
    plan_path=OUT/'plan.json';plan=t.read(plan_path);pin=t.sha(plan_path)
    script=ROOT/'scripts/cctv_dgp_head4_archive_cleanup_v1_vm.py';assert t.sha(script)==plan['remote_script_sha256']
    assert len(plan['candidates'])==15
    for row in plan['candidates']:
        assert row['full_gzip_CRC_verified'] and t.sha(ROOT/row['local_backup'])==row['sha256']
    readiness=t.read(ROOT/'outputs/cctv_dgp_head4_capacity_v1_maintenance/availability.json')
    assert readiness['VM_status']=='RUNNING' and readiness['instance_id']=='4410777042005672095'
    assert readiness['hostkey_pinned']==t.HOSTKEY
    cloud=Path(os.environ['LOCALAPPDATA'])/'Google/Cloud SDK/google-cloud-sdk/bin/gcloud.cmd'
    env=dict(os.environ);env['CLOUDSDK_CORE_CUSTOM_CA_CERTS_FILE']=str(ROOT/'scratch/gcloud-windows-roots-v1.pem')
    base=['--project=forensic-dgp-thesis','--zone=us-central1-a','--quiet']
    remote_script='/home/janusdominic0/cctv_dgp_head4_archive_cleanup_v1_vm.py';remote_plan='/home/janusdominic0/dgp_head4_archive_cleanup_v1_plan.json'
    for file,dest,label in [(script,remote_script,'script_upload'),(plan_path,remote_plan,'plan_upload')]:
        result=t.invoke([str(cloud),'compute','scp']+base+['--scp-flag=-batch','--scp-flag=-hostkey','--scp-flag='+t.HOSTKEY,
            str(file),'janusdominic0@forensic-dgp-thesis:'+dest],env,label,120)
        assert result['complete'],label
    def ssh(command,label,cap):
        result=t.invoke([str(cloud),'compute','ssh','janusdominic0@forensic-dgp-thesis']+base+
            ['--ssh-flag=-batch','--ssh-flag=-hostkey','--ssh-flag='+t.HOSTKEY,'--command='+command],env,label,cap)
        assert result['complete'],'Retain maintenance failure/partial ledger: '+label
    for phase in ['verify','apply']:
        if phase=='apply':
            receipt=t.read(RPC/'verify_readback_stdout.log')
            assert receipt['complete'] and receipt['plan_sha256']==pin and receipt['files_removed']==0
            assert receipt['candidate_count']==15 and receipt['protected_metadata_entries']>100 and receipt['protected_byte_hashes']>100
            for row in plan['candidates']:assert t.sha(ROOT/row['local_backup'])==row['sha256']
        print({'maintenance_phase':phase,'exact_archives':15,'model_or_training_calls':0},flush=True)
        ssh('python3 -B '+shlex.quote(remote_script)+' --plan '+shlex.quote(remote_plan)+' --plan-sha '+pin+' --phase '+phase,phase,960)
        if phase=='verify':
            code='from pathlib import Path;print(Path("/home/janusdominic0/dgp_head4_archive_cleanup_v1_receipts/verify_receipt.json").read_text())'
            ssh('python3 -B -c '+shlex.quote(code),'verify_readback',60)
    for name in ['dgp-head4-archive-cleanup-v1-receipts.tar.gz','dgp-head4-archive-cleanup-v1-export.json']:
        destination=OUT/name;assert not destination.exists()
        result=t.invoke([str(cloud),'compute','scp']+base+['--scp-flag=-batch','--scp-flag=-hostkey','--scp-flag='+t.HOSTKEY,
            'janusdominic0@forensic-dgp-thesis:/home/janusdominic0/'+name,str(destination)],env,'download_'+name.replace('.','_'),180)
        assert result['complete'],'Preserve download failure; do not repeat deletion'
    t.write(OUT/'execution.json',{'complete':True,'plan_sha256':pin,'remote_source_sha256':t.sha(script),
        'local_driver_sha256':t.sha(Path(__file__)),'declared_scope':'15 exact backed-up home transfer archive copies',
        'training_launched':False,'independent_audit_pending':True,'seconds':time.monotonic()-start})
    print({'complete':True,'cleanup_return_downloaded':True,'training_launched':False},flush=True)


if __name__=='__main__':main()
