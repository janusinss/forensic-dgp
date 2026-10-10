"""Bind complete Windows backups; perform only authorized pinned VM maintenance."""
import argparse
from pathlib import Path
import shlex
import time
import cctv_dgp_bank_archive_cleanup_v1 as t
import cctv_dgp_multiscale_archive_cleanup_v1 as transport

ROOT=t.ROOT
OUT=ROOT/'outputs/cctv_dgp_bank_split_offload_v1'
RETURN=ROOT/'outputs/cctv_dgp_bank_comparison_v1_return'
REMOTE='/home/janusdominic0/dgp_bank_split_offload_v1'
WORKER=ROOT/'scripts/cctv_dgp_bank_split_offload_v1_vm.py'

def cloud_copy(src,dest,stem):
    return t.invoke([str(t.CLOUD),'compute','scp']+t.BASE+['--scp-flag='+f for f in t.FLAGS]+[str(src),str(dest)],stem,180,False)
def vm_phase(plan_path,phase,stem):
    command='python3 -B '+shlex.quote(REMOTE+'_vm.py')+' --plan '+shlex.quote(REMOTE+'_plan.json')+' --plan-sha '+t.sha(plan_path)+' --phase '+phase
    return t.invoke([str(t.CLOUD),'compute','ssh','janusdominic0@forensic-dgp-thesis']+t.BASE+['--ssh-flag='+f for f in t.FLAGS]+['--command='+command],stem,960,phase=='inventory')

def backups(plan):
    for r in plan['candidates']:
        f=ROOT/r['local_backup'];assert f.is_file() and not f.is_symlink() and f.stat().st_size==r['bytes'] and t.sha(f)==r['sha256']
    for n,h in t.read(OUT/'local_protected_sha256.json').items():assert t.sha(ROOT/n)==h,n

def prepare():
    assert not OUT.exists();OUT.mkdir();(OUT/'transport').mkdir()
    audit=t.read(ROOT/'outputs/cctv_dgp_bank_comparison_v1_analysis/independent_audit.json')
    imp=t.read(ROOT/'outputs/cctv_dgp_bank_comparison_v1_analysis/import.json')
    assert audit['complete'] and imp['complete'] and audit['optimizer_updates_in_VM']==0
    assert audit['manifest_sha256']==imp['export_manifest_sha256']
    t.write(OUT/'instance_before.json',t.api('api_before'))
    manifest=t.read(RETURN/'export_manifest.json');entries=[]
    for n,h in manifest['files'].items():
        if not n.startswith('outputs/bank_comparison_v1/baseline/') or Path(n).suffix not in {'.npz','.npy','.png'}:continue
        f=RETURN/n;assert t.sha(f)==h
        entries.append(dict(path='/home/janusdominic0/forensic-dgp/cctv_dgp_bank_comparison_v1_vm/'+n,bytes=f.stat().st_size,sha256=h,local_backup=f.relative_to(ROOT).as_posix()))
    assert len(entries)==929
    archive=ROOT/'outputs/cctv-dgp-bank-comparison-v1-results.tar.gz'
    export=t.read(ROOT/'outputs/cctv-dgp-bank-comparison-v1-export.json')
    assert export['complete'] and t.sha(archive)==export['archive_sha256']==imp['archive_sha256'] and archive.stat().st_size==export['bytes']
    entries.append(dict(path='/home/janusdominic0/'+archive.name,bytes=export['bytes'],sha256=export['archive_sha256'],local_backup=archive.relative_to(ROOT).as_posix()))
    original=ROOT/'outputs/cctv_dgp_bank_comparison_v1_vm';p=t.read(original/'protocol.json')
    keep={str((original/n).relative_to(ROOT).as_posix()):h for n,h in p['assets_sha256'].items()}
    keep[str((original/'protocol.json').relative_to(ROOT).as_posix())]=t.sha(original/'protocol.json')
    # Include permanent incumbent and current app bindings, established prior to maintenance.
    keep.update(t.read(ROOT/'scratch/dgp-bank-app-preflight-20261010-v1/flow.json')['sources'])
    keep['outputs/cctv_dgp_face_code_fit_vm_v12_r2/weights/dgp_v2.pth']='646fbb11674e8882ea41e251d6e6e7d468c963979b48c93132def8292834e24b'
    for n,h in keep.items():assert t.sha(ROOT/n)==h
    t.write(OUT/'local_protected_sha256.json',keep)
    basic=dict(complete=True,instance_id=t.INSTANCE_ID,authorization_scope='inactive RGB output packs plus one archive; full verified local backup',worker_sha256=t.sha(WORKER),candidates=entries,original_return_import_sha256=t.sha(ROOT/'outputs/cctv_dgp_bank_comparison_v1_analysis/import.json'),independent_return_audit_sha256=t.sha(ROOT/'outputs/cctv_dgp_bank_comparison_v1_analysis/independent_audit.json'),local_protected_sha256=t.sha(OUT/'local_protected_sha256.json'),manual_training_pending=True,model_gradient_or_training_calls=0,maintenance_cap_seconds=900)
    initial=OUT/'candidate_request.json';t.write(initial,basic)
    cloud_copy(WORKER,'janusdominic0@forensic-dgp-thesis:'+REMOTE+'_vm.py','worker_upload')
    cloud_copy(initial,'janusdominic0@forensic-dgp-thesis:'+REMOTE+'_plan.json','request_upload')
    fresh=t.read(vm_phase(initial,'inventory','candidate_inventory'))
    assert fresh['complete'] and fresh['files_removed']==0 and len(fresh['candidates'])==930
    t.write(OUT/'candidate_inventory.json',fresh)
    plan={**basic,'candidates':fresh['candidates'],'candidate_inventory_sha256':t.sha(OUT/'candidate_inventory.json'),'allocated_reclaim_bytes':sum(r['allocated_bytes'] for r in fresh['candidates'])}
    t.write(OUT/'plan.json',plan)
    print(dict(complete=True,phase='prepare',candidates=930,reclaim_GiB=plan['allocated_reclaim_bytes']/1024**3,free_before_GiB=fresh['free_bytes']/1024**3,files_removed=0),flush=True)

def main():
    global OUT
    a=argparse.ArgumentParser();a.add_argument('--phase',choices=['prepare','verify','apply','post'],required=True);a.add_argument('--receipt-binding-repair',action='store_true');args=a.parse_args();phase=args.phase;started=time.monotonic()
    if args.receipt_binding_repair:OUT=ROOT/'outputs/cctv_dgp_bank_split_offload_v1_r1'
    transport.OUT=OUT;t.OUT=OUT
    if phase=='prepare':prepare();return
    assert not (OUT/(phase+'_execution.json')).exists()
    plan_path=OUT/'plan.json';plan=t.read(plan_path);assert plan['worker_sha256']==t.sha(WORKER)
    if phase in {'verify','apply'}:backups(plan)
    t.write(OUT/('instance_'+phase+'.json'),t.api('api_'+phase))
    if phase=='verify':
        cloud_copy(plan_path,'janusdominic0@forensic-dgp-thesis:'+REMOTE+'_plan.json','bound_plan_upload')
        receipt=t.read(vm_phase(plan_path,phase,'verify'));assert receipt['complete'] and receipt['files_removed']==0 and receipt['candidate_count']==930
        t.write(OUT/'verify_readback.json',receipt)
    elif phase=='apply':
        assert t.read(OUT/'verify_readback.json')['plan_sha256']==t.sha(plan_path)
        result=t.read(vm_phase(plan_path,phase,'apply'));assert result['complete'] and result['files_removed']==930
        t.write(OUT/'apply_readback.json',result)
        for n in ['dgp-bank-split-offload-v1-receipts.tar.gz','dgp-bank-split-offload-v1-export.json']:
            cloud_copy('janusdominic0@forensic-dgp-thesis:/home/janusdominic0/'+n,OUT/n,'download_'+n.replace('.','_'))
    else:
        assert t.read(OUT/'independent_audit.json')['complete']
        guest=t.read(t.ssh_source(t.INVENTORY_SOURCE,'inventory_after'))
        assert guest['gpu_processes']['exit_code']==0 and not guest['gpu_processes']['stdout'].strip()
        assert not any(r['path']=='/home/janusdominic0/cctv-dgp-bank-comparison-v1-results.tar.gz' for r in guest['archives'])
        # Reserve the route-A upload and complete installed packet plus 32 MiB.
        r=t.read(ROOT/'outputs/cctv_dgp_bank_comparison_v1_r2_preparation.json')['routes'][0]
        packet=ROOT/'outputs'/r['name'];p=t.read(packet/'protocol.json')
        install=r['archive_bytes']+sum((packet/n).stat().st_size for n in p['assets_sha256'])+(packet/'protocol.json').stat().st_size+32*1024**2
        guest.update(free_GiB=guest['disk']['free_bytes']/1024**3,install_reservation_bytes=install,conservative_A_post_install_GiB=(guest['disk']['free_bytes']-install)/1024**3,manual_A_space_requirement_met=guest['disk']['free_bytes']-install>=16*1024**3,current_snapshot_not_future_launch_guarantee=True)
        t.write(OUT/'inventory_after.json',guest)
        print(dict(complete=True,free_GiB=guest['free_GiB'],conservative_A_post_install_GiB=guest['conservative_A_post_install_GiB'],manual_A_space_requirement_met=guest['manual_A_space_requirement_met']),flush=True)
    t.write(OUT/(phase+'_execution.json'),dict(complete=True,phase=phase,plan_sha256=t.sha(plan_path),driver_sha256=t.sha(__file__),worker_sha256=t.sha(WORKER),model_gradient_or_training_calls=0,VM_started=False,seconds=time.monotonic()-started))
    print(dict(complete=True,phase=phase,model_gradient_or_training_calls=0),flush=True)

if __name__=='__main__':main()
