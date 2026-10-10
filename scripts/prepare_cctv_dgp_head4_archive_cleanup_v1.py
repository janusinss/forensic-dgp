"""Verify full local archive copies before freezing an exact maintenance plan."""
import gzip
from pathlib import Path
import sys
import time
import inspect_cctv_dgp_actual_step_review_v1_vm_inventory as t

ROOT=Path(__file__).resolve().parents[1];OUT=ROOT/'outputs/cctv_dgp_head4_archive_cleanup_v1'


def main():
    assert not OUT.exists();start=time.monotonic();OUT.mkdir()
    inventory=ROOT/'outputs/cctv_dgp_head4_capacity_v1_maintenance/guest_inventory.json';g=t.read(inventory)
    assert g['complete'] and g['gpu_processes']['exit_code']==0 and not g['gpu_processes']['stdout'].strip()
    assert t.read(ROOT/'outputs/cctv_dgp_head4_reactivation_v1_independent_audit_r2.json')['complete']
    rows=[]
    for a in g['home_transfer_archive_metadata_only']:
        assert a['regular_file'] and not a['symlink'] and a['uid']==1001 and a['nlink']==1
        name=Path(a['path']).name;f=ROOT/'outputs'/name;assert f.is_file() and not f.is_symlink() and f.stat().st_size==a['bytes']
        digest=t.sha(f);parts=Path(str(f)+'.sha256').read_text().split();assert parts==[digest,name]
        export=None
        if name.endswith('-results.tar.gz'):
            export=ROOT/'outputs'/name.replace('-results.tar.gz','-export.json');e=t.read(export)
            assert e['complete'] and e['archive_sha256']==digest and e['bytes']==a['bytes']
        total=0
        with gzip.open(f,'rb') as stream:
            for b in iter(lambda:stream.read(1024**2),b''):total+=len(b);assert total<30*1024**3 and time.monotonic()-start<600
        rows.append({**a,'sha256':digest,'local_backup':f.relative_to(ROOT).as_posix(),'full_gzip_CRC_verified':True,
            'uncompressed_stream_bytes':total,'export_sha256':t.sha(export) if export else None,
            'prior_transfer_copy_only':True,'unpacked_research_directory_not_a_target':True})
        print({'verified_local_archives':len(rows),'of':len(g['home_transfer_archive_metadata_only'])},flush=True)
    assert len(rows)==15
    t.write(OUT/'plan.json',{'complete':True,'authorization_scope':'inventory-first hash-bound inactive archive copies only',
        'research_root':'/home/janusdominic0/forensic-dgp','inventory_sha256':t.sha(inventory),
        'remote_script_sha256':t.sha(ROOT/'scripts/cctv_dgp_head4_archive_cleanup_v1_vm.py'),
        'candidates':rows,'estimated_allocated_reclaim_bytes':sum(x['allocated_bytes'] for x in rows),
        'current_new_capacity_packet_preserved':True,'current_capacity_packet_sha256':t.read(ROOT/'outputs/cctv_dgp_head4_capacity_v1_preparation/preparation.json')['packet_sha256'],
        'completed_prior_gradient_audit_sha256':t.sha(ROOT/'outputs/cctv_dgp_head4_reactivation_v1_independent_audit_r2.json'),
        'research_caches_checkpoints_splits_failures_runtime_not_delete_targets':True,
        'whole_research_metadata_and_critical_hashes_required_unchanged':True,'worker_stop_seconds':900,
        'training_launched':False,'files_removed':0,'seconds':time.monotonic()-start})
    print({'complete':True,'candidate_count':15,'reclaim_GiB':sum(x['allocated_bytes'] for x in rows)/1024**3,
        'plan_sha256':t.sha(OUT/'plan.json'),'files_removed':0},flush=True)


if __name__=='__main__':main()
