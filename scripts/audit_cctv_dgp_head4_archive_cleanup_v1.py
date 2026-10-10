"""Independent cleanup ledger, backup, protected-state and artifact audit."""
import gzip
import hashlib
import json
from pathlib import Path,PurePosixPath
import tarfile
import time
import inspect_cctv_dgp_actual_step_review_v1_vm_inventory as t

ROOT=Path(__file__).resolve().parents[1];OUT=ROOT/'outputs/cctv_dgp_head4_archive_cleanup_v1'


def canonical(x):return hashlib.sha256(json.dumps(x,sort_keys=True,separators=(',',':')).encode()).hexdigest()


def main():
    start=time.monotonic();destination=OUT/'remote_receipts';assert not destination.exists() and not (OUT/'independent_audit.json').exists()
    plan=t.read(OUT/'plan.json');pin=t.sha(OUT/'plan.json');execution=t.read(OUT/'execution.json')
    assert execution['complete'] and execution['plan_sha256']==pin and execution['remote_source_sha256']==plan['remote_script_sha256']
    assert t.sha(ROOT/'scripts/cctv_dgp_head4_archive_cleanup_v1_vm.py')==plan['remote_script_sha256']
    assert t.sha(ROOT/'scripts/run_cctv_dgp_head4_archive_cleanup_v1.py')==execution['local_driver_sha256']
    export=t.read(OUT/'dgp-head4-archive-cleanup-v1-export.json');archive=OUT/'dgp-head4-archive-cleanup-v1-receipts.tar.gz'
    assert export['complete'] and not export['training_launched'] and archive.stat().st_size==export['bytes'] and t.sha(archive)==export['archive_sha256']
    with tarfile.open(archive,'r:gz') as tar:
        members=tar.getmembers();assert len(members)<=8 and sum(m.size for m in members)<256*1024**2
        names=set()
        for m in members:
            path=PurePosixPath(m.name)
            assert m.isfile() and m.size>=0 and not m.issym() and not m.islnk()
            assert len(path.parts)==2 and path.parts[0]=='receipts' and '..' not in path.parts
            assert m.name==path.as_posix() and ':' not in m.name and '\\' not in m.name and m.name not in names
            names.add(m.name)
        with tar.extractfile('receipts/manifest.json') as f:manifest=json.load(f)
        assert manifest['complete'] and manifest['plan_sha256']==pin
        assert names=={'receipts/'+n for n in manifest['files_sha256']}|{'receipts/manifest.json'}
        for n,d in manifest['files_sha256'].items():
            with tar.extractfile('receipts/'+n) as f:
                h=hashlib.sha256()
                for b in iter(lambda:f.read(1024**2),b''):h.update(b)
            assert h.hexdigest()==d,n
        destination.mkdir()
        for m in members:
            file=destination/PurePosixPath(m.name).name
            with tar.extractfile(m) as f,file.open('xb') as g:
                for b in iter(lambda:f.read(1024**2),b''):g.write(b)
    def bounded_gzip(file):
        chunks=[];total=0
        with gzip.open(file,'rb') as f:
            for b in iter(lambda:f.read(1024**2),b''):total+=len(b);assert total<256*1024**2;chunks.append(b)
        return json.loads(b''.join(chunks))
    before=bounded_gzip(destination/'protected_before.json.gz');after=bounded_gzip(destination/'protected_after.json.gz')
    assert before==after
    digest=canonical(before);verify=t.read(destination/'verify_receipt.json');receipt=t.read(destination/'cleanup_receipt.json')
    assert verify['complete'] and receipt['complete'] and verify['plan_sha256']==receipt['plan_sha256']==pin
    assert verify['protected_canonical_sha256']==receipt['protected_canonical_before']==receipt['protected_canonical_after']==digest
    assert verify['files_removed']==0 and receipt['files_removed']==15 and receipt['disjoint_nlink1_home_only_deletion']
    assert len(before['metadata'])==verify['protected_metadata_entries']==receipt['protected_metadata_entries']
    assert len(before['critical_and_evidence_hashes'])==verify['protected_byte_hashes']==receipt['protected_byte_hashes']
    ledger=[json.loads(line) for line in (destination/'deletion_ledger.jsonl').read_text().splitlines()]
    targets={row['path']:row for row in plan['candidates']};assert len(ledger)==len(targets)==15
    assert {r['path'] for r in ledger}==set(targets)==set(receipt['removed_paths'])
    for row in ledger:
        original=targets[row['path']];assert row['sha256']==original['sha256'] and row['backup_verified']
        assert original['nlink']==1 and original['uid']==1001 and original['regular_file'] and not original['symlink']
        assert row['allocated_bytes']==original['allocated_bytes'] and t.sha(ROOT/original['local_backup'])==row['sha256']
        assert Path(row['path']).parent.as_posix()=='/home/janusdominic0' and 'capacity' not in Path(row['path']).name
    assert receipt['allocated_archive_bytes']==plan['estimated_allocated_reclaim_bytes']==sum(r['allocated_bytes'] for r in ledger)
    assert receipt['free_after_bytes']-receipt['free_before_bytes']==receipt['recovered_bytes']
    assert receipt['recovered_bytes']>plan['estimated_allocated_reclaim_bytes']-64*1024**2
    assert receipt['unpacked_research_assets_and_failures_unchanged'] and not receipt['training_launched'] and not receipt['VM_started']
    packet=ROOT/'outputs/cctv_dgp_head4_capacity_vm_v1';p=t.read(packet/'protocol.json')
    for n,d in p['assets_sha256'].items():assert t.sha(packet/n)==d
    for n,d in p['local_sources'].items():assert t.sha(ROOT/n)==d
    for n,d in before['critical_and_evidence_hashes'].items():assert after['critical_and_evidence_hashes'][n]==d
    result={'complete':True,'plan_sha256':pin,'cleanup_archive_sha256':export['archive_sha256'],'checker_sha256':t.sha(Path(__file__)),
        'exact_backed_up_archives_removed':15,'local_backups_reverified':15,'shared_links_deleted':0,
        'research_metadata_entries_unchanged':len(before['metadata']),'protected_byte_hashes_unchanged':len(before['critical_and_evidence_hashes']),
        'all_cache_bytes_rehashed':False,'cache_preservation_proof':'Deletion is outside research root, exact regular nlink1 files only; full research metadata unchanged',
        'reported_free_after_GiB':receipt['free_after_bytes']/1024**3,'observed_recovered_GiB':receipt['recovered_bytes']/1024**3,
        'current_training_packet_and_90_source_bindings_preserved':True,'training_launched':False,'goal_complete':False,
        'fresh_post_apply_inventory_required':True,'seconds':time.monotonic()-start}
    t.write(OUT/'independent_audit.json',result)
    print({k:result[k] for k in ['complete','exact_backed_up_archives_removed','observed_recovered_GiB','reported_free_after_GiB']},flush=True)


if __name__=='__main__':main()
