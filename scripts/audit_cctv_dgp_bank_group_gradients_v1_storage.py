"""Independently close the exact one-archive cleanup using retained receipts."""
import ast
from datetime import datetime, timezone
import gzip
import hashlib
import json
from pathlib import Path, PurePosixPath
import tarfile
import time

ROOT=Path(__file__).resolve().parents[1]
OUT=ROOT/'outputs/cctv_dgp_bank_group_gradients_v1_storage_cleanup'


def sha(path):
    with Path(path).open('rb') as stream:
        return hashlib.file_digest(stream,'sha256').hexdigest()


def read(path):
    return json.loads(Path(path).read_text(encoding='utf-8'))


def canonical(value):
    return hashlib.sha256(json.dumps(value,sort_keys=True,separators=(',',':')).encode()).hexdigest()


def main():
    start=time.monotonic()
    final=OUT/'independent_audit.json'
    assert not final.exists()
    plan=read(OUT/'plan.json');pin=sha(OUT/'plan.json')
    assert plan['complete'] and len(plan['candidates'])==1 and plan['only_one_home_archive_unlink'] and plan['no_research_tree_deletion']
    row=plan['candidates'][0]
    assert row['path']=='/home/janusdominic0/cctv-dgp-bank-photometric-guard-v1-results.tar.gz'
    backup=ROOT/row['local_backup']
    assert backup.is_file() and not backup.is_symlink() and backup.stat().st_size==row['bytes']
    assert backup.stat().st_mtime_ns==row['local_backup_mtime_ns'] and sha(backup)==row['sha256']
    pre=read(OUT/'local_preapply_backup_recheck.json')
    assert pre['complete'] and pre['plan_sha256']==pin and pre['full_archive_sha256']==row['sha256']
    worker=ROOT/'scripts/cctv_dgp_bank_group_gradients_v1_storage_vm.py'
    assert sha(worker)==plan['worker_sha256']
    tree=ast.parse(worker.read_text(encoding='utf-8'))
    unlinks=[n for n in ast.walk(tree) if isinstance(n,ast.Call) and isinstance(n.func,ast.Attribute) and n.func.attr=='unlink']
    assert len(unlinks)==1 and isinstance(unlinks[0].func.value,ast.Name) and unlinks[0].func.value.id=='TARGET'
    assert not any(isinstance(n,ast.Call) and isinstance(n.func,ast.Attribute) and n.func.attr in ['rmtree','remove','backward','step','grad','kill','terminate'] for n in ast.walk(tree))
    archive=OUT/'dgp-bank-group-gradients-v1-storage-receipts.tar.gz'
    exported=read(OUT/'dgp-bank-group-gradients-v1-storage-export.json')
    assert exported['complete'] and sha(archive)==exported['archive_sha256'] and archive.stat().st_size==exported['bytes']
    dest=OUT/'remote_receipts';assert not dest.exists();dest.mkdir()
    allowed={'protected_before.json.gz','protected_after.json.gz','verify_receipt.json','cleanup_receipt.json','deletion_ledger.jsonl','manifest.json'}
    seen=set()
    with tarfile.open(archive,'r:gz') as tar:
        members=tar.getmembers()
        assert len(members)==len(allowed) and sum(m.size for m in members)<128*1024**2
        for member in members:
            path=PurePosixPath(member.name)
            assert member.isfile() and not member.issym() and not member.islnk()
            assert len(path.parts)==2 and path.parts[0]=='receipts' and path.name in allowed and path.name not in seen
            with tar.extractfile(member) as source,(dest/path.name).open('xb') as sink:
                for block in iter(lambda:source.read(1024**2),b''):sink.write(block)
            seen.add(path.name)
    assert seen==allowed
    manifest=read(dest/'manifest.json')
    assert manifest['complete'] and manifest['plan_sha256']==pin and set(manifest['files_sha256'])==allowed-{'manifest.json'}
    assert all(sha(dest/n)==h for n,h in manifest['files_sha256'].items())
    verified=read(dest/'verify_receipt.json');cleanup=read(dest/'cleanup_receipt.json')
    assert verified['complete'] and cleanup['complete'] and verified['plan_sha256']==cleanup['plan_sha256']==pin
    assert verified['worker_sha256']==cleanup['worker_sha256']==sha(worker)
    assert cleanup['files_removed']==verified['candidate_count']==1 and cleanup['removed_paths']==[row['path']]
    ledger=[json.loads(line) for line in (dest/'deletion_ledger.jsonl').read_text().splitlines()]
    assert len(ledger)==1 and ledger[0]['backup_verified'] and ledger[0]['nlink']==1 and ledger[0]['uid']==1001
    assert all(ledger[0][k]==row[k] for k in ['path','sha256','bytes','inode','uid','nlink','allocated_bytes','local_backup'])
    with gzip.open(dest/'protected_before.json.gz','rb') as stream:before=json.load(stream)
    with gzip.open(dest/'protected_after.json.gz','rb') as stream:after=json.load(stream)
    assert before==after
    digest=canonical(before)
    assert digest==verified['protected_canonical_sha256']==cleanup['protected_canonical_before']==cleanup['protected_canonical_after']
    assert sha(dest/'protected_before.json.gz')==verified['protected_sha256']
    assert len(before['metadata'])==cleanup['protected_metadata_entries'] and len(before['packet_bytes_sha256'])==cleanup['packet_byte_hashes']
    assert all(before['packet_bytes_sha256'][n]==h for n,h in plan['critical_sha256'].items())
    live=read(OUT/'live_audit.json')
    assert live['complete'] and live['plan_sha256']==pin and live['worker_sha256']==sha(worker)
    assert live['deleted_exact_archive_absent'] and live['files_removed_by_audit']==0 and live['protected_canonical_sha256']==digest
    assert live['packet_hashes']==len(before['packet_bytes_sha256']) and live['free_bytes']>=4*1024**3
    assert live['fresh_runtime']['GPU_compute_idle'] and cleanup['model_gradient_or_training_calls']==0 and not cleanup['VM_started']
    assert cleanup['allocated_reclaim_bytes']==row['allocated_bytes']==plan['allocated_reclaim_bytes']
    assert cleanup['archive_logical_bytes']==row['bytes']
    assert cleanup['measured_free_change_bytes']==cleanup['free_after_bytes']-cleanup['free_before_bytes']
    protected=read(ROOT/'outputs/cctv_dgp_bank_group_gradients_v1_analysis/local_protected_before.json')['files_sha256']
    assert all(sha(ROOT/n)==h for n,h in protected.items())
    value=dict(complete=True,UTC=datetime.now(timezone.utc).isoformat(),checker_sha256=sha(__file__),plan_sha256=pin,
        exact_deleted_archives=1,full_local_backup_sha256=row['sha256'],full_local_backup_bytes=row['bytes'],
        receipt_archive_sha256=sha(archive),live_receipt_sha256=sha(OUT/'live_audit.json'),
        retained_research_metadata_entries=len(before['metadata']),retained_packet_byte_hashes=len(before['packet_bytes_sha256']),
        critical_byte_bindings_checked=len(plan['critical_sha256']),local_protected_bindings_checked=len(protected),
        protected_metadata_and_packet_bytes_identical=True,allocated_reclaim_bytes=row['allocated_bytes'],
        measured_free_change_bytes=cleanup['measured_free_change_bytes'],free_GiB_live=live['free_bytes']/1024**3,
        finalized_export_tails_bound_to_live_snapshots=True,all_scientific_cache_bytes_rehashed=False,
        checkpoints_research_caches_splits_logs_and_gate_failures_retained=True,
        model_gradient_or_training_calls=0,new_manual_measurement_unrun=True,goal_complete=False,seconds=time.monotonic()-start)
    with final.open('x',encoding='utf-8',newline='\n') as stream:stream.write(json.dumps(value,indent=2)+'\n')
    print({k:value[k] for k in ['complete','exact_deleted_archives','retained_research_metadata_entries','retained_packet_byte_hashes','free_GiB_live']},flush=True)


if __name__=='__main__':
    main()
