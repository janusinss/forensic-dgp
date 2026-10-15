"""Independent closure: exact deletions, full retained payloads and namespace."""
from datetime import datetime, timezone
import gzip
import hashlib
import json
from pathlib import Path
import shutil
import stat
import time

ROOT=Path(__file__).resolve().parents[1]
OUT=ROOT/'outputs/local_storage_cleanup_20261011_v1'


def read(p):
    return json.loads(Path(p).read_text(encoding='utf-8'))


def sha(p):
    with Path(p).open('rb') as f:
        return hashlib.file_digest(f,'sha256').hexdigest()


def safe(value):
    p=ROOT/value
    assert p.resolve().is_relative_to(ROOT.resolve()) and p.resolve()!=ROOT.resolve()
    assert not p.is_symlink()
    for parent in [p,*p.parents]:
        if parent==ROOT.parent:
            break
        assert not getattr(parent.lstat(),'st_file_attributes',0)&0x400
    return p


def main():
    start=time.monotonic()
    receipt=OUT/'independent_audit.json'
    assert not receipt.exists()
    plan=read(OUT/'plan.json')
    applied=read(OUT/'apply_receipt.json')
    assert plan['complete'] and applied['complete'] and applied['plan_sha256']==sha(OUT/'plan.json')
    assert applied['worker_sha256']==sha(ROOT/'scripts/apply_local_archive_cleanup_20261011_v1.py')
    rows=[json.loads(line) for line in (OUT/'deletion_ledger.jsonl').read_text().splitlines()]
    binding=read(OUT/'resume_binding.json')
    stopped=read(OUT/'apply_guard_stop_03.json')
    clarification=read(OUT/'coordination_clarification.json')
    assert applied['resumed_from_verified_partial_ledger'] and not applied['first_execution_complete']
    assert binding['complete'] and clarification['complete'] and not stopped['complete']
    assert applied['resume_binding_sha256']==sha(OUT/'resume_binding.json')
    assert applied['coordination_clarification_sha256']==sha(OUT/'coordination_clarification.json')
    assert applied['first_worker_sha256']==binding['first_deletion_worker_sha256']==stopped['worker_sha256']==sha(OUT/'worker_before_resume.py')
    assert applied['first_execution_deletion_ledger_sha256']==binding['first_deletion_ledger_sha256']==sha(OUT/'deletion_ledger_until_stop_03.jsonl')
    original_rows=[json.loads(line) for line in (OUT/'deletion_ledger_until_stop_03.jsonl').read_text().splitlines()]
    assert len(original_rows)==applied['first_execution_files_removed']==15
    assert rows[:len(original_rows)]==original_rows and applied['resume_files_removed']==37
    inventory_receipt=read(OUT/'complete_inventory_before.json')
    assert applied['free_before_bytes']==inventory_receipt['disk_free_bytes']
    assert applied['free_space_measurement_basis']=='complete_inventory_snapshot_to_cleanup_end_includes_concurrent_writes'
    assert applied['observed_net_free_increase_bytes']==applied['free_after_bytes']-applied['free_before_bytes']
    assert not applied['resume_execution_complete'] and applied['closure_removed_additional_files']==0
    assert applied['closure_worker_sha256']==sha(ROOT/'scripts/close_local_archive_cleanup_20261011_v1.py')
    assert applied['post_removal_stop_receipt_sha256']==sha(OUT/'post_removal_protection_stop.json')
    assert applied['worker_sha256']==sha(OUT/'worker_resumed_before_closure.py')
    guide=read(OUT/'concurrent_B_guide_change.json')
    assert applied['concurrent_B_guide_change_receipt_sha256']==sha(OUT/'concurrent_B_guide_change.json')
    assert guide['complete'] and guide['path']=='CCTV_DGP_BANK_COMPARISON_B_V1_VM.md'
    original_guide=(OUT/'concurrent_B_guide_review/original.md').read_bytes()
    updated_guide=(OUT/'concurrent_B_guide_review/after.md').read_bytes()
    assert sha(OUT/'concurrent_B_guide_review/original.md')==guide['original_sha256']==sha(safe(guide['original_backup_path']))
    assert updated_guide.endswith(original_guide) and len(updated_guide)-len(original_guide)==guide['prefix_bytes_added']==521
    assert sha(OUT/'concurrent_B_guide_review/after.md')==guide['after_sha256']==sha(safe(guide['path']))
    expected={r['archive']:r for r in plan['candidates']}
    assert len(rows)==applied['files_removed']==len(expected)==plan['exact_archive_count']
    assert len({r['path'] for r in rows})==len(rows)
    assert {r['path'] for r in rows}==set(expected)==set(applied['removed_paths'])
    digests={}
    member_count=0
    payload_bytes=0
    for number,row in enumerate(rows,1):
        source=expected[row['path']]
        assert source['verified'] and row['archive_sha256']==source['archive_sha256']
        assert row['former_bytes']==source['metadata']['bytes'] and row['former_inode']==source['metadata']['inode']
        assert source['metadata']['nlink']==1 and row['exclusive_archive_handle']
        assert row['complete_member_backup_reverified'] and row['retained_sources_held_read_only_during_removal']
        archive=ROOT/row['path']
        assert archive.parent==ROOT/'outputs' and not archive.exists() and 'bank-comparison' not in archive.name
        assert row['member_manifest_sha256']==source['member_manifest_sha256']==sha(safe(row['member_manifest']))
        assert row['retained_tree']==source['retained_tree']
        members=json.load(gzip.open(safe(row['member_manifest']),'rt',encoding='utf-8'))
        assert len(members)==row['retained_members']==source['complete_regular_members']
        for member in members:
            p=safe(member['retained_path'])
            assert p.is_relative_to(safe(row['retained_tree']))
            s=p.lstat()
            assert stat.S_ISREG(s.st_mode)
            details=dict(bytes=s.st_size,dev=s.st_dev,inode=s.st_ino,nlink=s.st_nlink,mtime_ns=s.st_mtime_ns)
            assert details==member['retained_metadata']
            key=(str(p),s.st_ino,s.st_mtime_ns,s.st_size)
            if key not in digests:
                digests[key]=sha(p)
            assert digests[key]==member['sha256']
            member_count+=1;payload_bytes+=member['bytes']
        assert time.monotonic()-start<3600, 'Finite independent audit limit'
        if number%5==0 or number==len(rows):
            print(dict(archive_payloads_rechecked=number,of=len(rows),retained_members=member_count),flush=True)
    before=read(OUT/'critical_protected_before.json')
    after=read(OUT/'critical_protected_after.json')
    documents=read(OUT/'concurrent_document_observations.json')
    shared=set(documents['allowed_shared_document_names'])
    assert shared=={'PROJECT_HANDOFF.md','SYSTEM_WORKFLOW_AND_GOAL.md','PRACTICAL_OUTPUT_SCOPE.md','CCTV_DGP_COMPARATIVE_IMPROVEMENT_PLAN_V2.md'}
    assert before['complete'] and after['complete'] and set(before['files_sha256'])==set(after['files_sha256'])
    for name,expected_hash in after['files_sha256'].items():
        if name in shared:
            assert sha(OUT/'shared_documents_before'/name)==before['files_sha256'][name]
            assert documents['after_snapshot_directory']=='shared_documents_after_closure'
            assert sha(OUT/documents['after_snapshot_directory']/name)==expected_hash
        elif name==guide['path']:
            assert before['files_sha256'][name]==guide['original_sha256']
            assert expected_hash==guide['after_sha256']==sha(safe(name))
        else:
            assert sha(safe(name))==expected_hash
            assert before['files_sha256'][name]==expected_hash
    with gzip.open(OUT/'complete_file_inventory_before.json.gz','rt',encoding='utf-8') as f:
        inventory=json.load(f)
    removed_paths=set(expected)
    existing=0
    metadata_changes=[]
    for row in inventory:
        name=row['path']
        if name.startswith('outputs/local_storage_cleanup_20261011_v1/'):
            continue
        if name in removed_paths:
            assert not (ROOT/name).exists()
            continue
        p=ROOT/name
        assert p.is_file() and not p.is_symlink(), ('Unexpected lost local file',name)
        s=p.stat()
        actual=(s.st_size,s.st_dev,s.st_ino,s.st_nlink,s.st_mtime_ns)
        original=tuple(row[k] for k in ['bytes','dev','inode','nlink','mtime_ns'])
        if actual!=original:
            metadata_changes.append(dict(path=name,before=original,after=actual))
            assert name in shared|{guide['path']}, ('Unexpected unrelated metadata change',name)
            if name==guide['path']:
                assert sha(p)==guide['after_sha256']
        existing+=1
        if existing%100000==0:
            print(dict(retained_namespace_files_rechecked=existing),flush=True)
        assert time.monotonic()-start<3600
    for row in plan['retained']:
        assert (ROOT/row['archive']).is_file()
    latest_documents=OUT/'shared_documents_at_audit'
    latest_documents.mkdir()
    latest_doc_hashes={}
    for name in sorted(shared):
        content=safe(name).read_bytes()
        (latest_documents/name).write_bytes(content)
        latest_doc_hashes[name]=hashlib.sha256(content).hexdigest()
    result=dict(complete=True,UTC=datetime.now(timezone.utc).isoformat(),checker_sha256=sha(__file__),
        plan_sha256=sha(OUT/'plan.json'),apply_receipt_sha256=sha(OUT/'apply_receipt.json'),
        deletion_ledger_sha256=sha(OUT/'deletion_ledger.jsonl'),exact_archive_deletions_checked=len(rows),
        full_retained_archive_payload_members_checked=member_count,
        retained_archive_payload_bytes_checked=payload_bytes,unique_retained_payload_files_hashed=len(digests),
        retained_namespace_files_checked=existing,unexpected_lost_files=0,unrelated_metadata_changes=0,
        concurrent_shared_document_metadata_changes=metadata_changes,
        concurrent_document_before_after_bytes_preserved=True,
        original_B_guide_backup_and_added_superseded_notice_verified=True,
        latest_shared_document_snapshot_sha256=latest_doc_hashes,
        critical_protected_file_hashes_checked=len(before['files_sha256']),
        model_data_cache_split_instruction_provenance_failure_records_retained=True,
        current_bank_packets_and_migration_backups_preserved=True,
        original_archive_checksums_and_member_metadata_retained=True,
        original_deleted_compressed_containers_not_retained=True,
        bytes_removed=sum(r['former_bytes'] for r in rows),free_bytes_now=shutil.disk_usage(ROOT).free,
        first_stopped_execution_ledger_preserved=True,resumed_execution_provenance_verified=True,
        post_removal_protection_failure_retained=True,zero_deletion_closure_verified=True,
        first_deletion_worker_sha256=applied['first_worker_sha256'],resume_worker_sha256=applied['worker_sha256'],
        observed_net_free_increase_bytes=applied['observed_net_free_increase_bytes'],
        free_space_measurement_basis=applied['free_space_measurement_basis'],
        VM_connections_or_neural_training_calls=0,goal_status='active',seconds=time.monotonic()-start)
    with receipt.open('x',encoding='utf-8',newline='\n') as f:
        f.write(json.dumps(result,indent=2)+'\n')
    print(dict(complete=True,exact_archive_deletions_checked=len(rows),
        free_GiB=result['free_bytes_now']/1024**3,unexpected_lost_files=0),flush=True)


if __name__=='__main__':
    main()
