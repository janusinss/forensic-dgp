"""Close completed removals after verifying the concurrent B guide publication.

This worker has no deletion, VM, neural, or training operation. Original stopped
workers and ledgers remain as evidence; the independent payload audit is separate.
"""
from datetime import datetime, timezone
import hashlib
import json
from pathlib import Path
import shutil
import time

ROOT=Path(__file__).resolve().parents[1]
OUT=ROOT/'outputs/local_storage_cleanup_20261011_v1'
SHARED={'PROJECT_HANDOFF.md','SYSTEM_WORKFLOW_AND_GOAL.md','PRACTICAL_OUTPUT_SCOPE.md',
        'CCTV_DGP_COMPARATIVE_IMPROVEMENT_PLAN_V2.md'}
GUIDE='CCTV_DGP_BANK_COMPARISON_B_V1_VM.md'
GUIDE_BACKUP='outputs/cctv_dgp_bank_comparison_v1_r2_B_analysis/documents_before/'+GUIDE


def read(p):
    return json.loads(Path(p).read_text(encoding='utf-8'))


def sha(p):
    with Path(p).open('rb') as stream:
        return hashlib.file_digest(stream,'sha256').hexdigest()


def save(name,data):
    with (OUT/name).open('x',encoding='utf-8',newline='\n') as stream:
        stream.write(json.dumps(data,indent=2)+'\n')


def main():
    start=time.monotonic()
    assert not (OUT/'apply_receipt.json').exists()
    plan=read(OUT/'plan.json'); binding=read(OUT/'resume_binding.json')
    stopped=read(OUT/'post_removal_protection_stop.json')
    clarification=read(OUT/'coordination_clarification.json')
    before=read(OUT/'critical_protected_before.json')['files_sha256']
    ledger=[json.loads(line) for line in (OUT/'deletion_ledger.jsonl').read_text().splitlines()]
    original=[json.loads(line) for line in (OUT/'deletion_ledger_until_stop_03.jsonl').read_text().splitlines()]
    assert plan['complete'] and binding['complete'] and clarification['complete']
    assert clarification['local_and_remote_maintenance_scopes_separate']
    assert sha(OUT/'plan.json')==binding['plan_sha256']
    assert len(ledger)==len(plan['candidates'])==stopped['actual_archives_removed']==52
    assert len(original)==15 and ledger[:15]==original
    assert not stopped['complete'] and stopped['worker_sha256']==sha(ROOT/'scripts/apply_local_archive_cleanup_20261011_v1.py')
    worker_copy=OUT/'worker_resumed_before_closure.py'
    worker_copy.write_bytes((ROOT/'scripts/apply_local_archive_cleanup_20261011_v1.py').read_bytes())
    assert sha(worker_copy)==stopped['worker_sha256']
    for actual,expected in zip(ledger,plan['candidates']):
        assert actual['path']==expected['archive'] and not (ROOT/actual['path']).exists()
        assert actual['archive_sha256']==expected['archive_sha256']
        assert actual['member_manifest_sha256']==expected['member_manifest_sha256']
        assert actual['complete_member_backup_reverified'] and actual['exclusive_archive_handle']
    assert {c['path'] for c in stopped['changes']}==SHARED|{GUIDE}
    review=OUT/'concurrent_B_guide_review'; review.mkdir()
    original_bytes=(ROOT/GUIDE_BACKUP).read_bytes()
    current_bytes=(ROOT/GUIDE).read_bytes()
    assert hashlib.sha256(original_bytes).hexdigest()==before[GUIDE]
    assert current_bytes.endswith(original_bytes) and len(current_bytes)-len(original_bytes)==521
    assert current_bytes.startswith(b'# Superseded B r2 commands - 11 October 2026')
    (review/'original.md').write_bytes(original_bytes)
    (review/'after.md').write_bytes(current_bytes)
    guide_record=dict(complete=True,path=GUIDE,original_backup_path=GUIDE_BACKUP,
        original_sha256=sha(review/'original.md'),after_sha256=sha(review/'after.md'),
        original_bytes_in_current_guide=True,prefix_bytes_added=521,
        source_publication_script='scripts/publish_cctv_dgp_bank_B2_review_v1.py',
        source_publication_script_sha256=sha(ROOT/'scripts/publish_cctv_dgp_bank_B2_review_v1.py'),
        explanation='Other chat marked completed B r2 commands historical and linked the pending manual B r3 guide',
        cleanup_did_not_edit_guide=True,original_protection_failure_preserved=True)
    save('concurrent_B_guide_change.json',guide_record)
    after={name:sha(ROOT/name) for name in before if name not in SHARED}
    assert after[GUIDE]==guide_record['after_sha256']
    for name,expected in before.items():
        if name not in SHARED|{GUIDE}:
            assert after[name]==expected, ('Unexplained protected change',name)
    snapshots=OUT/'shared_documents_after_closure'; snapshots.mkdir()
    changes=[]
    for name in sorted(SHARED):
        assert sha(OUT/'shared_documents_before'/name)==before[name]
        content=(ROOT/name).read_bytes(); (snapshots/name).write_bytes(content)
        after[name]=hashlib.sha256(content).hexdigest()
        if after[name]!=before[name]:
            changes.append(dict(path=name,before_sha256=before[name],after_sha256=after[name]))
    save('critical_protected_after.json',dict(complete=True,files_sha256=after))
    save('concurrent_document_observations.json',dict(complete=True,allowed_shared_document_names=sorted(SHARED),
        documents_changed=changes,before_after_full_bytes_preserved=True,
        after_snapshot_directory='shared_documents_after_closure',
        additional_original_guide_change_receipt='concurrent_B_guide_change.json',
        cleanup_did_not_edit_these_documents=True))
    inventory=read(OUT/'complete_inventory_before.json')
    free_before=inventory['disk_free_bytes']; free_after=shutil.disk_usage(ROOT).free
    assert time.monotonic()-start<900
    record=dict(complete=True,UTC=datetime.now(timezone.utc).isoformat(),root=str(ROOT),
        plan_sha256=sha(OUT/'plan.json'),worker_sha256=stopped['worker_sha256'],
        files_removed=52,removed_paths=[r['path'] for r in ledger],
        archive_logical_bytes_removed=sum(r['former_bytes'] for r in ledger),
        resumed_from_verified_partial_ledger=True,first_execution_complete=False,
        first_worker_sha256=binding['first_deletion_worker_sha256'],
        first_execution_deletion_ledger_sha256=binding['first_deletion_ledger_sha256'],
        first_execution_files_removed=15,resume_files_removed=37,resume_execution_complete=False,
        resume_binding_sha256=sha(OUT/'resume_binding.json'),
        coordination_clarification_sha256=sha(OUT/'coordination_clarification.json'),
        post_removal_stop_receipt_sha256=sha(OUT/'post_removal_protection_stop.json'),
        closure_worker_sha256=sha(__file__),closure_removed_additional_files=0,
        concurrent_B_guide_change_receipt_sha256=sha(OUT/'concurrent_B_guide_change.json'),
        free_before_bytes=free_before,free_after_bytes=free_after,
        observed_net_free_increase_bytes=free_after-free_before,
        free_space_measurement_basis='complete_inventory_snapshot_to_cleanup_end_includes_concurrent_writes',
        immediate_resume_free_before_measurement_unavailable=True,
        critical_protected_files=len(before),immutable_critical_hashes_unchanged=True,
        documented_guide_change_with_exact_original_preserved=True,
        concurrent_shared_document_changes=changes,
        extracted_payloads_checkpoints_data_caches_splits_logs_and_failures_retained=True,
        current_bank_transfers_and_migration_backups_preserved=True,no_directories_removed=True,
        VM_connections_or_training_calls=0,independent_closure_pending=True,
        maintenance_cap_seconds=900,seconds=time.monotonic()-start)
    save('apply_receipt.json',record)
    print(dict(complete=True,archive_removals_recorded=52,additional_files_removed=0,
        exact_original_B_guide_preserved=True,independent_closure_pending=True),flush=True)


if __name__=='__main__':
    main()
