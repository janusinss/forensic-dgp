"""Finish namespace checks after the passed byte audit's Git timestamp stop."""
from datetime import datetime, timezone
import gzip
import hashlib
import json
from pathlib import Path
import re
import shutil
import stat
import time
import zlib

ROOT=Path(__file__).resolve().parents[1]
OUT=ROOT/'outputs/local_storage_cleanup_20261011_v1'
SHARED={'PROJECT_HANDOFF.md','SYSTEM_WORKFLOW_AND_GOAL.md','PRACTICAL_OUTPUT_SCOPE.md',
        'CCTV_DGP_COMPARATIVE_IMPROVEMENT_PLAN_V2.md'}
GUIDE='CCTV_DGP_BANK_COMPARISON_B_V1_VM.md'

def read(p):
    return json.loads(Path(p).read_text(encoding='utf-8'))

def sha(p):
    with Path(p).open('rb') as f:
        return hashlib.file_digest(f,'sha256').hexdigest()

def save(name,data):
    with (OUT/name).open('x',encoding='utf-8',newline='\n') as f:
        f.write(json.dumps(data,indent=2)+'\n')

def main():
    start=time.monotonic()
    assert not (OUT/'independent_audit.json').exists()
    plan=read(OUT/'plan.json'); applied=read(OUT/'apply_receipt.json')
    assert plan['complete'] and applied['complete'] and applied['plan_sha256']==sha(OUT/'plan.json')
    source=ROOT/'scripts/audit_local_archive_cleanup_20261011_v1.py'
    log=OUT/'independent_audit_execution.log'; raw=log.read_bytes()
    transcript=raw.decode('utf-16' if raw[:2] in (b'\xff\xfe',b'\xfe\xff') else 'utf-8-sig')
    count=sum(r['complete_regular_members'] for r in plan['candidates'])
    assert count==276655 and len(plan['candidates'])==52
    assert "'archive_payloads_rechecked': 52, 'of': 52, 'retained_members': 276655" in transcript
    copied=OUT/'independent_payload_checker_before_namespace_tail.py'
    copied.write_bytes(source.read_bytes()); assert sha(copied)==sha(source)
    proof=dict(complete=True,archive_payload_phase_passed=True,archives=52,members=count,
        full_payload_bytes=sum(r['retained_payload_bytes'] for r in plan['candidates']),
        checker_source_sha256=sha(source),checker_execution_log_sha256=sha(log),
        original_checker_full_execution_complete=False,original_failed_check_preserved=True,
        original_stop='Namespace metadata check: .git/objects/05/6e14e1686ea86847876fd275a356bb6769eea4',
        all_regular_payload_hash_checks_precede_the_recorded_52_archive_pass=True)
    save('independent_payload_phase_pass.json',proof)
    ledger=[json.loads(line) for line in (OUT/'deletion_ledger.jsonl').read_text().splitlines()]
    expected={r['archive']:r for r in plan['candidates']}
    assert len(ledger)==len(expected)==applied['files_removed']==52
    assert len({r['path'] for r in ledger})==52 and {r['path'] for r in ledger}==set(expected)==set(applied['removed_paths'])
    assert applied['worker_sha256']==sha(ROOT/'scripts/apply_local_archive_cleanup_20261011_v1.py')==sha(OUT/'worker_resumed_before_closure.py')
    assert applied['closure_worker_sha256']==sha(ROOT/'scripts/close_local_archive_cleanup_20261011_v1.py')
    assert applied['first_worker_sha256']==sha(OUT/'worker_before_resume.py')
    original=[json.loads(line) for line in (OUT/'deletion_ledger_until_stop_03.jsonl').read_text().splitlines()]
    assert len(original)==15 and ledger[:15]==original and applied['resume_files_removed']==37
    assert applied['first_execution_deletion_ledger_sha256']==sha(OUT/'deletion_ledger_until_stop_03.jsonl')
    assert not applied['first_execution_complete'] and not applied['resume_execution_complete']
    parents={}; unique=set(); members_checked=0
    def safe_file(name):
        relative=Path(name); assert not relative.is_absolute() and '..' not in relative.parts
        p=ROOT/relative; assert p.is_relative_to(ROOT) and p!=ROOT
        for d in p.parents:
            if d==ROOT.parent: break
            if d not in parents:
                s=d.lstat(); assert stat.S_ISDIR(s.st_mode) and not getattr(s,'st_file_attributes',0)&0x400
                parents[d]=(s.st_dev,s.st_ino)
        s=p.lstat(); assert stat.S_ISREG(s.st_mode) and not getattr(s,'st_file_attributes',0)&0x400
        return p,s
    for row in ledger:
        wanted=expected[row['path']]; assert not (ROOT/row['path']).exists()
        assert row['archive_sha256']==wanted['archive_sha256'] and row['former_bytes']==wanted['metadata']['bytes']
        assert row['member_manifest_sha256']==wanted['member_manifest_sha256']==sha(ROOT/row['member_manifest'])
        assert row['complete_member_backup_reverified'] and row['exclusive_archive_handle']
        members=json.load(gzip.open(ROOT/row['member_manifest'],'rt',encoding='utf-8'))
        assert len(members)==wanted['complete_regular_members']==row['retained_members']
        for member in members:
            p,s=safe_file(member['retained_path'])
            assert dict(bytes=s.st_size,dev=s.st_dev,inode=s.st_ino,nlink=s.st_nlink,mtime_ns=s.st_mtime_ns)==member['retained_metadata']
            unique.add((str(p),s.st_ino,s.st_mtime_ns,s.st_size)); members_checked+=1
        assert time.monotonic()-start<900
    assert members_checked==proof['members']
    before=read(OUT/'critical_protected_before.json')['files_sha256']
    after=read(OUT/'critical_protected_after.json')['files_sha256']; guide=read(OUT/'concurrent_B_guide_change.json')
    assert set(before)==set(after) and guide['path']==GUIDE
    old=(OUT/'concurrent_B_guide_review/original.md').read_bytes(); new=(OUT/'concurrent_B_guide_review/after.md').read_bytes()
    assert new.endswith(old) and len(new)-len(old)==521
    assert hashlib.sha256(old).hexdigest()==before[GUIDE]==guide['original_sha256']
    assert hashlib.sha256(new).hexdigest()==after[GUIDE]==guide['after_sha256']==sha(ROOT/GUIDE)
    for name,expected_hash in after.items():
        if name in SHARED:
            assert sha(OUT/'shared_documents_before'/name)==before[name]
            assert sha(OUT/'shared_documents_after_closure'/name)==expected_hash
        elif name!=GUIDE:
            assert before[name]==expected_hash==sha(ROOT/name), ('Protected content changed',name)
    inventory=json.load(gzip.open(OUT/'complete_file_inventory_before.json.gz','rt',encoding='utf-8'))
    existing=0; changes=[]; git_changes=[]; lost=[]; unexplained=[]
    for row in inventory:
        name=row['path']
        if name.startswith('outputs/local_storage_cleanup_20261011_v1/'): continue
        p=ROOT/name
        if name in expected:
            assert not p.exists(); continue
        if not p.is_file() or p.is_symlink():
            lost.append(name); continue
        s=p.stat(); actual=tuple(getattr(s,k) for k in ['st_size','st_dev','st_ino','st_nlink','st_mtime_ns'])
        original=tuple(row[k] for k in ['bytes','dev','inode','nlink','mtime_ns'])
        if actual!=original:
            item=dict(path=name,before=original,after=actual)
            if name in SHARED|{GUIDE}: changes.append(item)
            elif re.fullmatch(r'\.git/objects/[0-9a-f]{2}/[0-9a-f]{38}',name) and actual[:4]==original[:4]:
                gp,_=safe_file(name); identifier=hashlib.sha1(zlib.decompress(gp.read_bytes())).hexdigest()
                assert identifier==''.join(name.split('/')[-2:]), ('Git content identity changed',name)
                git_changes.append({**item,'git_content_sha1':identifier,'only_mtime_changed':True})
            else: unexplained.append(item)
        existing+=1
        if existing%100000==0: print(dict(retained_namespace_files_rechecked=existing),flush=True)
        assert time.monotonic()-start<900
    save('namespace_tail_observations.json',dict(complete=not lost and not unexplained,
        retained_files=existing,unexpected_lost_files=lost,unexplained_metadata_changes=unexplained,
        shared_document_changes=changes,verified_git_timestamp_changes=git_changes))
    assert not lost and not unexplained, ('Namespace differences require review',lost,unexplained)
    for directory,expected_identity in parents.items():
        s=directory.lstat(); assert (s.st_dev,s.st_ino)==expected_identity and not getattr(s,'st_file_attributes',0)&0x400
    latest=OUT/'shared_documents_at_audit'; latest.mkdir(); latest_hashes={}
    for name in sorted(SHARED):
        content=(ROOT/name).read_bytes(); (latest/name).write_bytes(content); latest_hashes[name]=hashlib.sha256(content).hexdigest()
    result=dict(complete=True,UTC=datetime.now(timezone.utc).isoformat(),checker_sha256=sha(__file__),
        plan_sha256=sha(OUT/'plan.json'),apply_receipt_sha256=sha(OUT/'apply_receipt.json'),
        deletion_ledger_sha256=sha(OUT/'deletion_ledger.jsonl'),exact_archive_deletions_checked=52,
        full_retained_archive_payload_members_checked=count,retained_archive_payload_bytes_checked=proof['full_payload_bytes'],
        unique_retained_payload_files_hashed=len(unique),retained_payload_metadata_freshly_rechecked=members_checked,
        retained_namespace_files_checked=existing,unexpected_lost_files=0,unrelated_metadata_changes=0,
        concurrent_shared_document_metadata_changes=changes,verified_git_timestamp_changes=git_changes,
        concurrent_document_before_after_bytes_preserved=True,latest_shared_document_snapshot_sha256=latest_hashes,
        original_B_guide_backup_and_added_superseded_notice_verified=True,critical_protected_file_hashes_checked=len(before),
        model_data_cache_split_instruction_provenance_failure_records_retained=True,current_bank_packets_and_migration_backups_preserved=True,
        original_archive_checksums_and_member_metadata_retained=True,original_deleted_compressed_containers_not_retained=True,
        first_stopped_execution_ledger_preserved=True,resumed_execution_provenance_verified=True,
        post_removal_protection_failure_retained=True,zero_deletion_closure_verified=True,
        independent_payload_phase_receipt_sha256=sha(OUT/'independent_payload_phase_pass.json'),
        original_independent_worker_namespace_stop_preserved=True,
        independent_byte_phase_and_fresh_namespace_tail_are_separate=True,
        bytes_removed=sum(r['former_bytes'] for r in ledger),free_bytes_now=shutil.disk_usage(ROOT).free,
        observed_net_free_increase_bytes=applied['observed_net_free_increase_bytes'],free_space_measurement_basis=applied['free_space_measurement_basis'],
        VM_connections_or_neural_training_calls=0,goal_status='active',maintenance_cap_seconds=900,seconds=time.monotonic()-start)
    save('independent_audit.json',result)
    print(dict(complete=True,archives_checked=52,retained_members_checked=count,retained_namespace_files_checked=existing,
        verified_git_timestamp_changes=len(git_changes),unexpected_lost_files=0,free_GiB=result['free_bytes_now']/1024**3),flush=True)

if __name__=='__main__': main()
