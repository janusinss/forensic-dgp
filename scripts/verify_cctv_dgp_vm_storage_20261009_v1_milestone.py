"""Independent local evidence/backup/handoff closure after completed VM maintenance."""
import hashlib
import json
from pathlib import Path, PurePosixPath
import time

ROOT = Path(__file__).resolve().parents[1]
OUT = ROOT / 'outputs/cctv_dgp_vm_storage_cleanup_20261009_v1'


def main():
    start = time.monotonic()

    def sha(path):
        with Path(path).open('rb') as stream:
            value = hashlib.file_digest(stream, 'sha256').hexdigest()
        assert time.monotonic() - start < 900, 'Finite local closure cap exceeded'
        return value

    def read(path):
        return json.loads(Path(path).read_text(encoding='utf-8'))

    target = OUT / 'independent_closure_audit.json'
    assert not target.exists()
    milestone = read(OUT / 'milestone.json')
    for name, digest in milestone['new_evidence_sha256'].items():
        assert sha(ROOT / name) == digest, name
    for name, digest in milestone['app_bindings_sha256'].items():
        assert sha(ROOT / name) == digest, name
    for index, (name, digest) in enumerate(milestone['local_backup_bindings'].items(), 1):
        assert sha(name) == digest, name
        print({'retained_Windows_backups_reverified': index, 'of': len(milestone['local_backup_bindings'])}, flush=True)
    prior = ROOT / 'outputs/cctv_dgp_actual_step_review_v1_audited_milestone'
    assert sha(prior / 'milestone.json') == milestone['previous_milestone_sha256']
    assert sha(prior / 'independent_closure_audit_r1.json') == milestone['previous_closure_sha256']
    before = (OUT / 'PROJECT_HANDOFF_before_cleanup.md').read_bytes()
    now = (ROOT / 'PROJECT_HANDOFF.md').read_bytes()
    assert hashlib.sha256(before).hexdigest() == milestone['previous_handoff_sha256']
    assert now[milestone['new_intro_bytes']:] == before, 'Preserve the entire preceding handoff verbatim'
    live = read(OUT / 'independent_live_audit.json')
    cleanup = read(OUT / 'remote_receipts/cleanup_receipt.json')
    verify = read(OUT / 'remote_receipts/verification.json')
    plan = read(OUT / 'archive-duplicates_plan_20261009_v1.json')
    ledger = [json.loads(row) for row in (OUT / 'remote_receipts/deletions.jsonl').read_text(encoding='utf-8').splitlines()]
    count = len(plan['files'])
    assert {row['path'] for row in ledger} == {row['path'] for row in plan['files']}
    assert len(ledger) == count == cleanup['files_removed'] == live['deleted_files_verified_absent'] == milestone['files_removed']
    assert all(row['kind'] == 'duplicate_archive' and PurePosixPath(row['path']).parent == PurePosixPath('/home/janusdominic0') and row['nlink'] == 1 and row['uid'] == 1001 for row in verify['archives'])
    assert not plan['clear_pip_download_cache'] and not verify['pip_cache_files']
    assert cleanup['free_bytes_after'] - cleanup['free_bytes_before'] == cleanup['free_bytes_increase'] == milestone['observed_free_bytes_increase']
    assert live['free_bytes_live'] == milestone['independent_live_free_bytes'] and live['free_bytes_live'] >= 2 * 1024**3
    assert not milestone['training_or_diagnostic_launched'] and not milestone['goal_complete']
    assert milestone['manual_tail_execution_required'] and not milestone['app_promotion']
    assert read(OUT / 'source_adaptation.json')['owned_regular_non_symlink_single_link_checks_AST_unchanged']
    value = {'complete': True, 'scope': 'Independent local cleanup receipts, retained archive hashes, app bindings and exact handoff suffix',
             'milestone_sha256': sha(OUT / 'milestone.json'),
             'new_evidence_bindings_verified': len(milestone['new_evidence_sha256']),
             'Windows_archive_backups_fully_rehashed_after_cleanup': len(milestone['local_backup_bindings']),
             'Windows_archive_backup_bytes': sum(row['bytes'] for row in plan['files']),
             'app_bindings_verified': len(milestone['app_bindings_sha256']),
             'previous_handoff_exact_suffix_verified': True, 'archive_only_scope_verified': True,
             'live_VM_free_bytes_receipt_verified': live['free_bytes_live'],
             'no_training_or_diagnostic_launch': True, 'goal_complete': False,
             'seconds': time.monotonic() - start, 'cap_seconds': 900}
    with target.open('x', encoding='utf-8', newline='\n') as stream:
        stream.write(json.dumps(value, indent=2, allow_nan=False) + '\n')
    print(value, flush=True)


if __name__ == '__main__':
    main()
