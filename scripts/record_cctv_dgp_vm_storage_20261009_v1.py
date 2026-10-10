"""Record completed authorized cleanup and preserve the full preceding handoff."""
from datetime import datetime, timezone
import hashlib
import json
from pathlib import Path
import time

ROOT = Path(__file__).resolve().parents[1]
OUT = ROOT / 'outputs/cctv_dgp_vm_storage_cleanup_20261009_v1'
PRIOR = ROOT / 'outputs/cctv_dgp_actual_step_review_v1_audited_milestone'


def sha(path):
    with Path(path).open('rb') as stream:
        return hashlib.file_digest(stream, 'sha256').hexdigest()


def read(path):
    return json.loads(Path(path).read_text(encoding='utf-8'))


def write(path, value):
    with Path(path).open('x', encoding='utf-8', newline='\n') as stream:
        stream.write(json.dumps(value, indent=2, allow_nan=False) + '\n')


def main():
    started = time.monotonic()
    assert not (OUT / 'milestone.json').exists()
    plan_path = OUT / 'archive-duplicates_plan_20261009_v1.json'
    plan = read(plan_path)
    live = read(OUT / 'independent_live_audit.json')
    final_disk = read(OUT / 'final_disk_runtime_inventory.json')
    assert final_disk['complete'] and final_disk['GPU_idle'] and final_disk['manual_tail_free_requirement_met']
    assert not final_disk['training_or_diagnostic_launched'] and not final_disk['tail_root_present']
    verify = read(OUT / 'remote_receipts/verification.json')
    cleanup = read(OUT / 'remote_receipts/cleanup_receipt.json')
    ledger_path = OUT / 'remote_receipts/deletions.jsonl'
    ledger = [json.loads(line) for line in ledger_path.read_text(encoding='utf-8').splitlines()]
    protected = read(OUT / 'remote_receipts/protected_before.json')
    backend = ROOT / 'scripts/cleanup_cctv_dgp_vm_storage_20261009_v1.py'
    assert live['complete'] and verify['complete'] and cleanup['complete']
    assert live['plan_sha256'] == cleanup['plan_sha256'] == verify['plan_sha256'] == sha(plan_path)
    assert live['backend_sha256'] == cleanup['script_sha256'] == verify['script_sha256'] == sha(backend)
    for filename, field in (('verification.json', 'verification_sha256'), ('cleanup_receipt.json', 'cleanup_receipt_sha256'),
                            ('deletions.jsonl', 'deletions_sha256'), ('protected_before.json', 'protected_snapshot_sha256')):
        assert sha(OUT / 'remote_receipts' / filename) == live[field]
    count = len(plan['files'])
    assert count == len(ledger) == cleanup['files_removed'] == cleanup['archives_removed'] == live['deleted_files_verified_absent']
    assert {row['path'] for row in ledger} == {row['path'] for row in plan['files']}
    assert cleanup['archive_logical_bytes_removed'] == sum(row['bytes'] for row in plan['files'])
    assert all(cleanup[key] == 0 for key in ('inactive_cache_files_removed', 'duplicate_recognizer_files_removed', 'pip_cache_files_removed'))
    assert not cleanup['training_started'] and not cleanup['VM_stopped']
    assert live['GPU_idle'] and live['shared_VM_python_available'] and live['manual_tail_free_requirement_met']
    assert live['protected_hashed_files_live_verified'] == cleanup['protected_hashed_files_unchanged'] == len(protected['sha256'])
    assert live['retained_tensor_stamps_live_verified'] == cleanup['scientific_tensor_files_unchanged'] == len(protected['scientific_tensor_stamps'])
    assert live['explicit_current_research_bindings_live_verified'] == len(plan['protected_assets_sha256'])
    assert all(protected['sha256'][name] == digest for name, digest in plan['protected_assets_sha256'].items())
    before_verification = read(OUT / 'local_preapply_backup_recheck.json')
    assert before_verification['complete'] and before_verification['plan_sha256'] == sha(plan_path)
    for row in plan['files']:
        backup = Path(row['local_backup'])
        assert backup.is_file() and backup.stat().st_size == row['bytes']
        assert backup.stat().st_mtime_ns == row['local_backup_mtime_ns']
    prior = read(PRIOR / 'milestone.json')
    prior_closure = read(PRIOR / 'independent_closure_audit_r1.json')
    assert prior_closure['complete'] and prior_closure['milestone_sha256'] == sha(PRIOR / 'milestone.json')
    handoff = ROOT / 'PROJECT_HANDOFF.md'
    assert sha(handoff) == prior['new_evidence_sha256']['PROJECT_HANDOFF.md']
    app = prior['app_bindings_sha256']
    for name, digest in app.items():
        assert sha(ROOT / name) == digest
    guide = ROOT / 'CCTV_DGP_ACTUAL_STEP_TAIL_V1_VM.md'
    assert sha(guide) == prior['new_evidence_sha256']['CCTV_DGP_ACTUAL_STEP_TAIL_V1_VM.md']
    parent_archive = ROOT / 'outputs/cctv-dgp-actual-step-review-v1-results.tar.gz'
    assert parent_archive.is_file() and parent_archive.stat().st_size == 3204580880
    assert next(row for row in plan['files'] if row['local_backup'] == str(parent_archive))['sha256'] == 'd380ce08cf858eebc334fcad3c6f5590e01d526084239bbf807bd4b465e8c604'
    before = handoff.read_bytes()
    with (OUT / 'PROJECT_HANDOFF_before_cleanup.md').open('xb') as stream:
        stream.write(before)
    reclaimed = cleanup['free_bytes_increase'] / 1024**3
    initial = cleanup['free_bytes_before'] / 1024**3
    final = live['free_bytes_live'] / 1024**3
    document = ROOT / 'CCTV_DGP_VM_STORAGE_CLEANUP_20261009_V1.md'
    body = f'''# Verified VM storage cleanup — 9 October 2026

Authorized maintenance is complete on the existing forensic-dgp-thesis VM.
The exact plan removed {count} inactive top-level archive copies with retained,
SHA256-matched Windows backups. Observed free space increased by
{reclaimed:.3f}GiB during apply. The separate live audit reports {final:.3f}GiB free.

| Measurement | GiB |
| --- | ---: |
| Free immediately before apply | {initial:.3f} |
| Free at the independent live audit | {final:.3f} |
| Observed increase during apply | {reclaimed:.3f} |

Deletion was restricted to owned regular files in /home/janusdominic0, with one
link, exact inode/size/time identity and matching archive hashes. No recursive
delete occurred. Every Windows archive, checksum and export receipt stays local.
The original actual-step partial-return archive is retained at
outputs/cctv-dgp-actual-step-review-v1-results.tar.gz with hash
d380ce08cf858eebc334fcad3c6f5590e01d526084239bbf807bd4b465e8c604.

The independent VM auditor verifies all {count} deleted paths are absent,
{live['protected_hashed_files_live_verified']:,} protected research-file hashes and
{live['retained_tensor_stamps_live_verified']:,} retained scientific-tensor stamps.
It separately verifies all {len(plan['protected_assets_sha256']):,} explicitly pinned
current diagnostic bindings, including all 828 existing control/partial files.
Original checkpoints, instructions, source, splits, provenance, failure logs,
research caches and the shared VM Python runtime stay in place. All {len(app)} local
DGP-primary app bindings remain exact. The current execution archive is retained.

No model inference, gradient, optimizer, training, diagnostic or VM restart was
launched by maintenance. The GPU remains idle at the live audit. The final-state
tail remains uninstalled at the initial snapshot and unrun by this workflow.
Its manual guide CCTV_DGP_ACTUAL_STEP_TAIL_V1_VM.md is unchanged; its 2GiB free-space
requirement is met. The original 3GiB encoded-output protocol stop remains a
scientific failure record, separate from VM free-space availability.

Exact plan, local backup verification, pre-apply recheck, remote receipts,
deletion ledger and independent live audit are retained under
outputs/cctv_dgp_vm_storage_cleanup_20261009_v1/. Storage completion does not
qualify restoration or completion quality; the full research goal remains active.
'''
    with document.open('x', encoding='utf-8', newline='\n') as stream:
        stream.write(body)
    intro = f'''**Maintenance milestone — 9 October 2026: backed archive cleanup completed; final-state diagnostic remains manual.**

Removed {count} exact, inactive top-level VM archive copies after fresh inventory,
full local/remote SHA256 backup checks and a pre-apply Windows backup recheck.
Observed free-space increase: {reclaimed:.3f}GiB. Separate live audit: {final:.3f}GiB free.
All Windows originals remain retained. The audit confirms {count} paths absent,
{live['protected_hashed_files_live_verified']:,} protected research-file hashes unchanged,
{live['retained_tensor_stamps_live_verified']:,} scientific tensor stamps retained and
all {len(plan['protected_assets_sha256']):,} current diagnostic bindings exact, including
the 828 overlap files required by the frozen final-state packet.
Original checkpoints, data/splits, research caches, provenance, instructions,
sources and failure records remain on the VM. The shared Python runtime is present;
GPU idle. All {len(app)} DGP-primary local app bindings and the manual tail guide remain exact.

No training, inference or diagnostic was launched by maintenance. The user will
manually follow CCTV_DGP_ACTUAL_STEP_TAIL_V1_VM.md; its 2GiB free-space requirement
is met. Preserve the original 3GiB encoded-output protocol stop and all scientific
gate failures. Cleanup is not model qualification. Useful native restoration,
seven-family automatic/assisted completion and independent final review remain
outstanding. Evidence: CCTV_DGP_VM_STORAGE_CLEANUP_20261009_V1.md and
outputs/cctv_dgp_vm_storage_cleanup_20261009_v1/.

'''
    handoff.write_bytes(intro.encode('utf-8') + before)
    bound = {}
    for path in sorted(OUT.rglob('*')):
        if path.is_file():
            bound[str(path.relative_to(ROOT)).replace('\\', '/')] = sha(path)
    for name in ('scripts/inspect_cctv_dgp_vm_storage_20261009_v1.py', 'scripts/prepare_cctv_dgp_vm_storage_20261009_v1.py',
                 'scripts/storage_gcloud_20261009_v1.py', 'scripts/cleanup_cctv_dgp_vm_storage_20261009_v1.py',
                 'scripts/independent_cctv_dgp_vm_storage_cleanup_20261009_v1.py',
                 'scripts/record_cctv_dgp_vm_storage_20261009_v1.py',
                 'scripts/final_cctv_dgp_vm_storage_20261009_v1_inventory.py',
                 'scripts/observe_cctv_dgp_vm_storage_20261009_v1_deletions.py',
                 'scripts/verify_cctv_dgp_vm_storage_20261009_v1_milestone.py',
                 'CCTV_DGP_VM_STORAGE_CLEANUP_20261009_V1.md', 'PROJECT_HANDOFF.md',
                 'CCTV_DGP_ACTUAL_STEP_TAIL_V1_VM.md'):
        bound[name] = sha(ROOT / name)
    write(OUT / 'milestone.json', {'complete': True, 'UTC': datetime.now(timezone.utc).isoformat(),
          'previous_milestone_sha256': sha(PRIOR / 'milestone.json'),
          'previous_closure_sha256': sha(PRIOR / 'independent_closure_audit_r1.json'),
          'previous_handoff_sha256': hashlib.sha256(before).hexdigest(),
          'previous_handoff_exact_bytes_preserved': True, 'new_intro_bytes': len(intro.encode('utf-8')),
          'new_evidence_sha256': bound, 'app_bindings_sha256': app,
          'local_backup_bindings': {row['local_backup']: row['sha256'] for row in plan['files']},
          'files_removed': count, 'archive_logical_bytes_removed': cleanup['archive_logical_bytes_removed'],
          'observed_free_bytes_increase': cleanup['free_bytes_increase'],
          'independent_live_free_bytes': live['free_bytes_live'],
          'protected_research_hash_count': live['protected_hashed_files_live_verified'],
          'retained_scientific_tensor_stamp_count': live['retained_tensor_stamps_live_verified'],
          'explicit_diagnostic_binding_count': len(plan['protected_assets_sha256']),
          'training_or_diagnostic_launched': False, 'app_promotion': False,
          'manual_tail_execution_required': True, 'model_qualification': False, 'goal_complete': False,
          'seconds': time.monotonic() - started})
    print({'complete': True, 'archives_removed': count, 'free_GiB': final,
           'handoff_updated': True, 'manual_tail_guide_unchanged': True}, flush=True)


if __name__ == '__main__':
    main()
