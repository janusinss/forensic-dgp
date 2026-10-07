"""Close verified archive-only maintenance; preserve complete research history."""
from datetime import datetime, timezone
import hashlib
import json
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
OUT = ROOT / 'outputs/cctv_dgp_vm_storage_cleanup_20261006_v2'
PREVIOUS = ROOT / 'outputs/dgp_v26_gradient_return_and_feature_skips_v27_milestone/milestone.json'


def sha(path):
    digest = hashlib.sha256()
    with Path(path).open('rb') as stream:
        for block in iter(lambda: stream.read(4 * 1024**2), b''):
            digest.update(block)
    return digest.hexdigest()


def read(path):
    return json.loads(Path(path).read_text(encoding='utf-8-sig'))


def write(path, value):
    with path.open('x', encoding='utf-8', newline='\n') as stream:
        stream.write(json.dumps(value, indent=2, allow_nan=False) + '\n')


def main():
    assert not (OUT / 'closure_manifest.json').exists()
    plan_path = OUT / 'archive-duplicates_plan_20261006_v2.json'
    plan = read(plan_path)
    files = plan['files']
    verification_path = OUT / 'remote_receipts/verification.json'
    receipt_path = OUT / 'remote_receipts/cleanup_receipt.json'
    snapshot_path = OUT / 'remote_receipts/protected_before.json'
    ledger_path = OUT / 'remote_receipts/deletions.jsonl'
    verification, receipt = read(verification_path), read(receipt_path)
    audit = read(OUT / 'independent_VM_audit.log')
    runtime = read(OUT / 'final_runtime.json')
    snapshot = read(snapshot_path)
    pin = sha(plan_path)
    backend = ROOT / 'scripts/cleanup_cctv_dgp_vm_storage_20261006_v2.py'
    assert len(files) == receipt['archives_removed'] == receipt['files_removed'] == 14
    assert sum(row['bytes'] for row in files) == receipt['archive_logical_bytes_removed'] == 1756556632
    assert receipt['complete'] and audit['complete'] and runtime['complete'] and verification['complete']
    assert pin == receipt['plan_sha256'] == audit['plan_sha256'] == verification['plan_sha256']
    assert sha(backend) == receipt['script_sha256'] == audit['backend_sha256'] == verification['script_sha256']
    assert receipt['inactive_cache_files_removed'] == receipt['pip_cache_files_removed'] == 0
    assert plan['clear_pip_download_cache'] is False and plan['scientific_cache_removal_authorized'] is False
    assert not receipt['training_started'] and not receipt['VM_stopped'] and runtime['GPU_idle']
    assert runtime['runtime']['CUDA_available'] and runtime['GPU'] == 'NVIDIA L4'
    assert runtime['retained_cache_files'] == 4431 and runtime['retained_cache_bytes'] == 39448585279
    for field, path in (
        ('verification_sha256', verification_path), ('cleanup_receipt_sha256', receipt_path),
        ('deletions_sha256', ledger_path), ('protected_snapshot_sha256', snapshot_path),
    ):
        assert audit[field] == sha(path)
    assert verification['protected_snapshot_sha256'] == sha(snapshot_path)
    assert audit['deleted_files_verified_absent'] == 14
    assert audit['protected_hashed_files_live_verified'] == receipt['protected_hashed_files_unchanged'] == len(snapshot['sha256'])
    assert audit['retained_tensor_stamps_live_verified'] == receipt['scientific_tensor_files_unchanged'] == len(snapshot['scientific_tensor_stamps'])
    assert audit['explicit_current_research_bindings_live_verified'] == len(plan['protected_assets_sha256']) == 490
    assert all(snapshot['sha256'][name] == digest for name, digest in plan['protected_assets_sha256'].items())
    ledger = [json.loads(line) for line in ledger_path.read_text().splitlines()]
    assert len(ledger) == 14 and {row['path'] for row in ledger} == {row['path'] for row in files}
    for row in files:
        local = Path(row['local_backup']).resolve()
        assert local.is_relative_to(ROOT / 'outputs') and local.stat().st_size == row['bytes'] and sha(local) == row['sha256']
        deleted = next(item for item in ledger if item['path'] == row['path'])
        assert deleted['bytes'] == row['bytes'] and deleted['local_backup'] == row['local_backup']
    windows = {'complete': True, 'scope': 'Separate Windows recovery archive and returned live receipt audit',
               'archives_verified': 14, 'bytes_verified': 1756556632, 'receipt_sha256': sha(receipt_path),
               'live_audit_sha256': sha(OUT / 'independent_VM_audit.log'),
               'protected_snapshot_sha256': sha(snapshot_path), 'files_removed_by_this_audit': 0}
    write(OUT / 'Windows_independent_audit.json', windows)
    app_path = ROOT / 'outputs/dgp_app_v3_integration_record.json'
    app = read(app_path)
    assert sha(app_path) == 'e744f3708f248012469a111e204222f66261af4ca28bbaf9521ee4c5464b5e7a'
    for name, digest in {**app['sources_sha256'], **app['evidence_sha256']}.items():
        assert sha(ROOT / name) == digest
    goal = read(OUT / 'active_goal_snapshot.json')['goal']
    assert goal['status'] == 'active' and 'Latest explicit authorization:' in goal['objective']
    literal_goal = goal['objective'].split(' Latest explicit authorization:', 1)[0]
    assert 'No configured SSH connection exists:' in literal_goal
    assert sha(PREVIOUS) == '896204c16b02aa46e5b8613df2597bcf6d9bcf78a64fcd57ab4f6d6f922d832c'
    old = read(PREVIOUS)
    before = OUT / 'before_docs'
    before.mkdir(exist_ok=False)
    mapping = {}
    originals = {}
    for name in ('PROJECT_HANDOFF.md', 'SYSTEM_WORKFLOW_AND_GOAL.md', 'PRACTICAL_OUTPUT_SCOPE.md'):
        path = ROOT / name
        original = path.read_bytes()
        backup = before / name
        with backup.open('xb') as stream:
            stream.write(original)
        originals[name] = original
        mapping[name] = backup.relative_to(ROOT).as_posix()
    for name, digest in old['new_evidence_sha256'].items():
        assert sha(ROOT / mapping.get(name, name)) == digest, name
    free = runtime['free_bytes']
    added = receipt['free_bytes_increase']
    common = f'''**Latest authorization and maintenance — 6 October 2026: VM cleanup complete; full DGP goal active.**

The user explicitly authorizes direct connection/storage cleanup before continuing
the DGP workflow. The existing gcloud connection to forensic-dgp-thesis in
us-central1-a is verified. This maintenance exception supersedes the prior
no-connection restriction for cleanup only; actual training remains manual on
the existing NVIDIA L4/g2-standard-4 under ~/forensic-dgp. No historical pilot,
instance start/stop, model training, process termination or app promotion occurs.

Fourteen exact duplicate top-level transfer/result archives are removed after
complete retained Windows copies and VM SHA256/stamp checks: 1,756,556,632 bytes
(1.64 GiB). The apply receipt measures {added:,} additional free bytes. The final
live check reports {free:,} bytes ({free / 1024**3:.2f} GiB) free. Separate live
and Windows audits verify the deletion ledger, {len(snapshot['sha256']):,} protected
file hashes, {len(snapshot['scientific_tensor_stamps']):,} scientific tensor stamps
and 490 current V26/V27 bindings. Original unpacked checkpoints, sources, inputs,
splits, outputs and failed gates remain. All three scientific caches stay intact:
4,431 files/39,448,585,279 bytes. The existing CUDA runtime is available; L4 idle.
Current V27 execution/return archives and all Windows recovery copies stay intact.

V27 is reported stopped at50/800 updates for 0.221088578% structure improvement,
below the unchanged1% early requirement. Its complete 325,777,145-byte return,
checksum and export receipt are present on Windows, with reported SHA256
1bd4aec2bdc10cd6aca4455f24875b006bf4bca2d1a4f1cae8f4932400b386ab.
Cleanup confirms transfer hash only; independent result replay and visual review
are the next research actions. Export completion does not imply quality success.
Do not rerun V27 unchanged or relax its stop. Following the third spatial-path
failure, stop model modifications for the required architecture discussion.

All previous309 research bindings, deeper66/692/299/697/513 histories, three full
document bodies, earlier maintenance and app22 bindings remain preserved. The
user's restated goal takes precedence over historical manuscript/permission text.
Own-trained DGP remains primary; pretrained restoration models are comparisons.
All visible facial features, native unpaired evidence, separate paired synthetic
metrics, all seven automatic/assisted covering families, exact local app flow and
independent final review remain required. No new native/reserved-final pixels or
ethnicity/Zamboanga/hidden-identity claims enter maintenance. Goal incomplete.

[Cleanup evidence](<C:/xampp/htdocs/YEAR 4/Testing/outputs/cctv_dgp_vm_storage_cleanup_20261006_v2/closure_manifest.json>) ·
[Cleanup report](<C:/xampp/htdocs/YEAR 4/Testing/VM_STORAGE_CLEANUP_20261006_V2.md>)

Earlier document bodies below are preserved history. Their earlier pilot-launch
instructions and no-connection statements are historical; current authorization
and the retained quality failures above govern the next action.

'''
    for name, original in originals.items():
        split = original.index(b'\n') + 1
        prefix = common
        if name == 'SYSTEM_WORKFLOW_AND_GOAL.md':
            prefix += '**Exact goal restated by the user on6 October2026:**\n\n' + literal_goal + '\n\n'
            prefix += 'The explicit cleanup request permits current maintenance access; future training uses verified transfers and pasteable commands.\n\n'
        (ROOT / name).write_bytes(original[:split] + b'\n' + prefix.encode('utf-8') + original[split:])
    report = '# VM archive cleanup — 6 October 2026, second maintenance pass\n\n' + common
    report += '\nExact removed paths and Windows recovery files are in the pinned plan and returned deletion ledger. Verification precedes all removal; all unlinks target exact owned regular non-symlink/non-hardlinked files. No recursive deletion.\n'
    report_path = ROOT / 'VM_STORAGE_CLEANUP_20261006_V2.md'
    with report_path.open('x', encoding='utf-8', newline='\n') as stream:
        stream.write(report)
    evidence = {str(path.relative_to(ROOT)).replace('\\', '/'): sha(path)
                for path in OUT.rglob('*') if path.is_file()}
    for name in ('PROJECT_HANDOFF.md', 'SYSTEM_WORKFLOW_AND_GOAL.md', 'PRACTICAL_OUTPUT_SCOPE.md',
                 'VM_STORAGE_CLEANUP_20261006_V2.md', 'scripts/storage_gcloud_20261006_v2.py',
                 'scripts/prepare_cctv_dgp_vm_storage_cleanup_20261006_v2.py',
                 'scripts/cleanup_cctv_dgp_vm_storage_20261006_v2.py',
                 'scripts/independent_cctv_dgp_vm_storage_cleanup_20261006_v2.py',
                 'scripts/finish_cctv_dgp_vm_storage_cleanup_20261006_v2.py',
                 'scripts/verify_cctv_dgp_vm_storage_cleanup_20261006_v2.py'):
        evidence[name] = sha(ROOT / name)
    closure = {'complete': True, 'created_utc': datetime.now(timezone.utc).isoformat(),
               'scope': 'Authorized archive-only cleanup complete; full restoration/completion goal remains active',
               'archives_removed': 14, 'logical_bytes_removed': 1756556632,
               'measured_free_bytes_increase': added, 'final_free_bytes': free,
               'scientific_cache_files_removed': 0, 'protected_hashes_live_verified': len(snapshot['sha256']),
               'scientific_tensor_stamps_live_verified': len(snapshot['scientific_tensor_stamps']),
               'current_research_bindings_live_verified': 490, 'CUDA_available': True, 'GPU_idle': True,
               'training_started': False, 'VM_started_or_stopped': False, 'remote_task_killed': False,
               'current_V27_archives_retained': True, 'all_Windows_recovery_archives_retained': True,
               'previous_milestone_sha256': sha(PREVIOUS), 'previous309_original_locations': mapping,
               'previous309_preserved': len(old['new_evidence_sha256']), 'app22_bindings_preserved': True,
               'new_evidence_sha256': evidence, 'goal_status': 'active', 'goal_complete': False}
    write(OUT / 'closure_manifest.json', closure)
    print(json.dumps({key: closure[key] for key in ('complete', 'archives_removed', 'logical_bytes_removed',
        'measured_free_bytes_increase', 'final_free_bytes', 'protected_hashes_live_verified', 'goal_status')}, indent=2))


if __name__ == '__main__':
    main()
