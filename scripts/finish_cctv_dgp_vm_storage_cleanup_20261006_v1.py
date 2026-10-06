"""Record completed storage maintenance without additional VM actions."""
import hashlib
import json
from datetime import datetime, timezone
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
OUT = ROOT / 'outputs/cctv_dgp_vm_storage_cleanup_20261006_v1'


def sha(path):
    digest = hashlib.sha256()
    with path.open('rb') as stream:
        for block in iter(lambda: stream.read(4 * 1024**2), b''):
            digest.update(block)
    return digest.hexdigest()


def read(name):
    return json.loads((OUT / name).read_text(encoding='utf-8-sig'))


def write_new(name, value):
    with (OUT / name).open('x', encoding='utf-8', newline='\n') as stream:
        stream.write(json.dumps(value, indent=2) + '\n')


def main():
    assert not (OUT / 'closure_manifest.json').exists(), 'Preserve existing closure'
    plan = read('archive-duplicates_plan.json')
    receipt = read('remote_receipts/archive-duplicates/cleanup_receipt.json')
    windows = read('archive-duplicates_Windows_independent_audit.json')
    runtime = read('final_runtime_after_cleanup.json')
    stopped = read('partial_backup_stopped_live_check.json')
    assert receipt['complete'] and windows['complete'] and runtime['complete'] and stopped['complete']
    assert receipt['files_removed'] == receipt['archives_removed'] == windows['archives_removed'] == 7
    assert receipt['inactive_cache_files_removed'] == windows['cache_files_removed'] == 0
    assert receipt['archive_logical_bytes_removed'] == 2341409251
    assert sha(OUT / 'archive-duplicates_plan.json') == receipt['plan_sha256']
    assert sha(ROOT / 'scripts/cleanup_cctv_dgp_vm_storage_20261006_v1.py') == receipt['script_sha256']
    assert sha(OUT / 'remote_receipts/archive-duplicates/cleanup_receipt.json') == windows['receipt_sha256']
    assert sha(OUT / 'audit-archives_file_r1.log') == windows['live_audit_sha256']
    assert runtime['retained_cache_files'] == 4431 and runtime['retained_cache_bytes'] == 39448585279
    assert runtime['runtime']['CUDA_available'] and runtime['GPU_idle']
    assert not runtime['training_started'] and not runtime['VM_stopped']
    assert stopped['matching_assistant_backup_processes'] == stopped['remote_processes_stopped'] == 0
    partial = OUT / 'cache_backups/group0/anatomical_cache/features.bin'
    assert partial.stat().st_size == stopped['partial_backup_bytes'] == 3205955584
    before = read('before_docs/backup_receipt.json')
    for row in before:
        assert sha(Path(row['backup'])) == row['sha256']
        if Path(row['path']).name != 'PROJECT_HANDOFF.md':
            assert sha(Path(row['path'])) == row['sha256'], 'Storage report changed after backup'
    # Preserve a concurrent research milestone instead of restoring an older handoff.
    handoff_path = ROOT / 'PROJECT_HANDOFF.md'
    handoff_bytes = handoff_path.read_bytes()
    handoff_sha = hashlib.sha256(handoff_bytes).hexdigest()
    latest_before = OUT / 'before_docs_r1'
    latest_before.mkdir(exist_ok=False)
    with (latest_before / 'PROJECT_HANDOFF.md').open('xb') as stream:
        stream.write(handoff_bytes)
    write_new('documentation_concurrent_update.json', {
        'complete': True, 'original_handoff_sha256': before[0]['sha256'],
        'latest_handoff_sha256': handoff_sha,
        'latest_handoff_backup': str(latest_before / 'PROJECT_HANDOFF.md'),
        'cause': 'New V23 audit and V24 preparation research entry arrived during storage maintenance',
        'fix': 'Preserve the entire latest handoff body; prepend storage record only',
        'research_entry_modified': False})
    v23 = ROOT / 'outputs/cctv-dgp-detail-skip-v23-results.tar.gz'
    v23_sha = 'b71639cabd17476d5989dfd4582d47de9b747b8bf2a611232e5f62cc67bf9fd9'
    assert v23.stat().st_size == 76481832
    checksum = v23.with_name(v23.name + '.sha256')
    assert checksum.read_text().split()[0] == v23_sha
    export_path = ROOT / 'outputs/cctv-dgp-detail-skip-v23-export.json'
    export = json.loads(export_path.read_text())
    assert export['complete'] and export['archive_sha256'] == v23_sha and export['bytes'] == 76481832

    report = '''# VM storage cleanup - 6 October 2026

Completed on the existing `forensic-dgp-thesis` VM in `us-central1-a`.
Seven duplicate transfer/result archives were removed after SHA256 verification
against retained Windows backups: 2,341,409,251 bytes (2.18 GiB).
The final live check found **9,880,526,848 bytes free (9.20 GiB)** on the
103,865,303,040-byte filesystem. The VM remains running; its NVIDIA L4 is idle.

The user's decision, "pick what doesnt disrupt the future vm usage," closes the
cleanup at these backed-up archives. Scientific caches and the installed runtime
are retained for future work. Storage maintenance does not authorize training.

## Verified removal and preservation

The exact seven paths, sizes, hashes and Windows recovery copies are recorded in
`outputs/cctv_dgp_vm_storage_cleanup_20261006_v1/archive-duplicates_plan.json`
and `remote_receipts/archive-duplicates/deletions.jsonl` beneath that directory.
No recursive deletion was used. Pip-cache removals: 0. Scientific-cache removals: 0.

A separate live read-only auditor verifies all seven paths absent, all 2,630
protected research-file hashes unchanged, 4,817 retained scientific tensor stamps
unchanged and 1,139 explicit V22/V23 asset/result/archive bindings unchanged.
A separate Windows audit rehashes all seven recovery archives and checks the
downloaded receipts against the independent VM audit.

The apply receipt measures 2,331,934,720 additional free bytes. This differs
slightly from the archive-byte total because live filesystem usage also changes
during maintenance. Final free space comes from the later runtime/cache check.

## Future VM use preserved

All three historical scientific caches remain: expanded anatomical, expanded
fixed and V16 r2. Their 4,431 files total 39,448,585,279 bytes (36.74 GiB), including
metadata. The final check verifies exact file sets, retained tensor stamps and
metadata hashes against the protected snapshot. Original datasets, checkpoints,
splits, sources, logs, current V22/V23 packets and failed gates stay preserved.

The existing `~/forensic-dgp/cctv_dgp_vm_bundle/.venv/bin/python` successfully
imports Python 3.10.12, PyTorch 2.9.1+cu129, NumPy 1.26.4 and OpenCV 4.11.0.
CUDA is available; the GPU is NVIDIA L4. The existing `dgp_detail_skip_v23` tmux
pane contains an idle Bash shell. No model was loaded, no training was launched,
no remote task was killed, and the VM was neither started nor stopped here.

The unfinished Windows cache transfer was stopped after the user's decision.
Its 3,205,955,584-byte `features.bin` remains under
`outputs/cctv_dgp_vm_storage_cleanup_20261006_v1/cache_backups/group0/anatomical_cache/`.
It is a partial, unverified copy; it cannot authorize deleting the full VM cache.
Only the verified assistant-created local backup process tree was stopped.
The unified execution session ended with exit 1 from cancellation; the final
process check finds no matching backup process. No cache-removal plan was applied.

## Evidence and boundaries

Current evidence: `outputs/cctv_dgp_vm_storage_cleanup_20261006_v1/`.

1. `remote_receipts/archive-duplicates/`: verification, protected snapshot,
   cleanup receipt and exact deletion ledger.
2. `audit-archives_file_r1.log` and
   `archive-duplicates_Windows_independent_audit.json`: independent VM and
   retained Windows backup checks.
3. `final_runtime_after_cleanup.json`: final disk, cache, runtime, GPU and tmux
   inspection; no additional deletion.
4. `inactive_cache_manifest.json` and the two `partial_backup_stop*` receipts:
   full cache fingerprints and local cancellation scope.
5. `before_docs/`, `before_docs_r1/` and `closure_manifest.json`: original document bytes and
   completed maintenance evidence bindings.

Applied archive plan SHA256:
`0a4522835d221b2e7928966a9cf098c263e1849912730aa231d348b06b1dfb70`.
Applied backend SHA256:
`d5993d264ffdded86cc9c7d2ff1864cb4fd6c93d5d970b49131173cd48d13a8d`.
Verification/apply checks took 119.11/62.25 seconds; the independent live audit
took 43.79 seconds; the final runtime/cache inspection took 8.29 seconds.
Historical failed transport attempts and partial backups remain as evidence.

The earlier `20261005_r2` proposal stopped before upload/removal; its original
documents, plan and failures are preserved. This completed `20261006_v1` entry
supersedes the earlier stopped-VM and zero-removal notices.

The V23 return archive is now present on Windows and matches 76,481,832 bytes and
SHA256 `b71639cabd17476d5989dfd4582d47de9b747b8bf2a611232e5f62cc67bf9fd9`.
Its checksum and export receipt are present. A concurrent research milestone
records the independent V23 audit and prepares a distinct V24 experiment; its
entire handoff entry is preserved. This storage work performs no additional
training, gradient, replay or image-quality evaluation. The V23 structural stop
remains a failure; no unchanged rerun or checkpoint promotion follows from
cleanup. Actual training remains manual on the existing L4. The thesis
restoration/completion goal remains active.
'''
    (ROOT / 'VM_STORAGE_CLEANUP_20261006.md').write_text(report, encoding='utf-8', newline='\n')

    heading, history = handoff_bytes.split(b'\n', 1)
    entry = '''
**Latest 6 October 2026 - VM storage cleanup completed; future VM use preserved:**
The user authorizes direct VM storage maintenance and chooses "pick what doesnt
disrupt the future vm usage." On the existing running forensic-dgp-thesis VM,
seven duplicate archives totaling 2,341,409,251 bytes (2.18 GiB) are removed after
exact Windows backup/hash verification. Final live free space is 9,880,526,848
bytes (9.20 GiB). No pip-cache or scientific-cache file is removed.

Separate live VM and Windows audits verify the exact deletion ledger, all seven
retained recovery copies, 2,630 unchanged protected hashes, 4,817 retained tensor
stamps and 1,139 current V22/V23 bindings. Final runtime/cache inspection confirms
all 4,431 historical cache files (36.74 GiB), Python3.10.12/PyTorch2.9.1+cu129,
NumPy/OpenCV and available CUDA on NVIDIA L4. GPU idle; dgp_detail_skip_v23 tmux
contains Bash. The installed environment remains ready for future sessions.

Stop only the assistant-created incomplete Windows backup process tree; no
remote task is killed. Its 3,205,955,584-byte partial file is retained as
unverified evidence and cannot authorize VM cache removal. The VM stays running;
no instance start/stop, model load, actual training, app change or promotion.
This maintenance authorization is specific to storage; training remains manual.

The V23 results archive, checksum and export receipt are now present locally.
Archive size76,481,832 bytes/SHA256
`b71639cabd17476d5989dfd4582d47de9b747b8bf2a611232e5f62cc67bf9fd9`
match the reported return. The concurrent research milestone below records the
independent V23 audit and prepares a distinct V24 objective experiment. Preserve
that entire entry and its source/result/transfer artifacts. This maintenance
performs no additional model-quality evaluation; the V23 failure stays closed.

Original handoff/report bytes and the newer concurrent research handoff are
preserved under `outputs/cctv_dgp_vm_storage_cleanup_20261006_v1/before_docs/`
and `before_docs_r1/`. The entire current research body remains unchanged.
Evidence: `outputs/cctv_dgp_vm_storage_cleanup_20261006_v1/closure_manifest.json`.
Report: [VM storage cleanup](<C:/xampp/htdocs/YEAR 4/Testing/VM_STORAGE_CLEANUP_20261006.md>).
This current entry supersedes earlier stopped-VM/zero-removal notices below;
their prior evidence is retained. The full DGP restoration/covering goal remains
active and incomplete. V24 training remains on the existing manual VM path.

'''
    assert sha(handoff_path) == handoff_sha, 'Newer handoff update arrived; preserve it before merging'
    handoff_path.write_bytes(heading + b'\n' + entry.encode('utf-8') + history)
    assert handoff_path.read_bytes().endswith(history), 'Latest research body must remain byte-identical'

    bound = {}
    for path in OUT.rglob('*'):
        if path.is_file() and path.suffix.lower() in ('.json', '.jsonl', '.log', '.md'):
            bound[str(path)] = sha(path)
    helpers = ('cleanup_cctv_dgp_vm_storage_20261006_v1.py',
               'prepare_cctv_dgp_vm_storage_cleanup_20261006_v1.py',
               'storage_gcloud_20261006_v1.py',
               'independent_cctv_dgp_vm_storage_cleanup_20261006_v1.py',
               'verify_cctv_dgp_vm_runtime_after_storage_cleanup_20261006_v1.py',
               'backup_cctv_dgp_vm_inactive_caches_20261006_v1.py', Path(__file__).name)
    for name in helpers:
        path = ROOT / 'scripts' / name
        bound[str(path)] = sha(path)
    for name in ('PROJECT_HANDOFF.md', 'VM_STORAGE_CLEANUP_20261006.md'):
        bound[str(ROOT / name)] = sha(ROOT / name)
    for row in plan['files']:
        bound[row['local_backup']] = row['sha256']
    bound[str(v23)] = v23_sha
    bound[str(checksum)] = sha(checksum)
    bound[str(export_path)] = sha(export_path)
    for name in ('CCTV_DGP_DETAIL_SKIP_V23_RESULTS.md', 'CCTV_DGP_DEGRADED_DETAIL_V24_PLAN.md',
                 'CCTV_DGP_DEGRADED_DETAIL_V24_VM.md'):
        bound[str(ROOT / name)] = sha(ROOT / name)
    closure = {'complete': True, 'created_utc': datetime.now(timezone.utc).isoformat(),
        'scope': 'Storage maintenance only; seven backed-up duplicate archives removed',
        'user_decision': 'pick what doesnt disrupt the future vm usage',
        'archives_removed': 7, 'logical_archive_bytes_removed': 2341409251,
        'measured_free_bytes_increase': receipt['free_bytes_increase'],
        'final_free_bytes': runtime['free_bytes'], 'retained_cache_files': 4431,
        'retained_cache_bytes': 39448585279, 'cache_files_removed': 0,
        'protected_hashes_live_verified': 2630, 'retained_tensor_stamps_live_verified': 4817,
        'current_V22_V23_bindings_live_verified': 1139, 'runtime_CUDA_available': True,
        'VM_left_running': True, 'VM_start_stop_performed': False, 'training_started': False,
        'remote_task_killed': False, 'unfinished_local_backup_cancelled': True,
        'partial_backup': {'path': str(partial), 'bytes': partial.stat().st_size,
                           'complete': False, 'contents_verified': False, 'VM_original_retained': True},
        'V23_return_archive_present_and_hash_verified': True,
        'V23_research_audit_recorded_in_existing_handoff': True,
        'V23_quality_evaluation_performed_by_this_maintenance': False,
        'newer_research_handoff_body_preserved_byte_identically': True,
        'thesis_goal_complete': False, 'sha256': bound}
    write_new('closure_manifest.json', closure)
    # Independent readback covers document edits and all final evidence/recovery bindings.
    saved = read('closure_manifest.json')
    for name, digest in saved['sha256'].items():
        assert sha(Path(name)) == digest, 'Closure binding differs: ' + name
    write_new('closure_independent_readback.json', {
        'complete': True, 'closure_sha256': sha(OUT / 'closure_manifest.json'),
        'bindings_verified': len(saved['sha256']), 'all_seven_Windows_recovery_archives_rehashed': True,
        'V23_return_archive_rehashed': True, 'VM_actions_performed_by_this_readback': 0,
        'partial_backup_content_claimed_verified': False, 'thesis_goal_complete': False})
    print(json.dumps({'complete': True, 'archives_removed': 7, 'free_GiB': round(runtime['free_bytes'] / 1024**3, 2),
                      'bindings_verified': len(bound), 'closure_sha256': sha(OUT / 'closure_manifest.json')}))


if __name__ == '__main__':
    main()
