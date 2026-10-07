"""Record the independently audited cleanup, preserving complete prior documents."""
from datetime import datetime, timezone
import hashlib
import json
from pathlib import Path

from verify_cctv_dgp_vm_storage_cleanup_20261007_v2 import ROOT, OUT, BACKUP, PIN, read, sha, verify


def write(path, value):
    with path.open('x', encoding='utf-8', newline='\n') as stream:
        stream.write(json.dumps(value, indent=2, allow_nan=False) + '\n')


def main():
    assert not (OUT / 'closure_manifest.json').exists(), 'Preserve completed maintenance'
    report = verify()
    assert not report['V30_installed'] and not report['V30_outputs_present'], 'Use existing-run evidence before issuing new install steps'
    initial = report['initial_inventory_free_bytes']
    final = report['final_live_free_bytes']
    increase = report['measured_cleanup_free_bytes_increase']
    common = f'''**Latest maintenance â€” 7 October 2026: inactive VM caches removed after full local backup verification.**

The latest explicit user authorization permits direct maintenance connection to
the existing forensic-dgp-thesis/us-central1-a VM. Fresh inventory confirms the
original instance and 100 GB boot-disk identities, an idle NVIDIA L4 and no active
cache readers. The offered SSH key matches the previously trusted fingerprint;
the initial uncached-IP stop is retained. Linux restricted the first added
process-file check; a separate read-only privileged inspection established the
safe correction. The stopped verifier and original source remain preserved.
Exact file unlinks still run as the original VM user, without root deletion.

All 4,431 actual Windows cache files/39,448,585,279 bytes pass fresh full SHA256
verification against the frozen source inventory. All 4,431 VM hashes match those
copies. Cleanup removes only 4,429 inactive .bin/.npz files:39,442,892,400 logical
bytes. These are four expanded-feature binary files and 4,425 closed V16 r2 cache
files. Their two original state.json metadata files remain on the VM, together
with original DGP checkpoints, datasets, splits, source, research markdown,
provenance, logs and failed gates. The Windows actual cache copies remain intact.
Closed historical recipes must stay closed; restore a required historical cache
from the verified local files before a justified later read-only audit.

Initial live free space is {initial:,} bytes ({initial / 1024**3:.2f} GiB).
The removal receipt measures {increase:,} additional free bytes
({increase / 1024**3:.2f} GiB); final live free space is{final:,} bytes
({final / 1024**3:.2f} GiB). Independent VM verification confirms every planned
deletion, {report['protected_VM_file_hashes_verified']:,} retained file hashes,
{report['protected_VM_tensor_stamps_verified']:,} retained tensor stamps and all 5,757 current
V30 dependency bindings, including 5,467 TRAIN files. The existing CUDA runtime
remains available. No active workload is killed, VM stopped or model promoted.

V29 remains rejected for application promotion. V30's distinct broader TRAIN
coverage packet and finite 800-update protocol remain unchanged and verified.
V30 is not installed or started. Actual training remains the user's manual
upload/install/tmux workflow on the existing L4/g2-standard-4 at ~/forensic-dgp.
Use the five exact steps in CCTV_DGP_BROADER_MEAN_V30_VM.md, including separate
PuTTY-compatible downloads. The unchanged 6 GiB requirement is satisfied.
Returned results still require independent audit, paired synthetic and unpaired
native evidence kept separate, useful native structure review and preservation
gates. Useful DGP restoration, all seven covering families, independent final
review and full bundled inline Playwright app verification remain required.
Goal active/incomplete.

[Cleanup evidence](<C:/xampp/htdocs/YEAR 4/Testing/outputs/cctv_dgp_vm_storage_cleanup_20261007_v2/closure_manifest.json>) Â·
[V30 manual steps](<C:/xampp/htdocs/YEAR 4/Testing/CCTV_DGP_BROADER_MEAN_V30_VM.md>) Â·
[Actual Windows cache backup](<C:/xampp/htdocs/YEAR 4/Testing/outputs/cctv_dgp_local_research_cache_backup_20261007_v1/complete.json>)

Earlier complete document bodies remain preserved history. Current cache
locations and live free space above supersede earlier storage readings; prior
restoration findings, quality failures and research scope remain unchanged.

'''
    docs = []
    for name in ('PROJECT_HANDOFF.md', 'SYSTEM_WORKFLOW_AND_GOAL.md', 'PRACTICAL_OUTPUT_SCOPE.md'):
        original = (OUT / 'before_docs' / name).read_bytes()
        current = ROOT / name
        assert current.read_bytes() == original, 'Preserve intervening user edits'
        split = original.index(b'\n') + 1
        addition = b'\n' + common.encode('utf-8')
        after = original[:split] + addition + original[split:]
        current.write_bytes(after)
        assert current.read_bytes() == after and after[:split] + after[split + len(addition):] == original
        docs.append({'name': name, 'before_sha256': hashlib.sha256(original).hexdigest(),
                     'after_sha256': sha(current), 'complete_previous_body_preserved': True})
    guide = ROOT / 'VM_STORAGE_CLEANUP_20261007_V2.md'
    with guide.open('x', encoding='utf-8', newline='\n') as stream:
        stream.write('# Verified inactive-cache cleanup â€” 7 October 2026\n\n' + common)
        stream.write('The frozen plan records every removed absolute VM path, byte count, SHA256 and actual Windows recovery path. The returned deletion ledger and independent VM audit prove the exact scope. No recursive deletion. The existing local backup/restore guide describes the three-cache recovery layout; this cleanup does not constitute a whole-VM migration backup.\n')
    evidence = {path.relative_to(ROOT).as_posix(): sha(path) for path in OUT.rglob('*') if path.is_file()}
    for path in [guide, *(ROOT / row['name'] for row in docs),
                 *(ROOT / 'scripts' / name for name in (
                     'prepare_cctv_dgp_vm_storage_cleanup_20261007_v2.py',
                     'cleanup_cctv_dgp_vm_storage_20261007_v2.py',
                     'cleanup_cctv_dgp_vm_storage_20261007_v2_r1.py',
                     'independent_cctv_dgp_vm_storage_cleanup_20261007_v2.py',
                     'independent_cctv_dgp_vm_storage_cleanup_20261007_v2_r1.py',
                     'storage_gcloud_20261007_v2.py', 'storage_gcloud_20261007_v2_r1.py',
                     'verify_cctv_dgp_vm_runtime_after_storage_cleanup_20261007_v2.py',
                     'verify_cctv_dgp_vm_storage_cleanup_20261007_v2.py',
                     'finish_cctv_dgp_vm_storage_cleanup_20261007_v2.py'))]:
        evidence[path.relative_to(ROOT).as_posix()] = sha(path)
    closure = {**report, 'created_utc': datetime.now(timezone.utc).isoformat(),
               'scope': 'Authorized storage cleanup complete; DGP research goal active',
               'documents': docs, 'new_evidence_sha256': evidence,
               'local_backup_manifest_sha256': PIN,
               'local_backup_closure_sha256': sha(BACKUP / 'closure_manifest.json'),
               'previous_history_sha256': sha(OUT / 'local_history_pre_cleanup_audit.json'),
               'VM_maintenance_authorized': True, 'training_execution_remains_manual': True,
               'automatic_training_started': False, 'VM_stopped': False,
               'original_checkpoints_splits_gate_failures_preserved': True,
               'goal_status': 'active', 'goal_complete': False}
    write(OUT / 'closure_manifest.json', closure)
    print(json.dumps({key: closure[key] for key in ('complete', 'removed_inactive_cache_files',
                     'measured_cleanup_free_bytes_increase', 'final_live_free_bytes', 'goal_complete')}, indent=2))


if __name__ == '__main__':
    main()
