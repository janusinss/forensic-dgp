"""Record verified storage maintenance, retaining the entire research history."""
from datetime import datetime, timezone
import json
from pathlib import Path

from verify_cctv_dgp_vm_storage_cleanup_20261007_v1 import ROOT, OUT, PREVIOUS, PREVIOUS_PIN, history, maintenance, sha, read


def write(path, value):
    with Path(path).open('x', encoding='utf-8', newline='\n') as f:
        f.write(json.dumps(value, indent=2, allow_nan=False) + '\n')


def main():
    assert not (OUT / 'closure_manifest.json').exists()
    checked = maintenance()
    counts = history()
    runtime = read(OUT / 'final_runtime.json')
    free, added = checked['final_free_bytes'], checked['measured_free_bytes_increase']
    before = OUT / 'before_docs'
    before.mkdir(exist_ok=False)
    mapping, originals = {}, {}
    for name in ('PROJECT_HANDOFF.md', 'SYSTEM_WORKFLOW_AND_GOAL.md', 'PRACTICAL_OUTPUT_SCOPE.md'):
        original = (ROOT / name).read_bytes()
        backup = before / name
        with backup.open('xb') as f:
            f.write(original)
        originals[name] = original
        mapping[name] = backup.relative_to(ROOT).as_posix()
    common = f'''**Latest maintenance — 7 October 2026: VM cleanup complete; V30 remains ready for manual VM execution.**

The user explicitly authorizes direct connection and removal of unnecessary
VM files to reclaim at least1.3GiB, or more, before the6GiB V30 space check.
The existing forensic-dgp-thesis/us-central1-a connection is used for storage
maintenance only. Actual training remains manual VM execution on the existing
NVIDIA L4/g2-standard-4 at ~/forensic-dgp; no pilot or optimizer is launched.

Ten duplicate top-level transfer/result archives and four redundant pretrained
recognizer copies in closed V22/V23/V24/V25 packets are removed only after full
Windows backups and VM SHA256/stamp verification:1,915,161,691 logical bytes.
The four recognizer copies match the retained active V27 VM weight and their
original local packet manifests. V26's two-link copy stays; unlinking a shared
copy would not reclaim that storage. The initial conservative inventory stop
and the diagnostic that established this distinction are both retained.
Original trained DGP checkpoints, data, splits, logs, source, all failed gates
and all Windows recovery files remain. Historical V22–V25 recognizer copies
can be recovered from the exact Windows paths in the plan; do not rerun those
immutable failed pilots automatically. Research markdown files are preserved.

The pip download-cache scan yields{checked['pip_cache_files_removed']} eligible files.
Independent live and Windows audits verify
the exact deletion ledger, {checked['protected_hashed_files_live_verified']:,} protected file hashes,
{checked['scientific_tensor_stamps_live_verified']:,} retained tensor stamps and5757 current
dependency bindings, including all5467 V30 TRAIN assets. The three historical
scientific caches remain intact:4431 files/39,448,585,279 bytes. All11 tmux
sessions were at shell prompts before maintenance; the L4 remains idle and the
existing CUDA runtime is available. No process is killed or VM stopped.

The removal receipt measures {added:,} additional free bytes
({added / 1024**3:.2f}GiB). Final live free space is {free:,} bytes
({free / 1024**3:.2f}GiB), satisfying the unchanged6GiB V30 requirement.
V30 has not been uploaded or installed on this VM. Use the existing five manual
steps; its packet, protocol, finite800-update schedule and quality gates are
unchanged. Cleanup does not qualify V29 restoration or promote a model.

All previous5934 research bindings, the deeper371/1041/586/50/668/cleanup60/
309/66/692/299/697/513 history, complete document bodies and app22 bindings
remain preserved. Useful DGP native restoration, all seven covering families,
independent final review and the full local app flow remain required.
Goal active/incomplete.

[Cleanup evidence](<C:/xampp/htdocs/YEAR 4/Testing/outputs/cctv_dgp_vm_storage_cleanup_20261007_v1/closure_manifest.json>) ·
[V30 manual steps](<C:/xampp/htdocs/YEAR 4/Testing/CCTV_DGP_BROADER_MEAN_V30_VM.md>)

Earlier complete document bodies below remain preserved history. Their research
findings and quality failures retain their original scope; the current verified
free space above supersedes earlier storage readings.

'''
    for name, original in originals.items():
        split = original.index(b'\n') + 1
        (ROOT / name).write_bytes(original[:split] + b'\n' + common.encode('utf-8') + original[split:])
    report = ROOT / 'VM_STORAGE_CLEANUP_20261007_V1.md'
    with report.open('x', encoding='utf-8', newline='\n') as f:
        f.write('# Verified VM storage cleanup — 7 October 2026\n\n' + common)
        f.write('Exact removed paths and recovery locations are in the frozen plan and returned deletion ledger. Each unlink targets an owned regular canonical file with one hard link. No recursive deletion. Original maintenance evidence is unchanged.\n')
    evidence = {p.relative_to(ROOT).as_posix(): sha(p) for p in OUT.rglob('*') if p.is_file()}
    for p in (report, *(ROOT / name for name in originals),
              *(ROOT / 'scripts' / name for name in (
                  'storage_gcloud_20261007_v1.py', 'inspect_cctv_dgp_duplicate_recognizers_20261007_v1.py',
                  'prepare_cctv_dgp_vm_storage_cleanup_20261007_v1.py', 'cleanup_cctv_dgp_vm_storage_20261007_v1.py',
                  'independent_cctv_dgp_vm_storage_cleanup_20261007_v1.py',
                  'verify_cctv_dgp_vm_runtime_after_storage_cleanup_20261007_v1.py',
                  'finish_cctv_dgp_vm_storage_cleanup_20261007_v1.py', 'verify_cctv_dgp_vm_storage_cleanup_20261007_v1.py'))):
        evidence[p.relative_to(ROOT).as_posix()] = sha(p)
    closure = {'complete': True, 'created_utc': datetime.now(timezone.utc).isoformat(),
        'scope': 'Authorized VM storage cleanup complete; full research goal remains incomplete',
        **checked, 'archives_removed': 10, 'duplicate_pretrained_recognizer_copies_removed': 4,
        'scientific_cache_files_removed': 0, 'trained_DGP_checkpoint_files_removed': 0,
        'shared_V26_recognizer_retained': True, 'all_Windows_recovery_copies_retained': True,
        'current_V30_not_uploaded_or_installed': True, 'V30_training_started': False,
        'previous_milestone_sha256': PREVIOUS_PIN, 'previous5934_original_locations': mapping,
        'previous_history_binding_counts': counts, 'app22_bindings_preserved': True,
        'new_evidence_sha256': evidence, 'training_started': False, 'model_or_gradient_calls': 0,
        'VM_started_or_stopped': False, 'remote_task_killed': False,
        'goal_status': 'active', 'goal_complete': False}
    write(OUT / 'closure_manifest.json', closure)
    print(json.dumps({key: closure[key] for key in ('complete', 'archives_removed',
        'duplicate_pretrained_recognizer_copies_removed', 'pip_cache_files_removed',
        'measured_free_bytes_increase', 'final_free_bytes', 'goal_complete')}, indent=2))


if __name__ == '__main__':
    main()
