"""Record completed, independently audited maintenance; keep the full goal active."""
from datetime import datetime, timezone
import hashlib
import json
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
OUT = ROOT / 'outputs/cctv_dgp_vm_output_offload_20261010_v1_r2'
ARCHIVES = ROOT / 'outputs/cctv_dgp_bank_archive_cleanup_v1'


def read(path):
    return json.loads(path.read_text(encoding='utf-8'))


def sha(path):
    with path.open('rb') as stream:
        return hashlib.file_digest(stream, 'sha256').hexdigest()


def write(path, value):
    with path.open('x', encoding='utf-8', newline='\n') as stream:
        stream.write(json.dumps(value, indent=2, allow_nan=False) + '\n')


def main():
    assert not (OUT / 'closure.json').exists()
    audit = read(OUT / 'independent_cleanup_audit.json')
    backup = read(OUT / 'independent_backup_audit.json')
    post = read(OUT / 'inventory_after.json')
    original = read(ARCHIVES / 'independent_audit.json')
    plan = read(OUT / 'plan.json')
    export = read(OUT / 'recovery_export.json')
    presence = read(OUT / 'existing_cache_backup_presence_check.json')
    assert all(value['complete'] for value in [audit, backup, original, presence])
    assert read(OUT / 'post_execution.json')['complete'] and post['complete'] and post['manual_run_space_requirement_met']
    assert audit['plan_sha256'] == sha(OUT / 'plan.json')
    assert read(OUT / 'tail_recovery_v1/remote_receipts/verify_receipt.json') == read(OUT / 'verify_readback.json')
    assert backup['archive_sha256'] == audit['portable_backup_sha256'] == export['archive_sha256']
    assert audit['output_copies_offloaded'] == export['candidate_count'] == 76973
    assert not post['current_bank_installed'] and post['gpu_processes']['exit_code'] == 0 and not post['gpu_processes']['stdout'].strip()
    assert audit['model_gradient_or_training_calls'] == original['model_gradient_or_training_calls'] == 0
    pinned = read(ROOT / plan['local_protected_manifest'])
    for path, expected in pinned.items():
        assert sha(ROOT / path) == expected, path
    baseline = ROOT / 'outputs/cctv_dgp_face_code_fit_vm_v12_r2/weights/dgp_v2.pth'
    assert sha(baseline) == '646fbb11674e8882ea41e251d6e6e7d468c963979b48c93132def8292834e24b'
    initial = read(ARCHIVES / 'inventory_before.json')['disk']['free_bytes'] / 1024**3
    recovered = post['free_GiB'] - initial
    assert audit['original_apply_timeout_preserved'] and audit['additional_files_removed'] == 0
    destination = OUT / 'documents'
    destination.mkdir()
    handoff = ROOT / 'PROJECT_HANDOFF.md'
    before = handoff.read_bytes()
    (destination / 'PROJECT_HANDOFF.before.md').write_bytes(before)
    guide_path = ROOT / 'VM_IMAGE_OUTPUT_BACKUP_RESTORE_20261010_V1.md'
    assert not guide_path.exists()
    guide = f'''# Verified local image-output backup and VM cleanup — 10 October 2026

The user requested additional VM space and a local backup for a later VM change.
The existing instance remains `forensic-dgp-thesis`, project `forensic-dgp-thesis`,
zone `us-central1-a`, NVIDIA L4/g2-standard-4. No new VM was selected or started.

Cleanup is complete and independently audited: four redundant home archives and
76,973 exact inactive image-output copies from seven stopped/finished runs were
removed after full local backup checks. The net increase in guest free space
over maintenance is {recovered:.3f} GiB. Fresh free space is {post['free_GiB']:.3f} GiB,
compared with {initial:.3f} GiB at the initial inventory. This net measurement
includes storage consumed by new maintenance plans and receipts; it is not an
exact immediate-before/after deletion measurement. After the bank installation
reservation, the conservative free-space estimate is
{post['conservative_post_install_GiB']:.3f} GiB; its 16 GiB requirement passes.

The dedicated input/research-cache directories, all checkpoints, training
states, data/splits, sources, instructions, logs and gate/failure records remain
on the VM. 82 non-image NPY/NPZ gradient/optimizer records were expressly excluded
from offload and separately hashed. Full local return directories also remain.
The scientific cache check compares all retained metadata and critical/evidence
hashes; it does not rehash every cache byte again. No model, gradient or training
work ran on the VM as part of maintenance. Model quality and the full goal are
still unqualified; closed historical runs remain closed.

The original removal operation hit its 900-second stop while hashing retained
files after all 76,973 removals. Its failed transport receipt and traceback are
retained. A separate finite audit verified the complete original ledger, all
retained metadata and critical hashes, and exported evidence with zero additional
deletions. The independent local audit closed this evidence; the original
timed-stop operation is not relabeled as a successful execution.

## Portable output backup

Folder:
`C:\\xampp\\htdocs\\YEAR 4\\Testing\\outputs\\cctv_dgp_vm_output_offload_20261010_v1_r2`

Archive: `dgp-image-output-recovery-20261010-v1.tar`

Bytes: {export['bytes']:,}. Image payload bytes: {export['payload_bytes']:,}
({export['payload_bytes']/1024**3:.3f} GiB). The tar is deliberately uncompressed;
it includes exact original-path image/array packs, a recovery manifest and a
standalone standard-library verifier/restorer. Mixed RGB output packs retain
their measurement fields. The payload sources were already downloaded locally;
the new portable archive was created from those hash-verified copies and
independently read back. Every payload member and 127,334 full-return bindings
passed. The original return directories remain a second recovery source.

SHA256:
`{export['archive_sha256']}`

The `.sha256`, `recovery_export.json`, `independent_backup_audit.json`,
`independent_cleanup_audit.json` and `closure.json` sit beside the archive.
`tail_recovery_v1/remote_receipts` holds both research snapshots and the exact
deletion ledger. `transport/apply_stderr.log` retains the original timed stop.
The earlier local selector stop and inspected pack layouts are retained in
`outputs/cctv_dgp_vm_output_offload_20261010_v1`; they did not delete VM files.

## Verify the portable archive on Windows

```powershell
Get-FileHash -Algorithm SHA256 -LiteralPath "C:\\xampp\\htdocs\\YEAR 4\\Testing\\outputs\\cctv_dgp_vm_output_offload_20261010_v1_r2\\dgp-image-output-recovery-20261010-v1.tar"
```

The result must match the SHA256 above. Full member verification has already
passed independently; a matching archive hash binds that audited content.

## Restore after the destination VM is identified

Provide its project, instance name, zone, Linux user and GPU before transfer or
training. The output archive is not a boot-disk backup. Keep the separate code,
datasets, checkpoints/full states and environment/migration records too.
No guessed destination or automatic historical launcher is provided here.

The existing research-cache backup contains 4,431 files / 36.739 GiB. All files
remain present with the frozen sizes; its earlier full SHA256 audit is retained.
Its instructions are in
[VM_CACHE_BACKUP_RESTORE_20261007_V1.md](<C:/xampp/htdocs/YEAR 4/Testing/VM_CACHE_BACKUP_RESTORE_20261007_V1.md>).
The scientific-status paragraph in that older document is historical.

After checked transfer, verification on a VM with an existing research root is:

```bash
python3 -B restore_cctv_dgp_vm_output_offload_v1.py --archive dgp-image-output-recovery-20261010-v1.tar --expected-sha {export['archive_sha256']} --root ~/forensic-dgp
```

The verifier reads every member and checks existing outputs. Add `--restore`
to restore absent files only. Add `--group cctv_dgp_head4_capacity_vm_v1` to
restore one historical output group; repeat `--group` for several. Differing
existing files and symlink targets are refused. The helper enforces enough
space for the selected absent logical payload bytes plus 1 GiB. All-groups staging
requires approximately 25 GiB before upload for the archive, restored payload,
filesystem allocation and reserve.
This restores stored outputs; it does not rerun their models or resume training.
No restore on another VM has been performed or qualified by this cleanup.

The current finite bank comparison remains unrun and manual in tmux. Its
verified commands are in
[CCTV_DGP_BANK_COMPARISON_V1_VM.md](<C:/xampp/htdocs/YEAR 4/Testing/CCTV_DGP_BANK_COMPARISON_V1_VM.md>).
Its own live storage, CUDA, source/data and preservation checks remain required.
Current source-VM free space is a snapshot, not a future launch guarantee.
'''
    guide_path.write_text(guide, encoding='utf-8', newline='\n')
    prefix = f'''# Latest maintenance — 10 October 2026: verified local output backup and VM storage cleanup

The user requested VM space and a local backup before a later VM change. The
existing L4/g2-standard-4 instance was freshly identified; no VM was switched,
started, resized or stopped. New actual training remains manual in tmux.

Four hash-matched redundant home archives and 76,973 singly-linked inactive image
output copies from seven historical runs were removed after complete Windows
backup audits. Net increase in free space during maintenance: {recovered:.3f} GiB.
Fresh guest free space: {post['free_GiB']:.3f} GiB, from initial {initial:.3f} GiB.
Conservative free space after the ready bank-packet installation reservation:
{post['conservative_post_install_GiB']:.3f} GiB; its 16 GiB requirement is satisfied.
The current bank packet is not installed and has not run. This net space reading
includes storage used by new plans and receipts, not just deleted allocations.

All checkpoints/full states, dedicated research/training caches, inputs/splits,
source/instructions and gate/failure/log records remain on the VM. 82 non-image
gradient/optimizer files were explicitly retained and separately hashed.
The independent audit compared {audit['retained_research_metadata_entries']:,}
retained metadata entries and {audit['retained_critical_and_evidence_byte_hashes']:,}
critical/evidence byte hashes. Only direct output-parent directory times/size
changed besides the exact planned image files. Scientific cache bytes were not
all rehashed; unchanged metadata and disjoint singly-linked removal are recorded.
Local app/packet bindings and the permanent checkpoint SHA256
646fbb11674e8882ea41e251d6e6e7d468c963979b48c93132def8292834e24b are unchanged.

Portable backup: outputs/cctv_dgp_vm_output_offload_20261010_v1_r2/
dgp-image-output-recovery-20261010-v1.tar, {export['bytes']:,} bytes,
SHA256 {export['archive_sha256']}.
All 76,973 payload members and 127,334 full-return bindings passed independent
verification. Complete local returns remain too. The earlier 4,431-file / 36.739 GiB
cache backup remains present with its frozen sizes and prior full-hash audit.
This output archive is not a complete boot-disk/environment backup.

Evidence: outputs/cctv_dgp_bank_archive_cleanup_v1/independent_audit.json;
outputs/cctv_dgp_vm_output_offload_20261010_v1_r2/independent_backup_audit.json,
independent_cleanup_audit.json, tail_recovery_v1/remote_receipts and inventory_after.json.
Recovery: VM_IMAGE_OUTPUT_BACKUP_RESTORE_20261010_V1.md.
Manual next run: CCTV_DGP_BANK_COMPARISON_V1_VM.md, unchanged verified packet.
Use a freshly identified/verified destination before any later VM migration.

The original 900-second removal task stopped during retained-file hashing after
all deletions. Its failed receipt/traceback remain. A separate finite audit-only
task verified the complete deletion ledger and exact retained state with zero
additional removals. The independent local audit passed; the original execution
remains recorded as timed out. Immediate pre-removal free bytes were not saved,
so only ledger allocation and measured interval/net space changes are claimed.

The original local selector failure was retained before the corrected preparation;
the user chose to store complete verified image packs locally. V41 optimizer
step NPZ files were identified from their actual contents and stayed on the VM.
No historical pilot was relaunched, gate changed or model adopted. Maintenance
ran zero model/gradient/training calls. The full goal remains ACTIVE and
incomplete; useful restoration and all seven completion families still require
their agreed independent reviews. Earlier handoff sections remain historical.
Exact pre-maintenance handoff bytes are in this offload's documents directory.

---

'''
    handoff.write_bytes(prefix.encode('utf-8') + before)
    assert handoff.read_bytes().endswith(before)
    for path, expected in pinned.items():
        if path != 'PROJECT_HANDOFF.md':
            assert sha(ROOT / path) == expected, path
    closure = dict(complete=True, UTC=datetime.now(timezone.utc).isoformat(), checker_sha256=sha(Path(__file__)),
        archive_cleanup_audit_sha256=sha(ARCHIVES / 'independent_audit.json'),
        output_cleanup_audit_sha256=sha(OUT / 'independent_cleanup_audit.json'),
        output_backup_audit_sha256=sha(OUT / 'independent_backup_audit.json'),
        post_inventory_sha256=sha(OUT / 'inventory_after.json'),
        archive_copies_removed=4, image_output_copies_offloaded=76973,
        observed_total_recovery_GiB=recovered, free_space_measurement='Fresh final guest free bytes minus initial guest inventory; net of maintenance storage',
        original_apply_timeout_preserved=True, timed_stop_closed_by_separate_non_deleting_audit=True,
        initial_free_GiB=initial,
        final_live_free_GiB=post['free_GiB'], conservative_post_install_GiB=post['conservative_post_install_GiB'],
        manual_run_space_requirement_met=True, portable_backup_sha256=export['archive_sha256'],
        handoff_before_sha256=sha(destination / 'PROJECT_HANDOFF.before.md'), handoff_after_sha256=sha(handoff),
        previous_handoff_bytes_preserved_as_exact_suffix=True, recovery_guide_sha256=sha(guide_path),
        other_local_app_packet_source_bindings_unchanged=True, old_VM_not_migrated_or_stopped=True,
        model_gradient_or_training_calls=0, goal_complete=False, goal_status='active')
    write(OUT / 'closure.json', closure)
    print({key: closure[key] for key in ['complete', 'image_output_copies_offloaded',
           'observed_total_recovery_GiB', 'final_live_free_GiB', 'conservative_post_install_GiB']}, flush=True)


if __name__ == '__main__':
    main()
