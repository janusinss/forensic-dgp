# Verified local image-output backup and VM cleanup — 10 October 2026

The user requested additional VM space and a local backup for a later VM change.
The existing instance remains `forensic-dgp-thesis`, project `forensic-dgp-thesis`,
zone `us-central1-a`, NVIDIA L4/g2-standard-4. No new VM was selected or started.

Cleanup is complete and independently audited: four redundant home archives and
76,973 exact inactive image-output copies from seven stopped/finished runs were
removed after full local backup checks. The net increase in guest free space
over maintenance is 13.170 GiB. Fresh free space is 20.641 GiB,
compared with 7.471 GiB at the initial inventory. This net measurement
includes storage consumed by new maintenance plans and receipts; it is not an
exact immediate-before/after deletion measurement. After the bank installation
reservation, the conservative free-space estimate is
19.045 GiB; its 16 GiB requirement passes.

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
`C:\xampp\htdocs\YEAR 4\Testing\outputs\cctv_dgp_vm_output_offload_20261010_v1_r2`

Archive: `dgp-image-output-recovery-20261010-v1.tar`

Bytes: 12,579,983,360. Image payload bytes: 12,394,485,961
(11.543 GiB). The tar is deliberately uncompressed;
it includes exact original-path image/array packs, a recovery manifest and a
standalone standard-library verifier/restorer. Mixed RGB output packs retain
their measurement fields. The payload sources were already downloaded locally;
the new portable archive was created from those hash-verified copies and
independently read back. Every payload member and 127,334 full-return bindings
passed. The original return directories remain a second recovery source.

SHA256:
`5fe41326fa0ccd6399babf18f4f45582b0fa7d9815d506b998ca29fa637a4c1c`

The `.sha256`, `recovery_export.json`, `independent_backup_audit.json`,
`independent_cleanup_audit.json` and `closure.json` sit beside the archive.
`tail_recovery_v1/remote_receipts` holds both research snapshots and the exact
deletion ledger. `transport/apply_stderr.log` retains the original timed stop.
The earlier local selector stop and inspected pack layouts are retained in
`outputs/cctv_dgp_vm_output_offload_20261010_v1`; they did not delete VM files.

## Verify the portable archive on Windows

```powershell
Get-FileHash -Algorithm SHA256 -LiteralPath "C:\xampp\htdocs\YEAR 4\Testing\outputs\cctv_dgp_vm_output_offload_20261010_v1_r2\dgp-image-output-recovery-20261010-v1.tar"
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
python3 -B restore_cctv_dgp_vm_output_offload_v1.py --archive dgp-image-output-recovery-20261010-v1.tar --expected-sha 5fe41326fa0ccd6399babf18f4f45582b0fa7d9815d506b998ca29fa637a4c1c --root ~/forensic-dgp
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
