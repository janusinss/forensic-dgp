# Verified local output offload — 15 October 2026

The storage step is complete locally: exactly **76,973 historical output copies /
12,394,485,961 logical bytes (12.39 GB / 11.54 GiB)** were removed after verified
Drive recovery and an independent full current-byte audit. The original
CodeFormer checkpoint and all original DGP/model states remain local.

The existing 12.58GB Drive recovery TAR already contains these payloads, so no
second large upload was needed. Its ID/size/SHA256 still match the original full
streamed and all-member verification. A new actual range download restores one
archived output byte-for-byte. The new 20261015 recovery index and instructions
were uploaded and fully checked by downloading every byte before local removal.
Final closure transfer is recorded separately in the evidence folder.

| Measurement | Verified result |
|---|---:|
| Exact local output files removed | 76,973 |
| Logical file bytes removed | 12,394,485,961 |
| Observed C: free-space increase during removal | 12,508,647,424 bytes |
| C: free at removal completion | 40,171,778,048 bytes (40.17 GB / 37.41 GiB) |
| Retained files with unchanged full metadata | 606,152 |
| Protected file hashes unchanged | 12,230 |
| Retained non-image scientific arrays checked | 82 |
| Unexpected missing/added retained files | 0 / 0 |

Exact logical bytes and observed free-space change are different measurements.
C: free space is a time-specific snapshot and includes other system activity.
The retained namespace/hash audit completed before the documented publication
changes to this report and PROJECT_HANDOFF.md. The old handoff bytes remain
backed up exactly and as an unchanged suffix of the published handoff.

## Recovery location

Payload TAR:

```text
thesis_archive:FORENSIC_DGP_ARCHIVE/Testing/20261012_v1/image/outputs/cctv_dgp_vm_output_offload_20261010_v1_r2/dgp-image-output-recovery-20261010-v1.tar
```

Drive object ID: `1dAb7nZCps6K1dKIcX_IbXUUNrixK0MJ0`.
SHA256: `5fe41326fa0ccd6399babf18f4f45582b0fa7d9815d506b998ca29fa637a4c1c`.
[Open the retained payload](https://drive.google.com/file/d/1dAb7nZCps6K1dKIcX_IbXUUNrixK0MJ0/view).

New recovery index and closure:

```text
thesis_archive:FORENSIC_DGP_ARCHIVE/Testing/20261015_v1
```

[Exact restore commands](outputs/google_drive_offload_20261015_v1/RESTORE.md).
The local and uploaded plan maps original Windows paths to TAR members and
hashes. The helper restores only selected absent outputs and refuses to overwrite
different files. Seven historical return folders now require restoration before
replaying their old image/RAW checks. Their metadata, gates, logs, failed states,
checkpoints and protected scientific arrays remain local.

## Scope and evidence

The selected files are 70,400 PNGs, 1,900 RGB RAW arrays and 4,673 lossless RGB
output packs from seven completed experiments. Whole output packs, including
their associated measurement fields, are preserved in the TAR. This does not
remove training/research caches, original datasets/splits, provenance, model
checkpoints, failure records, the Python environment or current comparison files.
No directory was recursively deleted. Every exact deletion used an exclusive
handle with fresh hash/identity checks and a durable intent/removal ledger.
The complete backup is a recovery archive for this set, not a full VM migration.

- [Frozen exact plan](outputs/google_drive_offload_20261015_v1/plan.json), SHA256 `ddb180dd1a5154d6b00b595f247f253649c085f889c694b3525970c543907248`.
- [Independent before audit](outputs/google_drive_offload_20261015_v1/reviewer/independent_before_audit.json) and [independent after audit](outputs/google_drive_offload_20261015_v1/reviewer/independent_after_audit.json).
- [Removal receipt](outputs/google_drive_offload_20261015_v1/apply_receipt.json), [actual restore proof](outputs/google_drive_offload_20261015_v1/actual_restore_proof.json) and [restorer safety fixture](outputs/google_drive_offload_20261015_v1/restore_fixture_receipt.json).
- [Downloaded index verification](outputs/google_drive_offload_20261015_v1/metadata_transfer_receipt.json), [final closure transfer receipt](outputs/google_drive_offload_20261015_v1/closure_transfer_receipt.json) and [publication record](outputs/google_drive_offload_20261015_v1/publication.json).

The previous 52,028,568,639-byte / 52.03GB offload on 13 October remains a
separate completed action; it is not counted again in these new totals. Both
research and recovery records are preserved.

## Active thesis goal

The /goal is active: improve a separate CodeFormer-based restoration model by
our own fine-tuning, keeping untouched CodeFormer as benchmark, with accurate
attribution, visible-appearance preservation, separate native/paired evidence
and manual finite L4/tmux training. Further completion/removal-mask development
follows restoration acceptance. Current design:
[CCTV_CODEFORMER_FINETUNING_PLAN_V1.md](CCTV_CODEFORMER_FINETUNING_PLAN_V1.md).

No VM, inference, gradient, optimizer or training operation occurred in this
storage task. No app default changed. The earlier short VM diagnostics are
historical and are not reopened. Next is training-target quality and supported
training-path preparation; no new fine-tuning packet or trained CodeFormer
candidate is ready to launch. Storage completion is not model/thesis completion.
