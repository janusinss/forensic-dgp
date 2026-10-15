# Verified Drive offload — 13 October 2026

The first two selected cold-backup batches are uploaded, independently audited,
and removed from local storage. This is storage maintenance, not a model-quality
or complete-project migration result.

| Measurement | Verified result |
|---|---:|
| Drive files retained, including recovery metadata | 4,457 / 52,127,657,191 bytes |
| Exact local backup files removed | 4,432 / 52,028,568,639 bytes (52.03 GB / 48.45 GiB) |
| Observed C: free-space increase during removal | 52,027,830,272 bytes |
| C: free space at removal completion | 79,043,784,704 bytes (79.04 GB / 73.61 GiB) |
| Retained local files verified unchanged before publication | 665,797 |

The local removals comprise one historical output-recovery TAR and 4,431
local VM-cache recovery copies. The source VM was not contacted or changed.
No directory, cloud file, checkpoint, active model, split, provenance record,
failure evidence or current diagnostic asset was removed. Backup metadata and
restorers remain local. Other historical outputs await a separate dependency
review; they were not included in this first offload.

All 6,209 current/critical file hashes match. The complete project namespace
has zero unexpected missing files and zero retained metadata changes. The
original root documents are preserved before the publication changes below.

## Verified archive location

[Open the Drive archive](https://drive.google.com/drive/folders/15_PTa1yS09LlI2U5nmXccus-fraMzAKQ).

The rclone destination is:

```text
thesis_archive:FORENSIC_DGP_ARCHIVE/Testing/20261012_v1
```

The `image` and `cache` subfolders retain project-relative paths. The permanent
Desktop OAuth client and working profile remain
`%APPDATA%\rclone\forensic-dgp-drive-manual.conf`. Credentials and tokens are
outside the project and were not included in the archive.

The manual upload and streamed SHA256 checks completed for both batches.
Independent current Drive inventories before and after local removal agree
on every file ID, path, size and SHA256. A separate actual download of the
1,478,385-byte cache source inventory matches its original hash.

## Restore the removed copies to this Windows project

Restore only when the files are needed; it consumes the reclaimed space.
The commands below retrieve the removed payloads only. Existing local
metadata, instructions and failure evidence are not overwritten.

1. Set the exact Windows PowerShell variables:

   ```powershell
   $driveRestoreExe = "$env:LOCALAPPDATA\Programs\rclone\rclone.exe"
   $driveRestoreConfig = "$env:APPDATA\rclone\forensic-dgp-drive-manual.conf"
   $driveRestoreRoot = "C:\xampp\htdocs\YEAR 4\Testing"
   $driveRestoreLists = "$driveRestoreRoot\outputs\google_drive_offload_20261012_v1\closure_20261013_v1"
   ```

2. Retrieve the historical output TAR:

   ```powershell
   & $driveRestoreExe --config $driveRestoreConfig copy "thesis_archive:FORENSIC_DGP_ARCHIVE/Testing/20261012_v1/image" $driveRestoreRoot --files-from-raw "$driveRestoreLists\image_restore_files.txt" --checksum --immutable --transfers 2 --progress
   ```

3. Verify its restored bytes against the frozen SHA256:

   ```powershell
   & $driveRestoreExe check "$driveRestoreLists\image_restore_sha256.txt" $driveRestoreRoot --checkfile SHA256 --one-way
   ```

4. Retrieve the cache payloads:

   ```powershell
   & $driveRestoreExe --config $driveRestoreConfig copy "thesis_archive:FORENSIC_DGP_ARCHIVE/Testing/20261012_v1/cache" $driveRestoreRoot --files-from-raw "$driveRestoreLists\cache_restore_files.txt" --checksum --immutable --transfers 2 --progress
   ```

5. Verify every restored cache file:

   ```powershell
   & $driveRestoreExe check "$driveRestoreLists\cache_restore_sha256.txt" $driveRestoreRoot --checkfile SHA256 --one-way
   ```

A failed command must be reviewed before proceeding. Immutable copy refuses
different existing files. This guide's full payload restore has not been run;
only the actual metadata restore proof was performed during closure. The
TAR retains its embedded image-output manifest/restorer. Cache files retain
the original `home/janusdominic0/forensic-dgp/...` layout and the separately
saved standard-library source verifier.

Changing VM accounts requires preserving the same permanent OAuth application
and authorizing the same Drive storage account. The source-VM research assets,
checkpoint/full states, code, data/splits and Linux environment require their
own migration inventory. These two batches are not a complete VM image.
Windows `venv` is not a Linux environment. Closed historical pilots stay closed;
new actual training remains the user's manual L4/tmux workflow.

## Evidence and retained stops

The immutable upload plan SHA256 is
`2390ae75ca39b6a60383931b2d5fc658459a6051b4a640ef7cf608a1f38f716f`.
The exact removal plan SHA256 is
`cbd21863c6366c7a00cd4f6c89bcdb98d79ff7cffe76f7b0ef230967a7d94497`.

Evidence lives under
`outputs/google_drive_offload_20261012_v1/closure_20261013_v1`:

- `independent_before_audit.json` and `independent_after_audit.json` bind the
  completed manual operations, all cloud hashes, exact candidate namespace,
  active hashes and full before/after local inventories.
- `deletion_ledger.jsonl` records every exact removal with its exclusive-handle
  source hash and preserved Drive file ID. No recursive deletion was used.
- `cloud_inventory_before.json` and `cloud_inventory_after.json` retain every
  remote path/ID/size/hash. `metadata_restore_receipt.json` records the actual
  cache-inventory download proof.
- `documents_before/` preserves the complete pre-publication instructions and
  status documents. The first closure checker stopped on `SHA256` versus the
  observed lowercase `sha256` metadata key; its source and stop remain. The
  correction did not change data or scientific gates.
- `after_checker_review.json` preserves the review correction comparing
  removed-name counts with dictionary keys. The original checker is retained;
  the successful after audit has its own source fingerprint.

Logical removed bytes and observed free-space change are reported separately.
The ledger's allocation-query fields come from GetCompressedFileSizeW; they
are not a separate exact NTFS-sector allocation measurement.

The DGP-led thesis goal remains active. Storage offload does not establish
useful restoration, covering-family quality, identity recovery, native CCTV
generalization, or a completed migration. No inference, gradient query,
training, model adoption, billing change or VM maintenance was performed.
