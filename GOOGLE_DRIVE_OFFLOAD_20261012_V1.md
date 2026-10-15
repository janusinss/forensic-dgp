# Completed offload — 13 October 2026

Both manual uploads and streamed SHA256 verifications passed. Independent
Drive and local audits then removed exactly 4,432 backed-up local copies /
52,028,568,639 bytes. C: had 79.04 GB free at removal completion.
All 665,797 retained files and 6,209 protected hashes agree; no unexpected loss.
All 4,457 cloud files retain their IDs/sizes/hashes. Metadata and failure records
stay local. Other historical output folders were not in this first batch.

[Completed audit and exact restore commands](<C:/xampp/htdocs/YEAR 4/Testing/GOOGLE_DRIVE_OFFLOAD_COMPLETE_20261013.md>).
The original preparation guide is preserved under
`outputs/google_drive_offload_20261012_v1/closure_20261013_v1/documents_before/`.
The upload steps below are completed historical instructions; local payloads
are now in Drive and must be restored before repeating source-based checks.

# Manual cold-backup offload — 12 October 2026

Status: the user's manual My Drive connection is independently verified.
Total 5,497,558,138,880 bytes; free 5,477,794,986,957 bytes at the connection
check. The working profile is `%APPDATA%\rclone\forensic-dgp-drive-manual.conf`,
remote `thesis_archive`, own Desktop client, limited `drive.file` scope.
Credentials and OAuth tokens stay outside the project and the upload batches.

**Prepared only: no research files uploaded, moved or removed.**
The earlier agent profile `forensic-dgp-drive.conf` is not this working profile.
This guide uses the new manual profile throughout.

## Keep local

App/source/design, the Python environment and model checkpoints remain at their
existing paths. The deployed DGP, detector, completion and declared restoration
baseline are explicitly protected. The current signed VM diagnostic and its
parent directories remain local, alongside frozen data/splits/final identities,
current audits, research instructions, backup metadata and failure records.
Other output directories remain local pending a separate dependency review.
This first batch does not classify every historical output as inactive.

The preparation checked 5,658 current local file bindings. It also retains
208 required VM-parent bindings not duplicated at the corresponding local
parent path. Those missing local copies are not offload candidates. No VM
preflight or migration-completeness claim is made; the original VM-side
preflight remains necessary before any manual diagnostic.

## Two selected cold-backup batches

| Batch | Upload | Local-removal candidates after a separate cloud audit |
|---|---:|---:|
| Historical portable image-output recovery | 7 files / 12,674,051,375 bytes | One TAR / 12,579,983,360 bytes |
| VM research-cache recovery copies | 4,450 files / 39,453,605,816 bytes | 4,431 cache files / 39,448,585,279 bytes |

Potential local removal: **52,028,568,639 bytes / 48.45 GiB**. This is logical
file size, not a measured increase in free space. Metadata, verifiers and
failure records stay local. No new large archive or full staging copy is made.
The cache directory is copied with its original `home/janusdominic0/...` layout.

The existing portable TAR matches SHA256
`5fe41326fa0ccd6399babf18f4f45582b0fa7d9815d506b998ca29fa637a4c1c`.
A separate fresh standard-library verifier read every one of the 4,431 cache
files in 92.96 seconds and matched the frozen source inventory:
`af7a583f06cbc9a245e9c93815231be53d53c76d4a529578b4551c8bc318e291`.

The exact selected paths, hashes and active protections are in
`outputs/google_drive_offload_20261012_v1/plan.json`. Plan SHA256:
`2390ae75ca39b6a60383931b2d5fc658459a6051b4a640ef7cf608a1f38f716f`.

## Five exact steps — Windows PowerShell

Keep this PowerShell window open. These operations use your PC and Drive;
they do not contact the VM or run a model.

1. Set the command variables:

   ```powershell
   $driveOffload = "C:\xampp\htdocs\YEAR 4\Testing\scripts\run_google_drive_offload_20261012_v1.ps1"
   $drivePlanSHA = "2390ae75ca39b6a60383931b2d5fc658459a6051b4a640ef7cf608a1f38f716f"
   ```

2. Upload the historical output-backup batch:

   ```powershell
   powershell -NoProfile -ExecutionPolicy Bypass -File $driveOffload -Mode Upload -Batch image -ExpectedPlanSHA $drivePlanSHA
   ```

3. Verify that batch from downloaded bytes:

   ```powershell
   powershell -NoProfile -ExecutionPolicy Bypass -File $driveOffload -Mode Verify -Batch image -ExpectedPlanSHA $drivePlanSHA
   ```

4. Upload the cache-backup batch:

   ```powershell
   powershell -NoProfile -ExecutionPolicy Bypass -File $driveOffload -Mode Upload -Batch cache -ExpectedPlanSHA $drivePlanSHA
   ```

5. Verify the cache batch from downloaded bytes:

   ```powershell
   powershell -NoProfile -ExecutionPolicy Bypass -File $driveOffload -Mode Verify -Batch cache -ExpectedPlanSHA $drivePlanSHA
   ```

Wait for each command to return before running the next. Verification streams
Drive bytes against the frozen SHA256 list, without saving a second payload
copy on C:. A successful verification receipt reports `Complete: true` and
`DownloadedSHA256Verification: true`.

At a sustained 10 MB/s each way, payload transfer alone takes about 42 minutes
for the image upload/verification pair and 132 minutes for the cache pair.
Source checks and 4,450 small-file/metadata API operations add time. The displayed
transfer rate determines actual runtime. Each cloud command has an 8-hour hard stop; a stop retains local
originals and its operation log. Successful existing files can be skipped on
a repeat copy; a failed unfinished large file can require a fresh upload.

Upload uses `copy --immutable`; it does not mirror/delete other Drive files.
The new root is
`FORENSIC_DGP_ARCHIVE/Testing/20261012_v1`, with `image` and `cache` subfolders.
Logs and unique receipts are saved under
`outputs/google_drive_offload_20261012_v1/upload_*` and `verify_*`.

Return the final receipt from each verification for an independent cloud,
namespace and active-file audit. These commands contain no deletion operation.
Local removal is a separate exact-file, hash-bound step after that audit.
Upload completion alone does not reclaim local storage.

## Recovery and VM changes

Each batch retains project-relative paths beneath its Drive folder. A later
checked download restores the chosen absent paths into a fresh staging root.
The image TAR includes its own full recovery manifest and standard-library
restorer; the cache batch includes its frozen manifest and verifier. Hashes
must be checked before use and differing existing files must not be overwritten.

Keep the same permanent OAuth project/application and Desktop client when
changing VM accounts. Authorize the same Drive account on the new VM using
[rclone's headless setup](https://rclone.org/remote_setup/); keep the connection
profile separate from research archives. Windows `venv` is not a Linux
environment, and these two cold-backup batches are not a complete VM image.
VM code, current training assets/checkpoints/full states, splits and environment
records need their own migration inventory. No historical pilot is relaunched.

Testing-mode authorization expires after seven days; reconnect the same manual
profile when needed. The DGP restoration/covering goal remains active and all
scientific quality requirements stay unchanged.

References: [rclone copy](https://rclone.org/commands/rclone_copy/),
[downloaded checksum verification](https://rclone.org/commands/rclone_check/),
[Drive scope behavior](https://rclone.org/drive/#scopes).
