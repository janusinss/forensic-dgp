# Local research-cache backup — 7 October 2026

Destination: `C:\xampp\htdocs\YEAR 4\Testing\outputs\cctv_dgp_local_research_cache_backup_20261007_v1`.

Status: **complete**. Both `complete.json` and
`independent_local_cache_audit.json` confirm all **4,431 actual cache files /
39,448,585,279 bytes**, with every file SHA256 matching the frozen source.
The 147 redundant transfer blocks were removed after that audit; original
VM files, failed-transfer evidence and the earlier partial remain.

The actual files are materialized under `cache_files\home\janusdominic0\forensic-dgp`.
This preserves the original Linux path layout for a later restore audit.

| Original directory under `~/forensic-dgp` | Files | Bytes |
| --- | ---: | ---: |
| `expanded_feature_bundle/outputs/expanded_feature_training/anatomical_cache` | 3 | 15,083,101,710 |
| `expanded_feature_bundle/outputs/expanded_feature_training/fixed_cache` | 3 | 15,082,258,369 |
| `cctv_dgp_broader_codes_vm_v16_r2/outputs/broader_codes_v16_r2/cache` | 4,425 | 9,283,225,200 |

The combined data size is **36.74 GiB**. Cache bytes and cache metadata are
included; this is not a boot-disk image or a replacement for separately retained
code, datasets, model checkpoints, environments and failed-pilot evidence.
The `recovery_metadata` folder includes the frozen source inventory, read-only
Python verifier and complete project instructions from before this maintenance.

The source inventory SHA256 is
`af7a583f06cbc9a245e9c93815231be53d53c76d4a529578b4551c8bc318e291`.
All 4,431 original VM files were freshly checked against their frozen research
hashes before transfer. New 256 MiB transfer blocks have matching VM/Windows
SHA256 receipts. The earlier partial download remains unchanged; only its
verified full blocks are reused in the new backup. Every materialized file is
checked against the original inventory, then a separate verifier reads all
4,431 files again without neural inference or training.

## Corrected backup destination

The user requested a Windows backup in preparation for a possible future Google
Cloud VM. The agent initially selected a billed cloud snapshot instead. That
snapshot, `forensic-dgp-migration-backup-20261007-v1`, was deleted; a subsequent
snapshot listing confirms it is no longer active. The original VM and disk
remain. The rollback receipt is in `snapshot_rollback` inside the local backup.

The earlier `cctv-dgp-migration-recovery-kit-20261007-v1.zip` contains metadata
only and refers to the deleted snapshot. Preserve it as historical evidence;
use this local cache backup for the actual data. Deletion does not erase charges
already incurred. The running source VM and its disk retain their normal charges.
Official storage pricing: <https://cloud.google.com/compute/disks-image-pricing>.

## Verify the Windows backup

Run from **Windows PowerShell** after completion. This writes a new receipt and
stops within 1,800 seconds. Use a new receipt name when verifying again.

```powershell
cd "C:\xampp\htdocs\YEAR 4\Testing"
```

```powershell
& .\venv\Scripts\python.exe -B .\outputs\cctv_dgp_local_research_cache_backup_20261007_v1\recovery_metadata\verify_cctv_dgp_restored_caches_20261007_v1.py --manifest .\outputs\cctv_dgp_local_research_cache_backup_20261007_v1\cache_source_inventory.json --manifest-sha af7a583f06cbc9a245e9c93815231be53d53c76d4a529578b4551c8bc318e291 --root .\outputs\cctv_dgp_local_research_cache_backup_20261007_v1\cache_files --receipt .\outputs\cctv_dgp_local_research_cache_backup_20261007_v1\manual_recheck_01.json --max-seconds 1800
```

## Prepare a later transfer

No new VM is created or selected by this backup. Set the target project, zone,
instance and SSH user when a future VM is chosen. Do not change the authorized
training destination merely because a cache copy exists.

1. On Windows, verify the local backup with the command above.
2. Create a portable archive locally if one is needed. This requires about
   **37 GiB additional Windows space**; it does not consume extra VM disk space.

   ```powershell
   tar -cf "C:\xampp\htdocs\YEAR 4\Testing\outputs\cctv-dgp-research-caches-20261007-v1.tar" -C "C:\xampp\htdocs\YEAR 4\Testing\outputs\cctv_dgp_local_research_cache_backup_20261007_v1\cache_files" home
   ```

3. In Windows Google Cloud SDK Shell, upload the single archive with
   `gcloud compute scp --project=TARGET_PROJECT --zone=TARGET_ZONE`, targeting a
   fresh staging directory on `TARGET_USER@TARGET_VM`. Upload the manifest and
   verifier in separate calls. Exact target commands require the chosen VM;
   the source instance is not used as a placeholder destination.
4. Extract the checked archive into a fresh staging root on the target VM.
   Before extraction, its SHA256 and member paths must match the recorded
   transfer. Allow **at least 79 GiB free** for the archive, extracted files and
   a 5 GiB margin. Never extract over an existing research directory.
5. Run the verifier with `--root` set to the staging root. It expects
   `home/janusdominic0/forensic-dgp/...` beneath that root and uses Python 3.10
   or newer with only the standard library. All 4,431 files must pass.

A later checkout mapping must retain the original cache metadata and version
bindings. A different Linux username or changed absolute paths needs a
documented relocation audit. Keep the source VM data until restoration is
verified. Closed historical pilots remain closed.

## Research state

This maintenance makes no restoration-quality claim. V29 remains rejected for
application promotion; V30 is a distinct finite pilot prepared for manual
execution on the existing NVIDIA L4 VM. No optimizer, training launcher,
application update, source-cache deletion, VM stop or migration is performed
as part of the local backup. The DGP-led restoration and covering-family goal
remains active and incomplete.
