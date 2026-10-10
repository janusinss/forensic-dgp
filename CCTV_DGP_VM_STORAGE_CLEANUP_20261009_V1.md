# Verified VM storage cleanup — 9 October 2026

Authorized maintenance is complete on the existing forensic-dgp-thesis VM.
The exact plan removed 35 inactive top-level archive copies with retained,
SHA256-matched Windows backups. Observed free space increased by
16.298GiB during apply. The separate live audit reports 21.644GiB free.

| Measurement | GiB |
| --- | ---: |
| Free immediately before apply | 5.347 |
| Free at the independent live audit | 21.644 |
| Observed increase during apply | 16.298 |

Deletion was restricted to owned regular files in /home/janusdominic0, with one
link, exact inode/size/time identity and matching archive hashes. No recursive
delete occurred. Every Windows archive, checksum and export receipt stays local.
The original actual-step partial-return archive is retained at
outputs/cctv-dgp-actual-step-review-v1-results.tar.gz with hash
d380ce08cf858eebc334fcad3c6f5590e01d526084239bbf807bd4b465e8c604.

The independent VM auditor verifies all 35 deleted paths are absent,
382,930 protected research-file hashes and
3,569 retained scientific-tensor stamps.
It separately verifies all 1,095 explicitly pinned
current diagnostic bindings, including all 828 existing control/partial files.
Original checkpoints, instructions, source, splits, provenance, failure logs,
research caches and the shared VM Python runtime stay in place. All 14 local
DGP-primary app bindings remain exact. The current execution archive is retained.

No model inference, gradient, optimizer, training, diagnostic or VM restart was
launched by maintenance. The GPU remains idle at the live audit. The final-state
tail remains uninstalled at the initial snapshot and unrun by this workflow.
Its manual guide CCTV_DGP_ACTUAL_STEP_TAIL_V1_VM.md is unchanged; its 2GiB free-space
requirement is met. The original 3GiB encoded-output protocol stop remains a
scientific failure record, separate from VM free-space availability.

Exact plan, local backup verification, pre-apply recheck, remote receipts,
deletion ledger and independent live audit are retained under
outputs/cctv_dgp_vm_storage_cleanup_20261009_v1/. Storage completion does not
qualify restoration or completion quality; the full research goal remains active.
