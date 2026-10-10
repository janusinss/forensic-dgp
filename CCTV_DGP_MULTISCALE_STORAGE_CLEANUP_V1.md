# Verified L4 VM archive cleanup — 10 October 2026

Cleanup is complete and independently audited. It recovered **3.746 GiB**; the final guest snapshot has **11.181 GiB free**. Conservative upload/install projection is **10.776 GiB**, exceeding the next diagnostic's unchanged 7 GiB requirement.

Removed only these two obsolete, inactive, individually linked home copies:

- `/home/janusdominic0/cctv-dgp-head4-capacity-v1-execution.tar.gz`
- `/home/janusdominic0/cctv-dgp-head4-capacity-v1-results.tar.gz`

Complete matching backups remain in the Windows project `outputs/` directory. Both passed full SHA-256 and GZIP integrity checks before deletion; remote size, allocated blocks, owner, inode, mtime, nlink1 and SHA-256 matched the frozen plan. Active GPU, tmux, research processes and open archive readers were checked again before removal. No recursive removal was used.

Independent return checks verify all six receipt members, the exact deletion ledger, unchanged 468,473 research metadata entries and 4,111 critical/evidence hashes, 346 local evidence bindings and all251 current packet assets. Original/current checkpoints, optimizer/scheduler/RNG state, failed gates, raw/PNG outputs, datasets, splits, logs, provenance and scientific caches remain. Cache bytes were not all rehashed: preservation is established by disjoint nlink1 home-file deletion plus the full research metadata comparison and critical byte hashes.

Verified VM: project/instance `forensic-dgp-thesis`, zone `us-central1-a`, instance ID `4410777042005672095`, `g2-standard-4` with NVIDIA L4. Host key remains pinned and TLS validation remains enabled. The VM was already running and remains running. No model, gradient, diagnostic or training job was launched; there is no active GPU task or tmux session at the final snapshot. This is not a future launch availability guarantee.

Next action remains the user's manual upload/install/tmux launch in `CCTV_DGP_MULTISCALE_CALIBRATION_V1_VM.md`. The execution archive and protocol hashes are unchanged. Model usefulness, longer training, the app and all covering families remain unqualified/incomplete; cleanup is not a quality milestone.

Cleanup plan SHA-256: `a12f4db487e8925beaaef19adf311196c3d890c9c4807e68a914808404e81139`.

Final inventory UTC: `2026-10-10T09:00:57.574938+00:00`.

Evidence: `outputs/cctv_dgp_multiscale_archive_cleanup_v1/plan.json`, `remote_receipts/cleanup_receipt.json`, `independent_audit.json`, `inventory_after.json` and saved transport/runtime records.
