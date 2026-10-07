# Optional VM cleanup candidates — 7 October 2026

A fresh read-only inventory finds 6.43 GiB free on forensic-dgp-thesis. The GPU is idle. This audit removes no files and starts no training. The five groups below could reclaim about 0.46 GiB of allocated space, giving approximately 6.90 GiB free. Recheck active work and exact file state before any removal.

The 89 Markdown files in the research tree outside installed package folders total 602,885 logical bytes (0.58 MiB). Three redundant archived copies have verified Windows backups and an identical retained current VM copy. They occupy only 24 KiB. Project instructions, provenance, dataset terms, research results and software notices remain useful records.

| Candidate | Estimated space | Basis for removal |
|---|---:|---|
| Two inactive V6/V7 `w600k_r50.onnx` copies | 332.6 MiB | Complete matching Windows copies; identical retained V27 VM recognizer; singly linked, owned files outside V30's required dependency set. |
| 8,219 Python `.pyc` files in `__pycache__` folders | 101.3 MiB | Corresponding `.py` sources exist; files are owned and singly linked. Python regenerates bytecode. Remove only the exact listed bytecode files; retain source and runtime folders. |
| 29 downloaded `.deb` files in `/var/cache/apt/archives` | 39.8 MiB | Package-manager download cache. Use the package manager's `clean` operation when no package transaction is running; installed packages remain. |
| Empty `~/.cache/pip` directory tree | 1.65 MiB | 422 owned directories contain zero files. Empty-directory removal can reclaim their metadata blocks. |
| Three archived duplicate Markdown files | 24 KiB | Full Windows copies and the current V27 copy have matching SHA256. Keep those recovery and current copies. |

The exact recognizer paths are:

1. `~/forensic-dgp/cctv_dgp_targets_vm_v6/weights/w600k_r50.onnx`
2. `~/forensic-dgp/cctv_dgp_capacity_vm_v7/weights/w600k_r50.onnx`

Both match SHA256 `4c06341c33c2ca1f86781dab0e829f88ad5b64be9fba56e56bc9ebdefc619e43`. The current V30 pilot uses the retained V27 recognizer. These old copies can be restored from their exact Windows backup locations before any deliberate historical reproduction.

The three optional Markdown copies share the filename `lineage/closed_v22_10_CCTV_DGP_DETAIL_PRIOR_V22_R1_RESULTS.md` inside:

1. `~/forensic-dgp/cctv_dgp_detail_skip_vm_v23/`
2. `~/forensic-dgp/cctv_dgp_degraded_detail_vm_v24/`
3. `~/forensic-dgp/cctv_dgp_spatial_features_vm_v25/`

The identical V27 lineage record and all three original Windows packet files remain available. Their SHA256 is `c5d201de35f9d05f23c803eca2dc84df28b0fe40e664a5592f5cdd146761a751`.

The bytecode candidates are under `~/forensic-dgp/` and `~/.local/`; installed-package sources and both virtual environments remain. The exact 8,219 file paths and retained source paths are in the audit JSON.

Keep original and trained DGP checkpoints, V30 data and dependencies, splits, failure receipts, completion assets, and the large scientific `.bin`/`.npz` caches. A cache name alone does not establish that a research tensor is disposable. Full verified backups are still required for those caches. V12/V12r2 recognizer files have three hard links each; removing the inspected names alone would reclaim no weight-file blocks, so they are excluded.

[Exact candidates and backup evidence](<C:/xampp/htdocs/YEAR 4/Testing/outputs/cctv_dgp_vm_optional_cleanup_candidates_20261007_v1/safe_candidates.json>) · [Read-only inventory](<C:/xampp/htdocs/YEAR 4/Testing/outputs/cctv_dgp_vm_optional_cleanup_candidates_20261007_v1/read_only_candidates.log>)
