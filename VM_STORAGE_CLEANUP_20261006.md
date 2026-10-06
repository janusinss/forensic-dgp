# VM storage cleanup - 6 October 2026

Completed on the existing `forensic-dgp-thesis` VM in `us-central1-a`.
Seven duplicate transfer/result archives were removed after SHA256 verification
against retained Windows backups: 2,341,409,251 bytes (2.18 GiB).
The final live check found **9,880,526,848 bytes free (9.20 GiB)** on the
103,865,303,040-byte filesystem. The VM remains running; its NVIDIA L4 is idle.

The user's decision, "pick what doesnt disrupt the future vm usage," closes the
cleanup at these backed-up archives. Scientific caches and the installed runtime
are retained for future work. Storage maintenance does not authorize training.

## Verified removal and preservation

The exact seven paths, sizes, hashes and Windows recovery copies are recorded in
`outputs/cctv_dgp_vm_storage_cleanup_20261006_v1/archive-duplicates_plan.json`
and `remote_receipts/archive-duplicates/deletions.jsonl` beneath that directory.
No recursive deletion was used. Pip-cache removals: 0. Scientific-cache removals: 0.

A separate live read-only auditor verifies all seven paths absent, all 2,630
protected research-file hashes unchanged, 4,817 retained scientific tensor stamps
unchanged and 1,139 explicit V22/V23 asset/result/archive bindings unchanged.
A separate Windows audit rehashes all seven recovery archives and checks the
downloaded receipts against the independent VM audit.

The apply receipt measures 2,331,934,720 additional free bytes. This differs
slightly from the archive-byte total because live filesystem usage also changes
during maintenance. Final free space comes from the later runtime/cache check.

## Future VM use preserved

All three historical scientific caches remain: expanded anatomical, expanded
fixed and V16 r2. Their 4,431 files total 39,448,585,279 bytes (36.74 GiB), including
metadata. The final check verifies exact file sets, retained tensor stamps and
metadata hashes against the protected snapshot. Original datasets, checkpoints,
splits, sources, logs, current V22/V23 packets and failed gates stay preserved.

The existing `~/forensic-dgp/cctv_dgp_vm_bundle/.venv/bin/python` successfully
imports Python 3.10.12, PyTorch 2.9.1+cu129, NumPy 1.26.4 and OpenCV 4.11.0.
CUDA is available; the GPU is NVIDIA L4. The existing `dgp_detail_skip_v23` tmux
pane contains an idle Bash shell. No model was loaded, no training was launched,
no remote task was killed, and the VM was neither started nor stopped here.

The unfinished Windows cache transfer was stopped after the user's decision.
Its 3,205,955,584-byte `features.bin` remains under
`outputs/cctv_dgp_vm_storage_cleanup_20261006_v1/cache_backups/group0/anatomical_cache/`.
It is a partial, unverified copy; it cannot authorize deleting the full VM cache.
Only the verified assistant-created local backup process tree was stopped.
The unified execution session ended with exit 1 from cancellation; the final
process check finds no matching backup process. No cache-removal plan was applied.

## Evidence and boundaries

Current evidence: `outputs/cctv_dgp_vm_storage_cleanup_20261006_v1/`.

1. `remote_receipts/archive-duplicates/`: verification, protected snapshot,
   cleanup receipt and exact deletion ledger.
2. `audit-archives_file_r1.log` and
   `archive-duplicates_Windows_independent_audit.json`: independent VM and
   retained Windows backup checks.
3. `final_runtime_after_cleanup.json`: final disk, cache, runtime, GPU and tmux
   inspection; no additional deletion.
4. `inactive_cache_manifest.json` and the two `partial_backup_stop*` receipts:
   full cache fingerprints and local cancellation scope.
5. `before_docs/`, `before_docs_r1/` and `closure_manifest.json`: original document bytes and
   completed maintenance evidence bindings.

Applied archive plan SHA256:
`0a4522835d221b2e7928966a9cf098c263e1849912730aa231d348b06b1dfb70`.
Applied backend SHA256:
`d5993d264ffdded86cc9c7d2ff1864cb4fd6c93d5d970b49131173cd48d13a8d`.
Verification/apply checks took 119.11/62.25 seconds; the independent live audit
took 43.79 seconds; the final runtime/cache inspection took 8.29 seconds.
Historical failed transport attempts and partial backups remain as evidence.

The earlier `20261005_r2` proposal stopped before upload/removal; its original
documents, plan and failures are preserved. This completed `20261006_v1` entry
supersedes the earlier stopped-VM and zero-removal notices.

The V23 return archive is now present on Windows and matches 76,481,832 bytes and
SHA256 `b71639cabd17476d5989dfd4582d47de9b747b8bf2a611232e5f62cc67bf9fd9`.
Its checksum and export receipt are present. A concurrent research milestone
records the independent V23 audit and prepares a distinct V24 experiment; its
entire handoff entry is preserved. This storage work performs no additional
training, gradient, replay or image-quality evaluation. The V23 structural stop
remains a failure; no unchanged rerun or checkpoint promotion follows from
cleanup. Actual training remains manual on the existing L4. The thesis
restoration/completion goal remains active.
