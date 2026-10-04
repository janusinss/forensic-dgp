# Varied-covering VM result review — 3 October 2026

The frozen V2 pilot was executed as recorded in `VARIED_COVERING_VM.md`. This document and the
return auditor are separate local additions; they do not change that bundle.
Eleven return-audit counterexample tests pass. The actual return is now present
and independently audited: 1,653 archive members, 1,638 masks and 256 step records.
All ten return-preview rows, 36 practical masks and 20 downstream face estimates
have been inspected. Current outcomes are in `VARIED_COVERING_RESULTS.md` at
`C:\xampp\htdocs\YEAR 4\Testing\` (intended `~/forensic-dgp/` after future transfer).
The preparation-only receipt below remains historical; it is not a live VM status.

| Artifact | Windows local | Linux VM |
| --- | --- | --- |
| Frozen pilot runbook | `C:\xampp\htdocs\YEAR 4\Testing\VARIED_COVERING_VM.md` | `~/forensic-dgp/varied_covering_vm_bundle/VARIED_COVERING_VM.md` after bundle extraction |
| Return archive | `C:\xampp\htdocs\YEAR 4\Testing\outputs\varied-covering-results.tar.gz` | `~/forensic-dgp/varied_covering_vm_bundle/varied-covering-results.tar.gz` after successful completion |
| Checksum | `C:\xampp\htdocs\YEAR 4\Testing\outputs\varied-covering-results.tar.gz.sha256` | `~/forensic-dgp/varied_covering_vm_bundle/varied-covering-results.tar.gz.sha256` after successful completion |
| Local auditor | `C:\xampp\htdocs\YEAR 4\Testing\scripts\audit_varied_covering_results.py` | Intended `~/forensic-dgp/scripts/audit_varied_covering_results.py` only after future transfer; unnecessary for VM execution |
| Preparation evidence | `C:\xampp\htdocs\YEAR 4\Testing\outputs\varied_covering_return_audit_preparation_v1\verification.json` | Intended `~/forensic-dgp/outputs/varied_covering_return_audit_preparation_v1/verification.json` only after future transfer |

Do not repeat the completed V2 pilot or transfer. Preserve the executed workspace.
Completion wrote two unselected
`last.pth` files, not `best.pth`. Training remains VM-only.

The commands below are a reference for a distinct receiving workspace. The current
local return, extraction and audit already exist; the auditor refuses to overwrite
them. Use the configured Windows Google Cloud SDK Shell for a new destination.
One remote file per command avoids the observed PuTTY limitation.

1. Set the Windows destination:

   ```cmd
   cd /d "C:\xampp\htdocs\YEAR 4\Testing\outputs"
   ```

2. Download the archive:

   ```cmd
   gcloud compute scp --project=forensic-dgp-thesis "janusdominic0@forensic-dgp-thesis:/home/janusdominic0/forensic-dgp/varied_covering_vm_bundle/varied-covering-results.tar.gz" .
   ```

3. Download the checksum:

   ```cmd
   gcloud compute scp --project=forensic-dgp-thesis "janusdominic0@forensic-dgp-thesis:/home/janusdominic0/forensic-dgp/varied_covering_vm_bundle/varied-covering-results.tar.gz.sha256" .
   ```

4. Run the local audit from the same shell:

   ```cmd
   "C:\xampp\htdocs\YEAR 4\Testing\venv\Scripts\python.exe" "C:\xampp\htdocs\YEAR 4\Testing\scripts\audit_varied_covering_results.py"
   ```

5. Inspect all 10 preview rows and the per-case errors after the audit passes.

The archive hash must match the exact LF checksum and all inventory entries.
The auditor refuses missing/partial/unsafe returns and existing extraction/audit
directories before overwriting evidence. Its expected outputs, created only for
an actual passing return audit, are:

| Output | Windows local | Intended Linux counterpart after transfer |
| --- | --- | --- |
| Extraction | `C:\xampp\htdocs\YEAR 4\Testing\outputs\downloaded_varied_covering_v1\` | `~/forensic-dgp/outputs/downloaded_varied_covering_v1/` |
| Audit report | `C:\xampp\htdocs\YEAR 4\Testing\outputs\varied_covering_results_validation_v1\verification.json` | `~/forensic-dgp/outputs/varied_covering_results_validation_v1/verification.json` |
| Reconstructed preview | `C:\xampp\htdocs\YEAR 4\Testing\outputs\varied_covering_results_validation_v1\verified_preview.png` | `~/forensic-dgp/outputs/varied_covering_results_validation_v1/verified_preview.png` |

The audit recounts all 1,638 saved masks against the unchanged supervised support,
checks all 256 steps against the frozen sampling schedules, and inspects both
restricted-loaded checkpoint states. Each branch must log 128 fresh updates,
1,122 cumulative model updates and an unchanged reference head/BN state. All
V2 package/protocol/source/code/data bindings must remain intact.

Saved-mask counting and tensor inspection do not independently reproduce model
execution. Final optimizer tensors remain on the VM under
`~/forensic-dgp/varied_covering_vm_bundle/outputs/varied_covering_vm/<arm>/final_optimizer.pth`;
recorded hashes/steps remain claims until those tensors are separately inspected.
The local audit performs zero model forwards and constructs no optimizer.

These are exposed training-cohort detector measurements and approximate labels.
Even a complete audit and all five fit checks passing cannot qualify automatic
covering removal, generated hidden-face identity or population performance.
Review full covering footprints and clear controls before a separate evaluation
against the original 425-case gates and the fixed practical gallery. Automatic
and manually assisted face outputs remain separate evidence. Nearly hidden
automatic rejection remains an unresolved requirement.

The saved-footprint/cached-assisted follow-up is now complete. Next: run the
distinct matched RGB/grayscale exposure pilot from
`C:\xampp\htdocs\YEAR 4\Testing\GRAY_COVERING_VM.md`
(bundled `~/forensic-dgp/gray_covering_vm_bundle/GRAY_COVERING_VM.md`) after its
setup/preflight passes. Current return/practical reviews do not qualify an app
checkpoint; no local training or automatic promotion follows.
