# V18: structure-preserving capacity pilot — manual VM commands

**Closed5 October2026 — do not upload/launch/import V18 again.** The user returned
the successful481,723,919-byte archive.600 updates, transfer, independent60.29s
CPU audit and all ten original-cell visual reviews are complete. V18 shows
degraded training fitting gain but fails four clear preservation checks; it is
not selected or adopted. The complete audit uses separate numeric-only
corrections and preserves both original local failures plus frozen sources,
protocol and checkpoints. No retraining is needed to resolve those failures.
[Result report](<C:/xampp/htdocs/YEAR 4/Testing/CCTV_DGP_STRUCTURE_V18_RESULTS.md>).

The commands below are the historical user-run execution/collection record.
The user-selected transfer-files/pasteable-commands preference remains in force
for future justified VM pilots. No future pilot is authorized by these commands.

**Storage maintenance completed5 October2026:** after the user finished upload
steps1/2/3, they explicitly authorized direct gcloud cleanup.74 byte-identical
remote archive copies and104 pip cache files were removed; their Windows archive
backups remain. VM storage now87% used with13GiB available, recovering9.0GiB.
All10,480 protected hashes/4,739 tensor records match, including the three V18
upload files. Scientific caches, models, splits and failures are retained. The
assistant performed no training launch. The user subsequently ran the pilot;
the uploads/launch/collection need no repetition.

Windows workspace: `C:\xampp\htdocs\YEAR 4\Testing`.
Existing VM: NVIDIA L4 `forensic-dgp-thesis`, project `forensic-dgp-thesis`,
zone `us-central1-a`, user `janusdominic0`.
New VM root: `/home/janusdominic0/forensic-dgp/cctv_dgp_structure_vm_v18`.
Detached tmux session: `dgp_structure_v18`.

The launcher installs the thin package into a fresh root, checks the existing
data/runtime and starts its supervisor **inside detached tmux**. Launch it from
normal VM SSH. An idle outer tmux shell is also permitted. Active GPU tasks,
active tmux programs or an existing V18 root/session cause rejection; preserve
them and the printed evidence. No dependency installation or automatic resume.

## Historical upload and launch — closed

1. Upload the archive from **Windows Google Cloud SDK Shell (CMD)**:

   ```bat
   cd /d "C:\xampp\htdocs\YEAR 4\Testing\outputs"
   gcloud compute scp --project=forensic-dgp-thesis --zone=us-central1-a "cctv-dgp-structure-v18-execution.tar.gz" "janusdominic0@forensic-dgp-thesis:/home/janusdominic0/"
   ```

2. Upload the checksum from the same **Windows Google Cloud SDK Shell**:

   ```bat
   gcloud compute scp --project=forensic-dgp-thesis --zone=us-central1-a "cctv-dgp-structure-v18-execution.tar.gz.sha256" "janusdominic0@forensic-dgp-thesis:/home/janusdominic0/"
   ```

3. Upload the launcher from the same **Windows Google Cloud SDK Shell**:

   ```bat
   gcloud compute scp --project=forensic-dgp-thesis --zone=us-central1-a "launch_cctv_dgp_structure_v18.py" "janusdominic0@forensic-dgp-thesis:/home/janusdominic0/"
   ```

4. Verify the transfer in **VM SSH**:

   ```bash
   cd ~ &&
   sha256sum -c cctv-dgp-structure-v18-execution.tar.gz.sha256
   ```

   Expected: `cctv-dgp-structure-v18-execution.tar.gz: OK`.

5. Launch in **VM SSH**; this command creates detached tmux:

   ```bash
   python3 ~/launch_cctv_dgp_structure_v18.py \
     --protocol-sha e3a7567e9a9c53a1668464ab5c95edbf094c73e9a7c8e550635dcf8d8ef04adb \
     --archive-sha ecc8a24aa77e23f80bbd7e482573ed666e443e35c7ee5633ddf4e5a6edeb2ce7
   ```

   Expected: preflight success followed by a launch receipt with
   `"session": "dgp_structure_v18"`. The gradient preflight runs later in the
   trainer, before any optimizer update. Keep the printed traceback if rejected.

## Watch progress

1. Watch the two logs in **VM SSH**:

   ```bash
   tail -F ~/forensic-dgp/cctv_dgp_structure_vm_v18/trainer.log ~/forensic-dgp/cctv_dgp_structure_vm_v18/supervisor.log
   ```

   **Ctrl+C closes this viewer only.** The detached tmux supervisor continues.
   Its declared overall budget is 1,800 seconds (30 minutes), including audit
   and export. Trainer completion precedes supervisor audit/export completion.

2. After closing the viewer, check successful export readiness in **VM SSH**:

   ```bash
   cat ~/forensic-dgp/cctv_dgp_structure_vm_v18/supervisor_completion.json
   ```

   `"complete": true`, `"archive_sha256"`, `"bytes"` and
   `"optimizer_updates": 600` identify the successful export receipt. This
   establishes reported execution/export completion; usefulness still requires
   independent returned-file audit and visual review. If this file is absent
   after a traceback, use the failure collection below.

Optional live session view in VM SSH:

```bash
tmux attach -t dgp_structure_v18
```

Detach with **Ctrl+B**, release both keys, press **D**. A completed supervisor
can leave no live session; the saved logs and receipts remain the evidence.

## Download a successful export

Use these only when `supervisor_completion.json` reports a complete export.

1. Download the results from **Windows Google Cloud SDK Shell (CMD)**:

   ```bat
   cd /d "C:\xampp\htdocs\YEAR 4\Testing\outputs"
   gcloud compute scp --project=forensic-dgp-thesis --zone=us-central1-a "janusdominic0@forensic-dgp-thesis:/home/janusdominic0/forensic-dgp/cctv_dgp_structure_vm_v18/cctv-dgp-structure-v18-results.tar.gz" .
   ```

2. Download the checksum from the same **Windows Google Cloud SDK Shell**:

   ```bat
   gcloud compute scp --project=forensic-dgp-thesis --zone=us-central1-a "janusdominic0@forensic-dgp-thesis:/home/janusdominic0/forensic-dgp/cctv_dgp_structure_vm_v18/cctv-dgp-structure-v18-results.tar.gz.sha256" .
   ```

3. Download the receipt from the same **Windows Google Cloud SDK Shell**:

   ```bat
   gcloud compute scp --project=forensic-dgp-thesis --zone=us-central1-a "janusdominic0@forensic-dgp-thesis:/home/janusdominic0/forensic-dgp/cctv_dgp_structure_vm_v18/supervisor_completion.json" "supervisor_completion_v18.json"
   ```

The local receipt alias preserves earlier exports. Archive and sidecar names
stay exact for import. All three files land in
`C:\xampp\htdocs\YEAR 4\Testing\outputs`.

## Download a failure export

Do not restart training or edit the frozen bundle after a failed gate.

1. Check failure export readiness in **VM SSH**:

   ```bash
   cat ~/forensic-dgp/cctv_dgp_structure_vm_v18/failure_export.json
   ```

   `"complete": true` means the **failure export** finished. Training, output
   quality and the audit can still have failed. Preserve that distinction.

2. Download the failure archive from **Windows Google Cloud SDK Shell (CMD)**:

   ```bat
   cd /d "C:\xampp\htdocs\YEAR 4\Testing\outputs"
   gcloud compute scp --project=forensic-dgp-thesis --zone=us-central1-a "janusdominic0@forensic-dgp-thesis:/home/janusdominic0/forensic-dgp/cctv_dgp_structure_vm_v18/cctv-dgp-structure-v18-failure.tar.gz" .
   ```

3. Download its checksum from the same **Windows Google Cloud SDK Shell**:

   ```bat
   gcloud compute scp --project=forensic-dgp-thesis --zone=us-central1-a "janusdominic0@forensic-dgp-thesis:/home/janusdominic0/forensic-dgp/cctv_dgp_structure_vm_v18/cctv-dgp-structure-v18-failure.tar.gz.sha256" .
   ```

4. Download its receipt from the same **Windows Google Cloud SDK Shell**:

   ```bat
   gcloud compute scp --project=forensic-dgp-thesis --zone=us-central1-a "janusdominic0@forensic-dgp-thesis:/home/janusdominic0/forensic-dgp/cctv_dgp_structure_vm_v18/failure_export.json" "failure_export_v18.json"
   ```

If the bootstrap failed before trainer launch, preserve its printed evidence
and check `~/forensic-dgp/cctv_dgp_structure_vm_v18/bootstrap_preflight.json`.
If export itself failed, preserve `failure_export_incomplete.json` and partial
files. Neither condition permits an unchanged recipe rerun.

## Historical local import — complete return retained

The original importer already verified transfer/source/protocol and safely
extracted this return. Its original exact-arithmetic failure is preserved in
`outputs/cctv_dgp_structure_return_v18/local_audit.log`. Do not rerun it into
another directory or overwrite the existing return. The successful independent
audit is in `outputs/cctv_dgp_structure_v18_audit_recovery_r2/`, with250 CPU
replays and zero training/cloud operations. The following commands record the
original import behavior; future returns need their own fresh paths/protocols.

For a successful return, from PowerShell in
`C:\xampp\htdocs\YEAR 4\Testing`:

```powershell
.\venv\Scripts\python.exe -B -X utf8 -u scripts\import_cctv_dgp_structure_v18.py --archive outputs\cctv-dgp-structure-v18-results.tar.gz --completion outputs\supervisor_completion_v18.json --extract-to outputs\cctv_dgp_structure_return_v18
```

For a failure return:

```powershell
.\venv\Scripts\python.exe -B -X utf8 -u scripts\import_cctv_dgp_structure_v18.py --archive outputs\cctv-dgp-structure-v18-failure.tar.gz --completion outputs\failure_export_v18.json --extract-to outputs\cctv_dgp_structure_failure_return_v18
```

Both destinations must be fresh. A full result audit has a 300-second internal
limit and a 330-second importer subprocess timeout. A partial failure receives
only a trace/source/transfer audit; it is not a verified learned restoration.
Original failure receipts stay preserved even when a separate audit succeeds.

## Frozen transfer evidence

Archive: **9,063,630 bytes**, 23 regular unique members, 21 source/assets.
Protocol SHA256:
`e3a7567e9a9c53a1668464ab5c95edbf094c73e9a7c8e550635dcf8d8ef04adb`.
Archive SHA256:
`ecc8a24aa77e23f80bbd7e482573ed666e443e35c7ee5633ddf4e5a6edeb2ce7`.
Bootstrap SHA256:
`a0aa9c6ef4e6a36783ec22bc97d981c5789550f3027a7e130d6964cd47c46e2c`.

Preparation receipt:
[cctv_dgp_structure_v18_preparation.json](<C:/xampp/htdocs/YEAR 4/Testing/outputs/cctv_dgp_structure_v18_preparation.json>).
Separate transfer audit:
[cctv_dgp_structure_v18_transfer_audit.json](<C:/xampp/htdocs/YEAR 4/Testing/outputs/cctv_dgp_structure_v18_transfer_audit.json>).
Rationale, exact finite design and evidence limits:
[CCTV_DGP_STRUCTURE_V18_PLAN.md](<C:/xampp/htdocs/YEAR 4/Testing/CCTV_DGP_STRUCTURE_V18_PLAN.md>).
