# V19 r2 — corrected inference commands

**Closed 5 October 2026: successful execution/return audit and development review
are complete. Normalization parity passes; automatic/spatial preservation fails.**
Preserve this immutable package and returned evidence. Do not repeat the historical
upload/launch commands below. Download commands document the completed collection.
[Results and next-design evidence](<C:/xampp/htdocs/YEAR 4/Testing/CCTV_DGP_INPUT_SELECTION_V19_R2_RESULTS.md>).

The following records the original preparation and manual commands.

The returned diagnostic confirms the normalization cause. This separate package
matches the V15 comparison encoding while preserving the V18 spatial input path.
All original quality/parity gates remain. Eight regressions, the independent
transfer audit and actual frozen-package source/data preflight pass. R2 inference
has not been launched.

Upload the three new files below. The original V19 root/failure and diagnostic
remain intact. Do not use the earlier V19 or diagnostic launch commands again.
Detailed scope: [CCTV_DGP_INPUT_SELECTION_V19_R2_PLAN.md](<C:/xampp/htdocs/YEAR 4/Testing/CCTV_DGP_INPUT_SELECTION_V19_R2_PLAN.md>).

**V19 r2 runs inference only. No training or new checkpoint fitting.** It tests the
already trained V18 terminal decoder and frozen input-only restoration selector
on 104 development identities/520 synthetic cases, after 50 fresh training parity
checks. Prepared and independently transfer-audited on 5 October 2026; no VM launch
is claimed. Original V19 and the small diagnostic are closed; preserve both.

Existing NVIDIA L4 VM `forensic-dgp-thesis`, project `forensic-dgp-thesis`, zone
`us-central1-a`, user `janusdominic0`. Fresh root:
`/home/janusdominic0/forensic-dgp/cctv_dgp_input_selection_vm_v19_r2`.
Detached tmux session: `dgp_input_selection_v19_r2`.
The launcher creates tmux and starts the supervisor inside it. Use normal VM SSH;
an idle outer tmux shell is also allowed. Existing/partial R2 root or competing
GPU/tmux programs cause rejection; retain the printed evidence.

Inference cap 20 minutes; overall cap 30 minutes including audit/export. Do not
delete an existing result root, rerun a failed job or reinstall dependencies.

## Upload and launch

1. Upload the archive from **Windows Google Cloud SDK Shell (CMD)**:

   ```bat
   cd /d "C:\xampp\htdocs\YEAR 4\Testing\outputs"
   gcloud compute scp --project=forensic-dgp-thesis --zone=us-central1-a "cctv-dgp-input-selection-v19-r2-execution.tar.gz" "janusdominic0@forensic-dgp-thesis:/home/janusdominic0/"
   ```

2. Upload its checksum from the same **Windows Google Cloud SDK Shell**:

   ```bat
   gcloud compute scp --project=forensic-dgp-thesis --zone=us-central1-a "cctv-dgp-input-selection-v19-r2-execution.tar.gz.sha256" "janusdominic0@forensic-dgp-thesis:/home/janusdominic0/"
   ```

3. Upload the launcher from the same **Windows Google Cloud SDK Shell**:

   ```bat
   gcloud compute scp --project=forensic-dgp-thesis --zone=us-central1-a "launch_cctv_dgp_input_selection_v19_r2.py" "janusdominic0@forensic-dgp-thesis:/home/janusdominic0/"
   ```

4. Verify the uploaded files in **VM SSH**:

   ```bash
   cd ~ &&
   sha256sum -c cctv-dgp-input-selection-v19-r2-execution.tar.gz.sha256 &&
   printf '%s  %s\n' \
     '9eb87a551c2c916d3bec624bbc7c04a319a94d6f3c20f21278c8403879e148cd' \
     'launch_cctv_dgp_input_selection_v19_r2.py' | sha256sum -c -
   ```

   Expected: the archive and launcher each report `OK`.

5. Launch in **VM SSH**; this creates detached tmux automatically:

   ```bash
   python3 ~/launch_cctv_dgp_input_selection_v19_r2.py \
     --protocol-sha 5c128d6715785f84858f762035a168f396b46787d6a864b1d8d6437ffc69dc3f \
     --archive-sha 3397a6c3e81e1b04a492f51ec6e5fad939b5a85bc8b318253a04c7696eba581e
   ```

   Expected: preflight success and launch receipt containing
   `"session": "dgp_input_selection_v19_r2"`. Preserve any traceback; do not retry
   the same launcher against a partial R2 root.

## Watch and check export

1. Watch progress in **VM SSH**:

   ```bash
   tail -F ~/forensic-dgp/cctv_dgp_input_selection_vm_v19_r2/inference.log ~/forensic-dgp/cctv_dgp_input_selection_vm_v19_r2/supervisor.log
   ```

   **Ctrl+C closes the viewer only.** Detached tmux continues. Training parity
   must finish before the 520 development predictions. Audit and export follow
   inference completion; the maximum overall budget is 1,800 seconds.

2. Check successful export readiness in **VM SSH**:

   ```bash
   cat ~/forensic-dgp/cctv_dgp_input_selection_vm_v19_r2/supervisor_completion.json
   ```

   Expected: `"complete": true`, `"archive_sha256"`, `"bytes"`,
   `"validation_cases": 520`, `"optimizer_updates": 0` and `"backward_calls": 0`.
   `"scientific_guard_passed"` can be false even when export succeeds. Export
   completion is not adoption or output usefulness; all returns require local
   audit and visual review.

Optional live view in **VM SSH**:

```bash
tmux attach -t dgp_input_selection_v19_r2
```

Detach: **Ctrl+B**, release both keys, press **D**. A completed job can leave
no live tmux session; use saved receipts/logs.

## Download a successful export

Use these when `supervisor_completion.json` reports a complete export.

1. Download the result from **Windows Google Cloud SDK Shell (CMD)**:

   ```bat
   cd /d "C:\xampp\htdocs\YEAR 4\Testing\outputs"
   gcloud compute scp --project=forensic-dgp-thesis --zone=us-central1-a "janusdominic0@forensic-dgp-thesis:/home/janusdominic0/forensic-dgp/cctv_dgp_input_selection_vm_v19_r2/cctv-dgp-input-selection-v19-r2-results.tar.gz" .
   ```

2. Download its checksum from the same **Windows Google Cloud SDK Shell**:

   ```bat
   gcloud compute scp --project=forensic-dgp-thesis --zone=us-central1-a "janusdominic0@forensic-dgp-thesis:/home/janusdominic0/forensic-dgp/cctv_dgp_input_selection_vm_v19_r2/cctv-dgp-input-selection-v19-r2-results.tar.gz.sha256" .
   ```

3. Download its receipt from the same **Windows Google Cloud SDK Shell**:

   ```bat
   gcloud compute scp --project=forensic-dgp-thesis --zone=us-central1-a "janusdominic0@forensic-dgp-thesis:/home/janusdominic0/forensic-dgp/cctv_dgp_input_selection_vm_v19_r2/supervisor_completion.json" "supervisor_completion_v19_r2.json"
   ```

All three land in `C:\xampp\htdocs\YEAR 4\Testing\outputs`. The local receipt
alias preserves earlier exports. Keep archive/sidecar names exact.

## Download an execution or audit failure

Do not relaunch or edit the frozen bundle after a traceback.

1. Check failure export readiness in **VM SSH**:

   ```bash
   cat ~/forensic-dgp/cctv_dgp_input_selection_vm_v19_r2/failure_export.json
   ```

   `"complete": true` refers to the failure export. Inference/audit may have
   failed or stopped early.

2. Download the failure archive from **Windows Google Cloud SDK Shell (CMD)**:

   ```bat
   cd /d "C:\xampp\htdocs\YEAR 4\Testing\outputs"
   gcloud compute scp --project=forensic-dgp-thesis --zone=us-central1-a "janusdominic0@forensic-dgp-thesis:/home/janusdominic0/forensic-dgp/cctv_dgp_input_selection_vm_v19_r2/cctv-dgp-input-selection-v19-r2-failure.tar.gz" .
   ```

3. Download its checksum from the same **Windows Google Cloud SDK Shell**:

   ```bat
   gcloud compute scp --project=forensic-dgp-thesis --zone=us-central1-a "janusdominic0@forensic-dgp-thesis:/home/janusdominic0/forensic-dgp/cctv_dgp_input_selection_vm_v19_r2/cctv-dgp-input-selection-v19-r2-failure.tar.gz.sha256" .
   ```

4. Download its receipt from the same **Windows Google Cloud SDK Shell**:

   ```bat
   gcloud compute scp --project=forensic-dgp-thesis --zone=us-central1-a "janusdominic0@forensic-dgp-thesis:/home/janusdominic0/forensic-dgp/cctv_dgp_input_selection_vm_v19_r2/failure_export.json" "failure_export_v19_r2.json"
   ```

If bootstrap fails before inference, preserve the printed traceback and
`~/forensic-dgp/cctv_dgp_input_selection_vm_v19_r2/bootstrap_preflight.json`.
If archive export fails, preserve `failure_export_incomplete.json` and partial
files. Neither failure permits an unchanged repeat.

## Assistant's independent local audit after return

The assistant verifies transfer, safely imports into a fresh directory and
audits saved outputs plus 24 CPU decoder probes. Zero local training or cloud
operations. Original success/failure evidence is retained.

Successful return, PowerShell in `C:\xampp\htdocs\YEAR 4\Testing`:

```powershell
.\venv\Scripts\python.exe -B -X utf8 -u scripts\import_cctv_dgp_input_selection_v19_r2.py --archive outputs\cctv-dgp-input-selection-v19-r2-results.tar.gz --completion outputs\supervisor_completion_v19_r2.json --extract-to outputs\cctv_dgp_input_selection_return_v19_r2
```

Failure return:

```powershell
.\venv\Scripts\python.exe -B -X utf8 -u scripts\import_cctv_dgp_input_selection_v19_r2.py --archive outputs\cctv-dgp-input-selection-v19-r2-failure.tar.gz --completion outputs\failure_export_v19_r2.json --extract-to outputs\cctv_dgp_input_selection_failure_return_v19_r2
```

Full audit 300 seconds internal/330 seconds process cap. Partial failures receive
source/protocol/transfer verification only. All five original-cell sheets still
need development review before choosing the next experiment or app adoption.

Frozen archive 87,831,525 bytes/191 members. Protocol
`5c128d6715785f84858f762035a168f396b46787d6a864b1d8d6437ffc69dc3f`;
archive`3397a6c3e81e1b04a492f51ec6e5fad939b5a85bc8b318253a04c7696eba581e`.
Full design, provenance and limitations:
[V19 r2 plan](<C:/xampp/htdocs/YEAR 4/Testing/CCTV_DGP_INPUT_SELECTION_V19_R2_PLAN.md>).
