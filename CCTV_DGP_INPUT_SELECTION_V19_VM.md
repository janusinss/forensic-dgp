# V19 inference — step-by-step VM commands

**Closed5 October2026: original V19 stopped at the first development baseline
PNG equality check after all50 training parity cases passed.** The downloaded
failure is independently audited; no completed development outputs exist.
Preserve this frozen run. **Do not run the historical upload/launch commands below.**
The separate three-case diagnostic confirms the cause. Corrected V19 r2 has
also completed execution/audit/review: normalization passes, preservation fails.
Those launch commands are closed too. Current evidence:
[CCTV_DGP_INPUT_SELECTION_V19_R2_RESULTS.md](<C:/xampp/htdocs/YEAR 4/Testing/CCTV_DGP_INPUT_SELECTION_V19_R2_RESULTS.md>).
Failure evidence and limits:
[CCTV_DGP_INPUT_SELECTION_V19_FAILURE.md](<C:/xampp/htdocs/YEAR 4/Testing/CCTV_DGP_INPUT_SELECTION_V19_FAILURE.md>).
No parity relaxation, model-quality decision or V19 resume is claimed.

## Historical preparation — superseded by the failure closure

**V19 runs inference only. No training or new checkpoint fitting.** It tests the
already trained V18 terminal decoder and frozen input-only restoration selector
on104 development identities/520 synthetic cases, after50 fresh training parity
checks. Prepared and independently transfer-audited on5 October2026; no VM launch
is claimed. V18 is closed; do not repeat its training/import commands.

Existing NVIDIA L4 VM `forensic-dgp-thesis`, project `forensic-dgp-thesis`, zone
`us-central1-a`, user `janusdominic0`. Fresh root:
`/home/janusdominic0/forensic-dgp/cctv_dgp_input_selection_vm_v19`.
Detached tmux session: `dgp_input_selection_v19`.
The launcher creates tmux and starts the supervisor inside it. Use normal VM SSH;
an idle outer tmux shell is also allowed. Existing/partial root or competing
GPU/tmux programs cause rejection; retain the printed evidence.

Inference cap20 minutes; overall cap30 minutes including audit/export. Do not
delete an existing result root, rerun a failed job or reinstall dependencies.

## Upload and launch

1. Upload the archive from **Windows Google Cloud SDK Shell (CMD)**:

   ```bat
   cd /d "C:\xampp\htdocs\YEAR 4\Testing\outputs"
   gcloud compute scp --project=forensic-dgp-thesis --zone=us-central1-a "cctv-dgp-input-selection-v19-execution.tar.gz" "janusdominic0@forensic-dgp-thesis:/home/janusdominic0/"
   ```

2. Upload its checksum from the same **Windows Google Cloud SDK Shell**:

   ```bat
   gcloud compute scp --project=forensic-dgp-thesis --zone=us-central1-a "cctv-dgp-input-selection-v19-execution.tar.gz.sha256" "janusdominic0@forensic-dgp-thesis:/home/janusdominic0/"
   ```

3. Upload the launcher from the same **Windows Google Cloud SDK Shell**:

   ```bat
   gcloud compute scp --project=forensic-dgp-thesis --zone=us-central1-a "launch_cctv_dgp_input_selection_v19.py" "janusdominic0@forensic-dgp-thesis:/home/janusdominic0/"
   ```

4. Verify the archive in **VM SSH**:

   ```bash
   cd ~ &&
   sha256sum -c cctv-dgp-input-selection-v19-execution.tar.gz.sha256
   ```

   Expected: `cctv-dgp-input-selection-v19-execution.tar.gz: OK`.

5. Launch in **VM SSH**; this creates detached tmux automatically:

   ```bash
   python3 ~/launch_cctv_dgp_input_selection_v19.py \
     --protocol-sha 2d707ebba97524867c4ce7610666e3abeb29638eb9054a8025983b6476e628d2 \
     --archive-sha da75c16e25c0b83183beda230d90c552c9c24c8f6c070f8a992f77042350c899
   ```

   Expected: preflight success and launch receipt containing
   `"session": "dgp_input_selection_v19"`. Preserve any traceback; do not retry
   the same launcher against a partial root.

## Watch and check export

1. Watch progress in **VM SSH**:

   ```bash
   tail -F ~/forensic-dgp/cctv_dgp_input_selection_vm_v19/inference.log ~/forensic-dgp/cctv_dgp_input_selection_vm_v19/supervisor.log
   ```

   **Ctrl+C closes the viewer only.** Detached tmux continues. Training parity
   must finish before the520 development predictions. Audit and export follow
   inference completion; the maximum overall budget is1,800 seconds.

2. Check successful export readiness in **VM SSH**:

   ```bash
   cat ~/forensic-dgp/cctv_dgp_input_selection_vm_v19/supervisor_completion.json
   ```

   Expected: `"complete": true`, `"archive_sha256"`, `"bytes"`,
   `"validation_cases": 520`, `"optimizer_updates": 0` and `"backward_calls": 0`.
   `"scientific_guard_passed"` can be false even when export succeeds. Export
   completion is not adoption or output usefulness; all returns require local
   audit and visual review.

Optional live view in **VM SSH**:

```bash
tmux attach -t dgp_input_selection_v19
```

Detach: **Ctrl+B**, release both keys, press **D**. A completed job can leave
no live tmux session; use saved receipts/logs.

## Download a successful export

Use these when `supervisor_completion.json` reports a complete export.

1. Download the result from **Windows Google Cloud SDK Shell (CMD)**:

   ```bat
   cd /d "C:\xampp\htdocs\YEAR 4\Testing\outputs"
   gcloud compute scp --project=forensic-dgp-thesis --zone=us-central1-a "janusdominic0@forensic-dgp-thesis:/home/janusdominic0/forensic-dgp/cctv_dgp_input_selection_vm_v19/cctv-dgp-input-selection-v19-results.tar.gz" .
   ```

2. Download its checksum from the same **Windows Google Cloud SDK Shell**:

   ```bat
   gcloud compute scp --project=forensic-dgp-thesis --zone=us-central1-a "janusdominic0@forensic-dgp-thesis:/home/janusdominic0/forensic-dgp/cctv_dgp_input_selection_vm_v19/cctv-dgp-input-selection-v19-results.tar.gz.sha256" .
   ```

3. Download its receipt from the same **Windows Google Cloud SDK Shell**:

   ```bat
   gcloud compute scp --project=forensic-dgp-thesis --zone=us-central1-a "janusdominic0@forensic-dgp-thesis:/home/janusdominic0/forensic-dgp/cctv_dgp_input_selection_vm_v19/supervisor_completion.json" "supervisor_completion_v19.json"
   ```

All three land in `C:\xampp\htdocs\YEAR 4\Testing\outputs`. The local receipt
alias preserves earlier exports. Keep archive/sidecar names exact.

## Download an execution or audit failure

Do not relaunch or edit the frozen bundle after a traceback.

1. Check failure export readiness in **VM SSH**:

   ```bash
   cat ~/forensic-dgp/cctv_dgp_input_selection_vm_v19/failure_export.json
   ```

   `"complete": true` refers to the failure export. Inference/audit may have
   failed or stopped early.

2. Download the failure archive from **Windows Google Cloud SDK Shell (CMD)**:

   ```bat
   cd /d "C:\xampp\htdocs\YEAR 4\Testing\outputs"
   gcloud compute scp --project=forensic-dgp-thesis --zone=us-central1-a "janusdominic0@forensic-dgp-thesis:/home/janusdominic0/forensic-dgp/cctv_dgp_input_selection_vm_v19/cctv-dgp-input-selection-v19-failure.tar.gz" .
   ```

3. Download its checksum from the same **Windows Google Cloud SDK Shell**:

   ```bat
   gcloud compute scp --project=forensic-dgp-thesis --zone=us-central1-a "janusdominic0@forensic-dgp-thesis:/home/janusdominic0/forensic-dgp/cctv_dgp_input_selection_vm_v19/cctv-dgp-input-selection-v19-failure.tar.gz.sha256" .
   ```

4. Download its receipt from the same **Windows Google Cloud SDK Shell**:

   ```bat
   gcloud compute scp --project=forensic-dgp-thesis --zone=us-central1-a "janusdominic0@forensic-dgp-thesis:/home/janusdominic0/forensic-dgp/cctv_dgp_input_selection_vm_v19/failure_export.json" "failure_export_v19.json"
   ```

If bootstrap fails before inference, preserve the printed traceback and
`~/forensic-dgp/cctv_dgp_input_selection_vm_v19/bootstrap_preflight.json`.
If archive export fails, preserve `failure_export_incomplete.json` and partial
files. Neither failure permits an unchanged repeat.

## Assistant's independent local audit after return

The assistant verifies transfer, safely imports into a fresh directory and
audits saved outputs plus24 CPU decoder probes. Zero local training or cloud
operations. Original success/failure evidence is retained.

Successful return, PowerShell in `C:\xampp\htdocs\YEAR 4\Testing`:

```powershell
.\venv\Scripts\python.exe -B -X utf8 -u scripts\import_cctv_dgp_input_selection_v19.py --archive outputs\cctv-dgp-input-selection-v19-results.tar.gz --completion outputs\supervisor_completion_v19.json --extract-to outputs\cctv_dgp_input_selection_return_v19
```

Failure return:

```powershell
.\venv\Scripts\python.exe -B -X utf8 -u scripts\import_cctv_dgp_input_selection_v19.py --archive outputs\cctv-dgp-input-selection-v19-failure.tar.gz --completion outputs\failure_export_v19.json --extract-to outputs\cctv_dgp_input_selection_failure_return_v19
```

Full audit300 seconds internal/330 seconds process cap. Partial failures receive
source/protocol/transfer verification only. All five original-cell sheets still
need development review before choosing the next experiment or app adoption.

Frozen archive80,388,709 bytes/146 members. Protocol
`2d707ebba97524867c4ce7610666e3abeb29638eb9054a8025983b6476e628d2`;
archive`da75c16e25c0b83183beda230d90c552c9c24c8f6c070f8a992f77042350c899`.
Full design, provenance and limitations:
[V19 plan](<C:/xampp/htdocs/YEAR 4/Testing/CCTV_DGP_INPUT_SELECTION_V19_PLAN.md>).
