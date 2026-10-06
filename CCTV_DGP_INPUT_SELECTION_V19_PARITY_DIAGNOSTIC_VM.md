# V19 parity diagnostic — exact manual commands

**Closed 5 October 2026: the returned diagnostic and independent local audit
confirm the normalization cause. Do not repeat the historical commands below.**
Eight L4 DGP forwards completed; both V15 PNGs are recovered exactly with NumPy
normalization, and the fixed V18 training raw/PNG matches CUDA scalar division.
The original V19/diagnostic sources, checkpoints and gates remain unchanged.
Audit: [local_independent_audit.json](<C:/xampp/htdocs/YEAR 4/Testing/outputs/cctv_dgp_input_selection_v19_parity_diagnostic_return_r1/local_independent_audit.json>).
The separate V19 r2 has also completed execution, return audit and development
review. Its normalization correction passes; automatic preservation still fails.
[R2 results and remaining work](<C:/xampp/htdocs/YEAR 4/Testing/CCTV_DGP_INPUT_SELECTION_V19_R2_RESULTS.md>).

The following records the historical preparation and commands.

**Inference diagnostic only.** The downloaded original V19 failure is audited.
This small job measures both normalization paths on three fixed cases; it does
not resume V19, train a model, change a checkpoint or relax a gate. Four regressions
and the separate transfer audit pass. GPU diagnosis remains pending.

Existing L4 VM/project `forensic-dgp-thesis`, zone`us-central1-a`,
user`janusdominic0`. New root:
`~/forensic-dgp/cctv_dgp_input_selection_v19_parity_diagnostic_r1`.
The launcher automatically creates detached tmux`dgp_v19_parity_diagnostic_r1`.
Worker limit2minutes; whole supervisor/export limit4minutes. Original V19 remains
read only. No installation, cleanup, resume or automatic next pilot.

## Upload and launch

1. Select the upload directory in **Windows Google Cloud SDK Shell (CMD)**:

   ```bat
   cd /d "C:\xampp\htdocs\YEAR 4\Testing\outputs"
   ```

2. Upload the script and checksum from **Windows Google Cloud SDK Shell**:

   ```bat
   gcloud compute scp --project=forensic-dgp-thesis --zone=us-central1-a "diagnose_cctv_dgp_input_selection_v19_parity.py" "diagnose_cctv_dgp_input_selection_v19_parity.py.sha256" "janusdominic0@forensic-dgp-thesis:/home/janusdominic0/"
   ```

3. Verify the script in **VM SSH**:

   ```bash
   cd ~ &&
   sha256sum -c diagnose_cctv_dgp_input_selection_v19_parity.py.sha256
   ```

   Expected:`diagnose_cctv_dgp_input_selection_v19_parity.py: OK`.

4. Launch in **VM SSH**; it starts inside detached tmux automatically:

   ```bash
   python3 ~/diagnose_cctv_dgp_input_selection_v19_parity.py \
     --launch \
     --expected-sha cad6da856e74ae9dcaa28d463304c93997d7e21ebee49fb65e5fa8a6d030e729
   ```

   Expected:`"tmux_launched": true` and
   `"session": "dgp_v19_parity_diagnostic_r1"`.
   An idle outer tmux shell is allowed. Competing GPU/tasks, missing original
   evidence, changed hashes or an existing diagnostic root cause rejection.
   Preserve the error; do not delete the root or repeat the same diagnostic.

## Watch and check export

1. Watch the saved supervisor log in **VM SSH**:

   ```bash
   tail -F ~/forensic-dgp/cctv_dgp_input_selection_v19_parity_diagnostic_r1/supervisor.log
   ```

   Ctrl+C closes only this viewer. Detached tmux continues.
   Saved worker output appears when the bounded worker ends.

2. Check export readiness in **VM SSH**:

   ```bash
   cat ~/forensic-dgp/cctv_dgp_input_selection_v19_parity_diagnostic_r1/export.json
   ```

   Expected:`"complete": true`, `"archive_sha256"`, `"bytes"`.
   `"diagnostic_succeeded": true` means the measurements finished. The result
   can still report`"normalization_hypothesis_confirmed": false`.
   A worker failure is also exported. Download it without retrying.

Optional live tmux view:

```bash
tmux attach -t dgp_v19_parity_diagnostic_r1
```

Detach with **Ctrl+B**, release both keys, press **D**.

## Download after the export exists

Use **Windows Google Cloud SDK Shell (CMD)**, still in
`C:\xampp\htdocs\YEAR 4\Testing\outputs`.

1. Download the diagnostic archive:

   ```bat
   gcloud compute scp --project=forensic-dgp-thesis --zone=us-central1-a "janusdominic0@forensic-dgp-thesis:/home/janusdominic0/forensic-dgp/cctv_dgp_input_selection_v19_parity_diagnostic_r1/cctv-dgp-input-selection-v19-parity-diagnostic-r1.tar.gz" .
   ```

2. Download its checksum:

   ```bat
   gcloud compute scp --project=forensic-dgp-thesis --zone=us-central1-a "janusdominic0@forensic-dgp-thesis:/home/janusdominic0/forensic-dgp/cctv_dgp_input_selection_v19_parity_diagnostic_r1/cctv-dgp-input-selection-v19-parity-diagnostic-r1.tar.gz.sha256" .
   ```

3. Download its export receipt:

   ```bat
   gcloud compute scp --project=forensic-dgp-thesis --zone=us-central1-a "janusdominic0@forensic-dgp-thesis:/home/janusdominic0/forensic-dgp/cctv_dgp_input_selection_v19_parity_diagnostic_r1/export.json" "parity_diagnostic_export_v19_r1.json"
   ```

All three land in the Windows`outputs` directory. The receipt's local name
preserves other export receipts. Keep the archive/checksum filenames exact.

## Assistant's independent return audit

The assistant performs the following local command after all three files arrive:

```powershell
.\venv\Scripts\python.exe -B -X utf8 -u scripts\import_cctv_dgp_input_selection_v19_parity_diagnostic.py --archive outputs\cctv-dgp-input-selection-v19-parity-diagnostic-r1.tar.gz --receipt outputs\parity_diagnostic_export_v19_r1.json --extract-to outputs\cctv_dgp_input_selection_v19_parity_diagnostic_return_r1
```

It checks the transfer and safe members, worker source/protocol, every saved
normalized tensor/raw/PNG, original references, repeat stability, interpretation
and148 unchanged original fingerprints. Zero local neural/training calls.
No subsequent VM run starts automatically. The diagnostic does not establish
restoration quality or complete the full thesis/application goal.
