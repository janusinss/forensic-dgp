# V16 r2: corrected cache timing, manual VM execution

**Closed 5 October 2026:** the returned failure archive, sidecar and receipt
match the expected hash and 839,574,170 bytes. Import and the separate corrected
full audit passed; all 3,128 updates are verified. Both trained snapshots fail
the unchanged preservation guards. All ten original-cell sheets were reviewed.
Results: `CCTV_DGP_BROADER_CODES_V16_R2_RESULTS.md`.
Preserve this negative result. Do not launch or repeat the historical pilot below.

## Historical preparation and collection instructions

Prepared5 October2026 after independent audit of V16's zero-update cache failure.
The original V16 directory, protocol, checkpoint/cache and failure archive stay
unchanged. R2 is a separate finite pilot with corrected timing accounting. No
assistant upload/SSH/VM launch or local neural/backward/optimizer call occurred.
The package was verified locally before the manual run. The latest reported
training completion and post-training audit failure are recorded below;
returned-file verification and usefulness remain pending.

Windows: `C:\xampp\htdocs\YEAR 4\Testing\`.
VM: existing NVIDIA L4 `forensic-dgp-thesis`, `~/forensic-dgp/`.
Project: `forensic-dgp-thesis`; zone: `us-central1-a`; user: `janusdominic0`.
New root: `~/forensic-dgp/cctv_dgp_broader_codes_vm_v16_r2/`.
New detached tmux session: `dgp_broader_codes_v16_r2`.

## Historical collection after the checker failure

The user reports3,128/3,128 updates, epoch8 previews and trainer completion in
1,055.13s. The subsequent auditor rejects `Timing stop receipt differs`. Its
frozen source still checks20 cache references; the R2 protocol requires30.
This is a locally reproduced checker defect. Do not restart training or edit the
frozen VM bundle. The supervisor exports this run under the **failure** filename;
that archive contains the trained results/checkpoints, while preserving the audit
failure. File availability/hash/content still require collection and verification.

The user subsequently reported export readiness: `complete:true`,839,574,170
bytes (800.68MiB), SHA256
`3555eb65b37bcf4cbc24ff36a31f5a80a6dab02051fe3e2c9e7dd2cad026ed65`.
These are expected transfer values, pending independent local hash/size checks.
Continue with the downloads below; do not repeat the historical launcher.

1. Close the log viewer with **Ctrl+C**.

2. Check export readiness in **VM SSH**:

   ```bash
   cat ~/forensic-dgp/cctv_dgp_broader_codes_vm_v16_r2/failure_export.json
   ```

   `complete:true` and `archive_sha256` mean the failure export is ready. They
   do not mean the full audit or quality review passed.

3. Download the archive from **Windows Google Cloud SDK Shell**:

   ```bat
   cd /d "C:\xampp\htdocs\YEAR 4\Testing\outputs"
   gcloud compute scp --project=forensic-dgp-thesis --zone=us-central1-a "janusdominic0@forensic-dgp-thesis:/home/janusdominic0/forensic-dgp/cctv_dgp_broader_codes_vm_v16_r2/cctv-dgp-broader-codes-v16-r2-failure.tar.gz" .
   ```

4. Download its checksum from **Windows Google Cloud SDK Shell**:

   ```bat
   gcloud compute scp --project=forensic-dgp-thesis --zone=us-central1-a "janusdominic0@forensic-dgp-thesis:/home/janusdominic0/forensic-dgp/cctv_dgp_broader_codes_vm_v16_r2/cctv-dgp-broader-codes-v16-r2-failure.tar.gz.sha256" .
   ```

5. Download its receipt from **Windows Google Cloud SDK Shell**:

   ```bat
   gcloud compute scp --project=forensic-dgp-thesis --zone=us-central1-a "janusdominic0@forensic-dgp-thesis:/home/janusdominic0/forensic-dgp/cctv_dgp_broader_codes_vm_v16_r2/failure_export.json" "failure_export_v16_r2.json"
   ```

The local receipt alias preserves earlier V16 exports. The assistant will import
the failure export and perform the separate full arithmetic recovery audit,
then inspect all ten fixed256-cell grids and the unchanged quality guards.
`CCTV_DGP_BROADER_CODES_V16_R2_AUDIT_RECOVERY.md` records the bound correction,
seven passing regressions and the240s audit limit. No new VM training is needed
for this checker correction; usefulness and application readiness remain unproven.

## Verified transfer files

| File under `outputs\` | Verification |
| --- | --- |
| `cctv-dgp-broader-codes-v16-r2-execution.tar.gz` |1,098,502 bytes;27 independently rehashed members |
| `cctv-dgp-broader-codes-v16-r2-execution.tar.gz.sha256` |Exact filename/hash, ASCII with LF |
| `launch_cctv_dgp_broader_codes_v16_r2.py` |Exact checksum-bound copy of the new archived bootstrap |
| `cctv_dgp_broader_codes_v16_r2_preparation.json` |116 parent,6,195 data and2,804 baseline assets reverified in11.61s |
| `cctv_dgp_broader_codes_v16_r2_transfer_audit.json` |Separate archive/source/schedule/head/split preservation checks |

Protocol SHA256: `4f64cf2204458f8fadcab1824fc4bc293c65803e123fdebb304f7b5bd3a502a0`.
Archive SHA256: `6f78d84f0ba044897c41d629dc572f953cb62a9f26dacd9719d95f1ad4528453`.
Bootstrap SHA256: `c1a13b4938cd2ec6b085e8259f99e48612a551d6cb74d0eed50ef02803af9e15`.

Eight focused regressions pass in0.385s: startup/warmup accounting, balanced
source/role sampling, all885 references once, slow-rate rejection, invalid timing
rejection, unchanged objective/batch/snapshot code, Path bootstrap/local VM guard,
Python3.10 syntax and no neural audit imports. Receipt:
`outputs\cctv_dgp_v16_r2_boundary_tests.json`.

## Finite protocol

Same781 training references/3,905 training cases, separate104 development-validation
references/520 cases,8 epochs/3,128 updates/31,280 exposures, batch10 with original
balanced source/profile schedule. Same reset2,422,432-parameter code-only head,
frozen trained DGP and declared frozen pretrained prior/teacher/recognizer.
No V14/V15 fitted initialization, V16 partial-head resume, validation teacher
labels, native24/reserved32 use, automatic best.pth or promotion.

Same900s cache,1,200s fit/evaluation,240s VM arithmetic audit,2,400s overall
supervisor cap,20GiB VRAM and12GiB free-disk requirement. Early cache projection
now uses30 source/role-balanced references. Initial model setup and four first
reference warmups are counted once;18 train/8 validation steady reference timings
estimate remaining work. Safety1.25 and30s reserve remain. The same25-update fit
projection and epoch4 ≥1% training-preview CE improvement stop rules remain.
This changes timing and cache order, with unchanged data, learning and quality
criteria. If a gate fails, preserve it and collect the failure; do not repeat it.

This remains a component experiment on paired synthetic photographic camera
proxies. Source folders describe provenance, not ethnicity, native CCTV truth or
Zamboanga performance. No DGP app integration or useful output claim follows from
training/export completion alone. Review raw outputs and all ten original-cell
grids after independent returned-file audit before proposing an output path.

## Historical one-time launch — already executed; do not repeat

Use Windows **Google Cloud SDK Shell (CMD)** for transfers. Invoke the bootstrap
from normal VM SSH; it opens detached tmux itself. Existing outer tmux/GPU jobs
or an existing/partial r2 directory cause rejection. Preserve those jobs/directories.

1. Upload from **Windows Google Cloud SDK Shell**:

   ```bat
   cd /d "C:\xampp\htdocs\YEAR 4\Testing\outputs"
   gcloud compute scp --project=forensic-dgp-thesis --zone=us-central1-a "cctv-dgp-broader-codes-v16-r2-execution.tar.gz" "janusdominic0@forensic-dgp-thesis:/home/janusdominic0/"
   gcloud compute scp --project=forensic-dgp-thesis --zone=us-central1-a "cctv-dgp-broader-codes-v16-r2-execution.tar.gz.sha256" "janusdominic0@forensic-dgp-thesis:/home/janusdominic0/"
   gcloud compute scp --project=forensic-dgp-thesis --zone=us-central1-a "launch_cctv_dgp_broader_codes_v16_r2.py" "janusdominic0@forensic-dgp-thesis:/home/janusdominic0/"
   ```

2. Verify the transfer in **VM SSH**:

   ```bash
   cd ~ &&
   sha256sum -c cctv-dgp-broader-codes-v16-r2-execution.tar.gz.sha256
   ```

3. Launch from **normal VM SSH**, outside tmux:

   ```bash
   python3 ~/launch_cctv_dgp_broader_codes_v16_r2.py \
     --protocol-sha 4f64cf2204458f8fadcab1824fc4bc293c65803e123fdebb304f7b5bd3a502a0 \
     --archive-sha 6f78d84f0ba044897c41d629dc572f953cb62a9f26dacd9719d95f1ad4528453
   ```

4. Watch progress in **VM SSH**:

   ```bash
   tail -F ~/forensic-dgp/cctv_dgp_broader_codes_vm_v16_r2/trainer.log ~/forensic-dgp/cctv_dgp_broader_codes_vm_v16_r2/supervisor.log
   ```

   Wait for the supervisor's final `"complete": true` and `"archive_sha256"`.
   A bootstrap launch or trainer completion line is earlier than audit/export.
   Ctrl+C closes only the log viewer. The supervisor has a40-minute cap.

5. Download successful results from **Windows Google Cloud SDK Shell**:

   ```bat
   cd /d "C:\xampp\htdocs\YEAR 4\Testing\outputs"
   gcloud compute scp --project=forensic-dgp-thesis --zone=us-central1-a "janusdominic0@forensic-dgp-thesis:/home/janusdominic0/forensic-dgp/cctv_dgp_broader_codes_vm_v16_r2/cctv-dgp-broader-codes-v16-r2-results.tar.gz" .
   gcloud compute scp --project=forensic-dgp-thesis --zone=us-central1-a "janusdominic0@forensic-dgp-thesis:/home/janusdominic0/forensic-dgp/cctv_dgp_broader_codes_vm_v16_r2/cctv-dgp-broader-codes-v16-r2-results.tar.gz.sha256" .
   gcloud compute scp --project=forensic-dgp-thesis --zone=us-central1-a "janusdominic0@forensic-dgp-thesis:/home/janusdominic0/forensic-dgp/cctv_dgp_broader_codes_vm_v16_r2/supervisor_completion.json" "supervisor_completion_v16_r2.json"
   ```

The receipt is renamed locally to preserve previous exports. Downloaded archive
and sidecar filenames stay unchanged for their checksum/import checks.

## Failure collection

If a gate fails, close the viewer and check:

```bash
cat ~/forensic-dgp/cctv_dgp_broader_codes_vm_v16_r2/failure_export.json
```

`complete:true` in that receipt means the failure export finished. Collect it
from **Windows Google Cloud SDK Shell**:

```bat
cd /d "C:\xampp\htdocs\YEAR 4\Testing\outputs"
gcloud compute scp --project=forensic-dgp-thesis --zone=us-central1-a "janusdominic0@forensic-dgp-thesis:/home/janusdominic0/forensic-dgp/cctv_dgp_broader_codes_vm_v16_r2/cctv-dgp-broader-codes-v16-r2-failure.tar.gz" .
gcloud compute scp --project=forensic-dgp-thesis --zone=us-central1-a "janusdominic0@forensic-dgp-thesis:/home/janusdominic0/forensic-dgp/cctv_dgp_broader_codes_vm_v16_r2/cctv-dgp-broader-codes-v16-r2-failure.tar.gz.sha256" .
gcloud compute scp --project=forensic-dgp-thesis --zone=us-central1-a "janusdominic0@forensic-dgp-thesis:/home/janusdominic0/forensic-dgp/cctv_dgp_broader_codes_vm_v16_r2/failure_export.json" "failure_export_v16_r2.json"
```

The assistant imports and audits these locally. Successful import:

```powershell
Set-Location -LiteralPath 'C:\xampp\htdocs\YEAR 4\Testing'
.\venv\Scripts\python.exe -X utf8 -u scripts/import_cctv_dgp_broader_codes_v16_r2.py --archive outputs/cctv-dgp-broader-codes-v16-r2-results.tar.gz --completion outputs/supervisor_completion_v16_r2.json --extract-to outputs/cctv_dgp_broader_codes_return_v16_r2
```

Failed import uses the corresponding failure archive, `failure_export_v16_r2.json`
and fresh destination `outputs/cctv_dgp_broader_codes_failure_return_v16_r2`.
Preserve an existing/partial return. If the bootstrap fails before supervisor
launch, preserve the extracted directory and collect `bootstrap_preflight.json`;
it includes stdout/stderr and command/timeout records. Do not delete/re-extract it.

After collection, check `nvidia-smi --query-compute-apps=pid,process_name --format=csv,noheader`
and `tmux list-sessions` in VM SSH. When both are idle, the user can stop the VM
through Cloud Console. No assistant cloud shutdown or automatic next pilot occurs.
