# V16 finite code-learning pilot: manual VM transfer and return

**Closed failure,5 October2026:** downloaded export/hash and serialized evidence
are independently audited. The run has0 backward/optimizer updates,20 train-only
teacher arrays and the original cache timing rejection; no learned restoration.
See `CCTV_DGP_BROADER_CODES_V16_FAILURE_AUDIT.md`. Preserve this original package,
VM directory and failure. A separately verified timing correction is documented
in `CCTV_DGP_BROADER_CODES_V16_R2_VM.md`; follow that revision's new filenames/root
and manual commands. The original instructions/history below are retained.

Prepared on 4 October 2026. **The package is verified locally. No CUDA backward
preflight result, completed optimizer training, useful restored output or
application adoption has been reported.**
On 5 October, the user reported invoking the launcher inside tmux; the idle guard
rejected that attempt before archive extraction or training launch.
The user selected transfer files and pasteable commands only;
this supersedes earlier automatic gcloud execution instructions.
The last historical VM receipt recorded TERMINATED; no live cloud query was made.

**Current manual report, 5 October 2026:** recovery r1 confirmed the original path
error, passed source/data/CUDA availability preflight and opened tmux. The trainer
then stopped with `Cache projection exceeds900s` at line153; the supervisor
reported child exit1. Use the failure-download commands in **Collect success or
failure** below. Close the log viewer with Ctrl+C, check `failure_export.json`,
then download the failure archive, checksum and export receipt. Preserve the run;
do not rerun, resume or change limits before independent failure/timing review.
The user supplied `cache_timing.json`:22.75441105899995s elapsed after20 references,
1,874.5294464701833s projected versus900s cap. Local arithmetic reproduces the
estimate from the pinned protocol exactly (elapsed×81.0625+30s). This is a31.24-minute
projection from a startup-inclusive22.75-second sample. Serialized file integrity
and startup/steady-state cost separation still need independent failure review.
The gate runs after20 references and precedes backward preflight/optimizer updates.

**Earlier manual report, 5 October 2026:** after clearing the outer tmux session,
the original bootstrap extracted V16, then its preflight child returned exit1
at launcher line59. `capture_output=True, check=True` hid the child's stderr.
Local reproduction confirms that the launcher passes strings to `verify()`,
which expects `Path` objects. The underlying VM stderr has not yet been received.
The separate **preflight recovery r1** below checks that the VM error is this
specific bug before proceeding, uses `Path` arguments, and preserves every
frozen package byte. It refuses existing launch/training evidence and keeps
the original failure. Use that section for the currently extracted directory.

Windows workspace: `C:\xampp\htdocs\YEAR 4\Testing\`.
Existing Linux repository: `~/forensic-dgp/` on `forensic-dgp-thesis`, NVIDIA L4.
This runbook's intended counterpart is `~/forensic-dgp/CCTV_DGP_BROADER_CODES_V16_VM.md`
after separate transfer; it is not claimed uploaded.

## Verified files

| File under the Windows workspace | Purpose |
| --- | --- |
| `outputs\cctv-dgp-broader-codes-v16-execution.tar.gz` | 1,088,480-byte package; all23 members independently checked |
| `outputs\cctv-dgp-broader-codes-v16-execution.tar.gz.sha256` | Exact LF checksum sidecar |
| `outputs\launch_cctv_dgp_broader_codes_v16.py` | Checksum-bound one-time VM bootstrap |
| `outputs\cctv_dgp_broader_codes_v16_transfer_audit.json` | Independent archive/source-copy verification receipt |
| `outputs\cctv_dgp_broader_codes_v16_preparation.json` | Dataset, baseline and parent-asset verification receipt |

Protocol SHA256: `4331c28c97a659846eab76ee0dee9150811a00da3c6139d8d24b7905173a6681`.
Archive SHA256: `a40f68ccdeb7e7cf2e1420bef8c984a7faad6120461073da4362bcf2fb8ab65e`.
Bootstrap SHA256: `09b1a40613295842e45c30fe70880e5b8415ee66013e636ed8c11cf53b17197f`.

Separate recovery file: `outputs\recover_cctv_dgp_broader_codes_v16_preflight_r1.py`
(12,320 bytes) and its `.py.sha256` sidecar. SHA256:
`489e85d001b3cca74fdeca3c9fbd50edc2e5bf1fa7aa30c2b45db36bce99bb17`.
Ten targeted regressions pass in0.624s with zero local neural/backward/optimizer
calls. The original23 installed members and original bootstrap hashes have been
rechecked without changing the protocol, trainer, supervisor, splits or stops.
Receipt: `outputs\cctv_dgp_v16_preflight_recovery_r1_transfer_audit.json`.

## Finite scope and stop rules

Reset our2,422,432-parameter code-only head; do not load the fitted V14/V15 head.
Retained trained DGP, declared CodeFormer encoder/generator, clean-code teacher
and recognizer stay frozen. This follows the later documented face-prior
architecture authorization. A pretrained model alone is not our trained DGP.
The component still needs a separately evaluated DGP-preserving output path
before application adoption.

Use781 training references:390 from the Asian-source folder and391 FFHQ
counterparts, each with five frozen synthetic camera profiles. Source names
describe provenance, not ethnicity or native CCTV/Zamboanga performance.
The unchanged104-reference/520-case development-validation cohort receives no
optimization or teacher labels. Prior development use, unknown identity overlap
beyond exact content checks, pretrained FFHQ exposure and Asian target resolution
remain limitations. Native development24 and reserved32 remain unused.

Eight epochs contain3,128 updates/31,280 exposures. Each ten-case batch has one
case per source/profile; each epoch covers all3,905 training cases and replays
five shorter-source cases. Observed-token CE is the only loss. AdamW lr0.0003,
weight decay0.01, gradient clip1, no AMP. Fixed snapshots0/4/8 have no best.pth,
selection, app replacement, resume or automatic rerun. Preserve clear/degraded
and source/profile guards even when they reject the component.

| Stop rule | Frozen limit |
| --- | --- |
| Cache / fit including evaluation / complete supervisor | 900s /1,200s /2,400s; VM arithmetic audit has a240s child cap |
| Early timing | Conservative projection after20 references and25 updates; stop above the stage cap |
| Resources | Require12GiB free disk; peak allocated VRAM≤20GiB; stream at most ten cached cases per batch |
| Numerical/state checks | Stop on invalid arrays, nonfinite loss/gradients, frozen-state/count failures or bad starting/fresh-image parity |
| Fitting stop | At epoch4, stop if fixed training-preview CE improves less than1% from epoch0 |

Expected runtime is approximately20 minutes, an estimate rather than an executed
measurement. The full40-minute limit is enforced. The4,425-case float32 cache
uses approximately9.3GB and stays on the VM. Outputs, checkpoints, training
teacher codes, probes, traces and ten original256-cell grids are returned.

## Preflight recovery r1 for the reported extracted V16 directory

This is a standalone bootstrap correction, not a new training recipe or resume.
The original package and failed-launch directory stay in place. The recovery
checks idle GPU/tmux,12GiB free disk, original archive and every installed member.
It records the known original failure and corrected preflight with a120s cap per
child. Any different failure stops and prints its stderr. It starts the unchanged
finite supervisor only with explicit `--launch`. Do not open an outer tmux session.

1. Upload the two new files from **Windows Google Cloud SDK Shell**:

   ```bat
   cd /d "C:\xampp\htdocs\YEAR 4\Testing\outputs"
   gcloud compute scp --project=forensic-dgp-thesis --zone=us-central1-a "recover_cctv_dgp_broader_codes_v16_preflight_r1.py" "janusdominic0@forensic-dgp-thesis:/home/janusdominic0/"
   gcloud compute scp --project=forensic-dgp-thesis --zone=us-central1-a "recover_cctv_dgp_broader_codes_v16_preflight_r1.py.sha256" "janusdominic0@forensic-dgp-thesis:/home/janusdominic0/"
   ```

2. Verify the transfer in **normal VM SSH**:

   ```bash
   cd ~ &&
   sha256sum -c recover_cctv_dgp_broader_codes_v16_preflight_r1.py.sha256
   ```

3. Run the corrected one-time bootstrap from **normal VM SSH**:

   ```bash
   python3 ~/recover_cctv_dgp_broader_codes_v16_preflight_r1.py \
     --expected-sha 489e85d001b3cca74fdeca3c9fbd50edc2e5bf1fa7aa30c2b45db36bce99bb17 \
     --launch
   ```

   Expect confirmation of the original string/Path error, followed by
   `V16 source/data/CUDA availability preflight passed`, then
   `"tmux_launched": true`. This is not a CUDA backward/gradient check or a
   completion receipt. If it stops, provide the printed error before another
   attempt. Without `--launch` this command verifies only and starts no training.

4. Watch progress from **VM SSH**:

   ```bash
   tail -F ~/forensic-dgp/cctv_dgp_broader_codes_vm_v16/trainer.log ~/forensic-dgp/cctv_dgp_broader_codes_vm_v16/supervisor.log
   ```

The success/failure archive names, download commands,40-minute supervisor limit
and local result auditor remain unchanged. The recovery source/hash, original
error, corrected call and source import settings are embedded in
`supervisor_launch.json`, which the unchanged supervisor exports with results.
Independent return review must also bind that recovery source/hash to the local
transfer receipt. Source imports use a fresh bytecode prefix without writing
bytecode, preserving the caches left by the original failed attempt.

All bootstrap evidence stays under
`~/forensic-dgp/cctv_dgp_broader_codes_v16_preflight_recovery/<unique-attempt>/`.
If preflight stops before the supervisor can export an archive, collect that
evidence from **Windows Google Cloud SDK Shell**:

```bat
cd /d "C:\xampp\htdocs\YEAR 4\Testing\outputs"
gcloud compute scp --recurse --project=forensic-dgp-thesis --zone=us-central1-a "janusdominic0@forensic-dgp-thesis:/home/janusdominic0/forensic-dgp/cctv_dgp_broader_codes_v16_preflight_recovery" .
```

## Historical original upload/install instructions

These are retained to explain the original transfer and rejection. The original
launcher has the reproduced path bug. Do not invoke it again on the existing
directory; use the separate preflight recovery above.

Start the existing `forensic-dgp-thesis` VM and open its browser SSH terminal.
Windows commands below use the **Google Cloud SDK Shell (CMD)**. The user's
5 October2026 preference is separate SCP commands for each upload/download.

1. Upload from **Windows Google Cloud SDK Shell**:

   ```bat
   cd /d "C:\xampp\htdocs\YEAR 4\Testing\outputs"
   gcloud compute scp --project=forensic-dgp-thesis --zone=us-central1-a "cctv-dgp-broader-codes-v16-execution.tar.gz" "janusdominic0@forensic-dgp-thesis:/home/janusdominic0/"
   gcloud compute scp --project=forensic-dgp-thesis --zone=us-central1-a "cctv-dgp-broader-codes-v16-execution.tar.gz.sha256" "janusdominic0@forensic-dgp-thesis:/home/janusdominic0/"
   gcloud compute scp --project=forensic-dgp-thesis --zone=us-central1-a "launch_cctv_dgp_broader_codes_v16.py" "janusdominic0@forensic-dgp-thesis:/home/janusdominic0/"
   ```

2. Verify the uploaded archive in **VM SSH**:

   ```bash
   cd ~ &&
   sha256sum -c cctv-dgp-broader-codes-v16-execution.tar.gz.sha256
   ```

3. Install and launch **in detached tmux** from the normal **VM SSH prompt**:

   ```bash
   python3 ~/launch_cctv_dgp_broader_codes_v16.py \
     --protocol-sha 4331c28c97a659846eab76ee0dee9150811a00da3c6139d8d24b7905173a6681 \
     --archive-sha a40f68ccdeb7e7cf2e1420bef8c984a7faad6120461073da4362bcf2fb8ab65e
   ```

   This verified launcher performs extraction/preflight and opens tmux session
   `dgp_broader_codes_v16` itself. An existing outer tmux session would fail its
   competing-session check. No separate `tmux new-session` or historical normfix
   runner is needed. Run the launcher once; keep an existing/failed V16 root intact.

The bootstrap checks hostname, idle GPU/tmux, disk, transfer checksum, all116
frozen parent assets,6,195 data assets and2,804 baseline artifacts. It requires
the existing runtime and these previously prepared directories:

```text
~/forensic-dgp/cctv_dgp_vm_bundle/.venv/bin/python
~/forensic-dgp/cctv_dgp_face_code_fit_vm_v12_r2/
~/forensic-dgp/cctv_dgp_mixed_vm_v9_r2/
~/forensic-dgp/cctv_dgp_generalization_vm_v15/outputs/generalization_v15/
```

It creates a new `~/forensic-dgp/cctv_dgp_broader_codes_vm_v16/` with tmux session
`dgp_broader_codes_v16`. Existing/partial V16 roots, competing GPU/tmux tasks,
missing parents and runtime changes cause rejection. Preserve the rejection;
do not delete/rename a partial run or reinstall dependencies to bypass a guard.
Every graph/backward call requires the L4. Local CPU training is rejected.

4. Watch progress in **VM SSH**:

   ```bash
   tail -F ~/forensic-dgp/cctv_dgp_broader_codes_vm_v16/trainer.log ~/forensic-dgp/cctv_dgp_broader_codes_vm_v16/supervisor.log
   ```

   Wait for the supervisor's terminal receipt showing `"complete": true` and
   `"archive_sha256"`. Press Ctrl+C to close only the log viewer; tmux training
   continues. The trainer's own completion line precedes audit/export and is not
   the final download-ready receipt.

5. Download success files from **Windows Google Cloud SDK Shell** after that receipt:

   ```bat
   cd /d "C:\xampp\htdocs\YEAR 4\Testing\outputs"
   gcloud compute scp --project=forensic-dgp-thesis --zone=us-central1-a "janusdominic0@forensic-dgp-thesis:/home/janusdominic0/forensic-dgp/cctv_dgp_broader_codes_vm_v16/cctv-dgp-broader-codes-v16-results.tar.gz" .
   gcloud compute scp --project=forensic-dgp-thesis --zone=us-central1-a "janusdominic0@forensic-dgp-thesis:/home/janusdominic0/forensic-dgp/cctv_dgp_broader_codes_vm_v16/cctv-dgp-broader-codes-v16-results.tar.gz.sha256" .
   gcloud compute scp --project=forensic-dgp-thesis --zone=us-central1-a "janusdominic0@forensic-dgp-thesis:/home/janusdominic0/forensic-dgp/cctv_dgp_broader_codes_vm_v16/supervisor_completion.json" .
   ```

## Recover from an outer tmux preflight rejection

Location: `launch_cctv_dgp_broader_codes_v16.py`, initial `idle()` check.
Cause: `RuntimeError: Competing GPU/tmux task; do not launch` rejects any existing
tmux session or CUDA compute process. The reported attempt was inside tmux.
Fix: close that shell, verify idle state, and invoke the unchanged launcher from
normal VM SSH. This rejection occurred before extraction; it is not a failed
training run or a reason to change the frozen recipe.

1. In the tmux shell where the launch was rejected, close that shell:

   ```bash
   exit
   ```

   Detaching with Ctrl+B, D leaves the session alive and still triggers the guard.

2. From normal VM SSH, check remaining sessions:

   ```bash
   tmux list-sessions
   ```

   Expect no sessions (`no server running` or `no sessions`). If sessions remain,
   preserve them and provide the listing before proceeding.

3. Check CUDA compute processes:

   ```bash
   nvidia-smi --query-compute-apps=pid,process_name --format=csv,noheader
   ```

   Expect empty output. If processes are listed, preserve them and provide the
   output before proceeding.

4. When both checks are idle and V16 is already extracted, follow the preflight
   recovery section above from normal VM SSH. The recovery itself creates the
   detached training session after successful checks.

## Collect success or failure

Use the SDK Shell success-download commands above, or these **failure-download
commands** if the supervisor exported a failure instead:

```bat
cd /d "C:\xampp\htdocs\YEAR 4\Testing\outputs"
gcloud compute scp --project=forensic-dgp-thesis --zone=us-central1-a "janusdominic0@forensic-dgp-thesis:/home/janusdominic0/forensic-dgp/cctv_dgp_broader_codes_vm_v16/cctv-dgp-broader-codes-v16-failure.tar.gz" .
gcloud compute scp --project=forensic-dgp-thesis --zone=us-central1-a "janusdominic0@forensic-dgp-thesis:/home/janusdominic0/forensic-dgp/cctv_dgp_broader_codes_vm_v16/cctv-dgp-broader-codes-v16-failure.tar.gz.sha256" .
gcloud compute scp --project=forensic-dgp-thesis --zone=us-central1-a "janusdominic0@forensic-dgp-thesis:/home/janusdominic0/forensic-dgp/cctv_dgp_broader_codes_vm_v16/failure_export.json" .
```

Both sets save the three files in `C:\xampp\htdocs\YEAR 4\Testing\outputs\`,
preserving their filenames:

| Outcome | Files under `~/forensic-dgp/cctv_dgp_broader_codes_vm_v16/` |
| --- | --- |
| Success | `cctv-dgp-broader-codes-v16-results.tar.gz`, `.tar.gz.sha256`, `supervisor_completion.json` |
| Failure | `cctv-dgp-broader-codes-v16-failure.tar.gz`, `.tar.gz.sha256`, `failure_export.json` |

The terminal receipt is separate because it binds the finished archive checksum.
Check which exists with:

```bash
ls -l ~/forensic-dgp/cctv_dgp_broader_codes_vm_v16/supervisor_completion.json ~/forensic-dgp/cctv_dgp_broader_codes_vm_v16/failure_export.json
```

Exactly one is expected after export; `ls` reports the absent counterpart.
If neither exists, retain `supervisor.log`, `trainer.log` and any `failure.json`
for diagnosis. Do not repeat the unchanged recipe.

The assistant can import and independently audit returns locally. PowerShell
reproduction from the Windows project root:

```powershell
Set-Location -LiteralPath 'C:\xampp\htdocs\YEAR 4\Testing'
.\venv\Scripts\python.exe -X utf8 -u scripts/import_cctv_dgp_broader_codes_v16.py --archive outputs/cctv-dgp-broader-codes-v16-results.tar.gz --completion outputs/supervisor_completion.json --extract-to outputs/cctv_dgp_broader_codes_return_v16
```

For a failure:

```powershell
.\venv\Scripts\python.exe -X utf8 -u scripts/import_cctv_dgp_broader_codes_v16.py --archive outputs/cctv-dgp-broader-codes-v16-failure.tar.gz --completion outputs/failure_export.json --extract-to outputs/cctv_dgp_broader_codes_failure_return_v16
```

Import refuses overwrite, unsafe/duplicate paths and checksum mismatches.
The success auditor checks1,710 PNGs,300 raw previews,150 training code probes,
100 fresh-image parity cases,781 teacher arrays,3,128 traces/31,280 exposures and
600 grid cells. It rebuilds cosines and unchanged guards. Failure receives a
completed-prefix trace/scope audit. These check serialized evidence, not CUDA
gradients or recognizer replay. Full feature-cache content stays VM-side; the
local auditor checks its bindings and returned probes. Review all ten fixed
grids before claiming any output benefit; independent final human review remains
separate.

After return collection, inspect other work before stopping an instance started
solely for this experiment:

```bash
nvidia-smi --query-compute-apps=pid,process_name --format=csv,noheader
tmux list-sessions
```

Use Cloud Console Stop only when these show no competing work and collection is
complete. The supervisor stops its own timed-out child; it does not stop the
cloud instance or unrelated jobs. DGP-led app integration, covering-family
readiness and independent final thesis review remain unfinished. The Goal stays
active.
