# Executed and stopped: do not run this historical training guide again

The cleananchors continuation completed eight new updates and failed preservation.
Its retained quality stop is binding; do not resume, add updates or select an
earlier stopped state. For the new zero-update diagnostic, use the
[current manual guide](CCTV_CODEFORMER_U64_ROLE_GRADIENTS_V1_VM.md).
The preparation-time guide below is preserved as historical evidence.

---

# Continue our trained u64: five manual VM steps

**Prepared and unrun.** This experiment starts from the exact trained u64 that
achieved 1.627216% global and 1.210135% local PNG structure gain in the small TRAIN
screen. It retains its 36 learned heads, R2 base and U96 adapters. U96 remains
the application incumbent; both models and all prior failures stay preserved.

The change is photographic clean-target supervision for the clear/dark anchor
terms. Architecture, anchor coefficients and existing quality gates stay fixed.
This is a new finite protocol, not resuming the old closed 64-update experiment.
No automatic launch, retry, extra epochs or promotion is configured.

Archive: `cctv-codeformer-u64-cleananchors-continuation-v1-execution.tar.gz`, **89,566,574 bytes**.
Archive SHA256: `c8d7e22386ec24062072c089f065b20fc78b5d1eb3f64f4df49fca51ed734cd5`.
Protocol SHA256: `4db752a04222210a9fbccf176cf422c5fadeef443d39a15ae7b62086e9c41419`.

At most **16 new Adam updates**, four microbatches each: one pass through 64
correlated optimization rolecases, not 16 epochs. Same 36 trainable head tensors /
168,960 scalars; all 539 R2/U96 tensors remain frozen. One fresh Adam uses maximum
LR 0.0001 with four-update warmup, betas (0.9, 0.99) and norm-1 clipping. Old
optimizer/RNG/sampler states remain retained rather than resumed.

The maximum is **292 full forwards and 64 backwards**. There are 56 initial
monitor/guard calls, four determinism repeats, three 56-case banks at updates
4/8/16 and 64 training calls. Saved U96 controls add zero model calls. Initial
RAW/PNG must match the saved trained-u64 witnesses; the four repeat calls check
the same candidate, not a separate reference implementation. Historical native/
logits/LQ parity is not claimed where those old arrays are absent.

Planning estimate: **6–12 worker minutes**, unmeasured for this path. Scientific
work stops internally at 1,200 seconds; externally at 1,260 seconds plus up to
20 seconds termination grace. Whole phases stop at 180 seconds. Failure capture
has at most 30 seconds, followed by at most 5 for its receipt and 5 for caller-RNG
restoration inside the external budget when the internal stop returns promptly.
Native code that does not yield is subject to external termination. Export is
separate: 300 seconds internally / 330 externally.

Require **8 GiB free after installation**, with a continuing 4 GiB reserve. The
installer additionally needs approximately 377 MB for two exclusive initializer
copies. Limits: 10 GiB RSS, 18 GiB CUDA reserved, 2 GiB new output and 2 GiB return
archive. Do not run another GPU workload. No fresh VM inventory or connection was
performed while preparing this guide.

## 1. Upload from Windows Google Cloud SDK Shell

```bat
cd /d "C:\xampp\htdocs\YEAR 4\Testing\outputs"
gcloud compute scp --project=forensic-dgp-thesis --zone=us-central1-a "cctv-codeformer-u64-cleananchors-continuation-v1-execution.tar.gz" "janusdominic0@forensic-dgp-thesis:/home/janusdominic0/"
gcloud compute scp --project=forensic-dgp-thesis --zone=us-central1-a "cctv-codeformer-u64-cleananchors-continuation-v1-execution.tar.gz.sha256" "janusdominic0@forensic-dgp-thesis:/home/janusdominic0/"
```

## 2. Check and install in VM SSH

The package already contains the trained u64 heads and saved comparison controls.
The installer copies and rehashes only the retained R2/U96 files. It does not run
the historical smoothanchors-LR pilot. If any check fails, stop and preserve the
output and partial files; do not delete a partial installation to retry.

```bash
cd ~ &&
sha256sum -c cctv-codeformer-u64-cleananchors-continuation-v1-execution.tar.gz.sha256 &&
test -d ~/forensic-dgp/cctv_dgp_vm_bundle/.venv &&
test -f ~/forensic-dgp/cctv_codeformer_u96_postfusion_smoothanchors_lr_v1_vm/weights/codeformer_r2.pth &&
test -f ~/forensic-dgp/cctv_codeformer_u96_postfusion_smoothanchors_lr_v1_vm/weights/codeformer_u96_adapter.pth &&
test ! -e ~/forensic-dgp/cctv_codeformer_u64_cleananchors_continuation_v1_vm &&
tar -xzf cctv-codeformer-u64-cleananchors-continuation-v1-execution.tar.gz -C ~/forensic-dgp &&
cd ~/forensic-dgp/cctv_codeformer_u64_cleananchors_continuation_v1_vm &&
python3 -B research/codeformer_u64_cleananchors_continuation_v1/install_weights.py --root . --source-root ../cctv_codeformer_u96_postfusion_smoothanchors_lr_v1_vm --protocol-sha 4db752a04222210a9fbccf176cf422c5fadeef443d39a15ae7b62086e9c41419
```

Expected: installer `"complete": true`, two exact copies and zero neural/training
calls. Preserve the separate migration backup; this small execution archive
deliberately omits the two large R2/U96 source files.

## 3. Open tmux

```bash
tmux new-session -A -s u64_cleananchors_continuation_v1
```

## 4. Launch inside tmux

```bash
cd ~/forensic-dgp/cctv_codeformer_u64_cleananchors_continuation_v1_vm &&
source ~/forensic-dgp/cctv_dgp_vm_bundle/.venv/bin/activate &&
python -B -u scripts/run_u64_cleananchors_continuation_v1.py --root . --protocol-sha 4db752a04222210a9fbccf176cf422c5fadeef443d39a15ae7b62086e9c41419 --verify-transfer &&
bash scripts/run_continuation.sh 4db752a04222210a9fbccf176cf422c5fadeef443d39a15ae7b62086e9c41419
```

The static receipt verifies 469 file bindings, 96 retained paired TRAIN
cases, 64 optimization cases, 32 adaptive monitors, 24 legacy guards, 56 U96
controls and four u64 witnesses. It makes zero model calls. Actual parity,
gradients and model quality remain unverified until the guarded VM run.

The supervisor prints its child PID and log path. In another VM SSH window:

```bash
tail -F ~/forensic-dgp/cctv_codeformer_u64_cleananchors_continuation_v1_vm/run.log
```

Detach tmux with **Ctrl+B**, then **D**. Reattach with step 3. Do not launch again
while it runs. At update 8, clear target MAE must improve and dark MAE must not
worsen in RAW and PNG. At terminal 16, all four existing u64 structure gains
must be retained, clear/dark target MAE must improve, blur target MAE must not
worsen, and every original preservation/1% requirement must pass. No favorable
earlier checkpoint is selected after a stop.

Forward settings remain strict deterministic FP32. Each reserved backward records
the installed Torch CUDA bilinear-upsample warning in a temporary warning-only
scope and restores strict flags. Unexpected nondeterministic backward operations
stop. Bitwise CUDA gradient repeatability is not claimed.

Normal completion requires complete initial/closing 575-value witnesses, every
committed head/Adam/RNG/sampler state, gradients and all RAW/PNG records. On
failure, existing/partial bytes remain; additional capture is best effort within
time/output caps. Unknown operations stay unknown.

The final supervisor receipt reports `worker_exit`, `export_exit`,
`export_attempted` and `active_writer_outcome_unknown`. Worker exit 2 is a retained
quality stop. Worker exit 0 is a finite TRAIN-screen result, not qualification.
Export exit 0 confirms transport only. Known worker stops still export once
cleanup is confirmed; unknown cleanup skips export. Never rerun the experiment
or overwrite its partial export to force a success receipt.

## 5. Download from Windows Google Cloud SDK Shell

Wait for export completion. Windows PuTTY requires separate remote-source
commands, so download each file individually:

```bat
cd /d "C:\xampp\htdocs\YEAR 4\Testing\outputs"
gcloud compute scp --project=forensic-dgp-thesis --zone=us-central1-a "janusdominic0@forensic-dgp-thesis:/home/janusdominic0/cctv-codeformer-u64-cleananchors-continuation-v1-results.tar.gz" "."
gcloud compute scp --project=forensic-dgp-thesis --zone=us-central1-a "janusdominic0@forensic-dgp-thesis:/home/janusdominic0/cctv-codeformer-u64-cleananchors-continuation-v1-results.tar.gz.sha256" "."
gcloud compute scp --project=forensic-dgp-thesis --zone=us-central1-a "janusdominic0@forensic-dgp-thesis:/home/janusdominic0/cctv-codeformer-u64-cleananchors-continuation-v1-export.json" "."
```

If export fails or is skipped, preserve its terminal output, supervisor/status/
journal files and `export_failure.json` if present for inspection. Do not repeat
training, resume a stopped state or discard the failure evidence.

The named project/instance/zone/user is the existing user-identified L4 target;
this guide has not connected, uploaded, installed or trained anything. After
download, independently audit all returned states, Adam arithmetic, RAW/PNG and
gates, and personally review the 31 planned sheets or every available partial
output. Native/wild remain unpaired; TRAIN success does not erase broader
preservation failures. Wider DEV, visible usefulness, user acceptance, independent
FINAL and full app checks remain required before any promotion. Completion work
stays deferred. Earlier instruction snapshots inside this sealed package record
preparation-time readiness; later publication supersedes that historical status.
