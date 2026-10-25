# Completed diagnostic: audited zero-update return; do not rerun

23 October 2026. Worker exit0;72forwards,256queries,zero weight updates.
All575 tensors remain exactly originalu64. Independent archive,pixels,gradient,
state and12-sheet reviews passed. Measurement success is not model improvement.

[Return review](CCTV_CODEFORMER_U64_PNG_SURROGATE_MEASUREMENT_V1_RETURN_REVIEW.md).
[Next source-only design](CCTV_CODEFORMER_U64_PNG_CHECKED_STEP_V1_DESIGN.md).
The upload/tmux commands below are historical for the now-completed diagnostic.
Do not repeat them. No next runnable worker or manual transfer guide is ready.
U96 stays app incumbent;originalu64 remains retained. Full goal ACTIVE/incomplete.
Exact preceding guide bytes follow.

---

# Original trained u64: RAW/PNG signal measurement, zero updates

**Ready for one manual L4 diagnostic; not launched.** This measures why RAW
improvement failed to preserve the delivered PNG. It creates no new trained model.
The original trained u64 remains intact; U96 stays in the application.

[Audited update-two failure](CCTV_CODEFORMER_U64_GATE_ALIGNED_PILOT_V1_RETURN_REVIEW.md).
[Scientific design](CCTV_CODEFORMER_U64_PNG_SURROGATE_MEASUREMENT_V1_DESIGN.md).
The source-only status in the retained design describes its earlier proposal;
this guide records the newly sealed implementation. Independent archive and guide
readback must pass before this packet is listed as ready in PROJECT_HANDOFF.md.

Transfer: **243,674 bytes**. Archive SHA256:
`5585b5b36c4819e5a57b59a51fc4b8cc4d936bc8f364654722c494ad48469723`.
Protocol SHA256: `0a61c2d6433399e9e748f3782e4d87bf9da3089848b2502a761fcd029690797c`.

Require **8 GiB free** before launch and retain 4 GiB throughout. Keep the
completed parent `cctv_codeformer_u64_cleananchors_continuation_v1_vm`, its
469 static files and both weight files. All model weights and original-u64
parity witnesses are read from that parent. The failed pilot is not resumed.
Local audit aliases are not accepted on the VM. No VM connection or live-space
check was performed during preparation.

## 1. Upload from Windows Google Cloud SDK Shell

```bat
cd /d "C:\xampp\htdocs\YEAR 4\Testing\outputs"
gcloud compute scp --project=forensic-dgp-thesis --zone=us-central1-a "cctv-codeformer-u64-png-surrogate-measurement-v1-execution.tar.gz" "janusdominic0@forensic-dgp-thesis:/home/janusdominic0/"
gcloud compute scp --project=forensic-dgp-thesis --zone=us-central1-a "cctv-codeformer-u64-png-surrogate-measurement-v1-execution.tar.gz.sha256" "janusdominic0@forensic-dgp-thesis:/home/janusdominic0/"
```

## 2. Verify and install in VM SSH

```bash
cd ~ &&
sha256sum -c cctv-codeformer-u64-png-surrogate-measurement-v1-execution.tar.gz.sha256 &&
test -d ~/forensic-dgp/cctv_dgp_vm_bundle/.venv &&
test -d ~/forensic-dgp/cctv_codeformer_u64_cleananchors_continuation_v1_vm &&
test ! -e ~/forensic-dgp/cctv_codeformer_u64_png_surrogate_measurement_v1_vm &&
tar -xzf cctv-codeformer-u64-png-surrogate-measurement-v1-execution.tar.gz -C ~/forensic-dgp
```

The checksum file has Linux line endings. If a check fails, preserve its exact
message. Do not remove an existing or partial stage to force installation.

## 3. Open tmux in VM SSH

```bash
tmux new-session -A -s codeformer_u64_png_surrogate_v1
```

## 4. Launch once inside tmux

```bash
cd ~/forensic-dgp/cctv_codeformer_u64_png_surrogate_measurement_v1_vm &&
source ~/forensic-dgp/cctv_dgp_vm_bundle/.venv/bin/activate &&
python -B -u scripts/run_u64_png_surrogate_measurement_v1.py --root . --protocol-sha 0a61c2d6433399e9e748f3782e4d87bf9da3089848b2502a761fcd029690797c --verify-transfer &&
bash scripts/run_png_surrogate.sh 0a61c2d6433399e9e748f3782e4d87bf9da3089848b2502a761fcd029690797c
```

Detach with **Ctrl+B**, then **D**. Reattach using step 3. From a second SSH
terminal, inspect the existing worker without starting another:

```bash
tail -F ~/forensic-dgp/cctv_codeformer_u64_png_surrogate_measurement_v1_vm/run.log
```

The supervisor deliberately uses the completed parent's directory as a read-only
model-loading context; all new files go into the new diagnostic stage.

Fixed scope: **72 full forwards, 256 gradient queries, zero parameter
applications and zero optimizer constructions**. The same 16 TRAIN photographs
supply four matched roles. Four historical-u64 parity forwards and four same-state
repeats precede the 64 measured cases. Monitors, guards, native CCTV, wild and
FINAL images are excluded from gradient calculation.

Worker limit: **900 seconds**; external watchdog: **960 seconds**, plus at most
20 seconds cleanup. Every phase has a 180-second cap. Export runs separately:
300 seconds internally, 330 externally plus bounded cleanup. Exact runtime for
this route is unmeasured. Limits: 10 GiB RAM, 18 GiB reserved CUDA, 2 GiB outputs,
2 GiB return archive and the 4 GiB free-space reserve.

The PNG forward values must match the actual saved PNG decoded to FP32 exactly.
Its custom identity backward is explicitly a **biased surrogate**, not the true
PNG derivative or proof of finite-step improvement. All 575 tensors must remain
bitwise unchanged. All 256 gradient witnesses are retained for separate RAW/PNG
analysis of 16 pooled vectors; no common-direction solver or weight update runs.
The existing exporter, quality gates and failed states remain unchanged.

Final supervisor fields:

- `worker_exit: 0`: the measurement completed; independent returned-state,
  gradient and saved-pixel auditing is still required.
- `worker_exit` nonzero: execution, parity or resource failure; preserve the
  evidence and do not retry, resume or start a training run.
- `export_exit: 0`: the evidence archive was written; this does not qualify a
  model or prove restoration improvement.
- Unknown writer cleanup skips export, so download files can be absent.

## 5. Download from Windows Google Cloud SDK Shell

Wait for the supervisor's export result. Use one remote source per command;
Windows PuTTY does not support multiple remote sources in one invocation.

```bat
cd /d "C:\xampp\htdocs\YEAR 4\Testing\outputs"
gcloud compute scp --project=forensic-dgp-thesis --zone=us-central1-a "janusdominic0@forensic-dgp-thesis:/home/janusdominic0/cctv-codeformer-u64-png-surrogate-measurement-v1-results.tar.gz" "."
gcloud compute scp --project=forensic-dgp-thesis --zone=us-central1-a "janusdominic0@forensic-dgp-thesis:/home/janusdominic0/cctv-codeformer-u64-png-surrogate-measurement-v1-results.tar.gz.sha256" "."
gcloud compute scp --project=forensic-dgp-thesis --zone=us-central1-a "janusdominic0@forensic-dgp-thesis:/home/janusdominic0/cctv-codeformer-u64-png-surrogate-measurement-v1-export.json" "."
```

If export failed or was skipped, retain the logs and supervisor receipts; do not
rerun the measurement to manufacture an archive. After download, independently
check all gradients, exact PNG forward references, full575 initial/closing state,
RNG/sampler records, unchanged original-u64 parity and every planned output sheet.
No new training recipe is selected by a completed diagnostic.

Broader DEV failures remain binding. Native/wild evidence stays unpaired; FINAL
remains unopened and execution-unfrozen, with independent reviewers pending.
Protect/Apply/bypass preparation remains separate. Automatic matching and later
completion work remain deferred. All predecessors and failed states are retained.
The full goal remains **ACTIVE/incomplete**.
