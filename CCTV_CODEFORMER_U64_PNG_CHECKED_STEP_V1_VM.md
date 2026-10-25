# Completed failed experiment: do not rerun this guide

23 October 2026. The manual run reached its final monitor screen and stopped
with worker exit 2. Archive, state, saved-pixel and all 21-sheet audits are
complete; localized-blur PNG preservation failed. No qualified successor.

[Return review](CCTV_CODEFORMER_U64_PNG_CHECKED_STEP_V1_RETURN_REVIEW.md).
[Diagnostic matrix](CCTV_CODEFORMER_U64_POST_CHECKED_STEP_DIAGNOSTIC_MATRIX_V1.md).
U96 remains app incumbent; original u64 remains retained. A diagnostic choice
is pending under AGENTS.md after three failed preservation recipes. No next
runnable worker or upload guide is ready. The commands below are historical:
do not resume, add epochs or rerun this completed recipe.
Exact preceding guide bytes follow.

---

# Original trained u64: one finite PNG-checked step

**Prepared for one manual L4 test; not launched.** This continues original trained
u64 by choosing a fixed trial using actual TRAIN RAW/PNG results. It tests a
finite-step selection change, with all original preservation gates retained.
A completed or passing screen does not qualify a new restoration model.

[Audited diagnostic](CCTV_CODEFORMER_U64_PNG_SURROGATE_MEASUREMENT_V1_RETURN_REVIEW.md).
[Finite-step design](CCTV_CODEFORMER_U64_PNG_CHECKED_STEP_V1_DESIGN.md).
The source-only status in the retained design records its earlier proposal;
this guide describes the newly sealed implementation. Independent archive and
guide reviews must pass before PROJECT_HANDOFF.md lists this packet as ready.

Transfer: **798,435 bytes**. Archive SHA256:
`15ef783122ea90259aa8fc7fd86c912cb88bf3a425b2c85a1a86a41d0db126e7`.
Protocol SHA256: `7f4546d4d911bea7100f42870514f08a42e349ec524a2b9795f8f3c7f55dcf60`.

Require **8 GiB free** before launch; retain 4 GiB throughout. Keep these two
completed VM stages in place:

- `cctv_codeformer_u64_cleananchors_continuation_v1_vm`: all 469 static files,
  original trained-u64 head, R2 and U96 weights, data and control witnesses.
- `cctv_codeformer_u64_png_surrogate_measurement_v1_vm`: protocol, 21 sealed
  static files and 267 required runtime witnesses, including the 128 RAW gradient
  files. The new packet reuses them read-only; no new autograd queries run.

Both stages remain read-only. Local extraction/model aliases used for auditing
are not accepted as substitute paths on the VM. Do not remove these dependencies.
No VM connection or current free-space check was performed during preparation.

## 1. Upload from Windows Google Cloud SDK Shell

```bat
cd /d "C:\xampp\htdocs\YEAR 4\Testing\outputs"
gcloud compute scp --project=forensic-dgp-thesis --zone=us-central1-a "cctv-codeformer-u64-png-checked-step-v1-execution.tar.gz" "janusdominic0@forensic-dgp-thesis:/home/janusdominic0/"
gcloud compute scp --project=forensic-dgp-thesis --zone=us-central1-a "cctv-codeformer-u64-png-checked-step-v1-execution.tar.gz.sha256" "janusdominic0@forensic-dgp-thesis:/home/janusdominic0/"
```

## 2. Verify and install in VM SSH

```bash
cd ~ &&
sha256sum -c cctv-codeformer-u64-png-checked-step-v1-execution.tar.gz.sha256 &&
test -d ~/forensic-dgp/cctv_dgp_vm_bundle/.venv &&
test -d ~/forensic-dgp/cctv_codeformer_u64_cleananchors_continuation_v1_vm &&
test -d ~/forensic-dgp/cctv_codeformer_u64_png_surrogate_measurement_v1_vm &&
test ! -e ~/forensic-dgp/cctv_codeformer_u64_png_checked_step_v1_vm &&
tar -xzf cctv-codeformer-u64-png-checked-step-v1-execution.tar.gz -C ~/forensic-dgp
```

The checksum file uses Linux line endings. Preserve any failure message; do not
delete a previous or partially installed stage to force these checks to pass.

## 3. Open tmux in VM SSH

```bash
tmux new-session -A -s codeformer_u64_png_checked_v1
```

## 4. Launch once inside tmux

```bash
cd ~/forensic-dgp/cctv_codeformer_u64_png_checked_step_v1_vm &&
source ~/forensic-dgp/cctv_dgp_vm_bundle/.venv/bin/activate &&
python -B -u scripts/run_u64_png_checked_step_v1.py --root . --protocol-sha 7f4546d4d911bea7100f42870514f08a42e349ec524a2b9795f8f3c7f55dcf60 --verify-transfer &&
bash scripts/run_png_checked.sh 7f4546d4d911bea7100f42870514f08a42e349ec524a2b9795f8f3c7f55dcf60
```

The verification checks actual parent and measurement files before constructing
a model. The supervisor deliberately uses the original parent's directory as a
read-only model-loading context; all new artifacts go into the new stage.

Detach with **Ctrl+B**, then **D**. Reattach using step 3. From another SSH
terminal, follow the journal without starting another worker:

```bash
tail -F ~/forensic-dgp/cctv_codeformer_u64_png_checked_step_v1_vm/outputs/u64_png_checked_step_v1/events.jsonl
```

The initial bank and baseline can be quiet in the training terminal; trial-action
messages start after the initial 124 forwards. The journal records each operation.

Estimate **8–16 minutes**, unmeasured for this route. Worker stop: **1200 seconds**;
external watchdog: **1260 seconds**, plus at most 20 seconds cleanup. Every phase
has a 180-second cap. Export is separate: 300 seconds internally, 330 externally
plus bounded cleanup. Limits: 10 GiB RAM, 18 GiB reserved CUDA, 2 GiB new outputs,
2 GiB return archive and the 4 GiB free-space reserve. No automatic retry/resume.

The complete finite budget is **436 full forwards, zero new gradient queries,
zero optimizer constructions and nine explicit parameter applications**:
four trial writes, four restores to original u64, and one selected final commit.
At most **one selected training update** results; the nine writes are not epochs.
Every trial starts independently from original u64. The fixed maximum coordinate
bounds are **2e-5, 1.5e-5, 1e-5, 5e-6**, with fixed `.99` numerical headroom.
Actual rounded writes preserve the old 2.5e-5 per-application and 1e-4 cumulative
bounds; all 539 R2/U96 tensors stay exact. Every copy and restore is journaled.

All four complete 64-case optimization TRAIN banks are saved before selection.
The first eligible candidate in the registered order must preserve all 16 actual
RAW/PNG means and strictly improve both blur PNG adjacent means. Selection
receives no monitor/guard records. If none qualifies, stop after **380 forwards
and eight applications**, with original u64 restored. A selected candidate gets
one 56-case monitor/guard evaluation, including inherited 1% structure and all
preservation requirements. A monitor failure rejects the entire test; no next
candidate is tried. Invariant, numerical, copy or resource failures stop
immediately. A partial write stays unknown; no restore hides that outcome.

Final supervisor fields:

- `worker_exit: 0`: the selected finite TRAIN screen passed; returned outputs,
  states and all quality checks still need independent audit and visual review.
- `worker_exit: 2`: no candidate qualified or the selected monitor screen failed.
  Download the retained evidence. Do not resume or append updates.
- `worker_exit: 1` or another nonzero value: execution/resource/invariant failure;
  retain logs and states without relaunching.
- `export_exit: 0`: evidence was exported; this does not establish model quality.
  Unknown writer cleanup can skip export, leaving download files absent.

## 5. Download from Windows Google Cloud SDK Shell

Wait for the supervisor's export result. Use **one remote source per command**;
Windows PuTTY rejects multiple remote sources in one invocation.

```bat
cd /d "C:\xampp\htdocs\YEAR 4\Testing\outputs"
gcloud compute scp --project=forensic-dgp-thesis --zone=us-central1-a "janusdominic0@forensic-dgp-thesis:/home/janusdominic0/cctv-codeformer-u64-png-checked-step-v1-results.tar.gz" "."
gcloud compute scp --project=forensic-dgp-thesis --zone=us-central1-a "janusdominic0@forensic-dgp-thesis:/home/janusdominic0/cctv-codeformer-u64-png-checked-step-v1-results.tar.gz.sha256" "."
gcloud compute scp --project=forensic-dgp-thesis --zone=us-central1-a "janusdominic0@forensic-dgp-thesis:/home/janusdominic0/cctv-codeformer-u64-png-checked-step-v1-export.json" "."
```

If export failed or was skipped, retain logs and supervisor records; do not rerun
the test to manufacture an archive. Return review must authenticate actual
trial/restore/commit values, all saved RAW/PNG metrics, individual regressions,
selection membership/order, the separate final screen, frozen tensors, RNG/sampler,
resource records, operation accounting and every planned visual sheet.

U96 stays experimental app incumbent; original u64 and all predecessor/failure
states stay retained. This screen does not qualify native CCTV, wild images,
FINAL or Zamboanga performance. Broader DEV failures remain binding. Native/wild
stay unpaired; FINAL is unopened and execution-unfrozen, independent reviewers
pending. Reviewed Protect/Apply/bypass remains separate. Automatic matching and
later completion development remain deferred. Full goal **ACTIVE/incomplete**.
