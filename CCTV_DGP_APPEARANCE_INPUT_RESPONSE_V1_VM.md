# Appearance-input diagnostic completed; training still unqualified — 14 October 2026

The manual frozen-recognizer diagnostic completed in 77.61s plus 6.27s export:
40 forward batches, 200 RGB-input derivatives and zero optimizer/accepted
updates, restoration forwards, neural-weight displacements or epochs.
All 193 return / 1,246 stable response / 5,852 bank bindings pass. All 100
repeat-gradient checks pass the original bounds; maximum absolute difference
is 4.66e-10. Frozen weights/buffers and saved Torch CPU/CUDA RNG are preserved.
[Audited result](<C:/xampp/htdocs/YEAR 4/Testing/CCTV_DGP_APPEARANCE_INPUT_RESPONSE_V1_RESULTS.md>).

Finite RGB-input appearance responses still disagree in 9/6/3 of ten groups
from largest to smallest magnitude; saved double-dot reporting leaves the same
misses. Individual improvements/regressions cancel heavily in some groups.
This locates a disagreement at the recognizer input without uniquely proving
a backward bug. Earlier reconstruction-to-RGB behavior also contributes in
some cases. No new images, useful detail gain or candidate was produced.

USED/DIAGNOSTIC COMPLETE: do not rerun this returned packet. Next design:
compare a disposable full-recognizer precision copy and activation transitions
on ten exposed TRAIN cases before another optimizer/epoch schedule.
[Bounded next design; not yet a runnable packet](<C:/xampp/htdocs/YEAR 4/Testing/CCTV_DGP_RECOGNIZER_PRECISION_REVIEW_V1.md>).
All original RAW/PNG gates, previous failures, original checkpoints, splits,
caches, provenance, full states and app design remain. No more epochs qualified.
Training/derivatives stay human-manual L4 tmux; direct agent VM access remains
maintenance only. Native unpaired / paired TRAIN / reserved final roles,
all five milestones and seven automatic/assisted covering families remain.
Detector-confidence work remains prepared only and UNRUN. Goal incomplete.
Exact previous document follows.

# Local return-checker addendum — verified, 14 October 2026

Use [prospective local return auditor R1](<C:/xampp/htdocs/YEAR 4/Testing/scripts/audit_cctv_dgp_appearance_input_response_v1_return_r1.py>)
after importing this diagnostic's eventual return. A synthetic zero-reference
fixture caught an undefined relative-gradient ratio in the original shared
reporter. R1 reports that ratio as null while retaining elementwise1e-4/1e-8
and norm1e-4/1e-10 bounds; nonzero-reference reports are exactly original.
Both old checker and RED failure remain. NaN/Inf inputs and genuine absolute
bound violations are still rejected. Saved five-case replay passes unchanged.
[Exact source change and verification](<C:/xampp/htdocs/YEAR 4/Testing/outputs/cctv_dgp_appearance_input_response_v1_checker_addendum/audit.json>).

The VM archive,protocol,worker,200-query/40-forward limits,scientific image
gates and all five manual command blocks are unchanged. This is a local reporting
repair,not a training-recipe change or CUDA result. Diagnostic still UNRUN;
no model or more epochs qualified. Exact previous document follows.

# Frozen appearance-input response V1 — manual L4 commands

**Verified locally and UNRUN on the VM.** This diagnoses why the predicted
appearance changes differ from actual outputs. It trains no weights and
generates no new face images. Preserve the completed conditioning-response run.

Require **3 GiB free after installation**. Estimated measurement **2–5 minutes**,
or **3–7 minutes with export**, based on the previous returned workload.
The enforced external maximum is **12 minutes 30 seconds**, including both
termination graces. The VM's current free space and CUDA execution have not
been measured by this local preparation.

Packet: **406,879 bytes**, SHA256 `dea34f937f16808d3951267407dd791990306987da2cd80861e1450d2937caa5`.
Protocol: `3ba11275bc5680117ce40438b96987172fd0c772d70a049c7aca38a36f597f12`.
[Design, fixtures and exact limits](<C:/xampp/htdocs/YEAR 4/Testing/CCTV_DGP_APPEARANCE_INPUT_RESPONSE_V1_PREPARATION.md>).

1. Upload from **Windows Google Cloud SDK Shell**:

```bat
cd /d "C:\xampp\htdocs\YEAR 4\Testing\outputs"
gcloud compute scp --project=forensic-dgp-thesis --zone=us-central1-a "cctv-dgp-appearance-input-response-v1-execution.tar.gz" "janusdominic0@forensic-dgp-thesis:/home/janusdominic0/"
gcloud compute scp --project=forensic-dgp-thesis --zone=us-central1-a "cctv-dgp-appearance-input-response-v1-execution.tar.gz.sha256" "janusdominic0@forensic-dgp-thesis:/home/janusdominic0/"
```

2. Install in **VM SSH**:

```bash
cd ~ &&
sha256sum -c cctv-dgp-appearance-input-response-v1-execution.tar.gz.sha256 &&
test -d ~/forensic-dgp/cctv_dgp_vm_bundle/.venv &&
test -d ~/forensic-dgp/cctv_dgp_conditioning_response_v1_vm &&
test -d ~/forensic-dgp/cctv_dgp_bank_photometric_guard_v1_vm &&
test ! -e ~/forensic-dgp/cctv_dgp_appearance_input_response_v1_vm &&
tar -xzf cctv-dgp-appearance-input-response-v1-execution.tar.gz -C ~/forensic-dgp
```

3. Open **tmux**:

```bash
tmux new-session -A -s dgp_appearance_input_response_v1
```

4. Launch **inside tmux**:

```bash
cd ~/forensic-dgp/cctv_dgp_appearance_input_response_v1_vm &&
source ~/forensic-dgp/cctv_dgp_vm_bundle/.venv/bin/activate &&
python -B -u scripts/cctv_dgp_appearance_input_response_v1_vm.py --root . --response-root ../cctv_dgp_conditioning_response_v1_vm --bank-root ../cctv_dgp_bank_photometric_guard_v1_vm --protocol-sha 3ba11275bc5680117ce40438b96987172fd0c772d70a049c7aca38a36f597f12 --verify-transfer &&
bash scripts/run_input_response.sh 3ba11275bc5680117ce40438b96987172fd0c772d70a049c7aca38a36f597f12
```

The first receipt checks hashes only: 100 cases, at most 200 RGB-input
derivative queries and 40 recognizer forward batches; zero neural calls in
that receipt. The later measurement freezes every recognizer weight. Retain
any stop; do not delete outputs or repeat a used directory to bypass a guard.
The wrapper prints `review.log` when measurement ends. Detach with Ctrl+B,
release, then D. In a second VM SSH window, monitor with
`tail -F ~/forensic-dgp/cctv_dgp_appearance_input_response_v1_vm/review.log ~/forensic-dgp/cctv_dgp_appearance_input_response_v1_vm/export.log`.

5. Download from **Windows Google Cloud SDK Shell** after export finishes:

```bat
cd /d "C:\xampp\htdocs\YEAR 4\Testing\outputs"
gcloud compute scp --project=forensic-dgp-thesis --zone=us-central1-a "janusdominic0@forensic-dgp-thesis:/home/janusdominic0/cctv-dgp-appearance-input-response-v1-results.tar.gz" "."
gcloud compute scp --project=forensic-dgp-thesis --zone=us-central1-a "janusdominic0@forensic-dgp-thesis:/home/janusdominic0/cctv-dgp-appearance-input-response-v1-results.tar.gz.sha256" "."
gcloud compute scp --project=forensic-dgp-thesis --zone=us-central1-a "janusdominic0@forensic-dgp-thesis:/home/janusdominic0/cctv-dgp-appearance-input-response-v1-export.json" "."
```

Each download has one remote source, compatible with Windows PuTTY-backed SCP.
`complete: true` verifies export completion; the returned derivatives still
need independent audit. No new model, epoch continuation, native CCTV result
or covering-family result follows from this measurement alone.
