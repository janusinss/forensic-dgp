# Precision diagnostic complete; nearby-gradient training design — 14 October 2026

Recognizer precision V1 completed in 48.69s plus 2.98s export. All 462 return,
192 stable input, 1,246 response and 5,852 bank bindings pass. All 30 original
gradient-parity checks pass; frozen states/buffers and saved Torch CPU/CUDA RNG
are preserved. Zero restorer forwards, learned-weight/optimizer/accepted updates
or epochs. [Audited result](<C:/xampp/htdocs/YEAR 4/Testing/CCTV_DGP_RECOGNIZER_PRECISION_V1_RESULTS.md>).

Full FP64 changes sampled appearance misses from six to five of ten. It does
not resolve the two selected degraded failure cases. Both precisions show
PReLU region transitions at the signed endpoints; initial repeats show none.
No new restored images or structure gain. Original RAW/PNG gate failures stand.
USED/DIAGNOSTIC COMPLETE: do not rerun the precision packet.

Next design: collect derivatives at initial and two already measured nearby
states, retain all loss constraints, and require actual RAW/PNG proposal gates.
One solve / at most three proposals / one accepted research update; no epochs.
[Finite changed learning design; not yet a verified transfer](<C:/xampp/htdocs/YEAR 4/Testing/CCTV_DGP_NEARBY_GRADIENT_LEARNING_REVIEW_V1.md>).
Packet, regressions and return checker remain to prepare. Training/derivatives
and parameter changes stay human-manual L4 tmux; agent VM access stays maintenance
only. No new training or app change was launched.

All original checkpoints, splits, caches, provenance, full states and failures
remain. No new incumbent, model promotion or more epochs qualify. Native
unpaired / paired TRAIN / reserved final roles, all five milestones and seven
automatic/assisted covering families remain. Detector-confidence recording
remains prepared only and UNRUN. Goal incomplete. Exact previous document follows.

# Recognizer precision V1 — manual VM commands

**Verified and UNRUN. This is a computation diagnostic, not more restoration
training.** Appearance-input V1 completed and its gradients repeated, but finite
appearance predictions still disagreed in 9/6/3 groups. This check compares the
entire frozen recognizer in FP32 and FP64 on two problematic references and all
five profiles: ten already exposed TRAIN cases.

Require **3 GiB free after installation**. Estimate **2–6 minutes measurement**,
**3–8 minutes including export**; the external ceilings total **12 minutes
30 seconds**. FP64 speed has not been measured here; an early projection and
finite stops protect the run. Keep the existing input-response, conditioning-
response and brightness-guard VM directories intact. No cleanup is included.
An insufficient-space or competing-workload stop must remain retained.

1. Upload from **Windows Google Cloud SDK Shell**:

```bat
cd /d "C:\xampp\htdocs\YEAR 4\Testing\outputs"
gcloud compute scp --project=forensic-dgp-thesis --zone=us-central1-a "cctv-dgp-recognizer-precision-v1-execution.tar.gz" "janusdominic0@forensic-dgp-thesis:/home/janusdominic0/"
gcloud compute scp --project=forensic-dgp-thesis --zone=us-central1-a "cctv-dgp-recognizer-precision-v1-execution.tar.gz.sha256" "janusdominic0@forensic-dgp-thesis:/home/janusdominic0/"
```

2. Install in **VM SSH**:

```bash
cd ~ &&
sha256sum -c cctv-dgp-recognizer-precision-v1-execution.tar.gz.sha256 &&
test -d ~/forensic-dgp/cctv_dgp_vm_bundle/.venv &&
test ! -e ~/forensic-dgp/cctv_dgp_recognizer_precision_v1_vm &&
tar -xzf cctv-dgp-recognizer-precision-v1-execution.tar.gz -C ~/forensic-dgp
```

3. Open **tmux**:

```bash
tmux new-session -A -s dgp_recognizer_precision_v1
```

4. Launch **inside tmux**:

```bash
cd ~/forensic-dgp/cctv_dgp_recognizer_precision_v1_vm &&
source ~/forensic-dgp/cctv_dgp_vm_bundle/.venv/bin/activate &&
python -B -u scripts/cctv_dgp_recognizer_precision_v1_vm.py --root . --input-root ../cctv_dgp_appearance_input_response_v1_vm --response-root ../cctv_dgp_conditioning_response_v1_vm --bank-root ../cctv_dgp_bank_photometric_guard_v1_vm --protocol-sha ca8d2fd69dffb19e040a36eb059e4f4124f00f05755cad976d24f8a00f4dbbe7 --verify-transfer &&
bash scripts/run_precision.sh ca8d2fd69dffb19e040a36eb059e4f4124f00f05755cad976d24f8a00f4dbbe7
```

The wrapper retains `review.log`, exit code and an export after success or
failure. Detach with Ctrl+B, then D. This does not stop the run. Preserve the
logs and downloaded archive even if a stop occurs; do not edit or rerun it.
`complete: true` describes export and does not imply trained-model improvement.

5. Download to **Windows Google Cloud SDK Shell** after export finishes:

```bat
cd /d "C:\xampp\htdocs\YEAR 4\Testing\outputs"
gcloud compute scp --project=forensic-dgp-thesis --zone=us-central1-a "janusdominic0@forensic-dgp-thesis:/home/janusdominic0/cctv-dgp-recognizer-precision-v1-results.tar.gz" "."
gcloud compute scp --project=forensic-dgp-thesis --zone=us-central1-a "janusdominic0@forensic-dgp-thesis:/home/janusdominic0/cctv-dgp-recognizer-precision-v1-results.tar.gz.sha256" "."
gcloud compute scp --project=forensic-dgp-thesis --zone=us-central1-a "janusdominic0@forensic-dgp-thesis:/home/janusdominic0/cctv-dgp-recognizer-precision-v1-export.json" "."
```

Each download has one remote source, as required by Windows PuTTY. Return all
three files in the local `outputs` directory for an independent audit.

Protocol SHA256: `ca8d2fd69dffb19e040a36eb059e4f4124f00f05755cad976d24f8a00f4dbbe7`.
Execution archive: **460,140 bytes**, SHA256
`8d4f80ff2a54597fb4b11addfbae6b56beaac26cb9927566c6ea671eac7d42ef`. Checksum and shell script are ASCII/LF-compatible.
[Scope, limits and preparation proof](<C:/xampp/htdocs/YEAR 4/Testing/CCTV_DGP_RECOGNIZER_PRECISION_V1_PREPARATION.md>).

Maximum 16 recognizer forward attempts and 40 RGB-input gradient queries;
zero restoration forwards, learned-weight/optimizer/accepted updates or epochs.
The original recognizer and checkpoint remain intact. FP64 uses a disposable
precision copy, not new learned weights. Official FP32/PNG scientific scores
and every quality gate remain unchanged. No new restored image, native CCTV
or reserved-final identity is used. No automatic next training run follows.
The full goal and seven automatic/assisted covering families remain incomplete.
