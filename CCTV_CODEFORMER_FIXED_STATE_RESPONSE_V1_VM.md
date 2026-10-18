# Fixed-state return audited; direct restoration training being prepared — 17 October 2026

**CURRENT: 656-FORWARD MEASUREMENT COMPLETE / ZERO NEW UPDATES / NO QUALIFIED MODEL.**
[Returned evidence and all 24 reviewed sheets](CCTV_CODEFORMER_FIXED_STATE_RESPONSE_V1_RETURN_REVIEW.md).

All 2,536 return files and 528 saved RAW/PNG responses passed integrity and
independent pixel readback. Root reviewed all 24 sheets. The frozen final head
darkens all 264 TRAIN inputs; all 100 fitted dark and 32 unused-reference dark
cases worsen. This disconfirms a generalization-only explanation. The existing
trained composite remains research-only; no new checkpoint was made by this run.

The next preparation targets CodeFormer's actual restoration weights, with a
coherent full-spatial/token preservation objective on BOTH normal exposures,
full dark supervision and frozen-parent retention. Proposed finite first pilot:
50 restoration updates / 200 microbatches with fresh Adam, checkpoints and full
406-case DEV RAW/PNG review. **Not sealed or launched yet.** No more scalar-head
training, unchanged historical recipe, gate relaxation or app promotion is planned.

The full goal and manual L4/tmux boundary remain active. Untouched CodeFormer,
original DGP, trained failures, caches/splits/provenance/terms remain preserved;
native CCTV stays unpaired/source-separated, reserved FINAL separate. User
acceptance, independent FINAL and app Auto/override/PNG/Playwright verification
remain necessary. Separate covering/completion stays deferred. Storage offload
is already complete. Earlier status and commands below are historical only.

---

# Frozen CodeFormer response measurement: five manual steps

**PREPARED / UNRUN — 17 October 2026. Zero training updates.** [Purpose and scope](CCTV_CODEFORMER_FIXED_STATE_RESPONSE_V1_PREPARATION.md).

This measures the existing initial and trained brightness-controller states on the same 200 fitted observations and 64 matched observations from 32 unused TRAIN photographs. It answers whether selective brightness handling already fails on fitted inputs or changes on unused references. It does not train another model or authorize more epochs. Photograph IDs are not proven human identities. DEV and FINAL images are not used in this new measurement.

The upload is **4,909,256 bytes**, SHA-256:
`cb1fd0552665ad09a2ddfa59a1ab2ce5e9ae5746f72d2a1490a5091c1c2c1d1d`.
Protocol SHA-256:
`3ea4f1fe9ce90f3d29d29d6ace108eae7a809e6665cb770811f36773b888c089`.

Require **8 GiB free after installation**. The read-only parents `cctv_codeformer_tone_control_training_v1_vm` and `cctv_codeformer_adain_gradient_isolation_v1_vm` must remain under `~/forensic-dgp`. Preflight verifies 1,055 existing file bindings, including the original weights, initial/final composite states, fitted caches and runtime. It stops on missing or changed files; there is no automatic download, cleanup or retry. This preparation did not contact the VM or measure its current free space.

1. Upload from **Windows Google Cloud SDK Shell**:

```bat
cd /d "C:\xampp\htdocs\YEAR 4\Testing\outputs"
gcloud compute scp --project=forensic-dgp-thesis --zone=us-central1-a "cctv-codeformer-fixed-state-response-v1-execution.tar.gz" "janusdominic0@forensic-dgp-thesis:/home/janusdominic0/"
gcloud compute scp --project=forensic-dgp-thesis --zone=us-central1-a "cctv-codeformer-fixed-state-response-v1-execution.tar.gz.sha256" "janusdominic0@forensic-dgp-thesis:/home/janusdominic0/"
```

2. Install in **VM SSH**:

```bash
cd ~ &&
sha256sum -c cctv-codeformer-fixed-state-response-v1-execution.tar.gz.sha256 &&
test -d ~/forensic-dgp/cctv_dgp_vm_bundle/.venv &&
test -d ~/forensic-dgp/cctv_codeformer_tone_control_training_v1_vm &&
test -d ~/forensic-dgp/cctv_codeformer_adain_gradient_isolation_v1_vm &&
test ! -e ~/forensic-dgp/cctv_codeformer_fixed_state_response_v1_vm &&
tar -xzf cctv-codeformer-fixed-state-response-v1-execution.tar.gz -C ~/forensic-dgp
```

3. Open **tmux**:

```bash
tmux new-session -A -s codeformer_fixed_response_v1
```

4. Launch **inside tmux**, preserving this exact absolute-path invocation:

```bash
cd ~/forensic-dgp/cctv_codeformer_fixed_state_response_v1_vm &&
source ~/forensic-dgp/cctv_dgp_vm_bundle/.venv/bin/activate &&
TASK_ROOT="$(pwd -P)" &&
PYTHON_VM="$(python -c 'import sys; print(sys.executable)')" &&
"$PYTHON_VM" -B -u "$TASK_ROOT/scripts/supervise_codeformer_fixed_state_response_v1_vm.py" --root "$TASK_ROOT" --protocol-sha 3ea4f1fe9ce90f3d29d29d6ace108eae7a809e6665cb770811f36773b888c089
```

The supervisor performs transfer preflight, the frozen measurement and export. Maximum **656 neural forwards / zero optimizer updates**. Estimated measurement time **4–10 minutes**, unmeasured for this packet; the worker stops at **900 seconds**. Preflight and export each have a **300-second** cap, with **30-second** termination grace per stage. Phase limits are 300 seconds for data, 300 for initialization and 600 for measurement, all within the worker's 900 seconds. Detach using **Ctrl+B, then D**. Existing workloads are preserved; a conflict stops this launch.

5. Download the **four return files** from **Windows Google Cloud SDK Shell**, after the supervisor returns to the prompt:

```bat
cd /d "C:\xampp\htdocs\YEAR 4\Testing\outputs"
gcloud compute scp --project=forensic-dgp-thesis --zone=us-central1-a "janusdominic0@forensic-dgp-thesis:/home/janusdominic0/cctv-codeformer-fixed-state-response-v1-results.tar.gz" "."
gcloud compute scp --project=forensic-dgp-thesis --zone=us-central1-a "janusdominic0@forensic-dgp-thesis:/home/janusdominic0/cctv-codeformer-fixed-state-response-v1-results.tar.gz.sha256" "."
gcloud compute scp --project=forensic-dgp-thesis --zone=us-central1-a "janusdominic0@forensic-dgp-thesis:/home/janusdominic0/cctv-codeformer-fixed-state-response-v1-export.json" "."
gcloud compute scp --project=forensic-dgp-thesis --zone=us-central1-a "janusdominic0@forensic-dgp-thesis:/home/janusdominic0/cctv-codeformer-fixed-state-response-v1-supervisor-final.json" "."
```

Use one remote source per command because Windows PuTTY rejects multiple remote sources. If preflight stops before export, preserve the terminal output and `outputs/supervision/preflight.log`. If another stage stops, preserve all partial output and logs. Do not erase the new folder or relaunch it unchanged.

Optional monitoring from another **VM SSH** terminal:

```bash
tail -F ~/forensic-dgp/cctv_codeformer_fixed_state_response_v1_vm/outputs/supervision/diagnostic.log ~/forensic-dgp/cctv_codeformer_fixed_state_response_v1_vm/outputs/supervision/export.log
```

After return, independently verify both frozen states, exact counters, every RAW/PNG response, per-reference residuals and output sheets. A completed measurement is not a qualified restorer or evidence that more epochs will help. The failed tone-controller composite stays research-only. Existing restoration, clearer-input handling, user acceptance, independent FINAL review and app checks remain required by the full goal.
