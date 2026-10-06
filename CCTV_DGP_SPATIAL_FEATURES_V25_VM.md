# V25 — upload, finite tmux run and download

**Latest research milestone — 6 October 2026: V25 audited failure; fixed-state gradient diagnostic verified, manual L4 measurement pending.**

The325,762,305-byte V25 return is verified and safely imported; independent replay
checks628 returned files/235 assets,100 raw/PNG pairs, both0/50 snapshots,50 original
DGP CPU forwards and250 frozen feature arrays. All original replay/quality bounds
stay unchanged. The trainer correctly stops at50 updates/51 backwards:
**0.0282235213% delivered degraded structure gain, below the required1%.** Final800
never runs. complete:true packages the retained failure; it does not accept a model.

All ten original-cell sheets/50 paired photographic TRAINING cases are reviewed;
200 exact cells are independently checked. No convincing whole-face gain is visible.
All26 spatial-head tensors change, all five projection gradients are active and
the stopped correction now reaches the output without the old final filter
attenuation. The typical degraded correction is still0.12994 of one byte level.
The saved corrected loss decreases through degraded cases, with a clear-preservation
cost; the V23 clear-reward mismatch does not explain this small saved improvement.
Scalar evidence does not establish GPU gradient competition or optimizer causality.

The next transfer is a **zero-update fixed-state gradient diagnostic**, not a new
training recipe or an unchanged V25 retry. It uses the two saved heads, same50
TRAIN cases, original seven objective terms and250 frozen own-DGP feature arrays.
The15,259-byte packet is independently checked:20 head batches/140 gradient calls/
120 recognizer forwards, zero DGP forwards/optimizer updates. Worker420s, external
480s plus30s grace; export30s internally/60s externally plus10s grace. Require1GiB
free disk and an idle L4; preserve any stop. Actual L4 gradients and independent
returned-matrix audit remain pending before choosing another training recipe.

Thirteen V25 auditor regressions and nine new diagnostic packet/source/matrix
guards pass. Python3.10/actual Windows rejection/Bash syntax/frozen mask and affine
geometry checks pass. Preparation failure evidence is retained. No local gradients,
backwards or optimization and no assistant VM/cloud action occur. The user-selected
own-DGP spatial/feature direction and all visible facial features together remain
in scope. Exact manual upload/install/tmux/launch/download steps are in the new
diagnostic runbook; PuTTY downloads remain three separate remote-source calls.

Original checkpoints/splits/failed gates, previous513 milestone bindings, app22
bindings and concurrent completed VM maintenance are preserved. The app and its
historical34 regressions/bundled inline Playwright checks remain unchanged. No
candidate is promoted. Native CCTV stays unpaired; paired TRAINING metrics remain
separate. No native/reserved-final/new covering pixels, ethnicity or Zamboanga
performance claims are introduced. The previously useful CCTV crop stays usable
despite model softness. Useful native output, candidate app parity/full flow,
insufficient-information handling, all seven automatic/assisted covering families
and independent final review remain required. Goal active/incomplete.

[V25 audited result](<C:/xampp/htdocs/YEAR 4/Testing/CCTV_DGP_SPATIAL_FEATURES_V25_RESULTS.md>) ·
[Finite diagnostic design](<C:/xampp/htdocs/YEAR 4/Testing/CCTV_DGP_V25_GRADIENT_DIAGNOSTIC_V1_PLAN.md>) ·
[Manual diagnostic commands](<C:/xampp/htdocs/YEAR 4/Testing/CCTV_DGP_V25_GRADIENT_DIAGNOSTIC_V1_VM.md>) ·
[Milestone](<C:/xampp/htdocs/YEAR 4/Testing/outputs/dgp_spatial_features_v25_audit_and_gradient_diagnostic_milestone/milestone.json>)

Previous bodies below are preserved history. Their pending-return/manual-V25-launch
statements are superseded by the audited V25 failure and diagnostic-only next step.
Old V22–V25 training commands are historical; do not repeat those failed recipes.


6 October 2026. Your selected own-DGP spatial path is packaged and independently
verified. Training/feature-gradient/timing/quality evidence is pending the manual
L4 run. Keep the V24 failed run and all original checkpoints.

Packet: **218,140,185 bytes (208.03 MiB), 237 regular files**.
Archive SHA256: `7652fa82a95d218de18c31d943114124bfaf9299e6492e8c79da710b59831034`.
Protocol SHA256: `ed43355b1fa0bd768d78e4e2b9ed3bd32cc046fe6221629605899a6a9b2e3175`.

1. Upload from **Windows Google Cloud SDK Shell**:

```cmd
cd /d "C:\xampp\htdocs\YEAR 4\Testing\outputs"
gcloud compute scp --project=forensic-dgp-thesis --zone=us-central1-a "cctv-dgp-spatial-features-v25-execution.tar.gz" "janusdominic0@forensic-dgp-thesis:/home/janusdominic0/"
gcloud compute scp --project=forensic-dgp-thesis --zone=us-central1-a "cctv-dgp-spatial-features-v25-execution.tar.gz.sha256" "janusdominic0@forensic-dgp-thesis:/home/janusdominic0/"
```

2. Install in **VM SSH**:

```bash
cd ~ &&
sha256sum -c cctv-dgp-spatial-features-v25-execution.tar.gz.sha256 &&
test -x ~/forensic-dgp/cctv_dgp_vm_bundle/.venv/bin/python &&
test ! -e ~/forensic-dgp/cctv_dgp_spatial_features_vm_v25 &&
tar -xzf cctv-dgp-spatial-features-v25-execution.tar.gz -C ~/forensic-dgp
```

3. Open **tmux**:

```bash
tmux new-session -A -s dgp_spatial_features_v25
```

4. Launch **inside tmux**:

```bash
cd ~/forensic-dgp/cctv_dgp_spatial_features_vm_v25 &&
source ~/forensic-dgp/cctv_dgp_vm_bundle/.venv/bin/activate &&
python -B -u scripts/cctv_dgp_spatial_features_v25_vm.py --root . --protocol-sha ed43355b1fa0bd768d78e4e2b9ed3bd32cc046fe6221629605899a6a9b2e3175 --preflight &&
bash scripts/run_v25.sh ed43355b1fa0bd768d78e4e2b9ed3bd32cc046fe6221629605899a6a9b2e3175
```

5. Download **after the export receipt appears**, from **Windows Google Cloud SDK Shell**:

```cmd
cd /d "C:\xampp\htdocs\YEAR 4\Testing\outputs"
gcloud compute scp --project=forensic-dgp-thesis --zone=us-central1-a "janusdominic0@forensic-dgp-thesis:/home/janusdominic0/cctv-dgp-spatial-features-v25-results.tar.gz" "."
gcloud compute scp --project=forensic-dgp-thesis --zone=us-central1-a "janusdominic0@forensic-dgp-thesis:/home/janusdominic0/cctv-dgp-spatial-features-v25-results.tar.gz.sha256" "."
gcloud compute scp --project=forensic-dgp-thesis --zone=us-central1-a "janusdominic0@forensic-dgp-thesis:/home/janusdominic0/cctv-dgp-spatial-features-v25-export.json" "."
```

Downloads go to **C:\xampp\htdocs\YEAR 4\Testing\outputs**. Keep three separate
download commands: Windows PuTTY rejects multiple remote sources in one call.

`Ctrl+B`, then `D` detaches while the pilot continues. Step3 reattaches. The
training log is ~/forensic-dgp/cctv_dgp_spatial_features_vm_v25/trainer.log.
Do not interrupt a running trainer with Ctrl+C; the script has finite stops.
If preflight fails, preserve/paste its error and the feature_preflight receipt;
the && chain prevents training. Do not replace that failed folder or rerun an
unchanged failed recipe. An idle tmux shell is permitted; another GPU process
is rejected without stopping it.

The run has at most800 updates/80 epochs; it stops at50 if delivered degraded
structure gain misses1%. At20 it checks the fitting-time projection. Preflight
is bounded at5 minutes, fitting25, worker30, supervisor35 plus30s grace, export
2 minutes internally/2.5 externally plus30s grace. The explicit preflight in
step4 precedes another preflight in the supervised run. At least3GiB free disk
and an idle L4 are required. Do not increase limits after a stop.

An export **complete:true** means the result archive was packaged. Read
run_results_present and failure_present: a packaged failure is not a training
or usefulness pass. Download partial/failure evidence too. Real returned outputs
will be independently audited and all50 cases reviewed before any broader run
or app promotion. Goal active/incomplete.

[Frozen design and limits](<C:/xampp/htdocs/YEAR 4/Testing/CCTV_DGP_SPATIAL_FEATURES_V25_PLAN.md>) ·
[Independent packet check](<C:/xampp/htdocs/YEAR 4/Testing/outputs/cctv_dgp_spatial_features_v25_preparation/independent_execution_audit.json>)
