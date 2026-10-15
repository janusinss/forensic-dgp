# Nearby-gradient V1 R1 audited and rejected — 14 October 2026

The manual L4 run completed its finite computations, but all three proposals
failed actual preservation. Best structure gain 0.687495% RAW / 0.706917% PNG,
below 1%, with 32 RAW / 32 PNG failed group–metric checks. Smaller trials fail
24/25 and 14/9. Zero accepted updates, full epochs or new incumbent.
[Result, cause of stop and full evidence](<C:/xampp/htdocs/YEAR 4/Testing/CCTV_DGP_NEARBY_GRADIENT_LEARNING_V1_R1_RESULTS.md>).

All 1,144 return / 5,852 bank / 1,246 response bindings, seven snapshots,
700 RAW / 700 PNG outputs, 1,400 metric rows, all 96 local derivative constraints
and actual gate decisions pass saved-evidence auditing. One original
elementwise cross-run derivative bound still fails and remains reported;
all32 vector norms pass. Audit completion is not a model-quality pass.
All20 sheets/900 cells were directly reviewed; no convincing useful new clarity.
Initial/rollback outputs and saved RNG match; original/protected hashes remain.

New direction cosine with the failed RAW-only direction is 0.999539308.
Certified nearby descent has not established finite preserving improvement.
Joint, RAW-only and nearby learning attempts now require a diagnostic question
under AGENTS.md. Question sent; human answer pending. Stop training-code
changes while continuing completed evidence preservation and publication.
[Discussion boundary; no next training recipe](<C:/xampp/htdocs/YEAR 4/Testing/CCTV_DGP_POST_NEARBY_LEARNING_REVIEW_V1.md>).
USED / FAILED: retain this run; do not rerun it unchanged or append epochs.

No accepted checkpoint, app adoption, extra epochs or next manual pilot is
qualified. Preserve original checkpoints, splits, caches, provenance, full
states and gates. Actual training/gradients stay human-manual L4 tmux; agent
VM access stays inventory-first hash-bound maintenance only. Native unpaired,
paired TRAIN and final roles, original app design, all five milestones and
seven automatic/assisted covering families remain. Detector-confidence work
is prepared only and UNRUN. Goal incomplete. Exact prior document follows.

# Nearby-gradient learning V1 R1 — manual VM commands

**Verified and UNRUN. This is a new finite training comparison.** It samples
gradients at the original state and two already measured nearby states, fits
one direction using all 96 RAW group constraints, and tests at most three
steps. At most one preserving research update can be accepted. The earlier
failed states are derivative samples only; their failures remain binding.

**Use R1 only.** Original V1 was withdrawn before transfer after its local
preflight used the VM response-folder name for downloaded results. R1 requires
an explicit response root and retains exact Linux VM path checks. Both packets
and failed preparation checks remain preserved. No training recipe or gate
changed during this startup correction.

Require **9 GiB free after installation**. Estimate **8–20 minutes measurement**,
**10–25 minutes including export**, with an enforced **34-minute external
ceiling**. The estimate uses prior timings; this recipe's L4 runtime and current
VM space have not been measured. Timing, memory, storage, output and export
limits are enforced. Keep the brightness-guard and conditioning-response VM
directories, virtual environment, checkpoints, caches and prior failures intact.
If a stop occurs, retain it and download the exported evidence. A failure does
not authorize a rerun or a longer run. No cleanup or automatic training launch
is part of this guide.

1. Upload from **Windows Google Cloud SDK Shell**:

```bat
cd /d "C:\xampp\htdocs\YEAR 4\Testing\outputs"
gcloud compute scp --project=forensic-dgp-thesis --zone=us-central1-a "cctv-dgp-nearby-gradient-learning-v1-r1-execution.tar.gz" "janusdominic0@forensic-dgp-thesis:/home/janusdominic0/"
gcloud compute scp --project=forensic-dgp-thesis --zone=us-central1-a "cctv-dgp-nearby-gradient-learning-v1-r1-execution.tar.gz.sha256" "janusdominic0@forensic-dgp-thesis:/home/janusdominic0/"
```

2. Install in **VM SSH**:

```bash
cd ~ &&
sha256sum -c cctv-dgp-nearby-gradient-learning-v1-r1-execution.tar.gz.sha256 &&
test -d ~/forensic-dgp/cctv_dgp_vm_bundle/.venv &&
test ! -e ~/forensic-dgp/cctv_dgp_nearby_gradient_learning_v1_r1_vm &&
tar -xzf cctv-dgp-nearby-gradient-learning-v1-r1-execution.tar.gz -C ~/forensic-dgp
```

3. Open **tmux**:

```bash
tmux new-session -A -s dgp_nearby_gradient_learning_v1_r1
```

4. Launch **inside tmux**:

The free-space check runs before the wrapper. Below 9 GiB it stops while the
packet remains unrun; request inventory-first cleanup before launching.

```bash
cd ~/forensic-dgp/cctv_dgp_nearby_gradient_learning_v1_r1_vm &&
source ~/forensic-dgp/cctv_dgp_vm_bundle/.venv/bin/activate &&
python -B -c "import shutil; free=shutil.disk_usage('.').free; print(dict(free_GiB=round(free/1024**3,2),required_GiB=9)); assert free >= 9*1024**3, 'Need 9 GiB free; request inventory-first cleanup before launch'" &&
python -B -u scripts/cctv_dgp_nearby_gradient_learning_v1_r1_vm.py --root . --parent-root ../cctv_dgp_bank_photometric_guard_v1_vm --response-root ../cctv_dgp_conditioning_response_v1_vm --protocol-sha 913b2d201f0567328840f179418ad4f4c7649bf41d5820c8d060bb7ba5e9ad24 --verify-transfer &&
bash scripts/run_learning.sh 913b2d201f0567328840f179418ad4f4c7649bf41d5820c8d060bb7ba5e9ad24
```

For live progress in a second VM SSH terminal, use
`tail -F ~/forensic-dgp/cctv_dgp_nearby_gradient_learning_v1_r1_vm/review.log`.

5. Download after the export finishes, from **Windows Google Cloud SDK Shell**:

```bat
cd /d "C:\xampp\htdocs\YEAR 4\Testing\outputs"
gcloud compute scp --project=forensic-dgp-thesis --zone=us-central1-a "janusdominic0@forensic-dgp-thesis:/home/janusdominic0/cctv-dgp-nearby-gradient-learning-v1-r1-results.tar.gz" "."
gcloud compute scp --project=forensic-dgp-thesis --zone=us-central1-a "janusdominic0@forensic-dgp-thesis:/home/janusdominic0/cctv-dgp-nearby-gradient-learning-v1-r1-results.tar.gz.sha256" "."
gcloud compute scp --project=forensic-dgp-thesis --zone=us-central1-a "janusdominic0@forensic-dgp-thesis:/home/janusdominic0/cctv-dgp-nearby-gradient-learning-v1-r1-export.json" "."
```

Protocol SHA256: `913b2d201f0567328840f179418ad4f4c7649bf41d5820c8d060bb7ba5e9ad24`. Execution archive SHA256:
`9ce0a58559f059f417012421ce71af220225849b9938073c281762381fa9d84b`. The checksum is ASCII with LF line endings.
Every PuTTY download uses one remote source. The wrapper preserves the training
exit code and exports success or failure evidence. Export `complete: true`
means the archive was created; it does not establish a successful training
comparison or useful restoration.

Every retained step must pass the original 17 RAW and 17 delivered-PNG group
checks and positive structure progress globally and in both sources. The
endpoint still requires at least 1% structure gain globally and in both
sources in both representations. This percentage measures reduction in a
specific paired TRAIN feature error; it is not recovered identity or native
CCTV performance. Below 1%, a safely accepted research state remains saved
but the pilot stops. The starting state and RNG are restored in every normal
outcome. Independent saved-evidence auditing and direct review of all 20 new
visual sheets follow the download. No epoch extension, app adoption or final
review qualification follows automatically.
[Preparation, limits and prospective audit](<C:/xampp/htdocs/YEAR 4/Testing/CCTV_DGP_NEARBY_GRADIENT_LEARNING_V1_R1_PREPARATION.md>).
