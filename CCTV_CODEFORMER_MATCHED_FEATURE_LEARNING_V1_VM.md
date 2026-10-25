# 25 October 2026: matched feature-learning return audited; both arms closed at8

STATUS: EXECUTED_QUALITY_STOP_BOTH_ARMS_UPDATE8_NO_PROMOTION.
[Current returned-evidence review](CCTV_CODEFORMER_MATCHED_FEATURE_LEARNING_V1_RETURN_REVIEW.md).
The prepared/unrun and manual-launch entries below are historical. This packet
has now been executed; do not relaunch it or append epochs after its quality stop.

Both arms completed8 genuine Adam updates. A trained the36 post-fusion heads;
B trained those heads plus24 existing U96 adapters. The exact complete original
trained-u64 state initialized both;515 R2 tensors stayed frozen. Independent
transport,712 saved RAW/PNG outputs,18 closed checkpoints and16 saved update
recurrences verify. Both executed gates at8 failed preservation;24/48 never ran.
Worker exit2 was a quality stop, not a trainer crash. The complete archive and
export sidecar verify; final export-process exit is not independently recorded.

B's new PNG structure gain over original u64 was0.046731% local/0.079418% global.
Its cumulative1.256301%/1.705342% over frozen U96 includes inherited u64 gain.
B clear RAW MAE improved0.001109%, but clear PNG MAE worsened0.008849%.
The same five clear photographs improved and three worsened in both views;
clear/local coarse-color tradeoffs also remain. This is not an export-rounding
bug. Root reviewed all54 primary plus2 partial sheets, with inspection scale
documented; no consistent obvious new facial-detail improvement was found.

U96 remains the experimental app incumbent; original u64 remains retained.
Neither failed arm is promoted. [Additional saved-pixel photometric diagnosis](CCTV_CODEFORMER_MATCHED_FEATURE_LEARNING_V1_PHOTOMETRIC_DIAGNOSIS.md)
finds mixed global-color/spatial error. Next: source design review of condition-
sensitive feature learning with separate clear-color/local-structure controls;
all inference conditioning must use input-only information. No successor packet
is ready. Do not repeat this stopped recipe
unchanged, rescue earlier checkpoints, or weaken quality gates. The restoration-
first goal, exact Protect/Apply preparation, manual L4 training, separate native
CCTV/paired synthetic evidence and deferred covering development remain active.
No new native DEV, FINAL, human acceptance or Zamboanga performance claim.

---

# Matched feature-learning comparison: manual VM guide

**PREPARED / UNRUN.** This is a new finite training comparison, not a qualified successor. U96 stays the experimental app incumbent. Original u64, U96, R2, checkpoints, splits and failed recipes remain retained.

A updates the existing 36 post-fusion head tensors. B updates the same heads plus 24 existing U96 adapter tensors. Both start from the exact complete original trained-u64 state and use identical data, targets, loss, learning rate and order. Each arm has **48 updates: two passes through 24 TRAIN photographs, with five degradation/clear roles per photograph**. This is not two epochs over the full original face dataset. There are at most 96 optimizer updates and 1,736 model forwards across both arms, including evaluation.

The shared objective uses photographic RGB error, RGB edge residuals and coarse color residuals. Fresh Adam uses maximum LR **0.00001**, four-update warmup and norm clipping 1.0. This conservative LR is a prospective choice, not a measured guarantee of improvement. The pilot tests the changed shared task and added trainable feature path; it does not repeat the previous failed recipe unchanged.

Require **33 GiB free before installation**, keeping at least 32 GiB after the two retained weight files are copied. The run requires the existing single NVIDIA L4 and no competing GPU process. Worker scientific work stops at 90 minutes; the supervisor allows 91 minutes plus at most 20 seconds for owned-process cleanup. Export has a separate 10-minute internal / 11-minute external cap. Runtime is unmeasured; timing projections can stop early. Quality checks at updates 8, 24 and 48 can close an arm sooner. Do not append epochs, rerun or reuse this fresh directory after a stop.

Execution upload: **107,416,508 bytes** (about 102.4 MiB). R2/U96 large weights are reused by verified local copying on the VM; the trained-u64 heads are included in the upload.

Protocol SHA256:
`afd59b90951138f94f0fcf0dea6ef308f8860042ae3c4f25b43c279d13d0fbd8`

Execution archive SHA256:
`8a2748e3f124626fcfcb46551cad031deeacf630734e06dfc569cc707d158d3a`

## 1. Upload from Windows Google Cloud SDK Shell

```bat
cd /d "C:\xampp\htdocs\YEAR 4\Testing\outputs"
gcloud compute scp --project=forensic-dgp-thesis --zone=us-central1-a "cctv-codeformer-matched-feature-learning-v1-execution.tar.gz" "janusdominic0@forensic-dgp-thesis:/home/janusdominic0/"
gcloud compute scp --project=forensic-dgp-thesis --zone=us-central1-a "cctv-codeformer-matched-feature-learning-v1-execution.tar.gz.sha256" "janusdominic0@forensic-dgp-thesis:/home/janusdominic0/"
```

## 2. Install in VM SSH

The source below supplies the retained R2/U96 weight files. The installer verifies their exact hashes and copies bytes without loading a neural model. It preserves any partial install and refuses overwrite.

```bash
cd ~ &&
sha256sum -c cctv-codeformer-matched-feature-learning-v1-execution.tar.gz.sha256 &&
test -d ~/forensic-dgp/cctv_dgp_vm_bundle/.venv &&
test -d ~/forensic-dgp/cctv_codeformer_u96_postfusion_smoothanchors_lr_v1_vm/weights &&
test ! -e ~/forensic-dgp/cctv_codeformer_matched_feature_learning_v1_vm &&
tar -xzf cctv-codeformer-matched-feature-learning-v1-execution.tar.gz -C ~/forensic-dgp &&
cd ~/forensic-dgp/cctv_codeformer_matched_feature_learning_v1_vm &&
python3 -B research/codeformer_matched_feature_learning_v1/install_weights.py --root . --source-root ~/forensic-dgp/cctv_codeformer_u96_postfusion_smoothanchors_lr_v1_vm --protocol-sha afd59b90951138f94f0fcf0dea6ef308f8860042ae3c4f25b43c279d13d0fbd8
```

## 3. Open tmux

```bash
tmux new-session -A -s codeformer_matched_v1
```

## 4. Launch inside tmux

```bash
cd ~/forensic-dgp/cctv_codeformer_matched_feature_learning_v1_vm &&
source ~/forensic-dgp/cctv_dgp_vm_bundle/.venv/bin/activate &&
python -B -u scripts/run_matched_feature_learning_v1.py --root . --protocol-sha afd59b90951138f94f0fcf0dea6ef308f8860042ae3c4f25b43c279d13d0fbd8 --verify-transfer &&
bash research/codeformer_matched_feature_learning_v1/run_matched.sh afd59b90951138f94f0fcf0dea6ef308f8860042ae3c4f25b43c279d13d0fbd8
```

Detach with **Ctrl+B, then D**. Reattach with `tmux attach -t codeformer_matched_v1`. The supervisor prints child/log locations. To monitor from another VM SSH terminal:

```bash
tail -F ~/forensic-dgp/cctv_codeformer_matched_feature_learning_v1_vm/run.log ~/forensic-dgp/cctv_codeformer_matched_feature_learning_v1_vm/export.log
```

Wait for the supervisor summary before downloading. `worker_exit: 0` means both finite TRAIN screens finished; `worker_exit: 2` records an ordinary quality stop; `worker_exit: 1` records a fatal execution/integrity/resource stop. `export_exit: 0` means retained evidence was packaged. None establishes useful restoration, identity accuracy or promotion. Return stopped results for audit instead of changing gates or restarting.

## 5. Download from Windows Google Cloud SDK Shell

Run each command separately. Each has exactly one remote source, which avoids the PuTTY multiple-remote-source error.

```bat
cd /d "C:\xampp\htdocs\YEAR 4\Testing\outputs"
gcloud compute scp --project=forensic-dgp-thesis --zone=us-central1-a "janusdominic0@forensic-dgp-thesis:/home/janusdominic0/cctv-codeformer-matched-feature-learning-v1-results.tar.gz" "."
gcloud compute scp --project=forensic-dgp-thesis --zone=us-central1-a "janusdominic0@forensic-dgp-thesis:/home/janusdominic0/cctv-codeformer-matched-feature-learning-v1-results.tar.gz.sha256" "."
gcloud compute scp --project=forensic-dgp-thesis --zone=us-central1-a "janusdominic0@forensic-dgp-thesis:/home/janusdominic0/cctv-codeformer-matched-feature-learning-v1-export.json" "."
```

Keep all three files. Returned bytes, optimizer/RNG/sampler states, RAW/PNG metrics, preservation gates and all 54 planned primary sheets (or explicitly missing/stopped counterparts) require independent audit and root visual review. New paired results remain adaptive TRAIN evidence. Broader DEV failures, native unpaired CCTV, user acceptance and independent FINAL are separate requirements. Protect/Apply remains optional and bypassable; further completion development remains deferred.

[Build receipt](outputs/codeformer_matched_feature_learning_v1_source_preparation/build_receipt_v1.json) · [Independent packet audit](outputs/codeformer_matched_feature_learning_v1_source_preparation/independent_packet_audit_v1.json) · [Joined source review](outputs/codeformer_matched_feature_learning_v1_source_preparation/runner_review/complete_runner_source_review_v1.json) · [Root source tests](outputs/codeformer_matched_feature_learning_v1_source_preparation/root_runner_source_tests_v1.json) · [Visual plan](outputs/codeformer_matched_feature_learning_v1_source_preparation/visual_plan_v1.json)
