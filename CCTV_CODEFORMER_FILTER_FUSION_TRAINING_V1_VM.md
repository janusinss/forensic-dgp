# CodeFormer filter-fusion training V1 — manual VM guide

**Sealed preparation only; training has not run.** The experiment starts from our
retained R2 adaptation and targets readable blur, mosaic and repeated-line inputs.
It trains existing fusion branches while retaining clear-face and brightening
controls. No model is promoted by upload, training or export completion.

The packet contains **1,143 files / 1,216,503,758 bytes**.
The archive is **1,123,126,380 bytes**. Require **20GiB free after
installation** and preserve the enforced4GiB reserve. The install command checks
space before extraction. Stop if it fails; this guide authorizes no deletion.

The ceiling is **151 new optimizer updates /604 microbatches**, not151 epochs.
The pilot uses301 distinct filter cases,55 clear and55 brightness-only anchors;
one filter case is explicitly repeated for the four-slot update schedule.
Expect progress every50 baseline/snapshot inputs, every10 teacher inputs and each
training update. GPU runtime is not measured yet. Internal work is capped at80
minutes; the outer worker cap is5300seconds and export10minutes, with finite
verification/closure. Allow approximately110minutes for the bounded session;
quality/resource failures can stop it much earlier. No automatic retry/resume.

1. Upload in **Windows Google Cloud SDK Shell**:

```bat
cd /d "C:\xampp\htdocs\YEAR 4\Testing\outputs"
gcloud compute scp --project=forensic-dgp-thesis --zone=us-central1-a "cctv-codeformer-filter-fusion-training-v1-execution.tar.gz" "janusdominic0@forensic-dgp-thesis:/home/janusdominic0/"
gcloud compute scp --project=forensic-dgp-thesis --zone=us-central1-a "cctv-codeformer-filter-fusion-training-v1-execution.tar.gz.sha256" "janusdominic0@forensic-dgp-thesis:/home/janusdominic0/"
```

2. Install in **VM SSH**:

```bash
cd ~ &&
printf '%s  %s\n' '9609ab7386f9d2d29d4a0f99dc2fa64a44407aaeb65c120b7e6c387f9aeada51' 'cctv-codeformer-filter-fusion-training-v1-execution.tar.gz' | sha256sum -c - &&
sha256sum -c cctv-codeformer-filter-fusion-training-v1-execution.tar.gz.sha256 &&
test -d ~/forensic-dgp/cctv_dgp_vm_bundle/.venv &&
test ! -e ~/forensic-dgp/cctv_codeformer_filter_fusion_training_v1_vm &&
python3 -c "import shutil; free=shutil.disk_usage('/home/janusdominic0/forensic-dgp').free; print('Available bytes:', free, 'Required before extraction:', 22691340238); assert free >= 22691340238, 'Need20GiB free after installation; stop here'" &&
tar -xzf cctv-codeformer-filter-fusion-training-v1-execution.tar.gz -C ~/forensic-dgp
```

3. Open **tmux**:

```bash
tmux new-session -A -s codeformer_filter_fusion_v1
```

4. Launch **inside tmux**:

```bash
cd ~/forensic-dgp/cctv_codeformer_filter_fusion_training_v1_vm &&
source ~/forensic-dgp/cctv_dgp_vm_bundle/.venv/bin/activate &&
python -B -u scripts/supervise_codeformer_filter_fusion_training_v1_vm.py --root . --protocol-sha c1f7d8386856bd95ab3af3a8623f33fcecddd4b7e08f0696ea7694a0542c1949
```

5. Download after the supervisor finishes, using **Windows Google Cloud SDK Shell**:

```bat
cd /d "C:\xampp\htdocs\YEAR 4\Testing\outputs"
gcloud compute scp --project=forensic-dgp-thesis --zone=us-central1-a "janusdominic0@forensic-dgp-thesis:/home/janusdominic0/cctv-codeformer-filter-fusion-training-v1-results.tar.gz" "."
gcloud compute scp --project=forensic-dgp-thesis --zone=us-central1-a "janusdominic0@forensic-dgp-thesis:/home/janusdominic0/cctv-codeformer-filter-fusion-training-v1-results.tar.gz.sha256" "."
gcloud compute scp --project=forensic-dgp-thesis --zone=us-central1-a "janusdominic0@forensic-dgp-thesis:/home/janusdominic0/cctv-codeformer-filter-fusion-training-v1-export.json" "."
gcloud compute scp --project=forensic-dgp-thesis --zone=us-central1-a "janusdominic0@forensic-dgp-thesis:/home/janusdominic0/cctv-codeformer-filter-fusion-training-v1-supervisor-final.json" "."
```

Use each SCP command separately; Windows PuTTY does not support multiple remote
sources in one SCP call. The export receipt can say `complete: true` after a
training failure: it means the evidence archive was created. Actual update counts,
gate failures and physical checkpoint status remain separate. Preserve all four
downloaded files for independent local review.

To detach safely, press **Ctrl+B**, release, then **D**. Reopen with the same tmux
command. In another VM SSH terminal, monitor the saved log:

```bash
tail -F ~/forensic-dgp/cctv_codeformer_filter_fusion_training_v1_vm/outputs/supervision/worker.log
```

If verification, storage, initialization, gradient or quality checks fail, retain
the traceback and export. Do not change epochs, gates, source or checkpoints and
rerun this immutable packet. Earlier research runs and failure evidence remain.

**Protocol SHA256**: `c1f7d8386856bd95ab3af3a8623f33fcecddd4b7e08f0696ea7694a0542c1949`

**Upload archive SHA256**: `9609ab7386f9d2d29d4a0f99dc2fa64a44407aaeb65c120b7e6c387f9aeada51`

[Preparation and evidence](CCTV_CODEFORMER_FILTER_FUSION_TRAINING_V1_PREPARATION.md)
and [finite design](research/codeformer_filter_fusion_training_v1/DESIGN_PILOT_V1.md)
retain source terms, split/overlap limits, exact counters and original preservation
rules. Native CCTV remains unpaired; wild1–wild5 have no clean originals and remain
visual stress tests outside training. No automatic input cutoff, real Zamboanga
performance or exact reconstructed identity is established. Keep the current app
unchanged until returned outputs, user-visible usefulness and required independent
evaluation qualify a replacement. Completion development remains deferred.
