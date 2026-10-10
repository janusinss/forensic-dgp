# Original DGP feature-path diagnostic: manual L4 commands

Prepared diagnostic, not launched. The current app DGP remains selected.
V42's failed stop is retained; it is not resumed. This packet measures the
original feature path and reconstruction convolutions on a separate unchanged
current-checkpoint copy. It compares decoder-only, feature-only and joint
structure-gradient displacements. It trains no added decoder and substitutes
no pretrained restorer.

**Zero optimizer updates and zero epochs.** There are30 gradient queries on
50 photographic TRAIN cases, followed by nine finite trial directions on those
50 and50 different photographic TRAIN cases. All1000 raw images, delivered
PNGs, mean-only PNGs and embedding vectors are retained. Each trial starts from
the original parameters and restores them immediately afterward. These trial
parameters are diagnostic observations, not a selected trained checkpoint or
an exact-resume training state. Neither TRAIN cohort is a final evaluation.

The nine trials use three scopes and relative L2 displacements0.00001,0.0001
and0.001. This scale grid is a predeclared sensitivity test, not an AdamW
learning-rate recommendation. The same17 preservation groups,1% structure,
both-source nonregression and20% brightness limit are reported separately for
raw/PNG and each cohort. Passing this small diagnostic cannot qualify a model;
failing trials remain in the archive. No native, DEV or reserved final case,
completion generation or app change enters this run.

Expected diagnostic runtime **5–15 minutes**, export **1–5 minutes**; these are
preparation estimates. Require **4 GiB free after installation**. Actual bounds:
cache120s, gradients180s, trials600s, worker1200s/external1230s with30s kill grace,
export300s/external330s with30s grace,20GiB allocated VRAM,1.75GiB uncompressed
return and512MiB disk reserve. A measured baseline projects trial time and
output-plus-export storage with factor1.25 before any gradient query.
The packet is self-contained for100 inputs,20 targets/masks, model code,
retained DGP and fixed recognizer; it reuses the existing VM Python venv.
No file is deleted. Keep the archived failures and migration cache backup.

Execution archive: **183,278,655 bytes**.
Protocol SHA256: `ccc30870959c8f4f4f448c1d1e06bb0ae0403d450b81143c6bcb9032d122e94b`
Archive SHA256: `a76a7be6d7c1c7328f11241eb9c617695e786d3efc0e749f7297cd226a0c2ecf`

1. Upload from **Windows Google Cloud SDK Shell**:

```bat
cd /d "C:\xampp\htdocs\YEAR 4\Testing\outputs"
gcloud compute scp --project=forensic-dgp-thesis --zone=us-central1-a "cctv-dgp-original-feature-probe-v1-execution.tar.gz" "janusdominic0@forensic-dgp-thesis:/home/janusdominic0/"
gcloud compute scp --project=forensic-dgp-thesis --zone=us-central1-a "cctv-dgp-original-feature-probe-v1-execution.tar.gz.sha256" "janusdominic0@forensic-dgp-thesis:/home/janusdominic0/"
```

2. Install in **VM SSH**:

```bash
cd ~ &&
sha256sum -c cctv-dgp-original-feature-probe-v1-execution.tar.gz.sha256 &&
test -x ~/forensic-dgp/cctv_dgp_vm_bundle/.venv/bin/python &&
test ! -e ~/forensic-dgp/cctv_dgp_original_feature_probe_v1_vm &&
tar -xzf cctv-dgp-original-feature-probe-v1-execution.tar.gz -C ~/forensic-dgp &&
df -h ~/forensic-dgp
```

3. Open tmux:

```bash
tmux new-session -A -s dgp_original_feature_probe_v1
```

4. Launch **inside tmux**:

```bash
cd ~/forensic-dgp/cctv_dgp_original_feature_probe_v1_vm &&
source ~/forensic-dgp/cctv_dgp_vm_bundle/.venv/bin/activate &&
python -B -u scripts/cctv_dgp_original_feature_probe_v1_vm.py --root . --protocol-sha ccc30870959c8f4f4f448c1d1e06bb0ae0403d450b81143c6bcb9032d122e94b --verify-transfer &&
bash scripts/run_feature_probe.sh ccc30870959c8f4f4f448c1d1e06bb0ae0403d450b81143c6bcb9032d122e94b
```

The transfer check makes zero neural or gradient calls. The manually launched
diagnostic verifies the idle existing L4/g2-standard-4, unchanged normalization,
initial parity and declared tensor ownership. It exports partial evidence after
a stop. Detach with Ctrl+B, release, D. Reattach with
`tmux attach-session -t dgp_original_feature_probe_v1`.

5. Download from **Windows Google Cloud SDK Shell**, one remote source per command:

```bat
cd /d "C:\xampp\htdocs\YEAR 4\Testing\outputs"
gcloud compute scp --project=forensic-dgp-thesis --zone=us-central1-a "janusdominic0@forensic-dgp-thesis:/home/janusdominic0/cctv-dgp-original-feature-probe-v1-results.tar.gz" "."
gcloud compute scp --project=forensic-dgp-thesis --zone=us-central1-a "janusdominic0@forensic-dgp-thesis:/home/janusdominic0/cctv-dgp-original-feature-probe-v1-results.tar.gz.sha256" "."
gcloud compute scp --project=forensic-dgp-thesis --zone=us-central1-a "janusdominic0@forensic-dgp-thesis:/home/janusdominic0/cctv-dgp-original-feature-probe-v1-export.json" "."
```

Export `complete: true` confirms packaging, not model quality. Return all three
files and the terminal output for independent hashing, gradient/displacement
arithmetic, saved raw/PNG metric checks, CPU inference/embedding replay and
whole-face visual review before a new training design. Source labels are not
ethnicity labels; no Zamboanga or hidden-identity performance is inferred.
All five restoration milestones and seven automatic/assisted covering families
remain required. The full goal stays active/incomplete.
