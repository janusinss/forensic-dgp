# V40 spatial DGP learning pilot: five manual steps

V39's independently audited return passes all57 improvement-gradient partitions
and exact initial preservation, with zero optimizer updates. V40 learns that
same17,952-parameter spatial decoder using the existing781 TRAIN references and
3,905 cases. Original DGP weights, the fixed initial decoder and recognizer stay
frozen. All visible facial features remain in scope. Pretrained restoration is
not a training target or substituted primary model.

**Maximum800 updates: one781-reference epoch plus19 paired batches.** The fixed
V31 metadata schedule pairs one clear control and four degradations. Initial50
normalizers and all seven original losses stay fixed. A fixed AdamW3e-4 rate is
for the NEW initialized decoder, rather than original-weight fine-tuning. There
is no rate sweep, checkpoint selection from DEV or continuation of V38.

The unchanged1%-at50 and10%-at800 structure thresholds, all17 PNG preservation
groups, both-source nonregression and20% brightness-only limit remain. At50,
failure of structure OR preservation stops and exports evidence. At800 a failed
capacity result is exported without adoption. The broader/final/native quality
requirements remain separate. All50 raw previews and all3,905 PNG/mean-only
images/vectors are saved per complete snapshot at0/50/800; raw stages for other
cases are hashed, not retained. This is paired photographic TRAIN evidence.

Require the existing idle **NVIDIA L4/g2-standard-4**, existing venv, and **6GiB
free after installation**. This self-contained444,516,831-byte packet copies verified
inputs, original weights, code and the untrained seed; no old pilot runs.
Estimated training **15–40 minutes**, export **3–15 minutes**; these are estimates,
not measured V40 timing. Cache900s, fit3600s, worker4500s/external4800s+30s grace,
export900s/external930s+30s grace, VRAM20GiB, return3GiB are enforced. Cache and
update20 timing projections use1.25 safety factors. No automatic retry or cleanup.

Protocol SHA256: `b21597d7c50dc39ee7cb2bd70d7d0ad9f5cd967661b6f5e78413a7e4eb121010`
Execution archive SHA256: `b8113316de65ee965ffe71303adae6caf5446be7f853ab95c11242e776036b0f`

1. Upload from **Windows Google Cloud SDK Shell**:

```bat
cd /d "C:\xampp\htdocs\YEAR 4\Testing\outputs"
gcloud compute scp --project=forensic-dgp-thesis --zone=us-central1-a "cctv-dgp-spatial-fit-v40-execution.tar.gz" "janusdominic0@forensic-dgp-thesis:/home/janusdominic0/"
gcloud compute scp --project=forensic-dgp-thesis --zone=us-central1-a "cctv-dgp-spatial-fit-v40-execution.tar.gz.sha256" "janusdominic0@forensic-dgp-thesis:/home/janusdominic0/"
```

2. Install in **VM SSH**:

```bash
cd ~ &&
sha256sum -c cctv-dgp-spatial-fit-v40-execution.tar.gz.sha256 &&
test -x ~/forensic-dgp/cctv_dgp_vm_bundle/.venv/bin/python &&
test ! -e ~/forensic-dgp/cctv_dgp_spatial_fit_vm_v40 &&
tar -xzf cctv-dgp-spatial-fit-v40-execution.tar.gz -C ~/forensic-dgp
```

3. Open tmux:

```bash
tmux new-session -A -s dgp_spatial_fit_v40
```

4. Launch **inside tmux**:

```bash
cd ~/forensic-dgp/cctv_dgp_spatial_fit_vm_v40 &&
source ~/forensic-dgp/cctv_dgp_vm_bundle/.venv/bin/activate &&
python -B -u scripts/cctv_dgp_spatial_fit_v40_vm.py --root . --protocol-sha b21597d7c50dc39ee7cb2bd70d7d0ad9f5cd967661b6f5e78413a7e4eb121010 --verify-transfer &&
bash scripts/run_v40.sh b21597d7c50dc39ee7cb2bd70d7d0ad9f5cd967661b6f5e78413a7e4eb121010
```

Detach with Ctrl+B, release, D. Reattach with
`tmux attach-session -t dgp_spatial_fit_v40`. Keep the trainer log and every stop.
No other GPU task is terminated. A successful export is not a training-quality
pass; return the files for an independent audit and all50 preview review.

5. Download from **Windows Google Cloud SDK Shell**, one remote file per command:

```bat
cd /d "C:\xampp\htdocs\YEAR 4\Testing\outputs"
gcloud compute scp --project=forensic-dgp-thesis --zone=us-central1-a "janusdominic0@forensic-dgp-thesis:/home/janusdominic0/cctv-dgp-spatial-fit-v40-results.tar.gz" "."
gcloud compute scp --project=forensic-dgp-thesis --zone=us-central1-a "janusdominic0@forensic-dgp-thesis:/home/janusdominic0/cctv-dgp-spatial-fit-v40-results.tar.gz.sha256" "."
gcloud compute scp --project=forensic-dgp-thesis --zone=us-central1-a "janusdominic0@forensic-dgp-thesis:/home/janusdominic0/cctv-dgp-spatial-fit-v40-export.json" "."
```

The app, automatic and assisted completion, independent final review and useful
native output are not qualified by this pilot. No real Zamboanga CCTV exists yet;
source labels do not infer ethnicity. Reserved final crops stay unopened.
