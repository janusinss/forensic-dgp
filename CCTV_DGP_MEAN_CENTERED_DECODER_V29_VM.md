# V29: one finite mean-centered original-DGP pilot

V28 learned structure but failed preservation and brightness requirements.
The audited final-state diagnostic supports testing one fixed mean-centering
path during training. V29 starts from the original DGP in a separate copy;
all original weights, historical failures and the app remain preserved.
Loss weights, optimizer, schedule and acceptance gates are unchanged.

The seven-file packet is 30,293 bytes. Archive SHA256:
`de158e52c44494638c0477cd7fca2a67ed12f18ad40a9fc38e9ff45cdac67e4d`.
Protocol SHA256:
`77565ba437959305f22cff4dd967fc6c3caadbf9dbd4ac91abf8a366577fa72f`.
It uploads no images or weights; the existing V27, R2, V28 and diagnostic files
must remain in their original VM locations.

At most 800 updates/80 epochs run, with snapshots at 0/50/400/800.
Expect approximately 3–5 minutes based on V28's 126-second worker; actual V29
timing is unmeasured. It stops if the update-50 structure gain is below 1%.
Final acceptance requires at least 10% structure gain, both source gains
nonnegative, all 17 preservation groups and brightness fraction at most 20%.
The changed path must first pass 70 VM gradient queries and exact initial
50-case parity before an optimizer is created. Final800 only; no checkpoint
selection for deployment occurs here. A successful export alone is not a pass.

Require an idle existing NVIDIA L4/g2-standard-4 and 3 GiB free. Preflight is
capped at 300 seconds, fitting at 1,500 seconds, whole worker at 1,800 seconds,
external supervision at 2,100 seconds plus 30 seconds kill grace. Export allows
120 seconds internally, 150 externally plus 30 seconds grace. Allocated VRAM
must stay within 20 GiB. Every stop retains its evidence; no resume, unchanged
retry, deletion, competing-task termination or automatic follow-on occurs.

1. Upload from **Windows Google Cloud SDK Shell**:

```bat
cd /d "C:\xampp\htdocs\YEAR 4\Testing\outputs"
gcloud compute scp --project=forensic-dgp-thesis --zone=us-central1-a "cctv-dgp-mean-centered-decoder-v29-execution.tar.gz" "janusdominic0@forensic-dgp-thesis:/home/janusdominic0/"
gcloud compute scp --project=forensic-dgp-thesis --zone=us-central1-a "cctv-dgp-mean-centered-decoder-v29-execution.tar.gz.sha256" "janusdominic0@forensic-dgp-thesis:/home/janusdominic0/"
```

2. Install in **VM SSH**:

```bash
cd ~ &&
sha256sum -c cctv-dgp-mean-centered-decoder-v29-execution.tar.gz.sha256 &&
test -x ~/forensic-dgp/cctv_dgp_vm_bundle/.venv/bin/python &&
test -f ~/forensic-dgp/cctv_dgp_original_decoder_gradient_v1_r2_vm/outputs/failure.json &&
test -f ~/forensic-dgp/cctv_dgp_active_original_decoder_vm_v28/outputs/results.json &&
test -f ~/forensic-dgp/cctv_dgp_v28_preservation_diagnostic_v1_vm/outputs/results.json &&
test ! -e ~/forensic-dgp/cctv_dgp_mean_centered_decoder_vm_v29 &&
tar -xzf cctv-dgp-mean-centered-decoder-v29-execution.tar.gz -C ~/forensic-dgp
```

3. Open **tmux**:

```bash
tmux new-session -A -s dgp_v29
```

4. Launch **inside tmux**:

```bash
cd ~/forensic-dgp/cctv_dgp_mean_centered_decoder_vm_v29 &&
source ~/forensic-dgp/cctv_dgp_vm_bundle/.venv/bin/activate &&
python -B -u scripts/cctv_dgp_mean_centered_decoder_v29_vm.py --root . --protocol-sha 77565ba437959305f22cff4dd967fc6c3caadbf9dbd4ac91abf8a366577fa72f --verify-transfer &&
bash scripts/run_v29.sh 77565ba437959305f22cff4dd967fc6c3caadbf9dbd4ac91abf8a366577fa72f
```

Press `Ctrl+B`, then `D` to detach. The job stays in tmux. If it stops, retain
the exported failure and download it; do not repeat the same launch.

5. Download after **Export exit code: 0**, from **Windows Google Cloud SDK Shell**:

```bat
cd /d "C:\xampp\htdocs\YEAR 4\Testing\outputs"
gcloud compute scp --project=forensic-dgp-thesis --zone=us-central1-a "janusdominic0@forensic-dgp-thesis:/home/janusdominic0/cctv-dgp-mean-centered-decoder-v29-results.tar.gz" "."
gcloud compute scp --project=forensic-dgp-thesis --zone=us-central1-a "janusdominic0@forensic-dgp-thesis:/home/janusdominic0/cctv-dgp-mean-centered-decoder-v29-results.tar.gz.sha256" "."
gcloud compute scp --project=forensic-dgp-thesis --zone=us-central1-a "janusdominic0@forensic-dgp-thesis:/home/janusdominic0/cctv-dgp-mean-centered-decoder-v29-export.json" "."
```

Each download has one remote source for Windows PuTTY. The assistant will
independently audit the return and review all 50 final outputs before any
separately frozen native/app qualification. No local training or automatic
application promotion is authorized by a completed archive.
