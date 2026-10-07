# Final V28 preservation diagnostic: zero optimizer updates

V28 completed 800 updates but failed its fixed acceptance checks. All 50 final
training faces and all four checkpoints were audited. A fixed mean-colour control
retained structure gain but failed five preservation checks. No V28 checkpoint
or processed control is adopted by the application.

This distinct diagnostic measures the unchanged seven losses at the immutable
final V28 state, plus three diagnostic components: RGB mean shift, clear-face
SSIM regression and clear-face ArcFace regression. It performs at most 100
gradient queries in ten fixed five-case batches. It creates no optimizer,
performs zero updates/epochs/backwards and saves no new model checkpoint.
The three additional components are measurements; no new training objective or
loss weights have been selected. Local derivatives remain prohibited.

The 28,332-byte three-file packet SHA256 is
`d700acbb0bf6cdc03de9b6389132e77830937af518bc6ae563fee67698f11a51`.
Protocol SHA256 is
`a270f4631f00c55e8c0377f082bc5ec7d593b633c53f91126078a5452e3cf5b0`.
It uploads no images or weights. It requires the existing V27 assets, closed R2
failure and V28 final files to remain in their original VM directories.

Require an idle existing NVIDIA L4/g2-standard-4 and 2 GiB free. The worker is
capped at 300 seconds; external supervision allows 330 seconds plus 30 seconds
kill grace. Export allows 120 seconds internally and 150 seconds externally plus
30 seconds grace. Allocated VRAM must stay within 20 GiB. Source/state/parity,
finite-value or timing failures stop the diagnostic and retain all evidence.
No job is killed to make room; no resume, unchanged retry or follow-on runs.

1. Upload from **Windows Google Cloud SDK Shell**:

```bat
cd /d "C:\xampp\htdocs\YEAR 4\Testing\outputs"
gcloud compute scp --project=forensic-dgp-thesis --zone=us-central1-a "cctv-dgp-v28-preservation-diagnostic-v1-execution.tar.gz" "janusdominic0@forensic-dgp-thesis:/home/janusdominic0/"
gcloud compute scp --project=forensic-dgp-thesis --zone=us-central1-a "cctv-dgp-v28-preservation-diagnostic-v1-execution.tar.gz.sha256" "janusdominic0@forensic-dgp-thesis:/home/janusdominic0/"
```

2. Install in **VM SSH**:

```bash
cd ~ &&
sha256sum -c cctv-dgp-v28-preservation-diagnostic-v1-execution.tar.gz.sha256 &&
test -x ~/forensic-dgp/cctv_dgp_vm_bundle/.venv/bin/python &&
test -d ~/forensic-dgp/cctv_dgp_feature_skips_vm_v27/outputs &&
test -f ~/forensic-dgp/cctv_dgp_original_decoder_gradient_v1_r2_vm/outputs/failure.json &&
test -f ~/forensic-dgp/cctv_dgp_active_original_decoder_vm_v28/outputs/update800/dgp_candidate_v28.pth &&
test ! -e ~/forensic-dgp/cctv_dgp_v28_preservation_diagnostic_v1_vm &&
tar -xzf cctv-dgp-v28-preservation-diagnostic-v1-execution.tar.gz -C ~/forensic-dgp
```

3. Open **tmux**:

```bash
tmux new-session -A -s dgp_v28_preservation_diagnostic
```

4. Launch **inside tmux**:

```bash
cd ~/forensic-dgp/cctv_dgp_v28_preservation_diagnostic_v1_vm &&
source ~/forensic-dgp/cctv_dgp_vm_bundle/.venv/bin/activate &&
python -B -u scripts/cctv_dgp_v28_preservation_diagnostic_v1_vm.py --root . --protocol-sha a270f4631f00c55e8c0377f082bc5ec7d593b633c53f91126078a5452e3cf5b0 --verify-transfer &&
bash scripts/run_v28_preservation.sh a270f4631f00c55e8c0377f082bc5ec7d593b633c53f91126078a5452e3cf5b0
```

Press `Ctrl+B`, then `D` to detach. The diagnostic stays in tmux. A completed
diagnostic reports `gradient_queries: 100` and `optimizer_updates: 0`.
Export completion confirms packaging only. If it stops, retain its exported
failure and do not run the same command again.

5. Download after **Export exit code: 0**, using **Windows Google Cloud SDK Shell**:

```bat
cd /d "C:\xampp\htdocs\YEAR 4\Testing\outputs"
gcloud compute scp --project=forensic-dgp-thesis --zone=us-central1-a "janusdominic0@forensic-dgp-thesis:/home/janusdominic0/cctv-dgp-v28-preservation-diagnostic-v1-results.tar.gz" "."
gcloud compute scp --project=forensic-dgp-thesis --zone=us-central1-a "janusdominic0@forensic-dgp-thesis:/home/janusdominic0/cctv-dgp-v28-preservation-diagnostic-v1-results.tar.gz.sha256" "."
gcloud compute scp --project=forensic-dgp-thesis --zone=us-central1-a "janusdominic0@forensic-dgp-thesis:/home/janusdominic0/cctv-dgp-v28-preservation-diagnostic-v1-export.json" "."
```

Each download has one remote source to support Windows PuTTY. The assistant
will independently audit the returned matrices, execution and unchanged states.
Do not delete the diagnostic root or V28 files before that audit closes.
