# V28: finite active-original-decoder pilot

The failed R2 diagnostic is closed. Its two inactive `head4` tensors remain
frozen. V28 trains only the 12 demonstrated active original decoder tensors in a
separate copy, using the exact app input encoder. All original checkpoints,
sources, data and failure evidence are retained. No data or weights are uploaded.

The 29,527-byte, eight-file packet has SHA256
`e1484a675ad4330e4615d2f58a70f66ba8a8ad287b78cae66f8555ec4c4b1614`.
Its protocol SHA256 is
`27c140430673880f1aaab47a9df9af9b33758cc5d8adec53822cd9e05b13bbc8`.

At most 800 updates/80 epochs run. The pilot stops at update 50 if the delivered
structure error improves by less than 1%. Final 10% structure and all original
preservation/source/brightness gates remain. It first performs 70 component
gradient queries and exact initial same-batch parity; the optimizer is created
only if that selected12 proof passes. One fixed AdamW rate is 0.00003.

Require an idle existing NVIDIA L4/g2-standard-4 and 3 GiB free. Preflight is
capped at 300 seconds, fitting at 1,500 seconds and the whole worker at 1,800
seconds. External supervision allows 2,100 seconds plus 30 seconds kill grace.
Export is capped at 120 seconds, externally 150 seconds plus 30 seconds grace.
Allocated VRAM must stay within 20 GiB. No task is killed to make room; no
automatic follow-on, resume or unchanged retry is allowed. Every failure is kept.

1. Upload from **Windows Google Cloud SDK Shell**:

```bat
cd /d "C:\xampp\htdocs\YEAR 4\Testing\outputs"
gcloud compute scp --project=forensic-dgp-thesis --zone=us-central1-a "cctv-dgp-active-original-decoder-v28-execution.tar.gz" "janusdominic0@forensic-dgp-thesis:/home/janusdominic0/"
gcloud compute scp --project=forensic-dgp-thesis --zone=us-central1-a "cctv-dgp-active-original-decoder-v28-execution.tar.gz.sha256" "janusdominic0@forensic-dgp-thesis:/home/janusdominic0/"
```

2. Install in **VM SSH**:

```bash
cd ~ &&
sha256sum -c cctv-dgp-active-original-decoder-v28-execution.tar.gz.sha256 &&
test -x ~/forensic-dgp/cctv_dgp_vm_bundle/.venv/bin/python &&
test -d ~/forensic-dgp/cctv_dgp_feature_skips_vm_v27/outputs &&
test -f ~/forensic-dgp/cctv_dgp_original_decoder_gradient_v1_r2_vm/outputs/failure.json &&
test ! -e ~/forensic-dgp/cctv_dgp_active_original_decoder_vm_v28 &&
tar -xzf cctv-dgp-active-original-decoder-v28-execution.tar.gz -C ~/forensic-dgp
```

3. Open **tmux**:

```bash
tmux new-session -A -s dgp_active_original_decoder_v28
```

4. Launch **inside tmux**:

```bash
cd ~/forensic-dgp/cctv_dgp_active_original_decoder_vm_v28 &&
source ~/forensic-dgp/cctv_dgp_vm_bundle/.venv/bin/activate &&
python -B -u scripts/cctv_dgp_active_original_decoder_v28_vm.py --root . --protocol-sha 27c140430673880f1aaab47a9df9af9b33758cc5d8adec53822cd9e05b13bbc8 --verify-transfer &&
bash scripts/run_v28.sh 27c140430673880f1aaab47a9df9af9b33758cc5d8adec53822cd9e05b13bbc8
```

Detach with **Ctrl+B**, then **D**. The tmux session continues running. Completion
or a retained stop returns an export JSON with the archive hash and byte count.
`complete: true` in that export means packaging finished, not restoration success.

5. Download from **Windows Google Cloud SDK Shell** after the export JSON appears:

```bat
cd /d "C:\xampp\htdocs\YEAR 4\Testing\outputs"
gcloud compute scp --project=forensic-dgp-thesis --zone=us-central1-a "janusdominic0@forensic-dgp-thesis:/home/janusdominic0/cctv-dgp-active-original-decoder-v28-results.tar.gz" "."
gcloud compute scp --project=forensic-dgp-thesis --zone=us-central1-a "janusdominic0@forensic-dgp-thesis:/home/janusdominic0/cctv-dgp-active-original-decoder-v28-results.tar.gz.sha256" "."
gcloud compute scp --project=forensic-dgp-thesis --zone=us-central1-a "janusdominic0@forensic-dgp-thesis:/home/janusdominic0/cctv-dgp-active-original-decoder-v28-export.json" "."
```

Each download has one remote source so it works with Windows PuTTY. Preserve all
three downloaded files for the independent local audit. No candidate is adopted
by the app based on training completion. All 50 faces must be reviewed together
after the returned source/checkpoint/output/metric audit; native and final
qualification remain separate. No local training or assistant SSH launch occurs.
