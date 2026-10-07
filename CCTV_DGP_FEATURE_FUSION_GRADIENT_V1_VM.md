# Original DGP feature-fusion diagnostic: five manual steps

V31 stopped at50 with0.694525% against the1% early requirement. It is closed.
This new diagnostic measures gradients through11 original FPN fusion tensors and
compares them with12 decoder tensors. It performs **0 optimizer updates / 0 epochs**.
Original and stopped V31 weights are observations only; neither is resumed.
Two50-case TRAIN cohorts contain ten references each with their clear and four
degraded views, matched by source/profile. No native or reserved final cases are used.
All eyes, nose, mouth, outline and visible appearance remain in scope.

Use the existing NVIDIA L4/g2-standard-4 at ~/forensic-dgp, **6GiB free**, and an idle
GPU. Estimate **3-10 minutes plus1-5 minutes export**; runtime is not a success claim.
The worker stops at600s, the external supervisor at900s plus30s grace, export at300s
(external330s plus30s grace), allocated VRAM20GiB and uncompressed return1.5GiB.
Source/state mismatch or nonfinite output/loss/gradient stops and retains evidence.
No optimizer, .backward(), checkpoint writer, gate relaxation or app change is present.
Zero fusion gradients are recorded as a diagnostic finding, never a reason to train.

Protocol SHA256: 5926dea8c477a9d51ff97186e6fec2b0d8eab8e67a7cf7a0ed1a6c716756e0af
Execution archive SHA256: 9b1d3ab30224e24e02e1b00995b3d290bcbb4e037f121aa924d871b80fff4b5f

1. Upload from **Windows Google Cloud SDK Shell**:

```bat
cd /d "C:\xampp\htdocs\YEAR 4\Testing\outputs"
gcloud compute scp --project=forensic-dgp-thesis --zone=us-central1-a "cctv-dgp-feature-fusion-gradient-v1-execution.tar.gz" "janusdominic0@forensic-dgp-thesis:/home/janusdominic0/"
gcloud compute scp --project=forensic-dgp-thesis --zone=us-central1-a "cctv-dgp-feature-fusion-gradient-v1-execution.tar.gz.sha256" "janusdominic0@forensic-dgp-thesis:/home/janusdominic0/"
```

2. Install in **VM SSH**:

```bash
cd ~ &&
sha256sum -c cctv-dgp-feature-fusion-gradient-v1-execution.tar.gz.sha256 &&
test -d ~/forensic-dgp/cctv_dgp_vm_bundle/.venv &&
test -d ~/forensic-dgp/cctv_dgp_profile_batches_vm_v31/outputs/update50 &&
test ! -e ~/forensic-dgp/cctv_dgp_feature_fusion_gradient_v1_vm &&
tar -xzf cctv-dgp-feature-fusion-gradient-v1-execution.tar.gz -C ~/forensic-dgp
```

3. Open **tmux**:

```bash
tmux new-session -A -s dgp_feature_fusion
```

4. Launch **inside tmux**:

```bash
cd ~/forensic-dgp/cctv_dgp_feature_fusion_gradient_v1_vm &&
source ~/forensic-dgp/cctv_dgp_vm_bundle/.venv/bin/activate &&
python -B -u scripts/cctv_dgp_feature_fusion_gradient_v1_vm.py --root . --protocol-sha 5926dea8c477a9d51ff97186e6fec2b0d8eab8e67a7cf7a0ed1a6c716756e0af --verify-transfer &&
bash scripts/run_fusion.sh 5926dea8c477a9d51ff97186e6fec2b0d8eab8e67a7cf7a0ed1a6c716756e0af
```

Expected final measurement:280 queries,0 updates. Detach with Ctrl+B, release, D.
Keep any failed export result. Export complete:true means packaging only.

5. Download from **Windows Google Cloud SDK Shell**, one remote file per call:

```bat
cd /d "C:\xampp\htdocs\YEAR 4\Testing\outputs"
gcloud compute scp --project=forensic-dgp-thesis --zone=us-central1-a "janusdominic0@forensic-dgp-thesis:/home/janusdominic0/cctv-dgp-feature-fusion-gradient-v1-results.tar.gz" "."
gcloud compute scp --project=forensic-dgp-thesis --zone=us-central1-a "janusdominic0@forensic-dgp-thesis:/home/janusdominic0/cctv-dgp-feature-fusion-gradient-v1-results.tar.gz.sha256" "."
gcloud compute scp --project=forensic-dgp-thesis --zone=us-central1-a "janusdominic0@forensic-dgp-thesis:/home/janusdominic0/cctv-dgp-feature-fusion-gradient-v1-export.json" "."
```

The separate calls avoid PuTTY's multiple-remote-source error. The new prospective
local checker audits returned arrays/arithmetic and replays200 outputs with CPU
inference only. It never executes returned code or recomputes gradients locally.
A completed diagnostic alone does not select a new training recipe. Original
checkpoints, splits, research caches and every previous failure remain preserved.
Useful native restoration, all covering families and independent final review
are still required. The full goal remains active/incomplete.
