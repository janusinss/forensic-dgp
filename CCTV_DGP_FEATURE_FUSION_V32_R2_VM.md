# V32 r2: original DGP feature fusion — five manual steps

R1 stopped before any optimizer because its cache loader required the base V32
directory. R2 corrects both directory guards and checks the cache guard during
`--verify-transfer`, before model work. Keep the failed R1 directory and export.
This is a packaging repair of the same finite experiment; no numerical gate or
training recipe changes. The R1 full return still needs download and audit.

The independently checked 280-query diagnostic connects all 11 fusion and 12
active decoder tensors. V32 tests one parameter-partition change: train those
23 original tensors on a fresh original checkpoint copy. No stopped V31 state
is resumed. The eight original fusion convolutions retain their architecture.
Backbone, inactive head4 and all evaluation-normalization buffers stay frozen.
The paired 781-reference/3,905-case TRAIN data, frozen schedule, seven losses,
initial normalizers, AdamW and every structure/preservation gate remain fixed.

Maximum **800 updates**: 1 complete 781-batch epoch plus 19 reference batches.
Stop at update50 unless structure gain reaches **1%**. Final TRAIN gates require
10% structure gain, all17 preservation groups, both source gains >=0 and mean-only
fraction <=20%. A training pass still requires useful DEV/native output review.
Original models, splits and every failed run remain retained. No automatic app
promotion, native/reserved pixels or local training is included.

Use the existing NVIDIA L4/g2-standard-4 at ~/forensic-dgp, an idle GPU and
**8GiB free**. Estimate **15–35 minutes plus export**. Storage allowance protects
up to3GiB uncompressed return plus its archive and margin. Finite limits:
preflight300s, cache900s, fit3600s, worker4500s, external4800s +30s grace,
export900s/external930s +30s grace, allocated VRAM20GiB and return3GiB.
Initial raw/PNG equality and all23 finite nonzero improvement gradients are
required before any optimizer. Retain any source/nonfinite/time/gate failure.

Protocol SHA256: 87582314eb1313a6914b4940496eeeb246c3023b7b6bf4f84da43e3f5e739d6c
Execution archive SHA256: 82824cbf3239bc017a74f13982001b31409c5e77af1dfb1b3bd7b9e9f77d283a

1. Upload from **Windows Google Cloud SDK Shell**:

```bat
cd /d "C:\xampp\htdocs\YEAR 4\Testing\outputs"
gcloud compute scp --project=forensic-dgp-thesis --zone=us-central1-a "cctv-dgp-feature-fusion-v32-r2-execution.tar.gz" "janusdominic0@forensic-dgp-thesis:/home/janusdominic0/"
gcloud compute scp --project=forensic-dgp-thesis --zone=us-central1-a "cctv-dgp-feature-fusion-v32-r2-execution.tar.gz.sha256" "janusdominic0@forensic-dgp-thesis:/home/janusdominic0/"
```

2. Install in **VM SSH**:

```bash
cd ~ &&
sha256sum -c cctv-dgp-feature-fusion-v32-r2-execution.tar.gz.sha256 &&
test -d ~/forensic-dgp/cctv_dgp_vm_bundle/.venv &&
test -f ~/forensic-dgp/cctv_dgp_feature_fusion_gradient_v1_vm/outputs/results.json &&
test ! -e ~/forensic-dgp/cctv_dgp_feature_fusion_vm_v32_r2 &&
tar -xzf cctv-dgp-feature-fusion-v32-r2-execution.tar.gz -C ~/forensic-dgp
```

3. Open **tmux**:

```bash
tmux new-session -A -s dgp_feature_fusion_v32_r2
```

4. Launch **inside tmux**:

```bash
cd ~/forensic-dgp/cctv_dgp_feature_fusion_vm_v32_r2 &&
source ~/forensic-dgp/cctv_dgp_vm_bundle/.venv/bin/activate &&
python -B -u scripts/cctv_dgp_feature_fusion_v32_vm.py --root . --protocol-sha 87582314eb1313a6914b4940496eeeb246c3023b7b6bf4f84da43e3f5e739d6c --verify-transfer &&
bash scripts/run_v32.sh 87582314eb1313a6914b4940496eeeb246c3023b7b6bf4f84da43e3f5e739d6c
```

Detach with Ctrl+B, release, D. Export complete:true means packaging only;
the independent return checker and visible-structure review determine results.
Do not resume a stopped pilot or relax the1% gate.

5. Download from **Windows Google Cloud SDK Shell**, one remote file per call:

```bat
cd /d "C:\xampp\htdocs\YEAR 4\Testing\outputs"
gcloud compute scp --project=forensic-dgp-thesis --zone=us-central1-a "janusdominic0@forensic-dgp-thesis:/home/janusdominic0/cctv-dgp-feature-fusion-v32-r2-results.tar.gz" "."
gcloud compute scp --project=forensic-dgp-thesis --zone=us-central1-a "janusdominic0@forensic-dgp-thesis:/home/janusdominic0/cctv-dgp-feature-fusion-v32-r2-results.tar.gz.sha256" "."
gcloud compute scp --project=forensic-dgp-thesis --zone=us-central1-a "janusdominic0@forensic-dgp-thesis:/home/janusdominic0/cctv-dgp-feature-fusion-v32-r2-export.json" "."
```

The separate download calls avoid PuTTY's multiple-remote-source error.
All actual new training remains this human/manual existing-L4 workflow.
Useful native restoration, seven covering families with separate automatic and
assisted reviews, independent final review and the app goal remain incomplete.

Preserve the R1 failure first, using these separate **Windows Google Cloud SDK Shell** downloads:

```bat
cd /d "C:\xampp\htdocs\YEAR 4\Testing\outputs"
gcloud compute scp --project=forensic-dgp-thesis --zone=us-central1-a "janusdominic0@forensic-dgp-thesis:/home/janusdominic0/cctv-dgp-feature-fusion-v32-r1-results.tar.gz" "."
gcloud compute scp --project=forensic-dgp-thesis --zone=us-central1-a "janusdominic0@forensic-dgp-thesis:/home/janusdominic0/cctv-dgp-feature-fusion-v32-r1-results.tar.gz.sha256" "."
gcloud compute scp --project=forensic-dgp-thesis --zone=us-central1-a "janusdominic0@forensic-dgp-thesis:/home/janusdominic0/cctv-dgp-feature-fusion-v32-r1-export.json" "."
```

Expected R1 archive:169,708,697 bytes; SHA256
599d84dd2b84fdc05e55b774d115951d251e4013ae777d7f023791bdb60f59e2.
