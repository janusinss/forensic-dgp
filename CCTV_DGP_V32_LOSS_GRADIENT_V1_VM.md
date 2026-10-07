# Stopped V32 loss diagnostic: five manual steps

The audited V32 r2 pilot stopped at50/800 with0.970672% structure gain against
the unchanged1% requirement. This separate diagnostic inspects loss gradients at
the original and stopped states. **0 optimizer updates / 0 epochs / no new checkpoint.**
It does not resume V32 or start follow-on training.

Two50-case TRAIN cohorts have five references from each photographic source,
each with its clear control and four degradations. The exposed cohort uses the
first five touched references per source in the frozen schedule. The other uses
all50 fixed preflight/normalization previews, which were not optimized by50.
These are development TRAIN observations, not unseen or native CCTV evidence.
All visible facial features and the existing1%/10% preservation gates remain.

Require an idle existing NVIDIA L4/g2-standard-4 and **6GiB free**. Estimate
**1-5 minutes diagnostic plus1-3 minutes export**, based on earlier280-query runs.
Enforced limits: worker600s; external900s plus30s grace; export300s;
external export330s plus30s grace; allocated VRAM20GiB; raw return1.5GiB.
State/source mismatch or nonfinite output/loss/gradient stops and retains evidence.
No automatic historical launch, deletion, optimizer or application promotion.

Protocol SHA256: 61c04aef4ee184adc9830482477bf37ad9c936a8c0247e44d26eefc1a6bf483d
Execution archive SHA256: 22d76f3637a3ac580bc18fbac79ff5e3d24a61ec0a00ce082fde78c7069c6052

1. Upload from **Windows Google Cloud SDK Shell**:

```bat
cd /d "C:\xampp\htdocs\YEAR 4\Testing\outputs"
gcloud compute scp --project=forensic-dgp-thesis --zone=us-central1-a "cctv-dgp-v32-loss-gradient-v1-execution.tar.gz" "janusdominic0@forensic-dgp-thesis:/home/janusdominic0/"
gcloud compute scp --project=forensic-dgp-thesis --zone=us-central1-a "cctv-dgp-v32-loss-gradient-v1-execution.tar.gz.sha256" "janusdominic0@forensic-dgp-thesis:/home/janusdominic0/"
```

2. Install in **VM SSH**:

```bash
cd ~ &&
sha256sum -c cctv-dgp-v32-loss-gradient-v1-execution.tar.gz.sha256 &&
test -d ~/forensic-dgp/cctv_dgp_vm_bundle/.venv &&
test -f ~/forensic-dgp/cctv_dgp_feature_fusion_vm_v32_r2/outputs/update50/dgp_candidate_v32.pth &&
test ! -e ~/forensic-dgp/cctv_dgp_v32_loss_gradient_v1_vm &&
tar -xzf cctv-dgp-v32-loss-gradient-v1-execution.tar.gz -C ~/forensic-dgp
```

3. Open **tmux**:

```bash
tmux new-session -A -s dgp_v32_loss
```

4. Launch **inside tmux**:

```bash
cd ~/forensic-dgp/cctv_dgp_v32_loss_gradient_v1_vm &&
source ~/forensic-dgp/cctv_dgp_vm_bundle/.venv/bin/activate &&
python -B -u scripts/cctv_dgp_v32_loss_gradient_v1_vm.py --root . --protocol-sha 61c04aef4ee184adc9830482477bf37ad9c936a8c0247e44d26eefc1a6bf483d --verify-transfer &&
bash scripts/run_loss_gradient.sh 61c04aef4ee184adc9830482477bf37ad9c936a8c0247e44d26eefc1a6bf483d
```

Leave tmux with **Ctrl+B**, release, then **D**. Reconnect with the same step3
command. A completed diagnostic reports gradient_queries:280 and optimizer_updates:0.
Export complete:true means packaging only. If it stops, download the failure too;
keep all files, report the traceback and do not rerun or alter its assertions.

5. Download in **Windows Google Cloud SDK Shell**, after export finishes:

```bat
cd /d "C:\xampp\htdocs\YEAR 4\Testing\outputs"
gcloud compute scp --project=forensic-dgp-thesis --zone=us-central1-a "janusdominic0@forensic-dgp-thesis:/home/janusdominic0/cctv-dgp-v32-loss-gradient-v1-results.tar.gz" "."
gcloud compute scp --project=forensic-dgp-thesis --zone=us-central1-a "janusdominic0@forensic-dgp-thesis:/home/janusdominic0/cctv-dgp-v32-loss-gradient-v1-results.tar.gz.sha256" "."
gcloud compute scp --project=forensic-dgp-thesis --zone=us-central1-a "janusdominic0@forensic-dgp-thesis:/home/janusdominic0/cctv-dgp-v32-loss-gradient-v1-export.json" "."
```

Three separate remote downloads are required by Windows/PuTTY. Returned hashes,
all saved gradients, immutable state/source boundaries and CPU inference parity
need independent local audit before choosing a new finite training recipe.
Infinitesimal gradients alone do not reconstruct unsaved AdamW history or qualify
whole-face CCTV quality. Reserved final pixels remain unopened. Goal active.
