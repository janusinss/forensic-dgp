# V31: paired clear/degraded batches - five manual steps

The independently audited 280-query diagnostic is complete, with zero training
updates. Four unexposed clear controls show raw pixel drift; preservation terms
therefore retain a valid role. V31 tests one change: every minibatch contains a
reference's clear control and its four degradations. All 781 approved TRAIN
references, original initialization, 12 selected decoder tensors, seven losses,
normalizers, AdamW settings and numeric gates remain fixed.

This is a new finite experiment, not a resume of V30. Maximum 800 updates
(1 complete 781-reference epoch plus 19 batches). The unchanged early gate
requires 1% structure gain at update50. Final TRAIN gates require 10% gain,
all 17 preservation groups, both source gains >=0 and mean-only fraction <=20%.
Training success remains insufficient for native CCTV or app qualification.

Existing NVIDIA L4/g2-standard-4 VM only, below ~/forensic-dgp. Require 6 GiB
free. Estimate 15-35 minutes plus export. Enforced preflight 300s, cache 900s,
fit 3600s, worker 4500s, external 4800s + 30s grace, export 900s
(external 930s + 30s grace), allocated VRAM <=20 GiB, uncompressed export <=3 GiB.
No other GPU workload may be running. Original research and all failed gates remain.

Protocol SHA256: ae91d25066bcb532a8a88bdc48fe401c4b18d73e8d348b47b5d012d5df8e485e
Execution archive SHA256: bdf79346b10376b75328dfd2fab3ab3902935982cb9910b4478b401c0c55b9b8

1. Upload from **Windows Google Cloud SDK Shell**:

```bat
cd /d "C:\xampp\htdocs\YEAR 4\Testing\outputs"
gcloud compute scp --project=forensic-dgp-thesis --zone=us-central1-a "cctv-dgp-profile-batches-v31-execution.tar.gz" "janusdominic0@forensic-dgp-thesis:/home/janusdominic0/"
gcloud compute scp --project=forensic-dgp-thesis --zone=us-central1-a "cctv-dgp-profile-batches-v31-execution.tar.gz.sha256" "janusdominic0@forensic-dgp-thesis:/home/janusdominic0/"
```

2. Install in **VM SSH**:

```bash
cd ~ &&
sha256sum -c cctv-dgp-profile-batches-v31-execution.tar.gz.sha256 &&
test -d ~/forensic-dgp/cctv_dgp_vm_bundle/.venv &&
test -f ~/forensic-dgp/cctv_dgp_v30_sampling_gradient_v1_vm/outputs/results.json &&
test ! -e ~/forensic-dgp/cctv_dgp_profile_batches_vm_v31 &&
tar -xzf cctv-dgp-profile-batches-v31-execution.tar.gz -C ~/forensic-dgp
```

3. Open **tmux**:

```bash
tmux new-session -A -s dgp_profile_batches_v31
```

4. Launch **inside tmux**:

```bash
cd ~/forensic-dgp/cctv_dgp_profile_batches_vm_v31 &&
source ~/forensic-dgp/cctv_dgp_vm_bundle/.venv/bin/activate &&
python -B -u scripts/cctv_dgp_profile_batches_v31_vm.py --root . --protocol-sha ae91d25066bcb532a8a88bdc48fe401c4b18d73e8d348b47b5d012d5df8e485e --verify-transfer &&
bash scripts/run_v31.sh ae91d25066bcb532a8a88bdc48fe401c4b18d73e8d348b47b5d012d5df8e485e
```

Detach with Ctrl+B, release both keys, then D. Retain any stop or failed gate.
Export `complete:true` confirms packaging. The local independent return audit
and visible-structure review determine the next step. App qualification remains
pending useful development outputs and independent review.

5. Download from **Windows Google Cloud SDK Shell**, using separate PuTTY calls:

```bat
cd /d "C:\xampp\htdocs\YEAR 4\Testing\outputs"
gcloud compute scp --project=forensic-dgp-thesis --zone=us-central1-a "janusdominic0@forensic-dgp-thesis:/home/janusdominic0/cctv-dgp-profile-batches-v31-results.tar.gz" "."
gcloud compute scp --project=forensic-dgp-thesis --zone=us-central1-a "janusdominic0@forensic-dgp-thesis:/home/janusdominic0/cctv-dgp-profile-batches-v31-results.tar.gz.sha256" "."
gcloud compute scp --project=forensic-dgp-thesis --zone=us-central1-a "janusdominic0@forensic-dgp-thesis:/home/janusdominic0/cctv-dgp-profile-batches-v31-export.json" "."
```
