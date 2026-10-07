# V30 sampling-gradient diagnostic: five manual steps

V30 stopped at 50/800 with 0.805717% structure improvement against the unchanged 1% early requirement.
Do not resume or rerun V30. This distinct diagnostic measures 280 gradients with
zero optimizer updates. It compares the original and stopped update50 decoder on the
first 50 actually exposed cases and 50 unexposed TRAIN cases matched by source
and degradation profile. Reference repetition is also matched one to one.
Selection uses the frozen schedule and input metadata.
All visible regions remain in scope; no held-out/native/final cases are used.

Use the existing NVIDIA L4/g2-standard-4 at ~/forensic-dgp. Require 4 GiB free and
no other GPU workload. Estimated 2-8 minutes of measurements plus archive export.
Enforced worker limit: 600s; external limit: 900s + 30s grace; export limit: 300s
(external export: 330s + 30s grace), allocated VRAM <=20 GiB, return <=1.5 GiB.
No optimizer, backward,
checkpoint writer, loss change, app promotion or automatic training is present.
Read the new return locally before selecting any later finite training pilot.
Original checkpoints, splits, logs and failed gates remain preserved.

Protocol SHA256: d01fdee0b8b47feed2edfa231625150af40249f39eb13e57b3fd12fba3d61611
Execution archive SHA256: 55f46f8a576637ed268cdf70e230127eabe9366653ec622dd27c54cb719e59dc

1. Upload from **Windows Google Cloud SDK Shell**:

```bat
cd /d "C:\xampp\htdocs\YEAR 4\Testing\outputs"
gcloud compute scp --project=forensic-dgp-thesis --zone=us-central1-a "cctv-dgp-v30-sampling-gradient-v1-execution.tar.gz" "janusdominic0@forensic-dgp-thesis:/home/janusdominic0/"
gcloud compute scp --project=forensic-dgp-thesis --zone=us-central1-a "cctv-dgp-v30-sampling-gradient-v1-execution.tar.gz.sha256" "janusdominic0@forensic-dgp-thesis:/home/janusdominic0/"
```

2. Install in **VM SSH**:

```bash
cd ~ &&
sha256sum -c cctv-dgp-v30-sampling-gradient-v1-execution.tar.gz.sha256 &&
test -d ~/forensic-dgp/cctv_dgp_vm_bundle/.venv &&
test -d ~/forensic-dgp/cctv_dgp_broader_mean_vm_v30/outputs/update50 &&
test ! -e ~/forensic-dgp/cctv_dgp_v30_sampling_gradient_v1_vm &&
tar -xzf cctv-dgp-v30-sampling-gradient-v1-execution.tar.gz -C ~/forensic-dgp
```

3. Open **tmux**:

```bash
tmux new-session -A -s dgp_v30_sampling
```

4. Launch **inside tmux**:

```bash
cd ~/forensic-dgp/cctv_dgp_v30_sampling_gradient_v1_vm &&
source ~/forensic-dgp/cctv_dgp_vm_bundle/.venv/bin/activate &&
python -B -u scripts/cctv_dgp_v30_sampling_gradient_v1_vm.py --root . --protocol-sha d01fdee0b8b47feed2edfa231625150af40249f39eb13e57b3fd12fba3d61611 --verify-transfer &&
bash scripts/run_sampling.sh d01fdee0b8b47feed2edfa231625150af40249f39eb13e57b3fd12fba3d61611
```

The final diagnostic should report 280 queries and 0 updates. Export `complete:true`
confirms packaging only; retain any `failure_present:true` result. Detach using
Ctrl+B, release both keys, then D. The finite worker continues inside tmux.

5. Download from **Windows Google Cloud SDK Shell** using three separate calls:

```bat
cd /d "C:\xampp\htdocs\YEAR 4\Testing\outputs"
gcloud compute scp --project=forensic-dgp-thesis --zone=us-central1-a "janusdominic0@forensic-dgp-thesis:/home/janusdominic0/cctv-dgp-v30-sampling-gradient-v1-results.tar.gz" "."
gcloud compute scp --project=forensic-dgp-thesis --zone=us-central1-a "janusdominic0@forensic-dgp-thesis:/home/janusdominic0/cctv-dgp-v30-sampling-gradient-v1-results.tar.gz.sha256" "."
gcloud compute scp --project=forensic-dgp-thesis --zone=us-central1-a "janusdominic0@forensic-dgp-thesis:/home/janusdominic0/cctv-dgp-v30-sampling-gradient-v1-export.json" "."
```

Three separate remote-source calls avoid Windows PuTTY's multiple-source error.
The local prospective checker verifies archive boundaries, source/data provenance,
all saved derivative arithmetic and 100 cases at both states through CPU inference
only. Derivative computation and any future training remain manual VM work.
The CPU check replays outputs and loss values; it does not recompute gradients.
