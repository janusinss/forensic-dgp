# V39 spatial decoder gradient proof: five manual steps

V38 passes two small TRAIN displacement checks, but its fixed quarter step fails
paired development preservation and adds no convincing native clarity. V39 tests
a different spatial path within our DGP: observed256 RGB plus frozen original
multiscale features, a two-scale16/32-channel gated decoder, and subtraction of
its separately frozen initial response. It uses no pretrained-restorer targets
or pretrained NAFNet weights. Initial raw/PNG parity is independently verified
on all50 already-exposed TRAIN cases. All17,952 new parameters start untrained.

**This is70 gradient queries, zero optimizer/parameter updates and no epochs.**
All57 new tensors must receive finite nonzero improvement gradients; all initial
preservation values/gradients must remain exactly zero. Original DGP, fixed initial
decoder and recognizer must remain unchanged. Connectivity is not a quality pass.
The1%-at50/10%-at800 and all preservation gates remain for any later training.

Require the existing idle **NVIDIA L4/g2-standard-4** and **2GiB free**. The packet
contains its own verified model/data sources; only the existing Python venv is
required. No deleted historical pilot or research cache is recreated or rerun.
Estimated diagnostic **2–6 minutes**, export **1–2 minutes**. Worker600s;
external630s plus30s kill grace; export300s/external330s plus30s grace; allocated
VRAM20GiB; uncompressed return192MiB. A failed proof is exported and retained.
No cleanup, automatic follow-on or new actual training is launched by the agent.

Protocol SHA256: `d1775d67c8c80ddad9372e5af47be6ce987ca3fed7749defcdbdc190fb1153a4`
Execution archive SHA256: `d2093fc136d5305db5071a53be7fbdb0722e5505026051c68501b57959cc29bc`

1. Upload from **Windows Google Cloud SDK Shell**:

```bat
cd /d "C:\xampp\htdocs\YEAR 4\Testing\outputs"
gcloud compute scp --project=forensic-dgp-thesis --zone=us-central1-a "cctv-dgp-spatial-decoder-v39-execution.tar.gz" "janusdominic0@forensic-dgp-thesis:/home/janusdominic0/"
gcloud compute scp --project=forensic-dgp-thesis --zone=us-central1-a "cctv-dgp-spatial-decoder-v39-execution.tar.gz.sha256" "janusdominic0@forensic-dgp-thesis:/home/janusdominic0/"
```

2. Install in **VM SSH**:

```bash
cd ~ &&
sha256sum -c cctv-dgp-spatial-decoder-v39-execution.tar.gz.sha256 &&
test -d ~/forensic-dgp/cctv_dgp_vm_bundle/.venv &&
test ! -e ~/forensic-dgp/cctv_dgp_spatial_decoder_vm_v39 &&
tar -xzf cctv-dgp-spatial-decoder-v39-execution.tar.gz -C ~/forensic-dgp
```

3. Open **tmux**:

```bash
tmux new-session -A -s dgp_spatial_decoder_v39
```

4. Launch **inside tmux**:

```bash
cd ~/forensic-dgp/cctv_dgp_spatial_decoder_vm_v39 &&
source ~/forensic-dgp/cctv_dgp_vm_bundle/.venv/bin/activate &&
python -B -u scripts/cctv_dgp_spatial_decoder_v39_vm.py --root . --protocol-sha d1775d67c8c80ddad9372e5af47be6ce987ca3fed7749defcdbdc190fb1153a4 --verify-transfer &&
bash scripts/run_v39_gradient.sh d1775d67c8c80ddad9372e5af47be6ce987ca3fed7749defcdbdc190fb1153a4
```

Detach with **Ctrl+B**, release, then **D**. Step3 reconnects. The log prints
`V39_batch`1 through10 and70 gradient queries if completed. An export
`complete:true` means packaging, with optimizer_updates:0. Download failure
evidence too. Do not edit, resume or repeat the frozen packet unchanged.

5. Download from **Windows Google Cloud SDK Shell** after export completes:

```bat
cd /d "C:\xampp\htdocs\YEAR 4\Testing\outputs"
gcloud compute scp --project=forensic-dgp-thesis --zone=us-central1-a "janusdominic0@forensic-dgp-thesis:/home/janusdominic0/cctv-dgp-spatial-decoder-v39-results.tar.gz" "."
gcloud compute scp --project=forensic-dgp-thesis --zone=us-central1-a "janusdominic0@forensic-dgp-thesis:/home/janusdominic0/cctv-dgp-spatial-decoder-v39-results.tar.gz.sha256" "."
gcloud compute scp --project=forensic-dgp-thesis --zone=us-central1-a "janusdominic0@forensic-dgp-thesis:/home/janusdominic0/cctv-dgp-spatial-decoder-v39-export.json" "."
```

Each SCP has one remote source for Windows PuTTY. The prospective independent
checker verifies every returned file, all70 saved component queries,57 tensor
partitions, baseline PNGs, unchanged states and ten frozen CPU replay cases.
It runs no local gradients or optimizer and never executes returned Python.
A returned proof is required before choosing a new finite actual-training recipe.
Native useful structure, independent final review and separate automatic/assisted
quality for all seven covering families remain required. Goal active/incomplete.

[V38 development rejection](<C:/xampp/htdocs/YEAR 4/Testing/CCTV_DGP_V38_QUARTER_DEVELOPMENT_RESULTS.md>)
[V39 spatial design review](<C:/xampp/htdocs/YEAR 4/Testing/CCTV_DGP_POST_V38_SPATIAL_DECODER_REVIEW.md>)
