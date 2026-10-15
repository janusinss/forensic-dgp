# Completed and rejected — 12 October 2026

This packet completed its manual run;all four proposals failed preservation
and structure requirements. **Do not rerun the historical commands below.**
The original bytes are retained. [Verified failure report](<C:/xampp/htdocs/YEAR 4/Testing/CCTV_DGP_BANK_GROUP_DIRECTION_FORWARD_V1_RESULTS.md>).
Use the fresh [signed diagnostic manual guide](<C:/xampp/htdocs/YEAR 4/Testing/CCTV_DGP_BANK_GROUP_STEP_CHECK_V1_VM.md>) for the next
bounded diagnosis. It is UNRUN and does not authorize additional epochs.

# Four source/profile direction probes — manual L4 steps

**Prepared; not launched by the agent.** This is a forward-only comparison of
four disposable parameter proposals, not training or additional epochs.
Upload **2,838,507 bytes**. Require **4 GiB free after install**.
Estimate **2–5 minutes**, unmeasured for this new direction. Worker/measurement
1200/900s, outer1230s; export300s/outer330s plus30s kill grace.
VRAM20GiB/RSS12GiB/output768MiB/reserve1GiB. No optimizer, gradients or checkpoints.
Preserve every prior used run and failed gate. Do not bypass a stop or rerun
this immutable packet after it produces outputs/logs.

1. Upload from **Windows Google Cloud SDK Shell**:
```bat
cd /d "C:\xampp\htdocs\YEAR 4\Testing\outputs"
gcloud compute scp --project=forensic-dgp-thesis --zone=us-central1-a "cctv-dgp-bank-group-direction-forward-v1-execution.tar.gz" "janusdominic0@forensic-dgp-thesis:/home/janusdominic0/"
gcloud compute scp --project=forensic-dgp-thesis --zone=us-central1-a "cctv-dgp-bank-group-direction-forward-v1-execution.tar.gz.sha256" "janusdominic0@forensic-dgp-thesis:/home/janusdominic0/"
```

2. Install in **VM SSH**:
```bash
cd ~ &&
sha256sum -c cctv-dgp-bank-group-direction-forward-v1-execution.tar.gz.sha256 &&
test -d ~/forensic-dgp/cctv_dgp_vm_bundle/.venv &&
test -d ~/forensic-dgp/cctv_dgp_bank_photometric_guard_v1_vm &&
test ! -e ~/forensic-dgp/cctv_dgp_bank_group_direction_forward_v1_vm &&
tar -xzf cctv-dgp-bank-group-direction-forward-v1-execution.tar.gz -C ~/forensic-dgp
```

3. Open **tmux**:
```bash
tmux new-session -A -s dgp_group_direction_v1
```

4. Launch **inside tmux**:
```bash
cd ~/forensic-dgp/cctv_dgp_bank_group_direction_forward_v1_vm &&
source ~/forensic-dgp/cctv_dgp_vm_bundle/.venv/bin/activate &&
python -B -u scripts/cctv_dgp_bank_group_direction_forward_v1_vm.py --root . --parent-root ../cctv_dgp_bank_photometric_guard_v1_vm --protocol-sha 92783a76effd2409d4a77433a6a4f52ea80e48628dffc444716e3e6df2d7bd9e --verify-transfer &&
bash scripts/run_forward.sh 92783a76effd2409d4a77433a6a4f52ea80e48628dffc444716e3e6df2d7bd9e
```

The preflight reports available and required bytes. Detach with **Ctrl+B, D**;
reattach with `tmux attach -t dgp_group_direction_v1`. The exporter runs after
success or a retained diagnostic stop. Export completion does not establish
image quality or authorize more epochs.

5. Download from **Windows Google Cloud SDK Shell** after export finishes:
```bat
cd /d "C:\xampp\htdocs\YEAR 4\Testing\outputs"
gcloud compute scp --project=forensic-dgp-thesis --zone=us-central1-a "janusdominic0@forensic-dgp-thesis:/home/janusdominic0/cctv-dgp-bank-group-direction-forward-v1-results.tar.gz" "cctv-dgp-bank-group-direction-forward-v1-results.tar.gz"
gcloud compute scp --project=forensic-dgp-thesis --zone=us-central1-a "janusdominic0@forensic-dgp-thesis:/home/janusdominic0/cctv-dgp-bank-group-direction-forward-v1-results.tar.gz.sha256" "cctv-dgp-bank-group-direction-forward-v1-results.tar.gz.sha256"
gcloud compute scp --project=forensic-dgp-thesis --zone=us-central1-a "janusdominic0@forensic-dgp-thesis:/home/janusdominic0/cctv-dgp-bank-group-direction-forward-v1-export.json" "cctv-dgp-bank-group-direction-forward-v1-export.json"
```

Each download has one remote source, compatible with Windows PuTTY.
Return all three files for independent raw/PNG, source-specific gate and all20
sheet review. A newly inspected result is required before training is proposed.

Protocol SHA256: `92783a76effd2409d4a77433a6a4f52ea80e48628dffc444716e3e6df2d7bd9e`.
Execution archive SHA256: `528e3d7e957d5c85b0a29544eddc7c8b5dc5407711a33da44054f51079da721a`.
[Audited reason for this changed probe](<C:/xampp/htdocs/YEAR 4/Testing/CCTV_DGP_BANK_GROUP_GRADIENTS_V1_R1_RESULTS.md>).
