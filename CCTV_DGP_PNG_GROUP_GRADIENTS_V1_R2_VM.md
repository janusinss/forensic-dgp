# Completed and audited R2 measurement — do not rerun

The manual measurement completed; its return and canonical CUDA repair are
audited. Zero weight updates or new checkpoint. [Results](<C:/xampp/htdocs/YEAR 4/Testing/CCTV_DGP_PNG_GROUP_GRADIENTS_V1_R2_RESULTS.md>).
Next is a distinct finite actual-weight comparison: [New manual guide](<C:/xampp/htdocs/YEAR 4/Testing/CCTV_DGP_JOINT_RAW_PNG_FORWARD_V1_VM.md>).
The earlier prepared/unrun text and commands below are preserved history.

# Canonical PNG-gradient measurement V1-R2 — manual L4 commands

**Verified locally and unrun on the VM.** R1 stopped at the CPU/CUDA conversion
boundary before gradients; preserve that used packet. R2 changes canonical
forward construction, adds a pre-model256-byte CUDA test, and keeps all losses,
cases, scientific gates and320-query limits. Zero optimizer updates, model
proposals, epochs or new checkpoints. [Audited cause and structure status](<C:/xampp/htdocs/YEAR 4/Testing/CCTV_DGP_PNG_GROUP_GRADIENTS_V1_R1_RESULTS.md>).

Require **4 GiB free after installation**. Estimated runtime3–8min. Measurement
cap600s/worker900s/outer930s; export300s/outer330s;30s kill grace per stage.
First two batches must project below600s. VRAM20GiB/RSS12GiB/output1,280MiB/
reserve1GiB; export additionally reserves the full payload. Current VM free space
is checked at run time. Upload **2,845,924 bytes**.

1. Upload from **Windows Google Cloud SDK Shell**:
```bat
cd /d "C:\xampp\htdocs\YEAR 4\Testing\outputs"
gcloud compute scp --project=forensic-dgp-thesis --zone=us-central1-a "cctv-dgp-png-group-gradients-v1-r2-execution.tar.gz" "janusdominic0@forensic-dgp-thesis:/home/janusdominic0/"
gcloud compute scp --project=forensic-dgp-thesis --zone=us-central1-a "cctv-dgp-png-group-gradients-v1-r2-execution.tar.gz.sha256" "janusdominic0@forensic-dgp-thesis:/home/janusdominic0/"
```

2. Install in **VM SSH**:
```bash
cd ~ &&
sha256sum -c cctv-dgp-png-group-gradients-v1-r2-execution.tar.gz.sha256 &&
test -d ~/forensic-dgp/cctv_dgp_vm_bundle/.venv &&
test -d ~/forensic-dgp/cctv_dgp_bank_photometric_guard_v1_vm &&
test ! -e ~/forensic-dgp/cctv_dgp_png_group_gradients_v1_r2_vm &&
tar -xzf cctv-dgp-png-group-gradients-v1-r2-execution.tar.gz -C ~/forensic-dgp
```

3. Open **tmux**:
```bash
tmux new-session -A -s dgp_png_group_gradients_v1_r2
```

4. Launch **inside tmux**:
```bash
cd ~/forensic-dgp/cctv_dgp_png_group_gradients_v1_r2_vm &&
source ~/forensic-dgp/cctv_dgp_vm_bundle/.venv/bin/activate &&
python -B -u scripts/cctv_dgp_png_group_gradients_v1_r2_vm.py --root . --parent-root ../cctv_dgp_bank_photometric_guard_v1_vm --protocol-sha f6abe2abd6cbeb38f5184a8959d074e991c2917507ac21c896b3abae86453817 --verify-transfer &&
bash scripts/run_png_groups.sh f6abe2abd6cbeb38f5184a8959d074e991c2917507ac21c896b3abae86453817
```

Hash-only preflight:6 packet assets,5,852 parent assets,32 groups and320 planned
queries. The run must pass256-byte CUDA parity before any model. Progress reaches
20 reference batches/320 queries only on measurement success. Export also runs
after a retained failure; export completion is not diagnostic or model success.
Detach with **Ctrl+B,D**; reattach with
`tmux attach -t dgp_png_group_gradients_v1_r2`.
The worker preserves competing GPU jobs and refuses used packets. Do not delete
previous outputs, rerun a failed recipe unchanged or bypass a stop.

5. Download from **Windows Google Cloud SDK Shell** after export:
```bat
cd /d "C:\xampp\htdocs\YEAR 4\Testing\outputs"
gcloud compute scp --project=forensic-dgp-thesis --zone=us-central1-a "janusdominic0@forensic-dgp-thesis:/home/janusdominic0/cctv-dgp-png-group-gradients-v1-r2-results.tar.gz" "cctv-dgp-png-group-gradients-v1-r2-results.tar.gz"
gcloud compute scp --project=forensic-dgp-thesis --zone=us-central1-a "janusdominic0@forensic-dgp-thesis:/home/janusdominic0/cctv-dgp-png-group-gradients-v1-r2-results.tar.gz.sha256" "cctv-dgp-png-group-gradients-v1-r2-results.tar.gz.sha256"
gcloud compute scp --project=forensic-dgp-thesis --zone=us-central1-a "janusdominic0@forensic-dgp-thesis:/home/janusdominic0/cctv-dgp-png-group-gradients-v1-r2-export.json" "cctv-dgp-png-group-gradients-v1-r2-export.json"
```

Each PuTTY download has one remote source. Return all three files for independent
pixel-table, exact raw/PNG control, scientific scalar/embedding, saved-gradient
group and timing checks. No follow-on training is automatic. The existing named
L4 VM is required; a different VM needs identity and migration verification.

Protocol SHA256: `f6abe2abd6cbeb38f5184a8959d074e991c2917507ac21c896b3abae86453817`.
Execution archive SHA256: `217ca0e2f97f1fac95a536292e0cd1a1ced66e5ed72823178a8596fe5bc1ce07`.
Checksum text is verified LF. [Transfer audit](<C:/xampp/htdocs/YEAR 4/Testing/outputs/cctv_dgp_png_group_gradients_v1_r2_transfer_audit.json>).
