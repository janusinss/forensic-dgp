# Returned and audited — completed diagnostic; do not rerun

Live conditioning R1 passed activation and100 exact starting controls with zero
training updates. [Returned audit](<C:/xampp/htdocs/YEAR 4/Testing/CCTV_DGP_LIVE_CONDITIONING_V1_R1_RESULTS.md>).
[Next finite manual learning pilot](<C:/xampp/htdocs/YEAR 4/Testing/CCTV_DGP_CONDITIONING_LEARNING_V1_R2_VM.md>).
The completed diagnostic commands below are retained historical instructions.

# Live conditioning R1 diagnostic — manual L4 commands

**Corrected R1, verified and UNRUN. This is an architecture/gradient diagnostic, not training.**

R1 fixes an inherited export filename prefix. Its worker, download and importer
names agree. The original V1 packet and failed command audit are retained.
Architecture, cases, initialization, gradients and all stop/quality limits are
unchanged. Original human VM execution is unknown; this fresh directory preserves it.
It tests a changed feature path after the rejected joint comparison.
[Audited results and design rationale](<C:/xampp/htdocs/YEAR 4/Testing/CCTV_DGP_JOINT_RAW_PNG_FORWARD_V1_RESULTS.md>).

Same100 paired TRAIN cases;32 loss groups;320 gradient queries. Zero optimizer
updates, displacements, committed updates, epochs or new checkpoints. Starting
outputs must match the original DGP exactly. All12 style/spatial tensors must
receive finite nonzero detail and clear-preservation gradients for each source.
Activation alone is not useful restoration, PNG preservation or epoch approval.

Require **4 GiB free after installation**. Estimate **3–6 minutes** plus export.
Measurement/worker600/900s; outer930s. Export300s/outer330s plus30s kill grace.
VRAM20GiB/RSS12GiB/output1280MiB/reserve1GiB. Existing workloads and old outputs
are preserved. Upload **20,684,799 bytes**. No current free-space
or power-state claim is made; the VM checks them at execution.

1. Upload from **Windows Google Cloud SDK Shell**:
```bat
cd /d "C:\xampp\htdocs\YEAR 4\Testing\outputs"
gcloud compute scp --project=forensic-dgp-thesis --zone=us-central1-a "cctv-dgp-live-conditioning-v1-r1-execution.tar.gz" "janusdominic0@forensic-dgp-thesis:/home/janusdominic0/"
gcloud compute scp --project=forensic-dgp-thesis --zone=us-central1-a "cctv-dgp-live-conditioning-v1-r1-execution.tar.gz.sha256" "janusdominic0@forensic-dgp-thesis:/home/janusdominic0/"
```

2. Install in **VM SSH**:
```bash
cd ~ &&
sha256sum -c cctv-dgp-live-conditioning-v1-r1-execution.tar.gz.sha256 &&
test -d ~/forensic-dgp/cctv_dgp_vm_bundle/.venv &&
test -d ~/forensic-dgp/cctv_dgp_bank_photometric_guard_v1_vm &&
test ! -e ~/forensic-dgp/cctv_dgp_live_conditioning_v1_r1_vm &&
tar -xzf cctv-dgp-live-conditioning-v1-r1-execution.tar.gz -C ~/forensic-dgp
```

3. Open **tmux**:
```bash
tmux new-session -A -s dgp_live_conditioning_v1_r1
```

4. Launch **inside tmux**:
```bash
cd ~/forensic-dgp/cctv_dgp_live_conditioning_v1_r1_vm &&
source ~/forensic-dgp/cctv_dgp_vm_bundle/.venv/bin/activate &&
python -B -u scripts/cctv_dgp_live_conditioning_v1_r1_vm.py --root . --parent-root ../cctv_dgp_bank_photometric_guard_v1_vm --protocol-sha 754591c33b56e97d0b9f63336a3219ca12ddb316ddf6dcc2d883a34fda3c2d5f --verify-transfer &&
bash scripts/run_conditioning.sh 754591c33b56e97d0b9f63336a3219ca12ddb316ddf6dcc2d883a34fda3c2d5f
```

Progress visits20 reference batches and ends with320 queries. Export runs after
completion or a retained stop. Its `complete: true` certifies the archive, not
the diagnostic or model. Detach **Ctrl+B,D**; reattach with
`tmux attach -t dgp_live_conditioning_v1_r1`. No automatic training follows.

5. Download from **Windows Google Cloud SDK Shell** after export:
```bat
cd /d "C:\xampp\htdocs\YEAR 4\Testing\outputs"
gcloud compute scp --project=forensic-dgp-thesis --zone=us-central1-a "janusdominic0@forensic-dgp-thesis:/home/janusdominic0/cctv-dgp-live-conditioning-v1-r1-results.tar.gz" "cctv-dgp-live-conditioning-v1-r1-results.tar.gz"
gcloud compute scp --project=forensic-dgp-thesis --zone=us-central1-a "janusdominic0@forensic-dgp-thesis:/home/janusdominic0/cctv-dgp-live-conditioning-v1-r1-results.tar.gz.sha256" "cctv-dgp-live-conditioning-v1-r1-results.tar.gz.sha256"
gcloud compute scp --project=forensic-dgp-thesis --zone=us-central1-a "janusdominic0@forensic-dgp-thesis:/home/janusdominic0/cctv-dgp-live-conditioning-v1-r1-export.json" "cctv-dgp-live-conditioning-v1-r1-export.json"
```

Each PuTTY download has one remote source. Return all three files for an
independent audit of baseline parity, states, gradients, controls and resource
stops. Do not rerun this used diagnostic, delete failure records or bypass a
stop. The named L4/g2-standard-4 is required; a new VM needs migration/identity
verification. Agent access remains storage maintenance only.

Protocol SHA256: `754591c33b56e97d0b9f63336a3219ca12ddb316ddf6dcc2d883a34fda3c2d5f`.
Execution archive SHA256: `50a9e19dc9b5bdf48c784527b0e458cb55deb2a61dcc400e7a532f55909660e8`.
[Transfer/state evidence](<C:/xampp/htdocs/YEAR 4/Testing/outputs/cctv_dgp_live_conditioning_v1_r1_transfer_audit.json>);
[Actual old-gradient/adversarial gate fixtures](<C:/xampp/htdocs/YEAR 4/Testing/outputs/cctv_dgp_live_conditioning_v1_r1_gate_fixtures.json>).
