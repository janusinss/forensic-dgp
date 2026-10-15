# Completed signed diagnostic — historical commands, do not rerun

The user manually completed this immutable test. Its returned archive is audited;
all four proposals remain unqualified. +.0001 preserves raw checks but fails6
delivered checks and remains below1% structure. No training epoch or checkpoint
was created. [13 October result and next design](<C:/xampp/htdocs/YEAR 4/Testing/CCTV_DGP_BANK_GROUP_STEP_CHECK_V1_RESULTS.md>).
The original prepared/unrun status and commands below remain as historical bytes.

# Signed small-step diagnosis — exact manual L4 commands

**Verified and UNRUN.** This checks why predicted improvement failed to preserve
the finite outputs. Four signed disposable proposals; zero optimizer updates,
gradients,epochs or new checkpoints. Original checkpoints and failed runs stay
retained. Require **4 GiB free after install**. Estimate **2–5 minutes**.
Worker/measurement1200/900s,outer1230s; export300/outer330s plus30s grace;
VRAM20GiB/RSS12GiB/output768MiB/reserve1GiB. Upload **2,839,068 bytes**.

1. Upload from **Windows Google Cloud SDK Shell**:
```bat
cd /d "C:\xampp\htdocs\YEAR 4\Testing\outputs"
gcloud compute scp --project=forensic-dgp-thesis --zone=us-central1-a "cctv-dgp-bank-group-step-check-v1-execution.tar.gz" "janusdominic0@forensic-dgp-thesis:/home/janusdominic0/"
gcloud compute scp --project=forensic-dgp-thesis --zone=us-central1-a "cctv-dgp-bank-group-step-check-v1-execution.tar.gz.sha256" "janusdominic0@forensic-dgp-thesis:/home/janusdominic0/"
```

2. Install in **VM SSH**:
```bash
cd ~ &&
sha256sum -c cctv-dgp-bank-group-step-check-v1-execution.tar.gz.sha256 &&
test -d ~/forensic-dgp/cctv_dgp_vm_bundle/.venv &&
test -d ~/forensic-dgp/cctv_dgp_bank_photometric_guard_v1_vm &&
test ! -e ~/forensic-dgp/cctv_dgp_bank_group_step_check_v1_vm &&
tar -xzf cctv-dgp-bank-group-step-check-v1-execution.tar.gz -C ~/forensic-dgp
```

3. Open **tmux**:
```bash
tmux new-session -A -s dgp_group_step_check_v1
```

4. Launch **inside tmux**:
```bash
cd ~/forensic-dgp/cctv_dgp_bank_group_step_check_v1_vm &&
source ~/forensic-dgp/cctv_dgp_vm_bundle/.venv/bin/activate &&
python -B -u scripts/cctv_dgp_bank_group_step_check_v1_vm.py --root . --parent-root ../cctv_dgp_bank_photometric_guard_v1_vm --protocol-sha 72ad815c6514cfb54115e7f5f519ec99ad5f968cc682b3875168149ca9f2f215 --verify-transfer &&
bash scripts/run_forward.sh 72ad815c6514cfb54115e7f5f519ec99ad5f968cc682b3875168149ca9f2f215
```

Preflight reports available/required bytes. Detach with **Ctrl+B,D**;
reattach with `tmux attach -t dgp_group_step_check_v1`. Export runs after a
measurement success or retained stop. A complete export is not a quality pass.
Do not repeat this used packet or bypass a stop. If changing VMs, verify the
migration and new VM identity first; these commands target the existing L4 VM.

5. Download from **Windows Google Cloud SDK Shell** after export finishes:
```bat
cd /d "C:\xampp\htdocs\YEAR 4\Testing\outputs"
gcloud compute scp --project=forensic-dgp-thesis --zone=us-central1-a "janusdominic0@forensic-dgp-thesis:/home/janusdominic0/cctv-dgp-bank-group-step-check-v1-results.tar.gz" "cctv-dgp-bank-group-step-check-v1-results.tar.gz"
gcloud compute scp --project=forensic-dgp-thesis --zone=us-central1-a "janusdominic0@forensic-dgp-thesis:/home/janusdominic0/cctv-dgp-bank-group-step-check-v1-results.tar.gz.sha256" "cctv-dgp-bank-group-step-check-v1-results.tar.gz.sha256"
gcloud compute scp --project=forensic-dgp-thesis --zone=us-central1-a "janusdominic0@forensic-dgp-thesis:/home/janusdominic0/cctv-dgp-bank-group-step-check-v1-export.json" "cctv-dgp-bank-group-step-check-v1-export.json"
```

Each download has one remote source for Windows PuTTY. Return all three files
for independent raw/PNG,32-group signed-derivative,rounding,preservation and
all20-sheet visual review. Negative steps are diagnosis only. The1% structure
and visible-preservation requirements remain unchanged; no more epochs are
automatically authorized or launched.

Protocol SHA256: `72ad815c6514cfb54115e7f5f519ec99ad5f968cc682b3875168149ca9f2f215`.
Execution archive SHA256: `0f6342d2066bc3781c53c3f49efe57a681aac6531565ed5edc0a78ad0ab7a8c6`.
The checksum file is verified LF text to avoid the previous CRLF filename error.
[Audited finite failure and diagnostic purpose](<C:/xampp/htdocs/YEAR 4/Testing/CCTV_DGP_BANK_GROUP_DIRECTION_FORWARD_V1_RESULTS.md>).
