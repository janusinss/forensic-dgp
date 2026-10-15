# R1 returned an audited alignment failure — historical commands

Preserve this failed root; do not rerun its unchanged worker.
[Failure cause and repaired protocol](<C:/xampp/htdocs/YEAR 4/Testing/CCTV_DGP_BANK_SAVED_DIRECTION_FORWARD_V1_R1_RESULTS.md>).
[Use the fresh R2 manual guide](<C:/xampp/htdocs/YEAR 4/Testing/CCTV_DGP_BANK_SAVED_DIRECTION_FORWARD_V1_R2_VM.md>).

---

# Saved-direction finite forward R1: five exact manual steps

**Use R1 only.** The earlier V1 draft stays archived and unrun because its PNG
rounding policy differed from the retained baseline. This R1 packet passed
independent transfer and return-checker preparation checks. Its actual VM
forward proposals are **unrun**. [Audited hinge results and reason for this test](<C:/xampp/htdocs/YEAR 4/Testing/CCTV_DGP_BANK_HINGE_BOUNDARY_V1_RESULTS.md>).

Upload **20,632,115 bytes** (about 19.7 MiB). Require **4 GiB free
after installation** and idle L4. Estimate **2–6 minutes measurement plus 1–3
minutes export**, unmeasured for this packet. Worker/measurement/export stops
1200/900/300 seconds; external 1230/330 with 30-second kill grace. VRAM cap 20 GiB,
output 768 MiB, reserve 1 GiB. Preserve the parent and historical evidence.

This compares four temporary parameter proposals on 100 paired TRAIN cases,
with zero new gradient queries or optimizer steps. It restores the initial state
between proposals and at completion, saves no checkpoint and launches no training.
Raw/PNG metrics and all 20 visual sheets must be reviewed afterward.

1. Upload from **Windows Google Cloud SDK Shell**:

```bat
cd /d "C:\xampp\htdocs\YEAR 4\Testing\outputs"
gcloud compute scp --project=forensic-dgp-thesis --zone=us-central1-a "cctv-dgp-bank-saved-direction-forward-v1-r1-execution.tar.gz" "janusdominic0@forensic-dgp-thesis:/home/janusdominic0/"
gcloud compute scp --project=forensic-dgp-thesis --zone=us-central1-a "cctv-dgp-bank-saved-direction-forward-v1-r1-execution.tar.gz.sha256" "janusdominic0@forensic-dgp-thesis:/home/janusdominic0/"
```

2. Install in **VM SSH**:

```bash
cd ~ &&
tr -d '\r' < cctv-dgp-bank-saved-direction-forward-v1-r1-execution.tar.gz.sha256 | sha256sum -c - &&
test -d ~/forensic-dgp/cctv_dgp_vm_bundle/.venv &&
test -d ~/forensic-dgp/cctv_dgp_bank_photometric_guard_v1_vm &&
test ! -e ~/forensic-dgp/cctv_dgp_bank_saved_direction_forward_v1_r1_vm &&
tar -xzf cctv-dgp-bank-saved-direction-forward-v1-r1-execution.tar.gz -C ~/forensic-dgp
```

Archive SHA256: `4912d3d9c0f5f1c218210658ed20f4c7f19888e2c05ad21dc0aff014c6f5149a`.
Protocol SHA256: `0d45d64383c3b321ee50c099cdf0b0cb677289c52f6864c7943e2cbde37383a0`. Installation deletes nothing.

3. Open **tmux from the normal VM SSH shell**:

```bash
tmux new-session -A -s dgp_saved_direction_v1_r1
```

4. Launch **inside tmux**:

```bash
cd ~/forensic-dgp/cctv_dgp_bank_saved_direction_forward_v1_r1_vm &&
source ~/forensic-dgp/cctv_dgp_vm_bundle/.venv/bin/activate &&
python -B -u scripts/cctv_dgp_bank_saved_direction_forward_v1_r1_vm.py --root . --parent-root ../cctv_dgp_bank_photometric_guard_v1_vm --protocol-sha 0d45d64383c3b321ee50c099cdf0b0cb677289c52f6864c7943e2cbde37383a0 --verify-transfer &&
bash scripts/run_forward.sh 0d45d64383c3b321ee50c099cdf0b0cb677289c52f6864c7943e2cbde37383a0
```

Expected transfer: `complete: true`, `packet_assets: 4`, `parent_assets: 5852`,
with zero neural/gradient/optimizer calls in that transfer check. Successful
finite execution has 100 B and 160 recognizer forwards, four temporary proposals,
zero gradient queries/backwards/optimizer updates and no checkpoint. Export
`complete` confirms packaging; passing restoration is assessed after return.
Detach with **Ctrl+B, then D**; reattach with
`tmux attach -t dgp_saved_direction_v1_r1`. Stops are retained/exported.

5. Download from **Windows Google Cloud SDK Shell**, one remote source per call:

```bat
cd /d "C:\xampp\htdocs\YEAR 4\Testing\outputs"
gcloud compute scp --project=forensic-dgp-thesis --zone=us-central1-a "janusdominic0@forensic-dgp-thesis:/home/janusdominic0/cctv-dgp-bank-saved-direction-forward-v1-r1-results.tar.gz" "cctv-dgp-bank-saved-direction-forward-v1-r1-results.tar.gz"
gcloud compute scp --project=forensic-dgp-thesis --zone=us-central1-a "janusdominic0@forensic-dgp-thesis:/home/janusdominic0/cctv-dgp-bank-saved-direction-forward-v1-r1-results.tar.gz.sha256" "cctv-dgp-bank-saved-direction-forward-v1-r1-results.tar.gz.sha256"
gcloud compute scp --project=forensic-dgp-thesis --zone=us-central1-a "janusdominic0@forensic-dgp-thesis:/home/janusdominic0/cctv-dgp-bank-saved-direction-forward-v1-r1-export.json" "cctv-dgp-bank-saved-direction-forward-v1-r1-export.json"
```

The explicit local filenames prevent another accidental `.5` destination. Keep
all three files together. Do not rerun a used root or resume a failed historical
state. Independent audit and visual review precede any new training decision.
The full native/final/app and seven-family completion goal remains active.
