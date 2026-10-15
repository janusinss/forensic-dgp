# Superseding verified R2 status — 11 October 2026

R2 is completed and rejected for preservation, not prepared/unrun. Original
commands below are historical and must not be rerun unchanged. Larger proposals
exceed1% mean structure while regressing a source and preservation; smaller
ones also fail. All400 raw/PNG cases and20 sheets were checked. No checkpoint
or more-epoch qualification. [Audited R2 result](<C:/xampp/htdocs/YEAR 4/Testing/CCTV_DGP_BANK_SAVED_DIRECTION_FORWARD_V1_R2_RESULTS.md>).
[Fresh manual measurement](<C:/xampp/htdocs/YEAR 4/Testing/CCTV_DGP_BANK_GROUP_GRADIENTS_V1_VM.md>). Prior text is retained verbatim.

---

# Saved-direction finite forward R2: five exact manual steps

**Use the fresh R2 root.** The returned R1 alignment failure is retained and must
not be rerun unchanged. R2 restores the saved reference affine, checks all 100
matrices before neural import and saves raw outputs before identity processing.
Independent transfer, geometry, Bash and return-checker preparation checks passed.
Its actual VM forward proposals are **unrun**.
[Audited R1 cause and R2 repair](<C:/xampp/htdocs/YEAR 4/Testing/CCTV_DGP_BANK_SAVED_DIRECTION_FORWARD_V1_R1_RESULTS.md>).

Upload **20,633,183 bytes** (about 19.7 MiB). Require **4 GiB free
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
gcloud compute scp --project=forensic-dgp-thesis --zone=us-central1-a "cctv-dgp-bank-saved-direction-forward-v1-r2-execution.tar.gz" "janusdominic0@forensic-dgp-thesis:/home/janusdominic0/"
gcloud compute scp --project=forensic-dgp-thesis --zone=us-central1-a "cctv-dgp-bank-saved-direction-forward-v1-r2-execution.tar.gz.sha256" "janusdominic0@forensic-dgp-thesis:/home/janusdominic0/"
```

2. Install in **VM SSH**:

```bash
cd ~ &&
tr -d '\r' < cctv-dgp-bank-saved-direction-forward-v1-r2-execution.tar.gz.sha256 | sha256sum -c - &&
test -d ~/forensic-dgp/cctv_dgp_vm_bundle/.venv &&
test -d ~/forensic-dgp/cctv_dgp_bank_photometric_guard_v1_vm &&
test ! -e ~/forensic-dgp/cctv_dgp_bank_saved_direction_forward_v1_r2_vm &&
tar -xzf cctv-dgp-bank-saved-direction-forward-v1-r2-execution.tar.gz -C ~/forensic-dgp
```

Archive SHA256: `077884f14b374f5018ec71caeed1adbd764750d5191dd3895e58389f614cd183`.
Protocol SHA256: `52e0bba3b4e6f203bdfef6a2793fe851280feb0574982a64afa5eee587696641`. Installation deletes nothing.

3. Open **tmux from the normal VM SSH shell**:

```bash
tmux new-session -A -s dgp_saved_direction_v1_r2
```

4. Launch **inside tmux**:

```bash
cd ~/forensic-dgp/cctv_dgp_bank_saved_direction_forward_v1_r2_vm &&
source ~/forensic-dgp/cctv_dgp_vm_bundle/.venv/bin/activate &&
python -B -u scripts/cctv_dgp_bank_saved_direction_forward_v1_r2_vm.py --root . --parent-root ../cctv_dgp_bank_photometric_guard_v1_vm --protocol-sha 52e0bba3b4e6f203bdfef6a2793fe851280feb0574982a64afa5eee587696641 --verify-transfer &&
bash scripts/run_forward.sh 52e0bba3b4e6f203bdfef6a2793fe851280feb0574982a64afa5eee587696641
```

Expected transfer: `complete: true`, `packet_assets: 4`, `parent_assets: 5852`,
`fixed_target_affines_checked: 100`, with zero neural/gradient/optimizer calls in that transfer check. Successful
finite execution has 100 B and 160 recognizer forwards, four temporary proposals,
zero gradient queries/backwards/optimizer updates and no checkpoint. Export
`complete` confirms packaging; passing restoration is assessed after return.
Detach with **Ctrl+B, then D**; reattach with
`tmux attach -t dgp_saved_direction_v1_r2`. Stops are retained/exported.

5. Download from **Windows Google Cloud SDK Shell**, one remote source per call:

```bat
cd /d "C:\xampp\htdocs\YEAR 4\Testing\outputs"
gcloud compute scp --project=forensic-dgp-thesis --zone=us-central1-a "janusdominic0@forensic-dgp-thesis:/home/janusdominic0/cctv-dgp-bank-saved-direction-forward-v1-r2-results.tar.gz" "cctv-dgp-bank-saved-direction-forward-v1-r2-results.tar.gz"
gcloud compute scp --project=forensic-dgp-thesis --zone=us-central1-a "janusdominic0@forensic-dgp-thesis:/home/janusdominic0/cctv-dgp-bank-saved-direction-forward-v1-r2-results.tar.gz.sha256" "cctv-dgp-bank-saved-direction-forward-v1-r2-results.tar.gz.sha256"
gcloud compute scp --project=forensic-dgp-thesis --zone=us-central1-a "janusdominic0@forensic-dgp-thesis:/home/janusdominic0/cctv-dgp-bank-saved-direction-forward-v1-r2-export.json" "cctv-dgp-bank-saved-direction-forward-v1-r2-export.json"
```

The explicit local filenames prevent another accidental `.5` destination. Keep
all three files together. Do not rerun a used root or resume a failed historical
state. Independent audit and visual review precede any new training decision.
The full native/final/app and seven-family completion goal remains active.
