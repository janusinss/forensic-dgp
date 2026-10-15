# Hinge diagnostic completed and audited — historical commands

This immutable diagnostic is complete; do not rerun it unchanged.
[Audited hinge results](<C:/xampp/htdocs/YEAR 4/Testing/CCTV_DGP_BANK_HINGE_BOUNDARY_V1_RESULTS.md>).
[Use the prepared finite-forward R1 guide](<C:/xampp/htdocs/YEAR 4/Testing/CCTV_DGP_BANK_SAVED_DIRECTION_FORWARD_V1_R1_VM.md>).

---

# Inactive-guard diagnostic: five exact manual steps

**Prepared and independently transfer-verified; corrected L4 gradients are unrun.**
R1 completed and exposed a brightness guard with zero value but active gradient.
This packet tests the corrected boundary derivative with **zero optimizer updates**.
Scalar penalties, checkpoints, case lists and scientific gates remain preserved.
[Audited R1 findings and finite decision](<C:/xampp/htdocs/YEAR 4/Testing/CCTV_DGP_BANK_LOSS_ALIGNMENT_V1_R1_RESULTS.md>).

Upload **258,033 bytes**. Preserve both earlier loss diagnostics
and `~/forensic-dgp/cctv_dgp_bank_photometric_guard_v1_vm` (5,852 required assets).
Require **4 GiB free after installation** and idle L4. Estimate **2–5 minutes
measurement plus 1–3 minutes export**, unmeasured for this packet. Worker/
measurement/export caps 900/600/300 seconds (measurement checked between batches),
external 930/330 with 30-second kill grace; VRAM 20 GiB, output/reserve 1 GiB each.

1. Upload from **Windows Google Cloud SDK Shell**:

```bat
cd /d "C:\xampp\htdocs\YEAR 4\Testing\outputs"
gcloud compute scp --project=forensic-dgp-thesis --zone=us-central1-a "cctv-dgp-bank-hinge-boundary-v1-execution.tar.gz" "janusdominic0@forensic-dgp-thesis:/home/janusdominic0/"
gcloud compute scp --project=forensic-dgp-thesis --zone=us-central1-a "cctv-dgp-bank-hinge-boundary-v1-execution.tar.gz.sha256" "janusdominic0@forensic-dgp-thesis:/home/janusdominic0/"
```

2. Install in **VM SSH**:

```bash
cd ~ &&
tr -d '\r' < cctv-dgp-bank-hinge-boundary-v1-execution.tar.gz.sha256 | sha256sum -c - &&
test -d ~/forensic-dgp/cctv_dgp_vm_bundle/.venv &&
test -d ~/forensic-dgp/cctv_dgp_bank_photometric_guard_v1_vm &&
test ! -e ~/forensic-dgp/cctv_dgp_bank_hinge_boundary_v1_vm &&
tar -xzf cctv-dgp-bank-hinge-boundary-v1-execution.tar.gz -C ~/forensic-dgp
```

Archive SHA256: `e5f2fc6f9b65274117b0b244923ba7c103ccb7502b00f547d3fc84753844a404`.
Protocol SHA256: `1947d162217984694573bfcba8811f717630f66068474e6c9659d58c1a6b9099`. Installation deletes nothing.

3. Open **tmux from the normal VM SSH shell**:

```bash
tmux new-session -A -s dgp_bank_hinge_boundary_v1
```

4. Launch **inside tmux**:

```bash
cd ~/forensic-dgp/cctv_dgp_bank_hinge_boundary_v1_vm &&
source ~/forensic-dgp/cctv_dgp_vm_bundle/.venv/bin/activate &&
python -B -u scripts/cctv_dgp_bank_hinge_boundary_v1_vm.py --root . --parent-root ../cctv_dgp_bank_photometric_guard_v1_vm --protocol-sha 1947d162217984694573bfcba8811f717630f66068474e6c9659d58c1a6b9099 --verify-transfer &&
bash scripts/run_boundary.sh 1947d162217984694573bfcba8811f717630f66068474e6c9659d58c1a6b9099
```

Expected transfer receipt: `complete: true`, `packet_assets: 4`, `parent_assets: 5852`,
with zero neural calls/gradients/updates in that transfer check. Successful
measurement ends with **360 gradient queries and zero optimizer updates**.
Detach with **Ctrl+B, then D**; reattach with
`tmux attach -t dgp_bank_hinge_boundary_v1`. A stop is preserved/exported too.
Export `complete` confirms packaging only; there is no automatic training.

5. Download from **Windows Google Cloud SDK Shell**, one remote source per call:

```bat
cd /d "C:\xampp\htdocs\YEAR 4\Testing\outputs"
gcloud compute scp --project=forensic-dgp-thesis --zone=us-central1-a "janusdominic0@forensic-dgp-thesis:/home/janusdominic0/cctv-dgp-bank-hinge-boundary-v1-results.tar.gz" "."
gcloud compute scp --project=forensic-dgp-thesis --zone=us-central1-a "janusdominic0@forensic-dgp-thesis:/home/janusdominic0/cctv-dgp-bank-hinge-boundary-v1-results.tar.gz.sha256" "."
gcloud compute scp --project=forensic-dgp-thesis --zone=us-central1-a "janusdominic0@forensic-dgp-thesis:/home/janusdominic0/cctv-dgp-bank-hinge-boundary-v1-export.json" "."
```

Keep the three files together. The prepared independent return checker verifies
raw replay, scalar penalty parity and sampled gradient arithmetic before a new
training decision. Diagnostic success alone qualifies no model or extra epochs.
All five milestones and seven automatic/assisted completion families remain.
