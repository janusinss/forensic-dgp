# R1 executed and independently audited — historical commands

R1 completed; do not rerun this immutable diagnostic unchanged.
[Read the audited loss-direction result](<C:/xampp/htdocs/YEAR 4/Testing/CCTV_DGP_BANK_LOSS_ALIGNMENT_V1_R1_RESULTS.md>).
[Use the separately verified inactive-guard diagnostic guide](<C:/xampp/htdocs/YEAR 4/Testing/CCTV_DGP_BANK_HINGE_BOUNDARY_V1_VM.md>).

---

# Corrected loss-direction diagnostic R1: five manual steps

**Prepared and independently verified; actual L4 replay/gradients are unrun.**
The original diagnostic stopped before gradients because its input layout differed
from the retained baseline. R1 restores that layout and retains the exact-output
check. This is **zero-update diagnosis, not training or failed-state resume**.
[Audited failure and repair](<C:/xampp/htdocs/YEAR 4/Testing/CCTV_DGP_BANK_LOSS_ALIGNMENT_V1_RESULTS.md>).

Upload **256,341 bytes**. Preserve the original failed diagnostic
and `~/forensic-dgp/cctv_dgp_bank_photometric_guard_v1_vm` (5,852 bound assets).
Require **4 GiB free after installation** and idle L4. Unmeasured estimate:
**3–10 minutes measurement plus 1–5 minutes export**. Worker/measurement/export
limits are 900/600/300 seconds; measurement stop is checked between batches.
External stops 930/330 with 30-second kill grace. VRAM 20 GiB; output/reserve
1 GiB each. Full raw replay evidence is retained before any parity stop.

1. Upload from **Windows Google Cloud SDK Shell**:

```bat
cd /d "C:\xampp\htdocs\YEAR 4\Testing\outputs"
gcloud compute scp --project=forensic-dgp-thesis --zone=us-central1-a "cctv-dgp-bank-loss-alignment-v1-r1-execution.tar.gz" "janusdominic0@forensic-dgp-thesis:/home/janusdominic0/"
gcloud compute scp --project=forensic-dgp-thesis --zone=us-central1-a "cctv-dgp-bank-loss-alignment-v1-r1-execution.tar.gz.sha256" "janusdominic0@forensic-dgp-thesis:/home/janusdominic0/"
```

2. Install in **VM SSH**:

```bash
cd ~ &&
tr -d '\r' < cctv-dgp-bank-loss-alignment-v1-r1-execution.tar.gz.sha256 | sha256sum -c - &&
test -d ~/forensic-dgp/cctv_dgp_vm_bundle/.venv &&
test -d ~/forensic-dgp/cctv_dgp_bank_photometric_guard_v1_vm &&
test ! -e ~/forensic-dgp/cctv_dgp_bank_loss_alignment_v1_r1_vm &&
tar -xzf cctv-dgp-bank-loss-alignment-v1-r1-execution.tar.gz -C ~/forensic-dgp
```

Expected archive SHA256: `300030144f2a835a19f0c6258c99acc6dc4d2b0a6c5f18ab7c20ba10ce75251f`.
Expected protocol SHA256: `5b932745c73bd93d7ced1adcc0b5be9a53271e9fef86087dea638d697f972987`. Installation deletes nothing.

3. Open **tmux from the normal VM SSH shell**:

```bash
tmux new-session -A -s dgp_bank_loss_alignment_v1_r1
```

4. Launch **inside tmux**:

```bash
cd ~/forensic-dgp/cctv_dgp_bank_loss_alignment_v1_r1_vm &&
source ~/forensic-dgp/cctv_dgp_vm_bundle/.venv/bin/activate &&
python -B -u scripts/cctv_dgp_bank_loss_alignment_v1_r1_vm.py --root . --parent-root ../cctv_dgp_bank_photometric_guard_v1_vm --protocol-sha 5b932745c73bd93d7ced1adcc0b5be9a53271e9fef86087dea638d697f972987 --verify-transfer &&
bash scripts/run_alignment.sh 5b932745c73bd93d7ced1adcc0b5be9a53271e9fef86087dea638d697f972987
```

Expected transfer check: `complete: true`, `packet_assets: 3`,
`parent_assets: 5852`, `neural_calls: 0`, `gradient_queries: 0`,
`optimizer_updates: 0`. Successful measurement ends with **360 gradient queries,
zero optimizer updates**. Exact raw replay must pass first; no threshold is relaxed.
Detach with **Ctrl+B, then D**. Reattach with
`tmux attach -t dgp_bank_loss_alignment_v1_r1`. The shell exports a retained failure
too; export completion confirms packaging. It launches no subsequent training.

5. Download from **Windows Google Cloud SDK Shell**, one remote source per call:

```bat
cd /d "C:\xampp\htdocs\YEAR 4\Testing\outputs"
gcloud compute scp --project=forensic-dgp-thesis --zone=us-central1-a "janusdominic0@forensic-dgp-thesis:/home/janusdominic0/cctv-dgp-bank-loss-alignment-v1-r1-results.tar.gz" "."
gcloud compute scp --project=forensic-dgp-thesis --zone=us-central1-a "janusdominic0@forensic-dgp-the filethesis:/home/janusdominic0/cctv-dgp-bank-loss-alignment-v1-r1-results.tar.gz.sha256" "."
gcloud compute scp --project=forensic-dgp-thesis --zone=us-central1-a "janusdominic0@forensic-dgp-thesis:/home/janusdominic0/cctv-dgp-bank-loss-alignment-v1-r1-export.json" "."
```

Keep all three together for independent hash, frozen-replay and sampled-gradient
audits. Do not rerun the original packet, overwrite its failure, or promote a model
from diagnostic/export completion. Restoration, native DEV/final review, app flow
and all seven automatic/assisted completion families remain required.
