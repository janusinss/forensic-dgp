# Executed and audited — 11 October 2026

This historical frozen diagnostic completed successfully and all returned outputs
were independently audited and viewed. No variant qualified; **do not rerun it**.
Its original prepared commands below are retained as exact historical bytes.
[Current evidence and diagnosis](<C:/xampp/htdocs/YEAR 4/Testing/CCTV_DGP_BANK_PATH_ABLATION_V1_RESULTS.md>).
[New unrun manual loss-direction diagnostic](<C:/xampp/htdocs/YEAR 4/Testing/CCTV_DGP_BANK_LOSS_ALIGNMENT_V1_VM.md>).

---

# Frozen bank-path diagnostic: manual VM commands

**Prepared and independently transfer-verified; actual VM inference is unrun.**
This measures the existing rejected B50 state with fixed weights. It performs
zero training/gradient updates and cannot promote or resume that failed recipe.
[Audited result and diagnostic rationale](<C:/xampp/htdocs/YEAR 4/Testing/CCTV_DGP_BANK_PHOTOMETRIC_GUARD_V1_RESULTS.md>).

Upload only **253,174 bytes**. The diagnostic reads **5,852
hash-bound files** from `~/forensic-dgp/cctv_dgp_bank_photometric_guard_v1_vm`;
preserve that root, its state and saved outputs. It copies no large research asset.
Require **4 GiB free after installation** and an idle L4. No current VM storage
or power state is claimed. Estimate **5–15 minutes inference + 1–5 minutes export**;
new timing is unmeasured. Worker/inference/export stops are 20/15/5 minutes,
external worker/export caps 20.5/5.5, kill grace 30 seconds, VRAM 20 GiB,
output 1 GiB and disk reserve 1 GiB. No automatic follow-on training.

1. Upload from **Windows Google Cloud SDK Shell**:

```bat
cd /d "C:\xampp\htdocs\YEAR 4\Testing\outputs"
gcloud compute scp --project=forensic-dgp-thesis --zone=us-central1-a "cctv-dgp-bank-path-ablation-v1-execution.tar.gz" "janusdominic0@forensic-dgp-thesis:/home/janusdominic0/"
gcloud compute scp --project=forensic-dgp-thesis --zone=us-central1-a "cctv-dgp-bank-path-ablation-v1-execution.tar.gz.sha256" "janusdominic0@forensic-dgp-thesis:/home/janusdominic0/"
```

2. Install in **VM SSH**:

```bash
cd ~ &&
tr -d '\r' < cctv-dgp-bank-path-ablation-v1-execution.tar.gz.sha256 | sha256sum -c - &&
test -d ~/forensic-dgp/cctv_dgp_vm_bundle/.venv &&
test -d ~/forensic-dgp/cctv_dgp_bank_photometric_guard_v1_vm &&
test ! -e ~/forensic-dgp/cctv_dgp_bank_path_ablation_v1_vm &&
tar -xzf cctv-dgp-bank-path-ablation-v1-execution.tar.gz -C ~/forensic-dgp
```

Expected archive SHA256: `17dee12790ed01c0326c2c390281917dbce0851ce3780a246241370e5ddc9298`.
Expected protocol SHA256: `c3cdd781858dbfc0e9e73a4165f63fc34806fc4a2b74284bdcd59da390b7331f`.
Installation creates one small new folder; it deletes nothing. A launch preflight
checks free space, idle GPU and exact retained-parent hashes before diagnostic logs.
Preserve any failure; a used diagnostic root cannot run again unchanged.

3. Open **tmux from the normal VM SSH shell**:

```bash
tmux new-session -A -s dgp_bank_path_ablation_v1
```

4. Launch **inside tmux**:

```bash
cd ~/forensic-dgp/cctv_dgp_bank_path_ablation_v1_vm &&
source ~/forensic-dgp/cctv_dgp_vm_bundle/.venv/bin/activate &&
python -B -u scripts/cctv_dgp_bank_path_ablation_v1_vm.py --root . --parent-root ../cctv_dgp_bank_photometric_guard_v1_vm --protocol-sha c3cdd781858dbfc0e9e73a4165f63fc34806fc4a2b74284bdcd59da390b7331f --verify-transfer &&
bash scripts/run_paths.sh c3cdd781858dbfc0e9e73a4165f63fc34806fc4a2b74284bdcd59da390b7331f
```

Expected transfer receipt: `complete: true`, `packet_assets: 2`,
`parent_assets: 5852`, `neural_calls: 0`, `optimizer_updates: 0`.
The run prints one line for each of seven variants. Detach safely with
**Ctrl+B, then D**. Reattach using `tmux attach -t dgp_bank_path_ablation_v1`.
An export receipt with `complete: true` verifies packaging only; quality and
independent return review remain separate. Preserve logs if any control fails.

5. Download from **Windows Google Cloud SDK Shell**, one remote source per call:

```bat
cd /d "C:\xampp\htdocs\YEAR 4\Testing\outputs"
gcloud compute scp --project=forensic-dgp-thesis --zone=us-central1-a "janusdominic0@forensic-dgp-thesis:/home/janusdominic0/cctv-dgp-bank-path-ablation-v1-results.tar.gz" "."
gcloud compute scp --project=forensic-dgp-thesis --zone=us-central1-a "janusdominic0@forensic-dgp-thesis:/home/janusdominic0/cctv-dgp-bank-path-ablation-v1-results.tar.gz.sha256" "."
gcloud compute scp --project=forensic-dgp-thesis --zone=us-central1-a "janusdominic0@forensic-dgp-thesis:/home/janusdominic0/cctv-dgp-bank-path-ablation-v1-export.json" "."
```

Keep the three files together. Returned outputs will be hash checked, independently
recomputed and visually reviewed before a new training recipe or app decision.
Native CCTV remains unpaired; paired photographic TRAIN measurements, reserved
final identities and all seven automatic/assisted completion families stay separate.
