# Executed diagnostic failure — preserved historical commands

Do not rerun this V1 packet unchanged. The first raw replay failed before any
gradients. [Use the corrected, independently verified R1 guide](<C:/xampp/htdocs/YEAR 4/Testing/CCTV_DGP_BANK_LOSS_ALIGNMENT_V1_R1_VM.md>).
[Audited failure and layout reproduction](<C:/xampp/htdocs/YEAR 4/Testing/CCTV_DGP_BANK_LOSS_ALIGNMENT_V1_RESULTS.md>).

---

# Loss-direction diagnostic: exact manual VM commands

**Prepared and independently transfer-verified; actual L4 gradients are unrun.**
This is diagnosis before new training: **zero optimizer updates and no new model**.
It compares fixed initial and rejected B50 states; it does not resume the failed run.
[Audited results and rationale](<C:/xampp/htdocs/YEAR 4/Testing/CCTV_DGP_BANK_PATH_ABLATION_V1_RESULTS.md>).

Upload only **255,429 bytes**. Preserve
`~/forensic-dgp/cctv_dgp_bank_photometric_guard_v1_vm` and its 5,852 bound assets.
Require **4 GiB free after installation** and an idle L4. Estimate **3–10 minutes
measurement plus 1–5 minutes export**, unmeasured. Worker 15 minutes, measurement
stop 10 checked between reference batches, export 5; external 15.5/ 5.5 with 30-second
kill grace. VRAM 20 GiB, output/reserve 1 GiB each. Raw sampled-gradient storage
bound 530 MiB. Launch checks exact source/data/state hashes before model work.

1. Upload from **Windows Google Cloud SDK Shell**:

```bat
cd /d "C:\xampp\htdocs\YEAR 4\Testing\outputs"
gcloud compute scp --project=forensic-dgp-thesis --zone=us-central1-a "cctv-dgp-bank-loss-alignment-v1-execution.tar.gz" "janusdominic0@forensic-dgp-thesis:/home/janusdominic0/"
gcloud compute scp --project=forensic-dgp-thesis --zone=us-central1-a "cctv-dgp-bank-loss-alignment-v1-execution.tar.gz.sha256" "janusdominic0@forensic-dgp-thesis:/home/janusdominic0/"
```

2. Install in **VM SSH**:

```bash
cd ~ &&
tr -d '\r' < cctv-dgp-bank-loss-alignment-v1-execution.tar.gz.sha256 | sha256sum -c - &&
test -d ~/forensic-dgp/cctv_dgp_vm_bundle/.venv &&
test -d ~/forensic-dgp/cctv_dgp_bank_photometric_guard_v1_vm &&
test ! -e ~/forensic-dgp/cctv_dgp_bank_loss_alignment_v1_vm &&
tar -xzf cctv-dgp-bank-loss-alignment-v1-execution.tar.gz -C ~/forensic-dgp
```

Expected archive SHA256: `5cd8070253c0aa4b35defb8b9bbfb5311410c42ea11ba329ad42e2d371a26843`.
Expected protocol SHA256: `67838e1a9add105baf01a34fb14a83562d687540de61fc6338384143651241e1`.
Installation deletes nothing. Keep any stopped run and failure evidence.

3. Open **tmux from the normal VM SSH shell**:

```bash
tmux new-session -A -s dgp_bank_loss_alignment_v1
```

4. Launch **inside tmux**:

```bash
cd ~/forensic-dgp/cctv_dgp_bank_loss_alignment_v1_vm &&
source ~/forensic-dgp/cctv_dgp_vm_bundle/.venv/bin/activate &&
python -B -u scripts/cctv_dgp_bank_loss_alignment_v1_vm.py --root . --parent-root ../cctv_dgp_bank_photometric_guard_v1_vm --protocol-sha 67838e1a9add105baf01a34fb14a83562d687540de61fc6338384143651241e1 --verify-transfer &&
bash scripts/run_alignment.sh 67838e1a9add105baf01a34fb14a83562d687540de61fc6338384143651241e1
```

Expected transfer receipt: `complete: true`, `packet_assets: 3`,
`parent_assets: 5852`, `neural_calls: 0`, `gradient_queries: 0`,
`optimizer_updates: 0`. The diagnostic prints one line per reference/state.
Successful measurement finishes with 360 gradient queries and **zero optimizer
updates**. Detach safely with **Ctrl+B, then D**; reattach with
`tmux attach -t dgp_bank_loss_alignment_v1`. Export `complete` confirms packaging
only. The shell attempts to export a diagnostic failure too; no automatic training.

5. Download from **Windows Google Cloud SDK Shell**, one remote source per call:

```bat
cd /d "C:\xampp\htdocs\YEAR 4\Testing\outputs"
gcloud compute scp --project=forensic-dgp-thesis --zone=us-central1-a "janusdominic0@forensic-dgp-thesis:/home/janusdominic0/cctv-dgp-bank-loss-alignment-v1-results.tar.gz" "."
gcloud compute scp --project=forensic-dgp-thesis --zone=us-central1-a "janusdominic0@forensic-dgp-thesis:/home/janusdominic0/cctv-dgp-bank-loss-alignment-v1-results.tar.gz.sha256" "."
gcloud compute scp --project=forensic-dgp-thesis --zone=us-central1-a "janusdominic0@forensic-dgp-thesis:/home/janusdominic0/cctv-dgp-bank-loss-alignment-v1-export.json" "."
```

Keep the three files together. They will be independently hash checked and the
retained gradient samples recomputed before selecting a new finite training recipe.
Model selection, extra epochs, final identities, native unpaired evidence and all
seven automatic/assisted completion families remain separate and unqualified here.
