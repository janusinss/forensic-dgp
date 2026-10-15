# Executed historical B r4 recipe — do not rerun

The returned 50-update run was independently audited and rejected on quality.
This preserved guide records that executed recipe. Current evidence: [A/B review](<C:/xampp/htdocs/YEAR 4/Testing/CCTV_DGP_BANK_AB_COMPARISON_V1_RESULTS.md>). New unrun trial: [brightness-guard manual guide](<C:/xampp/htdocs/YEAR 4/Testing/CCTV_DGP_BANK_PHOTOMETRIC_GUARD_V1_VM.md>). Historical commands below do not authorize repeating the rejected run.

---

# Bank comparison B r4: manual VM commands

B r3 stopped before any model forward or optimizer update because its loader
used `hashlib.file_digest`, which the VM's Python 3.10 does not provide.
B r4 replaces only that hash helper with streaming SHA256. The same pinned
initializer, model computation, data, schedule and preservation gates remain.
Five real-initializer CPU checks and the independent transfer audit passed
with the unavailable API removed. Actual Python 3.10/CUDA loading, training
and image quality remain pending. B r2's original constructor differences
remain unknown; the loader still records and rejects unsafe differences.

Use the **3.55 MiB code transfer** below on the existing VM. The data-only
installer copies 5,618 hash-verified assets from the retained B r3 folder into
a new B r4 folder, overlays 11 assets, and verifies the complete new packet.
It preserves B r3 and its failure. Installation has a **300-second limit**.
The separate 812,764,956-byte full execution archive stays locally for recovery;
uploading it here consumes the space needed for this run.

The intended experiment is **50 updates**, **250 distinct TRAIN references**
(125 per source), **1,250 profile exposures**, **zero complete epochs**
(0.3201 epoch). Our conditioning/reconstruction uses a declared **frozen
external StyleGAN2 bank**. Its generator is not own-trained. Covered-region
completion is outside this experiment. Additional epochs require independently
reviewed improvement and preservation; export alone does not qualify a model.

Require **16 GiB free after installation**. B r4 runtime is unmeasured.
Finite limits: worker 90 minutes, external worker 90.5 minutes, fit 10 minutes,
preflight 5 minutes, snapshot 20 minutes, export 10 minutes (external 10.5),
VRAM 20 GiB, output 6 GiB and disk reserve 1 GiB. Read the current storage
snapshot in PROJECT_HANDOFF.md; the installer and launch check space again.

1. Upload from **Windows Google Cloud SDK Shell**:

```bat
cd /d "C:\xampp\htdocs\YEAR 4\Testing\outputs"
gcloud compute scp --project=forensic-dgp-thesis --zone=us-central1-a "cctv-dgp-bank-comparison-v1-r4-B-code-transfer.tar.gz" "janusdominic0@forensic-dgp-thesis:/home/janusdominic0/"
gcloud compute scp --project=forensic-dgp-thesis --zone=us-central1-a "cctv-dgp-bank-comparison-v1-r4-B-code-transfer.tar.gz.sha256" "janusdominic0@forensic-dgp-thesis:/home/janusdominic0/"
```

2. Install in **VM SSH**:

```bash
cd ~ &&
tr -d '\r' < cctv-dgp-bank-comparison-v1-r4-B-code-transfer.tar.gz.sha256 | sha256sum -c - &&
test -d ~/forensic-dgp/cctv_dgp_vm_bundle/.venv &&
test ! -e ~/cctv_dgp_bank_comparison_B4_transfer &&
test ! -e ~/forensic-dgp/cctv_dgp_bank_comparison_v1_r4_B_vm &&
tar -xzf cctv-dgp-bank-comparison-v1-r4-B-code-transfer.tar.gz -C ~ &&
python3 -B ~/cctv_dgp_bank_comparison_B4_transfer/install_packet.py --protocol-sha 8c906e8b1a3b9534872313414321a2c780560d9ea5ab50fec3e34232813cad85
```

Expected code-transfer SHA256:
`354f41abf71ec94631a8584cf57e77a7fce3815c37e890e3eaf1de234d8c9d95`.
The installer requires the intact B r3 source folder. It performs file-copy
and hashing only; it imports no model code and launches no training. A failed
or partial installation is retained for diagnosis; do not repeat it in place.

3. Open **tmux from the normal VM SSH shell**:

```bash
tmux new-session -A -s dgp_bank_comparison_B4
```

4. Launch **inside that tmux session**:

```bash
cd ~/forensic-dgp/cctv_dgp_bank_comparison_v1_r4_B_vm &&
source ~/forensic-dgp/cctv_dgp_vm_bundle/.venv/bin/activate &&
python -B -u scripts/cctv_dgp_bank_comparison_v1_vm.py --root . --protocol-sha 8c906e8b1a3b9534872313414321a2c780560d9ea5ab50fec3e34232813cad85 --verify-transfer &&
bash scripts/run_bank_comparison.sh 8c906e8b1a3b9534872313414321a2c780560d9ea5ab50fec3e34232813cad85
```

Transfer verification must print `"loader_hash_runtime_checked": true`.
It hashes the real initializer using the repaired helper before training.
Detach with **Ctrl+B**, then **D**; reattach with step 3. Do not repeat step 4
in a used root or rerun the historical B r2/r3 commands. Initialization receipts
retain tensor differences and exact loading results. Quality stops use trainer
exit code 2; other execution/resource/gradient stops also retain failure exports.

5. Download after export finishes, from **Windows Google Cloud SDK Shell**:

```bat
cd /d "C:\xampp\htdocs\YEAR 4\Testing\outputs"
gcloud compute scp --project=forensic-dgp-thesis --zone=us-central1-a "janusdominic0@forensic-dgp-thesis:/home/janusdominic0/cctv-dgp-bank-comparison-v1-r4-B-results.tar.gz" "."
gcloud compute scp --project=forensic-dgp-thesis --zone=us-central1-a "janusdominic0@forensic-dgp-thesis:/home/janusdominic0/cctv-dgp-bank-comparison-v1-r4-B-results.tar.gz.sha256" "."
gcloud compute scp --project=forensic-dgp-thesis --zone=us-central1-a "janusdominic0@forensic-dgp-thesis:/home/janusdominic0/cctv-dgp-bank-comparison-v1-r4-B-export.json" "."
```

Each download uses one remote source for Windows/PuTTY. Return all three files
for independent import and execution/state audits, separate raw/PNG checks,
native unpaired review and inspection of every planned visual sheet. A trained
matched A/B comparison requires actual B exposures. Final identities remain
outside tuning. App qualification, independent final review and all seven
automatic/assisted covering families remain part of the active goal.
