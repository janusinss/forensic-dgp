# Bank comparison v1 split run: final manual route A (r3)

Run the new **A** root only. The original v1 stopped for storage before training;
keep its history. A and B now run separately with unchanged scientific checks.
The local transfer audits passed; new CUDA/training/quality checks remain pending.

Latest verified VM free space: **19.03 GiB**. Conservative space
after A installation: **17.42 GiB**.
Launch requires **16 GiB free after installation**. Baseline alone took 9.4 minutes
in the parent run; full revised route timing is unmeasured. Hard worker cap:
90 minutes plus at most 10.5 minutes supervised export. Fit cap 10 minutes,
snapshot cap 20 minutes, preflight cap five minutes, peak VRAM 20 GiB, disk
reserve 1 GiB, outputs 6 GiB. Each route is limited to 50 updates, not more epochs.

1. Upload from **Windows Google Cloud SDK Shell**:

```bat
cd /d "C:\xampp\htdocs\YEAR 4\Testing\outputs"
gcloud compute scp --project=forensic-dgp-thesis --zone=us-central1-a "cctv-dgp-bank-comparison-v1-r3-A-execution.tar.gz" "janusdominic0@forensic-dgp-thesis:/home/janusdominic0/"
gcloud compute scp --project=forensic-dgp-thesis --zone=us-central1-a "cctv-dgp-bank-comparison-v1-r3-A-execution.tar.gz.sha256" "janusdominic0@forensic-dgp-thesis:/home/janusdominic0/"
```

2. Install in **VM SSH**:

```bash
cd ~ &&
tr -d '\r' < cctv-dgp-bank-comparison-v1-r3-A-execution.tar.gz.sha256 | sha256sum -c - &&
test -d ~/forensic-dgp/cctv_dgp_vm_bundle/.venv &&
test ! -e ~/forensic-dgp/cctv_dgp_bank_comparison_v1_r3_A_vm &&
tar -xzf cctv-dgp-bank-comparison-v1-r3-A-execution.tar.gz -C ~/forensic-dgp
```

Expected archive SHA256: `ab71d6510b10838db0b40fedb27ec836e75595eb1571965aa808394638504e65`.
The CR filter handles Windows checksum line endings without altering the upload.

3. Open **tmux from the normal VM SSH shell**:

```bash
tmux new-session -A -s dgp_bank_comparison_A
```

4. Launch **inside that tmux session**:

```bash
cd ~/forensic-dgp/cctv_dgp_bank_comparison_v1_r3_A_vm &&
source ~/forensic-dgp/cctv_dgp_vm_bundle/.venv/bin/activate &&
python -B -u scripts/cctv_dgp_bank_comparison_v1_vm.py --root . --protocol-sha 6ca05c6b87a4588bfea2bc5b80307927c3a2fd47fa419567ee35280b37a78d0b --verify-transfer &&
bash scripts/run_bank_comparison.sh 6ca05c6b87a4588bfea2bc5b80307927c3a2fd47fa419567ee35280b37a78d0b
```

Detach without stopping the run with **Ctrl+B**, then **D**. Reopen the same
tmux session to see progress. Do not repeat step 4 in a used/partial run root.

5. Download after export finishes, from **Windows Google Cloud SDK Shell**:

```bat
cd /d "C:\xampp\htdocs\YEAR 4\Testing\outputs"
gcloud compute scp --project=forensic-dgp-thesis --zone=us-central1-a "janusdominic0@forensic-dgp-thesis:/home/janusdominic0/cctv-dgp-bank-comparison-v1-r3-A-results.tar.gz" "."
gcloud compute scp --project=forensic-dgp-thesis --zone=us-central1-a "janusdominic0@forensic-dgp-thesis:/home/janusdominic0/cctv-dgp-bank-comparison-v1-r3-A-results.tar.gz.sha256" "."
gcloud compute scp --project=forensic-dgp-thesis --zone=us-central1-a "janusdominic0@forensic-dgp-thesis:/home/janusdominic0/cctv-dgp-bank-comparison-v1-r3-A-export.json" "."
```

Each SCP command has one remote source, as required by Windows/PuTTY.
An export `complete: true` means that evidence was packaged. A trainer error,
gate stop or missing candidate remains a failure even if export succeeds.
Return all three files; independent raw/PNG and visual audit precedes B.

B is already frozen locally as `cctv-dgp-bank-comparison-v1-r2-B-execution.tar.gz`, protocol
`68197fd20bef1eae52c4c232164a027e095d567d4bfb37aea79c4f908853ea42`. It will start its own initializer after A is audited
and VM capacity is checked again. No automatic B launch or app promotion occurs.
Full goal and all seven completion families remain incomplete.
