# Superseded used measurement packet — do not relaunch

R1's manual measurement is complete and independently audited. Its original
V1 storage failure and subsequent successful R1 are retained. No optimizer
update or checkpoint improvement occurred. Use the changed forward-only guide.
[Audited R1 measurements](<C:/xampp/htdocs/YEAR 4/Testing/CCTV_DGP_BANK_GROUP_GRADIENTS_V1_R1_RESULTS.md>).
[Fresh prepared/unrun four-step comparison](<C:/xampp/htdocs/YEAR 4/Testing/CCTV_DGP_BANK_GROUP_DIRECTION_FORWARD_V1_VM.md>).
Original commands below are retained historical evidence.

# Superseded used V1 packet: pre-model storage stop retained

V1 stopped before neural measurement at the unchanged 4 GiB guard. Authorized
cleanup is independently closed; use the fresh prepared/unrun R1 manual guide.
[Audited V1 result and cleanup](<C:/xampp/htdocs/YEAR 4/Testing/CCTV_DGP_BANK_GROUP_GRADIENTS_V1_RESULTS.md>).
[Five R1 recovery steps](<C:/xampp/htdocs/YEAR 4/Testing/CCTV_DGP_BANK_GROUP_GRADIENTS_V1_R1_VM.md>).
Original commands below are historical and must not be relaunched unchanged.

# Source/profile gradient diagnostic: five manual steps

**New diagnostic, prepared and independently transfer-checked; actual VM run
unrun.** Measures the preservation groups missed by the two-reference direction.
All20 photographic TRAIN references and five synthetic profiles are retained.
No native/final inputs, optimizer updates, parameter displacements or epochs.
The previous R2 return remains rejected and immutable; do not rerun it unchanged.

Upload **20,635,583 bytes** (19.7 MiB). Require **4 GiB free after
installation** and an idle existing L4. Estimate **3–8 minutes** for measurement
and export; this new run has not been timed. Hard caps: worker/measurement/export
900/600/300 seconds, external930/330 seconds with30-second kill grace, VRAM20GiB,
host RSS12GiB, outputs1.25GiB and disk reserve1GiB. Approximate uncompressed
gradient/sample/control arrays are948MiB; the resulting return size is unmeasured.

[Audited R2 result and reason for this diagnostic](<C:/xampp/htdocs/YEAR 4/Testing/CCTV_DGP_BANK_SAVED_DIRECTION_FORWARD_V1_R2_RESULTS.md>).

1. Upload from **Windows Google Cloud SDK Shell**:

```bat
cd /d "C:\xampp\htdocs\YEAR 4\Testing\outputs"
gcloud compute scp --project=forensic-dgp-thesis --zone=us-central1-a "cctv-dgp-bank-group-gradients-v1-execution.tar.gz" "janusdominic0@forensic-dgp-thesis:/home/janusdominic0/"
gcloud compute scp --project=forensic-dgp-thesis --zone=us-central1-a "cctv-dgp-bank-group-gradients-v1-execution.tar.gz.sha256" "janusdominic0@forensic-dgp-thesis:/home/janusdominic0/"
```

2. Install in **VM SSH**:

```bash
cd ~ &&
sha256sum -c cctv-dgp-bank-group-gradients-v1-execution.tar.gz.sha256 &&
test -d ~/forensic-dgp/cctv_dgp_vm_bundle/.venv &&
test ! -e ~/forensic-dgp/cctv_dgp_bank_group_gradients_v1_vm &&
tar -xzf cctv-dgp-bank-group-gradients-v1-execution.tar.gz -C ~/forensic-dgp
```

The checksum file is verified LF-only. Archive SHA256:
`e6a2149c8a7e4f3de237db0bc62e6801076eaf38e594b9c3c39cb20caacae896`. A missing hash-bound parent stops the transfer
check; it does not silently reacquire data, overwrite a checkpoint or launch
historical training. A failed checksum stops extraction.

3. Open **tmux**:

```bash
tmux new-session -A -s dgp_group_gradients_v1
```

4. Launch **inside tmux**:

```bash
cd ~/forensic-dgp/cctv_dgp_bank_group_gradients_v1_vm &&
source ~/forensic-dgp/cctv_dgp_vm_bundle/.venv/bin/activate &&
python -B -u scripts/cctv_dgp_bank_group_gradients_v1_vm.py --root . --parent-root ../cctv_dgp_bank_photometric_guard_v1_vm --protocol-sha c71bb683cf0c6c7bbb130e32415f4206e5a5136e9dc3ac9f6f97191ab487a794 --verify-transfer &&
bash scripts/run_groups.sh c71bb683cf0c6c7bbb130e32415f4206e5a5136e9dc3ac9f6f97191ab487a794
```

Expect the transfer receipt to show32 groups and320 planned queries, with zero
actual neural/gradient calls in that transfer check. The manual worker then
measures20 batches. Completion prints `Diagnostic exit code:0` and
`Export exit code:0`; successful export alone does not prove measurement success
or better image quality. Detach with **Ctrl+B**, release, then **D**; this keeps
tmux running. Leave the archive and source records untouched for audit.

5. Download from **Windows Google Cloud SDK Shell**, after export finishes:

```bat
cd /d "C:\xampp\htdocs\YEAR 4\Testing\outputs"
gcloud compute scp --project=forensic-dgp-thesis --zone=us-central1-a "janusdominic0@forensic-dgp-thesis:/home/janusdominic0/cctv-dgp-bank-group-gradients-v1-results.tar.gz" "cctv-dgp-bank-group-gradients-v1-results.tar.gz"
gcloud compute scp --project=forensic-dgp-thesis --zone=us-central1-a "janusdominic0@forensic-dgp-thesis:/home/janusdominic0/cctv-dgp-bank-group-gradients-v1-results.tar.gz.sha256" "cctv-dgp-bank-group-gradients-v1-results.tar.gz.sha256"
gcloud compute scp --project=forensic-dgp-thesis --zone=us-central1-a "janusdominic0@forensic-dgp-thesis:/home/janusdominic0/cctv-dgp-bank-group-gradients-v1-export.json" "cctv-dgp-bank-group-gradients-v1-export.json"
```

Each download uses **one remote source and an explicit local filename**, avoiding
the Windows PuTTY multiple-remote-source limitation. Hash/path import, all100
raw controls, all32 metric/gradient groups,320 scope bindings, four saved sample
vectors, original states, resource stops and gradient arithmetic must be checked
locally before a new training direction is chosen. Actual derivative execution
is measured on the VM; the saved-data audit does not independently rerun every
neural derivative. No automatic next pilot, more-epoch qualification or app
promotion follows this diagnostic.
