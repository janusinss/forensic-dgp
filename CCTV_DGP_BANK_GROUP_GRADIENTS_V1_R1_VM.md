# Superseded used measurement packet — do not relaunch

R1's manual measurement is complete and independently audited. Its original
V1 storage failure and subsequent successful R1 are retained. No optimizer
update or checkpoint improvement occurred. Use the changed forward-only guide.
[Audited R1 measurements](<C:/xampp/htdocs/YEAR 4/Testing/CCTV_DGP_BANK_GROUP_GRADIENTS_V1_R1_RESULTS.md>).
[Fresh prepared/unrun four-step comparison](<C:/xampp/htdocs/YEAR 4/Testing/CCTV_DGP_BANK_GROUP_DIRECTION_FORWARD_V1_VM.md>).
Original commands below are retained historical evidence.

# Storage-recovered source/profile diagnostic R1: five manual steps

**Prepared and independently transfer-checked; actual VM measurement unrun.**
V1 stopped before neural work because space was below 4 GiB. Authorized cleanup
is independently closed; final live free-space snapshot is
**8.42 GiB**. R1 uses a fresh directory and records any
pre-model storage stop. It retains all measurement, timing and quality limits.

[Audited V1 cause and cleanup proof](<C:/xampp/htdocs/YEAR 4/Testing/CCTV_DGP_BANK_GROUP_GRADIENTS_V1_RESULTS.md>).

Upload **20,636,122 bytes** (about19.7 MiB). Require **4 GiB free
after installation** and an idle existing L4. Estimate **3–8 minutes** for the
new measurement/export; actual R1 neural runtime is unmeasured. Hard caps:
worker/measurement/export 900/600/300 s, external 930/330 s with 30s kill grace,
VRAM 20GiB, RSS 12GiB, outputs1.25GiB and disk reserve 1GiB. Approximately948MiB
of uncompressed arrays are planned. No optimizer updates or epochs are planned.

1. Upload from **Windows Google Cloud SDK Shell**:

```bat
cd /d "C:\xampp\htdocs\YEAR 4\Testing\outputs"
gcloud compute scp --project=forensic-dgp-thesis --zone=us-central1-a "cctv-dgp-bank-group-gradients-v1-r1-execution.tar.gz" "janusdominic0@forensic-dgp-thesis:/home/janusdominic0/"
gcloud compute scp --project=forensic-dgp-thesis --zone=us-central1-a "cctv-dgp-bank-group-gradients-v1-r1-execution.tar.gz.sha256" "janusdominic0@forensic-dgp-thesis:/home/janusdominic0/"
```

2. Install in **VM SSH**:

```bash
cd ~ &&
sha256sum -c cctv-dgp-bank-group-gradients-v1-r1-execution.tar.gz.sha256 &&
test -d ~/forensic-dgp/cctv_dgp_vm_bundle/.venv &&
test ! -e ~/forensic-dgp/cctv_dgp_bank_group_gradients_v1_r1_vm &&
tar -xzf cctv-dgp-bank-group-gradients-v1-r1-execution.tar.gz -C ~/forensic-dgp
```

The checksum is verified LF-only. Archive SHA256:
`5c49d914a28b30d16a63a585fb861b383285ecd9d31b1559ba42c6af117b4087`. A missing or differing parent stops verification;
no historical pilot is launched or silently reacquired.

3. Open **tmux**:

```bash
tmux new-session -A -s dgp_group_gradients_v1_r1
```

4. Launch **inside tmux**:

```bash
cd ~/forensic-dgp/cctv_dgp_bank_group_gradients_v1_r1_vm &&
source ~/forensic-dgp/cctv_dgp_vm_bundle/.venv/bin/activate &&
python -B -u scripts/cctv_dgp_bank_group_gradients_v1_r1_vm.py --root . --parent-root ../cctv_dgp_bank_photometric_guard_v1_vm --protocol-sha 61c90e1d3ba6f9f4bd8ac1649612faabd25827a08b5f77bfcc06f12cb1ac5213 --verify-transfer &&
bash scripts/run_groups.sh 61c90e1d3ba6f9f4bd8ac1649612faabd25827a08b5f77bfcc06f12cb1ac5213
```

Transfer verification prints available and required free bytes and 32 groups /
320 planned queries, with zero actual neural/gradient calls in that check.
The worker checks current storage again, then measures20 batches. A successful
measurement/export prints `Diagnostic exit code:0` and `Export exit code:0`.
An export success alone does not prove measurement or training success. Detach
with **Ctrl+B**, release, then **D** to keep tmux running. Leave failed records
intact; do not relaunch a used directory unchanged.

5. Download from **Windows Google Cloud SDK Shell**, after export finishes:

```bat
cd /d "C:\xampp\htdocs\YEAR 4\Testing\outputs"
gcloud compute scp --project=forensic-dgp-thesis --zone=us-central1-a "janusdominic0@forensic-dgp-thesis:/home/janusdominic0/cctv-dgp-bank-group-gradients-v1-r1-results.tar.gz" "cctv-dgp-bank-group-gradients-v1-r1-results.tar.gz"
gcloud compute scp --project=forensic-dgp-thesis --zone=us-central1-a "janusdominic0@forensic-dgp-thesis:/home/janusdominic0/cctv-dgp-bank-group-gradients-v1-r1-results.tar.gz.sha256" "cctv-dgp-bank-group-gradients-v1-r1-results.tar.gz.sha256"
gcloud compute scp --project=forensic-dgp-thesis --zone=us-central1-a "janusdominic0@forensic-dgp-thesis:/home/janusdominic0/cctv-dgp-bank-group-gradients-v1-r1-export.json" "cctv-dgp-bank-group-gradients-v1-r1-export.json"
```

Each download uses one remote source and an explicit local filename. Return
import/audit must check all saved controls, group scopes, gradient arithmetic,
states and stops before a new direction is derived. No native/final inputs,
app promotion or more-epoch qualification follows this diagnostic alone. The
permanent baseline, splits, caches, failures and seven-family completion scope
remain preserved; the full goal stays active.
