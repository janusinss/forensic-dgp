# USED / FAILED — historical commands retained, do not rerun

The manual run and export finished. Independent return review confirms A PNG
gain 0.132699% with 8 failed checks; B 0.111965% with 18, both below1%.
All 20 sheets are reviewed. There is no new incumbent or epoch continuation.
[Audited result and exact evidence](<C:/xampp/htdocs/YEAR 4/Testing/CCTV_DGP_SPATIAL_READOUT_V1_RESULTS.md>).
[Next diagnostic direction; separate packet still unprepared](<C:/xampp/htdocs/YEAR 4/Testing/CCTV_DGP_POST_SPATIAL_READOUT_REVIEW_V1.md>).
The upload/launch commands below are preserved historical instructions for the
USED experiment. Do not use them to repeat this failed recipe. Return files
are already downloaded and independently imported. Original app unchanged.
Exact prior guide follows.

# Spatial reconstruction comparison V1 — manual L4 commands

**VERIFIED PREPARED / UNRUN.** This tests a reconstruction change; no improved
checkpoint or useful CCTV performance is established. All14 CPU boundary tests,
16 archive members, 203 required parent assets, actual hash-only CLI and host
Bash syntax pass. The first preparation metadata error and checker order/
sandbox errors are retained; the final sealed archive was unchanged by the
checker corrections.

Route A fits the corrected current decoder; B adds learned spatial bank
connections with frozen conditioning. Each has at most50 updates on20 exposed
TRAIN references /100 views. These are2.5 small-sample passes per route, not
additional epochs of the full DGP. The original checkpoint remains protected.

Require **8 GiB free after installation** and8 GiB available system RAM before
model loading. No current VM capacity or power state is claimed. Estimated
**20–40 minutes** including checks/export; this is not a new runtime measurement.
Hard supervised phase limits, including kill grace, total at most78.5 minutes
plus short hash/metadata checks: training51, replay6, saved audit10.5, export11.
Each fit itself is capped at10 minutes; cache4 and each snapshot5 minutes.
VRAM20 GiB / RSS12 GiB / outputs3 GiB / retained disk reserve1 GiB.

1. Upload from **Windows Google Cloud SDK Shell**:

```bat
cd /d "C:\xampp\htdocs\YEAR 4\Testing\outputs"
gcloud compute scp --project=forensic-dgp-thesis --zone=us-central1-a "cctv-dgp-spatial-readout-v1-execution.tar.gz" "janusdominic0@forensic-dgp-thesis:/home/janusdominic0/"
gcloud compute scp --project=forensic-dgp-thesis --zone=us-central1-a "cctv-dgp-spatial-readout-v1-execution.tar.gz.sha256" "janusdominic0@forensic-dgp-thesis:/home/janusdominic0/"
```

2. Install in **VM SSH**:

```bash
cd ~ &&
sha256sum -c cctv-dgp-spatial-readout-v1-execution.tar.gz.sha256 &&
test -d ~/forensic-dgp/cctv_dgp_vm_bundle/.venv &&
test -f ~/forensic-dgp/cctv_dgp_bank_photometric_guard_v1_vm/protocol.json &&
test ! -e ~/forensic-dgp/cctv_dgp_spatial_readout_v1_vm &&
tar -xzf cctv-dgp-spatial-readout-v1-execution.tar.gz -C ~/forensic-dgp
```

The SHA sidecar is verified LF UTF-8, with no CR or BOM. The203 reused parent
files must still exist and match their hashes. Hash-only verification below
stops before models if a file is missing; it never launches an older pilot.

3. Open **tmux**:

```bash
tmux new-session -A -s dgp_spatial_readout_v1
```

4. Launch **inside tmux**:

```bash
cd ~/forensic-dgp/cctv_dgp_spatial_readout_v1_vm &&
source ~/forensic-dgp/cctv_dgp_vm_bundle/.venv/bin/activate &&
python -B -u scripts/cctv_dgp_spatial_readout_v1_vm.py --root . --protocol-sha 3bfbaf4b4a519f2926f0503731cd9d22390a7fc0c09da20e2f022e6a1b93c97e --verify-transfer &&
bash scripts/run_spatial_readout.sh 3bfbaf4b4a519f2926f0503731cd9d22390a7fc0c09da20e2f022e6a1b93c97e
```

The supervisor displays live output and saves trainer/replay/audit/export logs.
At the endpoint it prints each route's actual RAW/PNG gain and failed checks.
Training exit2 is the retained quality stop; it is not a model-loading traceback.
The archive is still exported for independent review. Export `complete:true`
means packaging finished; `training_success_not_implied:true` remains binding.
Initial parity/nonzero derivatives and new CUDA timing remain unverified until
this manual execution. No failed recipe automatically resumes or gains epochs.

Detach with **Ctrl+B, then D**. Reattach with
`tmux attach -t dgp_spatial_readout_v1`. The process runs within that session.

5. Download after export from **Windows Google Cloud SDK Shell**:

```bat
cd /d "C:\xampp\htdocs\YEAR 4\Testing\outputs"
gcloud compute scp --project=forensic-dgp-thesis --zone=us-central1-a "janusdominic0@forensic-dgp-thesis:/home/janusdominic0/cctv-dgp-spatial-readout-v1-results.tar.gz" "."
gcloud compute scp --project=forensic-dgp-thesis --zone=us-central1-a "janusdominic0@forensic-dgp-thesis:/home/janusdominic0/cctv-dgp-spatial-readout-v1-results.tar.gz.sha256" "."
gcloud compute scp --project=forensic-dgp-thesis --zone=us-central1-a "janusdominic0@forensic-dgp-thesis:/home/janusdominic0/cctv-dgp-spatial-readout-v1-export.json" "."
```

Each PuTTY-compatible download has **one remote source**. Keep the returned
archive, sidecar and export receipt together for import, independent saved-array
audit and direct review of all20 sheets. No app adoption follows automatically.

Protocol SHA256: `3bfbaf4b4a519f2926f0503731cd9d22390a7fc0c09da20e2f022e6a1b93c97e`.
Execution archive SHA256: `83033e7207a9ce5bce5b7554d19a53202ace9375b8a74b5e9ba5bfe26903e3b5`.
[Design and unchanged requirements](<C:/xampp/htdocs/YEAR 4/Testing/CCTV_DGP_RECONSTRUCTION_DESIGN_REVIEW_V2.md>).
