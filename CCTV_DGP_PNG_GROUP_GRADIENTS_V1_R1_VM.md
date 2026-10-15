# R1 completed with a retained conversion stop — do not rerun

The user returned R1. It stopped before gradients at the canonical CPU/CUDA PNG
conversion boundary. Zero updates or new structure evidence. Preserve this
packet; use the changed R2 only. [R1 findings](<C:/xampp/htdocs/YEAR 4/Testing/CCTV_DGP_PNG_GROUP_GRADIENTS_V1_R1_RESULTS.md>).
[Fresh R2 manual guide](<C:/xampp/htdocs/YEAR 4/Testing/CCTV_DGP_PNG_GROUP_GRADIENTS_V1_R2_VM.md>). Original commands follow as history.

# PNG-loss gradient measurement V1-R1 — manual L4 commands

**Verified locally; R1 has not been uploaded or launched by the agent.**
Use the fresh **V1-R1** files. The original V1 had an incorrect export prefix;
its archive and code stay preserved. Its human VM execution state is unknown.
[Retained V1 naming error](<C:/xampp/htdocs/YEAR 4/Testing/outputs/cctv_dgp_png_group_gradients_v1_export_name_stop/receipt.json>).

This is a bounded diagnostic: 100 paired photographic TRAIN cases, 32 delivered
PNG loss groups, 320 gradient queries, 40 candidate and 20 recognizer forwards.
**Zero optimizer updates, parameter proposals, epochs or new checkpoints.**
It tests exact PNG forward values with an explicitly approximate backward;
it does not establish a true PNG derivative or improved model quality.
Native CCTV, final identities and completion images stay outside this measurement.

Require **4 GiB free after installation**. Estimated measurement/export runtime:
**3–8 minutes**, subject to actual L4 timing. Measurement cap10min, worker15min,
outer15.5min; export5min/outer5.5min, with30s termination grace per stage.
The first two complete batches must project within10min. Limits: VRAM20GiB,
RSS12GiB, output1,280MiB, storage reserve1GiB. Export retains enough free space
for the full payload and reserve. Current VM free space/power is not assumed.
Upload size: **2,844,623 bytes**; planned saved arrays about0.94GiB.

1. Upload from **Windows Google Cloud SDK Shell**:
```bat
cd /d "C:\xampp\htdocs\YEAR 4\Testing\outputs"
gcloud compute scp --project=forensic-dgp-thesis --zone=us-central1-a "cctv-dgp-png-group-gradients-v1-r1-execution.tar.gz" "janusdominic0@forensic-dgp-thesis:/home/janusdominic0/"
gcloud compute scp --project=forensic-dgp-thesis --zone=us-central1-a "cctv-dgp-png-group-gradients-v1-r1-execution.tar.gz.sha256" "janusdominic0@forensic-dgp-thesis:/home/janusdominic0/"
```

2. Install in **VM SSH**:
```bash
cd ~ &&
sha256sum -c cctv-dgp-png-group-gradients-v1-r1-execution.tar.gz.sha256 &&
test -d ~/forensic-dgp/cctv_dgp_vm_bundle/.venv &&
test -d ~/forensic-dgp/cctv_dgp_bank_photometric_guard_v1_vm &&
test ! -e ~/forensic-dgp/cctv_dgp_png_group_gradients_v1_r1_vm &&
tar -xzf cctv-dgp-png-group-gradients-v1-r1-execution.tar.gz -C ~/forensic-dgp
```

3. Open **tmux**:
```bash
tmux new-session -A -s dgp_png_group_gradients_v1_r1
```

4. Launch **inside tmux**:
```bash
cd ~/forensic-dgp/cctv_dgp_png_group_gradients_v1_r1_vm &&
source ~/forensic-dgp/cctv_dgp_vm_bundle/.venv/bin/activate &&
python -B -u scripts/cctv_dgp_png_group_gradients_v1_r1_vm.py --root . --parent-root ../cctv_dgp_bank_photometric_guard_v1_vm --protocol-sha 1ca9a1b6a1830b0e0224eb4b7ee97092c1baf52a186de3e1d68e031de61eba14 --verify-transfer &&
bash scripts/run_png_groups.sh 1ca9a1b6a1830b0e0224eb4b7ee97092c1baf52a186de3e1d68e031de61eba14
```

The hash-only preflight reports `packet_assets: 6`, `parent_assets: 5852`,
`groups: 32`, `planned_gradient_queries: 320` and zero actual model/gradient calls.
The run reports `reference_batches` progressing to20 and `gradient_queries` to320.
Export runs after a success or retained stop. `complete: true` in the export
means the archive was created; it is not a successful model or training result.
Detach with **Ctrl+B,D**; reattach with
`tmux attach -t dgp_png_group_gradients_v1_r1`.
A competing GPU workload is preserved. Used packets refuse unchanged reruns;
do not remove an existing outputs directory or bypass any stop.

5. Download from **Windows Google Cloud SDK Shell** after export finishes:
```bat
cd /d "C:\xampp\htdocs\YEAR 4\Testing\outputs"
gcloud compute scp --project=forensic-dgp-thesis --zone=us-central1-a "janusdominic0@forensic-dgp-thesis:/home/janusdominic0/cctv-dgp-png-group-gradients-v1-r1-results.tar.gz" "cctv-dgp-png-group-gradients-v1-r1-results.tar.gz"
gcloud compute scp --project=forensic-dgp-thesis --zone=us-central1-a "janusdominic0@forensic-dgp-thesis:/home/janusdominic0/cctv-dgp-png-group-gradients-v1-r1-results.tar.gz.sha256" "cctv-dgp-png-group-gradients-v1-r1-results.tar.gz.sha256"
gcloud compute scp --project=forensic-dgp-thesis --zone=us-central1-a "janusdominic0@forensic-dgp-thesis:/home/janusdominic0/cctv-dgp-png-group-gradients-v1-r1-export.json" "cctv-dgp-png-group-gradients-v1-r1-export.json"
```

Each download uses one remote source for Windows PuTTY. Return all three files
for saved raw/PNG controls, scalar/embedding, gradient-group and timing audits.
The prospective checker is prepared; no returned R1 results exist yet.
The1% structure and all raw/PNG preservation/brightness requirements stay binding.
More epochs need a separate justified finite test; nothing auto-launches here.
Existing app checkpoint, splits, research caches, prior failures and full thesis
restoration/completion requirements are retained. These commands target the
existing named L4 VM; a different VM requires identity/migration verification.

Protocol SHA256: `1ca9a1b6a1830b0e0224eb4b7ee97092c1baf52a186de3e1d68e031de61eba14`.
Execution archive SHA256: `4356cbd7d37baf5d1e391e3acf7eb2a24cb1e8263413ffdfcf307768cac5874b`.
The checksum is verified LF text, avoiding the earlier CRLF filename error.
[Independent transfer check](<C:/xampp/htdocs/YEAR 4/Testing/outputs/cctv_dgp_png_group_gradients_v1_r1_transfer_audit.json>).
[Delivered-loss design review](<C:/xampp/htdocs/YEAR 4/Testing/CCTV_DGP_DELIVERED_LOSS_LEARNING_REVIEW_V1.md>).
