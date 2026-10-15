# Used diagnostic — completed and rejected; do not rerun

The joint comparison completed and all four proposals failed the retained
requirements. Best raw/PNG gains0.865441%/0.886155% still have37 failures each.
[Audited results and new design decision](<C:/xampp/htdocs/YEAR 4/Testing/CCTV_DGP_JOINT_RAW_PNG_FORWARD_V1_RESULTS.md>).
[Fresh changed manual diagnostic](<C:/xampp/htdocs/YEAR 4/Testing/CCTV_DGP_LIVE_CONDITIONING_V1_VM.md>).
The original commands below are retained history, not an instruction to relaunch.

# Joint raw/PNG weight-change comparison — manual L4 commands

**Verified and UNRUN.** This tests the new64-loss direction on actual outputs.
Four disposable positive weight changes: .0001,.0005,.005,.02, each reset to the
same initial model. Zero optimizer updates, gradients, committed updates, epochs
or checkpoints. [Completed measurement and purpose](<C:/xampp/htdocs/YEAR 4/Testing/CCTV_DGP_PNG_GROUP_GRADIENTS_V1_R2_RESULTS.md>).

Require **4 GiB free after installation**. Estimate **2–5 minutes**. Worker/
measurement1200/900s; outer1230s. Export300s/outer330s plus30s kill grace per stage.
VRAM20GiB/RSS12GiB/output768MiB/reserve1GiB. Export reserves the full payload;
current VM storage/workload is checked at execution. Upload **2,879,650 bytes**.

1. Upload from **Windows Google Cloud SDK Shell**:
```bat
cd /d "C:\xampp\htdocs\YEAR 4\Testing\outputs"
gcloud compute scp --project=forensic-dgp-thesis --zone=us-central1-a "cctv-dgp-joint-raw-png-forward-v1-execution.tar.gz" "janusdominic0@forensic-dgp-thesis:/home/janusdominic0/"
gcloud compute scp --project=forensic-dgp-thesis --zone=us-central1-a "cctv-dgp-joint-raw-png-forward-v1-execution.tar.gz.sha256" "janusdominic0@forensic-dgp-thesis:/home/janusdominic0/"
```

2. Install in **VM SSH**:
```bash
cd ~ &&
sha256sum -c cctv-dgp-joint-raw-png-forward-v1-execution.tar.gz.sha256 &&
test -d ~/forensic-dgp/cctv_dgp_vm_bundle/.venv &&
test -d ~/forensic-dgp/cctv_dgp_bank_photometric_guard_v1_vm &&
test ! -e ~/forensic-dgp/cctv_dgp_joint_raw_png_forward_v1_vm &&
tar -xzf cctv-dgp-joint-raw-png-forward-v1-execution.tar.gz -C ~/forensic-dgp
```

3. Open **tmux**:
```bash
tmux new-session -A -s dgp_joint_raw_png_forward_v1
```

4. Launch **inside tmux**:
```bash
cd ~/forensic-dgp/cctv_dgp_joint_raw_png_forward_v1_vm &&
source ~/forensic-dgp/cctv_dgp_vm_bundle/.venv/bin/activate &&
python -B -u scripts/cctv_dgp_joint_raw_png_forward_v1_vm.py --root . --parent-root ../cctv_dgp_bank_photometric_guard_v1_vm --protocol-sha e5f004ce6381598d2372e51a171b796dc39d37c16687a019ffca495cc3e72b89 --verify-transfer &&
bash scripts/run_forward.sh e5f004ce6381598d2372e51a171b796dc39d37c16687a019ffca495cc3e72b89
```

Hash-only preflight expects4 packet assets,5,852 parent assets,100 fixed affines
and zero neural/gradient/update calls. Progress visits20 batches for each of four
proposals. Export runs after completion or a retained stop; its `complete: true`
is not a model-quality pass. Detach **Ctrl+B,D**; reattach with
`tmux attach -t dgp_joint_raw_png_forward_v1`.
Existing GPU workloads are preserved. Do not delete old outputs or bypass stops.

5. Download from **Windows Google Cloud SDK Shell** after export:
```bat
cd /d "C:\xampp\htdocs\YEAR 4\Testing\outputs"
gcloud compute scp --project=forensic-dgp-thesis --zone=us-central1-a "janusdominic0@forensic-dgp-thesis:/home/janusdominic0/cctv-dgp-joint-raw-png-forward-v1-results.tar.gz" "cctv-dgp-joint-raw-png-forward-v1-results.tar.gz"
gcloud compute scp --project=forensic-dgp-thesis --zone=us-central1-a "janusdominic0@forensic-dgp-thesis:/home/janusdominic0/cctv-dgp-joint-raw-png-forward-v1-results.tar.gz.sha256" "cctv-dgp-joint-raw-png-forward-v1-results.tar.gz.sha256"
gcloud compute scp --project=forensic-dgp-thesis --zone=us-central1-a "janusdominic0@forensic-dgp-thesis:/home/janusdominic0/cctv-dgp-joint-raw-png-forward-v1-export.json" "cctv-dgp-joint-raw-png-forward-v1-export.json"
```

Each Windows PuTTY download has one remote source. Return all three files for
independent parameter/image hashes,400 raw/PNG outputs,800 metric rows, each
source's1% structure requirement and all20 visual sheets. No automatic more
training or app adoption. The existing named L4 VM is required; changing VM needs
identity/migration verification. All research assets and failures stay retained.

Protocol SHA256: `e5f004ce6381598d2372e51a171b796dc39d37c16687a019ffca495cc3e72b89`.
Execution archive SHA256: `d672b53210c64e9eb99f62e0a7898dce9b9155f5a6ce52a704aeed27ddcbff33`.
LF checksum and future audit fixtures pass.
[Transfer evidence](<C:/xampp/htdocs/YEAR 4/Testing/outputs/cctv_dgp_joint_raw_png_forward_v1_transfer_audit.json>).
