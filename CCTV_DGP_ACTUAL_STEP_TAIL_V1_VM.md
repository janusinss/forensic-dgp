# Finish the final state of the storage-stopped diagnostic

The downloaded archive is verified. The separate R1 scientific audit checks all
29 complete conditions,3,045 raw/PNG/mean-only outputs and145 frozen CPU replays.
The original3GiB output-limit failure remains retained. The last update45 cone
condition has66 arrays and130 PNGs but no complete metrics receipt.

This verified43,992-byte packet reviews update45 only: unchanged, recorded
actual step and the same frozen cone proposal. All828 existing control/partial
files must replay exactly. It completes a diagnostic; it does not train or
qualify a model. Original checkpoints, splits, source and failures stay intact.

Require **2GiB free** after installation. Estimated runtime **3–7minutes** plus
download. Stops: cache120s, review300s, worker600s/external630s,
export300s/external330s, allocated VRAM20GiB and return512MiB.
The source-bound return auditor requires315 output checks,828 exact overlap
checks and15 frozen CPU replays within900s. Raw/PNG gates stay unchanged.

Protocol SHA256:
`ec6981ce0e6961b765fb5dc1e82f191bc3511db2a1ca6b87e2849e83ea40eef9`

Execution archive SHA256:
`0436e9477b1ce277856d9c54a1655d2915091c204637ab78e1c1fffe85d68950`

1. Upload from **Windows Google Cloud SDK Shell**:

```bat
cd /d "C:\xampp\htdocs\YEAR 4\Testing\outputs"
gcloud compute scp --project=forensic-dgp-thesis --zone=us-central1-a "cctv-dgp-actual-step-tail-v1-execution.tar.gz" "janusdominic0@forensic-dgp-thesis:/home/janusdominic0/"
gcloud compute scp --project=forensic-dgp-thesis --zone=us-central1-a "cctv-dgp-actual-step-tail-v1-execution.tar.gz.sha256" "janusdominic0@forensic-dgp-thesis:/home/janusdominic0/"
```

2. Install in **VM SSH**:

```bash
cd ~ &&
sha256sum -c cctv-dgp-actual-step-tail-v1-execution.tar.gz.sha256 &&
test -x ~/forensic-dgp/cctv_dgp_vm_bundle/.venv/bin/python &&
test -d ~/forensic-dgp/cctv_dgp_actual_step_review_v1_vm &&
test ! -e ~/forensic-dgp/cctv_dgp_actual_step_tail_v1_vm &&
tar -xzf cctv-dgp-actual-step-tail-v1-execution.tar.gz -C ~/forensic-dgp
```

3. Open **tmux**:

```bash
tmux new-session -A -s dgp_actual_step_tail_v1
```

4. Launch **inside tmux**:

```bash
cd ~/forensic-dgp/cctv_dgp_actual_step_tail_v1_vm &&
source ~/forensic-dgp/cctv_dgp_vm_bundle/.venv/bin/activate &&
python -B -u scripts/cctv_dgp_actual_step_tail_v1_vm.py --root . --parent-root ../cctv_dgp_actual_step_review_v1_vm --protocol-sha ec6981ce0e6961b765fb5dc1e82f191bc3511db2a1ca6b87e2849e83ea40eef9 --verify-transfer &&
bash scripts/run_actual_step_tail.sh ec6981ce0e6961b765fb5dc1e82f191bc3511db2a1ca6b87e2849e83ea40eef9
```

Transfer verification prints zero neural/gradient/optimizer calls. The supervisor
prints the child PID and log path, then review/export exit codes. A separate SSH
terminal can read progress with
`tail -F ~/forensic-dgp/cctv_dgp_actual_step_tail_v1_vm/review.log`.
Detach tmux with **Ctrl+B**, then **D**. An error is retained; do not restart it.

5. Download after export completes from **Windows Google Cloud SDK Shell**:

```bat
cd /d "C:\xampp\htdocs\YEAR 4\Testing\outputs"
gcloud compute scp --project=forensic-dgp-thesis --zone=us-central1-a "janusdominic0@forensic-dgp-thesis:/home/janusdominic0/cctv-dgp-actual-step-tail-v1-results.tar.gz" "."
gcloud compute scp --project=forensic-dgp-thesis --zone=us-central1-a "janusdominic0@forensic-dgp-thesis:/home/janusdominic0/cctv-dgp-actual-step-tail-v1-results.tar.gz.sha256" "."
gcloud compute scp --project=forensic-dgp-thesis --zone=us-central1-a "janusdominic0@forensic-dgp-thesis:/home/janusdominic0/cctv-dgp-actual-step-tail-v1-export.json" "."
```

Use one remote source per command for Windows PuTTY. After download, report
`downloaded`; the local checker independently imports and audits the return.
Export `complete` describes the archive; `tail_completed` describes the finite
diagnostic. Useful restoration still requires unchanged capacity, development,
native visual preservation and independent final review. All seven completion
families and separate automatic/assisted quality remain outstanding.
