**CLOSED — 10 October 2026: returned and independently audited; all12 calibration arms rejected. Do not rerun or resume these historical commands.**

The trainer/export completed, but no arm passes the unchanged restoration
requirements. All64 planned visual sheets are reviewed. Read
CCTV_DGP_MULTISCALE_CALIBRATION_V1_RESULTS.md and
CCTV_DGP_POST_MULTISCALE_TRAINING_REVIEW.md for the returned result and selected
next design. No next executable packet is ready in this guide. The prior ready
status, disk/workload values and transfer/launch commands below are historical.
Current checkpoint and original research/failure evidence remain preserved.

---

# Manual multiscale DGP calibration V1 — 10 October 2026

Prepared and independently transfer-checked; not run or qualified. The failed
Head4 capacity checkpoint remains rejected. This diagnostic starts fresh from
the current app checkpoint and compares two learning partitions, three rates
and two balanced TRAIN pools. There are twelve independent one-update arms,
600 fitting exposures in total and zero complete epochs. One arm is not a
continuation of another arm. The MobileNet/FPN features, stored normalization,
recognizer and anchors remain frozen.

The same 100 previously exposed paired TRAIN cases are evaluated in every arm.
All 24 input-reviewed native development crops are retained separately as
unpaired evidence, with no clean-reference PSNR/SSIM or hidden-identity claim.
Final identity pixels are excluded. All five milestones and seven completion
families remain required; this diagnostic cannot promote a model or app.

The existing L4 VM is maintenance-verified **RUNNING**, instance ID
4410777042005672095. After verified archive cleanup, **11.181 GiB** is
free; conservative projection after upload/install is **10.776 GiB**.
The GPU is idle and no tmux session is active at the final snapshot. Recheck
availability before launching. Require **7 GiB free after installation**. Estimated model work: **15–30 minutes**. Worker stop:
1800 seconds; external limit: 1830 seconds. Export stop: 600 seconds; external
limit: 630 seconds. The supervisor terminates only its own child process group
after a deadline, with a 30-second grace period. Output cap: 2.5 GiB; disk
reserve: 1 GiB; allocated VRAM cap: 20 GiB. All fifteen selected decoder tensors
must show finite nonzero improvement gradients before any optimizer. The
runtime storage projection must pass before fitting. Preserve any failed stop.

Protocol SHA256:
`c0239c2eebab3891cd4f42d34b93a5c7c2b07c8597bf93247df2cc16239dd491`

Execution archive SHA256:
`e40d35871f13ae3edcf69723022cd178f72f91c87f3c96d6b6fae1505f7ac13b`

Archive: 185,975,690 bytes (177.4 MiB). Checks cover all 252 archive members,
GZIP integrity, 251 bound assets, 20 canonical targets, 100 paired TRAIN inputs,
24 native development inputs, shell syntax and local training refusal. The
initializer matches all 100 current-DGP CPU outputs exactly; an independent
ten-case CPU replay verifies the result. Actual L4 gradients and outputs are
pending. The prepared incoming audit is source-bound before transfer.

Windows **Google Cloud SDK Shell**:

1. Check that the existing VM is running:
   ```bat
   gcloud compute instances describe forensic-dgp-thesis --project=forensic-dgp-thesis --zone=us-central1-a --format="value(status)"
   ```
   Expected: `RUNNING`. Maintenance left it running; no model job was launched.
2. Open the local outputs directory:
   ```bat
   cd /d "C:\xampp\htdocs\YEAR 4\Testing\outputs"
   ```
3. Upload the execution archive:
   ```bat
   gcloud compute scp --project=forensic-dgp-thesis --zone=us-central1-a --scp-flag=-batch --scp-flag=-hostkey --scp-flag=SHA256:pGOZAAcM7szdr2S1Ty+8UORXEPPAhxMDw7jDfzZ4I0E "cctv-dgp-multiscale-calibration-v1-execution.tar.gz" "janusdominic0@forensic-dgp-thesis:/home/janusdominic0/"
   ```
4. Upload the checksum file:
   ```bat
   gcloud compute scp --project=forensic-dgp-thesis --zone=us-central1-a --scp-flag=-batch --scp-flag=-hostkey --scp-flag=SHA256:pGOZAAcM7szdr2S1Ty+8UORXEPPAhxMDw7jDfzZ4I0E "cctv-dgp-multiscale-calibration-v1-execution.tar.gz.sha256" "janusdominic0@forensic-dgp-thesis:/home/janusdominic0/"
   ```

Open **VM SSH** for the existing instance:

1. Verify the uploaded archive:
   ```bash
   cd ~ && sha256sum -c cctv-dgp-multiscale-calibration-v1-execution.tar.gz.sha256
   ```
2. Install into the new directory:
   ```bash
   test -d ~/forensic-dgp/cctv_dgp_vm_bundle/.venv &&
   test ! -e ~/forensic-dgp/cctv_dgp_multiscale_calibration_vm_v1 &&
   tar -xzf ~/cctv-dgp-multiscale-calibration-v1-execution.tar.gz -C ~/forensic-dgp
   ```
3. Check free space before model execution:
   ```bash
   df -h ~/forensic-dgp
   ```
   Stop if available space is below 7 GiB. Cleanup requires a fresh inventory and
   hash-matched local backup; do not delete research caches or checkpoints.
4. Open tmux:
   ```bash
   tmux new-session -A -s dgp_multiscale_calibration_v1
   ```
5. Launch inside tmux:
   ```bash
   cd ~/forensic-dgp/cctv_dgp_multiscale_calibration_vm_v1 &&
   source ~/forensic-dgp/cctv_dgp_vm_bundle/.venv/bin/activate &&
   python -B -u scripts/cctv_dgp_multiscale_calibration_vm_v1.py --root . --protocol-sha c0239c2eebab3891cd4f42d34b93a5c7c2b07c8597bf93247df2cc16239dd491 --verify-transfer &&
   bash scripts/run_multiscale.sh c0239c2eebab3891cd4f42d34b93a5c7c2b07c8597bf93247df2cc16239dd491
   ```

Detach with **Ctrl+B, then D**. The supervised process continues inside tmux.
The initial gradient stage prints twenty batches/280 queries. Fitting prints
twelve arms, with one optimizer update per arm. A sampled quality failure is
retained for that completed arm; the remaining prospectively declared arms
start again from the original initializer. Runtime, state, storage, gradient
or finite-value failures stop the worker and retain partial evidence. The
supervisor attempts a bounded export after either success or failure. Do not
rerun a used directory or resume any failed state.

`complete:true` in the export JSON means the archive was written. Individual
quality gates still apply the unchanged 1% structure, appearance preservation
and 20% brightness-share requirements. A one-step diagnostic cannot satisfy
the full 3905-case capacity requirement, native usefulness, additional-epoch
study or app qualification. All 64 planned comparison pages must be inspected
after the return audit; automatic versus assisted completion remains separate.

After **Export exit code: 0**, download each remote file separately from
**Windows Google Cloud SDK Shell**. PuTTY supports one remote source per call:

1. Open the local outputs directory:
   ```bat
   cd /d "C:\xampp\htdocs\YEAR 4\Testing\outputs"
   ```
2. Download the results archive:
   ```bat
   gcloud compute scp --project=forensic-dgp-thesis --zone=us-central1-a --scp-flag=-batch --scp-flag=-hostkey --scp-flag=SHA256:pGOZAAcM7szdr2S1Ty+8UORXEPPAhxMDw7jDfzZ4I0E "janusdominic0@forensic-dgp-thesis:/home/janusdominic0/cctv-dgp-multiscale-calibration-v1-results.tar.gz" "."
   ```
3. Download its checksum file:
   ```bat
   gcloud compute scp --project=forensic-dgp-thesis --zone=us-central1-a --scp-flag=-batch --scp-flag=-hostkey --scp-flag=SHA256:pGOZAAcM7szdr2S1Ty+8UORXEPPAhxMDw7jDfzZ4I0E "janusdominic0@forensic-dgp-thesis:/home/janusdominic0/cctv-dgp-multiscale-calibration-v1-results.tar.gz.sha256" "."
   ```
4. Download the export receipt:
   ```bat
   gcloud compute scp --project=forensic-dgp-thesis --zone=us-central1-a --scp-flag=-batch --scp-flag=-hostkey --scp-flag=SHA256:pGOZAAcM7szdr2S1Ty+8UORXEPPAhxMDw7jDfzZ4I0E "janusdominic0@forensic-dgp-thesis:/home/janusdominic0/cctv-dgp-multiscale-calibration-v1-export.json" "."
   ```

Return all three files even if the trainer reports a failure. Preserve the VM
directory until full local hashes, weights, optimizer/scheduler/RNG state,
code/environment, inputs/provenance and failure evidence are independently
verified. Transfer check:
`outputs/cctv_dgp_multiscale_calibration_v1_transfer_audit_r1/independent_audit.json`.
Design rationale: `CCTV_DGP_POST_HEAD4_CAPACITY_DESIGN_REVIEW.md`.
