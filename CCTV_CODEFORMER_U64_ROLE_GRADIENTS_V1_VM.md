# Completed diagnostic: do not repeat the historical launch steps below

23 October 2026. **Executed and independently audited; zero updates.**
The user's returned archive and all 72 outputs/192 gradient files verify.
All 575 original-u64 model tensors remained unchanged. This diagnostic created
no new model. Its historical upload and tmux instructions below are retained
for provenance and must not be rerun or resumed.

Read the [return review](CCTV_CODEFORMER_U64_ROLE_GRADIENTS_V1_RETURN_REVIEW.md)
and [next source-only design](CCTV_CODEFORMER_U64_GATE_ALIGNED_V1_DESIGN.md).
The new design has no runnable worker, transfer package or launch guide yet.
U96 remains the experimental app incumbent; original trained u64 is retained.
Full goal ACTIVE/incomplete. Exact prior guide bytes follow.

---

# Diagnose trained u64: five manual VM steps

**Prepared and unrun.** This diagnostic measures why training can improve the
edge score while worsening photographic fidelity. It starts from the exact
original trained u64, makes **zero optimizer updates**, and leaves every model
value unchanged. It does not create a successor or repeat the failed update8 run.

Transfer: **238,801 bytes**. The installed parent supplies its existing weights
and data; this small archive contains the new diagnostic source and evidence.

Archive SHA256: `457c324448c6e6baefdcde7797722e6ad85a8d81ba467692bbe9d4a934003994`.
Protocol SHA256: `07d00785b4aef82c2e915a032fa2808064d0f728972f8c518694698a13f5d5e3`.

Use the existing user-identified L4 VM. Preparation did not connect to it, upload
files, start training or check its live storage. Require **8 GiB free** before
launch; a continuing 4 GiB reserve is enforced. Preserve the completed parent
`cctv_codeformer_u64_cleananchors_continuation_v1_vm`; the new stage reads its
469 exact static files, including its two installed weight files. It does not
resume that parent's stopped training or change its files.

## 1. Upload from Windows Google Cloud SDK Shell

```bat
cd /d "C:\xampp\htdocs\YEAR 4\Testing\outputs"
gcloud compute scp --project=forensic-dgp-thesis --zone=us-central1-a "cctv-codeformer-u64-role-gradients-v1-execution.tar.gz" "janusdominic0@forensic-dgp-thesis:/home/janusdominic0/"
gcloud compute scp --project=forensic-dgp-thesis --zone=us-central1-a "cctv-codeformer-u64-role-gradients-v1-execution.tar.gz.sha256" "janusdominic0@forensic-dgp-thesis:/home/janusdominic0/"
```

## 2. Check and install in VM SSH

```bash
cd ~ &&
sha256sum -c cctv-codeformer-u64-role-gradients-v1-execution.tar.gz.sha256 &&
test -d ~/forensic-dgp/cctv_dgp_vm_bundle/.venv &&
test -d ~/forensic-dgp/cctv_codeformer_u64_cleananchors_continuation_v1_vm &&
test ! -e ~/forensic-dgp/cctv_codeformer_u64_role_gradients_v1_vm &&
tar -xzf cctv-codeformer-u64-role-gradients-v1-execution.tar.gz -C ~/forensic-dgp &&
cd ~/forensic-dgp/cctv_codeformer_u64_role_gradients_v1_vm &&
python3 -B scripts/run_u64_role_gradients_v1.py --root . --protocol-sha 07d00785b4aef82c2e915a032fa2808064d0f728972f8c518694698a13f5d5e3 --verify-transfer
```

Expected: checksum `OK`, then `"complete": true`, `"own_static_bindings": 20`,
`"parent_static_bindings": 469`, and zero neural/training/VM calls in this
static verification. Missing or changed parent assets cause a stop. Preserve the
error and partial installation; do not delete files or change hashes to retry.
No separate weight installer is needed.

## 3. Open tmux

```bash
tmux new-session -A -s u64_role_gradients_v1
```

## 4. Launch inside tmux

```bash
cd ~/forensic-dgp/cctv_codeformer_u64_role_gradients_v1_vm &&
source ~/forensic-dgp/cctv_dgp_vm_bundle/.venv/bin/activate &&
python -B -u scripts/run_u64_role_gradients_v1.py --root . --protocol-sha 07d00785b4aef82c2e915a032fa2808064d0f728972f8c518694698a13f5d5e3 --verify-transfer &&
bash scripts/run_role_gradients.sh 07d00785b4aef82c2e915a032fa2808064d0f728972f8c518694698a13f5d5e3
```

The supervisor prints the child PID and log path. The worker deliberately uses
the read-only parent's working directory to retain its original initializer
guards. All new evidence is written under the new diagnostic stage.

In another VM SSH window, monitor:

```bash
tail -F ~/forensic-dgp/cctv_codeformer_u64_role_gradients_v1_vm/outputs/u64_role_gradients_v1/events.jsonl
```

Detach with **Ctrl+B**, then **D**. Reattach using step3. Do not launch again while
the diagnostic is running. The first state capture can be quiet while it writes
the complete model evidence. The journal reports forward/query reservations and
returns with their counters; `run.log` retains tracebacks.

Estimated worker time **3–8 minutes**, unmeasured for this new route. Internal
worker cap900 seconds; external960 seconds plus at most20 seconds termination
grace. Each phase has a180-second cap. Export is separate:180 seconds internally,
210 externally plus the same bounded cleanup. Limits:10 GiB RAM,18 GiB reserved
CUDA memory,2 GiB new output,2 GiB archive,4 GiB continuing disk reserve.

Maximum **72 full forwards and192 gradient queries**: four historical u64
RAW/PNG witnesses, four same-candidate repeats, and64 matched optimization cases
(16 photographs ×4 correlated degradations). Each optimization case measures the
role objective/4, photographic MAE and adjacent error separately. Monitor/guard
images receive no gradient queries. No optimizer is constructed and no parameter
direction is applied. All575 model tensors must remain bitwise unchanged.

The final supervisor prints `worker_exit` and `export_exit`. Both0 mean the
finite diagnostic and export completed; they do not mean the model improved.
A failed worker can still export retained evidence. Unknown writer cleanup
skips export. Missing or partial files stay preserved; do not rerun, resume or
overwrite the export to force success. Keep any traceback for the return audit.

## 5. Download from Windows Google Cloud SDK Shell

Wait for the supervisor's export result. Windows PuTTY requires one remote
source per command:

```bat
cd /d "C:\xampp\htdocs\YEAR 4\Testing\outputs"
gcloud compute scp --project=forensic-dgp-thesis --zone=us-central1-a "janusdominic0@forensic-dgp-thesis:/home/janusdominic0/cctv-codeformer-u64-role-gradients-v1-results.tar.gz" "."
gcloud compute scp --project=forensic-dgp-thesis --zone=us-central1-a "janusdominic0@forensic-dgp-thesis:/home/janusdominic0/cctv-codeformer-u64-role-gradients-v1-results.tar.gz.sha256" "."
gcloud compute scp --project=forensic-dgp-thesis --zone=us-central1-a "janusdominic0@forensic-dgp-thesis:/home/janusdominic0/cctv-codeformer-u64-role-gradients-v1-export.json" "."
```

If export failed or was skipped, retain its logs and supervisor receipts; the
three download files may not exist. Do not repeat the diagnostic to create them.

After download, independently audit the operation journal, state equality,
RAW/PNG parity, actual saved gradients, objectives, clipping and every generated
visual sheet. Gradient geometry is local RAW evidence; it cannot certify finite
training, delivered-PNG preservation or useful restoration. No optimizer/LR,
projection, routing rule or next training recipe is automatically chosen.

The original u64 stays preserved, U96 remains the experimental app incumbent,
and the update8 quality stop and inherited broader DEV failures remain binding.
Native/wild stay unpaired; reserved FINAL stays unopened and execution-unfrozen,
with independent reviewers pending. Reviewed Protect/Apply/bypass stays separate;
automatic matching and later completion development remain deferred.
