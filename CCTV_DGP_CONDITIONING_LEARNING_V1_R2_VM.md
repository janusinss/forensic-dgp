# Returned startup failure — retain R2; use corrected R3

R2 stopped before model inference or training because its pixel fixture was16×16.
Zero updates; no structure/preservation decision or improved output was produced.
[Audited result](<C:/xampp/htdocs/YEAR 4/Testing/CCTV_DGP_CONDITIONING_LEARNING_V1_R2_RESULTS.md>). [Current manual R3 commands](<C:/xampp/htdocs/YEAR 4/Testing/CCTV_DGP_CONDITIONING_LEARNING_V1_R3_VM.md>).
The used R2 instructions below remain historical evidence; do not rerun them.

# Conditioning learning V1 R2 — finite manual L4 pilot

**Verified and UNRUN. This is actual training, capped at2 accepted updates.**
[Returned activation result and design rationale](<C:/xampp/htdocs/YEAR 4/Testing/CCTV_DGP_LIVE_CONDITIONING_V1_R1_RESULTS.md>).

It learns only12 conditioning tensors on the fixed20 photographic TRAIN
references/100 cases. The original decoder and frozen generative bank stay fixed.
Every update needs actual raw and delivered PNG preservation, brightness<=0.2
and positive detail progress in both source groups. At most3 trials per update;
all rejected trials are saved. The finite endpoint still needs1% structure gain
globally and per source, on raw and PNG. It stops early if that endpoint passes.
Full epochs, useful native CCTV, final identities and covering-family qualification
require subsequent independent evidence. No automatic follow-on or app promotion.

Require **9GiB free after installation**. Estimate **8–15 minutes plus export**.
The fresh maintenance snapshot found99.93GiB free; launch checks current space.
Learning1350s/internal worker1500s/outer1530s; export420s/outer450s, each outer
timeout has30s kill grace. Total outer allowance34minutes. Gradient round600s,
solver60s; VRAM20GiB/RSS12GiB/output3.5GiB/reserve1GiB. Maximum180 candidate and360
recognizer forwards and1,280 gradient queries. The upload is344,149bytes.

1. Upload from **Windows Google Cloud SDK Shell**:
```bat
cd /d "C:\xampp\htdocs\YEAR 4\Testing\outputs"
gcloud compute scp --project=forensic-dgp-thesis --zone=us-central1-a "cctv-dgp-conditioning-learning-v1-r2-execution.tar.gz" "janusdominic0@forensic-dgp-thesis:/home/janusdominic0/"
gcloud compute scp --project=forensic-dgp-thesis --zone=us-central1-a "cctv-dgp-conditioning-learning-v1-r2-execution.tar.gz.sha256" "janusdominic0@forensic-dgp-thesis:/home/janusdominic0/"
```

2. Install in **VM SSH**:
```bash
cd ~ &&
sha256sum -c cctv-dgp-conditioning-learning-v1-r2-execution.tar.gz.sha256 &&
test -d ~/forensic-dgp/cctv_dgp_vm_bundle/.venv &&
test -d ~/forensic-dgp/cctv_dgp_bank_photometric_guard_v1_vm &&
test ! -e ~/forensic-dgp/cctv_dgp_conditioning_learning_v1_r2_vm &&
tar -xzf cctv-dgp-conditioning-learning-v1-r2-execution.tar.gz -C ~/forensic-dgp
```

3. Open **tmux** from the VM SSH shell:
```bash
tmux new-session -A -s dgp_conditioning_learning_v1_r2
```
If an older tmux session is attached, detach with **Ctrl+B,D** first.

4. Launch **inside tmux**:
```bash
cd ~/forensic-dgp/cctv_dgp_conditioning_learning_v1_r2_vm &&
source ~/forensic-dgp/cctv_dgp_vm_bundle/.venv/bin/activate &&
python -B -u scripts/cctv_dgp_conditioning_learning_v1_r2_vm.py --root . --parent-root ../cctv_dgp_bank_photometric_guard_v1_vm --protocol-sha d29d1e6631282cfcb09b7785164b82b8aec6b7cde65cd387f1f432288012a0b6 --verify-transfer &&
bash scripts/run_learning.sh d29d1e6631282cfcb09b7785164b82b8aec6b7cde65cd387f1f432288012a0b6
```
Detach **Ctrl+B,D**; reattach with `tmux attach -t dgp_conditioning_learning_v1_r2`.
`review.log` retains live progress. The shell prints it after the finite worker
ends. Export also runs after a retained stop. Archive `complete: true` certifies
transfer only; `accepted_updates` reports true learning counts. A stopped run can
contain partial accepted research components. Return failures for audit and keep
the used directory. Original checkpoints, splits and gate records stay retained.

5. Download from **Windows Google Cloud SDK Shell** after export:
```bat
cd /d "C:\xampp\htdocs\YEAR 4\Testing\outputs"
gcloud compute scp --project=forensic-dgp-thesis --zone=us-central1-a "janusdominic0@forensic-dgp-thesis:/home/janusdominic0/cctv-dgp-conditioning-learning-v1-r2-results.tar.gz" "cctv-dgp-conditioning-learning-v1-r2-results.tar.gz"
gcloud compute scp --project=forensic-dgp-thesis --zone=us-central1-a "janusdominic0@forensic-dgp-thesis:/home/janusdominic0/cctv-dgp-conditioning-learning-v1-r2-results.tar.gz.sha256" "cctv-dgp-conditioning-learning-v1-r2-results.tar.gz.sha256"
gcloud compute scp --project=forensic-dgp-thesis --zone=us-central1-a "janusdominic0@forensic-dgp-thesis:/home/janusdominic0/cctv-dgp-conditioning-learning-v1-r2-export.json" "cctv-dgp-conditioning-learning-v1-r2-export.json"
```
PuTTY downloads use one remote source each. Return all3 files for independent
raw/PNG metrics, gradients, state, rollback and every visual-sheet review before
another training decision. Preserve the original unqualified app restorer.
Use this R2 guide; earlier local drafts were superseded during preparation.

Protocol SHA256: `d29d1e6631282cfcb09b7785164b82b8aec6b7cde65cd387f1f432288012a0b6`.
Execution archive SHA256: `ca91a89234ece2e6c162b165cbbedb1897a0efa75cd5b3dab996210af4562435`.
[Transfer/state and retained-gate audit](<C:/xampp/htdocs/YEAR 4/Testing/outputs/cctv_dgp_conditioning_learning_v1_r2_transfer_audit.json>).
