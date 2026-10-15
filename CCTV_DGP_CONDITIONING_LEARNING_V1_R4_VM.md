# Used R4 pilot — failed actual preservation; do not rerun

Returned results are independently audited:640 gradient queries,three rejected
changes,zero accepted updates. All RAW/PNG gates and original checkpoints remain.
[Result and why it stopped](<C:/xampp/htdocs/YEAR 4/Testing/CCTV_DGP_CONDITIONING_LEARNING_V1_R4_RESULTS.md>).
[Next optimizer-design review; not a runnable packet](<C:/xampp/htdocs/YEAR 4/Testing/CCTV_DGP_POST_R4_GRADIENT_DESIGN_REVIEW_V1.md>).
The prior guide is retained below as exact historical command evidence.

# Conditioning learning V1 R4 — finite manual L4 pilot

**Verified and UNRUN. At most two accepted training updates.** R3 stopped at a
RAW SSIM loss-definition mismatch after420 gradient queries and zero updates.
R4 corrects the scientific forward value; its original approximate derivative,
model settings and quality gates remain. All600 saved RAW/PNG loss checks pass.
[Audited result and correction](<C:/xampp/htdocs/YEAR 4/Testing/CCTV_DGP_CONDITIONING_LEARNING_V1_R3_RESULTS.md>).

It learns only12 conditioning tensors on20 photographic TRAIN references/100
cases. Maximum three trials per update; all rejected trials are retained.
Each accepted change requires actual raw/PNG preservation,brightness<=0.2,
positive per-source detail and cumulative0.1% parameter distance. The endpoint
still needs1% structure gain globally and per-source on raw/PNG. Full epochs,
native DEV,final review and seven covering families remain later requirements.

Require **9GiB free after installation**. Estimate **8–15 minutes plus export**.
Launch checks current space. Learning1350s/internal worker1500s/outer1530s;
export420s/outer450s,each outer timeout has30s kill grace. Total outer allowance
34minutes. Gradient round600s,solver60s;VRAM20GiB/RSS12GiB/output3.5GiB/reserve1GiB.
Maximum180 candidate/360 recognizer forwards and1,280 gradient queries. Upload
size is347,790bytes. Export includes any failure and accepted research
components with true update counts. No automatic follow-on or app promotion.

1. Upload from **Windows Google Cloud SDK Shell**:
```bat
cd /d "C:\xampp\htdocs\YEAR 4\Testing\outputs"
gcloud compute scp --project=forensic-dgp-thesis --zone=us-central1-a "cctv-dgp-conditioning-learning-v1-r4-execution.tar.gz" "janusdominic0@forensic-dgp-thesis:/home/janusdominic0/"
gcloud compute scp --project=forensic-dgp-thesis --zone=us-central1-a "cctv-dgp-conditioning-learning-v1-r4-execution.tar.gz.sha256" "janusdominic0@forensic-dgp-thesis:/home/janusdominic0/"
```

2. Install in **VM SSH**:
```bash
cd ~ &&
sha256sum -c cctv-dgp-conditioning-learning-v1-r4-execution.tar.gz.sha256 &&
test -d ~/forensic-dgp/cctv_dgp_vm_bundle/.venv &&
test -d ~/forensic-dgp/cctv_dgp_bank_photometric_guard_v1_vm &&
test ! -e ~/forensic-dgp/cctv_dgp_conditioning_learning_v1_r4_vm &&
tar -xzf cctv-dgp-conditioning-learning-v1-r4-execution.tar.gz -C ~/forensic-dgp
```

3. Open **tmux** in VM SSH:
```bash
tmux new-session -A -s dgp_conditioning_learning_v1_r4
```
If an older tmux session is attached,detach **Ctrl+B,D** first.

4. Launch **inside tmux**:
```bash
cd ~/forensic-dgp/cctv_dgp_conditioning_learning_v1_r4_vm &&
source ~/forensic-dgp/cctv_dgp_vm_bundle/.venv/bin/activate &&
python -B -u scripts/cctv_dgp_conditioning_learning_v1_r4_vm.py --root . --parent-root ../cctv_dgp_bank_photometric_guard_v1_vm --protocol-sha 75699a3f94d390d8283ecefeb539a3169fbfc70c0546225d2f40d0353d18be85 --verify-transfer &&
bash scripts/run_learning.sh 75699a3f94d390d8283ecefeb539a3169fbfc70c0546225d2f40d0353d18be85
```
Detach **Ctrl+B,D**; reattach with `tmux attach -t dgp_conditioning_learning_v1_r4`.
Progress is retained in `review.log`; the wrapper prints it when the finite
worker ends. Export also runs after a retained stop. `complete: true` means
archive creation; `accepted_updates` records actual learning. Keep the used
directory and original checkpoints/splits/failure records.

5. Download from **Windows Google Cloud SDK Shell** after export:
```bat
cd /d "C:\xampp\htdocs\YEAR 4\Testing\outputs"
gcloud compute scp --project=forensic-dgp-thesis --zone=us-central1-a "janusdominic0@forensic-dgp-thesis:/home/janusdominic0/cctv-dgp-conditioning-learning-v1-r4-results.tar.gz" "cctv-dgp-conditioning-learning-v1-r4-results.tar.gz"
gcloud compute scp --project=forensic-dgp-thesis --zone=us-central1-a "janusdominic0@forensic-dgp-thesis:/home/janusdominic0/cctv-dgp-conditioning-learning-v1-r4-results.tar.gz.sha256" "cctv-dgp-conditioning-learning-v1-r4-results.tar.gz.sha256"
gcloud compute scp --project=forensic-dgp-thesis --zone=us-central1-a "janusdominic0@forensic-dgp-thesis:/home/janusdominic0/cctv-dgp-conditioning-learning-v1-r4-export.json" "cctv-dgp-conditioning-learning-v1-r4-export.json"
```
Use one remote source per PuTTY download. Return all3 files for independent
raw/PNG,state,derivative,preservation and every-sheet review. Do not rerun the
used R3 pilot. This R4 experiment is not a full epoch or a qualified app model.

Protocol SHA256: `75699a3f94d390d8283ecefeb539a3169fbfc70c0546225d2f40d0353d18be85`.
Execution archive SHA256: `62de722fb09ee4fdb4604595ae90c9799979bb3414e6f9ed6c580590690719ec`.
[Transfer/state/forward/gate audit](<C:/xampp/htdocs/YEAR 4/Testing/outputs/cctv_dgp_conditioning_learning_v1_r4_transfer_audit.json>).
