# Superseded prepared V1 commands — use corrected R1

The enforced query limit was640; this packet's hash-only summary printed1,280.
R1 corrects receipt metadata and namespace only. Prior human execution is unknown;
the original packet and exact guide are retained. The recipe and gates remain.
[Current verified R1 manual commands](<C:/xampp/htdocs/YEAR 4/Testing/CCTV_DGP_RAW_CONDITIONING_LEARNING_V1_R1_VM.md>).
The previous document follows as historical evidence.

# RAW conditioning learning V1 — verified manual L4 comparison

**Prepared and UNRUN. At most 2 accepted updates, 3 trials per update.**
The previous R4 run rejected all three changes. This separate test optimizes
32 RAW loss groups and checks every delivered PNG before accepting a change.
The starting model, 100 paired TRAIN cases, masks, 12 conditioning tensors and
all actual RAW/PNG preservation rules remain fixed. More epochs depend on useful
preserved learning and later development review.

Require **9 GiB free after installation**. Estimated **5–12 minutes plus export**,
based on R4's measured timings; the new CUDA runtime is not yet measured.
Maximum 640 gradient queries, 180 candidate and 320 recognizer forwards.
Learning cap 1,350s; worker 1,500s; outer 1,530s; export 420s/outer 450s;
each outer timeout has 30s kill grace. Total outer allowance is **34 minutes**.
Gradient round 600s, solver 60s, VRAM 20 GiB, RSS 12 GiB, output 3.5 GiB,
disk reserve 1 GiB. Runtime space checks retain failures. Upload size:
**378,041 bytes**. No automatic follow-on or app promotion.

1. Upload from **Windows Google Cloud SDK Shell**:
```bat
cd /d "C:\xampp\htdocs\YEAR 4\Testing\outputs"
gcloud compute scp --project=forensic-dgp-thesis --zone=us-central1-a "cctv-dgp-raw-conditioning-learning-v1-execution.tar.gz" "janusdominic0@forensic-dgp-thesis:/home/janusdominic0/"
gcloud compute scp --project=forensic-dgp-thesis --zone=us-central1-a "cctv-dgp-raw-conditioning-learning-v1-execution.tar.gz.sha256" "janusdominic0@forensic-dgp-thesis:/home/janusdominic0/"
```

2. Install in **VM SSH**:
```bash
cd ~ &&
sha256sum -c cctv-dgp-raw-conditioning-learning-v1-execution.tar.gz.sha256 &&
test -d ~/forensic-dgp/cctv_dgp_vm_bundle/.venv &&
test -d ~/forensic-dgp/cctv_dgp_bank_photometric_guard_v1_vm &&
test ! -e ~/forensic-dgp/cctv_dgp_raw_conditioning_learning_v1_vm &&
tar -xzf cctv-dgp-raw-conditioning-learning-v1-execution.tar.gz -C ~/forensic-dgp
```

3. Open **tmux** in VM SSH:
```bash
tmux new-session -A -s dgp_raw_conditioning_learning_v1
```
Detach an older attached session with **Ctrl+B,D** before opening this one.

4. Launch **inside tmux**:
```bash
cd ~/forensic-dgp/cctv_dgp_raw_conditioning_learning_v1_vm &&
source ~/forensic-dgp/cctv_dgp_vm_bundle/.venv/bin/activate &&
python -B -u scripts/cctv_dgp_raw_conditioning_learning_v1_vm.py --root . --parent-root ../cctv_dgp_bank_photometric_guard_v1_vm --protocol-sha b5ec5e10c68572ba2db610b03fe4af2929c2df00099d7069fe079c68fb4e59d5 --verify-transfer &&
bash scripts/run_learning.sh b5ec5e10c68572ba2db610b03fe4af2929c2df00099d7069fe079c68fb4e59d5
```
Detach **Ctrl+B,D**; reattach with `tmux attach -t dgp_raw_conditioning_learning_v1`.
Progress is saved in `review.log` and printed when the finite worker ends.
Export runs after a retained worker stop. `complete: true` describes archive
creation; `accepted_updates` describes actual learning. Keep the used directory,
original checkpoints/splits/caches and all failed trial outputs.

5. Download from **Windows Google Cloud SDK Shell** after export:
```bat
cd /d "C:\xampp\htdocs\YEAR 4\Testing\outputs"
gcloud compute scp --project=forensic-dgp-thesis --zone=us-central1-a "janusdominic0@forensic-dgp-thesis:/home/janusdominic0/cctv-dgp-raw-conditioning-learning-v1-results.tar.gz" "cctv-dgp-raw-conditioning-learning-v1-results.tar.gz"
gcloud compute scp --project=forensic-dgp-thesis --zone=us-central1-a "janusdominic0@forensic-dgp-thesis:/home/janusdominic0/cctv-dgp-raw-conditioning-learning-v1-results.tar.gz.sha256" "cctv-dgp-raw-conditioning-learning-v1-results.tar.gz.sha256"
gcloud compute scp --project=forensic-dgp-thesis --zone=us-central1-a "janusdominic0@forensic-dgp-thesis:/home/janusdominic0/cctv-dgp-raw-conditioning-learning-v1-export.json" "cctv-dgp-raw-conditioning-learning-v1-export.json"
```
Use one remote source per PuTTY download. Return all 3 files for independent
state/vector/RAW/PNG/preservation audit and direct review of every output sheet.
No qualified new DGP or longer-epoch continuation is claimed by preparation.

Protocol SHA256: `b5ec5e10c68572ba2db610b03fe4af2929c2df00099d7069fe079c68fb4e59d5`.
Execution archive SHA256: `06f26f3b1401382af17a61e540ca0986400754ca40fe450a5931ca8ce08787cf`.
[Transfer and scientific-gate audit](<C:/xampp/htdocs/YEAR 4/Testing/outputs/cctv_dgp_raw_conditioning_learning_v1_transfer_audit.json>).
[Why this comparison is justified](<C:/xampp/htdocs/YEAR 4/Testing/CCTV_DGP_RAW_CONDITIONING_LEARNING_V1_PREPARATION.md>).
