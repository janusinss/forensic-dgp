# Superseded B r3 commands - 11 October 2026

B r3 has run and stopped before training because its loader requires a hash
API unavailable in VM Python 3.10. Its return and failure remain preserved.
Use the [five manual upload, installation, tmux and download steps](<C:/xampp/htdocs/YEAR 4/Testing/CCTV_DGP_BANK_COMPARISON_B_V1_R4_VM.md>) for the repaired B r4 packet. Do not rerun the historical
commands below. Evidence: [B r3 returned evidence](<C:/xampp/htdocs/YEAR 4/Testing/CCTV_DGP_BANK_B_R3_RETURN_REVIEW.md>).

---

# Bank comparison B r3: manual VM commands

B r2 stopped with `Fresh initializer differs` before any B forward, gradient
or optimizer update. Its original packet and failure remain archived locally.
B r3 loads the same hash-pinned initializer strictly, records constructor
differences, and checks exact loaded state before the long baseline pass.
Differences outside the new style-conditioning tensors still stop the run.
The specific tensors that differed in B r2 were not saved and remain unknown.

The independently checked transfer contains 5,623 regular files. Model
computation, initializer bytes, data, objective, schedule, quality gates and
resource caps are unchanged. Five CPU loader checks passed; B r3 CUDA execution,
learning and image quality remain unverified. Read the latest storage snapshot
in PROJECT_HANDOFF.md before uploading. The launch checks resources again.

This finite comparison uses **50 updates**, **250 distinct TRAIN references**,
**1,250 profile exposures**, **zero complete epochs** (0.3201 epoch). B starts
from its original initializer, not A's rejected trained state. Our learned
conditioning/reconstruction uses a declared **frozen external StyleGAN2 bank**;
this does not make the bank an own-trained generator. Covered-region completion
is outside this pilot. No extra epochs or app promotion follow automatically.

Require **16 GiB free after installation**. B r2 spent 9.46 minutes producing
the baseline before its loading error, then 2.4 minutes exporting. B r3's full
runtime is unmeasured. Retained limits: worker 90 minutes, external worker
90.5 minutes, fitting 10 minutes, preflight 5 minutes, snapshot 20 minutes,
export 10 minutes (external 10.5), VRAM 20 GiB, output 6 GiB, disk reserve 1 GiB.

1. Upload from **Windows Google Cloud SDK Shell**:

```bat
cd /d "C:\xampp\htdocs\YEAR 4\Testing\outputs"
gcloud compute scp --project=forensic-dgp-thesis --zone=us-central1-a "cctv-dgp-bank-comparison-v1-r3-B-execution.tar.gz" "janusdominic0@forensic-dgp-thesis:/home/janusdominic0/"
gcloud compute scp --project=forensic-dgp-thesis --zone=us-central1-a "cctv-dgp-bank-comparison-v1-r3-B-execution.tar.gz.sha256" "janusdominic0@forensic-dgp-thesis:/home/janusdominic0/"
```

2. Install in **VM SSH**:

```bash
cd ~ &&
tr -d '\r' < cctv-dgp-bank-comparison-v1-r3-B-execution.tar.gz.sha256 | sha256sum -c - &&
test -d ~/forensic-dgp/cctv_dgp_vm_bundle/.venv &&
test ! -e ~/forensic-dgp/cctv_dgp_bank_comparison_v1_r3_B_vm &&
tar -xzf cctv-dgp-bank-comparison-v1-r3-B-execution.tar.gz -C ~/forensic-dgp
```

Expected archive SHA256:
`9ed188ce42881d33765544a355e687ef1ad1d34f044276e36fb6d6b929f0fc8b`.
The CR filter handles Windows checksum line endings. Keep the new root distinct
from the retained B r2 root.

3. Open **tmux from the normal VM SSH shell**:

```bash
tmux new-session -A -s dgp_bank_comparison_B3
```

4. Launch **inside that tmux session**:

```bash
cd ~/forensic-dgp/cctv_dgp_bank_comparison_v1_r3_B_vm &&
source ~/forensic-dgp/cctv_dgp_vm_bundle/.venv/bin/activate &&
python -B -u scripts/cctv_dgp_bank_comparison_v1_vm.py --root . --protocol-sha f34b7b54cd322d26a1a2159cffa28aabb1258fcb484914fd781d70e3e7bc0c35 --verify-transfer &&
bash scripts/run_bank_comparison.sh f34b7b54cd322d26a1a2159cffa28aabb1258fcb484914fd781d70e3e7bc0c35
```

Detach with **Ctrl+B**, then **D**; reattach with step 3. Do not repeat step 4
in a used or partial root. A quality stop is retained with trainer exit code 2.
An execution/resource/parity/gradient stop is also exported as failure evidence.
The new `initializer_load_B_*.json` receipts record exact loading checks and
constructor tensor differences even if initialization stops. Export completion
does not mean training or quality passed.

5. Download after export finishes, from **Windows Google Cloud SDK Shell**:

```bat
cd /d "C:\xampp\htdocs\YEAR 4\Testing\outputs"
gcloud compute scp --project=forensic-dgp-thesis --zone=us-central1-a "janusdominic0@forensic-dgp-thesis:/home/janusdominic0/cctv-dgp-bank-comparison-v1-r3-B-results.tar.gz" "."
gcloud compute scp --project=forensic-dgp-thesis --zone=us-central1-a "janusdominic0@forensic-dgp-thesis:/home/janusdominic0/cctv-dgp-bank-comparison-v1-r3-B-results.tar.gz.sha256" "."
gcloud compute scp --project=forensic-dgp-thesis --zone=us-central1-a "janusdominic0@forensic-dgp-thesis:/home/janusdominic0/cctv-dgp-bank-comparison-v1-r3-B-export.json" "."
```

Each SCP download uses one remote source for Windows/PuTTY. Return all three
files for independent import, saved raw/PNG and native-unpaired checks,
initializer and full-state/frozen-tensor checks where present, and review of
all planned sheets. A trained matched A/B comparison is possible only if B
actually completes the intended exposures. Reserved final identities remain
outside tuning; final review, app qualification and seven completion families
remain required by the active goal.
