# Historical FP32 comparison commands — returned 16 October 2026

**COMPLETED AND AUDITED AT 50 UPDATES; QUALITY STOP RETAINED. Do not relaunch.**
[Actual results](CCTV_CODEFORMER_FP32_COMPARISON_V1_RETURN_REVIEW.md) supersede
the PREPARED/UNRUN preparation below. All earlier bytes are retained as history.
The next [design review](CCTV_CODEFORMER_POST_FP32_DESIGN_REVIEW_V1.md) has no
prepared executable/transfer or next launch command. No unchanged resume.

---

# FP32 CodeFormer comparison V1 — five manual steps

**VERIFIED / PREPARED / UNRUN.** This trains a fresh copy of untouched
CodeFormer for at most **50 optimizer updates**, using V2's exact first
200 observations and unchanged preservation checks. This is a precision
comparison, not 50 epochs or a proven model improvement.

Upload: **96,576,884 bytes (96.58 MB)**.
The installer copies **1,333,086,155 bytes (1.24 GiB)**
of verified immutable dependencies from the existing V2 folder. It requires
that copy space **plus 16 GiB free before copying**, and **16 GiB free after
installation**. Runtime retains a **4 GiB reserve**. Original folders and
checkpoints remain protected; no automatic cleanup or package installation.

Planning estimate: **10–25 minutes** for training, comparisons and export;
FP32 optimization timing/memory remain unmeasured until this run. Hard limits:
15 minutes dependency copy; 5 minutes preflight; 50 minutes worker including
at most 15 minutes optimization; 10 minutes export, plus termination grace.
A resource or quality stop retains the evidence. No precision fallback,
automatic retry, resume or continuation beyond update50.

## 1. Upload — Windows Google Cloud SDK Shell

```bat
cd /d "C:\xampp\htdocs\YEAR 4\Testing\outputs"
gcloud compute scp --project=forensic-dgp-thesis --zone=us-central1-a "cctv-codeformer-fp32-comparison-v1-execution.tar.gz" "janusdominic0@forensic-dgp-thesis:/home/janusdominic0/"
gcloud compute scp --project=forensic-dgp-thesis --zone=us-central1-a "cctv-codeformer-fp32-comparison-v1-execution.tar.gz.sha256" "janusdominic0@forensic-dgp-thesis:/home/janusdominic0/"
```

## 2. Install — VM SSH

```bash
cd ~ &&
sha256sum -c cctv-codeformer-fp32-comparison-v1-execution.tar.gz.sha256 &&
test -x ~/forensic-dgp/cctv_dgp_vm_bundle/.venv/bin/python &&
test -f ~/forensic-dgp/cctv_codeformer_adaptation_v2_vm/protocol.json &&
test ! -e ~/forensic-dgp/cctv_codeformer_fp32_comparison_v1_vm &&
tar -xzf cctv-codeformer-fp32-comparison-v1-execution.tar.gz -C ~/forensic-dgp &&
cd ~/forensic-dgp/cctv_codeformer_fp32_comparison_v1_vm &&
timeout --signal=TERM --kill-after=30 900 python3 -B scripts/prepare_codeformer_fp32_comparison_v1.py --stage-dependencies --root . --protocol-sha 8acff86e31c3d135994fdac6ed781374e036150ce1ae2110db82ca0dc5a3aff1
```

Expected: checksum `OK`, followed by a dependency receipt with `complete: true`.
That receipt confirms installation only. If a dependency, space or existing-file
check stops, preserve the message and partial files; do not delete the folder
or rerun an old pilot. The installer has no neural or training calls.

## 3. Open tmux — VM SSH

```bash
tmux new-session -A -s codeformer_fp32_v1
```

## 4. Launch — inside tmux

```bash
cd ~/forensic-dgp/cctv_codeformer_fp32_comparison_v1_vm &&
source ~/forensic-dgp/cctv_dgp_vm_bundle/.venv/bin/activate &&
python -B -u scripts/supervise_codeformer_fp32_comparison_v1.py --root . --protocol-sha 8acff86e31c3d135994fdac6ed781374e036150ce1ae2110db82ca0dc5a3aff1
```

The supervisor immediately prints a child PID and log path. It verifies all
installed hashes, checks live GPU/workload availability, runs this one finite
training comparison and exports retained quality/resource-stop evidence too.
An idle tmux shell alone is allowed. No V1/V2 training is relaunched.

The training output goes to a log, so the tmux terminal can stay quiet during
the stage. Read progress from a **second VM SSH terminal**:

```bash
tail -F ~/forensic-dgp/cctv_codeformer_fp32_comparison_v1_vm/outputs/supervision/trainer.log
```

Leave the viewer with **Ctrl+C**. Detach tmux with **Ctrl+B**, release, then
**D**. Reattach with `tmux attach -t codeformer_fp32_v1`. Do not paste the launch
command into a second terminal while its child is still running.

At completion, the supervisor prints each stage's `exit_code`. Its
`outputs/supervision/finished.json` separately records `training_process_exit_code`
and `export_exit_code`. The file below reports archive status:

```bash
cat ~/cctv-codeformer-fp32-comparison-v1-export.json
```

`complete: true` confirms an exported archive. Separately check
`run_results_present` and `failure_present`; archive completion never means
the model passed quality review. Stop at 50 even if guards pass.

## 5. Download — Windows Google Cloud SDK Shell, after export

```bat
cd /d "C:\xampp\htdocs\YEAR 4\Testing\outputs"
gcloud compute scp --project=forensic-dgp-thesis --zone=us-central1-a "janusdominic0@forensic-dgp-thesis:/home/janusdominic0/cctv-codeformer-fp32-comparison-v1-results.tar.gz" "."
gcloud compute scp --project=forensic-dgp-thesis --zone=us-central1-a "janusdominic0@forensic-dgp-thesis:/home/janusdominic0/cctv-codeformer-fp32-comparison-v1-results.tar.gz.sha256" "."
gcloud compute scp --project=forensic-dgp-thesis --zone=us-central1-a "janusdominic0@forensic-dgp-thesis:/home/janusdominic0/cctv-codeformer-fp32-comparison-v1-export.json" "."
```

Each SCP command has one remote source for Windows PuTTY compatibility. All
three files land in the existing local `outputs` folder. Download stopped
results too. The result archive has a hard 3.6 GiB bound; actual size is unknown.
Independent review must check inputs, actual weight/Adam changes, all RAW/PNG
outputs, preserved frozen tensors, calls/timing/stops and visual comparisons
before deciding whether to continue with a separately designed longer run.

Execution archive SHA-256: `7889c9363e3941c66e59b79e3e0124496ec28c01d68cb77113d8faccebadecf3`.

Protocol SHA-256: `8acff86e31c3d135994fdac6ed781374e036150ce1ae2110db82ca0dc5a3aff1`.

[Preparation evidence](CCTV_CODEFORMER_FP32_COMPARISON_V1_PREPARATION.md).
[Frozen design](CCTV_CODEFORMER_FP32_COMPARISON_V1_DESIGN.md).
