# This diagnostic has returned — 16 October 2026

**COMPLETED AND AUDITED. Do not relaunch these historical commands.**
The diagnostic performed zero optimizer updates and created no new model.
[Current results](CCTV_CODEFORMER_PATH_DIAGNOSTIC_V1_RETURN_REVIEW.md) and
[next design](CCTV_CODEFORMER_FP32_COMPARISON_V1_DESIGN.md) supersede the old
PREPARED/UNRUN status below. The next design has no runnable packet yet.
All previous preparation/guide bytes are retained unchanged as history.

---

# CodeFormer path diagnostic V1 — five manual steps

**VERIFIED / PREPARED / UNRUN.** This measures why training changes brightness
and facial detail. It performs **zero optimizer updates** and exports no new
model. The untouched original and stopped V2 checkpoint stay unchanged.

Upload size: **7,445,777 bytes (7.45 MB)**. Require **6 GiB free after install**;
the worker preserves a **4 GiB** reserve. Runtime is unmeasured. Planning estimate:
**5–15 minutes** for measurement; hard worker limit **15 minutes**, export limit
**5 minutes**, transfer preflight limit **5 minutes**, plus termination grace.

The existing V2 directory and its stopped checkpoint are required. The worker
checks every required asset hash before any neural measurement. If a required
parent file was removed, retain the message and restore that exact dependency;
do not launch old training to recreate it.

## 1. Upload — Windows Google Cloud SDK Shell

```bat
cd /d "C:\xampp\htdocs\YEAR 4\Testing\outputs"
gcloud compute scp --project=forensic-dgp-thesis --zone=us-central1-a "cctv-codeformer-path-diagnostic-v1-execution.tar.gz" "janusdominic0@forensic-dgp-thesis:/home/janusdominic0/"
gcloud compute scp --project=forensic-dgp-thesis --zone=us-central1-a "cctv-codeformer-path-diagnostic-v1-execution.tar.gz.sha256" "janusdominic0@forensic-dgp-thesis:/home/janusdominic0/"
```

## 2. Install — VM SSH

```bash
cd ~ &&
sha256sum -c cctv-codeformer-path-diagnostic-v1-execution.tar.gz.sha256 &&
test -x ~/forensic-dgp/cctv_dgp_vm_bundle/.venv/bin/python &&
test -f ~/forensic-dgp/cctv_codeformer_adaptation_v2_vm/outputs/adaptation_v2/stopped_checkpoint.pt &&
test ! -e ~/forensic-dgp/cctv_codeformer_path_diagnostic_v1_vm &&
tar -xzf cctv-codeformer-path-diagnostic-v1-execution.tar.gz -C ~/forensic-dgp
```

Expected: `cctv-codeformer-path-diagnostic-v1-execution.tar.gz: OK`.
The checksum file uses Linux LF endings. Preserve any existing diagnostic
directory or result archive; this package does not support automatic reruns.

## 3. Open tmux — VM SSH

```bash
tmux new-session -A -s codeformer_path_v1
```

## 4. Launch — inside tmux

```bash
cd ~/forensic-dgp/cctv_codeformer_path_diagnostic_v1_vm &&
source ~/forensic-dgp/cctv_dgp_vm_bundle/.venv/bin/activate &&
python -B -u scripts/supervise_codeformer_path_diagnostic_v1.py --root . --protocol-sha 2d3dcd1dbc3be371de5e10ed9c059e455ce9cc8376bac99f04a1a514acc0d49f
```

The supervisor reports the live child PID and log path for each stage. It checks
current GPU and research processes, verifies the packet and parent assets, runs
one finite measurement, and exports stopped evidence too. An idle tmux session
alone is allowed. It does not install packages or launch another training run.

Detach with **Ctrl+B**, release, then **D**. Reattach with
`tmux attach -t codeformer_path_v1`.

The log can stay quiet while a measurement is running. To see the most recently
reserved call from a second VM SSH terminal:

```bash
python3 -c 'from pathlib import Path; p=Path.home()/"forensic-dgp/cctv_codeformer_path_diagnostic_v1_vm/outputs/path_diagnostic_v1/call_trace"; rows=sorted(p.glob("*.json")); print(rows[-1].read_text() if rows else "No neural call reserved yet; preflight or loading may still be running")'
```

This is read-only. A reservation reports the current stage and accumulated
counts; it does not prove that the reserved call has finished. It does not start
another worker or load a neural model.
After export, `complete: true` means the archive was written. The separate
`measurement_complete` and `failure_present` fields describe execution. A
completed diagnostic is still not model acceptance or training success.

## 5. Download — Windows Google Cloud SDK Shell, after export

```bat
cd /d "C:\xampp\htdocs\YEAR 4\Testing\outputs"
gcloud compute scp --project=forensic-dgp-thesis --zone=us-central1-a "janusdominic0@forensic-dgp-thesis:/home/janusdominic0/cctv-codeformer-path-diagnostic-v1-results.tar.gz" "."
gcloud compute scp --project=forensic-dgp-thesis --zone=us-central1-a "janusdominic0@forensic-dgp-thesis:/home/janusdominic0/cctv-codeformer-path-diagnostic-v1-results.tar.gz.sha256" "."
gcloud compute scp --project=forensic-dgp-thesis --zone=us-central1-a "janusdominic0@forensic-dgp-thesis:/home/janusdominic0/cctv-codeformer-path-diagnostic-v1-export.json" "."
```

Each download has one remote source for Windows PuTTY compatibility. Returned
files land in `C:\xampp\htdocs\YEAR 4\Testing\outputs`. Preserve all three files
even if the measurement stops. The next training design depends on independently
auditing these results; do not append epochs to V1/V2 unchanged.

Execution archive SHA-256:
`06300dab348b0bbbca71a5e21e6d275349c0e795d8a9a4d0ddb552bf4816b2c9`.

Protocol SHA-256:
`2d3dcd1dbc3be371de5e10ed9c059e455ce9cc8376bac99f04a1a514acc0d49f`.

[Preparation and exact scope](CCTV_CODEFORMER_PATH_DIAGNOSTIC_V1_PREPARATION.md).
[Frozen prospective design](CCTV_CODEFORMER_PATH_DIAGNOSTIC_V1_DESIGN.md).
