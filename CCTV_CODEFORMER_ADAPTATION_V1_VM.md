# CodeFormer V1 return audited: quality-stopped at 200 updates — 15 October 2026

**CURRENT: 200/1200 optimizer updates; 800 TRAIN exposures; preservation failed.**
The first fine-tune has visible improvements and mixed profile metrics, but is
not an accepted replacement. One clear control exceeds the unchanged individual
RAW/PNG bound. Do not relaunch the V1 commands or resume its stopped state unchanged.

[Returned result and diagnosis](CCTV_CODEFORMER_ADAPTATION_V1_RETURN_REVIEW.md).
[Every comparison page](outputs/codeformer_adaptation_v1_return_review/visual_sheets/INDEX.md).
All 52 pages / 406 case pairs were reviewed; user acceptance and reserved-final
evaluation remain pending. Native CCTV remains unpaired and unqualified.

All 350 trainable tensors changed; 165 frozen restoration tensors stayed exact.
The archive, checkpoints, failed guard and splits remain preserved. The measured
next design addresses exposure/loss imbalance using TRAIN-only data, with the
same preservation requirements. No new training has launched, no candidate was
adopted, and further completion/detector development remains deferred.

**The launch-ready / PREPARED / UNRUN text below is retained historical evidence.**
This current status supersedes it. The full thesis goal remains active.

---

# CodeFormer adaptation V1 — manual VM commands

Status: **VERIFIED / PREPARED / UNRUN.** Independent full-copy and archive audits pass.
Use this new CodeFormer packet. The original pretrained CodeFormer and all DGP
checkpoints remain comparison/rollback assets. No old training pilot is resumed.

The trusted SSH host key is verified. Read-only inventory confirms **78 GiB
free**, an idle NVIDIA L4 and the required installed packages. No cleanup or
package upgrade is needed. Use steps **1–4 now**; step **5 follows export**.

Allow approximately 19 GiB free before upload/unpacking, with **16 GiB free after
installation**. The packet contains approximately 1.31 GB of files. Preserve
research caches and checkpoints if more space is needed. A provisioned 200 GB disk
does not mean 200 GB is free.

The finite experiment allows 1200 optimizer updates,4 examples per update and 4800
photographic exposures. It saves updates 0/200/600/1200. Optimization caps at 2 hours;
the worker caps at 3 hours, export at 10 minutes. These are stop limits, not a measured
runtime estimate. First 20 updates measure feasibility and remain in the same run.

## 1. Upload from Windows Google Cloud SDK Shell

```bat
cd /d "C:\xampp\htdocs\YEAR 4\Testing\outputs"
gcloud compute scp --project=forensic-dgp-thesis --zone=us-central1-a "cctv-codeformer-adaptation-v1-execution.tar.gz" "janusdominic0@forensic-dgp-thesis:/home/janusdominic0/"
gcloud compute scp --project=forensic-dgp-thesis --zone=us-central1-a "cctv-codeformer-adaptation-v1-execution.tar.gz.sha256" "janusdominic0@forensic-dgp-thesis:/home/janusdominic0/"
```

## 2. Install in VM SSH

```bash
cd ~ &&
sha256sum -c cctv-codeformer-adaptation-v1-execution.tar.gz.sha256 &&
test -x ~/forensic-dgp/cctv_dgp_vm_bundle/.venv/bin/python &&
test ! -e ~/forensic-dgp/cctv_codeformer_adaptation_v1_vm &&
tar -xzf cctv-codeformer-adaptation-v1-execution.tar.gz -C ~/forensic-dgp
```

## 3. Open tmux

```bash
tmux new-session -A -s codeformer_adaptation_v1
```

## 4. Launch inside tmux

```bash
cd ~/forensic-dgp/cctv_codeformer_adaptation_v1_vm &&
source ~/forensic-dgp/cctv_dgp_vm_bundle/.venv/bin/activate &&
python -B -c "import sys,torch,numpy,cv2,PIL; print({'python':sys.version,'torch':torch.__version__,'numpy':numpy.__version__,'opencv':cv2.__version__,'pillow':PIL.__version__})" &&
python -B -u -m research.codeformer_adaptation_v1.worker --root . --protocol-sha e463222441c0930a01a85ee0711a1ea31b0d80ed3487a4a5459c71e65f517d3e --verify-transfer &&
bash scripts/run_adaptation.sh e463222441c0930a01a85ee0711a1ea31b0d80ed3487a4a5459c71e65f517d3e
```

If the import check fails, the `&&` chain stops before training. Preserve the
printed error for a dependency-specific correction; do not upgrade Torch blindly.
The run also checks L4 availability, active GPU processes and free space.

Detach safely with `Ctrl+B`, release, then `D`. Reattach with:

```bash
tmux attach -t codeformer_adaptation_v1
```

`Ctrl+C` during training requests a preserved stop. A stopped recipe is not
automatically resumed. Keep all receipts and checkpoints, including a gate failure.

## 5. Download after export finishes

Run each transfer in **Windows Google Cloud SDK Shell**. Each command has exactly
one remote source, avoiding PuTTY's multiple-remote-source error.

```bat
cd /d "C:\xampp\htdocs\YEAR 4\Testing\outputs"
gcloud compute scp --project=forensic-dgp-thesis --zone=us-central1-a "janusdominic0@forensic-dgp-thesis:/home/janusdominic0/cctv-codeformer-adaptation-v1-results.tar.gz" "."
gcloud compute scp --project=forensic-dgp-thesis --zone=us-central1-a "janusdominic0@forensic-dgp-thesis:/home/janusdominic0/cctv-codeformer-adaptation-v1-results.tar.gz.sha256" "."
gcloud compute scp --project=forensic-dgp-thesis --zone=us-central1-a "janusdominic0@forensic-dgp-thesis:/home/janusdominic0/cctv-codeformer-adaptation-v1-export.json" "."
```

Download failure exports too. `complete:true` in the export receipt proves archive
packaging; inspect `run_results_present` and `failure_present` separately. Local
independent RAW/PNG/state/visual review determines whether a trained candidate
is useful. No result is promoted automatically.

[Full protocol and acceptance boundaries](CCTV_CODEFORMER_ADAPTATION_V1_PREPARATION.md).

Execution archive SHA256: `8724496a7089d49ceb658f2f373d5e119c5923d0f780478b0a9ceaf150102cd5`.
Protocol SHA256: `e463222441c0930a01a85ee0711a1ea31b0d80ed3487a4a5459c71e65f517d3e`. SHA sidecars use Linux LF line endings.
