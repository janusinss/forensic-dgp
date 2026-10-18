# Bounded-fusion measurement V1 — five manual steps

**VERIFIED / PREPARED / UNRUN. This measures; it does not train.**
The previous FP32 run completed 50 updates but still failed preservation.
This new experiment tests a way to retain helpful dark-face improvement while
limiting changes to normal faces before choosing another training recipe.

Upload: **43,604,878 bytes (43.60 MB)**. Keep the existing FP32 folder and its
original/stopped weights. Those files are hash-checked and are not uploaded
again. Require **8 GiB free after installation**, retaining a **4 GiB reserve**.
Planning estimate: **5–12 minutes**, excluding transfer; the new intervention's
actual runtime remains unmeasured. Hard limits: 5 minutes preflight, 15 minutes
worker plus 60 seconds termination grace, 5 minutes export plus 30 seconds grace.
Within the worker: 5 minutes initialization and 10 minutes neural measurement.

## 1. Upload — Windows Google Cloud SDK Shell

```bat
cd /d "C:\xampp\htdocs\YEAR 4\Testing\outputs"
gcloud compute scp --project=forensic-dgp-thesis --zone=us-central1-a "cctv-codeformer-bounded-fusion-feasibility-v1-execution.tar.gz" "janusdominic0@forensic-dgp-thesis:/home/janusdominic0/"
gcloud compute scp --project=forensic-dgp-thesis --zone=us-central1-a "cctv-codeformer-bounded-fusion-feasibility-v1-execution.tar.gz.sha256" "janusdominic0@forensic-dgp-thesis:/home/janusdominic0/"
```

## 2. Install — VM SSH

```bash
cd ~ &&
sha256sum -c cctv-codeformer-bounded-fusion-feasibility-v1-execution.tar.gz.sha256 &&
test -x ~/forensic-dgp/cctv_dgp_vm_bundle/.venv/bin/python &&
test -f ~/forensic-dgp/cctv_codeformer_fp32_comparison_v1_vm/protocol.json &&
test -f ~/forensic-dgp/cctv_codeformer_fp32_comparison_v1_vm/weights/codeformer.pth &&
test -f ~/forensic-dgp/cctv_codeformer_fp32_comparison_v1_vm/outputs/fp32_comparison_v1/stopped_checkpoint.pt &&
test ! -e ~/forensic-dgp/cctv_codeformer_bounded_fusion_feasibility_v1_vm &&
python3 -c 'import shutil; from pathlib import Path; assert shutil.disk_usage(Path.home()).free >= 8*1024**3+45093599, "Require 8 GiB plus unpacking space"' &&
tar -xzf cctv-codeformer-bounded-fusion-feasibility-v1-execution.tar.gz -C ~/forensic-dgp
```

Expected checksum: `OK`. The LF sidecar avoids the earlier CRLF filename error.
If a check fails, retain the exact message and existing files. Do not overwrite,
delete the existing experiment or relaunch a historical training script.

## 3. Open tmux — VM SSH

```bash
tmux new-session -A -s cf_bounded_fusion_v1
```

## 4. Launch — inside tmux

```bash
cd ~/forensic-dgp/cctv_codeformer_bounded_fusion_feasibility_v1_vm &&
source ~/forensic-dgp/cctv_dgp_vm_bundle/.venv/bin/activate &&
python -B -u scripts/cctv_codeformer_bounded_fusion_feasibility_v1_vm.py --root . --protocol-sha c3b0f5f0f3ff83b508625df0259ed0c9acbc2d071da2fdde066a2bda429a15eb --verify-transfer &&
bash scripts/run_bounded_fusion_feasibility_v1.sh c3b0f5f0f3ff83b508625df0259ed0c9acbc2d071da2fdde066a2bda429a15eb
```

The transfer check should print `complete: true`, `cases: 32` and
`optimizer_updates: 0`. The supervisor prints a child PID before each stage.
Its bounded logs are displayed when that stage closes; the terminal can be
quiet during measurement. Completion means up to **170 model forwards and
960 internal fusion-block evaluations, zero optimizer/gradient calls**.
This is not another 50-update training run.

Press **Ctrl+B**, then **D** to detach safely. Reattach with the command in
step 3. Wait for the export stage and normal shell prompt before downloading.
A completed export can contain a failed measurement; it never proves quality.
Preserve any failure and download its evidence. Do not repeat the run unchanged.

## 5. Download — Windows Google Cloud SDK Shell

Use one remote source per command; Windows PuTTY rejects several remote sources
in the same `gcloud compute scp` command.

```bat
cd /d "C:\xampp\htdocs\YEAR 4\Testing\outputs"
gcloud compute scp --project=forensic-dgp-thesis --zone=us-central1-a "janusdominic0@forensic-dgp-thesis:/home/janusdominic0/cctv-codeformer-bounded-fusion-feasibility-v1-results.tar.gz" "."
gcloud compute scp --project=forensic-dgp-thesis --zone=us-central1-a "janusdominic0@forensic-dgp-thesis:/home/janusdominic0/cctv-codeformer-bounded-fusion-feasibility-v1-results.tar.gz.sha256" "."
gcloud compute scp --project=forensic-dgp-thesis --zone=us-central1-a "janusdominic0@forensic-dgp-thesis:/home/janusdominic0/cctv-codeformer-bounded-fusion-feasibility-v1-export.json" "."
```

After all three downloads, the saved arrays, state/content checks, actual call
counts, masks/padding if applicable, RAW/PNG conversion and every planned visual
comparison require independent local review. This TRAIN-only measurement does
not qualify native CCTV, FINAL identities, restoration or the application.

[Preparation and exact limits](CCTV_CODEFORMER_BOUNDED_FUSION_FEASIBILITY_V1_PREPARATION.md).
Protocol SHA-256: `c3b0f5f0f3ff83b508625df0259ed0c9acbc2d071da2fdde066a2bda429a15eb`.
Archive SHA-256: `9e355da89976885ad4d9935b483a1df8f449c916fe9dd0d480973ffa58af77d6`.
