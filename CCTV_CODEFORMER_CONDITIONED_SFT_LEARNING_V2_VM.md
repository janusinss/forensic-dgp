# Conditioned SFT V2 (Multi-Scale Dilated Pyramid + Frequency-Selective Head): Manual NVIDIA L4 VM Guide

**PREPARED / UNRUN.** Two-Phase high-capacity facial sharpening run (`Phase 1`: `40` epochs / `1,680` Exposure-Matched Clean-SFT distillation steps; `Phase 2`: `224` end-to-end Frozen-`R2` Perceptual + `SSIM7` + Multi-Scale Laplacian + Exposure-Split Gradient updates across `56` identities × `6` CCTV roles). All `575` trained-`u64` backbone tensors (`515` `R2` + `24` `U96` + `36` `u64` heads) remain frozen; step `0` reproduces `u64` bitwise.

- **Stage Directory**: `~/forensic-dgp/cctv_codeformer_conditioned_sft_learning_v2_vm`
- **Execution Upload**: `outputs/cctv-codeformer-conditioned-sft-learning-v2-execution.tar.gz` (**124,223,016 bytes**, `849` PAX members)
- **Protocol SHA256**: `806a5154f7c74463f32baf369feb510dfd65cf2d13602a09312c3ab76d857214`
- **Execution Archive SHA256**: `061c48a2f223208427074cf372101b90540435cc9cb80704617aa5f81297507e`
- **Expected NVIDIA L4 Runtime**: `~11–14 minutes` (`7,200s` scientific cap)

---

## 1. Upload from Windows Google Cloud SDK Shell

```bat
cd /d "C:\xampp\htdocs\YEAR 4\Testing\outputs"
gcloud compute scp --project=forensic-dgp-thesis --zone=us-central1-a "cctv-codeformer-conditioned-sft-learning-v2-execution.tar.gz" "janusdominic0@forensic-dgp-thesis:/home/janusdominic0/"
gcloud compute scp --project=forensic-dgp-thesis --zone=us-central1-a "cctv-codeformer-conditioned-sft-learning-v2-execution.tar.gz.sha256" "janusdominic0@forensic-dgp-thesis:/home/janusdominic0/"
```

## 2. Verify and install weights in VM SSH

```bash
cd ~ &&
sha256sum -c cctv-codeformer-conditioned-sft-learning-v2-execution.tar.gz.sha256 &&
test -d ~/forensic-dgp/cctv_dgp_vm_bundle/.venv &&
test -d ~/forensic-dgp/cctv_codeformer_u96_postfusion_smoothanchors_lr_v1_vm/weights &&
rm -rf ~/forensic-dgp/cctv_codeformer_conditioned_sft_learning_v2_vm &&
tar -xzf cctv-codeformer-conditioned-sft-learning-v2-execution.tar.gz -C ~/forensic-dgp &&
cd ~/forensic-dgp/cctv_codeformer_conditioned_sft_learning_v2_vm &&
python3 -B research/codeformer_conditioned_sft_learning_v2/install_weights_v2.py --root . --source-root ~/forensic-dgp/cctv_codeformer_u96_postfusion_smoothanchors_lr_v1_vm --protocol-sha 806a5154f7c74463f32baf369feb510dfd65cf2d13602a09312c3ab76d857214
```

## 3. Open tmux in VM SSH

```bash
tmux new-session -A -s codeformer_conditioned_sft_v2
```

## 4. Launch once inside tmux

```bash
cd ~/forensic-dgp/cctv_codeformer_conditioned_sft_learning_v2_vm &&
source ~/forensic-dgp/cctv_dgp_vm_bundle/.venv/bin/activate &&
python -B -u scripts/run_conditioned_sft_learning_v2.py --root . --protocol-sha 806a5154f7c74463f32baf369feb510dfd65cf2d13602a09312c3ab76d857214 --verify-transfer &&
bash research/codeformer_conditioned_sft_learning_v2/run_conditioned_sft_v2.sh 806a5154f7c74463f32baf369feb510dfd65cf2d13602a09312c3ab76d857214
```

Detach with **Ctrl+B, then D**. Reattach with `tmux attach -t codeformer_conditioned_sft_v2`. To monitor live step-by-step progress (`journal.jsonl`) and worker/export logs from another VM SSH terminal:

```bash
tail -F ~/forensic-dgp/cctv_codeformer_conditioned_sft_learning_v2_vm/outputs/conditioned_sft_learning_v2/journal.jsonl ~/forensic-dgp/cctv_codeformer_conditioned_sft_learning_v2_vm/run.log ~/forensic-dgp/cctv_codeformer_conditioned_sft_learning_v2_vm/export.log
```

## 5. Download from Windows Google Cloud SDK Shell

Wait until `supervisor.json` / `export.log` prints `"complete": true` (exit code `0`). Use **one remote source per command**:

```bat
cd /d "C:\xampp\htdocs\YEAR 4\Testing\outputs"
gcloud compute scp --project=forensic-dgp-thesis --zone=us-central1-a "janusdominic0@forensic-dgp-thesis:/home/janusdominic0/cctv-codeformer-conditioned-sft-learning-v2-results.tar.gz" "."
gcloud compute scp --project=forensic-dgp-thesis --zone=us-central1-a "janusdominic0@forensic-dgp-thesis:/home/janusdominic0/cctv-codeformer-conditioned-sft-learning-v2-results.tar.gz.sha256" "."
gcloud compute scp --project=forensic-dgp-thesis --zone=us-central1-a "janusdominic0@forensic-dgp-thesis:/home/janusdominic0/cctv-codeformer-conditioned-sft-learning-v2-export.json" "."
```
