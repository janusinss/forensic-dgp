# Conditioned SFT V2 (Multi-Scale Dilated Pyramid + Frequency-Selective Head): Manual NVIDIA L4 VM Guide

**PREPARED / UNRUN.** Two-Phase high-capacity facial sharpening run (`Phase 1`: `40` epochs / `1,120` zero-CPU-cache on-the-fly batched GPU Exposure-Matched Clean-SFT distillation steps; `Phase 2`: `224` end-to-end Frozen-`R2` Perceptual + `SSIM7` + Multi-Scale Laplacian + Exposure-Split Gradient updates across `56` identities × `6` CCTV roles). All `575` trained-`u64` backbone tensors (`515` `R2` + `24` `U96` + `36` `u64` heads) remain frozen; step `0` reproduces `u64` bitwise; peak host RSS stays `< 3.5 GiB` on `g2-standard-4`.

- **Stage Directory**: `~/forensic-dgp/cctv_codeformer_conditioned_sft_learning_v2_vm`
- **Execution Upload**: `outputs/cctv-codeformer-conditioned-sft-learning-v2-execution.tar.gz` (**124,224,129 bytes**, `849` PAX members)
- **Protocol SHA256**: `84e4976079dd0d29ccb7408254a500298160b08479ce739163ed7dc47d90e5a0`
- **Execution Archive SHA256**: `f373cf3b1eb217a409381ff133c47061f4bf06a4db0bb99b34bc78af3eb01d1d`
- **Expected NVIDIA L4 Runtime**: `~8–12 minutes` (`7,200s` scientific cap)

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
rm -rf ~/forensic-dgp/cctv_codeformer_conditioned_sft_learning_v2_vm ~/cctv-codeformer-conditioned-sft-learning-v2-results.tar.gz ~/cctv-codeformer-conditioned-sft-learning-v2-results.tar.gz.sha256 ~/cctv-codeformer-conditioned-sft-learning-v2-export.json &&
tar -xzf cctv-codeformer-conditioned-sft-learning-v2-execution.tar.gz -C ~/forensic-dgp &&
cd ~/forensic-dgp/cctv_codeformer_conditioned_sft_learning_v2_vm &&
python3 -B research/codeformer_conditioned_sft_learning_v2/install_weights_v2.py --root . --source-root ~/forensic-dgp/cctv_codeformer_u96_postfusion_smoothanchors_lr_v1_vm --protocol-sha 84e4976079dd0d29ccb7408254a500298160b08479ce739163ed7dc47d90e5a0
```

## 3. Open tmux in VM SSH

```bash
tmux new-session -A -s codeformer_conditioned_sft_v2
```

## 4. Launch once inside tmux

```bash
cd ~/forensic-dgp/cctv_codeformer_conditioned_sft_learning_v2_vm &&
source ~/forensic-dgp/cctv_dgp_vm_bundle/.venv/bin/activate &&
python -B -u scripts/run_conditioned_sft_learning_v2.py --root . --protocol-sha 84e4976079dd0d29ccb7408254a500298160b08479ce739163ed7dc47d90e5a0 --verify-transfer &&
bash research/codeformer_conditioned_sft_learning_v2/run_conditioned_sft_v2.sh 84e4976079dd0d29ccb7408254a500298160b08479ce739163ed7dc47d90e5a0
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
