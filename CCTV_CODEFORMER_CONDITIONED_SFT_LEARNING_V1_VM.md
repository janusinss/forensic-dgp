# Conditioned SFT feature-learning comparison: manual VM guide

**PREPARED / UNRUN.** This is a new finite training comparison, not a qualified successor. U96 stays the experimental app incumbent. Original u64, U96, R2, checkpoints, splits and failed recipes remain retained.

- **Stage Directory**: `~/forensic-dgp/cctv_codeformer_conditioned_sft_learning_v1_vm`
- **Execution Upload**: `outputs/cctv-codeformer-conditioned-sft-learning-v1-execution.tar.gz` (**107,283,842 bytes**, `690` PAX members)
- **Protocol SHA256**: `f54c37e8b6b05473b1530395549b31f89189445f642ecfc49b1280ba8442d7f0`
- **Execution Archive SHA256**: `4e3d07432ea1951a60050bb5aba5dfbbf30f6ad7b2cdc5e54232f0fa62b7af64`
- **Finite Schedule**: `1,736` forwards (`480` backwards, `96` optimizer updates across Arm `A` and Arm `B`)

---

## 1. Upload from Windows Google Cloud SDK Shell

```bat
cd /d "C:\xampp\htdocs\YEAR 4\Testing\outputs"
gcloud compute scp --project=forensic-dgp-thesis --zone=us-central1-a "cctv-codeformer-conditioned-sft-learning-v1-execution.tar.gz" "janusdominic0@forensic-dgp-thesis:/home/janusdominic0/"
gcloud compute scp --project=forensic-dgp-thesis --zone=us-central1-a "cctv-codeformer-conditioned-sft-learning-v1-execution.tar.gz.sha256" "janusdominic0@forensic-dgp-thesis:/home/janusdominic0/"
```

## 2. Verify and install in VM SSH

```bash
cd ~ &&
sha256sum -c cctv-codeformer-conditioned-sft-learning-v1-execution.tar.gz.sha256 &&
test -d ~/forensic-dgp/cctv_dgp_vm_bundle/.venv &&
test -d ~/forensic-dgp/cctv_codeformer_u96_postfusion_smoothanchors_lr_v1_vm/weights &&
test ! -e ~/forensic-dgp/cctv_codeformer_conditioned_sft_learning_v1_vm &&
tar -xzf cctv-codeformer-conditioned-sft-learning-v1-execution.tar.gz -C ~/forensic-dgp &&
cd ~/forensic-dgp/cctv_codeformer_conditioned_sft_learning_v1_vm &&
python3 -B research/codeformer_conditioned_sft_learning_v1/install_weights.py --root . --source-root ~/forensic-dgp/cctv_codeformer_u96_postfusion_smoothanchors_lr_v1_vm --protocol-sha f54c37e8b6b05473b1530395549b31f89189445f642ecfc49b1280ba8442d7f0
```

## 3. Open tmux in VM SSH

```bash
tmux new-session -A -s codeformer_conditioned_sft_v1
```

## 4. Launch once inside tmux

```bash
cd ~/forensic-dgp/cctv_codeformer_conditioned_sft_learning_v1_vm &&
source ~/forensic-dgp/cctv_dgp_vm_bundle/.venv/bin/activate &&
python -B -u scripts/run_conditioned_sft_learning_v1.py --root . --protocol-sha f54c37e8b6b05473b1530395549b31f89189445f642ecfc49b1280ba8442d7f0 --verify-transfer &&
bash research/codeformer_conditioned_sft_learning_v1/run_conditioned_sft.sh f54c37e8b6b05473b1530395549b31f89189445f642ecfc49b1280ba8442d7f0
```

Detach with **Ctrl+B, then D**. Reattach with `tmux attach -t codeformer_conditioned_sft_v1`. To monitor live step-by-step progress (`journal.jsonl`) and final worker/export logs from another VM SSH terminal:

```bash
tail -F ~/forensic-dgp/cctv_codeformer_conditioned_sft_learning_v1_vm/outputs/conditioned_sft_learning_v1/journal.jsonl ~/forensic-dgp/cctv_codeformer_conditioned_sft_learning_v1_vm/run.log ~/forensic-dgp/cctv_codeformer_conditioned_sft_learning_v1_vm/export.log
```

## 5. Download from Windows Google Cloud SDK Shell

Wait for the supervisor's export result. Use **one remote source per command**; Windows PuTTY rejects multiple remote sources in one invocation.

```bat
cd /d "C:\xampp\htdocs\YEAR 4\Testing\outputs"
gcloud compute scp --project=forensic-dgp-thesis --zone=us-central1-a "janusdominic0@forensic-dgp-thesis:/home/janusdominic0/cctv-codeformer-conditioned-sft-learning-v1-results.tar.gz" "."
gcloud compute scp --project=forensic-dgp-thesis --zone=us-central1-a "janusdominic0@forensic-dgp-thesis:/home/janusdominic0/cctv-codeformer-conditioned-sft-learning-v1-results.tar.gz.sha256" "."
gcloud compute scp --project=forensic-dgp-thesis --zone=us-central1-a "janusdominic0@forensic-dgp-thesis:/home/janusdominic0/cctv-codeformer-conditioned-sft-learning-v1-export.json" "."
```
