# Completion fine-tuning V1 R1: manual app-parity repair

25 October 2026. PREPARED, NOT RUN. The original V1 stopped before training:
zero backward calls and zero optimizer updates. This new R1 fixes a demonstrated
app-input/layout mismatch and saves parity evidence before asserting. GPU parity
remains unverified until the manual run; the `1e-6` tolerance is unchanged.

The installer is 120,369 bytes. It reuses the exact existing data and original
CodeFormer inpainting checkpoint. It verifies V1's protocol, 329 asset files and four
failed-run files before cloning only the sealed packet into a fresh R1 directory.
It does not modify V1, the original checkpoint or the app. No full archive or
weight re-upload is required. Use the same L4 VM as the returned V1 run.

## 1. Upload from Windows Google Cloud SDK Shell

Run these on your Windows computer, where the prompt starts with C:, not inside
VM SSH. Each scp command has one source.

```bat
cd /d "C:\xampp\htdocs\YEAR 4\Testing\outputs"
gcloud compute scp --project=forensic-dgp-thesis --zone=us-central1-a "install_cctv_completion_codeformer_finetune_v1_r1.py" "janusdominic0@forensic-dgp-thesis:/home/janusdominic0/"
gcloud compute scp --project=forensic-dgp-thesis --zone=us-central1-a "install_cctv_completion_codeformer_finetune_v1_r1.py.sha256" "janusdominic0@forensic-dgp-thesis:/home/janusdominic0/"
```

## 2. Install in VM SSH

```bash
cd ~ &&
sha256sum -c install_cctv_completion_codeformer_finetune_v1_r1.py.sha256 &&
test -d ~/forensic-dgp/cctv_dgp_vm_bundle/.venv &&
python3 -B install_cctv_completion_codeformer_finetune_v1_r1.py --expected-sha ca8f17a4a904c9e521f73d12151ecc385794690fdec6fd79e82c5a36298267cd
```

Expected:complete:true, 329 files, training_launched:false and new protocol
8815ad168ab429bf4f30b8d719d3a6de6d1e2667fc38412c89c9c95e74e5ff3d.
Installation needs 4 GiB plus about 30 MiB free and has a 180-second cooperative cap.
It refuses changed/missing parent assets and any existing R1. If it refuses,
retain the message; do not delete experiments or bypass hashes to continue.

## 3. Open tmux

If already inside tmux, use the current session. Otherwise:

```bash
tmux new-session -A -s completion_cf_finetune_v1_r1
```

## 4. Verify and launch inside tmux

```bash
cd ~/forensic-dgp/cctv_completion_codeformer_finetune_v1_r1_vm &&
source ~/forensic-dgp/cctv_dgp_vm_bundle/.venv/bin/activate &&
python -B -u scripts/cctv_completion_codeformer_finetune_v1_r1_vm.py --root . --protocol-sha 8815ad168ab429bf4f30b8d719d3a6de6d1e2667fc38412c89c9c95e74e5ff3d --verify-transfer &&
bash scripts/run_completion_codeformer_finetune_v1_r1.sh 8815ad168ab429bf4f30b8d719d3a6de6d1e2667fc38412c89c9c95e74e5ff3d
```

Transfer verification prints complete:true, 329 files, 128 fit cases, 32 monitor cases
and zero neural/optimizer calls. The next command starts the actual finite pilot.
Before updates it must pass four fresh original-app comparisons. On any mismatch,
paired arrays/layouts/errors remain in outputs/fresh_app_parity and export.
No check is weakened to allow training. The terminal may be quiet during baseline
preparation; monitor from a second SSH terminal:

```bash
tail -F ~/forensic-dgp/cctv_completion_codeformer_finetune_v1_r1_vm/trainer.log
```

Maximum 32 paired cycles, 64 Adam updates, 256 backward calls and 600 typed forward slots. Child
hard limit: 30 minutes; cooperative stop: 28 minutes reserves terminal-save time.
Export has a 15-minute cooperative limit and 1536 MiB raw/compressed cap. Start with 4 GiB free; stop if free space falls below 2 GiB; occupied GPU/nonfinite/storage/memory/quality checks can
stop earlier. Detach with Ctrl+B then D; do not launch a second copy.

Same 32 FIT/eight MONITOR original TRAIN sources, 128/32 paired mask cases and 32
public development cases. No original VAL/FINAL. Same two Adam learning rates 1e-5,
losses, code/fusion routes and frozen generator/codebook. One terminal state
retains all 332 trainable tensors, both Adam states, ordered names/RNG/schedule/
counters/partialgradients. Intermediate 8/16/32 images are not selectable recovery
checkpoints. No automatic resume or adoption.

At the end:

```bash
cat ~/forensic-dgp/cctv_completion_codeformer_finetune_v1_r1_vm/supervisor.json
cat ~/cctv-completion-codeformer-finetune-v1-r1-export.json
```

Worker exit 0 means only the prospective numerical screen passed;2 means finite quality
stop/nonqualification; exit 1/other errors require log audit. Export complete:true means
packaging succeeded, not useful completion. The screen is at least 1% relative
macro delivered-PNG hole-L1 gain with nonworsening raw/PNG hole/seam per geometry;
it is not restoration structure gain or proof of exact hidden identity. User
review, full-family usefulness and independent evaluation remain required.

## 5. Download from Windows Google Cloud SDK Shell

Run on your Windows computer, not VM SSH:

```bat
cd /d "C:\xampp\htdocs\YEAR 4\Testing\outputs"
gcloud compute scp --project=forensic-dgp-thesis --zone=us-central1-a "janusdominic0@forensic-dgp-thesis:/home/janusdominic0/cctv-completion-codeformer-finetune-v1-r1-results.tar.gz" "."
gcloud compute scp --project=forensic-dgp-thesis --zone=us-central1-a "janusdominic0@forensic-dgp-thesis:/home/janusdominic0/cctv-completion-codeformer-finetune-v1-r1-results.tar.gz.sha256" "."
gcloud compute scp --project=forensic-dgp-thesis --zone=us-central1-a "janusdominic0@forensic-dgp-thesis:/home/janusdominic0/cctv-completion-codeformer-finetune-v1-r1-export.json" "."
```

Tell this chat when all three are downloaded. Return transport, terminal state,
optimizer updates, raw/PNG outputs and every planned visual case will be audited
before a separate opt-in app test. Keep all prior failures and original weights.

Preparation checks passed: independent source/contract review, 10 pure CPU tensor tests, installer static review and seven full fixture checks. The actual new worker transfer verifier passed in the fixture with Torch imports blocked. Python 3.10 syntax was checked; the fixture ran under local Python 3.13, so the VM must still perform its own verification. No GPU parity, training or model-quality pass is claimed.
