# Coverage comparison VM runbook — 30 September 2026

Local bundle: `C:\xampp\htdocs\YEAR 4\Testing\outputs\coverage-vm-bundle.tar.gz`
(60,849,582 bytes), plus the adjacent `.sha256` file. SHA256:
`d72d3ed3de896856ac91be324829b9b2d2333ba6bb703b2cb40609a9b6058016`.
All 1,425 archive files verified. This is a standalone bundle; git pull is not
needed to obtain its code/data. It does not change the main checkout or app model.

## Recipe

Original completion-epoch2 detector; generator frozen. Both arms start from exact
same archived weights, AdamW LR1e-5/weight decay1e-4, BCE+Dice, hard-visible0.25,
synthetic Bernoulli KL1, clipping1. Ten epochs x21 updates, batch8; real dataset
68 versus73 is the intervention. Shared200 source paths, shared replay tensors,
and same synthetic order/positions. Prior consistency results are background,
not a substitute control for this experiment. No expectation of passing retention
is asserted; five examples make this a bounded coverage diagnostic.

Fourteen local tests passed (sampler equivalence, CPU refusal, lossless replay
conversion, real and original synthetic selection guards). No optimizer or GPU
preflight was run locally. CUDA runtime compatibility and memory are not yet
verified. Both preflight and training refuse an existing output directory.

## SSH commands after uploading both bundle files to home

```bash
cd ~
sha256sum -c coverage-vm-bundle.tar.gz.sha256 &&
test ! -e ~/forensic-dgp/coverage_vm_bundle &&
tar -xzf coverage-vm-bundle.tar.gz -C ~/forensic-dgp &&
tmux new-session -A -s dgp_training
```

Inside tmux:

```bash
cd ~/forensic-dgp/coverage_vm_bundle
if [ -f ../feature_vm_bundle/.venv/bin/activate ]; then
  source ../feature_vm_bundle/.venv/bin/activate
elif [ -f ../venv/bin/activate ]; then
  source ../venv/bin/activate
fi
python3 -u scripts/train_coverage_vm.py \
  --checkpoint outputs/downloaded_completion/outputs/completion_pilot/epoch_2.pth \
  --inventory outputs/coverage_protocol_v1/vm_inventory.json \
  --output outputs/coverage_preflight_vm --preflight &&
python3 -u scripts/train_coverage_vm.py \
  --checkpoint outputs/downloaded_completion/outputs/completion_pilot/epoch_2.pth \
  --inventory outputs/coverage_protocol_v1/vm_inventory.json \
  --output outputs/coverage_training_vm
```

Preflight checks the complete inventory, loads CUDA model, materializes exact
shared replay bytes, and runs one forward loss check with **zero optimizer updates**.
Training runs only if that exits successfully. If interrupted, inspect existing
logs/output before changing directories or restarting; there is no resume support.
The current bundle does not install dependencies; it reuses the existing VM runtime.

## Return results

After `outputs/coverage_training_vm/complete.json` exists:

```bash
cd ~/forensic-dgp/coverage_vm_bundle
test -f outputs/coverage_training_vm/complete.json &&
tar -czf coverage-results.tar.gz outputs/coverage_training_vm &&
realpath coverage-results.tar.gz
```

Download that printed path with SSH-in-browser Download File. Place it locally at
`C:\xampp\htdocs\YEAR 4\Testing\outputs\coverage-results.tar.gz`.
Return outputs contain exact initial state and replay bytes, every epoch checkpoint,
real/synthetic masks, baseline, aggregate and human/mannequin/glare metrics.
`best_detector.pth` exists only after unchanged real and synthetic gates pass.

Next after return: independent mask recount, matched-arm comparison, ten-row
visual grid and completion checks before any checkpoint promotion. No automatic
application/generator replacement. A threshold pass alone is not goal completion.

Local recount checker prepared: `scripts/evaluate_coverage_results.py`. It requires
both completed arms, the expected inventory/tensor archive hashes, all 8,500 masks,
matching mask-derived metrics and matching epoch selection decisions. It exports
a fixed first-ten-real-case preview. Two count/error-path tests pass; the checker
has not yet been run on actual VM results. Baseline inference and checkpoint-to-mask
reproduction remain separate required checks before promotion.
