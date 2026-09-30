# Matched projection experiment — VM commands

Upload `C:\xampp\htdocs\YEAR 4\Testing\outputs\projection-code.tar.gz` and its
adjacent `.sha256` to VM home. Two new files use the existing completed coverage
workspace/cache. No original training code or checkpoints are overwritten.

```bash
cd ~
sha256sum -c projection-code.tar.gz.sha256 &&
tar -xzf projection-code.tar.gz -C ~/forensic-dgp/coverage_vm_bundle &&
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
python3 -u scripts/train_projection_vm.py --preflight --output outputs/projection_preflight_vm &&
python3 -u scripts/train_projection_vm.py
```

Preflight performs a CUDA forward check, no optimization. Full run executes
ordinary and projected arms,210 updates each, same extended73 real data, original
parent and archived replay tensors. Loss weights, LR, AdamW, clipping and all
selection safeguards remain fixed. `steps.jsonl` records420 total actual-step
alignments, projection decisions and decomposition errors. Generator invariance
is asserted; per-epoch validation masks and checkpoints are saved.

23 local helper/guard/decomposition/sampler/selection tests passed. CUDA execution
is still unverified. Output directories cannot already exist; if a command fails,
inspect its error rather than deleting/restarting automatically. There is no
resume feature. Nothing is installed and no application weights change.

The script prints the finished archive path:
`~/forensic-dgp/coverage_vm_bundle/projection-results.tar.gz`.
Download it into `C:\xampp\htdocs\YEAR 4\Testing\outputs\projection-results.tar.gz`.
No separate tar command is needed after successful completion.

Next after return: independently recount masks and selection, compare ordinary
arm with prior context, verify actual-step alignment and inspect previews before
considering any completion trial. Raw projection does not guarantee retention
through AdamW; no improvement is claimed before evaluation.
