# Retention pilot — existing VM workspace

Nine local helper/guard/weighting/stop-rule tests pass. Actual CUDA execution is
unverified. No local model training occurred. This pilot has at most21 scheduled
batches, four candidate rates per batch, and stops after three fully rejected
batches. It is a feasibility experiment; lower training loss alone cannot qualify
its checkpoint for use in the application.

Upload these two local files through Google Cloud SSH's Upload File control:

- `C:\xampp\htdocs\YEAR 4\Testing\outputs\retention-code.tar.gz`
- `C:\xampp\htdocs\YEAR 4\Testing\outputs\retention-code.tar.gz.sha256`

The package uses the existing completed coverage workspace and replay cache. It
contains three new files only: runner, helper and protocol. Git pull is not a
substitute for uploading this package; these local files have not been pushed.

In SSH:

```bash
cd ~
sha256sum -c retention-code.tar.gz.sha256 &&
tar --keep-old-files -xzf retention-code.tar.gz -C ~/forensic-dgp/coverage_vm_bundle &&
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
python3 -u scripts/train_retention_vm.py --preflight --output outputs/retention_preflight_vm &&
python3 -u scripts/train_retention_vm.py
```

Preflight checks CUDA, hashes, the frozen schedule and all638 replay inputs, then
performs a mixed-batch forward pass with zero optimizer updates. The full pilot
measures parent ceilings independently and evaluates real/synthetic safeguards
at the retained final state. Parent baseline, generator and application weights
remain unchanged. A constraint-driven early stop still exports results.

Output directories and the result archive must not already exist. If a command
fails, preserve its output and inspect the error; do not delete directories or
start another run automatically. No packages are installed by these commands.

After successful completion, use SSH's Download File control with this path:

```text
/home/janusdominic0/forensic-dgp/coverage_vm_bundle/retention-results.tar.gz
```

Save locally as
`C:\xampp\htdocs\YEAR 4\Testing\outputs\retention-results.tar.gz`.
It includes final checkpoint/masks, parent metrics, trial logs, ceilings, stop
reason, executed source and hashes. `best_detector.pth` exists only if both
unchanged validation gates pass; it is not automatically deployed.

Next after return: verify each acceptance against fixed ceilings, recount final
masks and selection, inspect visual masks and verify generator invariance. Only
a qualifying candidate advances to end-to-end completion previews.
