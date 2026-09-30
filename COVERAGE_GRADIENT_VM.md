# Zero-update mixed-gradient audit

Upload `outputs/coverage-gradient-code.tar.gz` and its `.sha256` file from
`C:\xampp\htdocs\YEAR 4\Testing\` to the VM home directory.
These add two files to the existing coverage bundle without changing its training
code, checkpoints or inventory. No git pull or training restart is needed.

```bash
cd ~
sha256sum -c coverage-gradient-code.tar.gz.sha256 &&
tar -xzf coverage-gradient-code.tar.gz -C ~/forensic-dgp/coverage_vm_bundle &&
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
python3 -u scripts/audit_coverage_gradients_vm.py
```

Download the printed `coverage-gradient-results.tar.gz` path and save locally to
`C:\xampp\htdocs\YEAR 4\Testing\outputs\coverage-gradient-results.tar.gz`.

The audit uses the first six epoch1 extended batches at parent/control-final/
extended-final weights. It computes real-existing-covered, real-added-covered
(when present), real-clear, replay-supervised and teacher gradients. Actual sample
fractions preserve the full loss weighting. It records losses, norms, cosines,
clip scaling and sum-of-gradients relative error (tolerance1e-4, a numerical
verification tolerance, not a changed quality gate). Cosines with near-zero
norms are undefined and reported null. No validation/test gradients.

Two helper tests passed for objective/gradient equivalence and zero/opposed
cosines. This does not verify CUDA execution, which remains pending. There is no
optimizer construction or step; all model state is checked unchanged after each
checkpoint's six batches. Earlier VM files are hash-checked before use.

The result is a local-direction diagnostic, not a simulation of AdamW updates or
proof that conflict causes validation failure. Batch composition and added-image
coverage are reported explicitly; six batches do not represent the full trajectory.
Next: inspect returned evidence before proposing one specific training change.
