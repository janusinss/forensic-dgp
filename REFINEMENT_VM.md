# Refinement RGB ablation: VM execution

Upload both local files using Google Cloud SSH:
`C:\xampp\htdocs\YEAR 4\Testing\outputs\refinement-vm-code.tar.gz`
and `C:\xampp\htdocs\YEAR 4\Testing\outputs\refinement-vm-code.tar.gz.sha256`.
The checksum uses Linux line endings. This code-only bundle reuses existing cached
features and control checkpoint; no git pull or new encoder download is required.

```bash
cd ~ &&
sha256sum -c refinement-vm-code.tar.gz.sha256 &&
test ! -e ~/forensic-dgp/expanded_feature_bundle/refinement_experiment &&
tar -xzf refinement-vm-code.tar.gz -C ~/forensic-dgp/expanded_feature_bundle &&
tmux new-session -A -s dgp_training
```

Inside tmux:

```bash
cd ~/forensic-dgp/expanded_feature_bundle &&
source ../feature_vm_bundle/.venv/bin/activate &&
python -u refinement_experiment/train_refinement_vm.py --root "$PWD" --preflight &&
python -u refinement_experiment/train_refinement_vm.py --root "$PWD"
```

Preflight verifies original bundle/source/input/target provenance for all 3,540
training rows and tests a full-batch GPU forward/backward pass with zero optimizer
updates. It reports peak allocated memory. Real and synthetic RGB must match the
exact cached input hashes. Cached features are checksummed when consumed.
Any mismatch stops; preserve output and report the error rather than bypass checks.

Fixed experiment: RGB-off semantic control versus RGB-on, identical initialized
states and schedules, 10 epochs x 80 updates each, seed 42, batch 12, AdamW LR
0.001, weight decay 0.0001, clip 1. Parent pixel head/presence gate frozen. Original
BCE+covered Dice and constant frozen-gate BCE; no border weighting. All training
examples must be seen. Final checkpoints only. GPU runtime remains unmeasured.
Six local tests pass; bundle member bytes verified; no local optimizer updates.

After DONE, download:

```text
/home/janusdominic0/forensic-dgp/expanded_feature_bundle/refinement-results.tar.gz
```

Save to `C:\xampp\htdocs\YEAR 4\Testing\outputs\refinement-results.tar.gz`.
The archive includes both final checkpoints, protocol, training counts/history
and executed code. Existing application/generator checkpoints are untouched.
Next: verify provenance/frozen tensors and evaluate both arms on original real
and synthetic safeguards with visual review. Known frozen-gate failures remain;
this architecture diagnostic alone cannot establish full-pipeline qualification.
