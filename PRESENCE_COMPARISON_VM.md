# Presence architecture comparison — 29 September 2026

Status: implemented and unit-tested; GPU execution pending. This is a detector
diagnostic, not a new completion model or an application deployment.

## Fixed comparison

Global arm: 1x1 average pooling, channel LayerNorm, linear output (257 parameters).
Spatial arm: 4x4 average pooling, per-cell channel LayerNorm, flattened linear
output (4,097 parameters). Spatial layout and parameter count both change; an
improvement would support this architecture package, not isolate location from
capacity as a causal explanation. Both arms start fresh at seed 42. Their unequal
dimensions mean initial weights are not identical. The previous warmed-up global
head is not a matched control for this new experiment.

Both arms use 68 V3 real training cases and the same 400 cached synthetic training
cases, six balanced groups, batch 12, seed 42 batch order, 20 epochs, 80 updates
per epoch, AdamW lr 0.001, weight decay 0.0001, gradient clipping 1.0, BCE, threshold
0.5 and final-epoch checkpoints only. Encoder features are detached. The frozen
pixel detector is not loaded into the optimizer or modified. Real features are
recomputed with the same frozen GPU encoder; synthetic features are verified
against their saved provenance and input/mask hashes before reuse. Source hashes,
original split membership, checkpoint hash and V3 manifest are checked.

There is no threshold sweep or validation-dependent checkpoint selection.
The script exports both final heads, per-epoch training metrics, 468 per-case
training probabilities per arm, and protocol. Local evaluation will assess both
arms using existing fixed validation caches and unchanged pixel predictions.
Current raw segmentation still fails retention, so better presence classification
alone cannot qualify the whole pipeline. Further pixel-model work and end-to-end
completion evidence remain required.

## Execute on the existing VM

Upload `C:\xampp\htdocs\YEAR 4\Testing\scripts\compare_presence_heads_vm.py`
with Google Cloud SSH Upload File. Use the existing tmux session, with no other
training process running:

```bash
tmux new-session -A -s dgp_training
```

Inside tmux:

```bash
cd ~/forensic-dgp/feature_vm_bundle
source .venv/bin/activate
python -u ~/compare_presence_heads_vm.py --root "$PWD"
```

No new dependency install or full bundle upload is required. The script refuses
CPU execution and checks for at least 6 GiB free GPU memory via the bundle helper.
Existing output prevents a rerun; preserve results and inspect failures before
retrying. No remote execution is available from the local workspace.

Download the path printed by DONE:

```text
/home/janusdominic0/forensic-dgp/feature_vm_bundle/presence-comparison-results.tar.gz
```

Save to `C:\xampp\htdocs\YEAR 4\Testing\outputs\`. Next: inspect training fit and
evaluate both final heads on the fixed 25 real and 400 synthetic validation cases,
including glare and source strata. No model promotion based on training loss.

## Local checks

Three unit tests passed: CPU refusal before data access, encoder feature gradient
isolation, and location sensitivity versus global pooling invariance. The tests
perform forward/backward checks only, with zero optimizer updates. Python syntax
compilation passed. These checks do not substitute for GPU execution or quality
evaluation.
