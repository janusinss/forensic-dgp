# Matched pixel-head comparison — 29 September 2026

Prepared for GPU execution; not yet trained or validated. No local optimizer
updates occurred. Unit tests check equivalent starting predictions, feature
gradient isolation and CPU refusal; syntax compilation passes.

## Fixed protocol

Two arms start from the same mixed-pilot pixel weights. Pointwise uses the existing
1x1 hidden convolution. Context uses 3x3, padding 1, with the original kernel copied
into the center and all surrounding weights zero. Bias and output layer are copied.
GPU preflight compares initial logits on two training inputs before any updates;
differences above the fixed tolerance stop the run. Models have different parameter
counts, so spatial context is not isolated from model capacity.

Both arms: 468 training examples (68 reviewed V3 real, 400 synthetic), fixed six-group
balanced sampling, seed 42, batch 12, 20 epochs x 80 updates, AdamW lr 0.001,
weight decay 0.0001, gradient norm limit 1.0, all-image BCE plus nonempty Dice
terms averaged over the full batch. Identical initial functions, data, batch order,
loss and budget; separate fresh optimizers. Thresholds remain 0.5. Final epoch
only; no validation-driven checkpoint selection. The SAM encoder and the saved
spatial presence classifier are frozen. Presence is used only in metric evaluation.

Verified cached synthetic features are reused; the 68 real features are recomputed
on CUDA. The runner checks source byte hashes, original split membership and split
disjointness, V3 manifest, mask/input/cache provenance, encoder and checkpoint hashes.
Initializations and CPU exports are versioned diagnostic formats, not app models.
The recipe does not change the target masks or acceptance gates.

## Run on the existing VM

Upload `C:\xampp\htdocs\YEAR 4\Testing\scripts\compare_pixel_heads_vm.py` to the
VM home directory through Google Cloud SSH Upload File. Existing mixed-pilot and
presence-comparison results must remain in the bundle folder.

```bash
tmux new-session -A -s dgp_training
```

Inside tmux, with no other training running:

```bash
cd ~/forensic-dgp/feature_vm_bundle
source .venv/bin/activate
python -u ~/compare_pixel_heads_vm.py --root "$PWD"
```

The runner refuses CPU execution, checks at least 6 GiB free VRAM and refuses an
existing output directory. Preserve any interrupted output for inspection; it
does not automatically resume. Each epoch prints real and synthetic raw/gated
training metrics. No new dependency installation is needed.

Download the archive at DONE:

```text
/home/janusdominic0/forensic-dgp/feature_vm_bundle/pixel-comparison-results.tar.gz
```

Save in `C:\xampp\htdocs\YEAR 4\Testing\outputs\`. Both final checkpoints,
training history, source-case metadata and protocol are included. Next: local
inference on fixed 25 real and 400 synthetic validation cases, raw and composed
masks, unchanged safeguards, glare/source reports and visual inspection. Only a
qualifying full detector proceeds to end-to-end completion comparison. Improvement
is an experiment outcome, not guaranteed by passing the local code tests.
