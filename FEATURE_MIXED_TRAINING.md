# Bounded mixed-feature training — 29 September 2026

Motivation: real-only feature heads passed the real aggregate gate but failed
synthetic retention, especially after the presence gate. Train both heads on the
required domains before evaluating again; do not fit on cached benchmark cases.

## Source verification

Selected40 sources from previously verified replay membership:20 Asian and20 FFHQ,
seed42. All source hashes and original training-split membership checked. Excluded
4196 unique hashes covering original validation data, benchmark sources and reviewed
real crops/source images. No exact overlap; identity and near-duplicate separation
across datasets remains unverified. Source manifest
`outputs/feature_mixed_training/sources.json` SHA256
`2bc081baab19614dbc1d8b550e2320ce0a978675e276964a764c0328d48c77ca`.

Each training source yields10 fixed variants: five covering kinds, clean/degraded.
`CompletionDataset(validation=True)` is used only to enumerate these variants;
their source membership and partition are TRAINING. They are separate from the400
held-out benchmark cases. V3 real training has68 images (43 covered/25 uncovered).
Real embedding reuse verified image hashes/order and original cache provenance;
targets are loaded fresh from V3. Validation feature folders are never read.

## Frozen recipe

- SAM2.1 tiny encoder frozen; exact hashes recorded. Start pixel head from the
  nonempty-Dice diagnostic and presence classifier from its saved final probe.
- Six balanced groups: real covered/uncovered, Asian synthetic covered/uncovered,
  FFHQ synthetic covered/uncovered. Batch12 contains2 per group. Group sizes
  43,25,160,40,160,40; shuffled cycling exposes all468 examples every epoch.
-20epochs,80steps/epoch,1600updates. AdamW lr0.001 for both heads,weight decay0.0001,
  gradient norm clipping1.0. Pixel loss is all-image BCE plus nonempty Dice terms
  divided by batch size. Presence loss is BCE on balanced batches.
- Pixel/presence thresholds0.5. Final epoch only, no validation-driven checkpoint
  or threshold selection. Report training metrics separately for real/synthetic.
- After training, evaluate once on fixed V3 real and400 synthetic validation with
  unchanged safeguards, including mannequin/source/glare reports. Only a qualifying
  candidate proceeds to completion-output comparison. No deployment from train fit.

Glare data remains sparse: two real training examples, one also wearing a mask.
This run does not establish comprehensive glare coverage or recover hidden truth.

## Current execution

VM package prepared: see `FEATURE_VM_RUN.md`. All 468 embeddings will be recomputed
on CUDA rather than mixing the old CPU real cache with GPU synthetic features.
Numerical differences are recorded in the protocol. Four non-training safeguard
tests and Python/Bash syntax checks pass; GPU preflight and training remain pending.
Bundle location: `C:\xampp\htdocs\YEAR 4\Testing\outputs\feature-mixed-vm-bundle.tar.gz`;
VM extraction: `~/forensic-dgp/feature_vm_bundle/`.

**Stopped at user request before optimizer updates. All future training must use
the VM.** The local session was interrupted during feature extraction. No final
checkpoint or training history exists. Do not resume the local training script;
it is a research prototype requiring a portable VM runner/preflight bundle.
Local preparation and result evaluation can continue. The launch details below
are historical, not a live-run status.

Running locally on CPU, four torch threads. Approximately20–30minutes including
400 training encoder passes and head optimization. Windows sandbox requires
escalated access to isolated SAM dependencies. Do not duplicate a live run.
Runner `outputs/run_feature_mixed_training.py` compiles; runtime assertions check
feature dimensions, finite loss/gradients and complete per-epoch sample exposure.
The encoder, completion generator and application checkpoint remain unchanged.

Local root `c:\xampp\htdocs\YEAR 4\Testing\`; VM counterpart `~/forensic-dgp/`
only after explicit transfer. Local artifacts `outputs/feature_mixed_training/`:
sources/protocol, training-only input/mask/features, cache progress, training history
and final diagnostic `final_epoch_20.pth`. No VM job launched. No candidate promoted.
