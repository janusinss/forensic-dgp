# Expanded detector data — 29 September 2026

Data checks passed; GPU runner and package are still pending. This is detector
data preparation, not a trained improvement or verified clean completion targets.
All fitting remains on the VM.

## Frozen proposal

`expanded_feature_data.py` implements a separate training-only dataset. It requires
the original split hash and training membership, verifies every source hash, rejects
duplicate/held-out content and records invalid anatomical variants explicitly.
It retains square 256 resizing and the existing camera recipe. The original
`CompletionDataset` and validation benchmark are unchanged.

352 provisionally reviewed sources: 170 Asian-source and 182 FFHQ-source.
Ten variants per source would give 3,520 cases; 48 anatomical variants from 12
sources are rejected, leaving 3,472. Those sources remain usable for the six
generic/clear variants; rejected anatomy is never converted into an empty positive.
Anatomical coverings use explicit V2 clipping. They are partial procedural shapes,
not photorealistic accessories or simulated glare.

## Verification

Fourteen targeted tests pass across the expanded loader, anatomical augmentation
and existing VM runtime. Tests cover held-out membership, source mutation, duplicate
content, path escape, deterministic masks, crop-equivalent labels, sampling coverage,
fixed budget, overlapping groups and CPU refusal in the existing runtime.

`scripts/prepare_expanded_feature_data.py` inspected all 3,472 cases. Every mask is
binary with correct presence semantics; RGB is finite and in range. All 2,112
clear/object/irregular variants match the legacy inputs, targets, masks and geometry
exactly. Source membership was separately checked against the role review and flags.

Local artifacts: `C:\xampp\htdocs\YEAR 4\Testing\outputs\expanded_feature_data_v1\`.
Intended VM mirror: `~/forensic-dgp/feature_vm_bundle/outputs/expanded_feature_data_v1/`.
Files: `manifest.json`, `rejections.json`, `verification.json`.
Manifest SHA256: `a7644fd3da0d91186c1db4c87bb769a45f2b12fab4d4202f64bc057616130b2a`.
Preparation refuses to overwrite an existing output directory.

## Budget and balancing

Six groups contain 43 real covered, 25 real clear, 1,336 Asian covered, 340 Asian
clear, 1,432 FFHQ covered and 364 FFHQ clear cases. Two samples per group give
batch size 12. Twenty epochs of 80 steps retain the 1,600-update budget.
Shuffled streams continue across epoch boundaries, so every case appears during
the experiment. Synthetic covered cases receive two or three exposures each;
real clear cases receive 128. This balances domains/presence, not unique examples
or covering subtypes. Small real/glare coverage and repeated real exposure remain
overfitting risks. No sampling threshold was selected using validation scores.

## Next implementation

Build the VM-only runner using disk-backed float32 features: 3,540 real-plus-synthetic
embeddings require about 13.83 GiB before targets/runtime overhead, making the old
in-memory concatenation unsuitable for a 16 GiB VM. Preflight must verify GPU,
free disk, all packaged hashes, dataset/split evidence and one forward pass without
optimizer updates. Preserve incomplete caches with explicit completion records.

Predeclare model initialization, which heads train, fixed budgets and final-only
evaluation before launching. An augmentation comparison must use identical source
and variant membership in both arms. Keep existing real/synthetic gates and report
mannequin/glare separately. No validation/test fitting, threshold adjustment or
generator/application promotion. End-to-end completion improvement is still unproven.
