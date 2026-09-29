# Expanded detector data — 29 September 2026

Data checks passed; GPU runner and package are implemented. VM preflight remains
pending; use `EXPANDED_FEATURE_VM.md` for upload/run commands. This is detector
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

## Matched experiment design and storage implementation

Both arms use the same 352 sources, 3,472 variant identities and 48 rejection
records. `placement='fixed'` uses legacy eye/lower geometry with the anatomical
arm's texture RNG. This avoids confounding placement with a different color/noise
draw. Generic and clear cases retain exact legacy behavior. Camera RNG and source
targets match across arms. Geometry changes naturally change degraded boundaries.

The planned architecture is the existing 3x3 context pixel head and 4x4 spatial
presence head, with SAM2 frozen. Both trainable heads start from identical states
in both arms: center-expanded mixed pixel weights (checkpoint SHA
`eaa16229f88d416ad09d813a6f08ca0ae72bba214cb8266648bed382603864a4`) and spatial
presence weights (SHA `b7af5b8568fc1fd1a6be289a6599794c6c2976e162a351be90a53344997b40c4`).
AdamW LR .001, weight decay .0001, gradient norm cap 1, seed 42,
20 x 80 updates, batch 12. Loss: all-image pixel BCE plus nonempty per-image Dice
averaged over the whole batch, plus presence BCE. No hard-visible penalty in this
feature-head experiment. Thresholds remain .5; save final epoch only.

This isolates geometry between these two expanded-data arms. Comparisons against
earlier runs cannot attribute gains solely to added sources: presence is now
trainable, and the source mix differs. Real glare training coverage remains sparse.
No threshold/epoch sweep or repeated unchanged run is planned.

`feature_disk_cache.py` implements float32 feature / uint8 target memory maps,
row checksums including metadata, copied batch reads, atomic progress metadata and
explicit completion. Partial caches are preserved and refused for training rather
than automatically resumed or overwritten. Unit tests inject truncated files,
changed feature bytes, wrong provenance, nonfinite features and invalid masks.
The runner must reserve about 28 GiB for both arms' feature caches plus overhead;
use at least 35 GiB free disk in its preflight. Batch reads do not load the entire
cache into RAM. No actual feature extraction or fitting has run in this update.

The historical data manifest retains its original implementation hashes. The
matched check records the current code hashes separately; the future bundle must
pin those hashes rather than treating the historical loader hash as current.
Runner and packaging are implemented in `scripts/train_expanded_feature_vm.py`
and `scripts/build_expanded_feature_bundle.py`. GPU preflight remains pending.

Full matched replay subsequently passed: 3,472 cases per arm, 1,360 anatomical
variants per arm, identical 48 rejected variants and 2,112 legacy-equal generic
cases. The 2,954,499 pixels covered by both clean anatomical/fixed masks have
identical texture values. See `outputs/expanded_feature_data_v1/matched_check.json`
for current code hashes. Eighteen targeted tests pass. These checks establish
data/storage behavior only; no GPU run or improved detector/completion claim.
