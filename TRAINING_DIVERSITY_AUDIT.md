# Training diversity audit — 29 September 2026

## Anatomical outline review: 363 sources completed

Inspected all eight `anatomy_review/page_*.png` sheets covering 363 sources, each
with eye/lower geometry outlined on the source. The eye bands frequently cover
forehead/eyebrows or only part of the eyes, including aligned FFHQ examples.
Side poses and varying crop scale cause additional mismatch. Lower polygons
usually cover mouth regions but can reach background or miss intended structure.
This is qualitative review, not a measured landmark-coverage success rate.

Correction to interpretation: the geometry target still labels pasted pixels
correctly. An eye-named rectangle over a forehead is not necessarily invalid pixel
segmentation ground truth; it is weak anatomical coverage for the intended task.
Do not conflate this with pre-existing occlusion missing from empty-mask targets.

Eleven additional source-cleanliness flags were recorded (text overlays, objects,
hair/cap occlusions or ambiguous overlaps) in `anatomy_review/results.json`.
`source_status.json` covers every reviewed source. They are review flags, not
automatic pixel annotations. Earlier provisional passes were not certifications.

Next implementation: a separately versioned training augmentation that uses
reliable source landmarks to position anatomical eye/lower coverings. The existing
InsightFace detector path can supply eye anchors; mouth points and reliability
checks must be added explicitly. Use source landmarks only for synthesizing
training examples; inference must never require an uncovered reference. Preserve
generic object/irregular shapes as generic occlusion cases. Failed landmark
detection must be recorded and reviewed, not silently treated as anatomical
success or as evidence that the source has no face. No production preprocessing
or existing benchmark target changes. The expanded VM run remains pending.

## Detailed roles for all 59 flagged images

Inspected all six enlarged contact pages in `detail_review/`. The source-role
decisions are recorded in `source_roles_v2.json`, linked by hash to the previous
triage. Outcomes among the 59 flagged sources:

- 22 return as clean candidates pending geometry review; poses and transparent
  glasses are not automatically real-occlusion labels. Goggles above the eyes in
  index 347 were a thumbnail false alarm.
- 25 contain obscuring lenses or hands/objects/clothing/another face and belong
  in a separate real-occlusion annotation queue, not clean paired targets.
- Five require rotation/crop geometry review.
- Six have text overlays and one is unusable tiny rotated content on black.

The 341 thumbnail passes retain their provisional status; full-detail review of
the flagged set does not certify the unflagged set. All originals are preserved.
The new `real_occlusion_annotation_queue.json` uses only original training sources
and has `training_enabled: false` until reviewed pixel masks and grouping checks
exist. It does not manufacture an uncovered reference for these people.

Next: inspect anatomical covering placement and source cleanliness across the
363 provisional/returned clean candidates. Use those findings to define a
versioned generator with explicit geometry checks instead of blindly scaling the
same fixed-coordinate augmentations. Keep existing benchmark data unchanged and
report the shift in training data policy as part of the next experiment. A VM
training recipe has not yet been prepared for these source roles.

## Full thumbnail triage completed

All 400 source thumbnails were inspected across eight indexed sheets in
`source_review/`. `source_triage_v1.json` records a status for every source:
341 provisional thumbnail passes and 59 quarantined for closer review. Reasons
include pre-existing obscuring eyewear, hands/objects near facial regions,
overlaid text, severe crop/rotation defects and poses needing geometry review.
These categories overlap. Quarantine is not a claim that every flagged image is
unusable, and poses/normal glasses are not automatically excluded from the task.

Six flags belong to the earlier 40-source training pool, so data quality is not
only an expansion issue. In particular, using `kind=none` on a source already
wearing opaque sunglasses produces an empty synthetic label despite real facial
occlusion. It can also present the occluded face as the reconstruction target.
This conflicts with the intended completion task. The screen supports auditing
source labels, not attributing all prior failures to these six images.

`provisional_sources_v2.json` is a separately versioned, non-training-ready pool
of 163 Asian-source and 178 FFHQ images after quarantine. Original images, manifest,
splits and benchmark targets remain untouched. No source deletions. Near-duplicate
screen results refer to the original superset and remain applicable to this subset.
Thumbnail passes are not pixel-level annotation or proof of no hidden features.

Next: inspect flagged eyewear/objects at full resolution to distinguish ordinary
transparent glasses from real occlusion. Keep genuine occlusion cases for a
separate reviewed detector-training pool, never silently treat them as clean
completion targets. Review generated covering placements on the provisional pool
before selecting a balanced training recipe. The next experiment must document
that data quality as well as source count has changed; do not claim a pure
sample-size effect. Real-glare annotations and final independent evaluation remain
outstanding. No VM training is ready from these manifests yet.

## Follow-up screen and visual review

All 400 candidates screened against 4,200 original validation / benchmark /
reviewed crop and source paths. No exact decoded-RGB duplicates and no 63-bit
DCT perceptual hashes within Hamming distance six were found, including within
the candidate pool. This heuristic does not certify identity-disjointness or
exclude cropped/rotated near-duplicates. Evidence: `duplicate_screen.json`.

The ten-source geometry preview exposed input and placement defects. Source
`asian_face_07207.jpg` is predominantly black with tiny rotated facial content;
synthetic coverings mostly occupy background. `asian_face_01405.jpg` and
`asian_face_08959.jpg` show fixed eye bands below the visible eyes in the reviewed
examples. `asian_face_09450.jpg` contains rotation/black borders. These observations
are source/label geometry issues, not detector predictions. Other reviewed FFHQ
poses also show that fixed geometry need not align with actual facial regions.

**Do not train the candidate expansion unchanged.** The original manifest is
preserved as evidence. `geometry_review.json` records flags; `geometry_preview.png`
shows source plus synthetic lower/eye variants. Review all 400 source thumbnails
for unusable inputs and document a training-only quality filter before generating
a revised manifest. Any landmark-aware augmentation must be versioned and tested
separately; current synthetic benchmarks remain unchanged. Automatic face-detection
failure alone is not grounds to discard a valid occluded face.

Prepared a candidate 400-source training manifest, not a trained model. No fitting
ran locally and no original split or application checkpoint was changed.

## Inventory and selection

Original split membership contains 9,500 Asian-source and 66,500 FFHQ training
paths; all 76,000 are present locally. This presence count is not a full decode
audit of all files. The selected 400 images were decoded and hashed.

`outputs/training_diversity_audit/candidate_sources.json` contains 200 sources per
dataset, retaining the earlier 40 sources and adding 360. Selection uses seed 42,
original training membership and byte-hash deduplication. Excludes 4,196 unique
hashes from original validation, benchmark sources and reviewed real images/source
files. One duplicate encountered during selection was skipped. Near-duplicates
and repeated identities remain unverified; candidate is not ready for a final
generalization claim. No validation/test records were reassigned or relabeled.

The same ten synthetic variants would produce 4,000 examples, but no new training
schedule is implemented yet. The old sampler is hardcoded for at most 160 cases
per group, so this manifest must not be passed directly to the old runner.
Budget and exposure need explicit treatment in a controlled data expansion.

## Coverage and preprocessing gaps

Reviewed V3 training is still 68 cases: 43 covered, 25 clear, just two with strong
lens reflections. Validation contains one glare case and test one. Adding ordinary
face photos does not supply real glare supervision. These counts are insufficient
to establish coverage across the user's four approved reflection types.

Every selected FFHQ image is 128x128. Asian-source images have varied dimensions;
many are non-square. `completion_data.py` directly resizes every source to a square
before drawing synthetic occlusions. This changes facial proportions for non-square
portraits; the effect on performance is not measured. Do not silently change the
existing benchmark preprocessing or cached validation targets. A future preprocessing
change needs a versioned training pipeline and separately reported comparison.

## Next step

Screen the candidate sources for decoded-image duplicates and near-duplicate
matches against held-out sources; inspect a stratified crop/covering preview.
Document source geometry and whether simulated masks actually cover intended
facial regions. Keep architecture and thresholds fixed when testing any data
expansion; do not combine unmeasured cropping changes with a diversity claim.
Review additional training-only real glare candidates separately, preserving all
existing validation/test membership. No new VM run until this preparation is done.

Candidate path local:
`C:\xampp\htdocs\YEAR 4\Testing\outputs\training_diversity_audit\candidate_sources.json`.
Future VM transfer root: `~/forensic-dgp/feature_vm_bundle/`; not uploaded yet.
