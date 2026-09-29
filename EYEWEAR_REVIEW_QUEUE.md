# Training eyewear review — 29 September 2026

27 existing training-source candidates screened and visually reviewed on three
aspect-preserved pages. No labels were added and training remains disabled.

The pool contains 12 dark/mirrored-eyewear annotation candidates and 11 proposed
clear-eyeglass controls. Four require a different role or further review: index
14 shows skin highlights without visible eyeglasses; 52 has an ambiguous far-lens
region; 212 has colored circular overlay marks spanning face/lenses; 224 has a
hand/microphone near the lower face. These are source-only visual judgments at
the supplied resolution, not expert-verified pixel masks.

Dark/mirrored sunglasses hide eye detail, but they are not equivalent to localized
reflections on transparent glasses. This queue does not yet provide a useful new
set of confirmed localized-glare positives. Do not count all eyewear as glare,
fill entire transparent lenses, or silently turn ambiguous cases into negatives.

## Membership and evidence

Every selected source belongs to the original Phase 4 training split, with current
file hashes checked against the candidate manifest. Exact exclusions were rebuilt
from original validation content, the completion benchmark sources and all V3 real
reviewed source/image hashes (4,196 hashes). Prior candidate screening found no
decoded duplicates or DCT-hash flags against 4,200 reference paths. That heuristic
does not certify identity-disjointness or catch every near duplicate.

Selection used prior visual source roles and control appearance, not candidate
model predictions. Originals, dataset splits, validation labels and application
models remain unchanged. No uncovered reconstruction targets were invented.

Script: `scripts/prepare_eyewear_review_queue.py`.
Local artifacts: `C:\xampp\htdocs\YEAR 4\Testing\outputs\eyewear_review_v1\`
(`queue.json`, `source_review.json`, three review pages).
These are local review artifacts, not a VM training package. A future VM mirror
would be `~/forensic-dgp/outputs/eyewear_review_v1/` only after explicit packaging.

## Next

Search a bounded, deterministic additional pool from original **training** sources
for transparent lenses with localized reflections. Exclude this reviewed pool,
held-out membership/content and known duplicates before review. Inspect source
images without detector-score selection. Keep opaque eyewear, clear controls,
localized reflection candidates and uncertain cases distinct. Annotation proposals
must preserve visible eye regions; only reviewed masks can enter a versioned pilot.
Do not reuse validation/test glare examples for training. Synthetic retention and
independent end-to-end completion evaluation remain mandatory.
