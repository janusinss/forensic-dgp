# Nearby-gradient learning V1 R1 result — 14 October 2026

**The VM completed its finite computation, but all three proposals were
rejected. Zero updates were accepted and no improved checkpoint exists.** The
best measured structure gain was **0.687495% RAW / 0.706917% PNG**, below the
required 1% global and both-source endpoint. It also failed 32 preservation
checks in each representation. These are group–metric checks, not 32 faces.

| Trial | RAW structure gain | PNG structure gain | Failed preservation checks RAW / PNG | Decision |
| --- | ---: | ---: | ---: | --- |
| trial00 | 0.687495% | 0.706917% | 32 / 32 | Rejected |
| trial01 | 0.578604% | 0.597845% | 24 / 25 | Rejected |
| trial02 | 0.345259% | 0.361021% | 14 / 9 | Rejected |

Largest-trial source results remain separate:

| Photographic TRAIN source | RAW structure gain | PNG structure gain |
| --- | ---: | ---: |
| dataset/asian_faces | 0.643291% | 0.658014% |
| dataset/thumbnails128x128 | 0.698580% | 0.719187% |

This percentage is reduction in mean degraded landmark high-frequency error
on the same 100 exposed paired photographic TRAIN cases. It is not a fraction
of identity recovered, native CCTV performance or Zamboanga evidence. Twenty
references / five profiles were fixed; no native crop or reserved-final
identity entered optimization. Dataset names do not establish ethnicity.

## Location, cause and corrective action

Location: returned `scripts/cctv_dgp_nearby_gradient_learning_v1_r1_vm.py:394`,
`No actual raw/PNG preserving nearby-state proposal; retain stop`.
Cause of the recorded stop: none of the three actual outputs passed all
original preservation requirements. The largest proposal failed four pixel-
error, twelve SSIM and sixteen fixed-recognizer appearance checks on each
stage. The smallest still failed 14 RAW / 9 PNG checks. All proposals stayed
inside the original physical distance and brightness bounds. No unique
architectural, numerical or loss-gradient root cause is proved by this stop.
Corrective action: retain rejection and rollback; stop further training-code
changes for the required diagnostic discussion before another recipe. Adding
epochs or changing only the learning rate is not justified by these outputs.

The manual L4 worker took **380.04s** and export took
**125.61s**. It reports 200 restoration forward batches,
340 recognizer forward batches, 960 derivative queries, two derivative-sampling
displacements and three new proposals. Zero accepted updates, full epochs,
native pixels or final-identity pixels. No continuation or app change ran.
The archive's `complete: true` means export completion, not pilot success.

## What the nearby derivatives changed

All 96 original normalized dot constraints and the full Gram/dual/float32
certificate pass independent saved-vector replay. Numerical span rank is 96;
no inequality, source or loss was removed and no ridge was used. The Gram
condition number is approximately 902864859.
That sensitivity does not itself prove a wrong derivative or explain all
finite failures. The actual solve returned a certified direction in 3.865s.

The new direction's cosine with the failed RAW-only direction is
**0.999539308**. Inference: the three sampled
regions changed the tested update direction only slightly. The new sampling
recipe therefore did not produce a materially different direction in this
run, although the design and measured derivative inputs were new.

For the largest proposal, the initial derivative predicted **1.630269% RAW**
structure gain; actual RAW gain was **0.687495%**. Actual rounded displacement
dots predict descent for all 96 sampled constraints in each trial. Yet
19/14/10 of the 30 initial RAW preservation-loss groups
regress beyond their original tolerances. These derivative-group counts differ
from the overlapping 17-group scientific reporting counts in the table. Dots
from a nearby sampling state are local directional information, not predictions
of the entire initial-to-proposal finite output change.
[All saved predictions and actual outputs](<C:/xampp/htdocs/YEAR 4/Testing/outputs/cctv_dgp_nearby_gradient_learning_v1_r1_analysis/finite_response_review.json>).

The cited gradient-sampling paper uses random nearby gradients, a quadratic
program and a line search under stated assumptions. Our deterministic,
multi-constraint experiment differs; its theorem is not a RAW/PNG face-
preservation guarantee. The result rejects this particular finite recipe,
not the paper or every possible sampling approach.
[Burke, Lewis and Overton, primary author-hosted paper](https://cims.nyu.edu/~overton/papers/pdffiles/gradsamp.pdf).

## Independent verification and retained repeatability failure

All **1,144 return-file**, **5,852 bank**
and **1,246 response** bindings pass. Return archive:
2,163,166,224 bytes, SHA256 `0b9fc3fb95c2e1d4334165fd9024e32fa7fcb57e1142f8c07fd46471d4f156d3`.
Returned source was not executed locally. The prospective checker replays
seven snapshots, 700 RAW / 700 PNG outputs, 1,400 stage metric rows, observed
pixel composition, scientific aggregates, saved embeddings, component byte
schemas, three complete derivative rounds, all 96 gradient aggregates, the
certificate, actual rounded rates/proposals and exact decisions. Neural
derivatives and embedding inference were not repeated locally.
[Independent failed-pilot audit](<C:/xampp/htdocs/YEAR 4/Testing/outputs/cctv_dgp_nearby_gradient_learning_v1_r1_analysis/failure_audit.json>).

One original cross-run elementwise derivative bound still fails, now in
`RAW/dataset/thumbnails128x128/compound_lr24/ArcFace_loss` at index 2,509,339. Actual difference is
1.61351636e-08, above its unchanged
1.130096218e-08 bound. All 32 vector-norm checks pass; the failed
vector's relative difference is 5.124772258e-07.
The prospective checker reports this separately; its overall audit completion
is not an all-parity pass or model qualification. No bound was relaxed and no
cause was assigned to CUDA nondeterminism. A saved-data regression reproduces
the exact miss and rejects erased gate failures or false promotion for all
three real trials.
[Actual failures remain rejected](<C:/xampp/htdocs/YEAR 4/Testing/outputs/cctv_dgp_nearby_gradient_learning_v1_r1_analysis/retained_failure_regression.json>).

Initial and rollback RAW arrays equal the original DGP output by bits; PNG
pixels and cached control scores also agree. Saved Torch CPU/CUDA RNG arrays
restore exactly. Fresh VM receipts record original, recognizer, frozen and
buffer hashes unchanged and successful failure restoration. These runtime
hash receipts are distinguished from independently replayed saved arrays.
The permanent original checkpoint and twelve protected local files remain
exact. No accepted full-state file exists.

## All images reviewed and next requirement

All **20 original-resolution sheets / 900 cells** were directly inspected:
input, target, initial, both derivative sampling states, all three proposals
and rollback; five rows each. No convincing useful whole-face clarity gain
over initial is established. Eye, mouth, teeth, hair and transparent-glasses
detail remain soft. Some lower-resolution source targets are already soft;
the sharp-target examples still show missing visible detail in the outputs.
Hands, ordinary hair and head coverings in these restoration inputs are
observed content; they are not hidden-region completion evaluations.
[Every-sheet visual review](<C:/xampp/htdocs/YEAR 4/Testing/outputs/cctv_dgp_nearby_gradient_learning_v1_r1_analysis/visual_review.json>).

The joint, RAW-only and nearby-gradient recipes have not established a
preserving useful update. The invalid assumption to retire is that a certified
local direction in this conditioning path suffices for useful finite outputs.
The required diagnostic question was sent; its answer is pending. No next
training recipe is selected, prepared or launched from this result.
[Post-result discussion boundary](<C:/xampp/htdocs/YEAR 4/Testing/CCTV_DGP_POST_NEARBY_LEARNING_REVIEW_V1.md>).

Keep the original app DGP as the permanent baseline. Its architecture is
DeblurGAN-v2-compatible; the comparison's StyleGAN2 bank is externally
pretrained and frozen. No own-trained prior or replacement qualification is
claimed. Original checkpoints, splits, caches, provenance, full states and
failures remain. Real native CCTV stays unpaired; synthetic paired evidence
and final identities stay separate. All five milestones and seven automatic/
assisted covering families remain required. More epochs and app promotion
remain unqualified. Actual training/gradients/updates stay human-manual L4
tmux; direct agent VM access stays inventory-first, hash-bound maintenance
only. Detector-confidence recording remains prepared and UNRUN. Goal incomplete.
