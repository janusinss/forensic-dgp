# Automatic removal-area detector: scores before thresholding

Verified local diagnostic and development visual review, 10 October 2026.
The current detector produces weak or misplaced covering scores on these exposed
inputs before the fixed threshold and margin. A global threshold adjustment is
not supported by the saved score ordering. Automatic covering proposals remain
unqualified; the app, weights, threshold and three-pixel margin are unchanged.

## Evidence and execution

The frozen protocol is
`dac328b28b9a9d6a3a8aa908566e91013094f221de32faebbebc43973c1d63c4`.
Results: `outputs/automatic_proposal_score_v1/results.json`, SHA256
`7235d409f5f2ea0a0f9d6858a1f3895893865b7cfd171606672b8db070f5ca70`.
There are 36 production review-mask calls on 18 original photographs and their
fixed composite synthetic degradations. The `_native` suffix means original
photograph in this fixture; none is native CCTV. All seven covering families,
uncovered/clear-glasses controls and four historical input-only exclusions are
retained. These inputs have already been exposed to development review.

The trace saves the actual neural inputs and memory layout, logits, sigmoid
scores, resized scores, raw masks and delivered proposals. All 36 proposals
exactly match the historical current-app results. Only the retained detector
segmenter runs: no DGP, completion generator or internal generator forwards,
gradient queries, backwards, optimizer updates, VM connections or generation
requests occur. The detector state is identical before and after:
`73d2effeb52bb7dea4740ea614b78518b74c285f4d4bfefbda53ea9bc6db2661`.

The prospective independent checker verifies 174 source bindings, 153 artifacts,
180 exact 256-pixel gallery cells, all score/region arithmetic and 18 paired
records. Independent sigmoid arithmetic differs by at most 5.960465e-8 under the
declared 2e-7 calculation bound; thresholded masks remain exact. Two fresh
detector replays reproduce exact results with the recorded input memory layout.
This checks numerical and artifact integrity, not segmentation quality.

All nine pages and all 36 cases were actually inspected at original resolution.
Per-case observations are bound in `visual_review.json`; the later
`review_independent_audit.json` verifies all page hashes, 36 observations, four
exclusions and 28 saved score witnesses. The creation-time pending flag in the
first audit is superseded by these later records. The primary-assistant visual
review is development review, not independent final review.

## Findings

| Fixed exposed fixtures | Original photographs | Synthetic degraded photographs |
|---|---:|---:|
| Inputs | 18 | 18 |
| Empty proposals, all inputs | 5 | 14 |
| Eligible covering cases with fixed nonempty assisted core | 14 | 14 |
| Empty proposals among those covering cases | 4 | 11 |
| Cases marking fixed protected appearance | 6 | 2 |
| Marked protected pixels, summed across these fixtures | 14,009 | 514 |

Empty proposals after degradation also miss the covering: fewer marked protected
pixels do not establish better detection. Original versus degraded differences
describe the composite input perturbation; they do not isolate blur, measure
population performance or establish CCTV generalization. Assisted footprints are
fixed approximate operator annotations, not expert segmentation ground truth.

Cloth and pink masks leave central material while some proposals include eyes,
ordinary hair or background. Dark sunglasses, several opaque glasses and strong
glare have empty proposals. Mirrored lenses receive substantial scores but leave
gaps and include unwanted material. Hands, obstructing hair and small objects
are mostly missed. Knitted scarves produce substantial partial scores along
with unwanted visible appearance; several degraded scarf and hand cases become
empty. An uncovered original control falsely marks 11,683 protected pixels.
The clear-glasses control is correctly empty. Four held cases remain excluded;
reviewing a proposal does not authorize generating a face for them.

A separately labeled **post-hoc** saved-score analysis finds an ordering inversion
in each of the 28 eligible covering records: one fixed core pixel has a lower
score than one fixed protected-appearance pixel. Any single pointwise threshold
that captures that low-scoring pixel also captures the higher-scoring protected
pixel. The 28 exact coordinates and scores are independently checked. This
demonstrates the limitation for those witnesses only; it is not proof about
every family, spatial dilation, perfect-mask necessity or all possible detectors.
No threshold sweep, new annotations or new acceptance rule is used.

## Consequence for the goal

Repeated threshold/margin processing alone has no demonstrated basis to solve
these automatic proposals. A detector/data design review is warranted before a
new finite manual-VM learning recipe. Its evidence must retain genuine covering
labels, visible-appearance controls, source/exposure boundaries and separate
automatic versus assisted evaluations. Saved proposal inspection does not resolve
the separately observed central completion-generator defects.

No candidate is promoted. Automatic and assisted completion remain unqualified
as separate workflows. Original/mask/result presentation, optional correction,
PNG and bundle downloads, insufficient-information handling, all seven families,
independent final review and the full DGP-led restoration scope remain required.

Design: `CCTV_DGP_AUTOMATIC_PROPOSAL_SCORE_V1_DESIGN.md`.
Evidence: `outputs/automatic_proposal_score_v1/`.
