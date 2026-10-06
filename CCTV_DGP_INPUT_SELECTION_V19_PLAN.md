# V19 input selection — finite development inference

**Closed 5 October 2026:** original V19 failed its first development baseline
PNG check. The returned three-case diagnostic confirms a normalization mismatch.
Preserve this frozen plan/package/failure; do not repeat the original launch.
The separate [V19 r2 plan](<C:/xampp/htdocs/YEAR 4/Testing/CCTV_DGP_INPUT_SELECTION_V19_R2_PLAN.md>)
keeps both historical encodings and all scientific gates. Its verified manual
commands are in the [R2 runbook](<C:/xampp/htdocs/YEAR 4/Testing/CCTV_DGP_INPUT_SELECTION_V19_R2_VM.md>).
R2 inference, independent return audit and original-sheet review are complete;
its normalization correction passes but automatic preservation fails.
[R2 results](<C:/xampp/htdocs/YEAR 4/Testing/CCTV_DGP_INPUT_SELECTION_V19_R2_RESULTS.md>).
No useful native/app adoption claim follows.

The following is the original preparation snapshot.

Prepared and independently transfer-audited on 5 October 2026. **No VM launch,
training, V19 development outputs or production adoption is claimed.** V19
performs inference only; user-run transfer/launch/collection commands are in
[the VM runbook](<C:/xampp/htdocs/YEAR 4/Testing/CCTV_DGP_INPUT_SELECTION_V19_VM.md>).

## Evidence supporting this processing control

The returned V18 training is audited and all ten original-cell sheets are
reviewed. Its ten-photograph degraded fitting improves substantially, but four
clear-image preservation checks fail at update600. All original failures and
checkpoints stay intact:
[V18 results](<C:/xampp/htdocs/YEAR 4/Testing/CCTV_DGP_STRUCTURE_V18_RESULTS.md>).
Repeating V18 or scaling training before checking a concrete processing change
is unsupported.

The separate V19 input study uses only the fifty existing training inputs and
their canvas support. Runtime features use float64 luminance and a four-neighbour
Laplacian on fully supported inner pixels. Retain DGP when both Laplacian mean
squared energy is at least 5e-5 and its ratio to luminance variance is at least
0.002. Exactly constant observed luminance also retains DGP. Otherwise use the
fixed V18 terminal correction. These thresholds were chosen from the training
input study and frozen before the processing result was evaluated. Calibration
is training-based; it is not an independently validated universal blur detector.

The runtime selector accepts only an RGB256 input array and its input-canvas
support. It has no identity, filename, degradation label, clean reference,
prediction or quality score input. Profile labels are used for descriptive
training-group reports. Canvas support excludes padding; it is not a covering
removal mask or recognizer-based face qualification.

The fifty-case saved-output control takes 2.91 seconds and chooses retained DGP
for all ten clear cases, V18 update600 for all forty degraded cases. Selected
outputs are exact aliases of the audited originals, with no blending, sharpening
or preferred display processing. Its separate 2.53-second arithmetic audit
checks every choice, raw/PNG composition, selected artifact binding and metric.
All unchanged training-group preservation checks pass; degraded PSNR gain is
7.9674 dB and MSE reduction 84.0318%. This closes a training-only processing
prerequisite, not V18's failed unconditional guard or a generalization claim.

Four selector and six inference-contract regressions pass, covering flat-input
fallback, padding boundaries, rejected hidden label inputs, illegal training/
promotion/threshold refit/native use, changed automatic output aliases, exact
source grouping, safe paths, inference-only source and tmux preflight behavior.
Evidence:
[processing audit](<C:/xampp/htdocs/YEAR 4/Testing/outputs/cctv_dgp_input_selector_v19/independent_processing_audit.json>).

## Frozen development experiment

V19 uses the existing **104 development identities/520 paired synthetic cases**:
51 from `dataset/asian_faces`, 53 from `dataset/thumbnails128x128`. All five
profiles and the original ten preview identities are unchanged. Own training/
development exact identity IDs, source hashes and target RGB hashes do not
overlap. This cohort has been used in earlier diagnostics; it is development
validation, not independent final evaluation. Near-duplicate or pretrained
corpus overlap is not excluded. Folder names establish neither ethnicity nor
native CCTV capture provenance or Zamboanga performance.

Before development inference, all fifty training inputs must freshly reproduce
the retained DGP and fixed V18 terminal raw outputs within 2e-6 and the same PNG
pixels. The frozen input selector must reproduce its training choices. A failure
stops the run before development predictions; partial outputs/logs are retained.

Five arms use identical prepared 256×256 inputs:

1. Basic resizing: the already prepared RGB canvas, without model changes.
2. Retained trained DGP V2: freshly inferred, PNG pixels must match the separately
   audited V15 baseline.
3. Declared pretrained CodeFormer without statistics/fidelity w0: reuse its
   audited V15 PNG and fixed-affine embedding, with exact byte/pixel bindings.
4. Fixed V18 update600: fresh camera/DGP→R2 code head→prior64→spatial residual
   inference; no training or checkpoint search.
5. Automatic V19: exact alias of arm2 or arm4 according to frozen input features.

Pretrained CodeFormer is a comparison baseline and explicitly declared frozen
feature prior. Retained DGP remains the image base; our R2 head and new spatial
decoder are our trained components. This is a custom conditioning/residual
pipeline, not equivalence to the published DGP architecture. No AdaIN, prior RGB
tail, statistics or CodeFormer internal input-feature fusion is used in the
feature path. The separate comparison PNG is a historical pretrained output.

Raw float32 RGB arrays are exported for fresh DGP and spatial arms, separately
from their PNGs; the reused pretrained arm is PNG evidence, not a fresh raw-output
claim. PNG composition retains input padding and floors/clips to bytes. Automatic
output aliases original raw/PNG/embedding paths exactly. Source/profile/clear/
degraded reports retain every group; no aggregate gain waives a regression.

Use the unchanged guards: MSE may not increase beyond 1e-12, SSIM/fixed cosine
may not decrease beyond 1e-6 in any group; require at least 0.1 dB degraded PSNR
gain and 10% degraded MSE reduction. Both unconditional and automatic reports
are retained. The script does not require scientific pass to export; a successful
execution can report failed quality. No thresholds, loss, weights or checkpoints
are adjusted from these development results.

All baseline embeddings are checked against their audited sources. Target and
spatial output embeddings are freshly measured on the same saved affine/support.
The saved-output audit verifies fresh reference vectors against the existing
baseline within 2e-6, reconstructs cosine arithmetic and discloses embedding reuse.
No exact-identity conclusion follows from this recognizer score.

## Timing, stopping and returned evidence

| Stage | Finite limit |
| --- | --- |
| Inference, including parity and output generation | 1,200 seconds / 20 minutes |
| VM saved-output audit | 300 seconds / 5 minutes |
| Supervisor, including audit and export | 1,800 seconds / 30 minutes |
| Archive export/hash, within supervisor limit | 180 seconds / 3 minutes |
| Peak allocated VRAM / initial free disk | 20 GiB / at least 4 GiB |

At development case20, project the remaining500 cases from the19 steady samples
with a1.25 safety factor plus60 seconds. Stop if projection exceeds1,200 seconds.
The supervisor terminates a timed-out child process group with bounded grace;
it does not resume. Existing/partial root or competing GPU/tmux programs cause
launcher rejection. Idle outer tmux shells and log viewers are allowed.
No dependency installation, runtime changes, automatic repeat or shutdown.

All570 images (50 parity +520 development) run the retained DGP, prior encoder/
classifier, R2 head, prior feature generator and spatial decoder. Recognizer
forwards total624 (104 references +520 spatial outputs); prior RGB tail and unused
V11 forwards remain zero. All six loaded state hashes must match before/after,
all parameters remain frozen and no parameter may acquire a gradient. Optimizer
construction, backwards, autograd calibration and training updates remain zero.
These are planned counts; actual execution proof is pending the returned receipts.

The independent auditor rebuilds **2,080 physical PNG metric/cosine rows**,1,040
fresh raw/PNG compositions,570 input choices,520 automatic aliases, all50 fresh
training parity cases and300 original grid cells. Automatic summaries represent
2,600 logical arm rows including520 byte-identical selected aliases. Five fixed
1560×2904 sheets retain original256-pixel cells for visual review.

Twenty-four predeclared cache probes support optional local CPU head replay:
four training cases (first reference per source, clear/blur) plus twenty
development cases (fixed ten preview references, clear/blur). CPU bound5e-5 is
unchanged from V18. The local audit has300 seconds internal/330 seconds process
timeout. It does not independently replay the full CUDA/DGP/prior/recognizer
execution. Hash/receipt/saved arithmetic and CPU cache replay limits remain explicit.
Canonical `math.fsum` means avoid exact NumPy reduction equality; only derived
PSNR uses the separately regression-tested binary64 log10 ULP bound. Quality
means, decisions and scientific thresholds remain exact/unchanged.

## Transfer binding and next review

Preparation35.45 seconds uses zero neural calls and creates an immutable
80,388,709-byte archive. Separate transfer audit2.39 seconds verifies146 exact
safe members/144 assets,798 data files, Python3.10 syntax, lineage, fixed
checkpoint, own exact split separation, finite limits and derived neural counts.

Protocol SHA256:
`2d707ebba97524867c4ce7610666e3abeb29638eb9054a8025983b6476e628d2`.
Archive SHA256:
`da75c16e25c0b83183beda230d90c552c9c24c8f6c070f8a992f77042350c899`.
Launcher SHA256:
`a4aec055d34a035f02c9e27c771aad91d36efc0a22bd88415eb1b4fe7e2030b9`.
Receipts:
[preparation](<C:/xampp/htdocs/YEAR 4/Testing/outputs/cctv_dgp_input_selection_v19_preparation.json>),
[independent transfer audit](<C:/xampp/htdocs/YEAR 4/Testing/outputs/cctv_dgp_input_selection_v19_transfer_audit.json>).

After the user returns the archive, independently audit all evidence and inspect
all five original-cell sheets before choosing any further learned path or wider
training. Failed development guards stay failed; no unchanged rerun or promotion.
This processing study does not qualify insufficient crops, estimate coverings,
verify native CCTV, or fulfill the DGP-led app/override/final-review goal. The
full goal remains active; those separate requirements stay required.
