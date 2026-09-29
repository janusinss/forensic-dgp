# Raw pixel-head error audit — 29 September 2026

The errors are not confined to mask edges. Do not apply global mask dilation or
tune thresholds against validation. A spatial-context pixel-head comparison is
the next bounded experiment; benefit is unproven.

## Method and scope

Frozen mixed-pilot pixel checkpoint
`eaa16229f88d416ad09d813a6f08ca0ae72bba214cb8266648bed382603864a4`.
Presence gating disabled for this diagnosis. A fixed four-pixel square
morphological band at 256x256 splits false positives into edge/far and false
negatives into edge/interior. This is diagnostic geometry, not a replacement for
the existing acceptance metrics. Counts partition exactly. Two geometry tests
pass, including empty-target behavior.

68 real training cases evaluated with cached encoder features and current V3
targets; original V2 cache manifest/order/image hashes verified. 77 synthetic
training features from the stopped local cache verified against saved input,
target and encoder hashes. This partial subset covers the first eight Asian
training sources, not a representative full training set. No training occurred.
All 25 real and 400 synthetic validation raw masks reused and recounted against
targets; their TP/FP/FN totals match the previous evaluation exactly.

| Set | Cases | Missed pixels near edges | False positives far from edges |
|---|---:|---:|---:|
| Real training | 68 | 84.50% | 27.18% |
| Synthetic training subset | 77 | 78.06% | 22.61% |
| Real validation | 25 | 74.41% | 47.36% |
| Synthetic validation | 400 | 81.85% | 60.28% |

Percentages use error pixels as denominators, not all image pixels or cases.
Synthetic validation has 242,193 missed pixels: 228,213 on degraded cases versus
13,980 on clean cases. Degraded targets can include larger prescribed boundary
margins, so these absolute counts do not isolate a causal effect of blur.
Of 207,757 synthetic false-positive pixels, 125,237 are far from target edges;
41,304 are on clear controls. Far errors therefore also occur on covered images.
The visible-region preservation requirement remains unresolved.

## Next controlled experiment

Current pixel head is two pointwise (1x1) convolutions over a 64x64 embedding map,
then bilinear upsampling. It has no learned local spatial convolution. Training
evidence shows both missed boundary coverage and extra regions; test spatial
context before adding new losses or changing labels.

Compare the existing pointwise head with a version whose first layer is 3x3,
padding 1. Copy the existing 1x1 kernel into the center and zero other kernel
positions, preserving the initial function (within numerical tolerance). Copy
bias and output layer. Start both from the same mixed-pilot pixel checkpoint.
Keep encoder frozen, same 468 training cases, six-group batch order, optimizer,
loss and update budget. Freeze the already trained spatial presence classifier
for composed-mask evaluation. Save both final heads; no validation epoch or
threshold search. Parameter count and spatial context both change, so this is an
architecture comparison, not a capacity-controlled causal proof.

Implementation and GPU run are pending. Unit-test the initialization equivalence
and gradient isolation before preparing VM commands. All fitting stays on VM.
Keep original real/synthetic safeguards, report raw and composed masks separately,
and require completion-output evidence before application promotion. This audit
does not resolve limited real-glare coverage or establish hidden facial truth.

Evidence: `outputs/pixel_boundary_audit/results.json`.
Local root: `C:\xampp\htdocs\YEAR 4\Testing\`.
VM root: `~/forensic-dgp/feature_vm_bundle/`.
