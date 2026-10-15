# RAW conditioning R1 returned result — 14 October 2026

**All three trials were rejected; zero updates were accepted.** The manual L4
worker completed 320 RAW derivative queries and tested three finite changes of
the 12 style/spatial conditioning tensors. Every trial failed actual RAW/PNG
preservation. The stop is at `scripts/cctv_dgp_raw_conditioning_learning_v1_r1_vm.py:363`:
`No actual raw/PNG preserving conditioning step; retain stop`.

| Trial | RAW structure gain | PNG structure gain | Failed preservation checks RAW / PNG | Decision |
| --- | ---: | ---: | ---: | --- |
| r00_trial00 | 0.669337% | 0.682289% | 32 / 32 | Rejected |
| r00_trial01 | 0.577437% | 0.598276% | 24 / 25 | Rejected |
| r00_trial02 | 0.346369% | 0.361388% | 14 / 10 | Rejected |

The best trial also misses the retained 1% global and per-source endpoint.
Source gains are 0.616700% / 0.682537% RAW and 0.624365% / 0.696822% PNG,
reported under their dataset names rather than inferred ethnicity. Its failures
include 12 SSIM, 16 fixed-recognizer appearance and 4 pixel-error checks on each
stage. The smallest trial still has 14 RAW and 10 PNG failures. Brightness
fractions stay below 0.2. A fixed-recognizer comparison is not verified identity.

The worker took 168.55 seconds, with 100 restoration forward
batches, 180 recognizer forward batches, three temporary component proposals,
zero accepted updates and zero full epochs. Export took 50.67 seconds.
`complete: true` describes the archive export; the pilot did not pass. No
accepted checkpoint or new development incumbent exists. Saved trial components
and failure records remain. No fresh post-failure RAM state hash is claimed.

All 632 return-file bindings and 5,852 required
parent bindings pass path/size/hash checks. Archive is 862,975,919
bytes, SHA256 `b589020e0fe69ebe97e2063008c4048f9c52c5db3862fc31c040cfa17eed9e4a`. The failure-only supplemental audit
replays four snapshots, 400 RAW outputs and 400 delivered PNGs, 800 stage metric
rows, observed-pixel composition, component byte schemas, all 32 saved gradient
vectors, Gram/KKT arithmetic, rates, rounded saved changes and exact rejections.
Initial RAW outputs equal the original DGP. Gradients and embeddings have not
been independently recomputed through neural models locally.
[Failure-only forensic audit](<C:/xampp/htdocs/YEAR 4/Testing/outputs/cctv_dgp_raw_conditioning_learning_v1_r1_analysis/failure_forensic_audit.json>).

## Retained prospective-checker stop

The original returned checker **did not pass** cross-run gradient parity. One
of 82,587,648 compared elements fails its unchanged `rtol=1e-4, atol=1e-8` bound:
the RAW `dataset/thumbnails128x128/lowlight_lr32/ArcFace_loss` vector at index
2,485,488 differs by 2.3003668e-8. All 32 norm comparisons pass; the affected
vector's relative L2 vector error is about 4.9630e-7. The original checker,
traceback facts and failed criterion remain. This is distinct from the VM's
image-preservation stop. Its cause is not established as CUDA nondeterminism.

The separate supplement is restricted to an already rejected zero-update
pilot. It reports the parity failure and verifies the remaining saved-output
arithmetic. It does not loosen the cross-run bound, any scientific preservation
gate, the 1e-13 image metric replay bound or qualification conditions. Regression
checks keep the actual mismatched value rejected and reject NaN/Inf/shape errors.
[Original checker stop](<C:/xampp/htdocs/YEAR 4/Testing/outputs/cctv_dgp_raw_conditioning_learning_v1_r1_analysis/original_checker_stop.json>).
[Supplement preparation and exact source differences](<C:/xampp/htdocs/YEAR 4/Testing/outputs/cctv_dgp_raw_conditioning_learning_v1_r1_analysis/forensic_checker_preparation.json>).

## Images and the next decision

All 20 sheets and 600 cells were directly reviewed at original resolution.
Visible eye, mouth, tooth and hair details remain soft, particularly on the
degraded rows. No convincing whole-face clarity gain over initial is established.
Some targets from the existing lower-resolution photo source are already soft;
this is not evidence that the ten audited HQ FFHQ targets lack detail. These are
100 paired photographic TRAIN cases, not native CCTV, final identities or a
covering-family completion evaluation.
[Every-sheet visual review](<C:/xampp/htdocs/YEAR 4/Testing/outputs/cctv_dgp_raw_conditioning_learning_v1_r1_analysis/visual_review.json>).

Removing the approximate PNG constraints improved Gram conditioning but did not
solve finite preservation. The new direction's cosine with R4 is 0.998685.
The largest trial predicted 1.642168% RAW structure gain but delivered 0.669337%.
No rounded-step dot product changes the predicted descent sign, yet 19 of the
30 RAW source/profile preservation losses regress beyond their retained bounds.
These derivative-group counts differ from the 17-group scientific reporting
counts above. No new direction, solver or model proposal was computed locally.
[96 saved prediction-versus-output comparisons](<C:/xampp/htdocs/YEAR 4/Testing/outputs/cctv_dgp_raw_conditioning_learning_v1_r1_analysis/gradient_prediction_review.json>).

The recorded next decision is to review data support and the conditioning
path's measured finite response before issuing another training recipe.
Adding epochs, smaller untested rates, a random reset or a lower quality gate is
not justified by this result. A future manual derivative/response diagnostic
must address that uncertainty explicitly; it is not a runnable packet today.
[Design review and primary research](<C:/xampp/htdocs/YEAR 4/Testing/CCTV_DGP_POST_RAW_CONDITIONING_REVIEW_V1.md>).

Original checkpoint SHA256 remains
`646fbb11674e8882ea41e251d6e6e7d468c963979b48c93132def8292834e24b`.
R1 is used and stopped; do not rerun or resume its failed recipe. Checkpoints,
app design, splits/caches/provenance/full states and gate failures are retained.
No native CCTV, ethnicity, Zamboanga or exact hidden-identity claim follows.
Training remains human-manual L4 tmux. All five milestones and seven automatic
and assisted covering families remain required. The complete goal is incomplete.
