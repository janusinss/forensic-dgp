# Decoder activation V1: measurement completed, model stop retained

The manual L4 measurement completed in **86.30 seconds** with **70 gradient
queries, zero optimizer updates and no new candidate direction or checkpoint**.
The returned archive is fully hash-verified: 446,796,534 bytes,
SHA256 `a73582dc8804d649e62193d6acecdb61544b5b5728cf9f65d80845350828c77f`. Its 535 regular members were checked
before extraction and rehashed afterward. Protocol: `b93c7a8ce3ab730f1d20f8ebc8786db18469e8652db26554cf0b116dc5785481`.

The original VM saved-data checker stopped on missing gate metadata. The first
local checker also stopped on exact norm-reduction equality. A separately
reviewed local R2 checker completes the numerical audit; the original and R1 failures,
sources, protocol and returned files remain intact. This is a completed
measurement/audit, not a successful restoration model.

| Retained state | Mean actual batch-five objective | References with worse objective |
|---|---:|---:|
| initial | 5.200000023 | 0/20 |
| probe_1_16 | 5.218366084 | 14/20 |
| probe_1_4 | 5.288806810 | 18/20 |
| fit_u001 | 5.801023933 | 20/20 |

Lower objective is better. All 400 RAW/floor-PNG outputs equal the previously
audited, visually reviewed hinge-repair controls. All 80 fresh batch-five
objective records, 70 saved parameter gradients and exact before/after model
and CPU/CUDA RNG states are checked. Saved gradient arithmetic was independently
recomputed; no fresh neural or autograd replay occurred locally. The first
fitting-reference clipped gradient equals its retained original bytes exactly.

At the smallest retained placement, reconstruction plus anchors improve by
0.018446591, but regression penalties add 0.029102959 and brightness penalties
add 0.007709694. The total mean objective therefore worsens by 0.018366062.
The actual fitting reference has a negative initial selected slope, yet its
finite objective rises. Both sampled references improve all four degraded
landmark training slopes while worsening low-light and compound RGB MSE slopes.
Those positive preservation slopes exhaust the tiny regression allowances far
before the tested 1/16 placement. Re-evaluating the actual hinge offsets in a
finite affine scalar model explains the sign change substantially better than
extrapolating the infinitesimal slope alone. This is explanatory arithmetic,
not a certified finite neural or PNG prediction.

The largest batch-five versus historical batch-fifteen objective difference is
4.1781e-7. That observed discrepancy is too small to explain these finite
objective rises; exact batch parity is not claimed. Brightness activation
alone also does not reverse the fitting reference's initial descent prediction.
There is no evidence here that more epochs or scaling this same direction
will preserve the face.

Historical quality gates remain failed: at the full retained update the RAW
structure-error reduction is **0.113913%**, PNG **0.128470%**, with **30 RAW and
30 PNG group/metric preservation failures**. The smaller placements also fail.
Structure means landmark-supported high-frequency luminance MSE reduction;
recognizer similarity is a separate metric. These are exposed photographic
TRAIN results, not native CCTV, final review or identity recovery.
Dataset labels do not establish ethnicity or local performance; no Zamboanga
CCTV evidence was created by this diagnostic.

R2 changes only declared checker boundaries. Four missing historical metadata
fields match the frozen parent gate exactly. Exact dyadic integer arithmetic
proves the unchanged parameter radius; the saved approximate norm remains
explicitly unverified rather than rewritten or accepted with a ULP tolerance.
Only the derived `finite_arithmetic.objective` scalar uses the already frozen
absolute 1e-10 arithmetic bound: 14/80 rows differ, at most 8.8818e-16. Every
other field in that comparison remains exact. The original bitwise scalar
assertion did not pass. RAW/PNG scoring and scientific gates are unchanged.
Stored VM scalar differences are checked exactly against their stored operands,
separately from bounded local recomputation. Typed nested-record comparisons
retain exact keys, lengths and nonnumeric types with the existing per-call
numeric bounds. Actual records were prechecked before the full output audit.

[Full local R2 saved-data audit](<C:/xampp/htdocs/YEAR 4/Testing/outputs/cctv_dgp_decoder_activation_v1_analysis/independent_audit_r2.json>).
[Explicit R2 changes, exact radius proof and original failures](<C:/xampp/htdocs/YEAR 4/Testing/outputs/cctv_dgp_decoder_activation_v1_analysis/independent_audit_r2_schema_r2.json>).
[Independent causal arithmetic](<C:/xampp/htdocs/YEAR 4/Testing/outputs/cctv_dgp_decoder_activation_v1_analysis/independent_causal_review.json>).
[Independent return readback](<C:/xampp/htdocs/YEAR 4/Testing/outputs/cctv_dgp_decoder_activation_v1_analysis/independent_readback.json>).
[Selected next design requirements](<C:/xampp/htdocs/YEAR 4/Testing/CCTV_DGP_POST_ACTIVATION_V1_REVIEW.md>).

Only two references have these underlying gradients; the other 18 have finite
replay evidence. No new native/final images, completion masks or visual sheets
were used. No new visual review is claimed because all output pixels equal the
already reviewed controls. The original app model and design remain unchanged.
All five milestones and seven automatic/assisted covering families remain open.
