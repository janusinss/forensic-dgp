# Current readiness: V4 packet prepared, manual launch not yet run

26 October2026. The frozen prospective design below has now been implemented
and independently audited. Use COMPLETION_CODEFORMER_DELIVERED_OBJECTIVE_V4_VM.md
for the current five-step guide. Its archive/protocol and all source snapshots
are immutable. This readiness pointer does not claim trained improvement.

---

# Proposed V4: train for the delivered256 filled region

Status: DESIGN ONLY; no runtime, upload archive or new model is ready.
The working app remains trained R1. The V3 run is closed as a quality rejection.

The independent saved-output diagnosis found that equally weighted HQ512 boundary
improvement could outweigh worse filled-region error. It also found a clear-lens
aggregate difference between HQ512 and delivered256 error. Gradient clipping was
inactive; additional epochs of the unchanged objective are not the selected fix.

The proposed distinct pilot starts from the full retained R1 checkpoint. Keep the
443 frozen tensors,54 existing trainable fusion tensors, Adam/RNG restoration,
identical256FITcases/schedule,LR1e-5,64-step ceiling and all old limits. Change
only the objective package: compare the differentiable app float256 output with
the exact existing clean256 target inside the reviewed mask. Remove the separate
additive boundary loss; keep every raw/PNG boundary acceptance gate unchanged.

The exact FIT256targets already exist:192RGB and64strict-gray cases,64unique
clean256images,32FITreferences disjoint from8monitor references. Use their pinned
clean256RGBPNG/255 values directly. Do not derive targets from downsampledHQ512,
re-grayscale them, or use monitor/challenge/final targets for optimization.

The forward path is clamp((raw512+1)/2,0,1), app-identical bilinear256 and exact
source/mask composition. Divide masked RGB L1 by3timesmaskpixels and by4once
for four accumulated microbatches. Quantization stays in PNG evaluation; do not
use a hidden straight-through estimator. Record clipping/saturation and gradient
path integrity. Exact outside pixels and empty/refusal behavior remain required.

Keep clear0/16/32/64,challenge0/64,public0/64 evaluations:224actual outputs and20
all-case visual sheets. Each paired cohort must independently improve delivered
PNG hole error by at least1%, with every old RAW/PNG macro,geometry and challenge
condition hole/boundary gate nonworse. No cumulative original-model score can
replace the incremental R1 requirement. Preserve all stopped/failing states.

This tests a practical objective package, changing resolution/target alignment
and additive-boundary weighting together. It does not isolate one sole cause or
promise useful anatomy. Smoothing risk, saturated-channel gradients, pretraining
overlap and limited adaptive TRAIN identities remain. Full covering-family user
acceptance and independent human final evaluation are still required.

Next: implement and independently audit a fresh finite runtime/transfer packet,
then provide manual L4 commands. Nothing should be uploaded or launched from this
design alone. Read research/completion_codeformer_delivered_objective_v4_design_v1/
root_design.json for exact pins, gates, resource limits and unrun status.
