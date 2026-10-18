# After FP32: separate useful exposure adaptation from normal-face drift

**DESIGN REVIEW ONLY — 16 October 2026.** No new worker, protocol, execution
archive or actual training has been prepared by this review. The returned FP32
comparison is quality-stopped at 50 despite real learning and correct initial
FP32 behavior. The user selected the best supported approach after the required
three-attempt discussion: “great news, apply the best approach.”

[Actual evidence and all comparisons](CCTV_CODEFORMER_FP32_COMPARISON_V1_RETURN_REVIEW.md).

## What the next design must answer

Can input-conditioned exposure/fusion adaptation keep the useful blurred/dark
changes while preserving normal-face tone, contrast and visible features?
The current shared encoder/fusion update does not do this reliably. Prior
fixed-code fusion interventions improved four sampled normal cases but worsened
all four dark cases. Those are historical TRAIN measurements on V2, not a
guarantee about FP32 or a different architecture. Their scope rules out treating
blanket original-fusion rollback as an established solution.

The immediate direction is **conditional, bounded adaptation**. It does not
replace CodeFormer with an unrelated all-in-one model, claim an exact hidden
identity, or begin completion development. Untouched CodeFormer remains the
initializer and benchmark. All stopped states, splits and gates stay retained.

## First controlled experiment to specify

Review an original-base fusion block plus a bounded trainable delta. At each
existing fusion site, the frozen original block and its trainable copy see the
same current encoder/decoder features. Their difference is zero at initialization.
The proposed output is original fusion plus a smooth, bounded delta times an
input-conditioned gate. Original fusion weights are excluded from Adam; the
trainable copy still allows helpful lowlight adaptation. A feature bound does
not guarantee a bound on final pixels.

Use two separate finite arms to isolate conditioning from the common bound:

1. A bounded shared control with a constant gate.
2. The same bounded adaptation with input-conditioned gates.

Both start from untouched CodeFormer. Retain the same first 200 TRAIN observations,
exact original behavior targets, first 50 rates, Adam, losses, current encoder/token
trainable paths, frozen decoder/codebook, FP32, w=1/AdaIN, delivered output policy
and RAW/PNG guards. Each arm has at most 50 updates; no historical failed state
is resumed. The existing unbounded FP32 result is a historical comparator, not
a substitute for the new bounded control. The two arms are one controlled
comparison; a failed arm remains a failed arm.

The independent [design addendum](outputs/codeformer_fp32_comparison_v1_return_review/design_recommendation_v1.md)
contains a candidate formulation and its risks. Its beta/MLP values are proposals,
not accepted hyperparameters or verified executable instructions. In particular,
a learned gate starts with zero gradient when the fusion delta is exactly zero;
the inherited small LR may yield negligible conditioning within 50 steps.
Do not spend another run silently assuming that gate will learn useful routing.
An input-only fixed gate would be a different declared treatment and also needs
review: brightness statistics mix lighting, background and visible appearance.
During both training forwards and inference, the gate must use only the observed
input. It must never consume clean targets, simulator exposure labels, source/
reference IDs, DEV/FINAL labels or user annotations. Supervised losses may use
the declared training targets; the gate inputs may not. Do not equate dark pixels
with underexposure or infer ethnicity.

The current encoder/AdaIN route can still change appearance even with bounded
fusion. Keep that measurable in this first isolated comparison. Freezing that
route as well, adding a low-frequency exposure head, changing all losses or
raising LR would introduce separate factors. A learned exposure head is a
reasonable later architecture option if the evidence supports it; it is not
already implemented or proven here.

## Delivered images and preservation

Only 50/200 old observations have behavior losses, all normal-B controls. That
coverage gap remains a separate candidate cause. The existing behavior RGB
loss already uses delivered 256 pixels and exact FP32 teacher arrays. Other
clean image/photometry losses see unclipped native 512 output, while delivery
clamps, bilinearly resizes and floors PNGs; direct clean targets use LANCZOS256.
No target swap or hidden display processing was found.

Every new design must declare these paths and measure actual 256 RAW/PNG
preservation. Do not silently claim a new delivery loss is present in the
conditioning experiment. Broader normal-image behavior coverage or a separate
delivered 256 supervision comparison should be isolated if still necessary.
Useful dark brightening is allowed; anchoring every dark input to the old dark
output would oppose that goal. Loss scalar magnitude is not gradient dominance.

## Readiness work before manual launch

Resolve and audit the actual gate parameterization, initialization gradients,
numerical bounds and input dependence before sealing any packet. Recheck zero-
delta output/behavior parity, all original frozen states and no-target-leakage.
The live manual L4 preflight must measure actual gradient/update paths under
finite calls; CPU tests cannot establish learned-gate effectiveness.

Derive exact forward/backward/teacher/canary/LPIPS counts, checkpoint/temporary/
archive sizes, disk reserve and hard time/memory caps from the final source.
Reusing audited static targets requires exact source/state/input hash bindings.
Two fully measured arms imply at most 100 Adam updates,400 training exposures
and 1,624 snapshot outputs before extra explicitly budgeted canaries. More fusion
computation and larger states require a new ledger; old cap values are not
automatically reusable. No archive or verified upload/launch command exists yet.

After return, independently compare both arms with original CodeFormer and
retained FP32/BF16 on all identical 406 DEV cases, keep the 24+24 native sources
unpaired, inspect every planned sheet and retain every failure. Both unchanged
preservation checks and useful visible improvement are required before extending
training. A varying gate, lower loss, passing finite endpoint or export alone
does not qualify a replacement. FINAL remains untouched until independently
reviewed selection is fixed. User satisfaction remains required.

Restoration continues first. Detector/completion development, application
adoption and full Auto/override/download flow remain later goal milestones.
This review authorizes neither automatic training nor the unchanged failed
recipe. All actual training remains the user's manual verified L4/tmux workflow.
