# Post nearby-gradient learning review — 14 October 2026

**This learning recipe is closed and failed. Stop training-code changes for
the required diagnostic discussion.** The audit and direct image review are
complete; no accepted update, replacement checkpoint or epoch continuation
exists. The original DGP remains protected and the full goal is active.
[Measured result and exact failures](<C:/xampp/htdocs/YEAR 4/Testing/CCTV_DGP_NEARBY_GRADIENT_LEARNING_V1_R1_RESULTS.md>).

The latest best gain is 0.687495% RAW / 0.706917% PNG, with 32 preservation
failures per representation. Two smaller proposals also fail. All 96 sampled
local dot constraints pass, yet the actual finite output gains and appearance
do not. The direction remains almost parallel to the failed RAW-only direction
(cosine 0.999539308). The invalid assumption is that collecting these nearby
derivatives would yield a sufficiently different, preserving update.

The original question, assumptions and failed experiments remain binding.
The same 100 paired TRAIN exposures and targets cannot establish native CCTV
or independent final performance. No new automatic selection or blending
rule may be used merely to hide failed face preservation; future selection
requires its own input-only development evidence.

## Required diagnostic question

[Workspace rule](<C:/xampp/htdocs/YEAR 4/Testing/AGENTS.md>) requires: “If an issue remains unresolved
after 3 consecutive attempts, STOP modifying code. Name the invalid assumption
and ask one diagnostic question.” The three related failed recipes are joint
RAW/PNG R4, RAW-only V1 R1 and nearby-gradient V1 R1. The human's earlier “apply
best approach” responses authorized earlier reviews; the new three-attempt
discussion must still be asked and its actual answer retained.

The question sent is:

> The joint, RAW-only and nearby-gradient recipes all failed preservation. Before another VM pilot, which cause should the design review investigate first? Both directions retain every visible feature and the original quality gates.

Recommended discussion direction: review the reconstruction path and target
detail. Alternative: continue investigating loss gradients and PNG conversion.
The human answer is currently pending. This is a diagnostic discussion, not
approval of a concrete new training recipe or model. No fourth recipe is
prepared while it is pending. Auditing, preserving and publishing the already
returned result are completed independently of that choice.
[Bound question record](<C:/xampp/htdocs/YEAR 4/Testing/outputs/cctv_dgp_nearby_gradient_learning_v1_r1_analysis/diagnostic_question.json>).

## Evidence a subsequent design must resolve

1. Establish whether the current learned path can represent useful visible
   detail while preserving appearance, using retained architecture and saved
   signal/target evidence; do not infer this solely from nonzero gradients.
2. Audit genuine target resolution and visible detail across both existing
   sources before claiming more data is required. Acquire only for a measured
   gap with terms, provenance and overlap checks.
3. Compare against prior spatial/decoder failures as well as the recent bank
   conditioning route, so a new name does not repeat a failed recipe.
4. Keep all original actual RAW/PNG gates, brightness/distance limits, original
   checkpoint, failure records and TRAIN/DEV/final separation unchanged.
5. Freeze a materially justified finite design with source/component origins,
   meaningful regressions, resource/export limits and independent audit before
   issuing human-manual L4 commands.

The gradient-sampling paper motivates nearby derivatives and finite line
search under its assumptions. It does not prove our deterministic face-loss
recipe or hidden identity. Its primary text was refreshed for this review.
[Burke, Lewis and Overton](https://cims.nyu.edu/~overton/papers/pdffiles/gradsamp.pdf).
No universal derivative error, impossible target, or unique architectural
bottleneck has been proved by the latest finite failure.

No app, model, loss, mask, checkpoint or threshold was changed by this review.
Preserve visible structure and appearance, existing app design and insufficient-
information handling. Native unpaired/synthetic paired evidence, untouched
final identities, all five milestones and all seven automatic/assisted
covering families remain. Training and neural parameter changes stay manual
in tmux on the existing L4; agent VM access remains maintenance only. Goal
incomplete; no automatic historical retry or new epoch launch is authorized.
