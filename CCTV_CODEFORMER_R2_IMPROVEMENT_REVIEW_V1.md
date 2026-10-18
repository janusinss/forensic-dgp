# R2 improvement review — 18 October 2026

**The trained R2 model exists and is locally testable. Preservation still fails.**
Use [the testing guide](CCTV_CODEFORMER_R2_LOCAL_TEST_GUIDE.md) to compare it with
untouched CodeFormer. This review identifies the next question to investigate;
it prepares no training packet and changes no checkpoint, loss or quality gate.

The user subsequently reported: **“r2 is like a lot better with its higher
brightness its good.”** Useful brightening is an accepted preference to retain.
The comment does not identify particular input files or accept every preservation
case. It is positive subjective development feedback, not independent FINAL
evaluation. Do not turn this review into a requirement to remove all brightening.

## What the actual training established

R2 committed 50 Adam updates, not 50 epochs. Independent state inspection verified
350 changed trainable tensors and 165 unchanged frozen tensors. Every trainable
tensor has 50 recorded nonzero accumulated-gradient observations. The experiment
did train; an absent optimizer or an entirely disconnected network does not explain
the preservation failure.

The paired development comparison shows a specific tradeoff:

| Delivered PNG comparison | Parent | R2 | Interpretation |
| --- | ---: | ---: | --- |
| Clear cases exceeding the unchanged 0.03 regression limit | 3/61 | 7/61 | All three old failures remain; four additional cases fail |
| Normally blurred cases exceeding the coarse RGB limit | 14/53 | 29/53 | All fourteen old failures remain; fifteen additional cases fail |
| Normally blurred mean signed luminance error | +0.042793 | +0.056178 | Normal faces become brighter relative to paired clean references |
| Lowlight MAE / SSIM | 0.101779 / 0.614690 | 0.080872 / 0.627295 | Dark cohort improves on both means |
| Compound MAE / SSIM | 0.138071 / 0.520845 | 0.106888 / 0.538211 | Dark compound cohort improves on both means |

Clear PNG error worsens against the parent on 57/61 cases, including all three
paired source groups. Compound MAE improves on 53/53 cases and lowlight MAE on
48/53. RAW preservation also fails, so PNG quantization cannot explain the broad
tradeoff. SSIM alone would hide some clear-image changes.

These are paired synthetic development results. The 48 native CCTV crops remain
unpaired, source-separated evidence without clean-reference scores. Root's existing
34-sheet review and its limitations remain binding; this review performs no new
visual or independent FINAL evaluation.

## A supported training-path gap

The normal objective contains only original-teacher delivered-image L1 and token
KL. Its teacher target stores RGB and logits, with no teacher latent or style
statistics. Dark observations additionally receive clean token CE and latent MSE.

The unchanged hard token choice, detached encoder fusion inputs and candidate-only
AdaIN style detach matter here. The two saved before-Adam witnesses show image
losses reaching all 72 fusion tensors, with no connection to the 164 encoder or
114 predictor tensors. Token/HQ losses do reach the encoder and predictor.
Therefore normal image preservation cannot directly correct the shared encoder;
normal token KL remains an indirect constraint, while dark CE/latent training can
change that encoder. Detached feature values still affect the numerical forward.

The saved normal/dark witness fusion image-gradient norms are 1.251857 and
3.164641. Those are two source-bound VM readouts, not a population estimate or
proof that gradients oppose one another. Full gradient vectors were not exported.

Across the 100 saved normal TRAIN observations, candidate hard tokens disagree
with original-teacher argmax choices at 64.7422% of positions on average. These
are different inputs captured before their respective updates, not a fixed-input
learning curve or the final checkpoint's error rate. Teacher disagreement is
also not proof that a face is wrong: the untouched teacher is imperfect. It does
show that style alone is not isolated as the explanation; token/content behavior
must remain part of the investigation.

The architecture explicitly uses low-quality feature channel means and standard
deviations for AdaIN color/illumination. A normal feature/style preservation target
is therefore a defensible diagnostic direction. **Its absence is established;
its responsibility for the observed failure is still a hypothesis.** Token changes,
fusion changes, the already-biased parent initialization and limited training
coverage remain alternative or interacting causes.

## Next design decision

Keep R2 available for user testing while investigating whether normal appearance
preservation needs a direct encoder feature/style signal. The next finite manual
VM proposal should first distinguish this hypothesis from token/fusion drift,
using original/parent/R2 normal and dark TRAIN evidence. Existing saved returns
do not contain the normal teacher latent/style captures or component gradient
vectors needed to settle it. Do not invent that missing evidence.

Any proposed feature/statistic teacher term must demonstrate a finite nonzero
encoder gradient and unchanged inference geometry before optimization. Evaluate
actual normal and dark outputs together. Treat the parent initialization as a
separate factor; changing initialization and objective simultaneously would make
the result harder to interpret. No new route is selected or claimed successful by
this review, and more unchanged epochs or a larger learning rate are unsupported.

The next diagnosis must distinguish useful illumination correction from visible
feature drift. A teacher feature/style constraint could also suppress the user's
preferred brightening if applied indiscriminately, so it is not yet a selected
loss. Preserve unmodified RAW/PNG outputs; any illumination-normalized diagnostic
must be labeled separately and must not replace the existing acceptance scores.

The user's positive feedback on the currently testable model guides this priority. Preserve
all checkpoints, failure records, splits and sources. Retain current quality limits,
paired/unpaired separation and reserved FINAL identities. Actual training remains
manual on the L4 with a new finite reviewed protocol; the main app default and
deferred covering/completion scope stay as recorded in the goal.

## Evidence

- [Independent quality diagnosis](outputs/codeformer_role_preserving_training_v1_r2_return_review/design_review_v1/independent_quality_diagnosis_v1.json).
- [Independent TRAIN diagnosis and hash manifest](outputs/codeformer_role_preserving_training_v1_r2_return_review/design_review_v1/diagnosis_v1.json).
- [Independent state audit](outputs/codeformer_role_preserving_training_v1_r2_return_review/state_v1/state_audit_v1.json)
  and [saved-pixel audit](outputs/codeformer_role_preserving_training_v1_r2_return_review/pixels_v1/audit.json).
- [Actual return review and all-sheet limits](CCTV_CODEFORMER_ROLE_PRESERVING_TRAINING_V1_R2_RETURN_REVIEW.md).
- Local source: `research/codeformer_role_preserving_training_v1_r2/model.py`,
  `worker.py`, `research/codeformer_adain_gradient_isolation_v1/gradient_route.py`
  and `third_party/codeformer/codeformer_arch.py`. Root source/hash readback is
  retained beside the independent design receipts.
