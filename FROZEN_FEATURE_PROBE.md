# Frozen-feature detector feasibility — 29 September 2026

Hypothesis: generic pretrained image features can distinguish face coverings when
paired with a learned occlusion-specific head, without the point/box ambiguity
observed in SAM2 proposal refinement. This is one bounded training-only diagnostic.

## Fixed protocol

- Same pinned SAM2.1 tiny weights/revision as `SAM2_DETECTOR_BENCHMARK.md`, frozen
  encoder, public `get_image_embedding()` interface,256x64x64 embeddings.
- All68 V2 training crops (42 covered,26 uncovered),256px input; no validation/test
  fitting. Manifest loader verifies all dataset hashes/split integrity, but only
  training records enter feature extraction, normalization or optimization.
- `feature_detector.py`: per-position channel layer normalization, pointwise
  256->64->1 head with GELU, bilinear logits to256px. No target-derived prompts.
  Feature inputs are detached; no encoder gradients. No RGB shortcut or generator.
- Seed42,20 epochs,11 balanced batches of8 per epoch, AdamW lr0.001,
  weight decay0.0001, BCE+softDice, gradient clipping1.0, threshold0.5.
- Preliminary final-epoch training-fit criteria: IoU>=0.80, at most1 empty covered
  case and at most2 false-mask uncovered cases. These criteria decide feasibility,
  not selection; original real/synthetic deployment safeguards remain unchanged.

Three focused tests passed after the initial expected missing-module failure:
output shape, gradient isolation, wrong-channel rejection and learnability of a
synthetic separable-feature fixture. Fixture success is not real-mask evidence.

## Execution and next decision

Completed locally on CPU with four torch threads. Windows sandbox dependency reads
required escalation; no environment or application dependency changes. The script
creates a fresh output directory and must not be relaunched while the process is
live. It cached training embeddings and trained for exactly220 optimizer steps.
Final checkpoint has a distinct diagnostic format that the completion loader does
not accept. Application detector, generator and restoration baseline remain intact.

Local root `c:\xampp\htdocs\YEAR 4\Testing\`; VM equivalent `~/forensic-dgp/`.
Artifacts currently local only: `outputs/frozen_feature_probe/protocol.json`,
`train_features.pt`, progress/final `results.json`, final `head_epoch_20.pth` and
68 predicted masks. Runner: `outputs/run_frozen_feature_probe.py`.

## Results: training-fit criteria failed

All68 saved masks were independently scored and reproduced the final metrics
exactly. Fixed preview: first7 covered and first3 uncovered records in manifest
order, `outputs/frozen_feature_probe/training_preview.jpg`.

| Training metric | Final epoch20 |
|---|---:|
| IoU | 0.7340957 |
| Missed covered pixels | 24.949% |
| Visible-pixel false positives | 0.3217% |
| Empty covered cases | 0/42 |
| Uncovered cases with false masks | 3/26 |

Both IoU and negative-case criteria fail. No checkpoint was selected or deployed.
Epoch17 IoU was0.85736 with9 negative false-mask cases; final loss decreased while
recall deteriorated. This is evidence of an optimization tradeoff, not proof of
encoder inadequacy. Visual inspection found partial mask bodies and holes in
patterned/respirator regions, consistent with the missed-pixel metric.

Next: decompose BCE and softDice behavior for covered versus uncovered training
images to explain the late coverage decline before changing the recipe. No
validation sweep, extra epochs or checkpoint cherry-picking. Validation and
synthetic retention have not been evaluated for this failed feasibility candidate.

## Loss audit and matched empty-target Dice comparison

Audited final weights without updates; tensor equality verified. At offset0,
covered BCE0.25788/Dice0.18626 versus uncovered BCE0.000736/Dice0.38408.
On six balanced batches, covered and uncovered parameter-gradient cosines were
negative in6/6 (approximately-0.667 to-0.858); median uncovered/covered gradient
norm ratio0.8214. Empty-target Dice dominates uncovered BCE gradients. Training-only
logit offsets were diagnostic; none was selected for inference. This supports a
loss-conflict hypothesis, not a guarantee that removing Dice will control false
positives. Evidence: `outputs/frozen_feature_probe/loss_audit.json`.

Launched one matched comparison using the existing cached training features:
`outputs/compare_feature_empty_dice.py`, outputs `outputs/feature_empty_dice_comparison/`.
Both arms have identical saved initial head weights (hash recorded), batch order,
20epochs/220updates, AdamW settings and final training-fit criteria. Control keeps
all-image Dice; treatment zeros only empty-target Dice terms before the batch mean,
preserving covered Dice scale and all-image BCE. New shared seed42 initialization
is explicitly different from the earlier SAM-construction-dependent RNG state;
use the new matched control, not the earlier run, for causal comparison.
No validation/test fitting or threshold changes. Both arms completed successfully.

| Final training metric | All-image Dice control | Nonempty Dice treatment |
|---|---:|---:|
| IoU | 0.79353 | 0.86296 |
| Missed covered pixels | 17.995% | 8.220% |
| Visible-pixel false positives | 0.4809% | 0.9142% |
| Empty covered cases | 0/42 | 0/42 |
| Uncovered false-mask cases | 4/26 | 10/26 |

Both fail the fixed training-fit criteria. Removing empty-target Dice improves
coverage but worsens false masks, confirming a tradeoff rather than a complete
fix. Loss values across arms are not directly comparable because a term is removed.
No validation pass, deployment or additional training is authorized by these
results alone. Next diagnostic: inspect the location, area and confidence of
uncovered-face errors to distinguish local segmentation noise from image-level
occlusion confusion before choosing any further model change. Do not repeat loss
weight sweeps on this small training set.

## Uncovered-face error audit

Scored every26 uncovered training image at unchanged threshold0.5 for both saved
heads. Confirmed4 control and10 treatment false-mask cases. Treatment predicted
areas range1–737 of65536 pixels (at most1.125% per affected image), with maximum
probabilities0.517–0.925. Control false regions range28–320 pixels, max confidence
up to0.947. Small area does not imply low confidence or harmless output edits.

Inspected all10 treatment failures in `negative_audit.jpg` (input, control overlay,
treatment overlay). Regions occur on eyeglass reflections, teeth/chin, clothing and
isolated background pixels. Existing annotations classify these as uncovered;
they remain false positives under the current protocol. No labels or thresholds
were changed. Artifacts: `outputs/feature_empty_dice_comparison/negative_audit.json`
and `negative_audit.jpg`; script `outputs/audit_feature_negatives.py`.

Asked the user whether clear eyeglasses with strong glare should remain untouched
or count as occlusions requiring completion. This is a task-definition question,
not evidence that the present labels are erroneous. Until clarified, preserve
current labels and gates; do not relabel reflections to make the candidate pass.
Any alternate task definition needs a separately versioned annotation policy and
full consistent review, rather than edits only to failed examples. A size filter
would also risk dropping genuinely small occlusions and is not being introduced.
