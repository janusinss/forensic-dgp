# Matched region-weighting pilot — 1 October 2026

Status: locally verified and prepared for the existing Linux CUDA VM. The new
experiment has not trained or demonstrated improved model output. Actual
optimization is restricted to the VM.

Windows workspace: `C:\xampp\htdocs\YEAR 4\Testing\`.
VM workspace: `~/forensic-dgp/coverage_vm_bundle/`, inside the repository at
`~/forensic-dgp/`. Commands and downloads: `FACE_OCCLUSION_FOCUS_VM.md`.

## Evidence and hypothesis

The previous continuation is fully audited in
`FACE_OCCLUSION_CONTINUATION_RESULTS.md`. Epoch30 reaches real validation
IoU 0.81739 and synthetic IoU 0.92039. The synthetic parent is 0.97469; every
candidate fails the unchanged retention gate. On cached training replay,
epoch30 IoU is 0.93217, so poor fit also exists on seen examples. Degraded
irregular training/validation IoU is 0.84717/0.81573.

Only two real training examples have the accepted V3 reflection additions.
Their reflection recall is 59.48% over 2,031 pixels, while the single 491-pixel
validation reflection remains entirely missed. Whole-image mask scores hide
reflection errors when the same face also wears a large medical mask.

Hypothesis: equally weighting separate covered regions and penalizing spill
around their visible edges can improve small-region fit while preserving clear
controls. This is a bounded test of an auxiliary loss, not a proven best model.
The matched control distinguishes its effect from another 210 updates.

[Kervadec et al., Boundary loss for highly unbalanced segmentation](https://proceedings.mlr.press/v102/kervadec19a.html)
motivates complementing region-based losses when foreground and background are
imbalanced. That paper evaluates a contour-distance loss on medical datasets.
The region-balanced BCE below is a project-specific adaptation; it is **not**
their published boundary loss or evidence of effectiveness on faces.

## Fixed starts and recipe

| Setting | Both arms |
| --- | --- |
| Source | Verified pretrained detector `outputs/face_occlusion_continuation_vm/epoch_30.pth` |
| Model SHA256 | `9ba74a30719f01bfafd7ee9c060dc090c507c26eeeecb117b4cb342a8bf4df42` |
| Optimizer | Exact `outputs/face_occlusion_continuation_vm/final_optimizer.pth` |
| Optimizer SHA256 | `258a01cc4698668333e191d6d01bd2866bd1e847b1f9df84e1a33c8ec8204b00` |
| AdamW | Saved moments restored independently; encoder 1e-5, decoder/head 1e-4; decay 1e-4 |
| Data | Original 73 real training labels + 638 frozen training replay cases |
| Batch | 8: two each real-covered, real-clear, synthetic-covered, synthetic-clear |
| Order | Exact existing ten-epoch schedule, 21 batches per epoch, seed 42 |
| Budget | 10 additional epochs / 210 new optimizer updates per arm; 420 total |
| Candidates | Experiment epochs 5 and 10, global epochs 35 and 40 |
| Frozen state | Reference visible head, BatchNorm running statistics, original completion parent/generator |

This is the direct occlusion detector used by Track2 completion. It does not
train the completion generator or Track1 restoration model. Phase3
`checkpoints/dgp_zamboanga_final.pth` remains the restoration baseline; full
Phase5 identity training remains pending.

The existing supervised loss is BCE + Dice + 0.25 hard-visible BCE. The existing
synthetic consistency term has weight 1 against the original completion parent.
Each step logs supervised, consistency, auxiliary and total loss separately.

| Arm | Total loss |
| --- | --- |
| `control` | Existing supervised + existing synthetic consistency |
| `component_focus` | Same terms + 0.25 × region-balanced auxiliary BCE |

The auxiliary is also calculated and logged in the control, with zero weight.
Architecture, images, label unions, splits, sampling, rates, teacher and selection
are identical. No augmentation, new label or validation/test sample enters training.

## Auxiliary loss and provenance

For each covered target at 256×256, find its 8-connected foreground regions.
Each region receives equal mass. Half of that mass averages foreground pixel
BCE; half averages BCE on a three-pixel Chebyshev dilation ring restricted to
pixels outside the **entire** covered target. Overlapping visible rings accumulate
their contributions. A region with no visible ring uses all its mass on foreground.
The covered-image weight map sums to one. A fully clear image uses the mean
BCE of its highest-error `ceil(10% × pixel_count)` pixels. Average over the batch.
The original supervised and hard-visible terms still apply to every image.

Separate the previously accepted V3-minus-V2 reflection additions from the
remaining foreground in **training cases 15 and 46 only** before finding
components. This preserves every original target pixel and gives touching glare
its own region. It does not draw new boundaries or include validation glare.

| Training case | Reflection regions | Remaining original target |
| --- | --- | --- |
| 15, `uncovered_05.png` | 822 and 523 pixels | Empty |
| 46, `new_covered_48.png` | 686 pixels | 11,381-pixel mouth mask |

Case46 therefore assigns 0.25 positive weight to glare and 0.25 to the medical
mask, with 0.50 distributed across their visible rings. Without this separation,
the touching reflection and mouth mask form one connected region.

`outputs/face_occlusion_focus_bundle_v1/priority_regions.json` registers exact
flat reflection indices, source/mask hashes, original V2 mask hashes, V2/V3
manifest hashes and training split lineage. It contains only the two training
cases. `target_maps_v2.json` registers target bytes, float32 weight bytes and
region/ring counts for all 711 training targets. The VM recomputes every map
and requires exact registration agreement before optimization.

The earlier local `target_maps.json` was an unpackaged draft that treated touching
glare as part of the medical mask. It is preserved as preparation evidence and
is excluded from the new package. The V2 suffix refers to the new weighting
registration, not a dataset relabeling.

## Preflight, counters and stopping point

Linux CUDA is checked before any file, model or optimizer work. Three previous
immutable inventories and every listed input are reverified. New package hashes,
73/25 real train/validation membership, replay membership, 2/2/2/2 schedule,
source weights, exact optimizer settings/moments and all target maps must match.
The preflight restores saved optimizer state and performs one mixed CUDA forward;
it performs zero backward passes and zero updates. Model tensors must remain exact.

Full execution first reproduces all six epoch30 source metric domains and the
original common baseline. It then starts each arm from the same source tensors
and an independent copy of the saved optimizer. Counters have distinct origins:

| Candidate | New updates per arm | Lifetime model updates | Saved optimizer lifetime step |
| --- | ---: | ---: | ---: |
| Source30 | 0 | 630 | 420 |
| Global35 | 105 | 735 | 525 |
| Global40 | 210 | 840 | 630 |

The earlier source10-to30 continuation reset AdamW; the current pilot restores
that resulting step420 optimizer. There is no optimizer reset in this experiment.
Clip norm remains 1. Nonfinite loss/gradient, changed frozen state, unexpected
counter or provenance mismatch stops the run. Existing output/archive paths
are never overwritten. No automatic retry, extra epoch, resume or changed gate.

## Validation, exports and promotion

Keep probability threshold 0.5 and both original gates unchanged. The real gate
requires greater IoU than the best eligible real score and no worse original
visible false-positive, empty-mask or clear false-positive counts. The synthetic
gate requires at least original parent IoU and no worse missed fraction, visible
false-positive fraction, empty covered cases or clear false-positive cases.
Synthetic reference values are IoU 0.97468999, missed fraction 0.01647619,
visible false-positive fraction 0.00133278, at most one empty covered case and
zero clear false-positive cases.

Score the same 25 real validation, 400 synthetic validation and 73 real training
cases. Report human-only (24), mannequin (1) and reflection-containing validation
(1) separately. Training fit is diagnostic, never a selection substitute. No test
predictions are scored; previous development inspected some test sources.

Save `control/epoch_35.pth`, `control/epoch_40.pth` and equivalent
`component_focus/` states, initial source bytes and saved optimizer, per-arm
final optimizers, all 420 step logs and 2,915 binary masks. Each arm saves
`best_detector.pth` only when both unchanged gates pass. No eligible checkpoint
means that file is absent; this is an expected outcome, not cancellation.

The result archive is `face-occlusion-focus-results.tar.gz` at VM bundle root.
The runner includes its executed new code, specification, maps and inventory.
After return, independently check all source hashes, exact batch orders, separate
loss arithmetic, counters, restored optimizer binding, frozen tensors, mask
recounts and checkpoint-to-mask reproduction. Inspect the fixed ten-row
validation grid and reflection-only zoom. Compare both arms at the same update
counts before choosing an eligible detector for reviewed end-to-end completion.
No application or generator promotion occurs automatically.

Approximate assistant-drawn polygons, small reflection sample size, unresolved
external FFHQ pretraining overlap and repeated use of development validation
limit conclusions. Equal region weighting can amplify annotation artifacts or
small straps. These are hypotheses to assess after the matched run. Completion
of hidden features remains a plausible estimate; final output improvement
requires subsequent completion/restoration evaluation.

## Local verification

Seventeen fixture tests cover normalization, touching glare, visible spill,
clear controls, source/optimizer preservation, VM-only execution, matched order,
counter origins, training-only provenance and immutable package boundaries.
The read-only source check verified 92 optimizer states / 14,328,209 parameter
elements and all 711 maps. One CPU source forward gave supervised loss
0.03546993 and auxiliary loss 0.05711475; every model tensor stayed unchanged.
No optimizer was constructed in that source check and no local training update
was performed. This proves preparation integrity, not output quality or CUDA
execution of the new pilot.
