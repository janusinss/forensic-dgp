# CodeFormer normal-tone V1: prospective objective test

17 October 2026. **DESIGN / NOT RUN / NO NEW MODEL CHECKPOINT.**

The immediate aim is a trained CodeFormer adaptation with clearer degraded
faces and preserved visible appearance. Original CodeFormer remains the
initializer and comparison baseline. This test answers one remaining loss
question before the next checkpoint-producing training pilot; it does not
replace that pilot or establish a useful new model.

## Evidence and decision

The [completed loss-response audit](CCTV_CODEFORMER_LOSS_RESPONSE_V1_RETURN_REVIEW.md)
retains useful dark-image gains alongside normal-image regressions. Removing
the entire coarse RGB term improved some normal errors but weakened dark
benefit. That experiment removed both groups' contributions, so it did not
identify normal supervision as the cause. The
[independent next-design review](outputs/codeformer_loss_response_v1_return_review/next_design_independent_review_v1.md)
requires a falsifiable objective test before selecting another training recipe.

The new question is: with dark coarse supervision unchanged, can a normal-only
luminance-offset invariant coarse term protect normal appearance better than
the full objective and omitting normal coarse supervision entirely? Normal
coarse supervision may be protective; the omission control can disprove the
proposed explanation. All results, including contrary results, remain evidence.

## Fixed mathematical intervention

Keep the existing native512 unclipped RGB supervision. Let `P` be average
pooling with kernel and stride32, giving a16x16 grid. For each normal TRAIN
observation independently, define:

```text
e = P((native_prediction + 1) / 2) - P(clean_RGB512)
muY = mean_spatial(0.2126*e_R + 0.7152*e_G + 0.0722*e_B)
centered_normal_coarse = mean(abs(e - muY))
```

The scalar mean remains differentiable. Subtract the same scalar from each RGB
channel, not three separate channel means. This removes sensitivity to a
uniform additive RGB offset while retaining global chroma and spatial errors.
It is not an exposure-recovery parameter or an additive decomposition of L1.
No output clipping, delivered-PNG centering, objective-weight change or new
inference brightness gate is introduced. Coarse weight4 and accumulation
divisor16 remain fixed. Dark observations retain the original RGB coarse L1.

Keep CE, HQ latent, pixel RGB, LPIPS, behavior RGB and behavior KL contributions
unchanged. Preserve isolated AdaIN style routing, the frozen decoder/codebook,
the two fixed Adam group rates1e-5/2e-5, betas0.9/0.99 and epsilon1e-8. These
rates are measurement controls, not a continuation schedule. The known TRAIN
augmentation role selects supervision only; it is never a deployment input.

## Two states and five conditions

At original0 and stopped50, restore the exact saved model, Adam and RNG before
each condition. Use the same ordered16 TRAIN observations from updates1,6,15,24
and slots1–4: eight references, eight normal and eight dark observations.

1. Baseline, with no optimizer action.
2. Full existing objective.
3. Omit normal-only coarse, retaining all dark coarse.
4. Replace normal-only coarse with the formula above, retaining all dark coarse.
5. Explicit-zero-current-gradient momentum control.

There are eight real disposable Adam steps; proposals never chain. There are
zero committed training updates, no exported candidate checkpoint and no
automatic follow-on training. Saving these one-step weights would not establish
a useful model. The failed previous full61-clear/53-blur preservation guards
remain binding and are not rerun or replaced by this same-source study.

## Finite computation and reuse

Reuse the six unchanged component-gradient bases from the hash-bound completed
loss-response return, not a newly claimed full backward calculation. Recompute
all16 original coarse gradients and the eight normal-centered gradients per
state. Accumulate original-all, normal-only, dark-only, centered-normal and
hybrid coarse bases directly in canonical source order, with FP32 weighting
and division by16. Never derive dark-only by subtracting rounded aggregates.

Fresh original-all coarse must exactly match its cached control before any
treatment step. Full and momentum controls must reproduce their retained
assigned-gradient, model/Adam and saved-output identities. A discrepancy stops
the diagnostic and remains a cache/control-parity failure; no tolerance fallback,
package upgrade or automatic retry follows. Cached full-gradient proof remains
inherited source-bound evidence, not a new full-autograd check.

The finite new-call ledger is32 source forwards,160 probe forwards and32 repeat
forwards:224 CodeFormer forwards total. There are32 original coarse and16
centered-normal autograd queries:48 queries total. New teacher, LPIPS, benchmark,
full-gradient and backward calls are zero. Every attempted call and actual
completed disposable step is recorded separately.

Two explicit read-only parent roots are required: the AdaIN parent supplies
original assets/states/data; the completed loss-response parent supplies cached
components, source witnesses and full/momentum controls. The new upload contains
source, protocol, cases and provenance only. No weights or research caches are
duplicated or removed. Full hash verification precedes model imports and calls.

## Saved evidence and limits

Save all32 source witnesses,160 probe outputs and32 repeat arrays. Retain native
normalized512, RAW256 and exact floor-PNG outputs separately, with token and
source logit/feature witnesses. Export all ten72-array coarse bases and the
eight72-array actual fusion-parameter delta sets, plus complete assigned-gradient
and model/Adam/RNG digests. Full fusion deltas do not claim exported full
encoder/predictor deltas or bit-exact independent CUDA replay.

Record actual package/runtime versions, each canonical input/clean/noise-field
parity check, peak RSS, peak allocated/reserved CUDA and minimum sampled free
disk on the worker-root volume, including terminal/failure measurements. HOME
archive writes have separate reserve guards; disk minima are not continuous
measurements. Guard initial free disk10GiB with a
4GiB reserve, RSS8GiB, CUDA reserved22GiB, result payload2GiB, archive2.25GiB,
metadata128MiB and5,000 regular files. These are prospective caps. The conservative
planned output calculation is below2GiB; actual feasibility is unmeasured.

Worker total1,800 seconds, initialization300, source600 and probe900 seconds;
preflight300 and export600 seconds. The supervisor enforces finite external
limits and logs, preserves terminal failures, and exports without relaunching.
The new diagnostic may only be manually launched on the existing NVIDIA L4
under tmux. No local neural, gradient or optimizer execution is authorized.

## Review and checkpoint-producing follow-up

Independently audit every saved input, source/control identity, assigned basis,
RAW/PNG conversion and fusion delta. Compare normal/dark and A/B strata with
baseline, full and momentum, retaining every regression. Inspect every planned
exact256 visual sheet for visible eyes, brows, nose, mouth, jaw, glasses, color,
contrast and texture. Do not select a recipe from a favorable average alone.

If returned evidence supports the intervention, prepare one separate finite
training pilot that writes actual candidate/stopped checkpoints and runs the
full development/preservation reviews. Do not append epochs to the failed AdaIN
run. If the intervention fails, use its falsifier to reassess the learning path;
do not repeat it unchanged, enlarge a search grid or weaken the requirements.

Native CCTV remains unpaired and source-separated; this photographic TRAIN
test is neither DEV/FINAL nor evidence of Zamboanga performance. FINAL identities
remain untouched. User acceptance and independent final review remain required.
The later app keeps Auto/override, input-only refusal for insufficient visible
structure, PNG downloads and the existing design. Detection/completion development
stays deferred; retain its later seven covering families and separate automatic
versus assisted review. The full goal remains incomplete.
