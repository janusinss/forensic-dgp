# Frozen CodeFormer filter comparison — 18 October 2026

The controlled comparison is complete. Broad blur is visibly improved by all
three models, but repeated-line artifacts, mosaic blocks and localized central
blur remain. R2 adds the lightness the user prefers; it is not a consistently
better restoration model on these filter examples. Preserve that useful
brightening while investigating the spatial failures. No new training,
checkpoint replacement, automatic input threshold or app promotion occurred.

## What was actually run

Local CPU FP32 inference ran once and exited 0: **216 outputs in 1,258.56 seconds**
(20.98 minutes), **658 files / 194,318,386 bytes**, with **zero optimizer updates**.
The 96 fixed synthetic pairs were reviewed before model outputs: 28 readable,
44 borderline and 24 insufficient. The same 72 readable/borderline inputs went
through untouched CodeFormer, the retained full-coarse parent and our genuine
R2 trained state. All 24 insufficient inputs retain explicit blank model cells;
no result or model metric was produced for them. Basic resizing is a fourth,
zero-inference control.

These are **eight photographs**, four historical TRAIN and four DEV, each under
12 fixed profiles. They are not 96 independent people. All DEV remains excluded
from optimization, no case is currently admitted to new training, and the
391/53 historical split remains intact. The small selection is diagnostic, not
the complete future training dataset. Original-source attribution, terms and
overlap limitations remain in the [data preparation review](CCTV_CODEFORMER_FILTER_PAIRS_V1_PREPARATION.md).

The user's five `test_images/wild*.png` files have no unblurred originals and
remain separate, unknown-origin **unpaired visual stress tests**. They were not
used as clean targets, training data, these paired scores, native CCTV evidence
or reserved FINAL cases. This comparison does not establish Zamboanga or general
real-world performance.

## Exact model and image policy

| Arm | Retained state | File SHA-256 prefix |
| --- | --- | --- |
| Untouched CodeFormer | Official benchmark, `params_ema` | `1009e537e0c2a07d` |
| Full-coarse parent | R2 snapshot0000; exact previously audited parent50 | `53778f54589697a1` |
| Our trained R2 | Existing export; 50 new updates from the retained parent | `d4abda0883168100` |

Those R2 updates are not epochs or the model's lifetime training total. It began
from an already adapted parent and an originally pretrained CodeFormer. The
present frozen comparison adds no training at all.

Full file hashes, canonical 515-tensor hashes and paths are in
[model_binding_v1.json](outputs/codeformer_filter_comparison_v1_preparation/model_binding_v1.json).
All three model files and source files were freshly verified; before/after live
state readouts stayed exact. Those readouts are source-bound worker evidence,
not an independent neural replay. The saved-output auditors did not deserialize
checkpoints or run neural models; the earlier CPU model-binding audit did inspect
saved tensors without neural inference.

Every arm uses the current research-app path: identical RGB256 input,
float32/255, bilinear512, CodeFormer `w=1`, `detach_16=True`, AdaIN enabled,
clamp native output before bilinear256, and PNG `floor(float32_RAW*255)`.
No alignment, extra exposure adjustment, sharpening, completion or input-based
model routing was applied. All parameters stayed frozen in eval/no-grad mode.
All 216 saved PNGs exactly matched their saved RAW conversion.

## Paired synthetic measurements

These are unweighted means over all **72 inferred cases**, including the nine
cases from the separately flagged DEV appearance photograph. Lower error is
preferred; higher SSIM is preferred. Scores compare with known synthetic clean
references and are not recovered-identity scores.

| Delivered PNG | RGB MAE | SSIM | Mean-centered RGB MAE | Adjacent residual-gradient error |
| --- | ---: | ---: | ---: | ---: |
| Resize control | 0.028059 | 0.800253 | 0.028111 | 0.016755 |
| Untouched CodeFormer | 0.029859 | 0.797766 | 0.029426 | 0.022010 |
| Retained parent | 0.043596 | 0.761814 | 0.035445 | 0.021245 |
| Our trained R2 | 0.049231 | 0.779007 | 0.037801 | 0.020816 |

R2 versus the parent improves PNG SSIM on **53/72** and adjacent-gradient error
on **57/72**, while RGB MAE worsens on **66/72** and centered MAE on **55/72**.
Versus untouched CodeFormer, R2 has higher MAE and centered MAE on **72/72**;
SSIM improves on **28/72** and decreases on **44/72**, despite lower adjacent
residual-gradient error on **65/72**. RAW has the same all72 MAE/centered-MAE
direction; RAW SSIM improves on 29/72 and adjacent error on 66/72. Keep the RAW
and delivered-PNG counts distinct.

The core63 group retains the same tradeoff: original/R2 PNG MAE
**0.027986/0.048569**, SSIM **0.808353/0.786114**, adjacent error
**0.020858/0.019789**. On the 36 DEV cases, original/R2 PNG MAE is
**0.029531/0.047526** and SSIM **0.808481/0.789688**. The nine appearance-flag
cases remain separately reported and included in all72; they were not silently
removed to improve the outcome.

R2's mean signed PNG luminance bias is **+0.028663** versus the clean targets
(original **−0.000960**, parent **+0.022069**, normalized range 0–1). Brightening
is desired, so this is not automatically a defect. Subtracting one global RGB
offset still leaves R2's larger centered error on every case. That does not
isolate the cause: contrast, spatial illumination, color, texture and geometric
differences can remain. Centering is a numerical diagnostic; no centered or
color-corrected image was delivered. A lower adjacent-gradient error can coexist
with smoothing or visible blocks and does not establish preserved identity.

The [frozen metric contract](outputs/codeformer_filter_comparison_v1_preparation/metric_contract_v1.json)
defines the new adjacent-RGB metric, which differs from the earlier central
luminance-gradient metric. Exact float64 target normalization also differs from
older float32-target scalar audits. Do not compare their decimal values as the
same measurement or reinterpret them as the historical one-percent gate.

## Visual findings

Root personally inspected **all 12 sheets**, with an additional second observer
who authored the inference worker. This is a development review, not an
independent FINAL assessment. Exact 256-pixel saved cells were viewed from the
original-size 1280×2356 sheets; this is not a claim about physical monitor scale
or blinded evaluation. All models, inputs and references remain visible.

| Observed behavior | Sheets and conclusion |
| --- | --- |
| Already clear faces | 01: all models smooth or alter some fine appearance. R2 is lighter, without a consistent added structure advantage. Preserve clear glasses and visible styling. |
| Global/directional blur | 02–05: all arms make broad facial features sharper. Stronger blur still produces plausible but changed lips, teeth, hair or accessories. R2 mainly adds lightness/smoothing. |
| Repeated horizontal rows | 06: original generally removes stepping better. Parent and R2 retain more artifacts around eyes, nose, mouth and glasses. The limitation is already present in the parent. |
| Mosaic blocks | 07 and 11: global mosaic leaves grid/accessory/geometry departures; localized mosaic remains conspicuous across central features in all arms. R2 does not solve it. |
| Localized moderate blur | 09: central haze and transitions remain despite sharp surrounding hair/background. Extra brightness does not restore those boundaries. |

Sheets **08, 10 and 12** retain the 24 prospectively insufficient cases with
blank model columns. Clear surrounding hair or background did not qualify
usable facial structure. This is manual input-only refusal evidence, **not a
validated automatic threshold** and not an assertion that an unrun model failed.
The rectangular synthetic corruption footprint is not a covering/removal mask.

The flagged `va_ffhq_08025` visible lip/eye styling changes in all models and
remains a separate appearance challenge. Do not infer ordinary uncovered
anatomy, authorize its removal or claim hidden-identity recovery.

[All sheets and exact panel bindings](outputs/codeformer_filter_comparison_v1_independent_audit/visual_manifest_v1.json)
and [root's per-sheet observations](outputs/codeformer_filter_comparison_v1_preparation/root_visual_review_v1.json)
retain specific strengths and failures. The
[second-observer review](outputs/codeformer_filter_comparison_v1_independent_audit/independent_visual_review_v1.json)
discloses that observer's source-author role despite its historical filename.

## Next improvement direction

Prioritize **recoverable row, mosaic and localized-blur training coverage and
the spatial/fidelity path**, with clear-face and existing dark-face preservation
checked separately. Useful brightening must remain an explicit objective;
globally undoing it does not fix the observed blocks and haze. Adding epochs
to the unchanged recipe is not justified by this result.

The local CodeFormer implementation combines quantized prior features with
encoder-conditioned scale/shift residuals at `w>0`; AdaIN also uses encoder
statistics. Retained artifacts can therefore involve token prediction, style
statistics or the fidelity branches. **Their causal contribution is still a
hypothesis**, not established by these three `w=1` outputs. A small frozen
path-isolation comparison is the next design check before committing the next
finite, balanced filter-aware VM pilot. Keep its outputs tied to these fixed
inputs and the original/parent/R2 baselines. Do not train on DEV, unknown user
images, reserved FINAL or the severe refusal controls; do not replace the full
TRAIN cohort with only the four selected examples.

No new VM execution packet or training run was created in this comparison phase.
Actual training stays manual on the L4 through verified transfer/tmux commands,
with finite updates, timing/resource caps and preservation stops. Existing gate
failures remain authoritative. The research app stays experimental, the main
default is unchanged, completion development remains deferred, and the overall
restoration-first goal is active.

## Retained verification

- [Actual worker exit and final manifest](outputs/codeformer_filter_comparison_v1_preparation/root_actual_execution_v1.json): exit0, 216 forwards, no training.
- [Independent saved RAW/PNG audit](outputs/codeformer_filter_comparison_v1_independent_audit/audit_v1.json): all216 conversions, source/model joins and closed file namespace pass.
- [Independent strict journal audit](outputs/codeformer_filter_comparison_v1_journal_audit/journal_audit_v1.json): all655 ordered events, zero unfinished calls, all3 frozen model intervals and 221 saved resource samples pass. Sampled maximum RSS is 1,010,974,720 bytes; transient peaks are not independently profiled.
- [Full quantitative summary](outputs/codeformer_filter_comparison_v1_independent_audit/quantitative_summary_v1.json): all72/core63/flagged9, TRAIN/DEV, profiles and RAW/PNG remain separate.
- [Source readiness](outputs/codeformer_filter_comparison_v1_preparation/CPU_source_readiness_v1.json), [pixel-auditor review](outputs/codeformer_filter_comparison_v1_preparation/auditor_source_independent_review_v1.json) and [journal-auditor review](outputs/codeformer_filter_comparison_v1_preparation/journal_auditor_source_independent_review_v1.json): 10 worker, 14 metric and 6 journal fixtures passed. These are verification checks, not model-quality acceptance.

Worker manifest SHA-256:
`7b476134e4f0f7975eb4dd41385e1197b2b08f7094b970a25165b196c0dd7be8`.
Independent pixel receipt SHA-256:
`9f2ed9a1b0b249fa5626b56102422f9b4028a534dad346223896727cea03c17d`.
Independent journal receipt SHA-256:
`2d19d54c7671827591d59d3959bc85e29674647cef639e493ed9f7be1b045a68`.

No browser flow was changed in this phase; the earlier bundled Playwright
evidence remains a separate functional check, not new restoration qualification.
