# Filter-path isolation — 18 October 2026

The bounded comparison is complete. **The learned fidelity branches retain
part of the selected row, block and local-blur artifacts while also preserving
useful visible detail.** Bypassing them suppresses those artifacts but changes
glasses, mouths, expression and accessories. Disabling AdaIN alone leaves key
defects. Neither global switch is an accepted restoration improvement.

The next design to prepare is a **finite fusion-focused filter adaptation from
R2**, initially holding the encoder, token predictor and shared generative prior
fixed. It needs broader audited TRAIN corruption coverage and clear/dark
preservation checks. Useful brightening remains desired. No training or app
replacement occurred in this diagnostic.

## Actual execution and boundaries

One local CPU FP32 run exited 0 after **251.79 seconds** (4.20 minutes), making
**48 new top-level CodeFormer calls** and writing **150 files / 43,529,528 bytes**.
There were **zero optimizer updates or backward calls**. The independent saved
RAW/PNG, journal and lineage audit passed once. All original files and all
515 before/after tensor fingerprints stayed exact for each model.

Five probes were explicitly selected after the previous 72-case review: one
clear input, repeated horizontal rows, local Gaussian blur, local mosaic and
global mosaic. These are five observations from **three photographs**, with two
historical TRAIN observations and three DEV observations. Selection is purposive
and exploratory, frozen before the new settings, not unseen validation. DEV and
TRAIN admission remain unchanged. This does not replace the earlier all72,
core63, flagged9 or 24 insufficient-control evidence.

All three retained models were tested: untouched CodeFormer, the full-coarse
parent and our R2 export. R2 still means **50 new updates from the already
adapted parent**, not 50 epochs or its lifetime training total.

| Fidelity branches | AdaIN | Evidence |
| --- | --- | --- |
| On (`w=1`) | On | 15 cached, previously audited baseline cells |
| Off (`w=0`) | On | 15 new cells |
| Off (`w=0`) | Off | 15 new cells |
| On (`w=1`) | Off | 15 new cells |

An additional fresh clear baseline ran once per model before its interventions.
All **three** matched the earlier RAW values and PNG pixels exactly, including
the read-only instrumentation. Thus 45 new factorial outputs plus three parity
checks equal 48 new calls; all **60 factorial cells** remain in the comparison.
Parity outputs are not counted twice in quality measurements.

Input geometry, normalization, native clamp, bilinear512/256 and floored PNG
delivery match the [previous comparison](CCTV_CODEFORMER_FILTER_COMPARISON_V1_REVIEW.md).
No alignment, enhancement, exposure correction, completion, per-case output
selection or automatic routing was added. The user's five wild images remain
separate unpaired stress tests. No native CCTV or reserved FINAL input entered
this paired diagnostic.

## What was isolated

Read-only observations counted **48 top-level calls** and **72 fusion-module
calls** (18 at each of the four sites). `w=0` bypassed all four sites; `w=1`
enabled them. These are selected module counts, not a profiler of every layer.

Within each input/model, the three new settings returned exact matching logits,
LQ feature hashes and recomputed tokens. Only the clear inputs also have fresh
fourth-setting internal readouts; historical nonclear baseline internals were
not captured. Tokens were recomputed with the implementation's FP32
softmax/top-k rule, not directly captured from the codebook lookup. All 48
readouts had zero exact top-two probability ties. This is technical evidence,
not calibrated face confidence or identity assurance.

The **165 generator/codebook tensors** (164 generator and one embedding tensor)
also produced the same source-bound canonical digest in all three models:
`6e08299da2297f2953557ba8e91debb02189edb843fbf80ec736ea437aa6a8c4`.
The saved-output auditor separately streamed the full three model-file hashes;
it did not deserialize tensors or independently replay the neural network.

At fixed model/input and fixed AdaIN, enabling the learned fidelity branches
retains more of the selected corruptions. This is an observed conditional
inference effect. All four sites act within a nonlinear decoder; it is not a
decomposition into four independent additive errors or proof of which earlier
training loss caused the regression. AdaIN transforms latent statistics and
is not simply an image-brightness switch.

## Measurements and visible tradeoff

The following are unweighted means over the **five selected R2 probes**, with
known clean256 references and delivered PNG pixels. They are diagnostic values,
not a new gate, independent identities or estimates of CCTV performance.

| R2 setting | RGB MAE | SSIM | Centered RGB MAE | Signed luminance bias |
| --- | ---: | ---: | ---: | ---: |
| `w=0`, AdaIN off | 0.078312 | 0.685604 | 0.059408 | +0.061130 |
| `w=0`, AdaIN on | 0.061455 | 0.688272 | 0.051108 | +0.040528 |
| `w=1`, AdaIN off | 0.034076 | 0.834380 | 0.031248 | +0.013241 |
| `w=1`, AdaIN on — retained baseline | 0.035847 | 0.835161 | 0.032044 | +0.017269 |

Whole-image error strongly favors retaining fusion, despite its visible
artifact retention. With AdaIN off, enabling fusion lowers R2 MAE, centered MAE
and adjacent-gradient disagreement on all five probes; mean SSIM increases by
0.148776. Switching AdaIN on at `w=1` adds mean luminance +0.004028 and raises
MAE on all five, while SSIM improves on four and adjacent-gradient error on two.
These mixed outcomes do not select a universally better switch.

The predeclared local-mosaic example exposes why whole-image averages alone are
insufficient. For R2 with AdaIN on:

| Region in `va_ffhq_00383__local_mosaic_8` | Fusion off | Fusion on |
| --- | ---: | ---: |
| Centered MAE inside corruption footprint | 0.042293 | 0.049089 |
| Adjacent residual-gradient error inside | 0.022423 | 0.024597 |
| Centered MAE outside footprint | 0.052881 | 0.028300 |
| RGB MAE inside footprint | 0.070882 | 0.056339 |

Fusion-off suppresses the visible central grid and improves these centered and
gradient diagnostics inside it, but increases absolute RGB disagreement and
changes clear context. The two local probes share one DEV photograph. Their
exact 128×128 corruption footprint is not an anatomical/identity mask or a
completion/removal area. Partial-region SSIM remains null; no invented regional
SSIM or oracle-corrected image is used.

The R2 tone shift persists even with **both fusion and AdaIN disabled**: PNG
luminance bias +0.061130 versus original CodeFormer −0.00460 in that condition.
R2/original recomputed code indices differ on 134–163 of 256 sites across these
five probes. With the shared decoder/codebook and both late conditioning paths
bypassed, AdaIN alone cannot explain that between-model output difference.
This does not identify its historical loss cause, quantify identity changes or
mean that the user's preferred brightening should be removed.

RAW and PNG remain separately reported. Signed factorial pixel interactions
were computed without clipping and kept distinct from differences of scalar
metrics. The previous frozen metric conventions are unchanged; centered error,
token disagreement and adjacent gradients are not identity or acceptance scores.

## All five sheets reviewed

Root personally viewed all five 1536×916 sheets, containing **90 exact 256-pixel
panels**, with the input, clean reference and all four settings for every model.
A second observer wrote separate visual observations before consulting the
new scalar measurements; that observer also authored the inference worker.
This is development review, not an independent or blinded FINAL evaluation.

| Probe | Visual result |
| --- | --- |
| Clear appearance | Fusion-off changes glasses, mouth, hair and texture even when the source is already usable. |
| Repeated horizontal rows | Fusion-off suppresses stepping; parent/R2 fusion-on retain it with AdaIN off as well as on. |
| Local moderate blur | Fusion-off clears central haze but alters bow/headband and fine features; fusion-on preserves more context with a soft center. |
| Local mosaic | Fusion-off suppresses the grid; fusion-on retains central block planes. Sharper plausible features are not proof of exact recovery. |
| Global mosaic | Fusion-off reduces blocks while changing red glasses, expression and fine appearance; fusion-on retains appearance and artifacts together. |

[Root per-sheet observations](outputs/codeformer_filter_path_isolation_v1_preparation/root_visual_review_v1.json),
[second-observer observations](outputs/codeformer_filter_path_isolation_v1_independent_audit/second_observer_visual_review_v1.json)
and [all five exact sheets](outputs/codeformer_filter_path_isolation_v1_independent_audit/visual_manifest_v1.json)
retain the full comparison. No setting was chosen per input or promoted to the
app. All prior insufficient cases remain refused; no automatic threshold was
validated.

## Concrete next development step

1. Audit broader recoverable TRAIN variants for rows, blocks and localized blur, retaining original source terms and splits. Declare their parameters before generation; vary training footprint positions instead of learning only this fixed central rectangle.
2. Prepare an initially fusion-focused R2 pilot, freezing the encoder/token predictor and shared prior. Freezing them does not guarantee preserved final appearance or brightness; verify those outputs directly.
3. Balance paired spatial supervision with clear-face and existing dark-face preservation. Keep untouched CodeFormer, the parent and R2 as declared controls, retaining useful brightening and visible glasses, hair and features.
4. Produce a finite verified manual L4/tmux packet with measured resource/time limits, explicit updates/epochs and retained stop failures. Independently audit returned RAW/PNG and source-separated native DEV before any app promotion.

These steps change the demonstrated failure route and missing corruption
coverage; they do not append epochs to an unchanged failed recipe. No new
training packet or learned checkpoint is ready from this frozen diagnostic.
The research app and its default model remain unchanged. Completion development
stays deferred until restoration is accepted. The overall goal remains active.

## Retained verification

- [Root-observed execution](outputs/codeformer_filter_path_isolation_v1_preparation/root_actual_execution_v1.json): exit0, 48 calls, 150 files, zero training updates.
- [Independent saved-artifact audit](outputs/codeformer_filter_path_isolation_v1_independent_audit/audit_v1.json): 48 new RAW/PNG triples, 15 cached triples, three exact parity checks, 154 journal records, 101 saved resource samples and three fresh model-file hashes passed.
- [Quantitative summary](outputs/codeformer_filter_path_isolation_v1_independent_audit/quantitative_summary_v1.json): all model/settings, RAW/PNG, regions and conditional effects retained without an optimal-setting selection.
- [Frozen protocol](outputs/codeformer_filter_path_isolation_v1_preparation/protocol_v1.json) and [metric contract](outputs/codeformer_filter_path_isolation_v1_preparation/metric_contract_v1.json): fixed before new toggles; no gate changes or clean-target output routing.
- [Worker readiness](outputs/codeformer_filter_path_isolation_v1_preparation/worker_CPU_readiness_v1.json) and [independent auditor review](outputs/codeformer_filter_path_isolation_v1_preparation/auditor_source_independent_review_by_repair_v1.json): 12 worker and 21 auditor fixtures passed. Verification success is not model qualification.

Completed worker manifest SHA-256:
`aff85131e4dae26a0c27cab1e9903da6e52d9c648bf4ca18a13d26d9baf0ed99`.
Independent saved-artifact receipt SHA-256:
`75342f998f99eacc54bb36eb956033ba4593915cc28752def8316f672ba93ce2`.
Frozen protocol SHA-256:
`23772988d16e11f52351365e66ce1b6d8ef77694f873966db98e24512b0aa86c`.

No browser flow changed in this phase; earlier bundled Playwright results
remain separate functional evidence.
