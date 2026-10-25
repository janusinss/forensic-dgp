# U96: readable localized-blur study

21 October 2026. Completed local inference: seven historical DEV photographs, three roles and two U96 conditions, giving 21 inputs and 42 verified outputs. There were 35 new forwards and seven exact reused clear-control outputs. This is a preprocessing comparison with unchanged weights, not a newly trained model.

## Decision

Retain U96 as the selected development model. The prospective matched-operator preparation did not demonstrate a consistent visible facial improvement. Local central PNG error improves on three photographs and worsens on four; mean central and protected-face errors worsen. Small center SSIM/gradient/centered-error improvements coexist with increased luminance bias. These mixed measures do not pass preservation or establish useful automatic preparation.

Keep surroundings preparation optional and bypassable. Do not deploy an always-on blur rule or continue sweeping masks/strengths on this same DEV cohort. The single unpaired wild5 response remains valid as a separate context-sensitivity observation, but neither it nor this study establishes an automatic rule for arbitrary crops. The existing app and its default model were not changed by this study.

## Prospective admission and operator

Forty inputs were prepared from eight term-audited historical DEV photographs: clear plus local/global Gaussian1.2/1.8. A fresh isolated reviewer saw only32 anonymous degraded inputs, without clean references or restored outputs. It labelled17 readable and15 borderline; none was insufficient or out of scope. All8 local1.2 inputs were readable; seven global1.2 counterparts were readable. The selection contract, fixed before reading decisions, selected those seven paired photographs. Sigma1.8 was reviewed for coverage only and never used for neural selection. Exclusions and all original decisions remain retained. The clear control is the exact retained clean input; historical source knowledge is disclosed, not called new blind admission.

Each selected photograph contributes clear, global1.2 and local1.2 roles. Local corruption replaces only the half-open central128 patch[64:192,64:192]. Gaussian generation and preparation both use SciPy float64 reflect/truncate3 at nominal sigma1.2, finite spatial radius4, followed by rint/clipRGB8. Preparation filters the observed input, never the clean target. Whole-face rectangles protect readable eyes, mouth, jaw and glasses, expanded by6pixels with an8pixel outside smoothstep transition. Every protected input RGB value is unchanged. The auditor also verifies that no positive-alpha filter footprint samples the already blurred patch.

These are known synthetic operator/mask interventions. Broad protection preserves some sharp context and an outside transition, so the result is not globally uniform blur or exact unknown-blur estimation. Clear/global prepared arms intentionally test harm; globally soft inputs would normally bypass. The sigma, kernel, guard and feather differ from the earlier stress study, so cross-study differences do not isolate one causal factor. Input-only agent review is not a calibrated threshold or independent human FINAL assessment.

## Paired delivered-PNG outcomes

| Role | Central MAE original → prepared ↓ | Central SSIM original → prepared ↑ | Protected MAE original → prepared ↓ | Central MAE better / worse |
|---|---:|---:|---:|---:|
| clear | 0.024885 → 0.025473 | 0.947686 → 0.944764 | 0.025330 → 0.026240 | 3 / 4 |
| global_gaussian_1p2 | 0.029082 → 0.029384 | 0.917324 → 0.916578 | 0.029987 → 0.030436 | 2 / 5 |
| local_gaussian_1p2 | 0.028068 → 0.028493 | 0.907916 → 0.911257 | 0.027416 → 0.028772 | 3 / 4 |

On localized inputs, protected-face MAE worsens on6/7 photographs. Whole-image MAE worsens on all21 inputs, although intentionally altered surroundings mean that whole-image error alone is not the facial outcome. Center128 equals the corruption footprint only in the local role; protected rectangles are approximate face regions, not anatomical/identity ground truth. Paired metrics compare against the actual clean target; they are not CCTV identity-accuracy estimates.

| Role | Central gradient original → prepared ↓ | Central centered MAE original → prepared ↓ | Central absolute luminance bias original → prepared ↓ |
|---|---:|---:|---:|
| clear | 0.014673 → 0.015309 | 0.018452 → 0.019122 | 0.016157 → 0.016314 |
| global_gaussian_1p2 | 0.018297 → 0.018467 | 0.021122 → 0.021273 | 0.018864 → 0.018798 |
| local_gaussian_1p2 | 0.018432 → 0.018332 | 0.021791 → 0.021497 | 0.016619 → 0.017708 |

## RAW output kept separate

RAW here is clipped/resized delivery float32 before floorRGB8 PNG conversion, not unclipped native network activations. Neither RAW nor PNG uses display enhancement. The source-bound CSV retains both formats and all four regions.

| Role | Central RAW MAE original → prepared ↓ | Protected RAW MAE original → prepared ↓ |
|---|---:|---:|
| clear | 0.025089 → 0.025760 | 0.025587 → 0.026573 |
| global_gaussian_1p2 | 0.029619 → 0.029984 | 0.030580 → 0.031086 |
| local_gaussian_1p2 | 0.028424 → 0.028933 | 0.027775 → 0.029211 |

## Per-photograph local outcome

| Photograph | Appearance flag | Central PNG MAE original | Prepared | Protected PNG MAE original | Prepared |
|---|---|---:|---:|---:|---:|
| va_ffhq_28544 | False | 0.025135 | 0.023900 | 0.024283 | 0.024279 |
| va_ffhq_48646 | False | 0.023711 | 0.022283 | 0.022579 | 0.022826 |
| va_ffhq_37228 | False | 0.039297 | 0.041186 | 0.035178 | 0.037346 |
| va_ffhq_28565 | False | 0.025363 | 0.026597 | 0.035452 | 0.037739 |
| va_ffhq_59814 | False | 0.024117 | 0.025773 | 0.026287 | 0.029420 |
| va_ffhq_40375 | False | 0.028004 | 0.027562 | 0.023565 | 0.023925 |
| va_ffhq_19791 | True | 0.030847 | 0.032153 | 0.024565 | 0.025871 |

The makeup/jewelry19791 source remains in the cohort and is flagged. Source-specific errors remain available rather than silently removing it. Seven repeated photographs, with correlated roles, do not establish population performance. Source/photo separation from project TRAIN does not prove human-identity separation or absence from CodeFormer pretraining. Terms are retained source snapshots, not newly verified web terms. No native CCTV, reserved FINAL, ethnicity or Zamboanga-performance claim follows from this FFHQ synthetic study.

## Visual review and integrity

Clear sheets01-02: A/B faces remain close, with modest smoothing/tone changes relative to the exact clean input. A visibly softens more outer hair/context. Both reconstructions simplify some teeth, glasses and makeup/jewelry details; no clear-control preservation pass is inferred from visual similarity.
Global sheets03-04: A/B generate closely similar visible eyes,nose,mouth and jaw. The stronger surroundings softening in A does not produce a consistent facial advantage. Teeth, eye contours and forehead jewelry can still differ from the paired target. This forced-global arm is a harm probe, not recommended routing.
Local sheet05: both results are sharper than the mild central corruption, but A/B facial differences are small and inconsistent. Sharp-context preservation differs most at the hair/background. Glasses/teeth need target-based checks; more apparent detail does not establish identity fidelity.
Local sheet06: A/B for59814/40375/19791 remain close; none shows a repeatable substantial improvement of the visible facial structure. Makeup/jewelry19791 remains flagged and is not silently excluded.
All6 sheets and42 generated cells were personally viewed at saved256 before the key or score readout. Input admission was completed prospectively by a separate isolated agent. No automatic mask/strength, native CCTV or independent FINAL evidence is established. These observations do not justify universal context softening or a model promotion.

Root personally inspected all6 coded sheets/42 generated cells before opening the condition key or numerical results. The key mapped A=prepared/B=original for all three roles. This is author/agent review with prior project context, not an independent blinded human study. The independent saved-only auditor checked42 response bodies, lengths, ZIP/RAW/PNG/provenance records, all40 candidate inputs,32 anonymous admissions and21 preparations. A separate NumPy/skimage readback checked336 metric rows and2856 scalar values, agreeing within1.365e-12. Counter receipts prove35 new U96 forwards; seven reused outputs are physically joined to their original provenance and never counted as new.

A dedicated batch server completed in268.3seconds and was stopped after its terminal client result. The shared interactive server was left alone. The finite plan capped the worker at1200seconds and each request at150seconds, with4GiB disk/384MiB RAM reserves and1GiB outputs. No weights, optimizer, gradients, training, VM state, app source or quality gates changed.

## Next model work

Stop treating context softening as the general quality fix. The separate learning-path audit found that U96 continuation optimizes cached encoder-feature losses; its old delivery helper detaches the image graph. A future image-learning route needs a connected delivery expression and complete U96 parity. The current original-base prototype must not silently replace the user-selected U96 initializer. A new bounded post-fusion route is a hypothesis; source preparation or gradients alone do not qualify a trained model.

[U96 learning-path review](<C:/xampp/htdocs/YEAR 4/Testing/CCTV_CODEFORMER_U96_LEARNING_PATH_REVIEW_V1.md>). A finite manual-L4 zero-optimizer preflight must first prove initial complete U96 parity, unchanged frozen539 tensors, finite delivered-image gradients and residual leverage. No training pilot or new model acceptance is authorized by this study result alone. Keep all predecessors/gates; any later learning comparison must retain clear controls and use the same source-bound inputs/preparation for incumbent and challenger.

## Evidence

[Frozen plan](<C:/xampp/htdocs/YEAR 4/Testing/outputs/codeformer_u96_mild_context_v1/comparison_plan_v1.json>), [Input admission](<C:/xampp/htdocs/YEAR 4/Testing/outputs/codeformer_u96_mild_context_v1/input_admission_v1.json>), [Saved-output audit](<C:/xampp/htdocs/YEAR 4/Testing/outputs/codeformer_u96_mild_context_v1/saved_output_audit_v1.json>), [Independent readback](<C:/xampp/htdocs/YEAR 4/Testing/outputs/codeformer_u96_mild_context_v1/independent_saved_readback_v1.json>), [RAW/PNG metrics](<C:/xampp/htdocs/YEAR 4/Testing/outputs/codeformer_u96_mild_context_v1/region_metrics_v1.csv>).

| Sheet | Generated outputs |
|---|---:|
| [01_clear.png](<C:/xampp/htdocs/YEAR 4/Testing/outputs/codeformer_u96_mild_context_v1/visuals/01_clear.png>) | 8 |
| [02_clear.png](<C:/xampp/htdocs/YEAR 4/Testing/outputs/codeformer_u96_mild_context_v1/visuals/02_clear.png>) | 6 |
| [03_global_gaussian_1p2.png](<C:/xampp/htdocs/YEAR 4/Testing/outputs/codeformer_u96_mild_context_v1/visuals/03_global_gaussian_1p2.png>) | 8 |
| [04_global_gaussian_1p2.png](<C:/xampp/htdocs/YEAR 4/Testing/outputs/codeformer_u96_mild_context_v1/visuals/04_global_gaussian_1p2.png>) | 6 |
| [05_local_gaussian_1p2.png](<C:/xampp/htdocs/YEAR 4/Testing/outputs/codeformer_u96_mild_context_v1/visuals/05_local_gaussian_1p2.png>) | 8 |
| [06_local_gaussian_1p2.png](<C:/xampp/htdocs/YEAR 4/Testing/outputs/codeformer_u96_mild_context_v1/visuals/06_local_gaussian_1p2.png>) | 6 |

The full restoration, independent FINAL, automatic preparation, main-app and later covering-family goal remains ACTIVE and incomplete.
