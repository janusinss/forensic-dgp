# Matched feature-learning V1: returned evidence review

25 October 2026. **Both arms stopped for quality at eight updates. Neither is a qualified successor.** The manual run made and saved real Adam updates. Saved weights, source bindings, outputs and all executed quality decisions verify independently. This is a scientific preservation stop, not a trainer crash or missing export.

U96 remains the experimental app incumbent; original trained u64 remains the retained challenger. A/B failure checkpoints remain research evidence. Do not append epochs, rescue an earlier checkpoint, rerun this recipe unchanged, or publish either failed arm as the new baseline.

## What was tested and executed

Both arms began at the exact original complete575 trained-u64 state. A trained36 existing post-fusion head tensors (168,960 coordinates); B trained the heads plus24 existing U96 adapter tensors (220,288 coordinates). The515 R2 tensors stayed frozen. Both arms used the same admitted TRAIN photographs, degradation roles, loss, sampler order and LR schedule. This contrast changes trainable parameter scope; one matched seed cannot establish the universally best architecture.

The sealed ceiling was48 updates per arm, two passes through24 optimization photographs with five roles per photograph. These are not epochs over the full original face dataset. Only eight updates per arm ran. Updates24/48 and their terminal optimization readouts never ran; their absence is retained explicitly. Forty adaptive TRAIN monitor observations and the old56 quality bank remain separate by bank-qualified keys. Eight clear photographs recur across those banks, so they are not independent replications.

The journal reconciles712 forward returns,80 backward returns,16 Adam updates and two optimizer constructions. Of the forwards,432 initialized controls,8 repeated controls,192 later quality outputs and80 training outputs ran. All440 initializer/parity/repeat calls preceded optimizer construction. There were zero terminal optimization readout forwards. Worker exit2 was the intended quality-stop result; supervisor recorded403.502 seconds, no timeout and owned-process cleanup. No new model forward, gradient, optimizer or VM execution was performed by this local return audit.

## Numerical result: compare against the correct starting state

The table reports the old paired synthetic blur monitor bank at update8. Positive structure gain means lower adjacent RGB residual error against the photographic target. It is a proxy, not a percentage of recovered identity or a direct measure of visible sharpness. Positive MAE improvement means lower photographic pixel error. Relative percentages use the corresponding baseline error as denominator.

| Arm | Profile | NEW PNG structure gain vs original u64 | Cumulative PNG structure gain vs frozen U96 | PNG MAE improvement vs original u64 |
|---|---|---:|---:|---:|
| A | local_gaussian_1p2 | 0.048508% | 1.258056% | 0.149711% |
| A | global_gaussian_1p2 | 0.058913% | 1.685170% | -0.043915% |
| B | local_gaussian_1p2 | 0.046731% | 1.256301% | 0.241680% |
| B | global_gaussian_1p2 | 0.079418% | 1.705342% | 0.065361% |

**B's1.705342% global value includes the earlier u64 gain. Its new gain over exact original u64 is0.079418%, not1.705342%.** B is more favorable than A on several relevant aggregate pixel measures, especially clear preservation and degraded-profile MAE; it does not dominate every endpoint. Neither arm shows a consistent obvious visual improvement over u64.

## Why the run stopped

A worsened old clear MAE by0.095479% RAW and0.101698% PNG. Its supplemental clear and localized-blur MAE checks failed in both representations. B improved old clear RAW MAE by only0.001109%, but clear PNG MAE worsened0.008849%:0.03172407278620426 to0.03172688020022433. This is the sole failed supplemental new-bank check for B at8, and the inherited strict clear-PNG improvement check also failed. All legacy/ordinary guards and dark MAE checks passed. Their passes do not override failed clear preservation.

The1% terminal structure requirement was not binding at8; it would apply at24/48, which were not reached. This run did not stop because it missed a1% gate at8.

B improved mean PNG MAE on all four newly monitored degraded profiles. However, coarse color MAE already worsened on clear images by0.095692% RAW/0.115878% PNG, and localized blur by0.008838% RAW/0.031907% PNG. These supplemental coarse checks were prospective24/48 requirements, not the actual8-update stop. The saved8-update state would not meet them if evaluated by those same limits. Edge improvement coexists with a photometric tradeoff.

### The clear-image failure is broader than a PNG conversion issue

All712 RAW-to-PNG conversions are exact under the declared floor(RAW*255) rule. B improves the same five clear photographs and worsens the same three in both RAW and PNG; zero individual RAW-improved photographs become PNG-worse. The tiny aggregate RAW gain becomes a tiny aggregate PNG loss because the magnitudes change under quantization. It is incorrect to call the three failures export bugs.

For B,116,646 RGB components change across the eight clear PNGs. Exact total absolute RGB-code error increases1,126. The worst clear photograph contributes+14,492 code-error units and worsens PNG MAE1.297243%, offsetting gains elsewhere. Per-photograph results remain descriptive paired TRAIN evidence:

| TRAIN photograph | RAW MAE change; positive worse | PNG MAE change; positive worse | Integer RGB-code error delta |
|---|---:|---:|---:|
| tr_ffhq_08698 | -0.132878% | -0.121068% | -1997 |
| tr_ffhq_02147 | -0.203980% | -0.225668% | -2724 |
| tr_ffhq_05179 | -0.369155% | -0.372081% | -6760 |
| tr_ffhq_08949 | -0.231434% | -0.216394% | -2937 |
| tr_ffhq_11039 | +1.293351% | +1.297243% | +14492 |
| tr_ffhq_11154 | -0.127203% | -0.122073% | -3053 |
| tr_ffhq_08064 | +0.070605% | +0.058729% | +837 |
| tr_ffhq_08094 | +0.193294% | +0.198062% | +3268 |

Do not weaken the preservation gate or change PNG rounding to pass this state. Diagnose per-photograph signed RGB and coarse-block residuals before choosing a new loss/feature-path design. These results support investigating selective photometric learning and clear-image preservation in the joint B path; they do not prove a particular LR, coefficient, architecture or extra-epoch remedy. Preserve useful degraded-image brightening while testing clear and already-readable images explicitly. A new pilot would need a distinct justified recipe, finite stops and a separate verified manual VM packet.

## Independent audit and visual coverage

The returned archive is2,360,260,229 bytes, SHA256 `de7639af214d80be78ca73b24a3942afbf41223dae94591ce0e14076c079b860`. Its checksum and export sidecar agree. Full gzip CRC/ISIZE, tar EOF,3,842 members,705 returned static files and6,239 runtime binding joins passed. The three initializer weights are intentionally excluded from the results archive and were verified from sealed local dependencies. The return contains2,616,101,567 payload bytes. Archive completion does not imply training success.

The final supervisor/export-process exit receipts are absent inside the archive because they are written after export enumeration. Export-sidecar completion, the complete archive framing and scientific worker exit2 are verified. A final export-process exit code is not independently established; no such receipt is invented.

State verification decoded18 bounded closed checkpoints on CPU using weights-only loading. All575 original tensors, frozen subsets, final full states,16 actual step/moment recurrences, clipping, RNG and sampler bindings passed. A changed36 permitted tensors; B changed60. Saved graph/positive-gradient receipts verify the declared routes, not a fresh gradient reproduction.

An initial local state auditor required exact float64 diagnostic-norm equality and stopped on an observed1ULP representation difference. Its source, tests and failure evidence remain retained. The reviewed replacement allows8ULP only for recorded diagnostic norm scalars; finite/nonzero admission, tensor/state checks and every quality gate remain unchanged. No training rerun occurred.

The root assistant reviewed all54 prospective primary sheets and two additional partial sheets. Twelve quality PNG sheets and the two partial sheets were opened in full; the other42 primary sheets were inspected through11 labeled, resampled overview derivatives. Original evidence panels remain exact256 pixels with no enhancement or resampling. Overview inspection is not a claim of native-resolution inspection of every cell. RAW display uses floor(RAW*255), so its displayed RGB8 pixels match PNG; scientific RAW arrays were independently recomputed at float precision.

Independent readback checked all56 sheets/442 case-view rows:1,826 actual256 RGB cells,16 correctly empty dark-reference cells and480 missing terminal-arm cells. The latter represent240 unexecuted case-view rows, not fabricated results. The two partial optimization sheets show five roles BEFORE commit8, not predictions of the final8-update state.

Visual finding: A8/B8 remain very close to u64; existing softening and target departures in eyes, skin and mouth remain. No consistent obvious new detail gain was found. Similarity does not negate measured preservation failures. This assistant review is not independent blinded human FINAL review.

## Evidence boundaries and statistical interpretation

This return is paired synthetic TRAIN/adaptive monitor evidence, including reused photographs and correlated degradation roles. It adds no native CCTV DEV comparison, held-out FINAL result, independent identity count, PSNR/SSIM result, covering-family qualification, human acceptance or Zamboanga performance claim. Existing broader DEV failures remain binding. No app model or design was changed.

Academic Research Suite validation scan:11/11 checked, with no significance or population-effect claim:

| Pattern | Check and limit |
|---|---|
| Simpson's paradox | Role/view and per-photograph results are separated. Aggregate RAW/PNG signs differ; the same5/3 photograph directions persist. No universal per-face improvement claimed. |
| Ecological fallacy | Monitor means are not identity-level or person-level success rates. |
| Berkson's paradox | Curated/adaptive TRAIN samples are selected; no population representativeness assumed. |
| Collider bias | No observational causal regression or conditioned population model is fitted. Selection effects remain unquantified. |
| Base-rate neglect | No sensitivity, specificity, prevalence or readiness probability is estimated. |
| Regression to mean | Exact initializer/control pairs are retained; no extreme-case gain is generalized from one seed. |
| Survivorship bias | Both failures and all unexecuted terminal cells are reported; no earlier-checkpoint rescue. |
| Look-elsewhere effect | Both arms, all roles/views and every executed gate are retained; no significance claim from a selected metric. |
| Garden of forking paths | Sealed protocol and prior failures remain bound; diagnostic analyses are explicitly exploratory. |
| Correlation versus causation | This finite matched contrast describes these saved training paths; it does not prove a general remedy or native CCTV effect. |
| Reverse causality | Saved source/state precedes saved outputs in the journal; no person-level directional causal inference is made. |

## Current decision and next action

Keep the existing goal and app incumbent. Retain original u64 and both stopped states. Analyze the saved clear-face signed RGB/coarse-color changes and their contributions before designing any successor pilot. More updates alone are unsupported by this result. No automatic historical launch, unchanged rerun or promotion is authorized by this review.

Audit receipts:

- [Transport and journal](outputs/codeformer_matched_feature_learning_v1_return_audit_v1/transport.json), [saved-state audit](outputs/codeformer_matched_feature_learning_v1_return_audit_v1/state.json).
- [Independent saved-pixel gates](outputs/codeformer_matched_feature_learning_v1_return_audit_v1/pixels.json), [per-profile/per-photograph diagnosis](outputs/codeformer_matched_feature_learning_v1_return_audit_v1/diagnostic_summary_r1.json).
- [All comparison sheets](outputs/codeformer_matched_feature_learning_v1_return_audit_v1/visual_sheets.json), [independent pixel readback](outputs/codeformer_matched_feature_learning_v1_return_audit_v1/independent_visual_pixel_readback_v1.json).
- [Root visual review](outputs/codeformer_matched_feature_learning_v1_return_audit_v1/root_visual_review_v1.json).
