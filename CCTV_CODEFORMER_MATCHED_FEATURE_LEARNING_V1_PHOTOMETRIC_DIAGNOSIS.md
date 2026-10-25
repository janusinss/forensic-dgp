# Matched feature-learning: clear-image photometric diagnosis

25 October 2026. **Saved-output diagnosis complete; no successor model or VM packet is prepared by this document.** Both matched arms remain closed at eight updates. This supplements the [returned-evidence review](CCTV_CODEFORMER_MATCHED_FEATURE_LEARNING_V1_RETURN_REVIEW.md); it does not replace its failed gates.

## What the additional measurements establish

All eight clear TRAIN photographs were inspected using exact retained u64/A8/B8 RAW and PNG pixels and their photographic references. Analysis splits the output change into a constant per-channel image mean and a remaining zero-mean spatial field. Squared output energy splits orthogonally; target-error contributions use a symmetric two-order analytical allocation. These are descriptive mathematical decompositions, not causal proof or generated corrections. They do not run a model, clip or alter a saved restoration, or deploy target-derived information.

| B clear case | Constant RGB component share of output-change energy | Global contribution to RAW MAE change | Spatial contribution to RAW MAE change | Delivered PNG absolute RGB-code error delta |
|---|---:|---:|---:|---:|
| tr_ffhq_11039 | 82.833% | +0.000249562 | +0.000023278 | +14,492 |
| tr_ffhq_08064 | 4.335% | +0.000039687 | -0.000018472 | +837 |
| tr_ffhq_08094 | 5.792% | -0.000001102 | +0.000066206 | +3,268 |

Positive error deltas mean worse. Error magnitudes are on normalized RGB [0,1]; integer code deltas sum absolute target error over RGB8 components. Energy share is a different quantity from contribution to target error: the second row has low constant-component energy but a harmful constant-component error contribution, partly offset by spatial improvement.

The largest failure, tr_ffhq_11039, shifts channel means by approximately [-0.000551014, -0.000446714, -0.000290088]. Its error increase is widespread: 209 of 256 coarse blocks worsen. The third failure, tr_ffhq_08094, is instead dominated by the spatial contribution to error. A uniform brightness or channel-offset correction is not established as a solution for this mixed behavior.

Across all eight clear photographs, B's mean RAW MAE delta is -0.0000003595: the harmful mean-color contribution (+0.0000084400) is slightly outweighed by spatial improvement (-0.0000087995). Quantized PNG error still rises by 1,126 absolute code units. A's corresponding net PNG error rises by 12,940. Thus B is the more promising diagnostic feature path, while both still fail preservation.

## Next design review, before another training packet

1. Keep original trained u64 as the exact starting comparison. Treat B's failed state as evidence, without making it the incumbent or a rescued initializer.
2. Review a condition-sensitive feature update with separately measured channel-mean color and local-structure preservation on clear inputs. Preserve the existing useful degraded-image brightening; do not apply a universal brightness subtraction.
3. Require any inference selector or conditioning to use the uploaded input only. Photographic targets, known synthetic roles and measured residual maps are analysis tools, never runtime information.
4. Specify and source-review a distinct finite comparison before packaging it. Keep the same RAW/PNG gates, original checkpoints, historical failed recipes and adaptive TRAIN versus independent DEV/FINAL boundaries. Extra epochs, an LR change alone, or a different PNG rounding rule do not resolve the demonstrated mixed failure.
5. Admit another manual L4 run only after its packet, actual initialization, connected gradient path, controls, timing and stop rules verify. Qualification still requires useful visible improvement and independent development/final review.

This is a hypothesis for a prospective comparison, not proof that a particular architecture, coefficient, learning rate or extra training will work. The diagnosis provides no new native CCTV performance, identity recovery, independent human acceptance or covering-family qualification. U96 remains the experimental app incumbent; original u64 remains retained.

## Reproducibility and visual limits

[Full measurements](outputs/codeformer_matched_feature_learning_v1_return_audit_v1/photometric_diagnosis_v1.json), [grounded interpretation](outputs/codeformer_matched_feature_learning_v1_return_audit_v1/photometric_diagnosis_interpretation_r1.json), and [ten passing arithmetic tests](outputs/codeformer_matched_feature_learning_v1_return_audit_v1/photometric_author_tests_v1.json) retain their consumed-file hashes and actual operation boundaries.

The root assistant viewed the [diagnostic map sheet](outputs/codeformer_matched_feature_learning_v1_return_audit_v1/photometric_diagnostic_v1.png). Its colors deliberately amplify signed changes on fixed scales: red means brighter or more target error, blue means darker or less target error. These panels are diagnostic maps, **not restored face outputs**, new model predictions or a visual readiness test. References and source pixels remain retained; numerical computations use unmodified arrays. Encoded RGB/luma arithmetic is not calibrated physical illumination.
