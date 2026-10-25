# 23 October 2026: opt-in U96 workflow and retained comparison review

**The opt-in U96 restoration workflow works in the bounded local functional checks. U96 remains the incumbent experimental control; u64 is not promoted. Neither model is qualified against the full restoration goal.**

The recorded page is [the local U96 workflow](http://127.0.0.1:49619/restoration-workflow). Its PID/port are a runtime observation, not a permanent service address. The launcher receipt is `scratch/U96-restoration-staging-v1-server.json`; host process identity must be checked before reusing or restarting it. Existing main-app routes, defaults, checkpoints and design remain unchanged.

## Functional scope and repair

The new opt-in page uses the existing terminal design. It accepts one inspected crop, offers Auto/On/Off restoration and bypass or optional reviewed manual Protect/Apply, then shows original, selected input and one result. It offers PNG and optional evidence ZIP downloads. Completion and covering removal remain separate.

The readiness defect was in the controller: status could advertise availability while an owned worker was busy or uncertain. Status now validates the same finite clean worker counters used for restoration; typed worker-state and source-integrity failures return 409 and 503 respectively. The HTTP boundary, controller and launcher have 39 pure checks plus JavaScript syntax verification. Nine bounded HTTP checks exercised actual multipart/ASGI handling with fake or previously saved workers, including exact Off behavior, manual confirmation, stale-preview refusal and busy/uncertain status. Those checks ran no model.

The sealed [staging manifest](outputs/codeformer_restoration_staging_v1_preparation/opt_in_U96_staging_manifest_v1.json) binds seven source files, three retained lineage manifests and the existing model files. Independent source review and actual zero-neural launcher verification preceded server startup. Runtime metadata explicitly says `opt_in_U96_staging`, `model_qualified=false` and `main_app_default_changed=false`.

## Actual browser and delivery evidence

Root used the bundled inline Playwright runner. The retained checks cover the following four branches:

| Branch | Actual frozen U96 forwards | Observed behavior |
| --- | ---: | --- |
| Off, clear reviewed face | 0 | Original, selected input and result are the same resized image; PNG and ZIP downloaded. |
| On, bypass, readable degraded face | 1 | One U96 result; PNG and ZIP downloaded; owned worker exited. |
| On, confirmed manual preparation, same degraded face | 1 | Rebuilt preview ID matched the personally reviewed preview; one U96 result; PNG and ZIP downloaded; a separate owned worker exited. |
| Auto, bypass, same degraded face | 0 | Original-input signals did not suggest restoration; exact original resized result, optional ZIP omitted. |

The two actual inference requests took about 26.2 and 25.2 seconds at the controller boundary. Their total forward attempts, returns and completed requests are exactly two; status is inactive with zero uncertain requests or cleanup failures. A subsequent host-side process observation found both worker PIDs absent. Worker creation/command/terminal records are retained. The configured per-request deadline is 360 seconds with a 2.5 GiB available-RAM startup guard. No RAM/RSS time series was persisted in these app checks; configured limits and controller elapsed times are not a measured resource profile.

For both On results, Playwright found no accidental horizontal overflow, page exceptions or console errors at 375, 768 and 1280 CSS pixels. Root personally viewed all six result screenshots plus the manual preview/native prepared PNG, manual result PNG and Auto screenshot. Mobile manual captures are long and were downsampled by the image viewer; native-size screenshots remain retained. This is bounded functional evidence, not a soak test or complete main-app acceptance.

Manual preparation protected the reviewed face rectangle plus an 8-pixel guard, with radius 1 and a 16-pixel outside transition inside Apply. It rebuilt from the same immutable upload. Exact native Protect and outside-Apply preservation must be read from the independent saved-pixel audit; generation can still change protected facial appearance. This fixture tests mechanics, not optimal blur matching or useful preparation.

The [zero-neural saved audit](outputs/codeformer_restoration_staging_v1_actual_audit/independent_zero_neural_browser_saved_audit_v1.json) passed 311 independent checks. The [actual U96 pair saved audit](outputs/codeformer_restoration_staging_v1_actual_audit/independent_actual_U96_pair_saved_audit_v1.json) records 873 checks before 23 final evidence-pin checks reported at completion (896 total), covering actual transport, RAW/PNG delivery, downloaded bundles, manual masks and cleanup records, with five separate mutation regressions. Root's [personal visual receipt](outputs/codeformer_restoration_staging_v1_actual_audit/root_personal_actual_flow_visual_review_v1.json) records the screenshots actually inspected. No neural replay is implied by a saved audit.

Rebuilt preview noncontainer fields and all 16 decompressed ZIP payloads were exact; ZIP-container timestamps differed. The originally downloaded ZIP still matched its HTTP response byte for byte. Canonical decoded mask pixels, native filtering and preview identity were independently recomputed; original multipart mask PNG upload bytes were not retained, so their upload-byte hashes were not independently recomputed. Original and corrected audit-assumption evidence remains saved.

A browser-harness error is retained separately: the first check waited for `READY`, while the actual available UI label is `U96 · EXPERIMENTAL`. It sent no restoration POST and was stopped by exact owned process identity. A default sandbox process query also could not see the host server; a host query confirmed it remained alive. No server restart or production change was needed for that harness error.

## Broader comparison: partial, with observed failures

The old comparison is stopped; its absolute budget has expired. The [independent interrupted audit](outputs/codeformer_u96_postfusion_lr_broader_v3_interruption_audit_v1/run_v1/independent_interrupted571_saved_audit_v1.json) passed 959,556 checks on 571 saved outputs: 286 U96 and 285 u64, or 285 complete pairs plus one U96 output. Slot 572 has an unknown outcome; slots 573–716 are unattempted. Preserve all original, interrupted and failed records. Do not retry unknown 572 or resume under the expired budget.

The [partial renderer manifest](outputs/codeformer_u96_postfusion_lr_broader_v3_partial72_render_v1/partial72_visual_manifest_v1.json) and [root visual review](outputs/codeformer_u96_postfusion_lr_broader_v3_partial72_render_v1/root_personal_visual_review_v1.json) cover all 72 partial sheets and all 571 generated cells. Root found no consistent conspicuous useful u64 structural gain over U96. Several severe paired outputs change eyes, mouth, expression, hair or glasses; both arms introduce glasses absent from the target in `validation_va_ffhq_10401_blur`. The original complete 716-output/90-sheet plan is still incomplete.

The [partial counterexample analysis](outputs/codeformer_u96_postfusion_lr_broader_v3_interruption_audit_v1/partial_counterexample_analysis_v1.json) independently verified 278 strict individual scalar violations across 45 case IDs, not 45 people. Both arms already exceed the unchanged preservation limits on observed clear and normal-blur cases. Missing records cannot remove an observed maximum violation, but no full-cohort mean or completed 716-output gate is claimed. Native carried outputs remain separate, unpaired evidence.

The four-photo [paired Protect/Apply study](CCTV_CODEFORMER_PROTECT_APPLY_PAIRED_R1_REVIEW.md) was mixed/adverse: whole PNG MAE worsened in 11/12 correlated cases per arm, including 3/4 localized-blur photographs. All four photographs also occur in broader DEV; they are not an independent holdout. Manual preparation remains optional with confirmation and bypass. Automatic detection/matching and always-on surroundings blur remain deferred.

## Model diagnosis and next action

The [independent method review](outputs/codeformer_u96_u64_training_method_independent_review_v1/INDEPENDENT_METHOD_REVIEW_V1.md) confirms U96 used 96 optimizer updates of its adapter, not 96 epochs or full-model retraining. The experimental u64 trained downstream fusion heads while inherited tensors stayed frozen. Audited clean training photographs exist. More epochs, better edge scores or increased brightness alone do not establish useful facial restoration.

The next investigation is the [saved-lineage diagnosis](outputs/codeformer_u96_u64_training_method_independent_review_v1/NEXT_SAVED_DIAGNOSIS_AFTER_PARTIAL571_V1.md): separate coarse tone/illumination residuals from remaining spatial appearance errors on exact same-input saved original/R2/U96/u64 records. Missing RAW remains missing; no surrogate corrected image replaces the gate. This diagnosis does not establish a frozen-upstream bottleneck or justify a new trainable path before evidence is reviewed. No new training pilot is selected by this milestone.

Independent FINAL reviewers and adjudicator remain unassigned; final model/arms/paired cohort/budgets/acceptance rules remain UNFROZEN and reserved FINAL pixels remain unopened. Useful restoration improvement, full gate coverage, user acceptance, independent FINAL, accepted main-app integration and later separate covering-family scope remain unfinished. The goal stays ACTIVE. All actual training stays in the manual verified-transfer/tmux L4 workflow; this milestone used local inference and saved evidence only.
