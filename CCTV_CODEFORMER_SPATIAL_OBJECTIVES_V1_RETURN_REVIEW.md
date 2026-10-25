# Spatial-objective return: audited, small learning signal, no model promotion

21 October 2026. The export recovery succeeded. All three downloaded files agree on SHA256 `67ab66ad45e760565fdd04319a5578590bd96218eb3c0a6fd456c21e5cf2e545`. The 233,799,969-byte archive contains 842 verified regular files with 289,561,892 bytes of member payload. Import independently checked every original manifest binding, sealed return source/asset, physical extracted file and gzip trailer. Original VM-only weights and research caches were not duplicated into the return.

The two branches each completed **24 Adam updates**, with three microbatches per update: 48 optimizer commits and 144 backwards total. These are updates, not epochs. The source-bound ledger records 172 full outputs and 1,504 adapter evaluations. All six repeated-TRAIN preservation snapshots passed the unchanged rules. This successful finite screen does not resolve inherited R2 preservation failures on the full development comparison.

## What the images and feature measurements show

Both branches use the same frozen R2 base, adapter architecture, initialization, optimizer and sampler. U minimizes ordinary feature error; S minimizes per-channel centered feature error. The comparison optimized ten observations from six repeated TRAIN photographs. The strong rows below are four predeclared readable TRAIN observations, not held-out evaluation or native CCTV.

| Measured change at update 24 versus initial R2 | U: ordinary objective | S: centered objective |
| --- | ---: | ---: |
| Centered feature L1 reduction, all ten fitted observations/four sites | 0.0205% | 0.0684% |
| Moderate blur PNG mean absolute error reduction, eight TRAIN cases | 1.0192% | 0.1215% |
| Strong blur PNG mean absolute error reduction, four TRAIN cases | 0.5635% | 0.0758% |
| Strong blur PNG adjacent-pixel error reduction, four TRAIN cases | 0.1504% | 0.0020% |
| Preservation snapshots passed | 3 of 3 | 3 of 3 |

These percentages measure different quantities. **The 1.0192% image-error reduction is not the historical 1% facial-structure requirement.** Neither a small feature gain nor a lower whole-image error establishes recovered identity or useful visible facial improvement. Hair, clothing and background also affect image metrics. RAW and downloaded-PNG measurements remain separate in the detailed outcome receipt.

S directs more learning toward spatial corrections: the mean correction DC-energy fraction is about 34%, versus 93% for U. This is a feature decomposition, not an image-brightness measurement. S improves its targeted centered-feature quantity about 3.33 times as much as U, while U currently improves the paired image errors more. Selecting S solely from its feature loss would discard that output evidence.

Root personally inspected all **12 planned sheets**, covering 168 candidate PNG outputs at updates 6, 12 and 24. U, S and R2 remain visually similar across the clear, moderate-blur, strong-blur and synthetic dark groups. No material new improvement in visible eyes, nose, mouth or jaw is established. R2 smoothing and appearance differences from the paired clean photographs remain. Passing the small guard set does not make these differences acceptable. The dark anchors have no paired clean target; their scores measure distance to retained R2 behavior only.

Representative saved comparison: [strong blur at update 24](outputs/codeformer_spatial_objectives_return_review_v1/visuals/strong_u0024.png). The [visual manifest](outputs/codeformer_spatial_objectives_return_review_v1/visual_manifest_v1.json) binds all sheets and their source images; the [root visual review](outputs/codeformer_spatial_objectives_return_review_v1/root_visual_review_v1.json) records their findings.

## Training evidence and repaired checker assumptions

The independent checkpoint-content audit loaded all eight adapter states safely on CPU. It validated 24 finite FP32 tensors/51,328 parameters per state, matching identity starts, exact Adam steps, finite moments, source-bound RNG/reset records and the 48-commit/144-microbatch hash chain. Both final adapters changed. The adapter-only files do not themselves contain the frozen base; base immutability relies on the retained initializer plus source/runtime canonical-state checks. This is not an independent replay of every optimizer operation or a new neural inference.

Two local checker assumptions needed correction after preserving their failures. Linux current RSS and `getrusage` peak RSS use separately sampled accounting sources; the original auditor incorrectly required them to be ordered. The local revision checks each against the same unchanged 10 GiB cap, matching the worker's `max(rss, peak)` guard. The largest reported current RSS was 1,934,807,040 bytes; no reported sample crossed the cap.

The second checker mismatch came from Python 3.10 versus 3.13 floating-point summation. A separate local auditor explicitly reproduces the VM's ordered additions, restoring exact equality for all six saved gates without adding an epsilon or changing a quality threshold. Original sealed sources, protocol, returned files, first failures and both auditor revisions remain intact. The final pure audit passed all physical pixels/metrics, 70 feature case readouts/840 NPZ arrays, operation receipts, resource checks and closed namespace. Sufficient-statistic checks do not replay full feature activations.

The returned worker reported 126.44 seconds of optimization for 48 updates. Whole worker self-samples reached about 287.65 seconds; supervision finished in 297.26 seconds. Recovery packaging took 22.07 seconds. These are source-bound recorded measurements, not a fresh live VM observation. The original exporter failure was packaging-only; recovery added zero training or neural calls.

## Next controlled direction

The independent [method review](outputs/codeformer_spatial_objectives_return_review_v1/method_review_v1/METHOD_REVIEW.md) supports one finite, matched continuation of **both U and S from their audited update-24 states to update 96**. That means 72 additional updates per branch, retaining their own Adam/RNG states and the shared declared sampling schedule. Early guards at 30/36 and checkpoints/guards at 48/72/96 must stop a failed branch. The [96-update design review](outputs/codeformer_spatial_objectives_return_review_v1/method_review_v1/CONTINUATION_96_DESIGN_REVIEW.md) records the proposed operation budget and decision rules.

This is a design recommendation, **not a runnable execution packet**. No continuation has been staged or run. Exact closed output/storage bounds, transfer files, target Python-compatible export, resume parity and a return auditor must be verified before publishing manual commands. The original completed experiment must not be edited or relaunched. Increasing training without a bound or choosing a model from training loss alone remains unjustified.

The useful question is whether the measured spatial learning can grow enough to change actual faces while preserving clear and dark behavior. If feature gains grow but face outputs remain unchanged, inspect decoder/SFT response before another extension. If feature fitting plateaus, review adapter capacity or normalization. Neither outcome automatically justifies starting from scratch. Full development/native comparisons, independent reserved-final review, user acceptance and main-app Auto/override/download qualification remain outstanding. Restoration stays first; completion development remains deferred.

## Audit evidence

- [Verified import](outputs/codeformer_spatial_objectives_return_review_v1/import_v1.json)
- [Pure return audit R2](outputs/codeformer_spatial_objectives_return_review_v1/pure_return_audit_v1_r2.json)
- [Independent checkpoint contents](outputs/codeformer_spatial_objectives_return_review_v1/independent_checkpoint_contents_v1.json)
- [Separate RAW/PNG and feature outcomes](outputs/codeformer_spatial_objectives_return_review_v1/independent_outcome_summary_v1.json)
- [Root visual review](outputs/codeformer_spatial_objectives_return_review_v1/root_visual_review_v1.json)

The current archive, all saved states and failures remain preserved. No app default changed, no further actual training ran, and the overall goal remains active and incomplete.
