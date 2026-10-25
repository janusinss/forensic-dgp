# U96 and S96 created for experimental restoration testing

21 October 2026. The returned continuation archive passed independent import, saved-output and checkpoint-content audits. Both branches reached absolute update 96, adding 72 Adam updates each to their own audited update-24 states. All ten scheduled small TRAIN preservation checks passed. The unchanged pure auditor passed on its first run; no training recipe, quality threshold or return evidence was repaired.

The new inference artifacts are [U96](outputs/codeformer_spatial_continuation_v1_models/codeformer_spatial_u_u96.pth) and [S96](outputs/codeformer_spatial_continuation_v1_models/codeformer_spatial_s_u96.pth). They contain the exact learned adapters and use the existing hash-verified R2 base through the local test page. Each file is 212,827 bytes; these are adapter files, not standalone full CodeFormer weights. No training occurred locally while creating them.

**Test page:** [local spatial comparison](http://127.0.0.1:61815/restoration-spatial-test). Upload one cropped face, review whether visible structure is readable, then compare all four models or select one. The page includes untouched CodeFormer, R2, our U96 and our S96, with individual PNG and complete comparison downloads. It restores visible content; it does not remove coverings. This server is separate from the main application and does not change its model default.

## What improved

| Update 96 versus R2, repeated TRAIN evidence | U96 | S96 |
| --- | ---: | ---: |
| Moderate-blur PNG mean absolute error reduction, eight cases | 8.89% | 1.07% |
| Stronger-blur PNG mean absolute error reduction, four readable cases | 4.88% | 0.67% |
| Stronger-blur PNG adjacent-pixel error reduction | 1.40% | -0.0007% |
| Centered feature L1 reduction, ten fitted observations | 0.1296% | 0.6890% |
| Preservation snapshots passed | 5 of 5 | 5 of 5 |

These quantities are distinct. An image-error reduction or adjacent-pixel error reduction does not establish the historical facial-structure requirement. The fitting set has ten synthetic observations from six repeated photographs; the controls use eight historical TRAIN photographs. No new native CCTV, reserved FINAL or Zamboanga performance claim follows. Stronger blur here means the declared readable synthetic profile, not every severe user blur or privacy mosaic.

U96 improves the saved image errors more, while S96 improves its targeted centered features more. Relative to its own update-24 state, U lowers moderate/strong PNG error another 7.95%/4.34%. S lowers those errors another 0.95%/0.60%. S's stronger-blur adjacent-pixel error worsens slightly overall and on two individual cases; those regressions remain in the detailed record.

Root inspected all 18 planned sheets, covering 264 candidate PNGs at updates 30/36/48/72/96. U changes tones and local contrast modestly by update 96, with some excessive lightness reduced toward the paired photograph. It does not establish substantial new recovery of fine eyes, mouth or skin detail. S remains closer to the R2 appearance. Both inherit smoothing and differences from the actual clean photograph. These findings support experimental testing, not model acceptance.

Clear behavior distance improves on average, but one U and three S clear cases move farther from the retained untouched-CodeFormer teacher. Dark behavior distance to R2 rises to approximately 0.006908 for U and 0.000953 for S, within the existing 0.01 mean TRAIN stop. The dark anchors have no paired clean target; no clean-target score is invented. Passing this small control set does not resolve historical full-development preservation failures.

## Verified model and execution evidence

The 355,917,784-byte returned archive has SHA256 `a303657f04936d7b00e7bcb3cfbc1e6fe0c8f1f78b1b8f7a68f9f321f1f91c6c`. All 1,232 regular members and their 433,802,642-byte payload were checked against the original export manifest, sealed source/assets and physical imported files. The archive, checksum and companion agree. The full original checkpoint/optimizer/RNG records and all intermediate states remain retained.

The checkpoint audit safely loaded twelve states on CPU and verified exact own-24 adapter/Adam/RNG starts, finite FP32 tensors, absolute Adam steps, LR 0.0001 and all 144 additional commit/432 microbatch hash joins. The terminal export exactly preserves all 48 adapter tensor values. The 515 R2 base tensors remain separate and hash-bound; no unsupported independent replay of every optimizer operation is claimed.

The pure return audit checked both full image parity sets and all twenty initial feature-stat records before any additional optimization, then all saved pixels/metrics, 80 feature-case readouts/960 NPZ arrays, operation receipts, branch stops and resource limits. It records ten prefix passes, 272 full-model outputs and 3,136 adapter evaluations on the VM. Reported whole-worker time was about 590 seconds, including 378 seconds of optimization and 73 seconds of full parity/guard/probe work. These source-bound measurements are distinct from local inference timing.

Model files and their provenance are pinned in [model_exports_v1.json](outputs/codeformer_spatial_continuation_v1_models/model_exports_v1.json). U96 adapter canonical SHA256 is `fbfaa9d01c5ab04d2c4383485ba4e1eebb3d9f43cdab427773447248d14a586a`; S96 is `eb2e443e52ed4b3eace824afdfa6eff0f59a53613b7435304173b4f335e2ec42`. Neither artifact contains an optimizer or RNG. Their loading path is frozen, inference-only, verifies the exact base, removes all temporary adapter hooks and serializes shared-base inference.

## App verification boundary

The experimental page preserves the existing visual design. Bundled inline Playwright verified initial and populated layouts at 375, 768 and 1280 pixels without horizontal overflow, page exceptions or console errors. The readable-crop flow restored all four arms, downloaded four 256×256 PNGs plus a bundle and refused three unusable/out-of-scope review selections without invoking a model. These are functional checks, not a validated automatic input-quality threshold or full main-app acceptance.

The downloaded bundle retains the uploaded image, prepared input, observed support, PNGs, float delivery arrays and processing metadata. The float arrays represent native-RGB clipping followed by bilinear resizing, before PNG quantization; they are not unclipped native network activations. On the same audited readable crop, exported U96/S96 CPU delivery differs from saved L4 delivery by at most 0.000002921; PNG channel differences are at most one RGB8 level on a small fraction of channels. This is measured cross-device agreement, not bit-exact CPU/CUDA parity.

The user's `wild3.png` is an additional unpaired visual stress input with no clean original. The second four-model browser flow completed and downloaded all four PNGs plus its comparison bundle. Root inspected the populated screen: untouched CodeFormer removed more of the eye/forehead mosaic, whereas R2, U96 and S96 retained conspicuous blocks. U96 changed tones modestly; strong privacy pixelation remains unresolved. Neither this test nor subsequent preference establishes target accuracy or native CCTV performance.

Across both flows, ten downloads passed an independent payload audit. Each ZIP contains twelve closed, CRC-verified members; all eight standalone PNGs match their ZIP counterparts and all eight float arrays reproduce PNG quantization and observed-support handling. A separate repeat of R2 on the first readable TRAIN input, after both spatial models ran, returned exactly the same RAW and PNG hashes as before. Nine total successful CPU model calls and three refusals were recorded. This verifies experimental app behavior and shared-base isolation on these inputs, not full restoration qualification.

Full native/development comparisons, original preservation requirements, independent reserved FINAL review and user acceptance remain outstanding. No automatic promotion, unbounded extension or new training launch is justified by the test page alone. Restoration remains the priority; completion development is deferred.

Evidence: [import](outputs/codeformer_spatial_continuation_return_review_v1/import_v1.json), [pure saved-output audit](outputs/codeformer_spatial_continuation_return_review_v1/pure_return_audit_v1.json), [checkpoint contents](outputs/codeformer_spatial_continuation_return_review_v1/independent_checkpoint_contents_v1.json), [numeric review](outputs/codeformer_spatial_continuation_return_review_v1/INDEPENDENT_NUMERIC_REVIEW.md), [root image review](outputs/codeformer_spatial_continuation_return_review_v1/root_visual_review_v1.json), [independent export/app source review](outputs/codeformer_spatial_continuation_return_review_v1/independent_export_app_source_review_v1.json), [CPU/L4 output comparison](outputs/codeformer_spatial_continuation_return_review_v1/app_CPU_L4_output_comparison_v1.json), [independent downloaded payload audit](outputs/codeformer_spatial_continuation_return_review_v1/independent_app_download_payload_audit_v1.json).

The earlier storage offload was not repeated. Original models, research caches, splits, failures and migration assets remain preserved. The overall goal remains active and incomplete.
