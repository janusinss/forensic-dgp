# U96/S96: no accepted visible upgrade; interactive testing kept available

21 October 2026. The user's direct review was: “still the same i cant really tell the difference not a significant update”. U96/S96 have **not met the user-visible improvement requirement**. Their successful training/export and numerical changes do not qualify either as a replacement. Keep all four models available for explicit experimental comparison; do not extend the same U/S recipe unchanged.

[Interactive comparison](http://127.0.0.1:61815/restoration-spatial-test) remains running. The development client was stopped to free that shared server after the user received its busy message. No new background inference is scheduled while the user tests. No model weights, app defaults or quality thresholds changed during this review.

## Evidence completed and evidence still missing

A fixed cohort was independently frozen before inference: 36 paired synthetic filter DEV observations from four photographs, all 24 usable ChokePoint crops, and eight coarse QMUL-SurvFace crops. Its hash is `e42afd973305e42422bcb273dc419ab608bbc1437ab332e62d6eb006e0fe9307`. All 32 native original crops reproduce their exact saved prepared input, observed support and geometry. Sixteen other native crops and 24 insufficient synthetic filter controls remain excluded from new model calls. Existing source terms, memberships and input-only labels remain retained.

The four synthetic DEV source photographs exclude all six U/S TRAIN photographs and all 391 exact historical TRAIN source hashes. Person-level overlap and overlap with pretrained CodeFormer's face data remain unestablished. No reserved FINAL images were opened or optimized. Native CCTV has no aligned clean reference; the comparison cannot establish ethnicity or Zamboanga performance.

Only **10 of 68 inputs** completed before the interactive release: nine profiles of photograph `va_ffhq_00383` and the clear control `va_ffhq_01093`. All 40 candidate RAW/PNG outputs and original single-arm server responses are saved. **58 inputs, including all 32 native crops, remain pending**. This partial result does not replace a full development or final evaluation. User nonacceptance is sufficient to reject these candidates as upgrades without completing the interrupted batch.

Root personally reviewed all three partial sheets and all 40 candidate PNGs. U96 mainly changes tone/local contrast; S96 stays close to R2. For the local mosaic case, R2/U96/S96 retain conspicuous central eye/forehead blocks, while untouched CodeFormer suppresses more of them. All arms can change appearance or retain artifacts on severe corruptions. Sharper output alone does not prove recovered detail. Two clear controls also remain too small a scope to establish preservation.

The independent saved-output audit passed for these ten cases: all 40 original single-arm ZIP/body/header/metadata joins, ten explicit client merges, 40 RAW/PNG pairs and exact input/support/provenance/quantization checks. Its independently implemented NumPy measurements agreed within 0.000000000000358 arithmetic difference. This is output integrity, not model acceptance.

| PNG measurements: eight corruptions of **one** DEV photograph | Mean image MAE, lower is better | Mean SSIM, higher is better |
| --- | ---: | ---: |
| Untouched CodeFormer | 0.029780 | 0.807482 |
| R2 | 0.048498 | 0.784620 |
| U96 | 0.045699 | 0.788688 |
| S96 | 0.048178 | 0.785096 |

U96/S96 reduce image MAE relative to R2 on all eight saved corruptions, but untouched CodeFormer has lower MAE on all eight. These are global paired photographic measurements, not facial-structure gains or population performance. The two clear controls are separate: mean MAE original/R2/U/S = 0.016186/0.025956/0.022689/0.025544; mean SSIM = 0.930774/0.924182/0.932080/0.926426. U's higher mean SSIM on those two clear cases does not prove preservation or override the user's nonacceptance. No new native CCTV PSNR/SSIM was computed.

## Transport and interactive release

The original combined-response client stopped with a connection reset after four server forwards completed, without a saved response. The next single-arm client retained a partial response of 1,566,720 bytes against 1,592,163 declared bytes; its original-model forward completed, but its bundle was incomplete. Both failures and the partial body remain preserved. Two synthetic zero-model transport probes passed; they did not establish the reset's root cause.

The third reviewed transport used a separate HTTP/1.1 keep-alive connection for each request and read the exact declared Content-Length. Its completed outputs retained all original response bodies, headers, six-member single-arm ZIPs and metadata. Each four-arm comparison is explicitly marked as a client merge into a twelve-member ZIP. A client merge is not an untouched server response.

To free the interactive test, root directly stopped the hash-identified development launcher PID 28996. Its previously observed child PID 31532 was subsequently absent; no direct child-kill claim is made. The original release record overstated two direct stops and is preserved alongside the corrected release. Independent process inventory confirmed that server PID 32392 remained running. The execution handle was terminal, with its original partial run receipt preserved rather than edited into a completed run.

At release the server recorded 55 attempted, returned and successful forwards. The R2-transport interval added 41: 40 responses saved across ten completed inputs, plus one next-case original-model forward completed without a saved response. The five earlier undelivered forwards remain separate. No incomplete payload is counted as a verified restoration output. No automatic retry or resume ran after release.

## Next method decision

[Post-rejection method review](outputs/codeformer_spatial_dev_review_v1/design_review/POST_USER_REJECTION_METHOD_DECISION.md) recommends a distinct research candidate initialized from untouched CodeFormer, with all 515 original tensors frozen. A zero-initialized correction after the multiscale SFT blocks would condition jointly on observed encoder and decoder context. Actual image reconstruction and preservation, rather than clean-feature matching alone, would guide learning on a broader audited set of recoverable corruptions.

This is a design recommendation, not an implemented or proven improvement. Earlier original-based training already changed too much appearance; earlier R2 fusion/skip trials already used image and adjacent-error losses. Rebase alone, image loss alone, a larger learning rate or more unchanged updates is therefore insufficient justification. The new base, conditioning mechanism and data coverage must be specified and independently reviewed together; no causal attribution to one factor is established without an ablation.

R2 remains a lowlight/brightness comparator. Original initialization has known deficits relative to R2 on some dark-image measures. A future protocol must record those deficits and retain the existing dark benchmarks and preservation requirements; it cannot silently waive an inconvenient stop or claim zero-step R2 parity. Clean target images and synthetic corruption footprints are training supervision only. No clean target or true corruption mask may be required at user inference; restoration reliability is separate from completion/removal masks.

The next preparation task is to audit suitable detailed TRAIN source photographs and fix the correction architecture, loss weights, sampler, initial parity/gradient witnesses and finite resource/stop contract before sealing any manual L4 packet. No new training or VM action ran here. Completion development remains deferred until restoration acceptance. The overall goal remains active and incomplete; the earlier storage offload is closed and was not repeated.

Evidence: [frozen cohort](outputs/codeformer_spatial_dev_review_v1/cohort_manifest_v1.json), [user feedback](outputs/codeformer_spatial_dev_review_v1/user_visual_feedback_v1.json), [partial root visual review](outputs/codeformer_spatial_dev_review_v1/root_partial_visual_review_v1.json), [independent partial saved-output audit](outputs/codeformer_spatial_dev_review_v1/independent_partial_saved_output_audit_v1.json), [corrected interactive release](outputs/codeformer_spatial_dev_review_v1/interactive_release_r1.json), [independent lifecycle audit](outputs/codeformer_spatial_dev_review_v1/independent_interactive_release_audit_v1.json).
