# Trained tone-control composite: actual return review

**17 October 2026 — MODEL CREATED / 50 HEAD UPDATES VERIFIED / QUALITY NOT ACCEPTED.**

[Exported trained composite](outputs/codeformer_tone_control_training_v1_candidate_unqualified/README.md). [All 35 comparison sheets](outputs/codeformer_tone_control_training_v1_return_review/pixels_v1/visual_sheets/INDEX.md). [Next design review](CCTV_CODEFORMER_TONE_CONTROL_TRAINING_V1_DESIGN_REVIEW.md).

The human's manual L4 run trained a genuine 8,257-parameter input-only tone controller for 50 Adam updates and 200 microbatches. The earlier fine-tuned CodeFormer remained frozen: all 515 restorer tensors exactly match the retained parent. All four controller tensors changed. The initial, update-50 and stopped checkpoints, optimizer/RNG/sampler records, 3,090 operation events and all output banks were preserved. This is a frozen adapted restorer plus a trained controller; it is not 50 additional CodeFormer updates or a new architecture trained from scratch.

The local inference artifact is **376,657,585 bytes**, SHA-256 `807f92a90960b559e2fa9ee7201ef6a1c00ca50c47c7f86dcdcaca0904f486a0`. It contains **515 base tensors plus four trained controller tensors** in a distinct composite format. It has no `params` or `params_ema` alias. Standard CodeFormer loaders and the current app cannot reproduce this composite. A dedicated loader must preserve both tensor sections and the exact input-feature/support/output policy; silently running the base alone would discard the newly trained component. The full stopped checkpoint separately retains the optimizer and RNG. CPU serialization is not new training.

The returned archive is **2,858,634,754 bytes**, SHA-256 `6a4386f84ebc27d6201ad188bfb19dc36bdb0a6fbab3c6f77c0662ecd4e547bf`; protocol `f10ce6d52c2ec259632750af2d3afbdf97c5c925b99849fae2a2c4d9cd04a5b1`. Independent transport readback verified all 8,533 members / 3,209,400,295 payload bytes, the manifest, 24 frozen static files, single gzip CRC and full USTAR end, checksum and all external companion records. The files were imported as inert evidence; returned Python was not executed.

Independent saved-pixel readback verified all **1,218 DEV RAW/PNG pairs**, 812 base RAW arrays, historical original/zero-head parity, repeated frozen base outputs, exact scalar composition, input features and protected padding pixels. There were no saved-scalar, parity or gate-record mismatches. Root personally viewed all 34 DEV sheets covering 406 cases plus the two-case known TRAIN stress sheet. Exact256 panel pixels were independently checked; the viewing tool displayed resized previews of the large sheets. Integrity and visual review do not establish useful restoration.

| Preservation check against untouched CodeFormer | Mean regression | Worst regression | Cases above 0.03 |
|---|---:|---:|---:|
| Clear RAW MAE, 61 cases | 0.013815 | 0.041288 | 4 |
| Clear PNG MAE, 61 cases | 0.014620 | 0.039971 | 5 |
| Normal-blur RAW luminance bias, 53 cases | 0.017846 | 0.052370 | 12 |
| Normal-blur PNG luminance bias, 53 cases | 0.015709 | 0.048467 | 8 |
| Normal-blur PNG coarse RGB MAE, 53 cases | 0.007821 | 0.031836 | 1 |

The unchanged preservation limits are **0.01 mean / 0.03 worst**. Normal-blur RAW coarse RGB MAE also fails its worst-case limit: mean 0.008477, worst 0.033666, one case above 0.03. Both preservation families fail. Clear PNG failures are fewer than the prior normal-tone run's 60/61, but this different frozen-parent/controller experiment is still unqualified; that comparison does not establish a better accepted model.

| Additional dark-retention check, 53 cases each | Frozen base PNG MAE | Trained composite PNG MAE | Frozen base / trained SSIM |
|---|---:|---:|---:|
| Lowlight | 0.101779 | 0.114762 | 0.614690 / 0.611274 |
| Compound | 0.138071 | 0.150575 | 0.520845 / 0.513941 |

Both dark profiles regress against the frozen adaptation. The known normal TRAIN stress case passes its tone requirement; the known dark TRAIN case fails. These are two previously seen TRAIN cases, not independent generalization evidence. All 406 final DEV offsets are negative, including lowlight, compound and native crops. The head therefore darkens both normal and dark outputs. It still retains some benefit of the adapted restorer relative to the untouched original on dark examples, but fails the explicitly required retention against its own frozen base.

The saved TRAIN audit confirms that the target construction is correct: normal observations use original CodeFormer means and dark observations use frozen-base means. In the 200 recorded pre-update responses, **98/100 dark observations received unwanted negative offsets and worsened**, despite zero desired dark correction. Normal loss improved on 85 observations and worsened on 13, with two below readout resolution. The mean recorded online objective is 38.5% lower than zero-offset reference losses on the same observations. That is not a fixed-final-head before/after evaluation or proof of a monotonic training curve. Improving the aggregate objective did not ensure selective preservation.

The prior clean-reference oracle demonstrated scalar correction capacity only. Selective input-only learning was an explicit hypothesis; this run did not establish it. There is no returned frozen-final evaluation of all TRAIN examples or per-step head-gradient record to uniquely distinguish objective interference, limited features/capacity, finite optimization and generalization. Additional unchanged epochs, a larger learning rate or a weakened gate are not justified by this evidence. [Detailed TRAIN response review](outputs/codeformer_tone_control_training_v1_return_review/state_v1/training_response_review_v1.md).

Visual findings agree with the saved-pixel evidence. Many usable ChokePoint faces remain coherent, with little change beyond darker tone. Paired blurred examples often have smoother surfaces than untouched CodeFormer, while eye opening, expression and mouth contour can still change. The head cannot repair these features because it adds one common RGB offset to frozen restoration outputs. Severe QMUL block-like, glare and covered crops still have missing, misplaced or invented facial features. Request clearer or less-covered input for insufficient cases; do not infer a recovered identity from realistic texture.

The 48 native CCTV cases remain **unpaired** and have no clean-reference PSNR/SSIM claim. Source groups, terms/provenance and overlap labels remain separate; reserved FINAL identities were not used. The observed-support mask here is the input/letterbox support, not a user covering/removal mask. This experiment does not improve covering detection or completion. It supports no ethnicity or Zamboanga performance claim.

The worker took 464.19 seconds; supervised worker time was 479.24 seconds. Preflight took 16.26 seconds and export 237.12 seconds. VM-reported peaks were 3,314,655,232 bytes RSS and 1,543,503,872 bytes CUDA reserved; sampled minimum free space was 35,845,939,200 bytes. The recorded stop was the declared quality decision, not a timeout, transfer, gradient-math or storage failure. Local audits used CPU arrays and `weights_only` state serialization; no local neural forward, autograd, optimizer or VM launch ran.

Retain this genuine model and every failure record. Do not rerun the old guide, resume its optimizer, alter preservation limits or promote it in the app. The next design review must distinguish fixed-state TRAIN selectivity from held-out TRAIN response before another training recipe, while keeping facial structure as the actual restoration objective. User acceptance, independent FINAL review, qualified app selection/override/PNG flow and bundled inline Playwright verification remain outstanding. Later separate seven-family completion work remains deferred until restoration acceptance. The full goal stays active and incomplete.
