# CodeFormer AdaIN gradient isolation V1 — returned result, 17 October 2026

**50 real Adam updates verified; preservation failed; no model promotion or unchanged resume.** The transfer, export and independent audits succeeded. The trained candidate did not satisfy the restoration requirements. The full CodeFormer restoration-first thesis goal remains active.

[All 55 comparison sheets](outputs/codeformer_adain_gradient_isolation_v1_return_review/visual_sheets/INDEX.md) · [Three failed clear cases at exact 256 pixels](outputs/codeformer_adain_gradient_isolation_v1_return_review/critical_clear_256.png) · [Independent numerical tables](outputs/codeformer_adain_gradient_isolation_v1_return_review/SCIENTIFIC_READBACK_V1.md).

## 1. Actual run and retained evidence

The manually launched L4 pilot started from untouched pretrained CodeFormer. It changed one training gradient route: the image objective no longer backpropagates through the encoder's AdaIN style argument. CE/HQ supervision, token prediction and fusion learning remained. This was a finite 50-update comparison against the retained FP32 pilot, not additional epochs on a previously stopped state.

The archive `outputs/cctv-codeformer-adain-gradient-isolation-v1-results.tar.gz` is **2,864,673,701 bytes**, SHA-256 `7f7fc6b5a1fd58ea1ae5801d5f1edc95e66d11fe5dbc0256e1b4fe88ab7fa633`. Protocol SHA-256 is `4e2cc86857628617edbb313b6ef2911602a63ad4e1f9862afe07c57c0ba770c5`. All **6,558 regular members / 3,143,169,457 payload bytes**, exact namespaces, export manifest, 31 local overlay files and 1,939 immutable dependency/source/asset bindings passed hash checks before exclusive import. No returned Python was executed during import.

The authoritative worker stop was `PilotStop: Declared RAW/PNG clear-preservation guard failed; retained snapshot`. The state audit confirms 50 committed Adam updates, 200 TRAIN exposures/backwards, 100 photographic reference IDs, 1,614 recorded neural forwards, nine gradient queries and 2,073 ordered operation reservations. Those IDs are not independently established unique people. All 350 trainable tensors changed, all 165 frozen tensors stayed exact, and update-50 and stopped model/Adam/RNG/sampler states agree. All 391 HQ array targets and 50 behavior anchors match retained FP32 evidence. The recorded `resume_safe` state flag does not authorize resuming a quality failure.

Measured times: 16.04 seconds for the route witness, 214.25 seconds for optimization, 675.53 seconds for the worker's internal run, 692.32 seconds for supervised worker execution, and 191.44 seconds for export. These intervals overlap and must not be added as independent costs. The supervised preflight took 7.24 seconds. Runtime was NVIDIA L4, Torch 2.9.1+cu129/Python 3.10.12. FP32/autocast-off controls and seeds were checked; CUDA deterministic-algorithm flags were not enabled, so global deterministic execution is not claimed.

The retained stopped checkpoint SHA-256 is `43c0f313365e13849ddcd852f401511e4c7ae52cbac55fdd98a6b032f06ef37b`. Untouched CodeFormer, original DGP checkpoints, prior FP32 evidence, splits, source data and failure records remain preserved. No app restorer was replaced.

## 2. What improved and what failed

The independent pixel audit recomputed **812 new RAW/PNG output pairs**, 406 retained FP32 update-50 pairs and 406 exact original baseline pairs. All delivered PNG conversion/padding checks passed. There are 358 paired synthetic cases and 48 native/unpaired cases per state. The original 61-clear and 53-blur preservation cohorts and thresholds remain unchanged.

| Control and metric | RAW mean / worst increase | Delivered PNG mean / worst increase | Cases exceeding 0.03, RAW / PNG |
|---|---:|---:|---:|
| Clear paired MAE | 0.008292 / 0.057293 | 0.007719 / 0.055917 | 3/61 / 3/61 |
| Normal-blur luminance bias | 0.043238 / 0.074893 | 0.040136 / 0.070992 | 47/53 / 37/53 |
| Normal-blur coarse RGB MAE | 0.024078 / 0.055026 | 0.022473 / 0.053090 | 15/53 / 14/53 |

These are increases relative to untouched CodeFormer against paired clean references, on the declared observed support. The mean limit is 0.01 and the per-case limit is 0.03. Clear mean MAE passes, but the worst-case requirement fails. Both RAW and delivered PNG fail the overall clear and normal-blur guards. The three clear failures are `comparison_va_asian_04382_clear`, `comparison_va_asian_09306_clear` and `validation_va_ffhq_27440_clear`, the same as the prior FP32 run. Dataset names do not establish ethnicity.

There are real gains over untouched CodeFormer on synthetic degraded inputs. Across all 53 HQ examples per profile, delivered compound PNG mean MAE falls by 0.065617 and SSIM rises by 0.131179; lowlight MAE falls by 0.051769 and SSIM rises by 0.087146. Normal-blur SSIM rises on 50/53 cases while its tone guard still fails. Helpful moderate brightening remains acceptable, but these gains do not cancel clear-face changes, incorrect features or other preservation failures.

**The new AdaIN intervention does not improve the previous FP32 result.** All six HQ profile mean PNG MAEs are slightly worse and all six mean SSIMs slightly lower than retained FP32. The changes are small: mean MAE differences range approximately 0.000010–0.000125 and SSIM differences −0.000118 to −0.000564. Failing case counts remain the same. This rules out this unchanged intervention as the preservation fix; it does not establish that all further justified adaptation is impossible.

The 40 older paired cases remain separate by dataset/source/profile. Some photometric measurements are unavailable for masked observed support, so their valid denominators differ from row counts. All 53 HQ measurements per profile are available. Native ChokePoint and QMUL-SurvFace each contribute 24 unpaired DEV crops; their clean-reference PSNR/SSIM/MAE remain null. Input-relative native descriptors measure changes, not correctness. Reserved FINAL pixels were not read.

## 3. Mechanism diagnosis and personal image review

Before Adam, both isolated TRAIN witnesses made all 164 encoder image gradients `None`, while CE/HQ encoder/predictor and image-to-fusion gradients stayed nonzero. Saved fixed-state native/RAW/PNG/token outputs matched between the original and isolated paths, and saved RNG restoration matched. Full gradient/logit/latent vectors were not exported; gradient scalars and source-controlled restoration remain source-bound evidence, not a local neural replay.

After training, encoder update direction cosine to the prior FP32 run is **0.99903**, with L2 changes 0.502029 versus 0.502155. The optimizer was active and the intended route was removed, yet the remaining supervised representation still moved almost the same way. This supports rejecting the assumption that this route alone explained preservation drift. It does not prove CE/HQ alone caused the failure.

A saved-pixel algebraic decomposition covers all 61 clear cases, three states and two views: 366 records and 244 deltas. In the three failed delivered PNG cases, **82.4–92.9% of the added MSE** lies in the global per-channel mean-error component. Residual MSE also increases in all three. This is not a decomposition of the MAE acceptance gate, a pure structure/identity score or proof of an illumination cause. No corrected image, clean-target inference adjustment or replacement gate was generated.

Root personally viewed **all 55 planned sheets** and the separate three-case figure at original-detail 256 pixels. Saved sheet cells have no enhancement or resampling; viewing-tool page overviews can be resized. The personal notes are bound in [visual_review_v1.json](outputs/codeformer_adain_gradient_isolation_v1_return_review/visual_review_v1.json). Independent readback verifies 1,988 source cells, 1,222 output cells, complete 406-case coverage and 15 critical figure cells; that check adds no second personal visual judgment.

Both trained states remove many false wrinkles/rough textures seen in original CodeFormer and improve visibility on several compound/lowlight examples. Their images remain nearly indistinguishable. Some severe examples still change eyes, expression, headwear, glasses or geometry. ChokePoint examples often look clearer but have no aligned clean reference. Several QMUL crops remain distorted, featureless or obstructed and cannot support a useful restoration. These are observation-level findings, not recovered-identity, ethnicity or Zamboanga claims. Earlier metric/sheet captions saying root review was pending are preserved as creation-time history; the completed visual record supersedes that pending status.

## 4. User-requested usable-input threshold

The user identified block-like native crops with too little distinguishable facial structure. **Record a pre-generation, input-only usability requirement:** assess the original crop's usable visible facial relationships, effective/native resolution, blur/blocking, exposure/clipping, coverage and supported pose. If usable information is insufficient, request a clearer, more frontal or less-covered crop instead of generating a face. Darkness or blur alone must not decide usability, and resizing to 256 does not create captured detail.

The [source-bound requirement review](outputs/codeformer_adain_gradient_isolation_v1_return_review/input_usability_requirement_v1.md) shows the current app requires operator review and checks flat/empty support and near-total covering. It **does not yet implement a calibrated automatic structural threshold**. Its Auto blur/noise heuristic selects restoration behavior and is not a usability qualification. A future hard insufficiency decision must remain binding despite restoration On/Off overrides. An uncertain input may receive input-only review; confirmed insufficiency requires a new crop. Coverage alone does not reject a face with enough useful visible context for later separate completion.

Define scores/cutoffs prospectively using DEV input-only labels and independent review, retaining ambiguous cases. The displayed page already has five `insufficient`, one out-of-scope full-profile/object case and two `weak` pre-output labels. No labels were changed from generated appearance. Preserve all historical 406 inputs/results and all failing preservation cohorts; report future accepted/rejected/uncertain counts and refusal rates separately. Do not remove hard cases to manufacture improvement. No numerical cutoff, app change or gate qualification is claimed by this audit.

## 5. Next action and audit limits

The [next design review](outputs/codeformer_adain_gradient_isolation_v1_return_review/next_design_review_v1.md) investigates which retained training objectives change normal-image colour/exposure after preservation anchors, while supplying useful dark-face restoration. Clear training examples already exist: 26 normal-clear and 26 dark-clear exposures, including ten normal clear256 examples, plus all 50 normal behavior-anchor slots. Missing clear data, corrupt teacher arrays, absent optimizers and the removed AdaIN image route alone are unsupported explanations. Earlier rollback/gating evidence also does not establish blanket encoder freezing or brightness gating as the solution.

This is **design review only**. No next packet, recipe, new training or unchanged resume is prepared or authorized by these findings. Any next finite diagnostic must distinguish objective effects using actual bounded responses and exported outputs, retain original initializers/data roles/RAW/PNG gates, and remain manually launched by the user on the L4 with verified transfers and finite limits.

Transport, state, pixel, decomposition and closure boundary fixtures pass. The local audit performed zero neural forwards, gradient queries, optimizer updates, VM connections, cleanup or new Drive transfers. CPU-only saved-tensor inspection used `weights_only=True`; no returned model was instantiated. Two local transport-checker assumption failures and one summary-checker denominator failure remain preserved. Fixes restored the frozen writer's clean512 cap, manifest accounting and available-metric denominator policy; archive bytes, training recipe, quality thresholds and outputs did not change.

Primary receipts reside in [the review directory](outputs/codeformer_adain_gradient_isolation_v1_return_review/): transport, independent state/pixels/scientific readbacks, root visual record, input-usability requirement, design review and final chain. This audit is independent return verification, not the reserved independent FINAL evaluation. User restoration acceptance, calibrated refusal handling, useful native outputs, CodeFormer app integration/Auto/override/PNG verification and independent final review remain incomplete. Separate detector/completion development remains deferred until restoration acceptance, while all seven agreed covering families and original/mask/result downloads stay in the overall goal.
