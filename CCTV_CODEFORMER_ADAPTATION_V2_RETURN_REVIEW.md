# CodeFormer V2 return: real learning, stopped on quality — 15 October 2026

**V2 reached 50/600 optimizer updates and stopped under its declared preservation
rules. The return is intact. The model is not an accepted replacement.**

The visible results are mixed: many severe synthetic faces have fewer conspicuous
texture artifacts than untouched CodeFormer; V2 is usually less bright than the
stopped V1 model. Some clear faces still change tone and detail. Several native
CCTV inputs remain too degraded, turned or obstructed for supported restoration.
These observations come from all 52 planned sheets, covering all 406 cases, plus
the three failed clear controls inspected at the original 256-pixel size.

## What actually ran

The human-launched L4 run performed 50 Adam updates, 200 TRAIN exposures over
100 distinct TRAIN references, 100 normal and 100 matched dark observations.
There were 148 fine-image/LPIPS exposures and 50 frozen-behavior teacher anchors.
This is a short adaptation experiment, not 600 completed updates or 600 epochs.
All 350 declared trainable tensors changed. All 165 frozen tensors stayed exact;
all 350 Adam parameter states record step 50. The snapshot50 and stopped model,
optimizer, RNG and sampler states match exactly. Training therefore did happen.

The measured optimization interval was 125.359 seconds; worker time was about
402.484 seconds. Peak reserved CUDA memory was 8,971,616,256 bytes. The failure
receipt names the clear-preservation guard. The same saved snapshot also fails
the prospective normal-blur photometric guard; this is not a second independent
runtime exception. Export completed after the quality stop.

## Why the declared checks stopped it

The clear-control bound was mean MAE regression at most 0.01 and each case at
most 0.03 relative to unchanged CodeFormer on the identical input. The cohort
means pass, but three individual cases fail in both unmodified RAW and PNG:

| Clear case | RAW MAE regression | PNG MAE regression |
|---|---:|---:|
|comparison_va_asian_04382_clear|+0.038339|+0.036793|
|comparison_va_asian_09306_clear|+0.056840|+0.055460|
|validation_va_ffhq_27440_clear|+0.036869|+0.035243|

Overall, 58/61 clear controls have higher MAE. This does not mean 58 faces are
visually unusable: pixel errors are sensitive to tone. It does establish the
recorded per-case failures and prevents calling the frozen V2 recipe qualified.
The three-case [original-size comparison](outputs/codeformer_adaptation_v2_return_review/visual_sheets/clear_failures_256.png)
shows brighter/softer output and remaining feature differences. Historical source
identifiers such as `asian` are provenance labels, not an ethnicity inference.

On 53 high-quality paired normally exposed blur cases, every case worsens both
absolute luminance bias and coarse RGB error. The PNG mean regressions are
+0.039549 and +0.022058; the worst are +0.070144 and +0.052306. The predeclared
mean 0.01 and worst 0.03 bounds fail. RAW checks fail independently as well.

## Improvements and remaining tradeoffs

These mean per-case PSNR values concern only the 53 aligned HQ references per
synthetic profile. They are not measurements of native CCTV or identity recovery.

| Delivered PNG profile | Original CodeFormer | V2 update50 | Change |
|---|---:|---:|---:|
|Clear|31.416|29.705|−1.711 dB|
|Light|28.233|26.599|−1.634 dB|
|Normal blur|20.485|20.522|+0.037 dB|
|JPEG|30.805|29.207|−1.598 dB|
|Lowlight|15.228|18.341|+3.112 dB|
|Compound dark/blur|12.602|15.655|+3.053 dB|

V2 can visibly reduce false textures and some invented accessories from the
initializer on severe synthetic examples. It retains more natural exposure than
V1 in many rows. Residual smoothing, changed eyes/expression, missing fine detail
and occasional incorrect facial structures remain. For example, the compound
21514 case is less altered than V1, but this single comparison is not a global
preservation or identity claim.

Mean normal-blur PNG luminance is 0.414582 for original CodeFormer, 0.553847 for
V1 update200 and 0.459114 for V2 update50. V2's rise is +11.356/255 versus
V1's +35.513/255. This descriptive reduction is 68.0%; the states have different
update counts and jointly changed recipes. It is not an equal-budget comparison
or proof that one loss caused the difference.

The [user accepts some brightening for clearer visibility](CCTV_CODEFORMER_ADAPTATION_V2_USER_REVIEW_20261015.md).
Helpful exposure improvement must be distinguished from washed-out detail and
altered appearance in future design. The historical guard failures stay intact.
Brightness-sensitive pixel metrics and structure-sensitive descriptive metrics
must remain separate; neither alone establishes useful restoration.

The saved-PNG [structure and contrast supplement](outputs/codeformer_adaptation_v2_return_review/metrics_supplement_v1/INTERPRETATION.md)
confirms genuine paired-image improvements alongside these failures. With the
same retained V1 SSIM formula and the same 53 HQ references per profile:

| Profile | Original CF SSIM | V2 update50 SSIM | Cases higher |
|---|---:|---:|---:|
|Clear|0.917899|0.920830|41/53|
|Light|0.813974|0.806240|31/53|
|Normal blur|0.471176|0.555472|50/53|
|JPEG|0.900561|0.900234|41/53|
|Lowlight|0.527545|0.615528|51/53|
|Compound|0.389666|0.521563|52/53|

Higher SSIM on most blurred/dark paired images supports retaining what the
adaptation learned. It does not establish identity or recovery of unsupported
detail. The three failed clear cases also lose SSIM: 04382 falls 0.895875→0.864012,
09306 falls 0.891817→0.800608, and 27440 falls 0.931321→0.891370. Their issue is
therefore not explained away by a preference for brighter faces. These separate
measurements assign no retrospective pass or new acceptance threshold.

Contrast and target-relative intensity-gradient errors are also saved. On HQ
blur, V2 has lower gradient-vector error than original CF, but gradient magnitude
remains softer than the target. Higher sharpness/contrast alone could indicate
false texture. All 1,074 paired PNG rows across original/V1/V2 remain separated
by source/profile; all 48 native CCTV inputs are excluded from these formulas.

## Native CCTV review

All 24 ChokePoint and 24 QMUL-SurvFace development crops were reviewed separately.
ChokePoint V2 often resembles original CodeFormer closely; V1 tends to be softer.
There is no consistent independently established native improvement. Many tiny,
turned, bright, obstructed or very dark QMUL examples still produce implausible
features, blank facial regions or false textures. Request a clearer, frontal or
less-covered crop when visible structure is insufficient. Do not equate a crisp
synthetic face with a recovered CCTV identity.

Native crops have no aligned clean reference; paired PSNR/SSIM does not apply.
No reserved FINAL pixels were opened and no Zamboanga performance claim follows.
The old 40 paired low-resolution comparisons are reported separately from the
318 HQ synthetic validation inputs and the 48 native cases.

## Independent verification and limitations

The archive, all 2,250 members, their exact namespace and all original packet
bindings were verified before import. All 812 RAW arrays and 812 delivered PNGs
were checked and 1,624 metric rows recomputed. V1/V2 update0 original-CodeFormer
outputs are exactly identical on all 406 RAW arrays and PNGs. All 200 actual
TRAIN inputs and 50 teacher anchors are bound to retained records. The 41 CPU
audit fixtures and seven supplemental metric fixtures pass. No local neural inference, gradient, optimizer, VM launch,
application model change or automatic resume occurred during this review.

The first trajectory audit stopped on 18 unused noise-field hashes. The original
failed audit and source are preserved. The repaired auditor explicitly reports
those 18 cross-runtime hashes as unverified warnings because their exact zero
noise coefficients make their contribution zero. All 148 active fields replay
exactly; stored normal/dark field bindings and actual PNG hashes remain exact.
This does not establish identical random fields across NumPy versions or replay
neural losses/gradients. It changes no V2 recipe or quality threshold.

| Retained evidence | Path |
|---|---|
|Archive SHA-256|`2ab8082f695a11bc4a54f6433a34c31f809cc04f2fa80f288116858d816ef218`|
|Frozen V2 protocol|`029291a2d58d7bc8b33fb99b86774e13db4e5c098bb8af5f508fdc5e24b20183`|
|Transport audit|[archive_import_v1.json](outputs/codeformer_adaptation_v2_return_review/archive_import_v1.json)|
|Pixel audit|[Independent pixel review](outputs/codeformer_adaptation_v2_return_review/metrics_v1/INDEPENDENT_PIXEL_REVIEW.md)|
|Training-state audit|[Repair and evidence receipt](outputs/codeformer_adaptation_v2_return_review/trajectory_v2/SUMMARY_AND_REPAIR_RECEIPT.json)|
|Independent cross-review|[Status readback](outputs/codeformer_adaptation_v2_return_review/independent_status_readback_v1.json)|
|All visual pages|[52-page index](outputs/codeformer_adaptation_v2_return_review/visual_sheets/INDEX.md)|

## Next decision

Keep untouched CodeFormer as benchmark and preserve V1/V2 stopped states. Do not
resume V2 unchanged or remove its safeguards to force more epochs. Before another
training recipe, diagnose the original-weight training/evaluation path and the
loss gradients: the first saved clear behavior anchor is already nonzero before
the first optimizer step. That is a testable path/precision difference, not proof
that it caused the 50-update brightness change. The diagnostic must run manually
on L4 with zero optimizer updates, finite calls/time/storage and an independent
returned-evidence audit. Any subsequent training design must separately evaluate
helpful brightness, visible detail and supported appearance preservation.

Restoration is still the first milestone for model improvement. Completion and
covering-detector development stay deferred. User acceptance, independent final
evaluation and the full application workflow remain outstanding.
