# FP32 CodeFormer return: trained, but preservation still stops it — 16 October 2026

**The return is intact. Training performed all 50 planned Adam updates. FP32
fixed the initial numerical mismatch, but did not resolve the preservation
tradeoff. This checkpoint remains a stopped experiment.**

Root inspected all 52 comparison sheets covering 406 cases, plus the three
failed clear controls at their actual 256-pixel size. Many blurred/dark paired
outputs retain the useful improvements seen in V2. FP32 and V2 are visually
very similar; there is no consistent visible FP32 improvement over V2.

[All comparison sheets](outputs/codeformer_fp32_comparison_v1_return_review/visual_sheets/INDEX.md)
and [the three clear failures at 256px](outputs/codeformer_fp32_comparison_v1_return_review/visual_sheets/clear_failures_256.png)
are available for user review. Columns are input, paired target/native note,
original CodeFormer, BF16 V2 update 50 and FP32 update 50. Sheets use only
256-to-192 Lanczos display resizing; the failure detail retains exact 256 PNGs.

## What ran and what passed verification

This human-launched L4 comparison started from untouched pretrained CodeFormer.
It completed 50 updates,200 TRAIN exposures and 100 photographic reference IDs.
These are updates and exposures, not 50 epochs or 100 independently proven people.
All 350 declared trainable tensors changed; all 165 frozen tensors stayed exact.
All 350 Adam parameter states record step 50. Snapshot 50 and the stopped state
agree on model, Adam, committed RNG and sampler. Real training therefore occurred.

The actual 200 input/clean-target bindings, first 50 learning rates, sampler,
FP32 dtypes and all 2,057 operation reservations match the frozen recipe.
All 50 used behavior RGB/logit targets match the retained original targets.
The separately saved 50 online FP32 pairs also match those arrays in this run.
All 391 saved HQ feature arrays match their frozen dictionary lookups.
The initial behavior image error is exactly 0 and token KL is
1.5076004444836144e-08, within the predeclared 1e-6 bound before the first Adam step.

Measured optimization took 214.235 seconds; the worker took 651.765 seconds and
the supervised run 668.483 seconds. Peak reserved CUDA memory was 8,453,619,712
bytes. Runtime and resource caps passed. The stop was a declared quality check
after the saved 50-update snapshot, not an optimizer, CUDA or transfer crash.
The finite cap was 50 even if quality had passed; the 600 schedule horizon was
retained only to match the original first 50 learning rates/data positions.

## Why it stopped

The clear guard compares error against unchanged CodeFormer on identical inputs:
mean MAE regression at most 0.01 and each case at most 0.03, independently on RAW
and delivered PNG. The means pass: RAW+0.008269 and PNG+0.007705. The same three
individual cases fail as in V2:

| Clear control | RAW MAE regression | PNG MAE regression |
|---|---:|---:|
|comparison_va_asian_04382_clear|+0.038422|+0.036878|
|comparison_va_asian_09306_clear|+0.057173|+0.055788|
|validation_va_ffhq_27440_clear|+0.037000|+0.035366|

The source label `asian` is retained provenance, not an inferred ethnicity.
The first two targets belong to the older low-resolution comparison source;
they are not new HQ references. The third is an HQ validation control.

All three also have lower delivered-PNG SSIM against their respective targets:
0.895875→0.863565,0.891817→0.799865 and 0.931321→0.888621. At 256px both adaptations
brighten/smooth these faces;09306 loses some cheek/nose contrast, and 27440 has
flatter skin and softer eye/cheek texture. SSIM is sensitive to tone as well
as structure; these findings do not prove an identity change or make every
clear output worse. Mean HQ clear SSIM actually increases, with 42/53 higher.

The 53 normally exposed HQ blur cases also fail the unchanged photometric guard.
PNG absolute-luminance-bias regression is mean+0.040048/worst+0.070932; coarse-RGB
regression is mean+0.022442/worst+0.053083. Both exceed mean 0.01/worst 0.03 limits.
RAW checks fail independently. Mean PNG luminance rises 0.414582→0.459613
(target 0.416908), about+11.48/255, versus BF16's0.459114. This is slightly more
normal-image brightening, not a resolved FP32 exposure correction.

The failure receipt names clear preservation first. Normal-blur failure is a
separate independently checked property of the same snapshot, not another
runtime exception. Successful export packages the failed evidence; it does
not override either quality stop. No historical 1% DGP feature gate applies here.

## Real improvements retained alongside those failures

These are mean per-case delivered-PNG scores on the same 53 aligned HQ references
per synthetic profile. They are not CCTV, identity or Zamboanga measurements.
PSNR and the retained RGB/7-window observed-support SSIM formula are reported
separately from the preservation gates.

| HQ synthetic profile | Original PSNR / SSIM | BF16 V2 u50 PSNR / SSIM | FP32 u50 PSNR / SSIM |
|---|---:|---:|---:|
|Clear|31.416 /0.917899|29.705 /0.920830|29.723 /0.920696|
|Light|28.233 /0.813974|26.599 /0.806240|26.597 /0.805757|
|Blur|20.485 /0.471176|20.522 /0.555472|20.495 /0.554298|
|JPEG|30.805 /0.900561|29.207 /0.900234|29.217 /0.900000|
|Lowlight|15.229 /0.527545|18.341 /0.615528|18.361 /0.615127|
|Compound|12.602 /0.389666|15.655 /0.521563|15.676 /0.521100|

Against original CodeFormer, FP32 has higher SSIM on 50/53 blur,51/53 lowlight
and 52/53 compound cases. Root's sheets show reduced exaggerated skin texture,
false jewelry and invented glasses on several severe synthetic observations.
Some dark faces become easier to read. Residual softness, changed expressions
or eye/mouth geometry and false accessory detail remain on difficult examples.

Against BF16 V2 at the same 50-update budget, differences are small and mixed:
blur PSNR−0.0266dB and SSIM−0.001174; lowlight PSNR+0.0199dB and SSIM−0.000401.
The old V2 HQ cache was not exported, so exact old/new HQ encoder-target cache
equality remains unproven. This is a closely controlled precision comparison,
not proof that precision is the sole causal difference or that FP32 never helps.

The user's accepted preference for moderate helpful brightening is retained.
Improved visibility and paired blur structure are real benefits worth retaining.
They do not waive clear appearance, contrast or unchanged preservation checks.

## Native CCTV and visual scope

All 24 ChokePoint and 24 QMUL-SurvFace DEV crops were reviewed separately, alongside
318 HQ synthetic and 40 older paired comparisons. No aligned clean native targets
exist here; native PSNR/SSIM/reference fidelity fields remain null.

ChokePoint outputs become a little brighter/softer than original CodeFormer,
with broadly similar contours. FP32 and BF16 look closely matched. This review
does not establish a consistent native fidelity improvement. Many QMUL crops
remain too tiny, overexposed, turned, obstructed or dark for supported restoration;
blank features, false texture and implausible structure persist. Request clearer,
more frontal or less-covered input when usable visible structure is insufficient.
No ethnicity, recovered identity or Zamboanga performance claim follows.

All 406 baseline RAW arrays and PNGs match the retained V2 update 0 outputs exactly.
All 812 new and 812 retained RAW/PNG pairs passed hash, floor-conversion and protected
padding checks. The pixel audit recomputed 3,248 source/profile/view metric rows.
All 52 page hashes and 1,982 rendered source cells were independently checked.
Root's visual review is complete; user acceptance and independent FINAL review
remain outstanding. No reserved FINAL pixels were read.

## Decision and next design review

The tested assumption that precision alignment would be sufficient is rejected.
Do not resume this failed recipe, append epochs, raise LR blindly or relax gates.
The current run is learning, but its shared update still trades useful dark/blur
changes against normal-face appearance. That tradeoff is the strongest remaining
hypothesis, not an isolated causal proof from this FP32 run.

Only 50/200 observations receive the original-CodeFormer behavior penalties:
26 normal clear-B and 24 normal light-B controls. The 50 normal-severe and all 100
dark observations have no behavior anchors; encoder/fusion updates are shared.
Earlier fixed-code interventions improved normal cases but removed helpful dark
brightness when original fusion was restored. This argues against blanket fusion
rollback or indiscriminately anchoring dark outputs to the old dark result.
Logged loss magnitudes alone do not establish which gradients dominate.

The user answered the required post-three-attempt design discussion:
**“great news, apply the best approach.”** The selected direction is to review
input-conditioned, bounded exposure/fusion adaptation that protects facial detail,
with delivered 256 preservation explicitly accounted for. Start with a controlled
test of conditioning; do not change LR, loss coverage, all trainable paths and
architecture simultaneously. Native 512 supervision versus delivered 256 filtering
and clipping remains a declared secondary issue to isolate separately.

[Next design review](CCTV_CODEFORMER_POST_FP32_DESIGN_REVIEW_V1.md).
This is a design review, not a prepared executable, verified transfer or next
launch command. Its remaining conditioning/resource decisions must be resolved
before another finite manual-L4 packet. The original model and every failed
state remain retained. No local neural inference, gradient, optimizer, VM action,
cleanup, application adoption or new training occurred in this return audit.
Restoration stays first; completion/detector development is deferred. The full
thesis goal remains active and incomplete.

## Evidence and exact bindings

| Evidence | Location |
|---|---|
|Strict transport/import|[transport_v1.json](outputs/codeformer_fp32_comparison_v1_return_review/transport_v1.json)|
|State/Adam/data/call audit|[trajectory_audit.json](outputs/codeformer_fp32_comparison_v1_return_review/trajectory_v1/trajectory_audit.json)|
|Independent metrics and gates|[metrics_audit.json](outputs/codeformer_fp32_comparison_v1_return_review/metrics_v1/metrics_audit.json), [guards.json](outputs/codeformer_fp32_comparison_v1_return_review/metrics_v1/guards.json)|
|Source-separated pixel readback|[SCIENTIFIC_READBACK.md](outputs/codeformer_fp32_comparison_v1_return_review/metrics_v1/SCIENTIFIC_READBACK.md)|
|Visual and next-design evidence|[visual_review_v1.json](outputs/codeformer_fp32_comparison_v1_return_review/visual_review_v1.json), [state/loss review](outputs/codeformer_fp32_comparison_v1_return_review/next_design_state_review_v1.md)|

Results archive:2,848,638,408 bytes;
SHA-256 `8b2361147a1d12b7d2bad8a7d3a8965e0e90c077bb1cdbb4ad7378ad8703ef78`.
All 6,501 regular members/3,120,598,370 payload bytes were hash-verified before
exclusive import. The audit separately verified all 1,925 original packet
bindings, including the 1,399 retained parent dependencies. Protocol SHA-256:
`8acff86e31c3d135994fdac6ed781374e036150ce1ae2110db82ca0dc5a3aff1`.
Stopped checkpoint:855,229,944 bytes;
SHA-256 `133c5a1e15e069bdd2abda4ebf818d3f4ff59b966677762753e894ff42e2583d`.

The 50 CPU audit fixtures pass:30 transport,12 trajectory and 8 pixel-metric tests.
Neural losses/gradients and HQ encoder nearest-code selection were not rerun;
the audits check recorded scalar reductions, state, saved indices/features,
inputs and output pixels. Runtime captures are bound frozen-source evidence,
not an independent live probe. A supplemental summary initially assumed an
incorrect trajectory JSON field; its source/repair record is retained and no
scientific metric or worker changed. Successful audits establish evidence
consistency, not useful-model or thesis completion.
