# CodeFormer restoration adaptation V2 — prospective design

Status: **DESIGN ONLY / REQUIRES INDEPENDENT REVIEW BEFORE IMPLEMENTATION**.
Revised15 October 2026 after independent design review; the original e0309142
draft is retained in `outputs/codeformer_adaptation_v2_design_review/before_revision_v1/`.
No V2 executable, protocol hash, transfer archive, training,
gradient or VM launch exists by this document. This proposes one changed finite
experiment; it does not authorize continuing the quality-stopped V1 trajectory.

## Question and evidence

Can balanced normal/dark TRAIN pairs, direct coarse photometric supervision at
every generated resolution, and preservation of pretrained behavior on TRAIN
controls retain useful lowlight gains while avoiding V1's normal-blur brightening
and clear-appearance changes?

V1 actually updated all 350 intended trainable tensors with the 165 frozen
restoration tensors unchanged, then stopped at200 updates. Only20/162 normal-blur
TRAIN exposures received image L1/LPIPS, compared with188/188 lowlight exposures.
All 53 normally exposed HQ blur validation images brightened; their mean output
luminance rose0.41458→0.55385 against a0.41691 clean-reference mean. Lowlight
received a similar+0.13878 shift, where it usually helped. One clear comparison
also violated both RAW and PNG preservation limits. These results support fixing
supervision before adding a detector, router or new architecture path.

The root's complete52-sheet review found reduced false textures/false glasses
in several severe paired cases, alongside brightness/appearance regressions;
native ChokePoint often softened, and many QMUL inputs lacked useful information.
Those visual observations do not establish native identity recovery. See
[independent trajectory review](outputs/codeformer_adaptation_v1_return_review/training_review/INDEPENDENT_TRAINING_REVIEW_V1.md)
and [saved-pixel diagnosis](outputs/codeformer_adaptation_v1_return_review/post_v1_review/POST_V1_DESIGN_REVIEW.md).

The user subsequently judged most V1 outputs great or personally better than
original CodeFormer, while explicitly observing amplified brightness. This is
positive development feedback supporting this next controlled recipe; brightness
and preservation remain unresolved, and the feedback is not full model acceptance.

## Immutable model and evidence roles

Initialize a separate candidate from the untouched official restoration asset
`outputs/codeformer_restoration_pretrained_v1/codeformer.pth`, SHA-256
`1009e537e0c2a07d4cabce6355f53cb66767cd4b4297ec7a4a64ca4b8a5684b7`.
Do not initialize from V1 update200. Retain all V1 checkpoints, outputs, failed
gates and archives; no app/default checkpoint is changed.

Keep the same six trainable groups: encoder, position embedding, feature
embedding, transformer layers, token head and four SFT fusion modules. Keep
restoration quantizer/generator, official HQ teacher, original CodeFormer behavior
teacher and LPIPS completely frozen. No GAN, discriminator, EMA, new adapter,
detector, exposure router or source-specific exception is introduced.

Inference remains RGB256→bilinear512→CodeFormer→clippedRGB→bilinear256,
`w=1`, `adain=True`, no alignment change and no display enhancement. Training
predicts unclipped normalized512 output; clean512 is derived directly from the
audited1024photograph. The supported clean target roles remain HQ-teacher token
indices and restoration-codebook features at those indices. Retain the released
codebook compatibility receipt and immutable asset fingerprints.

| Role | Fixed scope |
|---|---|
| Optimization |Same 391 approved genuine 1024 FFHQ TRAIN sources; all rights/provenance retained|
| Repeated development |Same 53 held-out historical validation sources; never optimizer data|
| Fixed comparison |Same 40 prior synthetic views and 48 native CCTV DEV crops|
| Final evaluation |Existing reserved identities remain excluded from pixels, training and selection|
| Prior overlap |FFHQ pretraining and repeatedly exposed DEV limits remain explicit; no unseen-subject claim|

No additional low-resolution `asian_faces` reference is promoted to clean512
ground truth. Do not train on the failed DEV face, its output, its source label
or a manual mask. Native CCTV remains unpaired. No ethnicity or Zamboanga
performance inference is introduced. All model/data notices remain separate.

## Exact finite TRAIN sampling

Use a single seeded sampler (`20261015`) over the 391 TRAIN IDs, shuffled without
replacement. Each optimizer update consumes two distinct IDs, A and B. If a
shuffle-boundary draw duplicates A, deterministically rotate the new permutation
until B differs and log that rotation. No image/identity is selected by DEV error.

Each update accumulates four microbatches of one image, in this fixed order:

| Slot | Reference | Camera role | Loss role |
|---|---|---|---|
|1|A|Severe normal exposure|Clean-code/latent +coarse photometry; detail loss only if edge≥32|
|2|A|Matched severe dark exposure|Same eligibility and clean target as slot1|
|3|B|Normal clear/light resolution control|All clean losses +pretrained-behavior anchors|
|4|B|Matched dark version of slot3|All clean losses; no pretrained dark-image anchor|

For zero-based update `u=update-1`, set A edge to `[24,28,32,40][u%4]` and
B variant to the six-element list below at `[(u//4)%6]`. This gives the full
24-combination cycle25 times over600 updates:150 updates per A edge and100 per
B variant. The first50-update snapshot covers two complete cycles plus two
combinations; no claim of full first50 balance is made.

For A, draw Gaussian sigma uniformly[0.6,2.0] in256-pixel units,
JPEG quality uniformly among integers35–65, shot coefficient uniformly
[0.0004,0.0012] and read sigma uniformly[0.001,0.003]. No motion kernel or warp.
The two observations share source, edge, blur, JPEG, noise parameters and random
noise fields; only exposure changes from1.0 to a draw in[0.15,0.65]. Apply shot/read
noise after exposure using the exact linear-light equations below. The
signal-dependent noise therefore changes with exposure, while the sampled noise
field remains matched. Both observations receive the same clean512 target.

For B, use these six normal variants:
`clear256, clear128, clear64, light256, light128, light64`,100 updates each.
Clear variants have sigma0, no JPEG/noise, and normal exposure1. Light variants draw
sigma[0.5,1.5], RGB noise[1,12]/255 and integer JPEG quality35–95. Down/up sampling
uses the exact V2 order below; no crop or face alignment changes.
Its dark partner shares all those camera draws and independently keyed fields, changing
only linear exposure to a draw in[0.20,0.65]. No additional shot/read noise is
introduced in the B pair. The64/128/256variants explicitly expose the preservation
anchor to input-resolution variation without forcing imitation of CodeFormer's
more ambiguous24–40pixel hallucinations. A clear64/128 variant still downsamples
and upsamples, and its dark partner still applies exposure; there is no early
return for a clear profile.

### Exact paired camera and random fields

Each update uses deterministic NumPy `Generator(PCG64(seed64))` instances.
For pair reference ID `r`, role `A` or `B`, and purpose `parameters`, `linear_noise`
or `rgb_noise`, derive `seed64` from the first eight bytes of
`SHA256(UTF8("codeformer-adaptation-v2|20261015|" + str(u) + "|" + r + "|" + role
+ "|" + purpose))`, interpreted little-endian unsigned. Do not use Python hash,
global NumPy state, one conditionally consumed RNG, or normal/dark in this key.
The parameters RNG draws in the order sigma, JPEG quality, shot coefficient,
read sigma, dark exposure for A; sigma, RGB noise sigma, JPEG quality, dark
exposure for B light; only dark exposure for B clear. All uniform ranges are
half-open except the integer JPEG range whose stated endpoints are inclusive.
Discrete edge/variant choices use the cycle above, not RNG draws.

Start from verified source1024→Pillow BILINEAR256 RGB8, then float32 `/255`.
For both pair members use the following fixed operations:

1. Gaussian blur with OpenCV BORDER_REPLICATE if sigma>0, then INTER_AREA
   downsample to the declared square edge (identity at256).
2. Invert sRGB: `L=where(C<=.04045,C/12.92,((C+.055)/1.055)**2.4)`.
   Apply `S=e*L`, with normal `e=1` and the recorded dark exposure for its partner.
3. Precompute `Nlinear` and `Nrgb` as float64 standard-normal arrays of shape
   `(edge,edge,3)` from their separate keyed generators, even when a coefficient
   is zero. Reuse each exact field within the pair. Set
   `Lnoisy=clip(S+Nlinear*sqrt(shot*S+read**2),0,1)`; here A has the declared
   shot/read parameters and B has both zero. Identical noise fields do not imply
   identical signal-dependent noise values after exposure.
4. Encode sRGB: `Cnoisy=where(Lnoisy<=.0031308,12.92*Lnoisy,
   1.055*Lnoisy**(1/2.4)-.055)`, then add `Nrgb*rgb_sigma_255/255`.
   A and B clear have RGB sigma0. Do not skip the linear/sRGB path at exposure1.
5. Quantize using `floor(clip(Cnoisy,0,1)*255).astype(uint8)`, no rounding-to-nearest.
   Apply recorded OpenCV JPEG encode/decode with RGB↔BGR conversions if enabled,
   then INTER_CUBIC upsample uint8 to256 if edge<256. Convert final RGB8 to
   contiguous HWC float32 `/255` for the model input. No output warp or cropping.

Record NumPy/Pillow/OpenCV versions, full seed keys/seed64s, SHA256 of each
contiguous float64 noise-field byte sequence, pair ID/member, full parameters,
RGB8 and float32 input hashes, and direct clean512/reference hashes. Floating
rounding or codec identity across runtime versions is not assumed. The final
same-VM generated input PNGs and records are evidence, not regenerated local GT.

All draws use recorded local seeds, never a global RNG hidden inside preparation.
Save camera parameters, input-pixel hashes, reference/clean-target hashes, role,
paired-slot linkage, eligibility and exact runtime versions for every exposure.
TRAIN images are generated on the VM from verified originals; cross-platform
OpenCV output identity is not assumed.

At 600 updates: **2400 image exposures**,1200 reference selections,391 unique
available TRAIN references. The reference sampler covers three full passes plus
27selections:27 IDs selected four times and 364 IDs three times, each selection
creating two exposures. Thus image-exposure equivalents are2400/391=6.138,
while photographic selection passes are1200/391=3.069. Neither number is the
initializer's lifetime epoch count. At every update exactly 50% of exposures are
normal and 50% dark. Half of all exposures are B controls/dark-controls. A detail
loss is eligible for exactly two of its four edge positions; no random imbalance
in the normal/dark loss eligibility is possible.

## Exact prospective losses and reductions

Let `z` be candidate raw normalized512 output and `P=(z+1)/2` its **unclipped**
RGB value. Let `T` be verified clean512 RGB[0,1]. All listed means reduce equally
over their stated channels/pixels/tokens; microbatch losses are divided by 4
before backward. Do not renormalize loss weights by the number of eligible slots.
This retains the meaning of missing detail terms instead of secretly amplifying
the other samples.

| Term | Weight | Definition and eligibility |
|---|---:|---|
|Clean token CE|0.5|FP32 mean over256tokens, HQ-teacher indices; all four slots|
|Clean latent MSE|1.0|FP32 mean encoder feature versus frozen restoration-codebook features; all slots|
|Native RGB detail L1|1.0|`mean(abs(P−T))`; all slots with observation edge≥32|
|LPIPS v0.1/VGG16|1.0|Standard FP32 `LPIPS(z,2T−1)`; same edge≥32 eligibility|
|Coarse RGB photometry|**4.0**|`mean(abs(pool32(P)−pool32(T)))`; **all slots**, including24/28pixel observations|

`pool32` is exact nonoverlapping average pooling with kernel/stride32 on512,
producing16×16RGB grids, no padding. It supervises surviving color/illumination
without requiring high-frequency texture where the observation is insufficient.
No output clipping precedes this loss, so excessive-range predictions retain
photometric gradients. A0.14RGB bias contributes approximately0.56 to the weighted
term, making V1's measured failure consequential without changing the code prior.
Weight4 is a prospective engineering choice, not an optimized/verified optimum.

Add **two behavior-preservation terms only to normal B (slot3)**:

| Anchor | Weight | Exact definition |
|---|---:|---|
|Delivered-image anchor|**2.0**|Mean L1 between candidate and untouched teacher's clipped/bilinear256RGB outputs|
|Token-distribution anchor|**0.1**|Temperature2, `4*mean_tokens(KL(teacher_probs || candidate_probs))`, sum over 1024 codes|

The teacher sees exactly the same B input with `w=1`, AdaINTrue and FP32 evaluation.
Use the teacher's own native forward logits and matching delivered256 image.
Compute teacher outputs under `no_grad()`, then detach/clone ordinary tensors as
needed for the differentiable candidate loss; do not pass unsafe inference-mode
tensors into backward. Teacher targets are pretrained-behavior anchors, **not
clean identity truth**. Clean-reference supervision remains active on B, so
teacher errors are not made the sole objective. No anchor is applied to dark
slots or severe A, preserving room for the improvements sought there.

For KL, calculate `log_softmax(candidate_logits/2)` and
`softmax(teacher_logits/2)` in FP32, sum per-token KL over the code axis, average
over batch/tokens, multiply by 4, then by 0.1. Do not use a reduction that introduces
an extra factor of 256. Record raw and weighted terms separately. The general
teacher-preservation rationale is supported by [Learning without Forgetting](https://arxiv.org/abs/1606.09282)
and [knowledge distillation](https://arxiv.org/abs/1503.02531); these sources do not
establish the correct weights or restoration benefit for this experiment.

This is a joint changed-recipe comparison: matched sampling, coarse supervision,
behavior anchors, the600-update cosine horizon and the new50 snapshot change
together. V1 used a1200-update horizon and snapshots0/200/600/1200; therefore even
the common200 snapshot has a different learning rate. This is not a causal
ablation of one loss or of sampling. Architecture and inference operations stay
unchanged. Generator operations
stay differentiable despite frozen parameters. Hard token selection remains;
CE/KL supervise logits directly, while coarse/detail/anchor image losses reach
fusion and the AdaIN encoder-moment route. No straight-through token replacement
is introduced. The design does not claim that AdaIN alone caused V1's failure.

## Schedule, outputs and finite resources

| Control | Prospective value |
|---|---|
|Optimizer trajectory|Maximum600 updates; microbatch1, accumulate4; no automatic resume|
|Adam|Betas(0.9,0.99), epsilon1e-8, weight decay0; no gradient clipping|
|LR|Encoder/token1e-5; fusion2e-5;50-update linear warmup, cosine to2e-6 at 600|
|Precision|Candidate BF16; teacher/token assignment/CE/KL/latent reductions/LPIPS/photometry FP32; no GradScaler|
|Snapshots|0,50,200,600; same406cases and61clear controls; all snapshots retained|
|Initialization/cache cap|600seconds; no forced repeated gradient-diagnostic loop|
|Measured benchmark|First20updates,600seconds cap; retained in trajectory; project600 updates≤3600seconds|
|Optimization wall cap|3600seconds, including online behavior-teacher forwards|
|Each snapshot cap|600seconds; no cohort dropping to fit time|
|Whole worker / export|7200seconds worker plus180seconds termination grace;600seconds export plus30seconds grace|
|Storage|At least20GiB free **after** installation; final result payload≤7GiB, archive≤7.1GiB, checkpoint temporary≤1GiB, free reserve≥4GiB|
|GPU memory|22GiB reserved-memory ceiling, fresh idle NVIDIA L4 check|

V1 measured approximately1.71seconds/update plus snapshot overhead, but V2 adds
one frozen behavior-teacher forward and losses per update. A preliminary planning
range is20–40minutes including snapshots/export on the existing L4; this is not a
runtime promise. The measured first20-update projection overrides that estimate.
Stop/export if the fixed schedule cannot fit; do not extend caps automatically.

Maximum new training forwards/backwards are2400, behavior-teacher forwards 600,
HQ cache forwards 391, snapshot forwards 1624, and initialization parity forwards
two candidate/two original. Permit at most one declared LPIPS input-gradient
prerequisite on the manual VM, as in V1; backward cap 2400 and optimizer cap 600.
All counters distinguish cache, teacher, training and evaluation calls. No
additional sweep or retry is hidden inside the worker.

Bind and reuse **exact V1 VM-saved318 validation input PNGs**, their pair records,
the same53 retained targets, and the88 comparison inputs/masks/targets. Do not
regenerate DEV degradations for V2. Update0 must match untouched CodeFormer on
the prospective VM; numerical/platform variation must be reported if comparison
with saved V1 baseline differs. Baseline quality guards use same-VM V2 update0.

Save update0/50/200/600 full candidate+Adam+committed RNG/sampler states; retain
stopped/interrupted state and partial outputs. No automatic resume follows a
quality stop. Record per-tensor gradients and changes during the first 20 updates
and frozen-state fingerprints at snapshots/export. Check module coverage without
requiring every individual tensor to be nonzero in one batch.

Persist each behavior teacher's exact delivered256RAW tensor and FP32 token
logits used for slot3, plus input hash/parameters and teacher state fingerprint.
This may add approximately 1.1 GB at 600 updates; full original teacher/data/weights
are not duplicated in result exports. The pre-freeze modeled ledger is:

| Evidence allocation | Bytes |
|---|---:|
|Update0/50/200/600 states plus one retained stopped state, based on V1 sizes|3,797,526,165|
|600 delivered256 float32 teacher anchors and256×1024 float32 logits, payload|1,101,004,800|
|Four406-case RAW256 snapshots|1,277,373,440|
|Four406-case RGB PNGs at bounded198,656 bytes each|322,617,344|
|Modeled subtotal before headers/cache/inputs/records|6,498,521,749 (6.052GiB)|

Enforce final output/result payload cap7GiB (`7516192768` bytes), including
anchors, cache, states, inputs, PNGs, records and logs; archive cap7.1GiB
(`7623566951` bytes), checkpoint-temporary cap1GiB (`1073741824` bytes), and
runtime/export free reserve4GiB (`4294967296` bytes). The combined allocation is
`20508468839` bytes (19.1GiB), below startup20GiB (`21474836480` bytes).
Checkpoint temporary means one atomic in-flight state; after rename it is counted
only in final payload. Audit the prospective ledger before freeze, enforce actual
payload size at safe boundaries, check checkpoint reservation before each save,
and preflight actual final payload+TAR/gzip allowance+reserve before exporting.
The archive writer must stop before exceeding its cap and retain a labelled
partial archive on failure; do not accumulate repeat failed exports. Result
exports never duplicate immutable source photographs, original model weights,
full teacher weights or unbounded training image dumps. Bind ordinary RGB PNG
encoding with no ancillary metadata and enforce the per-file PNG bound. Retain
all license, source, provenance and failure companions. These storage stops are
resource failures, not grounds for dropping cases or deleting research evidence.

## Prospective stops and qualification boundaries

Keep **all V1 clear RAW/PNG guards unchanged** at each nonzero snapshot: across
the same 61 clear cases, stop/export if mean candidate-minus-update0 clean MAE is
greater than 0.01 or any per-case regression is greater than 0.03 RGB[0,1]. Compare
RAW and delivered PNG separately; equality to the bound passes. Preserve failures.

Add a prospective **normal-blur photometric abort**, separately for RAW and PNG,
over the fixed 53 HQ `blur` views. Define RGB-derived luminance
`Y=.2126R+.7152G+.0722B`, without gamma inversion. For each case define
`bias_i(model)=abs(mean(Y(model_i)−Y(target_i)))`, and explicitly
`delta_bias_i=bias_i(candidate)−bias_i(update0)`. Stop if
`mean_i(delta_bias_i)>0.01` or `max_i(delta_bias_i)>0.03`. Also define
`coarse_i=mean(abs(pool16(candidate256_i)−pool16(target256_i)))` over a16×16RGB
grid (kernel/stride16), and `delta_coarse_i=coarse_i(candidate)−coarse_i(update0)`.
Stop if `mean_i(delta_coarse_i)>0.01` or `max_i(delta_coarse_i)>0.03`.
Equality passes. Both metrics use all53 complete aligned paired targets and full
image support; native CCTV is excluded from clean-target formulas. These bounds
directly reject V1-scale exposure leakage;
they are additional prospective safeguards, not a replacement of old gates or
proof of useful structure. Report luminance contrast and PSNR/SSIM, but introduce
no new claimed structural-pass percentage.

Retain all nonfinite, frozen-mutation, missing-learning-path, active-workload,
space, VRAM, timing and manual-signal stops. Honor signals at safe initialization,
per-case and microbatch boundaries, with optimizer-in-flight state marked unsafe.
A stopped export is evidence completion, never successful model qualification.

At returned review, report each degradation/source separately. The changed recipe
must show normal-blur improvements **and** retained useful lowlight/compound
benefits relative to untouched CodeFormer, with preserved clear appearance.
No average or new coarse guard can overrule loss of visible eyes, brows, nose,
mouth, jaw, clear glasses, hair, brightness or expression. Recompute the retained
historical structure/preservation analyses where applicable and leave their
previous failures intact; passing these new abort guards is not a 1% structure
pass. No candidate is accepted merely because600 updates finish.

Independently audit the return archive, saved anchors, state/Adam/RNG, actual
exposures, counters and output conversions; recompute RAW/PNG metrics and inspect
all planned sheets. Native CCTV remains qualitative/unpaired, including
insufficient/profile/covered exclusions. User acceptance of visible restoration
improvement and independent final review remain required before integration and
bundled inline Playwright Auto/override/PNG/app verification. Completion and
covering-family work stays deferred until restoration acceptance.

## Review before implementation

An independent reviewer must check exact loss scaling/eligibility, matched-pair
construction and sampler counts, teacher gradient isolation, finite resource byte
budget, all old/new guards and immutable DEV reuse. Resolve implementation
questions before producing code/transfer hashes. A later implementation must
remain a separate V2 tree; retain the stopped V1 unchanged. No executable or VM
commands should be issued from this unreviewed design document.
