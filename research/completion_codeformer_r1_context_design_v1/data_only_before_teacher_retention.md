# Completion R1: data-coverage-only continuation design

25 October 2026. FINALIZED DESIGN FOR INDEPENDENT REVIEW. No trainer, transfer
packet, model call, gradient, app edit or VM operation is prepared here. The user
prefers R1 after same-image/mask review. Keep it as the next working completion
baseline, with original CodeFormer and every prior state/failure retained.
Working selection is not full-family or hidden-identity qualification.

## Evidence that justifies this direction

All40 original1024 sources and their exact256/512 targets were independently
rehashed. The original32FIT/eight adaptiveTRAIN monitor IDs are separate; source
separation does not establish independent identities or eliminate unknown FFHQ
pretraining overlap. Every source retains originalTRAIN role and per-photo terms:
21 attribution,18 noncommercial-attribution,oneCC0. No ethnicity, native CCTV or
Zamboanga-performance claim follows from these sources.

R1 exposed each128 FIT reference/mask pair once per learning route, with32 Adam
steps per route. This is a small bounded pass, not many large-dataset epochs.
The128 mask files collapse to four shapes repeated over all32 sources:
lowerface15,277pixels; eyeband8,640; singlelens2,219; cheek/object2,908.
There are zero strict-gray FIT sources and zero strict-gray MONITOR sources;
there was no simulated degraded-context training. Missing condition and shape
coverage is therefore a demonstrated gap before replacing the architecture.

The two working routes changed all332 intended tensors, while165 generator/
codebook tensors stayed exact. Discrete code selection and detached fusion inputs
mean fusionRGB loss does not train code prediction; preserve the explicit routes.
Mean CE1.794, latentMSE0.02898, holeRGBL10.05549 and boundaryL10.02735 describe
scales, not relative gradient influence. All32 code steps reached clip1
(preclip5.659–23.634); fusion did not (0.233–0.671). Raising LR/removing clipping
is not the evidence-led first action. First/last training blocks use different
pairs and cannot be called a matched quality improvement. Retain the independent
6.1853% monitor gain and six individual regressions separately.

All four scarf cases preserve588,336 source RGB bytes outside their fixed masks;
their outside3pixel rings and raw coordinates are exact too. Knit-scarf masks
have14,259pixels, scarf/gloves18,757. R1-versus-original changes inside support
average8.636/10.710 levels for original/degraded knit scarf,9.663/9.399 for
scarf/gloves. These locate copied/generated support, not cloth or anatomy accuracy.
No uncovered clean reference or covering-segmentation truth exists for these
public photographs. Root's visual review records copied covering fragments
outside support and cloth/beard-like generation defects inside. Keep masks fixed;
reviewed mask correction is separate assisted evidence, not a model improvement.

## Comparison and fixed data

Continue from verified R1 logical55feaa296...4122d6 and candidate7ae248673...da37e0,
with frozen R1 as the matched control on identical inputs/masks. Original
CodeFormer remains the historical baseline. Use one new learning arm; no GAN,
architecture, generator-unfreeze, optimizer-reset or LR sweep.

Keep the exact32FIT/eight adaptiveTRAIN monitor sources. Plan256FIT cases:
128 original byte-identical clear anchors, plus128 augmentation pairs, four new
pairs per FIT source. Rotate conditions by source/geometry index to give32
gray-clear,32 RGB-mild-degraded,32 gray-mild-degraded and32 irregular-clear-RGB
cases. Each source receives one of every new condition; each geometry gets eight
per condition. Only32 irregular-RGB cases use new masks. Other augmentations keep
the original geometry. Each64cycle contains exactly two distinct anchors and two
distinct augmented IDs. Every256case appears once in each route, and code/fusion
use the same four IDs per cycle. Freeze a deterministic new-phase order before
viewing model outputs; preserve the old sampler evidence.

Strict gray is replicated Y=.299R+.587G+.114B applied to both observed input and
known target256/HQ512. Never convert only the input and implicitly teach
colorization. Mild input-only corruption is proposed blur sigma0.4–0.8pixel,
downsample side128–192 then bilinear256, shared-channel noise0–2/255 and JPEG
quality75–90, applied once before masking. Freeze actual transform bytes and
local per-case seeds. These are synthetic ranges, not measured CCTV degradation.
No real user-upload processing changes here. Low-chroma real photographs remain
separate stress evidence; gray training is not universal near-gray qualification.

Targets are appearance-matched clean hidden references. Delivery copies the real
observed/degraded prepared source outside removal exactly, so this trains
completion, not global restoration. Clean-island seams, arbitrary colorization
and hidden expression remain review concerns. Unpaired DEV never supplies a
clean target. Irregular masks have bounded translation±8pixels,scale.85–1.15,
rotation±10degrees, reviewed irregular boundaries and existing padding/visibility
checks. They simulate missing support; they are not real hand/scarf segmentations.
Existing DEV and original-MONITOR masks never change. Review actual augmented
input/target/mask previews and freeze their hashes before an execution packet.

## Optimizer continuation; existing losses and teacher unchanged

Keep code CE_all256+original-prequantencoderlatentMSE and fusion unclamped
HQ512maskedRGBL1+innerboundaryRGBL1(kernel7), each with the original1+1 weights.
No new chroma, delivered256, perceptual, identity or GAN loss. Those auxiliary
ideas are retained in the superseded draft, for a separate target/loss test only
if this data-only adaptation is insufficient. Do not call our custom teacher
an exact reproduction of official CodeFormer training.

Before loading R1 into the learner, cache original frozen inpainting encoder/
quantizer targets:40RGB+40gray references=80sets=160 component calls. Keep oldRGB
teacher semantics exact. Then load R1 state and restore both named Adam states
at step32, LR1e-5,betas(.9,.99),eps1e-8,wd0,clip1. Maximum64 additional steps
per route ends at step96:128 new Adam updates,192 cumulative across both phases.
There are512 new backwards and64 new paired cycles. Phase-local and cumulative
counts must be recorded separately.165 generator/codebook tensors remain exact.

Retain R1 CPU/CUDA/Python/NumPy RNG and hash-check restoration at the declared
optimization boundary after deterministic preflight. Prematerialized new data
and its separately seeded sampler phase do not silently consume restored global
RNG. This is optimizer continuation with a declared new data/sampler phase,
not replay of the historical dataset or a restart from original weights.

## Endpoints, frozen cohorts and stops

Keep the original32 clear-monitor input/target/mask cohort and every existing
raw/PNG macro/per-geometry hole/boundary gate. Require at least1% incremental
mean delivered-PNG hole-L1 improvement against frozenR1 with all other declared
endpoints nonworsening. Report cumulative improvement against original CodeFormer
on that identical32case cohort separately, never pooled with challenge cases.

Add16 prospectively frozen adaptiveTRAIN challenges from the same eight monitor
sources: one gray+degraded and one irregularRGB+degraded case per source, balanced
four cases per geometry. Require at least1% mean PNGhole-L1 gain versus frozenR1
and no condition/geometry regression in raw/PNG hole/boundary endpoints. Report
strict-gray raw/PNG chroma as a separate diagnostic; it is not a silently added
selection gate. These are adaptiveTRAIN checks, not FINAL/independent identities.

Original-monitor full snapshots0/16/32/64; challenge0/64; fixed public32 cases
(28neural+4empty) only0/terminal. Retain exact masks and original/degraded/family
groups for unpaired public evidence; no hiddenPSNR/SSIM. Preserve the same >10%
catastrophe stop on every existing monitor endpoint at16/32, plus input-parity,
nonfinite, exact-pixel, empty-bypass and resource stops. Review every planned
visual sheet and the same-image/mask user comparison. Optional gray/display
postprocessing cannot conceal raw model color defects or replace qualification.

## Exact work and storage admission

64cycles×4microbatches×2routes=512 learning forwards/backwards.128 original-monitor
+32challenge+56public+8fresh parity calls gives736 whole-model forwards. Add160
teacher components:896 typed slots planned, capped1000. Four fresh frozen-R1
app-wrapper comparisons must pass the unchanged1e-6 gate before any update.
No automatic extra cycles, historical rerun or intermediate-state rescue.

Root approved these prospective bounds for source preparation:1800s hard child,
1680s cooperative work,20GiB CUDA allocation,7GiB initial free disk/2GiB stopfree,
2304MiB output/raw-and-compressed-export caps and900s cooperative export. Preserve
one complete terminal trainable/Adam/RNG/sampler/counter/partial-gradient state,
including failed states. The old1536MiB allowance is not assumed sufficient.

A source-enforced conservative serialization budget is216 nonempty evaluation
NPZ arrays×4,980,736bytes, eight empty arrays×1,835,008bytes,60,833,792bytes for
four paired-parity arrays,1%ZIP/header reserve,224 PNGs capped256KiB each,
a1GiB terminal/partial-state write bound and8MiB receipts/logs/protocol bound.
This totals2,303,717,049bytes; plus<=2MiB archive overhead fits2304MiB
(2,415,919,104bytes).7GiB startfree leaves more than2GiB after both maximum output
and archive. Implementation must enforce these reservations and the saved state
bound. If it cannot, redesign before execution rather than drop failed evidence.
Timing is an enforced stop, not a promise that64cycles will finish or help.

## Primary sources and pending work

CodeFormer's original paper motivates a discrete prior and contextual prediction;
it does not qualify our adaptation. [Paper](https://arxiv.org/abs/2206.11253).
Its official project separates aligned inpainting, restoration and colorization.
[Pinned README](https://raw.githubusercontent.com/sczhou/CodeFormer/b33cc7d639d6545bfcccc7e0bc6ae51f24e79c2b/README.md).

Official dataset code applies corruption in the non-inpainting branch and gray
conversion to input alone. Our task-specific paired-gray choice preserves the
known target's appearance. [Pinned dataset](https://raw.githubusercontent.com/sczhou/CodeFormer/b33cc7d639d6545bfcccc7e0bc6ae51f24e79c2b/basicsr/data/ffhq_blind_dataset.py).
Root's primary-source review found official feature supervision uses quantized
frozen-HQ-VQGAN features, unlike our declared original-prequantlatent teacher.
Preserve our definition for this isolated coverage test; changing it needs a
separate target audit/ablation. [Official model source](https://raw.githubusercontent.com/sczhou/CodeFormer/master/basicsr/models/codeformer_model.py).
Do not copy a1024-code training config into this actual admitted512-code model.

Full40source/term pins, actual schedule/mask/loss/scarf evidence:
research/completion_codeformer_r1_context_design_v1/saved_source_and_support_evidence.json.
Its prospective-case section is an earlier unexecuted draft; finaldesign_review.json
supersedes it. The earlier auxiliary-loss design is retained in that folder.
Next: independent design review, then actual source-only augmented-preview/terms/
mask/target/serialization admission before any new trainer or execution packet.
All actual training remains a distinct manual user launch on the L4.
