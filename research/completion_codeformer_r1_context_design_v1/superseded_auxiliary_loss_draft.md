# Completion R1: context and mask coverage continuation review

25 October 2026. DESIGN REVIEW ONLY. No execution packet, model call, gradient,
training, app edit or VM operation follows from this document. The user prefers
R1 after a same-image/mask comparison; keep it as the next working completion
baseline while retaining original CodeFormer and all failed states.

## What the actual first pilot covered

The saved schedule verifies exactly 32 FIT source photos, eight separate
adaptive TRAIN monitor sources and 128 FIT reference/mask pairs. Every pair was
seen once by each learning route: 128 code and 128 fusion backward calls, with
32 Adam steps per route. This is one exposure per small source/mask pair, not
many full-dataset epochs. All 40 originals were rehashed and are genuine
1024×1024 files; retained 256/512 targets and case masks match the sealed protocol.

The 128 FIT mask files contain only four unique masks, repeated over all sources:
lower face 15,277 pixels; eye band 8,640; image-left lens 2,219; cheek/object 2,908.
There are zero strictly grayscale FIT sources and zero strictly grayscale monitor
sources. No synthetic degraded context or grayscale condition was part of this
training/monitoring design. Broad public DEV images were output checks, not
training examples. This is the clearest demonstrated coverage gap.

Source roles and terms stay unchanged: all 40 originals have original TRAIN role;
32 FIT/eight MONITOR IDs and target hashes are distinct. These facts do not
establish independent identities. License counts are 21 attribution, 18
attribution-noncommercial and one CC0. Per-photo URLs/licenses remain in the
protocol/evidence. Official FFHQ pretraining overlap is unknown. No ethnicity,
local capture provenance or Zamboanga performance is inferred.

## What optimization actually did

The two routes worked and the saved R1 state changed all 332 intended tensors.
Codebook/generator 165 tensors remain frozen. Code uses CE plus latent MSE;
fusion uses masked HQ512 RGB plus inner-boundary RGB loss. Because code lookup
uses discrete selection and the code/encoder features are detached in the
fusion path, fusion RGB gradients do not train the code prediction route. Keep
the explicit two-route learning design. The fixed generator still constrains
possible textures and expression, so additional data cannot promise exact hidden
identity or eliminate every hallucination.

Mean CE was 1.7940 and latent MSE 0.02898. Mean masked RGB512 L1 was 0.05549 and
boundary L1 0.02735. These scalar scales do not prove gradient dominance. All 32
code steps were clipped at norm 1 (preclip 5.659–23.634); no fusion step reached
that bound (0.233–0.671). Raising LR or removing clipping is not the evidence-led
first change. First/last loss means use different pairs and are not a valid
paired improvement measurement. The independently matched monitor reduction of
6.1853% is the appropriate existing small-pilot result, with six individual
hole-L1 regressions retained.

## Scarf evidence: support errors and generation errors stay distinct

All four saved scarf cases preserve every source byte and raw coordinate outside
the fixed removal area. Knit-scarf support is 14,259 pixels; scarf/gloves support
is 18,757. Their outer three-pixel bands are also exact copies. Across all four
cases 588,336 outside RGB bytes are unchanged; no pixel outside these reviewed
masks can have been removed by R1.

Within the masks R1 differs from original CodeFormer: mean absolute PNG changes
are 8.636/10.710 levels for original/degraded knit scarf and 9.663/9.399 for
original/degraded scarf/gloves. These numbers locate generated changes; they
are not a cloth-removal or anatomy score. No aligned uncovered target or actual
covering segmentation truth exists for these public photographs. The root visual
review records both copied scarf fragments outside support and beard/cloth-like
generated defects inside support. Increasing mask area silently would confound
them. Keep their current delivery masks fixed for this model comparison; any later
manual correction is a separate assisted result.

## Recommended next finite comparison

Continue from R1 weights, using a frozen R1 control on identical inputs and masks.
Do not restart from original CodeFormer or unfreeze its generator first. The
current trained code/fusion state has a verified useful signal and user preference;
its most direct missing experience is grayscale/low-chroma, degraded visible
context and shape diversity. This supports one new data/loss adaptation phase.
It does not prove a broad new model is already qualified.

Keep the exact 32 FIT and eight adaptive TRAIN monitor source IDs. Create 256
FIT cases: 128 byte-identical original clear anchors plus 128 new deterministic
irregular variants, one additional case per reference/geometry. Rotate four
conditions by source/geometry index: 32 strict-gray clear; 32 mildly degraded
RGB; 32 mildly degraded low-chroma; 32 irregular clear RGB. Each FIT source gets
four old and four new cases. New masks have bounded translation/scale/rotation
and reviewed irregular boundaries, pass existing visibility/padding checks and
are frozen prospectively. They simulate missing support; they are not real
hand/scarf segmentation labels. Existing DEV and monitor masks never change.

Define appearance A_s(X)=sX+(1-s)Y(X), with replicated luminance
Y=.299R+.587G+.114B and s=0/0.05/1 for gray/low-chroma/RGB. Apply the same appearance
transform to observed input and both clean targets. Apply only mild deterministic
blur/downsample/noise/JPEG corruption to the input256 context before masking:
sigma0.4–0.8 pixels, downsample side128–192 then bilinear256, noise0–2/255 and
JPEG quality75–90. Use shared-channel noise on gray/low-chroma cases. These are
proposed simulation ranges, not calibrated CCTV measurements. Do not duplicate
unknown blur/noise on real user uploads or change app preprocessing here.

Targets remain appearance-matched HQ hidden regions; the visible delivery always
copies the actual degraded prepared input exactly. This teaches missing-region
completion from imperfect context and does not train global restoration. Mild
corruption limits, but cannot rule out, a sharp-looking generated island; inspect
inside seams and preserve a separate degraded-context monitor. No unpaired DEV
photograph becomes a clean target.

Keep CE+latent MSE unchanged. Retain fusion HQ512 hole+inner-edge L1 and add only
0.25 times actual delivered256 hole L1 plus 0.25 times target-chroma residual L1
for s<=.1. The chroma term matches predicted (P-Y(P)) to known transformed-target
(T-Y(T)); it does not force real colored faces to gray. Cache original frozen
teacher RGB targets for all40 sources plus gray and low-chroma targets for32FIT:
104 label sets/208 encoder+quantizer component calls. Then load R1 into the
learner. This avoids drifting existing RGB teacher semantics into R1's changing
encoder. Teacher bias remains declared. Auxiliary weights are prospective small
weights, not tuned on public hidden targets; the combined design does not isolate
which addition caused a future difference.

Use fresh identical Adam settings for this explicitly new adaptation phase:
LR1e-5, betas(.9,.99), eps1e-8, no weight decay, clip1. Preserve R1's complete old
Adam/RNG/terminal state; do not call the new phase an exact optimizer resume.
Maximum64 paired cycles, four batch-one microbatches per route, 64 code +64
fusion steps =128 Adam updates, 512 backwards. One learning arm plus frozen R1
control is sufficient; there is no need for a new broad architecture sweep.

## Gates and finite resource admission

Keep original32 clear-monitor inputs/targets/masks, all original raw/PNG hole and
boundary macro/per-geometry gates, the 1% relative delivered-PNG hole gain screen
and exact preservation, now compared with frozen R1. Keep original CodeFormer
as the permanent historical comparator. Add 16 prospectively frozen adaptive
TRAIN challenge cases from the same eight monitor sources, two conditions per
source and four per geometry. Report strict-gray, low-chroma and degraded
subgroups separately; their hole/boundary and target-chroma endpoints must be
nonworsening against R1. No new independent identity or FINAL claim follows.

Save full original-monitor outputs at0/16/32/64, new challenge outputs at0/64 and
same32 exposed unpaired public cases at0/terminal. This plans 736 whole-model
forwards plus208 teacher-component calls =944 typed slots, capped at1100. Keep
30-minute hard child,28-minute cooperative work stop,4GiB initial/2GiB stopfree,
20GiB CUDA allocation bound,1536MiB output/raw-compressed export caps and900s
cooperative export guard. Retain one full terminal trainable/Adam/RNG/sampler/
counter/partial-gradient state. Preserve failures and never rescue an unsaved
intermediate checkpoint. More epochs are not appended automatically.

Before sealing a packet, independently size actual NPZ retention: extra challenge
arrays may approach the existing1536MiB cap. If that layout cannot fit, revise
the prospective retention schedule or finite design before running; do not discard
failed evidence afterward. Existing >10% monitor-regression, nonfinite, input-parity,
resource and exact-pixel stops remain. A quality screen still requires every
planned visual sheet, appearance review and user comparison. Strict-gray/display
processing cannot conceal raw model color artifacts or substitute for qualification.

## Primary-source basis and boundaries

CodeFormer's learned discrete codebook and contextual prediction support adapting
the existing conditioning routes before replacing its prior. This is method
motivation, not evidence that our specific adaptation will improve. [Original
paper](https://arxiv.org/abs/2206.11253).

The official aligned inpainting path is separate from its restoration and
colorization entrypoints. [Pinned official README](https://raw.githubusercontent.com/sczhou/CodeFormer/b33cc7d639d6545bfcccc7e0bc6ae51f24e79c2b/README.md).

The official dataset code applies degradation in the branch where inpainting
masks are disabled; its grayscale probability affects input alone. Reusing that
input-only grayscale behavior would conflict with our task's appearance-preserving
grayscale targets. The proposed both-sides transform is our task-specific design,
not a claim about the authors' completion qualification. [Pinned official dataset
source](https://raw.githubusercontent.com/sczhou/CodeFormer/b33cc7d639d6545bfcccc7e0bc6ae51f24e79c2b/basicsr/data/ffhq_blind_dataset.py).

Local source evidence: third_party/codeformer/codeformer_arch.py (discrete lookup,
quantizer/generator freeze, detached fusion inputs); scripts/
cctv_completion_codeformer_finetune_v1_r1_vm.py (actual routes/losses/schedule).
Detailed source hashes, all 40 source/term records, actual mask/schedule coverage,
loss receipts and scarf saved-support counts are in
research/completion_codeformer_r1_context_design_v1/saved_source_and_support_evidence.json.
The machine-readable proposal is design_review.json in the same folder.

Next action is independent design review and actual augmented TRAIN/monitor preview
admission. No upload bundle, manual launch command or training is prepared here.
