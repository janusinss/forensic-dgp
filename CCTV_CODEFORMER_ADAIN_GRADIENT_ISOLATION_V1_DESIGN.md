# CodeFormer AdaIN gradient isolation V1: prospective controlled training design

**16 October 2026. A new finite comparison from untouched CodeFormer. This
document specifies a hypothesis and controls; it is not a training result or
model acceptance. A separately sealed packet and exact manual guide are required
before the user launches anything on the existing L4.**

## Question and reason for this factor

The [bounded-fusion return](CCTV_CODEFORMER_BOUNDED_FUSION_FEASIBILITY_V1_RETURN_REVIEW.md)
is intact, but rejects a fixed brightness gate: normal tone improves while dark
visibility and other descriptors worsen. A normal, low-luminance input receives
almost the full correction, and a bright simulated-dark input receives little.
Do not tune that gate or its feature bound after seeing those outputs. Keep all
returned states and failures. The untouched CodeFormer benchmark remains fixed.

The next question is whether image-objective gradients entering the encoder
through its AdaIN style statistics contribute to the learned appearance tradeoff.
This is a plausible mechanism to test, not an established cause. Earlier gradient
conflict evidence came from the older V2 state; a frozen-content intervention
cannot establish FP32 learning dynamics or identify the encoder as the cause.

The retained architecture already detaches the hard-selected codebook features
with `detach_16=True` and the encoder skips supplied to fusion. It still sends
`lq_feat` into `adaptive_instance_normalization` as the differentiable style
input. That route lets image objectives update encoder moments. It also serves
useful exposure changes, so freezing the entire encoder is not justified.

Change exactly that training gradient route:

```
# Existing forward value and style statistics:
adaptive_instance_normalization(quant_feat, lq_feat)

# Proposed candidate training graph; same values at identical weights:
adaptive_instance_normalization(quant_feat, lq_feat.detach())
```

Keep the encoder trainable through token CE, latent/HQ feature supervision and
the existing token-KL anchors. Keep fusion learning and the unchanged losses.
No new learned gate, output enhancement, exposure classifier, decoder, loss
weight, learning rate or input policy is introduced. Inference uses ordinary
CodeFormer with the resulting weights; detachment changes no inference value.
The encoder's moments can still change through its remaining supervision, so
the intervention does not guarantee preserved tone or successful training.

The wrapper must affect only the candidate forward being tested. Preserve and
restore the exact original callable in `finally`, including exceptions. Do not
edit the retained architecture or let the wrapper leak into independent teachers.
No checkpoint parameter is added, removed or renamed.

## Frozen initialization, data and objective

Use untouched official CodeFormer SHA-256
`1009e537e0c2a07d4cabce6355f53cb66767cd4b4297ec7a4a64ca4b8a5684b7`.
The retained FP32 packet, protocol
`8acff86e31c3d135994fdac6ed781374e036150ce1ae2110db82ca0dc5a3aff1`,
is a read-only dependency. The stopped checkpoint is a historical comparator,
not this run's initializer. Do not restore an old optimizer or continue a stopped
run. All original DGP/CodeFormer states, targets, splits and gate failures stay.

Retain the exact first 50 FP32 update positions and their 200 canonical TRAIN
observations, 100 photographic reference IDs, canonical input/clean tensors,
sampler cycle and degradation fields. These are not 50 epochs or independently
established distinct people. The renderer produces RGB8 observations before
normalizing to float32/255; all 200 saved PNG/255 pixel hashes match the original
canonical training tensors. Keep exact regeneration and first200 record parity.
The bounded diagnostic's selected32 observations do not replace that full
training schedule. Earlier continuous-input wording is a documentation error,
not a different numerical input path or an explanation for the quality stops.

Keep all 350 declared trainable tensors in the original six encoder/token/fusion
groups, all 165 frozen tensors, w=1, `detach_16=True`, AdaIN enabled, the retained
training hybrid mode (`net.train()` with dictionary/generator in eval),
teacher/perceptual and snapshot eval behavior, seeded FP32, disabled
autocast/TF32 and the original
input256-to-bilinear512 geometry. Keep native512 losses, exact delivered-256
behavior anchors and clamp-before-bilinear256/floor-to-PNG delivery unchanged.
Retain the prior algorithm settings rather than adding a determinism treatment.
Record actual deterministic-algorithm/cuDNN/CUBLAS flags in runtime metadata;
seeded execution alone does not establish universally deterministic gradients.
Exact fixed-state forward and HQ-target parity are still mandatory VM checks.

Adam, betas(.9,.99), epsilon1e-8, zero weight decay, encoder/token LR1e-5,
fusion LR2e-5, first50 warmup rates and the retained600-position schedule/data
horizons remain. **The new run stops after at most50 actual updates.** A schedule
horizon is not authorization for600 updates. No GAN, EMA, gradient clipping or
precision sweep is added.

Retain loss weights: token CE.5, latent MSE1, native RGB L1 1, LPIPS1, coarse
RGB L1 4, delivered behavior RGB L1 2 and behavior token KL.1. Keep existing
image eligibility and the50 normal-B behavior anchors. Additional full-RGB
teacher anchoring on severe inputs would risk preserving known original false
detail and is not silently combined with this factor.

Recompute the original391 HQ cache entries through the same pinned teacher and
require exact parity with a new metadata-only manifest derived from the closed
FP32 return. Bind reference, clean/source, teacher, restoration dictionary,
selected-index array and feature-array fingerprints. Check all391 entries before
constructing Adam. Reusing the prior targets' evidence avoids treating a new
teacher target as an unexplained second factor; it does not independently rerun
the old nearest-code calculation locally. Teacher and all original assets remain
hash-bound. No bulk cache upload or broad parent-output copying is needed.

## Fixed pre-Adam VM witness

Choose update3 slots1/2, the normal/dark A pair at eligible severe edge32. The
first update's edge24 pair is image-ineligible and is deliberately not used for
a witness of LPIPS/fine-image terms. Sampling metadata chooses these two fixed
TRAIN observations; it never enters inference routing.

At original initializer weights, run each observation once through the unmodified
FP32 candidate graph and once through the isolated graph: four full CodeFormer
and four LPIPS forwards. Use identical verified HQ targets and the unchanged
image eligibility/loss definitions. Save actual native512, RAW256, PNG/token
evidence and scalar components, and require exact fixed-state forward/value
parity. A changed style value, token choice or geometry must stop the experiment.

For each forward, reserve two `autograd.grad` queries over the declared trainable
parameters: weighted image objectives and weighted CE/HQ objectives, eight new
queries total. Require finite nonzero original image-to-encoder signal, no
isolated image-to-encoder or image-to-token-predictor path, finite nonzero isolated
image-to-fusion signal, and finite nonzero CE/HQ-to-encoder and predictor signal.
Unused gradients and exact zero tensors must be distinguished from nonfinite or
missing expected gradients. Report all parameter-group/named gradient norms;
these are source-bound gradient captures, not archived complete gradient vectors.

Do not construct Adam, call backward or update any model in this witness. Record
and restore CPU/CUDA RNG around the diagnostic, require all model states unchanged
and parameter `.grad` fields empty, and restore the original function namespace.
Only a passing witness allows the new finite training section to begin. Existing
LPIPS prerequisite, initial target/parity and snapshot checks remain in force.
A failure is retained and exported; there is no automatic repair, retry or resume.

## Exact finite accounting and resource boundaries

The original2057 guarded reservations remain, including391 actual HQ teacher
forwards,200 candidate training forwards,148 training perceptual forwards,
50 behavior-teacher forwards,812 snapshot forwards,2 baseline and2 initializer
parity forwards,1 perceptual prerequisite forward,1 prerequisite gradient query,
200 backward calls,50 optimizer attempts and200 training-input writes.

The fixed witness adds4 candidate forwards,4 perceptual forwards and8 gradient
queries: **2073 total reservations,1614 actual neural forwards,9 gradient queries,
200 backward calls and at most50 optimizer attempts/committed updates.** Successful
endpoint totals must be derived from and checked against the final worker.
Early failure must retain actual counters rather than fabricate successful totals.
Model construction and state loading are not neural forward calls.

Retain the prior finite limits as ceilings: initialization600seconds,
benchmark600, projected/actual training900, each snapshot600, whole worker3000
and export600. The pre-Adam witness has its own180second cap and remains inside
the600second initialization and3000second whole-worker clocks, without resetting elapsed time or
excluding it from resource accounting. Use external process-group timeouts with
finite grace, streaming capped logs, fresh workload/source/disk checks and proof
that the worker is terminal before export. Existing active workloads stay.

Require16GiB free after installation, preserve4GiB, cap CUDA reserved memory at
22GiB, metadata at128MiB, each PNG at198656bytes and checkpoint temporary writes
at1GiB. Before sealing, derive the full checkpoint/Adam/state/RNG/cache/snapshot/
witness/log/source/manifest/TAR/gzip ledger, all file/array caps and installed
overlay/dependency sizes from actual code. The old3.5GiB result/3.6GiB archive
ceilings can be retained only if that complete new ledger proves them sufficient;
otherwise revise the new declared limits before sealing, never mid-run.

Proposed new VM root is `~/forensic-dgp/cctv_codeformer_adain_gradient_isolation_v1_vm`.
All actual neural/gradient/optimizer work is user-launched on the existing NVIDIA
L4 through verified transfer/tmux commands. No local neural proof or automatic
training is authorized by CPU fixtures or a prepared archive.

## Outcome and unchanged quality requirements

Save initial and update50 outputs on the same406 development/comparison cases:
318 paired HQ synthetic cases,40 older paired comparisons and48 source-separated
unpaired native CCTV crops. As in the retained FP32 experiment, the812 full
comparison outputs save RAW256 and exact delivered PNGs. Native512 image/loss
semantics remain separately declared; only the four witness native512 arrays
are additionally saved for direct forward parity. Do not silently add812
native512 exports to the bounded payload. Inspect all planned sheets. Preserve
failed states and initial/stopped checkpoint
bindings. Do not turn unpaired CCTV into PSNR/SSIM evidence or read reserved FINAL
pixels. No ethnicity, local Zamboanga performance or recovered-identity claim.

The unchanged61-clear guard requires mean MAE regression<=.01 and every
case<=.03 versus original, independently on RAW and PNG. The53-normal-blur
luminance-bias/coarse-RGB guards retain their existing mean.01/worst.03 limits and
implementation. No historical DGP1% feature-gain requirement replaces these
CodeFormer checks. Quality failure must stop and export even if losses improve.

A passing gradient witness proves this gradient route was isolated, not that it
caused previous failures or will improve delivered images. Compare trained output
with untouched CodeFormer and the retained FP32/BF16 results on identical inputs.
Useful dark visibility, visible facial structure, normal appearance, moderate
helpful brightening and severe insufficient-information cases must all be reviewed.
Only later independent full review, user restoration acceptance and the local
Auto/override/PNG app flow can qualify adoption. Completion/detector model work
stays deferred until restoration acceptance. The complete thesis goal remains.

## Research basis and limits

The [CodeFormer paper](https://arxiv.org/abs/2206.11253) separates code prediction
from controllable feature transformation; the [official training guide](https://github.com/sczhou/CodeFormer/blob/master/docs/train.md)
describes its training stages. The [official StageIII configuration](https://github.com/sczhou/CodeFormer/blob/master/options/CodeFormer_stage3.yml)
fixes the dictionary/generator while retaining joint learning and image losses.
These primary sources were checked16October2026. They do not prescribe our
AdaIN style-detachment experiment or prove that it improves CCTV restoration.
The inference and gradient-route observations above come from the retained,
fingerprinted local implementation. Credit our method as a trained adaptation
of CodeFormer, not a newly invented architecture or exact upstream reproduction.

The independent recommendation and its CPU saved-pixel checks are retained in
[design_recommendation_v1.md](outputs/codeformer_bounded_fusion_feasibility_v1_return_review/design_recommendation_v1.md)
and its JSON companion. Near-equivalent changes to coarse native/delivered
pooling were not selected as a claimed remedy. No source, weight, gradient,
training, VM or app execution is established by this design document itself.
