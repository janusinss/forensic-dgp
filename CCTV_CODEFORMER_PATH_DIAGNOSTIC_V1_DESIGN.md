# Finite CodeFormer training-path diagnostic V1 — prospective design

**Purpose:** distinguish numerical path differences, encoder illumination moments
and learned fusion changes before selecting another training recipe. This is a
manual L4 measurement with zero optimizer updates. It is not V3 training, a new
restoration checkpoint, a quality pass or an application model change.

The independently audited V2 model changed all 350 intended tensors but stopped
at 50 updates. Its outputs improve SSIM on 50/53 HQ blur cases and many dark
paired images; three clear controls lose fidelity and SSIM. The user accepts
moderate brightening when it improves facial visibility. Helpful exposure,
contrast, facial detail and appearance preservation will therefore be measured
separately. Historical V1/V2 failures remain unchanged.

## Fixed evidence and paths

The parent is the retained sibling `cctv_codeformer_adaptation_v2_vm`, protocol
SHA-256 `029291a2d58d7bc8b33fb99b86774e13db4e5c098bb8af5f508fdc5e24b20183`.
The initializer, stopped50 checkpoint, HQ teacher and LPIPS assets are read-only
external dependencies with exact file hashes. The stopped checkpoint must match
`3863236cd5b2ef01432e7498836d3c37096903d84a293f76b4a65bcd1fa7037e`.
No optimizer is loaded for execution and no replacement weights are exported.

Use exactly the eight already saved TRAIN PNGs from updates 1 and 24, slots 1–4,
and the four corresponding genuine photographic clean512 targets. Chronological
selection covers severe24, severe40, clear256 and light64 in matched normal/dark
pairs. No DEV, native CCTV, reserved-final identity or failed clear-control case
is used for these gradient measurements. Keep provenance, terms, role records
and every input hash. Regenerate no degraded observation.

The frozen V2 data code formed its input float32 tensor directly as observed
RGB8 divided by float32(255). Preparation and transfer verification must prove
that each saved PNG reconstructs the exact recorded input float32 hash. The
clean512 float32 target must also match the recorded tensor hash. A failure
stops preparation; do not silently accept a new tensor or camera rendering.
The two saved FP32 behavior anchors are retained. Their comparison with a new
inference-mode pass is descriptive because inference/no-grad numerical paths
have not yet been measured as identical.

## Four numerical paths and repetitions

At both the untouched initializer and stopped50, evaluate all eight inputs as:

| Path | Conditions |
|---|---|
|E32|Evaluation, FP32, inference context|
|E32-grad|Evaluation, FP32, autograd enabled|
|T32|Training, FP32, autograd enabled; frozen decoder/codebook stay evaluation|
|TBF16|Actual V2 BF16 training path, FP32 losses and LPIPS|

Repeat each path once on update1 slot1 and slot3 at each state. Record RAW256,
floor-quantized PNG, preclip range, clipping fraction, luminance and contrast.
Record actual code indices selected by the architecture's native-dtype
softmax→top-k, native probability margins and FP32 logit margins. Do not replace
the actual selected codes with FP32 logit argmax; BF16 ties need explicit evidence.
Report repeat stability and cross-path disagreement separately.

At original weights before the first V2 step, saved clear-anchor image L1 is
0.0015124048804864287 and token KL is 0.0011029532179236412. These nonzero values
motivate the test; they do not establish the cause or size of later drift.
No output is exposure-corrected, sharpened or selected as a winning model.

## Conditional moment and fusion interventions

At stopped50 E32 on all eight inputs, compare the existing current/current arm
with three disposable inference arms: original AdaIN encoder channel moments,
original fusion weights, and both. Capture the original moments during already
budgeted E32 passes. Keep stopped encoder/token content and actual selected
codes fixed. Verify selected indices are identical for every intervention.

Only the declared moment substitution and/or fusion weights may differ during
each arm. Restore scoped hooks and exact module state after each stage, even on
failure. Save fingerprints before/after; export no hybrid model. Effects are
conditional on these inputs and the retained content path. Residual differences
cannot be attributed uniquely to code selection or interpreted as native quality.

## Gradient signatures without an optimizer

Reuse the six selected base TBF16 graphs at each state: update1 slots1/3 and
update24 slots1/2/3/4. Query each actual V2 weighted loss divided by four using
`autograd.grad`, without `.backward()`, parameter `.grad` accumulation or any
optimizer. This is 32 queries per state, 64 total. HQ codes come from four
frozen-teacher passes; ten eligible cases use frozen FP32 LPIPS. The two exact
saved FP32 normal-B delivered/logit anchors provide ordinary detached behavior
targets, matching the actual V2 loss targets. Compare new original E32 outputs
with those saved anchors separately and report any numerical gap rather than
substituting a newly computed target.

Report absent/zero/finite gradient status, per-group norms, and normal/dark
photometric and behavior cosine signatures. Retain only transient CPU vectors
needed for the declared comparisons; do not archive full gradients. Some terms
cannot reach every group through hard detached code selection. Do not require
all terms/groups to have nonzero gradients, or infer a successful Adam trajectory
from these measurements. Frozen teacher/perceptual and all model tensors must
remain unchanged, and parameter `.grad` fields must remain empty.

## Finite execution and review

| Resource | Hard bound |
|---|---:|
|CodeFormer forwards|104: 64 base + 16 repetitions + 24 interventions|
|HQ teacher / LPIPS forwards|4 / 10|
|Gradient queries / optimizer updates|64 / 0|
|Worker / export wall time|900 / 300 seconds|
|CUDA reserved memory|22 GiB|
|Initial free storage / retained reserve|6 / 4 GiB|
|Result payload / compressed archive|256 / 272 MiB|

The supervisor applies an external process-group timeout and preserves partial
evidence. Reserve every call before execution. An existing tmux shell alone is
not a competing workload; inspect current GPU compute processes and actual
training commands. Refuse an occupied GPU, existing run evidence or changed
dependencies. Never kill an unrelated process, relaunch historical training,
install packages automatically or silently retry this packet.

Prepare sources and eight exact inputs locally with CPU-only checks. Actual
neural/gradient measurements run only when the human uploads and launches this
new package in tmux on the existing NVIDIA L4. Independently inspect returned
RAW/PNG images, tokens, scalar formulas, intervention/state invariants and every
planned visual sheet before choosing another finite training recipe. Diagnostic
completion alone changes neither quality status nor application defaults.

The investigation follows the saved independent recommendation in
`outputs/codeformer_adaptation_v2_return_review/next_diagnostic_review_v1.md`.
The full restoration goal remains active. User acceptance and independent final
evaluation are still required; completion/detector development remains deferred.
