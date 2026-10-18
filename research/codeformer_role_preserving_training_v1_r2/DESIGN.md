# Role-preserving CodeFormer training V1 r2 — numerical prerequisite repair

Status: new finite experiment specified for independent source review. No model
from this recipe has been trained or evaluated. Preparation is not authorization
for an automatic VM launch, retry, continuation, model qualification or app change.

## Immutable prerequisite and numerical test repair

Original V1 and R1 stopped before Adam construction, with zero new updates and
no checkpoints. V1 compared unequal eval/train autograd contexts and retained
no failing arrays. R1 corrected those contexts and retained its actual evidence.
Its first normal witness has exact native512, RAW256, latent and selected tokens,
but 223,123/262,144 logits differ by at most 7.62939453125e-6. All 256 saved
indices match both argmax arrays; every parent top-two margin exceeds twice
the actual per-token maximum difference. The 515-tensor parent/candidate state
fingerprints match. The operation responsible for logit differences remains
unproven. This is not evidence of a restoration-quality failure or improvement.

R2 explicitly replaces ONLY cross-object initializer bit-exact logits with
`initializer-numerical-contract-v2`. Following the public PyTorch2.9 float32
comparison convention, all finite FP32 logits must satisfy
`abs(candidate-parent) <= 1e-5 + 1.3e-6*abs(parent)` in float64 audit arithmetic.
These constants are fixed from the documented datatype defaults, not selected
by a learning/quality sweep. The two arrays must have shape1x256x1024. Every
saved int64 token must exactly equal both actual logit argmax indices; every
parent top1-minus-top2 margin must strictly exceed twice that token's actual
maximum logit difference. This additional certificate rejects unstable choices,
even when scores satisfy the numerical bound.

Native1x3x512x512, RAW256x256x3 and latent1x256x16x16 must remain finite FP32
and bit-exact between both objects, and all256 int64 tokens must be exact.
Complete state, context and RNG invariants remain exact. Arrays, differences,
contexts and RNG are saved before the initializer decision. A separate pair of
training-mode, autograd-active unscoped captures still supplies the reference
for the scoped gradient witness. That scoped-versus-unscoped array/loss equality
requirement is unchanged and receives no new tolerance. No extra neural call is
added: maximum1,399 forwards/1,854 reservations,50updates and200backwards.

This changes a self-imposed numerical initialization test; it does not change
restoration acceptance thresholds, loss weights, data, architecture, learning
rates, snapshot0 image parity, checkpoints or app-promotion requirements. The
second dark witness and the scoped gradient prerequisite are still untested on
L4 and must pass prospectively before Adam is constructed. Generic floating-point
comparison defaults do not establish model quality, exact identity, cross-device
reproducibility or a specific CUDA cause. V1/R1 sources, returns and failures
remain immutable; R2 is a new manual finite training pilot, not an automatic retry.

Sources: [PyTorch2.9 numerical accuracy](https://docs.pytorch.org/docs/2.9/notes/numerical_accuracy.html)
and [PyTorch2.9 assert_close defaults](https://docs.pytorch.org/docs/2.9/testing.html#torch.testing.assert_close).
Actual R1 review: `CCTV_CODEFORMER_ROLE_PRESERVING_TRAINING_V1_R1_RETURN_REVIEW.md`.

## Evidence and testable hypothesis

The returned fixed-state response test completed 656 frozen forwards and no
updates. The final scalar controller applied a negative correction to all 264
observations. It worsened behavioral mean error on all 100 already fitted dark
observations and all 32 unused-source dark observations. Fitted normal mean
squared behavioral error fell from 0.001052076389 to 0.000338663753, while fitted
dark error rose from approximately zero to 0.000509992052. The role-balanced
recorded objective still fell by 19.34%. These are recorded FP32 behavioral mean
losses, not clean-reference restoration scores. The independent saved-pixel
audit agrees with the response pattern.

This disproves an explanation based solely on failure to generalize to unused
sources. It does not uniquely identify optimizer, features, capacity or objective
weighting as the cause. A common scalar and an improved aggregate loss did not
preserve the useful dark behavior. More controller epochs or an unchanged retry
are not justified by these observations.

The next hypothesis concerns the actual restorer: give each normal observation
one coherent behavioral target over the full delivered image and tokens, while
giving each dark observation clean restoration supervision and retention of the
useful full-coarse parent. This replaces the old competing clean-normal graph
and its B-normal-only original anchor. It is a training-only assignment of
targets; the network and inference receive no role label or brightness gate.
The normal teacher remains an imperfect benchmark, including on severe A-normal
inputs. Its behavior is not clean truth or proof of identity. Dark clean and
parent targets can also compete; their outcome remains an empirical question.

The evidence pins are the fixed-state state audit
`d14848349a86186971960f8927d231c4223c4522b46e6d07020430d64d2ef504`,
pixel audit `05bb345f6128175cbcf9115feb01e6d7dbd774c0264e41a63e30754db3da87ff`
and mechanism review
`3005dab2986bc6fc8db3bc092913dffee9ddb614a277707dccf3c429b0848bb3`.
The returned archive is
`e60258356e148ae426dcc8863794ad62377559f858a2b5dbf75c3e0eee0d50ad`.
These observations motivate the experiment; they do not establish its success.

## Initialization and exact objective

Initialize all 515 candidate tensors from the actual full-coarse AdaIN update50
model in `outputs/adain_gradient_isolation_v1/stopped_checkpoint.pt`, SHA-256
`43c0f313365e13849ddcd852f401511e4c7ae52cbac55fdd98a6b032f06ef37b`.
Its canonical model digest is
`92ac81750b86ef1da6f1f7c0480b9e64773fafe07b094e4a5ad7e88b3f7a0c8b`.
This retained research parent failed qualification. Load its model tensors only;
do not install its optimizer, RNG, sampler, schedule or completed-update counter.
Freeze an independent copy as the dark behavior teacher. Load an independent,
frozen untouched original CodeFormer, original asset SHA-256
`1009e537e0c2a07d4cabce6355f53cb66767cd4b4297ec7a4a64ca4b8a5684b7`,
as the normal behavior teacher. Load the same frozen HQ teacher and LPIPS assets.

Keep the architecture, fidelity weight1, detach16, AdaIN numerical forward,
256-to-512 input interpolation and delivered output policy unchanged. Keep
350 trainable tensors: 278 encoder/token tensors and 72 fusion tensors. The
165 quantizer/generator tensors remain frozen. The candidate alone uses the
previously audited candidate-scoped AdaIN style detach. There is no tone head,
output centering, new clipping rule, alignment or display processing.

For both normal slots 1 and 3, the entire objective is:

`2 * L1(delivered_candidate256, delivered_original256)
 + 0.1 * KL_T2(original_logits || candidate_logits)`.

No clean CE, clean latent, clean coarse, clean pixel or LPIPS term is built for
normal observations. The full RGB grid replaces a mean-luminance-only target.
Pass `clean512=None` and `codes=None` to the normal helper so an unused
clean-supervised graph cannot silently survive dispatch.

For both dark slots 2 and 4, use the existing clean objective plus parent
retention, in the following fixed summation order:

| Term | Weight | Target and domain |
| --- | ---: | --- |
| Token cross entropy | 0.5 | Frozen HQ indices, all dark observations |
| Latent MSE | 1 | Frozen restoration-dictionary features at HQ indices |
| Coarse RGB L1 | 4 | Unclipped RGB512, 32x32 average-pool to 16x16 grid |
| Pixel RGB L1 | 1 | Unclipped RGB512 against clean512, image-eligible only |
| LPIPS v0.1/VGG16 | 1 | Native signed output against clean512*2-1, eligible only |
| Behavior RGB L1 | 2 | Full delivered256 against frozen full-coarse parent |
| Behavior token KL | 0.1 | Frozen full-coarse parent logits, temperature2 |

KL uses teacher probabilities and candidate log probabilities, sums over the
1024 vocabulary, means over batch/token positions and multiplies by T². Behavior
RGB delivery clamps `(native.float()+1)/2` before bilinear resizing to 256 with
`align_corners=False`. Clean dark image losses do not clamp the prediction.
LPIPS retains its existing internal scaling; add no external normalization.
All candidate terms and original/parent behavior targets are newly computed for
each observation. Only independently verified frozen HQ targets are cached.
No cached gradients or former head state are reused. Each observation's complete
weighted loss is divided by4 before backward, with one step after four slots.

## Finite TRAIN data and optimization

Use exactly the retained first50 matched TRAIN updates: A-normal, A-dark,
B-normal, B-dark, 200 observations from 100 photographic source IDs. A-edge
cycles24/28/32/40; B-control uses the existing six-variant schedule. There are
100 normal and 100 dark observations. A-dark is image-eligible on24 updates;
all50 B-dark observations are eligible, so exactly74 training LPIPS calls occur.
Historical `behavior_anchor` metadata marked only slot3; it is retained as
provenance and must not decide this new role objective.

Retain all200 input/camera/source/pair joins and clean-source derivation hashes.
Build fresh HQ targets for exactly the100 selected TRAIN references and compare
indices/features/record provenance against the corresponding audited cache.
Do not load unused291 HQ sources or regenerate DEV inputs. Source IDs do not
prove different human identities, and CodeFormer pretraining overlap is unknown.
No DEV/FINAL image, metric or returned role response selects a loss/rate grid.

Construct a new empty Adam with betas(0.9,0.99), eps1e-8 and weight_decay0. Use
the existing base rates1e-5 for encoder/token and2e-5 for fusion. Apply a fresh
linear warmup `base_lr * new_update / 50` for new updates1 through50: encoder
2e-7 to1e-5; fusion4e-7 to2e-5. This is a declared new finite schedule, not a
continuation at historical update51. No post50 schedule or resume is authorized.
Maximum50 actual optimizer updates and200 backwards; FP32, no autocast/scaler,
gradient clipping, GAN or EMA. No selection of an intermediate better-looking
checkpoint, extension, automatic retry or promotion is permitted.

## Prerequisites and maximum operation census

Before Adam construction, use the retained image-eligible A-normal/A-dark pair
at update3, slots1/2. Capture two eval/no-grad candidate and two frozen eval/no-grad
parent forwards with identical inputs, forward helper, all module modes and RNG.
Save every native, RAW, logits, feature and token array, actual execution-context
metadata, RNG evidence and exact differences before the declared numerical initializer gate.
Native/RAW/latent/tokens remain exact; only logits use the bounded rule and
strict argmax-margin certificate described above.

Separately capture two train-mode, gradient-active unscoped candidate forwards.
These use the full actual role helper, adding one eligible-dark LPIPS forward.
Preserve candidate module modes and CPU/CUDA/Python/NumPy RNG so these training
captures can be compared with two candidate forwards through the same actual
role helper under the isolated-style scope. Capture all arrays, actual role terms
and weighted totals. Frozen original/parent behavior targets require two
additional forwards for this witness. Eval initializer captures are never reused
as train-mode gradient-witness references.

Query the weighted image partition and token/HQ partition for each role: four
queries total over all350 trainable tensors. The isolated image partition must
have `None` encoder/predictor gradients and finite nonzero aggregate fusion
gradients. The token/HQ partition must retain finite nonzero aggregate
encoder/predictor routes. Do not require an individual dark parent-retention
gradient to be nonzero when candidate and teacher initially coincide. These
source-bound gradient summaries are not an independent full-gradient replay.
Restore RNG, modes, callable and empty `.grad` fields before constructing Adam.

Run one additional LPIPS forward/query on the detached saved dark native output
as a fresh differentiable leaf against the same clean target; require a finite
nonzero prediction derivative. This adds no CodeFormer forward. The normal
witness makes zero LPIPS calls; the eligible dark role witness makes one.

| Reserved operation | Maximum |
| --- | ---: |
| Candidate TRAIN / original teacher / parent teacher | 200 / 100 / 100 |
| Fresh HQ / training LPIPS | 100 / 74 |
| Snapshot candidate forwards, 406 at each of0/50 | 812 |
| Initial candidate / parent parity forwards | 2 / 2 |
| Separate train-mode unscoped candidate forwards | 2 |
| Initial unscoped eligible-dark LPIPS | 1 |
| Role candidate / behavior teacher / LPIPS witness | 2 / 2 / 1 |
| Separate LPIPS prerequisite forward | 1 |
| Gradient queries / backwards / Adam attempts | 5 / 200 / 50 |
| Retained TRAIN input writes | 200 |

The neural maximum is1399 forwards. The total reservation maximum is1854,
including writes and gradient/backward/step attempts. Record actual attempts,
completed updates, partial microbatches and any optimizer-inflight flag honestly;
50 successful steps are not established by reaching a loop index or export.
No local neural/framework/gradient/optimizer work is authorized.

## Evaluation, failure evidence and resource boundary

Save real initial, update50 and stopped/failure checkpoints including model,
fresh Adam, RNG, sampler, rates, protocol and actual counters. Do not serialize
teachers into each checkpoint. Canonical paths are `snapshot_0000/checkpoint.pt`,
`snapshot_0050/checkpoint.pt` and `stopped_checkpoint.pt` beneath the new outputs.
Retain partial temporary files and the last honest completed/partial state on
failure; never silently discard evidence to make an export fit.
An explicit stopped-state alias may replace only the final stopped duplicate
after50 committed updates: it must bind the retained checkpoint50 SHA/size and
an exact current committed model/Adam/RNG/sampler/rates/counters digest. It is
forbidden when an optimizer is in flight, any microbatch is partial, a stopped
checkpoint already exists or any checkpoint temporary remains. An uncertain
failure cannot use this alias. Initial and update50 checkpoints remain real files.
Write checkpoint50 only after every snapshot50 neural call has completed.
Checkpoint50 and an actual stopped checkpoint are mutually exclusive: a failure
before checkpoint50 retains the actual stopped state, while an exact committed
post-evaluation state uses checkpoint50 plus its verified stopped alias. No new
reservation, RNG mutation or model operation may occur between checkpoint50 and
the terminal/alias proof. A failed partial write remains in the same updated-state
slot; do not create another updated checkpoint copy around it.

Evaluate all406 unchanged DEV cases at both snapshots:53 saved validation
references times six profiles plus88 comparisons, including40 historical paired
and48 native unpaired cases. Save all812 RAW/PNG pairs and metric/input/mask
joins. New snapshot0 must reproduce the retained full-coarse outputs exactly.
The normal preservation baseline remains the retained untouched original0;
substituting the new full-coarse snapshot0 would hide its existing regression.
Record snapshot0's known gate failures without aborting before the experiment.
Snapshot50 must pass the unchanged61-case clear MAE and53-case normal-blur
absolute luminance-bias/coarse-RGB-MAE rules in RAW and PNG separately: mean
regression<=0.01 and worst per-case regression<=0.03 against original.

Retain the existing additional dark rule separately: for each53-case lowlight
and compound validation cohort, delivered PNG mean MAE must not exceed the
full-coarse parent and mean SSIM must not fall. Record RAW dark metrics too.
These are prospective stopping/preservation controls, not the historical one
percent overall restoration qualification gate. Native clean/identity accuracy
remains unmeasured. All planned visual sheets and an independent returned-output
audit remain required before any usefulness or model/app decision. FINAL is
reserved; no gate threshold or split is altered.

Only a manual Linux NVIDIA L4/tmux launch in the exact new VM root may import
the neural runtime. Verify all sources and parent hashes before framework
imports. Parent assets and historical failure evidence remain read-only.
The physical dependency set is a selected subset of the hash-bound2011-file
historical catalog; the complete catalog is metadata, not a demand to read all
unused sources. Bind all1624 historical original/base RAW/PNG witnesses plus
the100 selected HQ cache triplets, two metric files, runtime and checked export
manifest. Keep the packet's synthetic empty `research/__init__.py` explicit.

Admission requires16GiB free after installation and a4GiB reserve. Caps are
worker3000s, initialization600s, witness180s, training900s, each snapshot600s,
preflight300s and export600s, with supervised30s external grace. RSS<=8GiB,
CUDA reserved<=22GiB, aggregate result payload<=4GiB, archive<=4.25GiB,
temporary checkpoint bytes<=1GiB, metadata<=128MiB and regular files<=7000.
Initial checkpoint<=384MiB; trained/stopped checkpoint<=832MiB each.
The three canonical checkpoint names allow at most two physical state slots:
one384MiB initial and one832MiB updated final-or-temporary slot, totaling1216MiB.
812 RAW/PNG pairs can reach
roughly0.75GiB before behavior targets, HQ cache and metadata. A source-derived
full output/file/archive/disk ledger must close before sealing. This contract
does not claim that a measured runtime peak or complete byte projection exists.
Use prospective per-call/per-write guards and retain failures if a bound is hit.

The target remains useful visible-structure-preserving256x256 face restoration
with credited untouched CodeFormer benchmark, sufficient-evidence refusals and
separate native/paired evidence. This pilot is not completion, hidden-identity
recovery, ethnicity/Zamboanga evidence, user acceptance or overall goal completion.
