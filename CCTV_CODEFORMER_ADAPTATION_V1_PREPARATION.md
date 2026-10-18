# CodeFormer V1 return audited: quality-stopped at 200 updates — 15 October 2026

**CURRENT: 200/1200 optimizer updates; 800 TRAIN exposures; preservation failed.**
The first fine-tune has visible improvements and mixed profile metrics, but is
not an accepted replacement. One clear control exceeds the unchanged individual
RAW/PNG bound. Do not relaunch the V1 commands or resume its stopped state unchanged.

[Returned result and diagnosis](CCTV_CODEFORMER_ADAPTATION_V1_RETURN_REVIEW.md).
[Every comparison page](outputs/codeformer_adaptation_v1_return_review/visual_sheets/INDEX.md).
All 52 pages / 406 case pairs were reviewed; user acceptance and reserved-final
evaluation remain pending. Native CCTV remains unpaired and unqualified.

All 350 trainable tensors changed; 165 frozen restoration tensors stayed exact.
The archive, checkpoints, failed guard and splits remain preserved. The measured
next design addresses exposure/loss imbalance using TRAIN-only data, with the
same preservation requirements. No new training has launched, no candidate was
adopted, and further completion/detector development remains deferred.

**The launch-ready / PREPARED / UNRUN text below is retained historical evidence.**
This current status supersedes it. The full thesis goal remains active.

---

# VM prerequisites verified for manual CodeFormer V1 — 15 October 2026

The user's trusted ED25519 host fingerprint matches the presented VM key.
Read-only guest inventory confirms **78 GiB free**, an idle NVIDIA L4, Python
3.10.12, Torch 2.9.1+cu129, NumPy 1.26.4, Pillow 11.3.0 and OpenCV 4.11.0.86.
There is sufficient space for the verified archive, unpacked packet and the
16 GiB post-install requirement. No cleanup or dependency upgrade is needed.

Proceed manually through steps 1–4 in
[CCTV_CODEFORMER_ADAPTATION_V1_VM.md](CCTV_CODEFORMER_ADAPTATION_V1_VM.md).
Step 5 is for downloading after export completes, including a stopped run.
Actual CUDA/import, gradient, parameter-change, timing and quality checks remain
inside the manual finite worker. No training, optimizer or model inference was
launched by this inventory. The archive/protocol and original checkpoints are
unchanged. The previous pending-host-key state below is historical.

Evidence: `outputs/codeformer_training_readiness_v1/runtime_inventory/r3_verified_host/`.

---

# Execution packet independently verified —15 October 2026

**PREPARED / UNRUN.**36 local tests and independent 779-member archive/copy audits pass.
No actual gradients, optimizer updates or restoration qualification.
[Manual commands](CCTV_CODEFORMER_ADAPTATION_V1_VM.md).
The complete finite design below is unchanged; its earlier packaging status is historical.

---

# CodeFormer restoration adaptation V1

Status: **SOURCE REVIEWED /36 LOCAL TESTS PASS / READY TO PACKAGE / NOT LAUNCHED**.
This is a separate supervised adaptation of CodeFormer, not a continuation of
the failed DGP optimizers and not an original model architecture. The original
restoration checkpoint remains the initializer and principal benchmark. No app
default, completion model, removal-mask detector or manuscript has changed.

## What this experiment tests

The user selected CodeFormer after reviewing its stronger restoration outputs.
This experiment asks whether genuine-detail paired supervision can adapt its
restoration paths to the declared CCTV-like degradations without losing visible
appearance. It uses the original degraded crop, not a Phase 3-restored
intermediate. Success requires reviewed outputs better than untouched CodeFormer;
more updates or a completed export alone is insufficient.

CodeFormer uses discrete code prediction and spatial feature fusion. Image loss
does not propagate through its hard token selection. The new worker therefore
uses explicit clean-code cross entropy, clean latent features, pixel loss and
fixed perceptual loss. It follows the roles in the [official joint training
implementation](https://github.com/sczhou/CodeFormer/blob/b33cc7d639d6545bfcccc7e0bc6ae51f24e79c2b/basicsr/models/codeformer_joint_model.py),
with the deviations below declared before training. The official eight-GPU,
150,000-iteration schedule is not copied onto the single L4.

## Frozen data and evidence roles

| Set | Count | Permitted use |
|---|---:|---|
| Approved historical FFHQ TRAIN | 391 genuine 1024×1024 sources | Paired optimization; clean 512 derived directly with LANCZOS |
| Approved historical FFHQ validation | 53 separate sources | Repeated development validation only; retained direct 1024→256 targets |
| Existing synthetic comparison DEV | 40 views of eight references | Comparison only, never optimizer data |
| Existing native CCTV DEV | 48 crops:24 QMUL,24 ChokePoint | Unpaired input/output review; no aligned target loss or PSNR/SSIM |
| Reserved native final identities | Existing frozen roles | No pixel access, training, tuning or selection in this pilot |

The 444 selected sources total 604,400,889 bytes. All 510 acquired originals were
rechecked against official file/pixel metadata; excluded sources stay excluded.
The 1024 dimension describes the official aligned file, not a guarantee that
every source face contains 512 independently captured pixels of optical detail.
The retained source review, rather than upscaling alone, guides target suitability.
The 390 retained `asian_faces` references have short edges 129–217 pixels and lack
a complete per-photo license chain. They remain historical controls, not newly
claimed high-resolution ground truth. Folder names imply no ethnicity.

FFHQ photos retain their individual source URL, author, license and modification
records in `references.json`. TRAIN includes CC-BY-NC photos; this is a
non-commercial thesis experiment. The FFHQ metadata notice and per-photo terms
remain separate from the CodeFormer S-Lab terms and LPIPS/TorchVision notices.
FFHQ's facial-recognition restriction remains: this worker trains restoration,
not a recognizer. See the [official dataset terms](https://github.com/NVlabs/ffhq-dataset#licenses).

Historical roles are preserved even where they differ from FFHQ's official
training/validation categories. Exact image and photo-URL overlap across our
roles is excluded; subject overlap is unestablished. CodeFormer was pretrained
on FFHQ, so these FFHQ validation images are not claimed unseen by its prior.
Native DEV has also been reviewed repeatedly. No Zamboanga samples or local
performance claim is introduced; Asian capture sources remain separately named.

Six deterministic camera profiles are clear, light, blur, JPEG, lowlight and
compound. Their recorded bounds reuse existing synthetic camera stress code,
not a fitted native CCTV distribution. Expected sampling proportions are 10%,
10%,20%,20%,20%,20%. There is no new crop, warp, occluder simulation or face
alignment. Clear glasses and non-obstructing hair stay part of the reference.
Generated observations below 32 pixels receive code/latent supervision without
direct image losses. This loss-routing rule is not a usable-input or quality gate.

## Model, targets and losses

The app comparison contract remains RGB256→bilinear 512→CodeFormer→RGB256.
Fidelity is 1.0, AdaIN isTrue, and output clipping precedes bilinear return to 256,
matching the selected untouched benchmark. Source computations match the pinned
upstream architecture after the declared logger/registry/import adapters.

Trainable parameters: **59,788,608** in encoder, position/feature embeddings,
transformer, token head and four spatial fusion modules. Frozen parameters:
**34,324,099** in the restoration generator and quantizer. The separate clean HQ
teacher and complete LPIPS network are also frozen. Generator operations remain
differentiable so image losses reach fusion weights; no detached prediction loss.

| Loss | Weight | Contract |
|---|---:|---|
| Clean token cross entropy |0.5| FP32 logits reduction against frozen HQ-teacher indices |
| Clean latent MSE |1.0| Encoder feature against frozen restoration-codebook features at those indices |
| Pixel L1 |1.0| Native 512 RGB units, unclipped prediction, eligible paired samples |
| LPIPS v0.1/VGG16 |1.0| FP32 raw normalized prediction and clean[-1,1] target, standard scaling once |

The official teacher and restoration codebooks are not byte-identical: maximum
entry difference 2.169609×10^-5, with 1024/1024 identical nearest matching row
indices. The failed exact-equality assumption is retained in
`initializer_preflight_failure_v1.json`. The worker verifies compatibility and
uses the upstream target roles: teacher indices, restoration dictionary features.
It changes neither original asset. Tiny EMA arithmetic drift is a plausible
explanation, not an established release-history fact.

Standard LPIPS is deliberately used instead of the upstream wrapper's additional
range/ImageNet transforms. Its compact VGG16 feature tensors are identical to the
official ImageNet backbone, and three CPU fixtures match `lpips==0.1.4` exactly.
The implementation preserves input gradients but freezes all perceptual weights.
Its own [source and license](https://github.com/richzhang/PerceptualSimilarity)
are retained. No new discriminator, adversarial objective or EMA is introduced
in this first adaptation. These are explicit deviations from full StageIII.

## Finite manual L4 schedule and stops

| Control | Predeclared setting |
|---|---|
| Maximum optimizer updates |1200; microbatch 1, accumulate 4, effective batch 4 |
| Maximum photographic exposures |4800; 391 unique TRAIN references; 12.276 reference-pass equivalents, not the original model's lifetime epochs |
| Optimizer |Adam, betas(0.9,0.99), epsilon1e-8, no weight decay |
| Learning rates |Encoder/token1e-5; fusion2e-5; 50-update linear warmup; cosine to2e-6 |
| Precision |L4 BF16 restoration autocast; FP32 clean token assignment and perceptual objective; no GradScaler |
| Snapshot updates |0,200,600,1200; 406 outputs and 61 clear controls each |
| Initialization/cache cap |600 seconds |
| First 20-update benchmark cap |600 seconds; updates remain part of the trajectory |
| Projected/actual optimization cap |7200 seconds; measured projection required after 20 updates |
| Snapshot cap |600 seconds each; fixed cohorts cannot be silently dropped |
| Whole worker cap |10800 seconds; external timeout and termination grace |
| Export cap |600 seconds; export is evidence packaging, not training success |
| Storage |Require 16 GiB free after installation; keep 4 GiB reserve; retain model/Adam/RNG states and failed snapshots |
| VRAM |Predeclared 22 GiB reserved-memory ceiling on the 24 GB L4; stop/export if exceeded |

The storage reserve covers three roughly 855 MB trained states, the update 0 state,
RAW/PNG outputs, a stopped state and the compressed return archive. The verified
input dataset/initializer is already local and is not duplicated in result exports.
Every return binds those retained inputs by hash. Shared VM caches stay untouched.

Before optimizing, the same manually launched worker checks hashes, geometry,
benchmark parity, ordered dictionary compatibility, frozen perceptual input
gradient and actual runtime details. At 20 updates it checks finite nonzero
module-level gradients, actual intended weight changes, timing projection and
VRAM. Individual inactive tensors need not all be nonzero on one batch.
The dictionary, generator, HQ teacher and perceptual state must remain identical
to their own initialization hashes at every snapshot and export decision.

Stop/export for nonfinite values, frozen mutation, missing learning paths,
resource limits, manual signal or gross clear-preservation regression. The latter
is an additional abort guard, separately on RAW and delivered PNG: mean clear
MAE regression over same-VM update 0 greater than 0.01, or any case greater than
0.03, in RGB[0,1] units. These thresholds do not replace the historical 1% feature
gate or certify preservation. No acceptance requirement is relaxed after results.

## Review and qualification remain separate

The worker saves 406 cases at each declared snapshot:53 validation×six profiles,
48 native DEV and 40 retained synthetic controls. RAW256 and floor-quantized
nativePNG remain separate; original unobserved padding is restored only in PNG.
No display enhancement or palette operation is smuggled into the model result.
Paired MAE/MSE/PSNR are measured only against clean paired targets. Native scores
remain null where a clean reference would be required.

Returned evidence needs an independent local archive/state/trajectory audit,
replayed RAW/PNG measurements and all planned comparison-sheet inspection.
Recompute additional meaningful structure, perceptual and preservation checks
with their declared controls; historical gate failures remain preserved. Review
eyes, brows, nose, mouth, jaw, glasses, hair, brightness and expression against
the original input and appropriate paired reference. Report insufficient inputs
separately. A favorable average must not hide individual visible-feature loss.

The user must accept visible restoration improvement over untouched CodeFormer
before primary-model integration. Then freeze the selected checkpoint for
independent final evaluation and bundled inline Playwright Auto/override/PNG/app
checks. Completion and the seven covering families follow that restoration
acceptance. No model promotion, final-set result or completion qualification is
created by packet preparation or a finished training loop.

Fresh API inventory on 15 October 2026 confirmed the existing g 2-standard-4/L4 VM
**running after the user's manual start**. Guest free space, packages and active
workloads remain unverified because its current SSH host key is uncached; a
trusted fingerprint is requested. Actual gradient/optimization timing remains
unmeasured until manual launch. The 10800-second cap is a limit, not a runtime
estimate. No agent launches this experiment.
