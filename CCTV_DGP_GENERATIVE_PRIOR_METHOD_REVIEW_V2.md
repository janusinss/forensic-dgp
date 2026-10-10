# DGP method decision and verified generative-bank component — 10 October 2026

Status: interpretation selected; standalone prior component acquired and
independently checked. A trained conditioned restoration candidate and matched
manual VM comparison are not ready. The current checkpoint and application
remain unchanged. This review is the next preparation decision under
CCTV_DGP_COMPARATIVE_IMPROVEMENT_PLAN_V2.md, not a completed quality milestone.

## What the current model actually implements

The retained checkpoint is a fine-tuned DeblurGAN-v2-compatible MobileNet/FPN
conditional restoration network. Its executed forward builds an input feature
pyramid, combines convolutional heads, and adds a clipped RGB residual to the
observed input. It does not call a separate face-generating prior. The
`DGPSynthesizer` class name remains for compatibility; its name and old docstring
do not establish the published DGP method or useful CCTV quality.

Fresh strict loading confirms **181 unique parameter tensors /3,312,707
elements**, and **622 state entries** including buffers and shared feature-path
aliases. Calling the latter “622 parameters” is inaccurate. The original
reconstruction section has14 parameter tensors/609,219 elements. The later
calibration graph has15 selected tensors with the same element count because
fusion is split and a repaired deep branch is represented separately. It is
not an unmodified fourteen-tensor decoder.

Keep the permanent app comparator:
`outputs/cctv_dgp_face_code_fit_vm_v12_r2/weights/dgp_v2.pth`, SHA256
`646fbb11674e8882ea41e251d6e6e7d468c963979b48c93132def8292834e24b`.
Its selected continuation added two epochs/226 updates; lifetime epochs remain
unconfirmed. Other rejected versions are not accumulated training of this file.
Its weights-only file is not a complete optimizer-resume state.

Published DGP uses a pretrained generative model as an image prior and adapts
the latent/model for restoration. That establishes a method distinction, not
a claim about our CCTV performance. [Original DGP paper](https://arxiv.org/abs/2003.13659),
[authors' implementation](https://github.com/XingangPan/deep-generative-prior).

## Selected comparison and why

The user answered “apply the best approach here.” Select an **own-trained
input-conditioned hybrid generative-prior restorer** for the replacement route.
Reuse useful current observation/reconstruction weights. Declare the external
face-generating bank separately from our learned conditioning and reconstruction.
This is a proposed DGP adaptation; it is not an exact reproduction of Pan et al.
or a claim that we trained the external generator ourselves.

| Route | Preparation decision | Required evidence |
| --- | --- | --- |
| A: corrected current copy | Retain the repaired current reconstruction graph; add always-active clear and blur RGB target anchors to the existing normalized reconstruction objective. | Fresh finite manual learning must preserve raw/PNG appearance and improve useful DEV detail. Saved initial slopes are insufficient. |
| B: conditioned generative bank | Use retained camera feature maps to condition a frozen continuous StyleGAN2 face bank; join its multiscale features to our trained256 reconstruction path. | Verify initial behavior, conditioning learning, actual bank contribution through ablation, resources, and the same preservation/usefulness checks. |

For A, the selected contrast adds fixed unit-coefficient clear and blur target
MSE terms normalized by the saved clear/blur TRAIN baseline means. It retains
MSE/SSIM/fixed-recognizer/HF terms and the original guards. No new coefficient
search or local gradient calculation occurs. The saved contrast still has
adverse per-reference slopes, so it cannot qualify an optimizer or extended
epoch schedule.

GLEAN motivates an encoder–generative-bank–decoder connection with multiscale
features and a single inference pass. The new route will use that connection
as a design reference, rather than its pretrained restoration checkpoint.
Its published super-resolution evidence does not demonstrate CCTV performance
or justify recovering exact missing identity. [GLEAN paper](https://arxiv.org/abs/2012.00739),
[official architecture](https://github.com/open-mmlab/mmagic/blob/0a560bba9b79ebe78574e1d4cbbdd0e798e63568/mmagic/models/editors/glean/glean_styleganv2.py).

This decision addresses the previous failures rather than repeating them.
The old discrete CodeFormer conditioning experiments missed preservation and
generalization; V20's clean-target-informed `DGP+VQ(target)-VQ(input)` correction
failed both motion cases and visibly changed tones/texture. Neither discrete
code classification nor that RGB subtraction is selected here. Continuous
input-conditioned feature fusion remains a hypothesis requiring training.
Resetting an entire face generator from random weights has no established data
or compute feasibility in this project, so it is not the selected starting point.

## Acquired component and checks actually completed

The component is the official **standalone**
`StyleGAN2_512_Cmul1_FFHQ_B12G4_scratch_800k.pth` generator from TencentARC's
GFPGAN release, documented as a training prior. It is **not GFPGAN's trained
restorer**. No GFPGAN/GLEAN restoration encoder or restoration checkpoint was
acquired. [Official prior download documentation](https://github.com/TencentARC/GFPGAN/tree/7552a7791caad982045a7bbe5634bbf1cd5c8679#computer-training).

The file has204,535,545 bytes and SHA256
`05f5d33d79b32a3355cae3ede30e7ee06a90e56c60b1c2efe4ddd0d0e5a2959f`.
Official release metadata matches its name, byte count and asset ID. This older
release supplies no publisher digest: the locally computed hash binds our copy,
not an independently published checksum. The selected `params_ema` has139
state entries/124 unique parameter tensors/24,860,935 elements. All remain
frozen. Source, licences, release metadata and bytes are retained under
`outputs/cctv_dgp_generative_bank_source_v1/`.

The finite local component probe took24.32 seconds with four CPU threads. It
used generator seeds0 and1 plus a seed0 repeat: six generator forwards, zero
CCTV/photographic inputs, gradients, backwards or optimizer updates. Native
generator RGB is512; separate bilinear256 display arrays and floor-quantized
PNGs are retained. Internal16/32/64/128/256 feature maps have512/512/256/128/64
channels. Extracting features left the original generator RGB path exactly
unchanged; the fixed-noise repeat was exact.

The independent checker took7.56 seconds. It verified31 output and11 source
bindings,11 unchanged generator source bodies, the exact upstream CPU filter
body,14 NumPy operator fixtures, three raw/display/PNG compositions, one fresh
CPU generator replay and five feature replays. Replay error was zero; operator
error was2.22e-16 and independent bilinear recomputation differed by at most
5.22e-8. Seven protected checkpoint/source/protocol/report hashes remained
unchanged. The public component API refused local gradient-enabled execution.

The source adaptation retains original generator operations and FIR filtering;
only imports/registration and pure PyTorch operator adapters change. It does
not substitute bilinear filtering into the original GAN or silently convert
weights to a different clean architecture. CUDA equality, GPU gradients,
runtime and VRAM are **not yet verified**.

The one512×256 seed sheet was inspected at original resolution. It shows
face-like eyes, nose, mouth, eyewear, hair and skin detail, with some texture
artifacts. These are unconditioned generated faces, not reconstructions of a
person or quality evidence for our model. No ethnicity or nationality is inferred.

Evidence: `outputs/cctv_dgp_generative_bank_probe_v1/plan.json`, `results.json`,
`manifest.json`, `outputs/cctv_dgp_generative_bank_probe_v1_independent_audit.json`,
and `outputs/cctv_dgp_method_comparison_v2/selection.json`.

## Terms, overlap and next execution boundary

Retain the BasicSR/GFPGAN licences and their third-party notices plus NVIDIA's
noncommercial StyleGAN2 terms with any derived packet. This is a research
component, with no unrestricted commercial-deployment claim.
[GFPGAN licence and notices](https://github.com/TencentARC/GFPGAN/blob/7552a7791caad982045a7bbe5634bbf1cd5c8679/LICENSE),
[NVIDIA StyleGAN2 licence](https://github.com/NVlabs/stylegan2/blob/bf0fe0baba9fc7039eae0cac575c1778be1ce3e3/LICENSE.txt).

FFHQ is the declared prior-training corpus, not native CCTV or an Asian capture
source. Its corpus source overlaps our FFHQ TRAIN source; exact person-level
pretrained overlap is unexcluded. Own TRAIN/DEV/final separation does not prove
disjointness from this external corpus. Keep dataset-level terms distinct from
individual image licences. This acquisition downloads weights and code, not new
FFHQ photographs. [Official FFHQ provenance and terms](https://github.com/NVlabs/ffhq-dataset/tree/4826aa6ea77aa7f1a7802b938ed7c40afb985cda#licenses).

Next implement the **conditioned** B copy and matched independent return checker.
Declare retained/new/frozen tensors and initial raw/PNG behavior. Freeze identical
inputs, roles, objective/exposures, snapshots, bank ablation and scientific gates.
Measure finite L4 resource projections before allowing any optimizer; retain
failure and full model/optimizer/scheduler/RNG states on every stop. Only then
issue verified transfer/tmux commands for the human's manual run. The component
probe is not a training packet and does not justify extra epochs by itself.

The original1% early/10% later structure criteria and MSE/SSIM/embedding,
brightness and visible preservation checks remain. Raw and delivered PNGs,
paired synthetic and unpaired native evidence stay separate. Final identity
pixels were not accessed. The DGP-primary app, rolling accepted incumbent,
independent final review and all seven automatic/assisted covering families
remain unqualified. Completion needs separately trained/reviewed masked inputs;
a generative bank or restoration win alone cannot qualify it. Full goal active.
