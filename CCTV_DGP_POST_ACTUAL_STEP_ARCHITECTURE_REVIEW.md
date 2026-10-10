# Post actual-step design review

The complete finite-step diagnostic fails preservation on both fixed TRAIN
cohorts and shows no convincing extra facial clarity. The user previously
selected revising our DGP's spatial/feature path and authorized the best
research-informed approach. That decision supports this design review. It
does not authorize automatic VM training or an unchanged historical retry.

**Selected next direction: inspect reconstruction-first supervision of our
own spatial DGP decoder before another full training recipe.** Preserve the
original DGP, its normalization, initialization parity, source splits, failed
gates and the existing app. All visible facial features remain required.

## What the completed evidence establishes

The diagnostic includes ten saved V41 states, three parameter proposals and
3,150 distinct output slots. Recorded and closest-cone proposals fail raw and
PNG preservation on both fixed TRAIN cohorts at every state. Their degraded
landmark changes are very small and sometimes adverse. No new optimizer or
gradient query was needed to establish that result. It rejects the tested
proposal as sufficient evidence for continuing its training recipe.

The assumption to retire is that making the **current combined-loss applied
step** locally nonascending automatically supplies a useful structure-learning
step with acceptable finite outputs. The cone's current-batch local certificate
does not cover every group, finite nonlinear image response or PNG rounding.
It also seeks the closest displacement, rather than choosing a new direction
to prioritize visible reconstruction. These are different questions.

Earlier original-decoder V33–V38 evidence already tested restoration-only and
group/delivered-guard proposals at finite scales. Those failures remain relevant;
this review is not permission to repeat them with a new name. V40/V41 added the
spatial path but have not established its useful learned capacity. No unique
loss, optimizer or architecture cause is proved by these observations.

## Output geometry check completed locally

Before proposing a larger decoder or removing mean protection, the new bounded
saved-array check uses all existing 145 TRAIN cases once: 29 photographic
references, 29 clear and 116 synthetically degraded cases. It reads the saved
original-DGP output and the aligned clean TRAIN target; no neural, gradient,
backward or optimizer call occurs. No new DEV/native/final sample is selected.

For observed pixels, let `d = target - original_DGP`. The first oracle uses
`d - mean(d)` and clips the resulting image into `[0,1]`. The bounded oracle
first clips `d - (min(d)+max(d))/2` to `[-0.499999,0.499999]`, subtracts its
observed RGB mean and applies the same image clamp. Outside the observed region,
input pixels remain exact. Raw and floor-quantized PNG metrics stay separate.
These calculations relax the network's representational/learning constraints;
they are not predictions produced by our decoder.

| Degraded TRAIN oracle | Raw landmark error reduction | PNG landmark error reduction |
| --- | ---: | ---: |
| Mean-centered paired target correction | 96.5340% | 96.4448% |
| Bounded mean-centered paired target correction | 96.4534% | 96.3637% |

An uncapped mean-centered target correction has a bounded delta construction
for 88 of 145 cases. For the remaining cases, this particular uncapped
construction exceeds the chosen range; that does not prove no other clipped
construction exists. The bounded oracle still retains substantial aggregate
detail improvement. Maximum required target RGB mean shift is 0.3878903, so
mean protection can leave a brightness/color residual even in the oracle.

The independent checker verifies 870 case/variant/stage metric rows and 54
aggregates. It recomputes the high-frequency filter with OpenCV against SciPy;
maximum detail-MSE difference is 2.6020852139652106e-18. Pixel MSE and SSIM
recomputation are exact. The oracle has access to unavailable clean targets.
ArcFace, complete preservation, network representability and useful restoration
are not tested. None of its percentages belongs in a model-performance table.

**Inference from these results:** global mean centering and the bounded output
formula are not, by themselves, a sufficient explanation of the observed
near-zero detail gains. Learning the correction and preserving appearance
remain unresolved. Broadening the amplitude or removing the protected mean
has no demonstrated need from this check.

## Primary research and its practical limits

GEM's conversion from loss constraints to gradient inequalities assumes local
linearity around small steps and representative memory. Its reported tasks are
continual classification, not CCTV face restoration. This supports checking
finite images and the actual protected cohorts rather than treating our local
certificate as a global guarantee. [Lopez-Paz and Ranzato, GEM, section 3](https://proceedings.neurips.cc/paper_files/paper/2017/file/f87522788a2be2d171666752f97ddebb-Paper.pdf)

NAFNet demonstrates a simple gated architecture on image-restoration benchmarks.
Our spatial block is inspired by that design; it is not the published network
and uses none of its pretrained restoration weights. The paper supports a
testable spatial reconstruction path but does not establish that our 17,952
parameter decoder has sufficient capacity or useful CCTV performance.
[Chen et al., Simple Baselines for Image Restoration](https://arxiv.org/abs/2204.04676)

The perception-distortion paper distinguishes reference distortion from
perceptual image quality. It supports reporting paired metrics and visual
structure separately. It does not authorize identity changes, relaxation of
preservation gates or claims that our current model is at a theoretical optimum.
[Blau and Michaeli, The Perception-Distortion Tradeoff](https://openaccess.thecvf.com/content_cvpr_2018/html/Blau_The_Perception-Distortion_Tradeoff_CVPR_2018_paper.html)

## Conditions before another executable pilot

The next design should isolate **direct supervised visible reconstruction** in
the spatial path from the combined regression-penalty objective. Identity,
pixel/SSIM, clear-input anchoring, mean protection and visible-feature review
remain acceptance requirements. A diagnostic that cannot pass them does not
qualify by improving one term.

Preparation must resolve five concrete items:

1. Verify the proposed reconstruction objective and its normalization against
   saved paired targets; compare it explicitly with the failed V33 control and
   V40/V41 objective so the next recipe is materially different.
2. Use a separate initialized copy of our spatial decoder, frozen original DGP
   and unchanged TRAIN-only case schedule. Record the trainable tensors and
   exact starting parity before learning.
3. Specify finite optimizer/update, cache, inference, export, wall-time, VRAM
   and output-storage bounds, including an early measurable structure stop and
   raw/PNG preservation checks. An unsuccessful diagnostic retains its failure.
4. Freeze an independent return auditor and exact original checkpoint/split/
   source bindings, with a fresh manual-only transfer packet and tmux commands.
5. Require broader TRAIN capacity, paired DEV preservation, useful native
   development images and independent final review before app promotion.

This document is a **design decision and completed arithmetic diagnostic**.
It contains no new executable training protocol or transfer archive. No new
pilot is launched. More epochs, renamed copies of failed recipes or weakened
gates are not justified by the returned evidence.

All seven covering families remain a separate completion qualification problem:
masks, sunglasses, strong lens glare, hands, obstructing hair, scarves and
objects. Automatic and assisted results must be reported separately; preserve
clear glasses, ordinary hair and visible appearance, show the removal area for
correction, and request a less-covered crop when usable information is too
limited. The current DGP remains primary locally, but the full thesis goal is
active and incomplete.

[Completed finite-step findings](<C:/xampp/htdocs/YEAR 4/Testing/CCTV_DGP_ACTUAL_STEP_TAIL_V1_RESULTS.md>)
[Oracle independent audit](<C:/xampp/htdocs/YEAR 4/Testing/outputs/cctv_dgp_output_geometry_v1/independent_audit.json>)
[Earlier applied-step review](<C:/xampp/htdocs/YEAR 4/Testing/CCTV_DGP_POST_V41_OPTIMIZER_REVIEW.md>)
