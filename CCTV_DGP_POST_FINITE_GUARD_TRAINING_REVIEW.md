# Training review after finite-guard R1 and the fixed conversion diagnostic

The current DGP remains the starting checkpoint. Three initial common-direction
proposals accept no parameter changes; nearest rounding improves float-to-byte
agreement but still qualifies none. More epochs of that unchanged recipe are
not an evidence-based remedy. These studies do not prove that the original DGP
cannot learn; they show that the tested common direction and small placements
do not provide sufficient preserved, useful reconstruction.

The retained checkpoint represents original Phase 3 plus two selected identity-v2
fine-tuning epochs (226 selected-branch updates). Its complete ancestral epoch
count is not established. Failed later pilots and zero-accepted-change studies
do not add epochs to it. The previously proposed additional epochs 1, 2 and 5
remain a design target, with a five-epoch maximum, not a ready training command.
The immutable current-training improvement plan and all historical failures remain.

## Evidence and the remaining cause

The 64 initial raw objectives admit an improving common direction, and every
finite-guard raw preservation comparison passes. Delivered quantization changes
recognition and high-frequency measurements enough to change failing groups.
Nearest conversion removes most bias but retains recognition regressions and
introduces negative source structure gains. Both conversions have extremely
small structure gains, below 1%. The original images remain visibly soft.
Thus quantization contributes to the measurement discrepancy, while a useful
learned reconstruction improvement remains undemonstrated. A unique architecture,
normalization, objective or data cause has not been isolated.

Source inspection confirms that the current synthesizer already has four FPN
heads, low-level spatial fusion and an RGB input residual. Adding a generic
input skip, declaring more trainable tensors, or renaming the same added decoder
would not constitute a demonstrated new remedy. The previous 1,996,035-parameter
original-path copy already establishes nonzero gradients and initial parity;
these do not establish useful finite outputs. Existing coarse PNG gradients and
empirical-clearance trials V37/V38 also failed to qualify native/paired development.
They must not be repeated unchanged or promoted because a small TRAIN trial passes.

## Primary research and what it permits us to infer

DeblurGAN-v2 introduces an FPN generator for motion deblurring and supports
different backbones. This matches the inspected architectural lineage, but its
benchmark results do not establish forensic face preservation for our checkpoint.
[Original DeblurGAN-v2 paper](https://arxiv.org/abs/1908.03826).

PromptIR uses degradation-conditioned prompts to guide a single restoration
network across denoising, deraining and dehazing. It supports investigating
input-dependent conditioning as a design principle; it does not validate a
particular face decoder, clear-image bypass, or CCTV identity result.
[NeurIPS proceedings abstract](https://papers.nips.cc/paper_files/paper/2023/hash/e187897ed7780a579a0d76fd4a35d107-Abstract-Conference.html).

Real-ESRGAN expands synthetic degradation through repeated degradation stages
and considers ringing/overshoot. This motivates auditing whether training
degradation coverage represents our native inputs. It does not provide paired
truth for our CCTV or justify adopting adversarial texture synthesis for identity.
[Original ICCV workshop paper](https://openaccess.thecvf.com/content/ICCV2021W/AIM/html/Wang_Real-ESRGAN_Training_Real-World_Blind_Super-Resolution_With_Pure_Synthetic_Data_ICCVW_2021_paper.html).

These are primary-paper/abstract findings, not reproductions. The NeurIPS full
PDF exceeded the browser limit, and CVF direct opens were blocked; the cited
Real-ESRGAN proceedings abstract was available through indexed primary content.
No new pretrained weights or external model outputs are used as clean targets.
No population, ethnicity or Zamboanga performance inference follows from them.

## Selected next design hypothesis

Investigate **broader realistic degradation coverage and input-conditioned
spatial refinement of a separate current-DGP copy**. This is a hypothesis requiring
measurement and an ablation, not a proven best model or a prepared VM pilot.
Conditioning must depend only on the observed crop; clean targets, dataset/source
names, identity labels and synthetic-profile labels cannot enter inference.
All visible facial features remain in scope. Clear glasses and ordinary hair
must remain. Completion of hidden areas remains a separate component.

Before writing an executable training packet, complete these five bounded tasks:

1. Audit existing input-only blur/noise/resolution/framing coverage for the
   exposed TRAIN cases and frozen native development crops. Native file dimensions
   alone do not establish effective detail. Retain input-only usability labels;
   model softness cannot relabel usable crops as insufficient.
2. Determine whether observed-input conditioning can reduce interference between
   clear preservation and degraded reconstruction. Preserve full reference groups
   and every old failure. Test routing separability rather than assuming clear
   and degraded images can be perfectly recognized by a threshold.
3. Specify the changed spatial path, parameter ownership, multiscale reconstruction
   supervision and normalization policy. Establish exact initial parity against
   the current DGP. Keep paired targets outside inference and avoid substituting
   a pretrained restorer for our trained primary model.
4. Freeze a finite manual L4 ablation with the original raw/PNG preservation,
   source/profile, brightness and structure requirements. Retain the original
   1%-at50/10%-at800 capacity decisions where applicable; declare any additional
   schedule explicitly. No rejected pilot is resumed or automatic continuation
   allowed. Freeze timing, storage, VRAM, update/exposure and export caps first.
5. Prepare the additional-epoch 1/2/5 study only after that ablation justifies it.
   Compare native development visually and paired development numerically;
   preserve reserved final identities. Export a portable optimizer/scheduler/RNG/
   schedule/data/provenance state for any successful continuation. Independently
   audit returned files before selecting a checkpoint or changing the app.

Task 1 now has an initial bounded result on the exposed100 TRAIN and all24 native
development inputs: projected degraded TRAIN eye spacing is5.38–21.40 sampling
pixels, compared with23.02–47.04 native annotated pixels. This is a geometric
coverage finding, not an equivalence of resolved detail or a full-corpus audit.
The current Auto rule selects all10 asian_faces clear cases but none of10 FFHQ
clear cases, so it cannot guarantee a clear-image bypass. No threshold is fitted
and all24 native usable labels remain. These findings justify examining additional
resolution/framing coverage from genuine HQ TRAIN references while retaining
legacy profiles and replay anchors. They do not yet justify an executable pilot.
See CCTV_DGP_POST_FINITE_GUARD_INPUT_COVERAGE_V1_RESULTS.md for exact measurements,
sampling limitations and the older transitive-source binding omission.

This review prepares the next investigation; it does not freeze numerical gates,
loss weights, learning rate, a conditioning threshold or trainable new modules.
There is no transfer archive or launch command for this hypothesis yet. Existing
VM maintenance authorization does not authorize automatic training. All actual
training and autograd studies remain manually launched in tmux on the existing L4,
until the user identifies another VM. Credits are user-reported, not queried here.

The current checkpoint, retained Phase 3 checkpoint, app design, source data,
split roles, terms, provenance, caches and failed gates stay preserved. All five
milestones remain binding, including useful native outputs, independent final
review and bundled inline Playwright. Masks, sunglasses, glare, hands, obstructing
hair, scarves and objects require separate automatic/assisted completion review,
removal-area preview/correction, preserved visible appearance, clearer/less-covered
requests and original/mask/plausible-result downloads. No exact hidden identity
claim is permitted. The complete goal remains active/incomplete.

Evidence:

- [Finite-guard return report](<C:/xampp/htdocs/YEAR 4/Testing/CCTV_DGP_FINITE_GUARD_V1_R1_RESULTS.md>)
- [Quantization report](<C:/xampp/htdocs/YEAR 4/Testing/CCTV_DGP_FINITE_GUARD_QUANTIZATION_V1_RESULTS.md>)
- [Retained training-history plan](<C:/xampp/htdocs/YEAR 4/Testing/CCTV_DGP_CURRENT_TRAINING_IMPROVEMENT_PLAN.md>)
- [Original spatial-path inventory](<C:/xampp/htdocs/YEAR 4/Testing/CCTV_DGP_POST_V42_ORIGINAL_PATH_INVENTORY.md>)
- [Rejected V38 development evidence](<C:/xampp/htdocs/YEAR 4/Testing/CCTV_DGP_V38_QUARTER_DEVELOPMENT_RESULTS.md>)
