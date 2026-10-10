# DGP learning-path and rate review — 10 October 2026

The Head4 capacity return is independently audited and rejected. Preserve that
run, its original 1% requirement, every raw/PNG result and its stopped optimizer
state. The current app checkpoint remains Phase3 plus two selected identity-v2
epochs/226 updates; its complete ancestral epoch count remains unconfirmed.

The next investigation compares learning through the complete original RGB
decoder with learning through the repaired deepest branch alone. It starts from
an isolated current-checkpoint copy with exactly the same initial output. It
does not introduce an external pretrained face generator or resume a failed
checkpoint. All visible facial features, the five milestones, the proposed
additional-epoch 1/2/5 study and all seven completion families remain required.

## New evidence and the assumption it retires

The saved-gradient worker uses all 160 existing initial VM vectors. For the
repaired route, it evaluates one averaged first-Adam direction and 20 individual
reference-batch directions against all 20 references/four loss components. No
model forward, new gradient, optimizer, parameter assignment or training occurs.
An independent elementwise-product implementation verifies all 1,680 products;
maximum recomputation difference is 6.3949e-14 and every sign decision agrees.

The averaged direction predicts descent in all four components for both TRAIN
cohorts and both photographic sources. In contrast, 8/20 individual directions
predict increasing detail error in at least one source/cohort group. Of the 20
directions, 19 predict increasing at least one group component, 17 predict worse
detail on another reference, and one predicts worse detail on its own reference.

**Retire the assumption that a favorable averaged initial-gradient calculation
validates the implemented per-reference stochastic training recipe.** These
initial-state slopes do not prove which steps caused the observed finite
failure. The 100 cases are historically exposed TRAIN evidence, not independent
validation or native CCTV. Actual Adam moments, changing activations, clipping,
regression penalties and delivered-PNG rounding still need finite checking.

The selected historical head rate was 1e-5. At that rate, the four sampled-group
first-step linear detail-gain estimates are only 0.00051%–0.00248%. Larger rates
scale these initial slopes, but their actual outputs and preservation are
unverified. These values must not be extrapolated across 50 steps or epochs.
They justify a finite rate-response diagnostic, not a claim that 1e-4 or 1e-3
will pass. There is no demonstrated optimal learning rate yet.

## Spatial-path limitation to test

The retained forward traces show the original heads at 64×64, 32×32, 16×16 and
8×8. The repaired pilot trained only the 8×8 head's two kernels and its connected
fusion slice. The other heads and later RGB reconstruction convolutions stayed
frozen. The official architecture combines all four scales and uses additional
reconstruction layers. This supports testing its complete reconstruction path;
it does not prove that coarse features alone are the unique cause of failure.
[Official DeblurGAN-v2 FPN implementation](https://github.com/VITA-Group/DeblurGANv2/blob/master/models/fpn_mobilenet.py)

The proposed complete decoder includes all four original heads, both independently
owned fusion slices, fusion bias, second smoothing convolution and final RGB
projection: 15 tensors with the same 609,219 effective original decoder elements.
The split fusion ownership protects the stored original tensor and avoids
updating an unused slice through weight decay. The MobileNet/FPN feature
extractor, five stored normalization statistics, fixed anchors and recognizer
remain frozen. This is the complete original RGB decoder, not whole-model
training. A new initializer must demonstrate exact inference parity before any
optimizer and finite, nonzero gradients for every selected tensor on the L4.

NAFNet's controlled ablations support testing restoration design and rate changes
with matched controls. Its GoPro/SIDD results do not establish useful CCTV face
restoration, qualify our decoder or justify importing its pretrained weights.
[Chen et al., Simple Baselines for Image Restoration](https://arxiv.org/abs/2204.04676)

## Data remains a separate unresolved limitation

The review joins every one of the 3,905 already audited TRAIN input/output records.
Retained-DGP landmark high-frequency error is slightly worse than the input mean
for the `dataset/asian_faces` blur and compound groups, while its motion group
improves approximately 3.93% relative to input. HQ FFHQ degraded groups improve
approximately 0.36%–2.51%. These are paired TRAIN filter errors, not useful-image
ratings, recovered identities or native CCTV performance. Input filtering uses
float64 RGB conversion; model errors use saved float32 outputs with the same
support/kernel. The small conversion distinction is recorded explicitly.

Native development eye spacing is 23.02–47.04 pixels. Most retained severe replay
sampling grids have 4.55–15.80 pixels between eyes, with partial overlap through
larger motion cases. This measures geometry, not equivalent captured detail.
Before a longer recipe is chosen, review realistic degradation coverage against
the frozen native input criteria; do not label native CCTV as a clean target.
Research supports examining synthetic-to-real degradation mismatch, without
proving it caused this project's failure or promising a transferable result.
[Wang et al., Real-ESRGAN](https://openaccess.thecvf.com/content/ICCV2021W/AIM/html/Wang_Real-ESRGAN_Training_Real-World_Blind_Super-Resolution_With_Pure_Synthetic_Data_ICCVW_2021_paper.html)

Dataset directory names do not establish ethnicity or capture country. Native
CCTV stays unpaired. Original roles, provenance, terms, exposure history and
incomplete ancestral overlap knowledge remain. No validation or reserved-final
identity pixels were opened by this review. Additional acquisition is justified
only for an audited, demonstrated gap, with Asian capture sources prioritized
where available.

## Next finite manual diagnostic

Prepare a controlled calibration on the existing L4: compare deep-only versus
complete-original-decoder learning at rates 1e-5, 1e-4 and 1e-3. Use the same
initializer, normalized four-term objective, clipping, coupled weight decay and
balanced accumulation of ten reference batches in each of the two existing
TRAIN pools. This is a factorial diagnostic: partition comparisons match rate
and data; rate comparisons match partition and data. It changes aggregation
and investigates the fine path, rather than relaunching the failed 50-update
recipe or extending its stopped state.

Each of the twelve independent arms starts from the original initialization
and has one actual optimizer update/50 fitting exposures. Include zero-update
parity and real per-profile gradient evidence for blur/clear preservation and
RGB-mean changes. Raw floats, PNGs, original masks, all model/optimizer/scheduler/
RNG/schedule states, source/environment hashes and every failure must return.
Native development outputs are separate unpaired review evidence; final
identities stay outside fitting, rate selection and visual tuning.

The original 1%/10% capacity and appearance thresholds stay unchanged. A
one-step calibration cannot qualify a model or authorize a failed-state resume.
Any subsequent capacity pilot must start again from the declared initializer,
be materially justified by independently audited finite outputs and pass the
original full-TRAIN and native development requirements before additional
epochs 1/2/5 are considered. No automatic follow-on, historical pilot execution
or app promotion is permitted.

The new packet is prepared and independently transfer-checked. Its bound
protocol limits twelve updates/120 backwards/280 gradient queries, 1800 seconds
of model work, 600 seconds of export, 2.5 GiB output, 20 GiB allocated VRAM and a
1 GiB disk reserve. Require 7 GiB free after installation and a measured runtime
projection before fitting. CPU parity is exact for all 100 cases; the independent
replay checks ten cases and separate learning ownership. L4 gradient connectivity,
finite learning, all 64 planned visual pages and usefulness remain pending.
The prepared incoming audit is source-bound. See
CCTV_DGP_MULTISCALE_CALIBRATION_V1_VM.md for manual start/upload/tmux/download
commands. The VM was API-verified stopped; no current free-space value is assumed.
Actual training remains manual inside tmux; direct VM access is maintenance only.

Evidence: `outputs/cctv_dgp_head4_learning_signal_review_v1/plan.json`,
`results.json`, `directions.json` and `independent_audit.json`. Model and app
selection are unchanged. The complete goal remains active and incomplete.
