# Next bounded DGP pilot: preparation decision — 3 October 2026

## Executable specification — 3 October 2026

The input gate and independent reconstruction are complete. V1's 90% full112
context requirement was disproved on tight crops with visible facial features;
it remains immutable. V2 requires >=50% context, >=95% feature-core support and
>=95% support around all five landmarks, retaining confidence/pose rules and
known input exceptions. It admits 917 training and110 validation references.
The 64 input previews identified two additional training exceptions (hand over
mouth; watermark over face). A separate reviewed manifest excludes those and
truncates by inherited ordering to **451/source, 902 training references**.
All110 qualified validation references remain (59FFHQ/51Asian-source), including
explicit original-quality warnings. Geometry is not an all-cohort pristine label.

The executable is `scripts/run_cctv_dgp_pilot_vm.py`, paired utilities are in
`cctv_dgp_pilot.py`, and preparation/auditors are versioned separately. The recipe
uses two precomputed camera epochs, approximately20% clear anchors, batch8,
226 updates per arm (452 total), and90 minutes including preflight/validation.
Both arms start at retained Phase3 and differ only in identity coefficient0/0.1.
The common loss is observed Charbonnier +0.05 pooled color +0.1VGG +0.05Sobel;
no FAN, FFT, EMA or AMP. Normalization running statistics are held fixed.
Adam uses backbone2e-6/head1e-5, decay1e-5, clip1. Preflight backward is VM-only,
batch8 and zero optimizer updates; local preparation/tests remain forward-only.

Targets/input/observation PNGs and target-derived fixed112 transforms are frozen.
Unsupported identity-crop context is filled identically with RGB128. New metrics
evaluate exported PNGs and use MSE-derived aggregate PSNR and the newly named
`ArcFace_observed_fixed`; do not compare them directly to prior raw/unmasked
numbers. Every source/profile and clear-control MSE/SSIM/cosine must preserve
starting-baseline agreement; degraded MSE-derived PSNR must gain>=0.1dB over the
current best. Epoch0 remains selectable. Native/visual review still decides
usefulness; no automatic application promotion.

Local package: `C:\xampp\htdocs\YEAR 4\Testing\outputs\cctv_dgp_vm_bundle_v1\`.
VM extraction: `~/forensic-dgp/cctv_dgp_vm_bundle/`.
Archive: `C:\xampp\htdocs\YEAR 4\Testing\outputs\cctv-dgp-vm-bundle.tar.gz`
↔ `~/cctv-dgp-vm-bundle.tar.gz` for upload. Exact setup/tmux/transfer commands:
`C:\xampp\htdocs\YEAR 4\Testing\CCTV_DGP_VM.md`
↔ `~/forensic-dgp/cctv_dgp_vm_bundle/CCTV_DGP_VM.md` after extraction.
Preparation and independent archive verification must finish before transfer;
**actual CUDA compatibility and trained output remain pending on the VM**.
Native reserved crops remain unused, and independent final review is pending.

## Earlier preparation decision (superseded by the specification above)

Prepare a targeted DGP experiment on the existing L4 VM after qualifying the
reference pool. Training is justified by the fixed camera-stress regression:
DGP smooths clear controls, darkens low-light inputs and lowers reference
agreement/identity cosine across the input-selected development cases. The
three-case alignment comparison did not establish a general processing fix.
CodeFormer remains a comparison; sharpening is insufficient for promotion.

**Current readiness: data pool audited, training implementation/bundle pending.**
This document is a bounded preparation decision, not an executable recipe or an
assertion that a new model will improve. Do not run historical Phase 5 unchanged
to stand in for this experiment. All optimizer/backpropagation preflights and
training run on the user's VM; local input audits/inference remain permitted.

## Locations and starting state

| Item | Windows local | Linux VM |
| --- | --- | --- |
| Project | `C:\xampp\htdocs\YEAR 4\Testing\` | `~/forensic-dgp/` |
| Retained start | `C:\xampp\htdocs\YEAR 4\Testing\checkpoints\dgp_zamboanga_final.pth` | `~/forensic-dgp/checkpoints/dgp_zamboanga_final.pth` |
| Original split | `C:\xampp\htdocs\YEAR 4\Testing\outputs\downloaded_phase4\outputs\phase4_with_progress\split.json` | `~/forensic-dgp/outputs/phase4_with_progress/split.json` |
| Fixed paired evidence | `C:\xampp\htdocs\YEAR 4\Testing\outputs\cctv_paired_regression_v1\` | `~/forensic-dgp/outputs/cctv_paired_regression_v1/` after transfer |
| Candidate reference pool | `C:\xampp\htdocs\YEAR 4\Testing\outputs\cctv_dgp_pilot_pool_v1\` | `~/forensic-dgp/outputs/cctv_dgp_pilot_pool_v1/` after transfer |

Start from Phase 3 weights, SHA256
`b6376f56c4161ef1f26a8f9efb0bb309b3014f4ecdeaca5399a47c1c4607586c`.
Keep the Phase 4 rejection history and pending original Phase 5 full-GPU status.
New experiment epochs start at 1; they do not continue an epoch 31 optimizer.

## Preparation stages

1. Derive a separate deduplicated pool manifest. Retain the candidate audit and
   inherited role/source membership; do not delete or relabel original images.
2. Freeze input-only reference-quality/pose and landmark-coverage gates. Separate
   strong profiles, obscured/closed-eye and already dim/noisy reference limitations;
   report source counts. Do not infer clean target quality from file readability.
3. Implement aspect-preserving camera augmentation and clear-image anchors in
   a versioned pilot dataset; preserve historical degradation code and metrics.
4. Freeze the complete matched experiment, GPU preflight, stopping criteria and
   independent return auditor. Verify source/model/code fingerprints and exports.
5. Package verified transfer files and exact tmux commands for the VM. Only after
   this gate is complete describe the experiment as available for training.

## Proposed matched diagnostic budget

Use at most 1,024 deduplicated/qualified inherited training references and 128
inherited validation references. The current raw pool is 512/64 per source, before
qualification; source balance is an experiment choice, not an ethnicity label or
established optimal ratio. Record any qualification losses and actual sampler
weights before running. Historical validation remains development data.

The intended comparison is **matched camera/clear-anchor training without versus
with fixed-reference identity supervision**. Both arms begin from the same Phase 3
checkpoint and share data, order, degradation seeds, optimization and evaluation;
the identity term is the declared changed factor. Use 20% clear anchors and a
bounded mixture of blur/resolution, motion, noise/compression and explicit low-light
exposure, preserving target/source proportions. The final augmentation and loss
configuration must be frozen before the GPU run. Do not call the simplified
processed-RGB stress a calibrated CCTV sensor or train on native degraded CCTV
as if it were a clean pixel target.

Initial conservative head/backbone learning-rate candidates are 1e-5/2e-6, with
identity weight 0 versus 0.1; these are hypotheses. The final loss/EMA/norm-buffer
policy and eligibility implementation must be specified in the executable version.
Avoid a short-pilot EMA that mostly retains the starting checkpoint while hiding
whether the trained weights changed. Export declared raw/EMA stages separately
if both are evaluated; selection rules must precede their outputs.

Cap each arm at two epochs and 256 optimizer updates, batch 8: **512 updates
maximum across both arms**, never an open-ended run. Cap the whole VM diagnostic
at 90 minutes including preflight/validation. Record timings at the first 32 updates
and stop when the finite runtime cap cannot be respected. A CUDA preflight may
use one forward/backward batch with **zero optimizer updates**, on the VM only.
The executable must refuse CPU training, changed assets, missing eligibility,
nonfinite losses/gradients, OOM and incompatible resumes. Retain partial evidence
on failure; do not repeat a failed configuration unchanged.

## Selection and required return

Keep baseline weights selectable as epoch 0. A pilot candidate must improve
degraded validation reference agreement and preserve fixed-cohort structure/
identity metrics against its own starting baseline, both overall and per source/
profile. Evaluate clear controls separately; improved aggregate degraded scores
cannot excuse additional clear-input damage. Report the resizing baseline too;
beating a weak old DGP does not by itself establish useful restoration.

Export baseline/candidate raw predictions and a ten-row preview. Compare the
frozen native development cases without inventing clean-reference PSNR/SSIM.
Keep the 32 native reserved crops for the frozen candidate and final evaluation;
do not use them for choosing this recipe. No checkpoint automatically becomes
the app default from these pilot averages or recognition scores.

The return package must include frozen protocol, selected/qualified manifests,
baseline and per-epoch/source/profile metrics, clear-control metrics, reference
eligibility, raw float/PNG previews, timings, optimizer-update counts, weight/code
fingerprints, selection decision and partial/error records. Independently audit
the returned files before selecting a further training run or app integration.
Independent human final assessment, actual local DGP-led workflow and the agreed
covering-family behavior still determine Goal completion.
