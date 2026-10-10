# Direct correction supervision: arithmetic review, 9 October 2026

The existing 145-case TRAIN cohort supports testing direct supervision of the
bounded spatial correction before the final image clamp. The independent audit
passes the target and loss arithmetic. It establishes neither learned capacity
nor improved restoration. No new VM training occurred.

## Evidence and decision

V33–V38's original-decoder proposals and V40/V41's spatial-path training failed
the retained structure or preservation requirements. Their failure records remain
unchanged. In particular, a locally compatible combined-loss displacement did
not protect every finite raw output and delivered PNG. The cause is not uniquely
identified; output clipping has not been proved to be the cause.

The preceding bounded-target geometry audit found that an oracle paired
correction can substantially reduce landmark detail error while retaining the
mean-centered output formula. Its aligned clean target is unavailable at
deployment. This supports investigating the learning path without removing mean
protection or expanding the correction amplitude.

The distinct next hypothesis is to teach our spatial decoder the desired
**correction itself**, before the final clamp, rather than reward only the
final-image high-frequency error. The current DGP checkpoint supplies the
starting restoration and image-specific features. Its weights and stored
normalization, the fixed initial decoder and the recognizer remain frozen.
Only our own spatial decoder would learn. This is an extension of the current
DGP reconstruction path, not additional training epochs of the frozen encoder.
A successful result would still need independent native development review
before becoming the primary app model.

## Target and loss definition

For a synthetically degraded TRAIN case, let `d = clean_target - retained_DGP`.
Over observed pixels, subtract each RGB channel's `(min(d)+max(d))/2`, bound that
delta to `[-0.499999, 0.499999]`, and remove its observed channel mean. This is
the same bounded oracle construction from the preceding geometry review.
Clear controls explicitly receive a zero correction target: retain the current
DGP's output. Unobserved pixels do not enter the correction losses.

Three robust terms supervise the predicted mean-centered correction:

1. RGB correction over the observed region.
2. RGB correction over the original five 24×24 landmark patches intersected
   with the observed region eroded by six pixels.
3. RGB Laplacian correction at three scales, with weights 1, 0.5 and 0.25
   divided by their sum, and strict observed-support radii 4, 10 and 22.

The robust error is `sqrt(error² + 0.001²) - 0.001`. The fixed blur taps are
`[0.05, 0.25, 0.4, 0.25, 0.05]`; downsampling, zero insertion and reconstruction
follow the published Laplacian construction. MPRNet's primary paper and author
code inform these fixed filters and robust losses. Our masked, multiscale,
pre-clamp teacher is a separate design; no MPRNet model or pretrained restoration
weights are used. This research does not establish CCTV or identity preservation.
[MPRNet paper](https://arxiv.org/abs/2102.02808),
[author's loss implementation](https://raw.githubusercontent.com/swz30/MPRNet/main/Deblurring/losses.py).

## Completed local verification

The analysis uses exactly the existing 145 TRAIN cases from 29 photographic
references: 29 clear controls and 116 synthetic degradations. Source groups
contain 65 Asian-source and 80 FFHQ-source cases; these labels describe capture
or collection provenance, not ethnicity. Native CCTV, DEV and reserved final
pixels were not used. Fractions 0, 0.25, 0.5 and 1 of the oracle teacher test the
loss arithmetic; they are not learned predictions.

The independent NumPy/OpenCV checker recomputes 1,740 term values and 21 source/
profile aggregate rows, rechecking 356 source bindings. Maximum independent
float64 discrepancy is 2.7714e-14. Maximum Torch float32 versus independent
float64 initial-term discrepancy is 2.6807e-8. These are arithmetic tolerances,
not relaxed appearance gates.

| Initial 40 degraded calibration cases | Mean loss scale |
| --- | ---: |
| Observed RGB correction | 0.06827913638 |
| Landmark RGB correction | 0.09418982418 |
| RGB pyramid correction | 0.01376023039 |

All 29 clear cases have zero initial correction loss. The separate regressions
reject empty or insufficient support, nonfinite corrections, a landmark region
outside observed support and a local gradient-bearing path. Padding and hole
poisoning leave valid losses exactly unchanged. A chromatic test whose luminance
cancels still registers RGB detail error. This verifies different loss coverage,
not successful preservation of real appearance.

The analysis took 68.03 seconds and the independent check 36.16 seconds, each
within its 300-second cap. Learned-model forwards, gradient queries, backward
calls and optimizer updates were all zero; fixed filter operations did execute.

## Finite VM design to prepare

Keep the same accepted 781 TRAIN references and 3,905 prepared cases, five
profiles per reference, current DGP checkpoint and own spatial initialization.
Compare additional spatial-decoder epochs 1, 2 and 5, with five epochs maximum.
Native development crops guide the subsequent unpaired review; clean training
targets do not enter the restoration forward path. No reserved final sample
is used to choose an objective, learning rate or checkpoint.

The existing 1% structure requirement at update 50, source nonregression,
pixel/SSIM/embedding preservation and 20% brightness-only limit still apply.
The final necessary capacity requirement remains 10%. Stop and export the first
failed early requirement. Finishing five epochs cannot qualify a failed candidate.
Retain identity supervision, clear-input controls and raw versus PNG reporting.
Freeze the objective weights, learning rate, schedule, gradient preflight,
resource limits, full migration states and prospective return auditor before
issuing the new manual tmux command. Do not execute a historical pilot.

This arithmetic milestone has no trained checkpoint, executable VM packet,
app promotion, independent final review or covering-family qualification.
The complete five-milestone restoration and seven-family completion goal remains
active and incomplete.

## Bound artifacts

- Analysis: `outputs/cctv_dgp_residual_supervision_v1/analysis.json`, SHA256
  `c7706f7e13214a6be2113cc60543899162d63bacbe2ce04e0f3da97315a09b24`.
- Plan: `outputs/cctv_dgp_residual_supervision_v1/plan.json`, SHA256
  `1cb14a62e6d561dc14357a6be0942614c186775a5b3b4f24cfda621cbf906cba`.
- Independent receipt: `outputs/cctv_dgp_residual_supervision_v1/independent_audit.json`.
- Producer: `scripts/analyze_cctv_dgp_residual_supervision_v1.py` and
  `cctv_dgp_residual_supervision_v1.py`.
- Independent checker: `scripts/verify_cctv_dgp_residual_supervision_v1.py`, SHA256
  `34489433e763853ed5add4c3c1bc099b72f188104ee2967539f4bd536a9f832e`.
