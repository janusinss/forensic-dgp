# DGP structure feedback, saved-output trace and closed detail controls — V21

The user reviewed `choke_dev_02_t033` and answered: **“its useful but needed
clearer structure.”** This is positive usefulness feedback for the shown
development result with a remaining facial-structure defect. The uploaded crop
was already judged usable. Do not relabel that crop as insufficient information
to avoid correcting the output. Full cohort, covering-family and independent
final acceptance remain incomplete. Visible eyes, nose, mouth, contour, glasses
and hair remain the targets; some softness is acceptable.

## Saved-array evidence

The frozen V21 trace reads 50 previously exposed paired photographic TRAINING
cases and 24 previously exposed ChokePoint native DEVELOPMENT crops. It produces
no corrected images and runs no neural model. All 379 source bindings and 122
original raw-to-PNG compositions pass; 244 regional measurements and 26 group
summaries are independently recomputed with a second convolution implementation.
Execution takes 5.00 seconds and saved-output audit 2.85 seconds. Maximum scalar
audit difference is 2.34e-15. Native CCTV remains unpaired.

The fixed diagnostic uses a 13-tap separable Gaussian, sigma2 at 256 pixels,
half-sample reflection and float64 luma. Observed support is eroded6 pixels;
input landmark patches are 24×24. Native metadata supplies only the two eyes;
other native facial features are not automatically localized. This is an
input-relative detail diagnostic, not an identity or restoration-accuracy score.

| Arm on 24 native crops | Median input-aligned detail slope in eye patches |
| --- | ---: |
| Retained Phase3 | 0.187889 |
| Retained own DGP | 0.704009 |
| Declared CodeFormer w1 baseline | 1.027428 |

Every own-DGP case has a negative effective residual projection onto captured
eye-patch detail. The shown case's slope is 0.563223. Attenuation is already in
the untouched raw floats; PNG quantization does not account for it. Captured
high frequencies can include noise and compression, and sharper pretrained
details remain unverified. The trace does not prove a specific training loss
caused the defect or that missing identity information can be recovered.

Training floats inherit V18's internal CUDA-scalar normalization. They are not
relabeled as canonical fresh-app parity. Initial diagnostic preparation pointed
at the wrong external data root. The failure and a separate recovery receipt
are retained; the actual V18 training images come from mixed V9r2. Image bytes,
checkpoints, split membership and original output files were not changed.

## Fixed detail-preservation control: closed

Before output generation, freeze one formula retaining captured detail and only
the normalized Gaussian-lowpassed DGP correction. Use the same 50 TRAINING cases;
the target is used for scoring only. Keep processed diagnostic arrays separate
from untouched model raw. Execution takes 9.46 seconds; the 5.19-second independent
audit verifies 174 sources, 110 artifacts, 50 raw/PNG compositions, 150 paired
metric rows and 200 grid cells. All ten 1072×1504 sheets are actually viewed at
original resolution.

The control fails 15 predeclared paired-preservation checks. Mean degraded MSE
rises 0.08998%; blur and motion include regressions. Clear photographs retain
more already visible glasses, hair and texture, but degraded outputs remain
soft and can retain stripes and blocks. No convincing new facial structure is
established. Close before native, reserved or broader evaluation and do not
adopt this control in the app.

V21 control SSIM uses a separately declared Gaussian11 population formula over
eroded support. Its numbers are not substituted for historical win7 SSIM gates.
All original failures remain available in `results.json`.

## Restricted learned band gate: closed before a VM pilot

A second frozen TRAINING-only diagnostic analytically bounds a restricted
spatial gate of the DGP residual high-frequency band. Permit independent RGB
coefficients in [-1,1] and a target-informed per-pixel interval projection, more
freedom than a practical scalar gate. Expand the delivered integer interval one
unit at each end to make the PNG error bound conservative. No coefficient,
restoration image, neural call, optimizer or gradient is produced.

Even that optimistic family cannot improve mean degraded paired PNG MSE by more
than **5.44026%**, below the predeclared10% capacity requirement. This closes that
restricted band-rescaling recipe, not all DGP architectures or all possible
structural improvements. It does not bound SSIM, identity or visual usefulness.
Execution takes 1.85 seconds; the independent 2.97-second OpenCV-based audit
checks 179 source bindings, 50 intervals and 17 groups to 2.23e-16.

The first audit catches a bookkeeping defect: the two source/clear groups report
ten observations for five unique cases. Preserve the original runner, results
and failed audit. A separate R1 corrects only those two counts; every numerical
value and stop decision stays identical. The independent audit verifies the
correction and rederives all unique counts and metrics.

## Next finite experiment: V22, manual VM execution pending

Inference controls do not resolve the demonstrated raw-model limitation. V22
therefore prepares a new 45,443-parameter own-DGP detail head conditioned on
camera pixels and retained own-DGP pixels, with no CodeFormer generator features,
RGB, codes or weights. Unlike V18's broad RGB residual, the new correction is
projected to a fixed Gaussian high-pass and zero observed channel mean before
clamp. Post-clamp brightness contribution is explicitly measured and bounded.
This extends the existing feed-forward FPN residual implementation; it does not
make it equivalent to published GAN-based DGP.

Freeze 800 updates/80 epochs, batch5, using only the same ten exposed training
references/50 cases. No new data, native crops, reserved identities or covering
training is admitted. The endpoint is final800 only; no checkpoint selection
or automatic follow-on run. The existing DGP and ArcFace remain frozen.

The new prospective capacity requirement is at least10% reduction of paired
landmark-luma high-frequency MSE, with nonregression on both provenance sources.
Keep historical MSE/SSIM/ArcFace source/profile preservation tolerances unchanged
and limit brightness-only contribution to20% of degraded MSE gain. This responds
to the structure feedback and V19's brightness-dominated gain; it does not erase
V18's failed ten-percent RGB-MSE requirement or any older failed recipe.

Stop at update50 without at least1% structural gain. Measure projection at
update20 with a1.25 timing margin. Caps: preflight300s, fit1500s, worker1800s,
external supervisor2100s, export120s; 3GiB free disk and20GiB peak allocated VRAM.
Preserve every partial head, original checkpoint, timing stop and failed gate.

Three CPU no-gradient contract forwards verify exact initial output on two
training cases and padding/range behavior under fixed disposable test weights.
The diagnostic Gaussian matches SciPy to1.60e-7. The final class AST is identical
to the tested class. Bundle audit verifies195 assets,192 original source bindings,
all800 scheduled batches and correct Windows training rejection before model
execution. Git Bash `-n` passes; its initial sandbox signal-pipe failure is
recorded. All197 safe regular archive members and the LF checksum are verified.
These are preparation checks; VM CUDA gradients, timing and quality are pending.

Training must use the existing NVIDIA L4/g2-standard-4 VM at `~/forensic-dgp`.
The assistant performs no SSH, upload, launch, stop, cleanup or cloud action.
The user follows [the exact gcloud/tmux commands](<C:/xampp/htdocs/YEAR 4/Testing/CCTV_DGP_DETAIL_PRIOR_V22_VM.md>).

### Active execution revision R1: Python3.10 compatibility

The VM trace identifies Python3.10. Final source review catches the original
unexecuted V22 helper's use of `hashlib.file_digest`, a Python3.11 API. Preserve
that source, protocol and archive. A separate R1 uses streaming SHA256 and
separate execution/return filenames. The tested head AST and the complete
verify/VM-guard/prepare/run functions are otherwise identical. Model, data,
schedule, objective, gate definitions and budgets stay unchanged.

R1 independently verifies196 assets, Python3.10 grammar for13 source files,
streaming checksums, Windows training rejection and Bash syntax. Its198 safe
regular archive members and LF checksum pass read-back audit in1.93 seconds.
Archive217,556,126 bytes/207.48MiB, SHA256
6978e423c23909caebff65c7299267ce1a6803d15e2818e38bac6b1f685fdaba.
Protocol c431af07b8e0bd2310f67fdc1b9ccfac7118adc08dc6cddd22131af6a5a4ca96.
A separate nonempty-padding fixture adds one fixed-weight, no-gradient CPU head
forward:20,992 outside pixels remain exact in raw and delivered output. It corrects
the limitation of the earlier full-support fixture's vacuous padding assertion.
Total local head contract forwards are four; optimizer/backward calls stay zero.
No fixed test weights or outputs are promoted.

Use only the R1 commands in the runbook. No VM run or training has occurred.

On return, independently verify archive/source bindings, budgets, update counts,
all saved metrics and exact PNG compositions; replay the head and recognizer
locally without training and review all50 cases. A capacity pass requires a
separately frozen broader development/native experiment. It is not app promotion,
final review or useful CCTV qualification. The existing app's22 bindings remain
unchanged; historical34 regressions and Playwright are not rerun for this work.
All restoration and automatic/assisted covering-family requirements remain open.

Evidence: `outputs/dgp_structure_feedback_20261005/`,
`outputs/dgp_structure_trace_v21/`, `outputs/dgp_structure_detail_control_v21/`,
`outputs/dgp_detail_gate_capacity_bound_v21/`,
`outputs/cctv_dgp_detail_prior_v22_preparation/`.
Goal active; original failed gates and unviewed final reserves remain intact.
