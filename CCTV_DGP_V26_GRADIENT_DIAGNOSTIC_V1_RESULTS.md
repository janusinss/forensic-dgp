# Corrected V26 gradient diagnostic: verified return

6 October 2026. **Measurement passed; V26 restoration remains a closed failure.**
The downloaded archive is 1,665,230 bytes and matches SHA256
`bb07f90ef10496a8b464d61d9e446826e4f8d824c12478f35765b3d3b160c201`.
Ten regular files were safely imported without executing returned code.
The diagnostic made **zero optimizer updates** and produced no new checkpoint.
`complete:true` refers to this measurement/export, not useful reconstruction.

The independent readback checks 240 original assets, 118 saved-input bindings,
250 frozen own-DGP feature arrays, both saved heads, two 7×53,781 float64 gradient
matrices, 14 component rows and all 26 parameter groups per state. All 752,934
saved gradient values, statistics, batch contexts, counters, logs and finite limits
pass. CPU saved-scalar differences peak at 4.5858324e-8 within the original bounds.
It checks saved derivatives; no local derivative is calculated or independently
rerun on the L4. Full AdamW moments/trajectory were not recorded.

| Evidence | Initial state 0 | Stopped state 50 |
|---|---:|---:|
| Seven-term objective | 1.3000000081 | 1.2997388303 |
| Combined degraded improvement gradient norm | 0.0141790984 | 0.0137879958 |
| Combined preservation gradient norm | 0 | 0.0051036254 |
| Preservation / improvement norm | 0 | 0.3701499124 |
| Improvement/preservation cosine | Undefined: zero preservation | −0.7547647029 |
| Total gradient norm | 0.0141790984 | 0.0104848509 |

The corrected initial identity value and **every identity gradient element are
exactly zero**, in all ten five-case batches. This corroborates the V26 reference
processing correction. At stopped 50, clear-input anchoring, SSIM and identity
preservation contribute opposing gradients. Their vector sum does not cancel the
improvement gradient. These observations do not justify removing preservation
terms, adding an identity tolerance or relaxing the original structure stop.

The actual L4 worker took 19.0893201 seconds; its external supervisor took
20.2279370 seconds. Counters are 20 head batches, 140 component gradient queries,
70 frozen recognizer forwards, zero DGP forwards and zero optimizer updates.
Independent return audit took 3.4482148 seconds. Export took 0.1687637 seconds.
The original 420/480-second worker/supervisor caps and all other limits remain intact.

## Spatial-path evidence and next test

A separate pure-array analysis and independent coordinate/second-moment checker
verify 269 source bindings, 84 parameter-family/term rows and all 250 input-only
feature summaries. No neural call, derivative, weight fitting or optimization runs
locally. This analysis took 5.2464302 seconds; its independent checker took
4.8187577 seconds.

The direct RGB branch carries 99.3184264% of the initial landmark gradient's squared
magnitude and 98.4251552% at stopped 50. The five original feature projections,
decoder, camera and fuse gradients are initially zero because the output tail is
zero-initialized; they become active during training. At stopped 50, the feature
projections account for 0.0779167% of landmark squared-gradient magnitude. These
fractions depend on parameterization and cannot by themselves prove the cause of
poor optimization or useful restored appearance.

Median degraded spatial RMS drops from 0.0164787 at decoder0 to 0.00914027 at the
fused representation. The particular saved correction is 0.000521559; removing its
channel mean does not cause the earlier V24 high-pass attenuation. The independent
V26 raw/PNG audit already excludes quantization alone as the remaining cause.

The user-selected own-DGP spatial direction therefore supports one distinct,
prospective hypothesis: add zero-initialized readouts from the five frozen DGP
feature scales directly into the existing bounded RGB residual. Keep the original
decoder and direct RGB branch. Normalize each new branch using only that input's
observed-channel spatial RMS, with a fixed 0.05 floor; no target or source/person
label enters inference. Only 162 of 28,800 case/channel combinations lie below
that floor; degraded median channel RMS by scale is 0.106605, 0.501077, 0.654099,
0.594127 and 0.196132. This is input scaling evidence, not a selected quality optimum.

Short residual connections are supported as an optimization design principle by
the [author residual-learning paper](https://arxiv.org/abs/1512.03385). Applying that
principle to this particular DGP feature path is our hypothesis, with no claim of
CCTV or identity validation from the paper. No external restoration weights/code
are substituted. A new source/parity audit and finite manual L4 pilot are required
before accepting that design. Every original loss coefficient, seed, 50 exposed
TRAIN cases, schedule, 1% early/10% final structure requirement, pixel/SSIM/identity
preservation and finite stop must remain fixed.

V26 delivered only 0.0335635587% degraded structure gain against 1% at update 50;
final800 was never run. All 50 whole-face TRAIN cases lacked convincing gain in
eyes, nose, mouth, outline and visible appearance. That failure stays closed.
Do not rerun V26 or this completed diagnostic unchanged. A favorable new gradient
or capacity score cannot replace whole-face review, native CCTV development,
canonical app inference or independent reserved final evaluation.

Native CCTV remains unpaired; paired synthetic photographic evidence is separate.
The 58 reserved native crops/45 namespaced labels remain unviewed. No new covering
pixels, ethnicity or Zamboanga performance claims enter this measurement. No app
checkpoint is promoted. All seven automatic/assisted covering families, useful
native output, input-only insufficient-information handling and final review remain
required. **Goal active/incomplete.**

Evidence: [independent return audit](<C:/xampp/htdocs/YEAR 4/Testing/outputs/cctv_dgp_v26_gradient_diagnostic_v1_independent_audit.json>),
[saved-array path analysis](<C:/xampp/htdocs/YEAR 4/Testing/outputs/cctv_dgp_v26_gradient_paths_v1/analysis.json>),
[independent path readback](<C:/xampp/htdocs/YEAR 4/Testing/outputs/cctv_dgp_v26_gradient_paths_v1/independent_readback.json>).
