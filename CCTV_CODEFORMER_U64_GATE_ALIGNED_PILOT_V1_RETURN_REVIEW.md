# Original-u64 gate-aligned pilot: audited quality stop at update two

23 October 2026. **Integrity passes; pilot quality fails.** The manual L4 run
applied two of four permitted updates, then correctly stopped on the localized
blur PNG structure requirement. It did not crash from a missing file, optimizer
or broken gradient. No state from this rejected pilot replaces original u64 or
U96. Do not resume it, append epochs or repeat its launch guide.

## What failed

Update one passed its scheduled bank. At update two, the same central128 local
blur endpoint failed two comparisons: against original u64 and against the prior
accepted state. Every inherited guard still passed. The additional PNG
nonincrease requirement remains binding, with no quality tolerance added.

| Local-blur central structure error, lower is better | Original u64 | Update 1 | Update 2 |
| --- | ---: | ---: | ---: |
| RAW adjacent error | 0.0169674276970 | 0.0169672431862 | 0.0169670592095 |
| Delivered PNG adjacent error | 0.0169709439626 | 0.0169706574920 | 0.0169714465417 |

The PNG error increased **5.025791143e-7**, or **0.0029614%**, against original
u64. It increased **7.890497216e-7**, or **0.00464949%**, against update one.
All eight local RAW cases improved; six of eight corresponding PNG cases worsened
against original u64, and seven worsened against update one. This is a small
measured regression, not evidence of a dramatic visible degradation or permission
to ignore the gate. Terminal update four was never reached.

Keep the comparison denominators explicit:

| PNG structure gain against frozen U96, same adaptive TRAIN monitors | Original u64 | Update 1 | Update 2 |
| --- | ---: | ---: | ---: |
| Global blur | 1.627216% | 1.629008% | 1.630374% |
| Local central128 blur | 1.210135% | 1.211803% | 1.207209% |

The global score rose slightly while the local score fell. Neither a favorable
single column nor an earlier passing snapshot qualifies an overall winner.
The original 1% endpoint threshold was not the failing check at update two;
the new RAW/PNG retention requirement stopped the run before its terminal screen.

## Why RAW improvement did not carry into PNG

The renderer is unchanged: clamp native512 RGB, bilinear-resize to RAW256, then
save the existing floor-to-RGB8 PNG. All 300 delivered pairs match that renderer's
saved RAW-to-PNG conversion exactly. No double blur, new filter, altered mask,
round-to-nearest substitution or corrupted image export was introduced.

For the eight local monitor images, update two versus update one changed only
**1,132 of 393,216 RGB components** in central128: 589 increased by one RGB8 level,
543 decreased by one, and no component moved further. These sparse code crossings
affected 4,485 of 780,288 residual-edge components. Their integer edge-error
increments sum to +2,321 and decrements to -2,164, a net +157. This independently
reconstructs the observed PNG mean-error increase within 3.15e-13 of the original
FP32 normalization arithmetic. The reconstruction tolerance concerns arithmetic
readback only; the quality threshold remains zero.

Both updates passed all eight actual rounded RAW gradient-dot checks. Those
checks predicted local RAW behavior on optimization photographs. They did not
certify the discrete PNG endpoint on different monitor photographs. This return
demonstrates that distinction with actual saved pixels. It does not prove a
complete explanation of every earlier training or development failure.

## Verified execution and preserved state

- Return archive: **1,166,765,803 bytes**, SHA256
  `005167e9be87c37efab55382652b9e5776466a61ef4f201f485fdafd527281d0`.
  Protocol SHA256:
  `35a5cfd877a97355622c6c69d9792fd9238ccf312335dd679081a8636ba23b48`.
  All 1,643 member payloads, gzip CRC, tar framing, static sources, source
  admissions and 469 parent bindings authenticate. Local model aliases remain
  explicitly local audit evidence; they did not substitute VM admission paths.
- The journal replays **1,679 events**, giving 558 complete operation pairs:
  **300 full forwards, 256 gradient queries and two parameter applications**.
  No pending operation or partial file is declared. Worker exit **2** authenticates
  the quality stop; timeout is false. Its child was reaped and process group
  absent. Worker elapsed time was about **256 seconds**. Post-export supervisor
  and export-process receipts were not inside the exported archive by design;
  their final exit codes are not independently claimed from these files.
- All 256 saved gradients have the exact 36-tensor/168,960-coordinate schema.
  Independent pooling and direction calculations match both saved certificates.
  Actual FP32 commits reconstruct exactly from the fixed scalar rule, including
  the 0.99 headroom factor. Maximum coordinate step is 2.474989742e-5; cumulative
  change after two updates is 4.949979484e-5, within the original limits.
- Complete initial575 state matches original u64. All **539 frozen R2/U96
  tensors remain bitwise exact** in the failed closing state. Head states
  0/1/2, proposed arrays, actual deltas, RNG, fixed sampler, null optimizer and
  failure receipts agree. No Adam object or momentum was used; the two plain
  gradient-descent applications were actual manual VM training.
- Independent saved-pixel review passes **4,138 checks**: all 300 RAW/PNG pairs,
  256 scalar receipts, three complete 56-case banks, four historical parity
  cases and four full repeats. Root personally inspected all **20 planned
  sheets**. They cover 296 distinct case/state outputs and four hash-identical
  repeat aliases, accounting for all 300 forwards. Fine texture remains smooth;
  the two small updates do not show convincing useful visible improvement.

These are adaptive paired TRAIN measurements, including inherited dark guards
whose protocol has no aligned photographic target. No native CCTV, wild-image,
DEV or FINAL model call was made by this pilot. Dataset source labels establish
neither ethnicity nor Zamboanga performance.

## Next action supported by this result

Keep original u64 as a retained challenger and U96 as experimental app incumbent.
The next source-only design review should address **delivered-PNG-aware learning**
before selecting another training pilot. Preserve RAW fidelity/structure checks
as well; do not optimize a PNG surrogate while allowing RAW deterioration.
Do not alter the PNG quantizer, threshold, baseline or downloaded images to
manufacture a passing result. Do not promote update one retrospectively.

Hard quantization is nondifferentiable. Straight-through estimators supply a
surrogate backward rule rather than the true derivative; their choice matters.
This is documented in [Bengio et al., 2013](https://arxiv.org/abs/1308.3432) and
[Yin et al., ICLR 2019](https://arxiv.org/abs/1903.05662). The latter's theoretical
setting is a small binarized network, not CCTV restoration, and poor estimator
choices can be unstable. [PyTorch's QAT explanation](https://pytorch.org/blog/quantization-aware-training/)
illustrates simulated quantization during learning; its reported LLM results
are not evidence for this face-restoration application. Applying that general
idea to our RGB8 output is a proposed research direction, not a proven fix.

The immediate safe follow-up is a bounded, separately reviewed **zero-update
measurement design** from exact original u64: retain existing RAW gate queries,
measure explicitly labeled PNG-surrogate gate queries, require exact forward
PNG parity and examine their shared current-state geometry before any new update.
An exporter loss with biased gradients must never be described as the actual
PNG derivative. No next worker, transfer archive or launch guide has been
prepared for that follow-up yet.

Authoritative receipts:
`outputs/codeformer_u64_gate_aligned_pilot_v1_return_transport/audit_v1.json`,
`outputs/codeformer_u64_gate_aligned_pilot_v1_return_updates/receipt_v1.json`,
`outputs/codeformer_u64_gate_aligned_pilot_v1_return_pixels/audit_v1.json`,
`outputs/codeformer_u64_gate_aligned_pilot_v1_png_diagnosis/receipt_v1.json`, and
`outputs/codeformer_u64_gate_aligned_pilot_v1_return_audit/root_visual_review_v1.json`.

All failure states, checkpoints, gradients, original weights, splits and
provenance remain retained. Broader DEV failures remain binding. Reviewed
Protect/Apply/bypass stays separate; automatic matching and later completion
development remain deferred. Native/wild remain unpaired. FINAL remains unopened
and execution-unfrozen with independent reviewers pending. Automatic goal
advancement did not trigger. Full goal **ACTIVE/incomplete**.
