# Rounding reduces conversion error but does not qualify a proposal

The fixed local diagnostic and its frozen prospective checker both completed.
Nearest-half-up rounding removes most downward byte bias and approximately
halves error relative to the saved raw floats. **It does not make any of the
three rejected proposals acceptable.** Recognition regressions remain in the
second TRAIN cohort, and some first-cohort structure/source gains turn negative.
The original 1% structure requirement remains unmet under both policies.
No rounding policy, trained state or app change is adopted.

This tests exactly 100 exposed photographic TRAIN cases, the original baseline
and all three finite-guard R1 proposals. Baseline and candidate use the same
policy in each comparison. The original float32 floor conversion is compared
with one fixed `floor(float64(raw)*255 + 0.5)` conversion. No scale, policy,
case selection or quality threshold was fitted from these outputs. Raw pixels
are unchanged; camera padding stays exact. This is not another VM training run.

| Saved variant | Mean floor error versus raw, bytes | Mean nearest absolute error versus raw, bytes | RGB components changed |
|---|---:|---:|---:|
| Baseline | -0.498000 | 0.248898 | 49.8049% |
| Proposal 0.000020 | -0.498207 | 0.249127 | 49.8035% |
| Proposal 0.000010 | -0.498038 | 0.248982 | 49.7941% |
| Proposal 0.000005 | -0.497976 | 0.248944 | 49.7963% |

Floor signed and absolute errors have equal magnitude here. Nearest signed
error is between -0.000172 and +0.000050 bytes across these variants. These are
conversion errors versus raw floats, not errors versus clean facial targets.
Each changed channel differs by at most one byte; the independent checker
verifies that bound and every outside-support camera pixel. This arithmetic
improvement does not establish sharper faces or recovered identity.

| Proposal | Floor structure gain: first / second cohort | Nearest structure gain: first / second cohort | Floor failures: first / second | Nearest failures: first / second |
|---|---:|---:|---:|---:|
| 0.000020 | 0.043497% / 0.039478% | 0.011271% / 0.035845% | 0 / 1 | 0 / 1 |
| 0.000010 | 0.033851% / 0.018666% | -0.000922% / 0.028325% | 0 / 3 | 0 / 1 |
| 0.000005 | 0.026661% / 0.005028% | 0.002507% / 0.009518% | 2 / 6 | 5 / 1 |

All 12 same-policy quality decisions fail. Under nearest conversion, the
first-cohort FFHQ-thumbnail degraded source gain is negative in all three
proposals. The largest proposal's second-cohort ArcFace regression moves from
low-light to motion; both smaller proposals retain a second-cohort low-light
regression. The smallest also has five first-cohort FFHQ group regressions.
Changing the conversion changes which groups fail; it does not solve finite
preservation. Failure entries are source/profile/metric aggregates, not counts
of identities or people. Historical VM floor failures remain in their original
records; no proposal is relabelled as passing.

The worker took 213.137 seconds inside a 215.618-second supervisor. It made 80
fixed-recognizer CPU forwards in a declared 15-image context (five floor, five
nearest and five paired-target images). The largest embedding difference from
the original VM floor/target vectors is 3.371388e-7, within the previously
declared 5e-5 replay tolerance. Every original floor failure-key/pass decision
is reproduced. Scientific comparisons still use 1e-12 MSE, 1e-6 SSIM/ArcFace,
nonnegative source structure gains, 0.2 brightness share and 1% overall gain.
Replay tolerance does not change any quality requirement.

The prospective checker verifies 1,438 source bindings and 806 artifact bindings,
all 400 new PNGs, 800 metric rows, all source/profile/mean-shift controls, 12
quality decisions and all quantization sums. It independently reconstructs
nearest pixels through fractional-part arithmetic. Four prospectively selected
fresh embedding replays have maximum error zero. The checker took 70.647
seconds inside a 72.358-second supervisor. Recognizer state is unchanged.
There are zero DGP or completion forwards, gradient queries, training parameter
updates and epochs in both phases. Both are within their frozen runtime caps.

The original 400 floor outputs have full primary-assistant development visual
review. The 400 newly rounded images are arithmetically audited, **not separately
visually qualified**. A one-byte bound is not a replacement for visual review
or native development usefulness. There is no app adoption, new checkpoint,
independent final evaluation or claim of native CCTV performance here.

Protocol SHA256:
`d749160e6938d0539a93842c328668e1bccf3771dc160998e0c5a0234ab0c1f4`.
The original finite-guard R1 protocol, return checker, checkpoints and failures
remain unchanged. The hypothesis that rounding alone can unlock useful
preserved training is rejected on these saved cases. Further floor/rounding
variants or unchanged smaller steps are not justified by this result.

The next design review examines learned spatial reconstruction and degradation
coverage from the retained current DGP; see
CCTV_DGP_POST_FINITE_GUARD_TRAINING_REVIEW.md. It does not yet provide an
executable epoch-extension recipe. The overall DGP-led restoration, seven
automatic/assisted covering families and independent app/final verification
goal remains active/incomplete.

Evidence:

- [Frozen local protocol](<C:/xampp/htdocs/YEAR 4/Testing/outputs/cctv_dgp_finite_guard_quantization_v1/protocol.json>)
- [All measured comparisons](<C:/xampp/htdocs/YEAR 4/Testing/outputs/cctv_dgp_finite_guard_quantization_v1/results.json>)
- [Independent audit](<C:/xampp/htdocs/YEAR 4/Testing/outputs/cctv_dgp_finite_guard_quantization_v1/independent_audit.json>)
- [Original return and visual findings](<C:/xampp/htdocs/YEAR 4/Testing/CCTV_DGP_FINITE_GUARD_V1_R1_RESULTS.md>)
- [Prospective processing design](<C:/xampp/htdocs/YEAR 4/Testing/CCTV_DGP_FINITE_GUARD_QUANTIZATION_V1_DESIGN.md>)
