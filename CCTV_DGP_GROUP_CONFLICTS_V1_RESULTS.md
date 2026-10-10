# DGP group conflicts: verified return and finite-output review

10 October 2026. The manual VM diagnostic completed successfully, but all three
reset trials remain unqualified. Its initial shared descent direction does not
establish a safe finite change or useful facial restoration. Current app and
original Phase 3 checkpoints remain unchanged. No optimizer update or epoch was
performed in this diagnostic.

## Independent evidence checks

Returned archive: 608,243,422 bytes, SHA256
`701cfd773d48f69cdbf5a933fd453db11bcd9ac3839de0fc88f7b50592e30dfb`.
Frozen protocol:
`bc0dcbd92b876c53ebe4f484cc5743fad7c8e7cdce6ca8222e04de3d485ec4b2`.
The frozen prospective checker passes without a repair: 1,624 regular members,
400 raw/PNG output records, all 32 saved aggregate gradient vectors, the common
direction certificate, three float32 displacements and all 12 scientific
decisions. Eighty fresh CPU cases agree within the prospective limits: maximum
raw error 2.272427e-6, PNG error one byte, embedding error 2.831221e-7.
Audit time is 183.61 seconds; supervised total is 186.36 seconds.

The VM performed 160 autograd queries and three independent reset placements,
with zero optimizer updates, epochs or committed trajectory updates. Runtime
was 164.49 seconds, including 53.92 seconds for trials; peak allocated VRAM was
1,676,872,192 bytes. Original, candidate, normalization buffers and fixed
recognizer states return exactly to their initial values. Gradient metadata
and aggregate-vector arithmetic are audited; individual query vectors were
not exported, so their autograd results/aggregation were not independently
recomputed locally. This limitation remains explicit.

Both cohorts are exposed photographic TRAIN evidence: 20 references, each with
five profiles, producing 100 cases per variant. The second cohort is a TRAIN
cross-check, not DEV or final evaluation. No native CCTV, full 3,905-case TRAIN
capacity, paired DEV or reserved final identities were tested.

## Scientific results

| Reset displacement / original decoder L2 | Gradient cohort PNG structure gain | TRAIN cross-cohort PNG gain | PNG preservation failures, first / cross |
|---|---:|---:|---:|
| 1e-5 | 0.034730% | 0.018856% | 0 / 4 |
| 1e-4 | 0.106404% | 0.061254% | 4 / 4 |
| 1e-3 | -1.139494% | -1.439851% | 46 / 47 |

All 12 cohort/stage decisions fail. Positive small gains remain below the
unchanged 1% structure requirement. The smallest step preserves the sampled
gradient cohort but fails raw and delivered preservation in the other TRAIN
cohort. Medium steps also regress sampled low-light/motion SSIM and compound
recognition; the sampled Asian-source structure gain becomes negative. The
largest worsens structure and appearance in both cohorts. Failure counts are
group/metric records, not people, identification mistakes or population rates.

Raw failures persist, so display PNG processing alone is not the cause. Very
large brightness-explanation fractions on the largest step occur when total MSE
has worsened and the denominator is clamped at its retained 1e-12 floor. They are
failure indicators, not meaningful percentages of recovered facial information.

## What the new diagnostic resolves

The previous balanced direction has adverse initial descent cosines in six of
the 32 sampled groups. The new bounded solve finds a direction with minimum
normalized descent cosine 0.19841845, comfortably above its 1e-7 certificate
requirement. Every proposed float32 displacement still has a negative initial
dot product with all 32 gradients. This establishes a shared local improving
direction for these losses, not global infeasibility or useful restoration.

Post-hoc saved-evidence attribution checks finite scientific raw losses against
those predictions. The smallest trial has no sampled-group regressions beyond
the existing allowances; the medium has four and the largest 27. The SSIM
derivative remains a declared surrogate. The 100 initial raw and PNG outputs
are byte-identical to the prior loss-balance baseline. These results support
investigating finite-step behavior and missing TRAIN coverage before any epoch
extension; they do not prove that one cause explains every failure.

## Visual review and consequence

All 20 sheets, 600 exact 256-pixel cells and 400 unique outputs were actually
reviewed. The two smaller placements look close to baseline and do not provide
convincing useful eye, nose or mouth detail. Many clear-image details are still
softened; degraded blur/compound faces remain indistinct. The largest changes
tone and introduces a green canvas-edge strip on several full-canvas fixtures.
No trial is adopted. This is primary-assistant development review, not
independent final review. A separate saved-pixel checker verifies every cell,
page hash and 20 recorded observations; code does not prove subjective judgments.

After three unsuccessful diagnostic designs, the invalid assumption is that an
initially shared improving direction remains safe across finite changes and
other TRAIN faces. The AGENTS.md circuit-breaker question was asked; the user
answered "apply the best approach". The next design review therefore prioritizes
finite, measured preservation checks and broader TRAIN coverage, with fresh
directions after accepted changes. An unchanged constant-direction continuation
or bypass of the earlier 1% stop is not justified.

V42's 3,905-case failure and every earlier gate failure remain binding. More
epochs need a distinct verified recipe, full TRAIN capacity, paired DEV
preservation, useful native DEV output review and independent final evaluation.
The full restoration and seven-family completion goal remains active/incomplete.

Evidence: `outputs/cctv_dgp_group_conflicts_v1_independent_audit.json`,
`outputs/cctv_dgp_group_conflicts_v1_return_review_v1/visual_review.json`,
`outputs/cctv_dgp_group_conflicts_v1_return_review_v1/finite_direction_analysis.json`.
