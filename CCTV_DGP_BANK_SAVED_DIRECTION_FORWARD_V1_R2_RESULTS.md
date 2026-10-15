# Saved-direction finite forward R2 results — 11 October 2026

**Execution completed, but none of the four proposals qualifies.** The larger
changes exceed the 1% mean structure requirement and still regress preservation.
The smaller changes remain below 1% and also regress preservation. More epochs
and app adoption remain unqualified. The original checkpoint is preserved.

The human manual L4 diagnostic completed in **53.96 seconds**,
using **100 B forwards, 160 recognizer forwards, zero gradient queries,
zero optimizer updates and zero epochs**. Four temporary parameter proposals
were evaluated from the pinned initial B state. The model was restored between
proposals and at completion; original and recognizer state hashes were retained.
No new checkpoint was saved. Peak allocated VRAM was 1.106 GiB.
This measured peak is allocated Torch memory, not total device usage.

Return SHA256: `247e295221cb996769df905c69a2dca9ac043d0dcbce58621dae4c61bab083f1` (313,779,973 bytes).
All **596 returned file bindings**, **5,852 parent hashes**, **100 control-case
hashes**, four proposed-state/tensor hashes, **400 raw output cases**, **400
delivered PNGs**, **800 raw/PNG metric rows** and **17 source/profile groups**
were checked. Saved embedding calculations were audited; all embeddings were
not independently regenerated through a new recognizer forward. The audit used
saved arrays and parameter bytes; it did not run local neural gradients,
optimizers or model-parameter displacements.

| Proposal | Raw degraded structure gain | PNG gain | Raw preservation failures | PNG failures | Decision |
|---|---:|---:|---:|---:|---|
| ordinary_0.0001 | 1.1095% | 1.1323% | 42 | 42 | Rejected |
| ordinary_0.00001 | 0.1427% | 0.1647% | 20 | 20 | Rejected |
| constrained_0.0001 | 1.0941% | 1.1093% | 44 | 45 | Rejected |
| constrained_0.00001 | 0.1469% | 0.1606% | 21 | 20 | Rejected |

Failure counts are **group/metric checks**, not numbers of distinct faces.
The threshold remains 1%; MSE must not increase beyond 1e-12, SSIM and fixed
observed ArcFace must not decrease beyond 1e-6 in any retained group. Both
source-level degraded structure changes must be nonnegative. Brightness-only
gain fractions were zero for all four proposals; colour changes can still occur
and remain subject to appearance preservation and visual review.

At the larger rate, ordinary source-level raw/PNG structure changes were
**−0.2820%/−0.2770%** for `dataset/asian_faces`, versus **+1.4585%/+1.4858%**
for `dataset/thumbnails128x128`. Constrained changes were
**−0.3425%/−0.3302%** versus **+1.4544%/+1.4705%**. These names are source
labels. They do not establish ethnicity, native capture location, or Zamboanga
performance. The 100 cases are paired photographic TRAIN inputs across five
synthetic profiles, not native CCTV or reserved final identities.

All **20 exact comparison sheets and 800 cells** were inspected at original
resolution. Clear glasses, visible hair, hands and fabric remain present;
the sheets do not test covering removal or hidden-region completion. Eyes,
mouths, teeth and hair detail remain soft. Larger ordinary proposals often
change skin and background colour. No convincing whole-face clarity gain
supports promotion from this review. This assistant review is not the independent
final reviewer required by the goal.

[All20 bound comparison sheets](<C:/xampp/htdocs/YEAR 4/Testing/outputs/cctv_dgp_bank_saved_direction_forward_v1_r2_analysis/sheets>).
[Complete visual notes](<C:/xampp/htdocs/YEAR 4/Testing/outputs/cctv_dgp_bank_saved_direction_forward_v1_r2_analysis/visual_review.json>).
[Independent raw/PNG audit](<C:/xampp/htdocs/YEAR 4/Testing/outputs/cctv_dgp_bank_saved_direction_forward_v1_r2_analysis/independent_audit.json>).
[All800 saved metric rows](<C:/xampp/htdocs/YEAR 4/Testing/outputs/cctv_dgp_bank_saved_direction_forward_v1_r2_analysis/finite_metrics.json>).

## Cause and next diagnostic

The invalid assumption was that **two-reference, first-order aggregate
constraints would preserve every source/profile group after a finite change**.
The saved direction used only two of 20 TRAIN references. SSIM and ArcFace
constraints averaged five profiles, leaving separate profile behaviour
unprotected. The detail constraint and evaluation both use the four degraded
profiles: a clear/degraded detail-scope mismatch was checked and rejected.

The baseline losses agree in original metric units. For the sampled
`tr_asian_00048` at constrained rate1e-5, the predicted degraded-MSE improvement
was **0.0010%**, while the finite result worsened it by **0.0193%**. This is a
counterexample even on a derivation reference. It does not prove which individual
nonlinear layer causes the discrepancy. The other18 references also lack
constraints, and source/profile averaging leaves additional regressions visible.

The next bounded packet **measures** MSE, SSIM loss and ArcFace loss for each
source/profile combination, plus the two source/degraded feature groups, over
all20 frozen references. It saves 32 aggregate gradient vectors from 320 queries
and four sampled clear-profile vectors. The supplied previous directions are
used only for dot-product diagnosis. It performs **zero parameter displacements,
optimizer updates or training epochs**. Returned evidence must be audited before
a new direction is derived. Any later finite proposal needs a bounded line
search and actual raw/PNG gates; no preserved requirement is relaxed.

The original [GEM paper](https://papers.nips.cc/paper_files/paper/2017/file/f87522788a2be2d171666752f97ddebb-Paper.pdf)
explicitly bases gradient constraints on local linearity and representative
samples. [Boyd and Vandenberghe's descent-method notes](https://web.stanford.edu/~boyd/cvxbook/bv_cvxslides.pdf)
describe testing finite step lengths through backtracking. These inform the
diagnostic design. Their results do not establish convergence, preservation or
quality for this nonconvex face model.

[Saved first-order versus finite comparison](<C:/xampp/htdocs/YEAR 4/Testing/outputs/cctv_dgp_bank_saved_direction_forward_v1_r2_analysis/finite_direction_review.json>).
[New five-step manual diagnostic](<C:/xampp/htdocs/YEAR 4/Testing/CCTV_DGP_BANK_GROUP_GRADIENTS_V1_VM.md>).

## Local checker repairs

The prepared return checker incorrectly equated two independent clock reads.
The actual projection timestamp followed the second logged timestamp by
**72.156 microseconds**. Its repaired
check verifies recorded timestamp order and the unchanged 900-second cap.
Both invalid clock-order fixtures were rejected; metric/gate/support/PNG/state
hash functions remained AST-identical. The VM return and original checker were
preserved; no VM rerun occurred.

An additional diagnostic checker compared normalized SSIM loss with a tolerance
intended for original metric units. The original saved visual receipt and checker
remain preserved. The corrected check compares original metric units: maximum
sampled SSIM arithmetic discrepancy **5.023e-7**, below 1e-6. This numerical
cross-implementation check did not change scientific quality gates or the
rejected preservation result.

## Goal remains active

Permanent baseline checkpoint SHA256:
`646fbb11674e8882ea41e251d6e6e7d468c963979b48c93132def8292834e24b`.
The current restorer remains a fine-tuned DeblurGAN-v2-compatible baseline.
Replacement B uses an externally trained frozen StyleGAN2 prior; its prior is
not our trained model, and this diagnostic does not qualify replacement B.
Original checkpoints, splits, caches, failed runs and existing app design are
preserved. No new incumbent or agent VM connection/compute occurred.

All five milestones, native/final/app validation and all seven automatic versus
assisted covering families remain required. Clear glasses, non-obstructing hair
and visible appearance must be preserved. Insufficient usable structure needs
a clearer or less-covered crop. Completion must show the original, automatic
editable removal mask and one plausible estimate, with PNG and optional bundle
downloads; it must not claim exact hidden identity. No local training, automatic
historical pilot, ethnicity or Zamboanga-performance claim is permitted.
