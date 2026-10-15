# Signed small-step result — 13 October 2026

The manual diagnostic completed successfully in **58.14 seconds**.
It produced useful learning-path evidence, but **no qualified improved model**.
There were zero optimizer updates, gradients or training epochs and no new
checkpoint. The original initial model was restored. All four proposals remain
unqualified; the two negative steps are diagnosis only.

| Additive step | Raw structure gain | PNG structure gain | Raw failed checks | PNG failed checks | Decision |
|---:|---:|---:|---:|---:|---|
| -0.001 | -0.102793% | -0.079746% | 51 | 49 | Diagnosis only |
| -0.0001 | -0.010033% | -0.009201% | 51 | 47 | Diagnosis only |
| 0.0001 | 0.009962% | 0.018381% | 0 | 6 | Rejected |
| 0.001 | 0.096543% | 0.107710% | 6 | 9 | Rejected |

Failure counts are source/profile group–metric checks, including the brightness
requirement where applicable; they are not subjects. The 1% structure requirement
and both-source raw/PNG requirements remain unchanged. At +.0001 the source raw
gains are0.009976% for `dataset/asian_faces` and0.009959% for
`dataset/thumbnails128x128`; both are below1%. Dataset names are source labels,
not ethnicity or native CCTV evidence.

## What the signed experiment established

All32 measured central slopes are negative at both .001 and .0001. Raw MSE,
SSIM loss and feature-loss slope magnitudes are close to their saved first-order
predictions. ArcFace central-slope error falls on the smaller radius for all10
groups, but the largest remaining relative difference is about34.7%. Accounting
for actual float32 parameter rounding leaves that discrepancy substantially
unchanged. These results support a local descent tendency; they do not prove
every backward path correct or identify a specific broken layer.

Finite changes can still reverse a favorable local prediction. For example,
`dataset/thumbnails128x128/clear/MSE` at +.001 has a central linear contribution
of−9.18650e−8 and a positive midpoint departure of1.49353e−7, giving an actual
increase of5.74876e−8. This is measured behavior along this direction, not an
explanation of every layer. Simply increasing a fixed step or adding epochs
to the failed recipe is unsupported.

At +.0001 **raw preservation has zero failed checks**, while the PNG has six.
The delivered checks include clear-source MSE/SSIM/ArcFace regressions, two
degraded ArcFace regressions, and brightness fraction0.279498 above0.2.
The raw brightness fraction is0.077726. Both views remain required.

The unchanged converter is `floor(float32*255)`. At +.0001 it changes159,154
observed channel values across156,922 pixels in100 images, by at most one byte
level. About18.67million raw channel values changed before quantization.
Two individual cases improve raw MSE while worsening PNG MSE. This traces a
delivery difference; it is not a claim that quantization alone causes all model
defects, a reason to weaken the gates, or proof of visible identity preservation.

[Signed slopes](<C:/xampp/htdocs/YEAR 4/Testing/outputs/cctv_dgp_bank_group_step_check_v1_analysis/signed_derivative_review.json>).
[Actual rounded-step gradient recount](<C:/xampp/htdocs/YEAR 4/Testing/outputs/cctv_dgp_bank_group_step_check_v1_analysis/actual_step_direction_review.json>).
[All400 PNG/raw processing recounts](<C:/xampp/htdocs/YEAR 4/Testing/outputs/cctv_dgp_bank_group_step_check_v1_analysis/quantization_review.json>).

## Independent saved-evidence checks and visual outcome

Archive bytes:296,743,929; SHA256
`0b1885ca09688d376bee0c1a19dcd5a37c16537a59787bd0b8f5e603df1181d8`. All596 returned files and5,852 parent
bindings passed. The separate R1 saved-output audit verifies100 exact control
receipts,400 raw arrays/400 exact floor PNGs,800 metric rows,17 source/profile
groups, four reconstructed disposable state hashes, and20 sheets/800 RGB cells.
Peak allocated CUDA memory:1.106 GiB. This is an
independent audit of saved data, not a fresh neural reproduction of the proposals.

All20 sheets were actually inspected at requested original detail. Visible
eyes, lips, teeth, ordinary glasses and hair remain soft. The four small changes
are visually close to the retained restorer; there is no convincing useful
whole-face clarity gain. Narrow observed canvases preserve their side padding.
This is primary-assistant development review, not independent final human review.

The original prospective checker stopped on exact equality of derived squared
parameter-change sums. All148 parameter-byte hashes match; 835 of888 statistics
match exactly. The other53 differ only in three float64 reduction summaries,
by at most two representable rounding steps. The original source and failure
remain. A separate R1 checker permits that two-ULP reduction difference only;
counts, maxima, parameter/state hashes, image bytes and every quality gate remain
unchanged. Its inverse source/AST and rejection fixtures pass. NumPy explains
that summation precision depends on reduction details.
[NumPy summation reference](https://numpy.org/doc/stable/reference/generated/numpy.sum.html).

[R1 complete saved-output audit](<C:/xampp/htdocs/YEAR 4/Testing/outputs/cctv_dgp_bank_group_step_check_v1_analysis/independent_audit_r1.json>).
[Original checker stop](<C:/xampp/htdocs/YEAR 4/Testing/outputs/cctv_dgp_bank_group_step_check_v1_analysis/original_parameter_statistics_audit_stop/failure.json>).
[All20 actual visual observations](<C:/xampp/htdocs/YEAR 4/Testing/outputs/cctv_dgp_bank_group_step_check_v1_analysis/visual_review.json>).
[Independent review readback](<C:/xampp/htdocs/YEAR 4/Testing/outputs/cctv_dgp_bank_group_step_check_v1_analysis/review_binding_audit.json>).

## Next design and the user's five-item summary

The selected next design accounts for both raw and delivered losses, uses small
actual checked changes and refreshes directions as weights move. It retains the
current app checkpoint and the frozen initial replacement state; no random-weight
restart is justified. A prospective exact-PNG-forward/surrogate-backward design
needs a finite manually run L4 test before extra epochs. It is design work only,
not a sealed packet or launch-ready training recipe.
[Concrete next learning design review](<C:/xampp/htdocs/YEAR 4/Testing/CCTV_DGP_DELIVERED_LOSS_LEARNING_REVIEW_V1.md>).

The user's summary accurately describes model diagnosis, fair comparison inputs,
functional app testing, cold-backup offload and a finite VM diagnostic. The
recorded storage offload freed52,028,568,639 logical bytes (**52.03 GB/48.45 GiB**)
locally; it did not change VM storage. Its recorded Drive audit retains4,457 files
and all active assets. No new cloud inventory is claimed by this model audit.
[Verified storage closure](<C:/xampp/htdocs/YEAR 4/Testing/GOOGLE_DRIVE_OFFLOAD_COMPLETE_20261013.md>).

Fresh prior app checks verified14 actual PNG/ZIP downloads,10 backend refusals
and required responsive layouts. This is functional evidence, not model usefulness
or covering-family qualification.
[Bounded app flow](<C:/xampp/htdocs/YEAR 4/Testing/CCTV_DGP_APP_FLOW_AUDIT_20261012.md>).

The separate completion saved-evidence recount verifies2,529 bindings/1,182 files
and exact outside-mask preservation across five retained comparisons. Among14
eligible covering photographs per condition, the retained automatic detector is
empty on4 original photos and11 degraded photos. An uncovered original control
also has11,683 false-marked pixels. Masks are approximate operator footprints,
not expert segmentation truth. Generated anatomy and outside-mask remnants remain
distinct problems; all seven automatic/assisted families stay unqualified.
[Saved completion readiness recount](<C:/xampp/htdocs/YEAR 4/Testing/outputs/dgp_covering_readiness_20261012_v1/audit.json>).

## Goal and evidence boundaries

The full goal is unchanged and incomplete. The permanent app baseline SHA256 is
`646fbb11674e8882ea41e251d6e6e7d468c963979b48c93132def8292834e24b`.
It is an own fine-tuned DeblurGAN-v2-compatible restorer. The proposed replacement
has an own conditioning/reconstruction path with an externally pretrained frozen
StyleGAN2 prior; there is no own-trained-prior claim. No model or app is promoted.
Original checkpoints, splits, full states, caches, provenance, rollback versions
and failures remain retained.

All five milestones still apply: native DEV/separate final identities and source
contracts; identical-input model comparisons with raw/display separation;
demonstrated finite improvements; qualified primary DGP with Auto/override;
independently useful DEV/final review, regressions and full app verification.
These100 paired photographic TRAIN cases are not native CCTV. No native PSNR/SSIM,
ethnicity, Zamboanga-performance or exact hidden-identity claim is made. Reserved
final identities remain outside tuning and their pixels remain unopened.

Separately retain completion for masks, sunglasses, strong glare, hands,
obstructing hair, scarves and objects. Show automatic removal support for optional
correction; preserve clear glasses, ordinary hair and visible appearance with a
documented small margin. Request clearer/less-covered input when insufficient.
Return original/mask/one plausible estimate with PNG and optional bundle. Automatic
and assisted results and independent final review remain separate and outstanding.

All actual training, gradients and live model proposals remain human-manual L4
tmux work. Agent VM access remains inventory-first hash-bound maintenance only.
This turn runs saved-data audits; it launches no VM or local neural job. Every
future pilot needs finite updates/epochs, timing/projection, VRAM/RSS, free-space,
reserve, output and export stops. Credits/migration notifications and the
November2026 thesis deadline remain. More epochs require demonstrated preserved
learning; the conditional five-epoch plan is not activated by this result.
