# Conditioning learning R4 returned result — 14 October 2026

**No trial qualified and no training update was accepted.** R4 completed the
previously failing scientific-loss alignment stage, all 640 gradient queries,
and three actual temporary conditioning changes. Every change failed retained
RAW/PNG preservation. The worker stopped at
`scripts/cctv_dgp_conditioning_learning_v1_r4_vm.py:364` with
`No actual raw/PNG preserving conditioning step; retain stop`.

| Trial | RAW structure gain | PNG structure gain | Failed preservation checks RAW / PNG | Decision |
| --- | ---: | ---: | ---: | --- |
| r00_trial00 | 0.688521% | 0.705580% | 31 / 31 | Rejected |
| r00_trial01 | 0.578503% | 0.600541% | 25 / 25 | Rejected |
| r00_trial02 | 0.345093% | 0.359206% | 13 / 8 | Rejected |

The largest trial still missed the 1% global and per-source endpoint. Its two
source gains were 0.635030% / 0.701935% RAW and 0.639203% / 0.722234% PNG.
Reducing the step improved preservation but left 13 RAW and 8 PNG failed checks
at the smallest trial. Brightness fractions stayed below 0.2 in all trials.
The failures include clear-image preservation, SSIM and the fixed-recognizer
appearance comparison. Recognizer similarity is not verified identity.

The manual L4 worker reports 237.05 seconds, 100 candidate
forwards, 200 recognizer forwards, 640 gradient queries, three parameter
proposals and zero accepted updates/epochs. The export took 80.13 seconds;
`complete: true` means the failure archive was created. No accepted component
checkpoint or new incumbent exists. All trial components and outputs remain.
The exception handler reports no rollback error; no fresh post-failure RAM-state
hash is claimed.

All 663 returned files and 5,852 required parent
bindings pass hash checks. Archive size is 1,384,730,860 bytes;
SHA256 `528fbc2315bb97c9378658e9e3aee99460589c087ae946ca91979dd34f6f541f`. The separate R1 checker independently
recomputes four snapshots/400 RAW and PNG model outputs/800 metric rows, visible
support compositing, selected-component state schemas, all 64 saved derivative
vectors, Gram/KKT arithmetic, rounded changes, rates and all three rejections.
Initial RAW controls equal the permanent DGP exactly. No neural derivative or
embedding inference was repeated locally.
[Independent returned audit](<C:/xampp/htdocs/YEAR 4/Testing/outputs/cctv_dgp_conditioning_learning_v1_r4_analysis/failure_audit.json>).

The original checker stopped on exact equality of derived receipt floats.
Four-snapshot replay reproduced differences up to 4.8225313e-13 while scientific
image-score replay passed its original 1e-13 bound. R1 binds the stored receipt
exactly to stored scientific scores and requires every independent quality
branch/failure member to agree exactly. It adds no scientific or replay
tolerance. Three real receipt regressions pass; 18 corrupted records are
rejected. Original checker, stopped check and failed regression remain retained.
[Checker repair evidence](<C:/xampp/htdocs/YEAR 4/Testing/outputs/cctv_dgp_conditioning_learning_v1_r4_analysis/return_audit_r1_preparation.json>).

All 20 full-resolution sheets and 600 cells were directly inspected. The clear
images retain softened eye/lip/teeth/hair detail; degraded features remain
diffuse. The three trials supply no convincing whole-face clarity advantage.
This is 20 exposed photographic TRAIN references and 100 synthetic paired
cases, not native CCTV, independent final identities or completion evidence.
No ethnicity, Zamboanga performance or hidden-identity claim is supported.
[Every-sheet visual review](<C:/xampp/htdocs/YEAR 4/Testing/outputs/cctv_dgp_conditioning_learning_v1_r4_analysis/visual_review.json>).

The invalid assumption was that a feasible local descent certificate and a
small relative parameter change establish finite image preservation. After the
third stop, the diagnostic question was asked; the human answered “apply best
approach.” The saved-data review is now complete: float32 parameter rounding
did not reverse any predicted descent sign, yet actual losses worsened.
The joint 64-gradient Gram condition number is 884,961,456, compared with
4,164 for RAW32 and 4,138 for PNG32. This identifies a loss/step-design hypothesis,
not proof of a unique cause or a new passing model.
[192 saved prediction-versus-output comparisons](<C:/xampp/htdocs/YEAR 4/Testing/outputs/cctv_dgp_conditioning_learning_v1_r4_analysis/gradient_prediction_review.json>).
[Design decision and primary research](<C:/xampp/htdocs/YEAR 4/Testing/CCTV_DGP_POST_R4_GRADIENT_DESIGN_REVIEW_V1.md>).

Original checkpoint SHA256 remains
`646fbb11674e8882ea41e251d6e6e7d468c963979b48c93132def8292834e24b`.
R4 is used and stopped; do not rerun or resume its failed recipe. New training,
longer epochs, app promotion and completion qualification are not authorized by
archive completion. Actual learning stays human-manual on the existing L4.
All five milestones and seven automatic/assisted covering families remain
required; the full goal is active and incomplete.
