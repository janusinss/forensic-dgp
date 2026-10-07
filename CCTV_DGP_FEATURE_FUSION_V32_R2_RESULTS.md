# V32 r2: verified early structure stop — 7 October 2026

V32 r2 trained50 of its maximum800 updates on the existing NVIDIA L4. It stopped
at the prescribed **1%** early structure requirement: measured improvement was
**0.9706721697%**. This is a deliberate research stop. The routing repair worked;
the result is not another directory/preflight error. Export finished successfully,
but the model did not qualify. Preserve the stopped checkpoint, logs and failed
gate. Do not resume this run, rerun the recipe unchanged or promote it to the app.

**Location:** `cctv_dgp_feature_fusion_v32_training.py:152` in the immutable R2 packet.
**Cause:** delivered-PNG landmark high-frequency error decreased by less than1%.
**Next action:** audit/review the stopped result and discuss the learning design
before another model change. Removing the assertion does not fix restoration.

The original own-trained DGP remains the application's primary restorer. Its
checkpoint, observed-canvas256 processing, frozen normalization, selector, app
design and current frontend are unchanged. No local training or agent-launched
VM training occurred during download, independent audit or development review.

## Return and independent verification

The user authorized direct download after a local search found only the R1 return.
The three R2 files were downloaded separately through gcloud with the known host
key, into a distinct staging folder. Transfer/verification finished in302.07s,
with zero VM writes or training launches. The1,324,635,165-byte archive matches
SHA256 `576b3897a9a9589ddad8896e71c827481086b26ffab6e6cbeadb9a5dbc4032e9`.
Both original sidecars match. Existing files were preserved.

The prospective local checker verifies27,625 regular allowlisted return files,
the frozen R2 protocol/source,5,467 approved TRAIN assets,75,324,711 saved
gradient values, finite selected23 connectivity and unchanged frozen parameters.
It recomputes all7,810 rows from the two complete3,905-case snapshots, including
PNG metrics, saved vectors and mean-only controls. Returned code is not executed.
Audit runtime was1004.73s. No local autograd, backward or optimizer update occurs.

CPU inference replay covers the initial50 proof cases and50 fixed previews at
each snapshot:150 original-DGP,100 candidate and250 recognizer forwards. It does
not replay all3,905 stopped neural outputs. Maximum raw preview discrepancy is
0.0000025034, within the existing0.00001 bound; PNG difference is at most one byte,
and maximum saved-vector discrepancy is0.0000002874. Numerical replay tolerances
do not change the1% gate or its failed decision.

[Independent return audit](<C:/xampp/htdocs/YEAR 4/Testing/outputs/cctv_dgp_feature_fusion_v32_r2_independent_audit.json>)
[Verified download receipt](<C:/xampp/htdocs/YEAR 4/Testing/outputs/cctv_dgp_feature_fusion_v32_r2_download_v1/download_receipt.json>)

## What improved, and what did not qualify

| Fixed paired photographic TRAIN observation | Result |
| --- | ---: |
| Original degraded feature error | 0.001876217941608562 |
| R2 stopped50 feature error | 0.001858006016206468 |
| Relative gain / early minimum | 0.970672% /1% — failed |
| Delivered MSE/SSIM/recognizer preservation groups at50 | All17 pass |
| Completed epochs / updates | 0 complete epochs /50 |

The feature error measures high-frequency disagreement inside fixed landmark
regions in delivered PNGs against paired TRAIN photographs. It is not PSNR,
native CCTV accuracy, a whole-face quality certificate or recovered identity.
Passing preservation-group averages does not certify every individual feature
and does not waive the structure requirement.

| Photographic source | Degraded cases | Structure-error reduction |
| --- | ---: | ---: |
| dataset/asian_faces | 1,560 | 0.952033% |
| dataset/thumbnails128x128 | 1,564 | 0.975071% |

Both source groups improve, but neither reaches1%. Source labels are not ethnicity.
No Zamboanga CCTV performance is inferred. Synthetic paired observations remain
separate from unpaired native CCTV and reserved final evaluation.

| TRAIN diagnostic subset | Degraded cases | Error reduction |
| --- | ---: | ---: |
| All TRAIN | 3,124 | 0.970672% |
| Optimized by update50 | 200 | 0.919562% |
| Not yet optimized by update50 | 2,924 | 0.974922% |
| Fixed preview subset | 40 | 0.991558% |

The50 paired updates touch50 references/250 cases, including50 clear controls.
None of the fixed50 previews has been optimized by update50, but they are TRAIN
and already used for preflight/normalization. They are not held-out evaluation.
Worker runtime was1033.71s; allocated VRAM8,942,142,976 bytes is within20GiB.
No800-update result exists, and the stopped run cannot establish its outcome.

## All50 previews and the measured parameter path

All10 original-detail sheets and250 exact256px image cells were actually viewed:
input, retained DGP, V31 stopped50, R2 stopped50 and paired TRAIN target. Every
row was inspected for eyes, nose, mouth, face outline and overall visible
appearance. No resizing or display enhancement is applied to these cells.

Broad expression/outline generally remain. Clear glasses, hair, cap/earrings,
facial hair and adjacent hands remain in review. Clear inputs are softened;
degraded eyelids, nostrils and lips stay diffuse. Motion cases retain more readable
broad structure than severe compound cases. R2 gives small changes over V31,
without a convincing incremental whole-face clarity gain. Severe compound inputs
need a clearer crop. This development agent review is not independent final review.

A saved-array analysis confirms changes in both fusion and decoder weights;
their relative L2 movements are0.338703% and0.338540%, respectively. Frozen
parameters stay identical. The initial three improvement-gradient dot products
with the actual50-update displacement are negative; the cosine with the negative
initial total gradient is0.122446. These are coordinate-dependent first-order
observations at the original point. They neither reconstruct AdamW history nor
identify a unique cause of the weak finite result. No model, gradient or optimizer
is constructed for that analysis.

[All50 observations](<C:/xampp/htdocs/YEAR 4/Testing/outputs/cctv_dgp_feature_fusion_v32_r2_failure_review_v1/visual_review.json>)
[Independent comparison-cell readback](<C:/xampp/htdocs/YEAR 4/Testing/outputs/cctv_dgp_feature_fusion_v32_r2_failure_review_v1/independent_preparation_audit.json>)
[Saved parameter-path measurements](<C:/xampp/htdocs/YEAR 4/Testing/outputs/cctv_dgp_feature_fusion_v32_r2_displacement_v1/analysis.json>)

## Next boundary

V30, V31 and V32 missed the same early requirement. The invalid assumption is
that favourable initial gradients plus more original fusion parameters would
deliver sufficient finite whole-face improvement with the unchanged learning
design. The measured gains do not support that assumption. They do not prove
that every original DGP decoder or possible optimization scheme is incapable.

The user answered the required design discussion: "apply the best approach and
do research also if needed". The selected direction inspects the current objective
and effective updates using saved evidence and primary research before another
training change. Keep all visible features,
preservation losses, original checkpoints, frozen splits and1%/10% gates.
No new trainer or automatic historical launch is released by this milestone.
The user retains manual control of any justified finite L4 pilot.

[Post-V32 design discussion](<C:/xampp/htdocs/YEAR 4/Testing/CCTV_DGP_POST_V32_ARCHITECTURE_REVIEW.md>)

The separate converted-MAT32-case assisted comparison is technically verified
but unqualified for consistent covering-family quality. It is not promoted.
Automatic/assisted completion, useful native DGP output and independent final
review remain outstanding. The goal is active and incomplete.
