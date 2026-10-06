# V19 r2 returned results — 5 October 2026

**Execution, independent local audit and development sheet review are complete.
The normalization correction passes; both the spatial candidate and automatic
selector still fail the unchanged appearance-preservation gates. Close these
launch commands. Keep the original models, splits, protocols and failures.**

## Verified return and execution

The 1,507,038,572-byte archive, checksum sidecar, exported receipt and safely
imported results agree. The protocol and all 189 frozen assets match the prepared
package. Fresh training parity passes all 50 cases. All 520 canonical DGP PNGs
exactly match the V15 baseline; 50 saved raw previews meet the unchanged 2e-6
limit. The separate internal spatial DGP bases remain explicit.

| Binding | SHA256 |
| --- | --- |
| Protocol | 5c128d6715785f84858f762035a168f396b46787d6a864b1d8d6437ffc69dc3f |
| Returned archive | fefc78e16ab9a1f4d78d7faf4a8344ad3eede088918ce5d522b266760728ef10 |
| Returned results JSON | 98c311465351b36c20d26eca2a1d2310194b1ac255ebfad82f6d49b5c039f9c8 |

L4 inference takes 175.04s, VM saved-output audit 68.95s, and full supervisor/export
356.31s. The case-20 timing projection is 265.42s against the original 1,200s cap.
Peak allocated VRAM is 855,550,976 bytes, approximately 0.797GiB. All six state
hashes agree before/after. Reported forwards are 1,090 DGP; 570 each prior encoder,
classifier, R2 head, feature generator and spatial decoder; 624 recognizer; zero
prior RGB tail/unused V11. No optimizer was constructed; zero updates/backwards.

Independent Windows audit takes **97.61s**. It verifies 5,957 artifact bindings,
2,080 physical PNG metric/cosine rows, 1,040 raw/PNG compositions, 520 internal DGP
bases, 50 canonical raw previews, 570 input decisions, 520 exact automatic aliases,
50 training parity cases and 300 original grid cells. All 24 cached CPU decoder
replays pass, maximum difference **7.15255737e-7**, below the unchanged 5e-5 limit.
It does not independently replay the full DGP/prior/recognizer or CUDA execution.

Audit: [local_full_audit.json](<C:/xampp/htdocs/YEAR 4/Testing/outputs/cctv_dgp_input_selection_return_v19_r2/local_full_audit.json>).
Transfer/import: [local_import_and_audit.json](<C:/xampp/htdocs/YEAR 4/Testing/outputs/cctv_dgp_input_selection_return_v19_r2/local_import_and_audit.json>).

## Paired synthetic development comparison

The repeatedly used cohort contains 104 photographic development identities:
51 from `dataset/asian_faces`, 53 from `dataset/thumbnails128x128`. Each has five
synthetic profiles. Own exact identity/source/target-hash overlap with training
is excluded; near duplicates and pretrained corpus overlap remain unexcluded.
Folder labels are not ethnicity or CCTV capture provenance. These results do
not establish native CCTV, Zamboanga or independent final performance.

PSNR below is derived from each group's mean observed-support MSE. The declared
pretrained baseline reuses audited V15 `starting_prior_none` PNGs/embeddings
(statistics none, fidelity w0); no fresh pretrained raw output is claimed.
All arms use the same source RGB256 crops, support and original floor composition.
Raw arrays and composed PNGs remain separate; there is no display enhancement.
ArcFace cosine is a fixed-affine comparison proxy, not an identity verification.

**416 degraded synthetic cases:**

| Arm | PSNR dB | SSIM | Fixed ArcFace cosine |
| --- | ---: | ---: | ---: |
| Basic resizing | 14.482 | 0.57553 | 0.33262 |
| Retained DGP | 16.027 | 0.61930 | 0.33011 |
| Declared pretrained prior baseline | 14.055 | 0.49438 | 0.19348 |
| Fixed V18 spatial output | 19.053 | 0.60451 | 0.22278 |
| Automatic V19 | 19.054 | 0.60458 | 0.22346 |

**104 clear synthetic controls:**

| Arm | PSNR dB | SSIM | Fixed ArcFace cosine |
| --- | ---: | ---: | ---: |
| Basic resizing | Exact input/target match | 1.00000 | 1.00000 |
| Retained DGP | 31.132 | 0.92512 | 0.95701 |
| Declared pretrained prior baseline | 24.071 | 0.73311 | 0.59986 |
| Fixed V18 spatial output | 26.265 | 0.87990 | 0.82579 |
| Automatic V19 | 31.065 | 0.92364 | 0.95109 |

Automatic degraded PSNR gains 3.027dB and MSE falls 50.19%, but SSIM falls
0.61930→0.60458 and cosine falls 0.33011→0.22346. **30 group/metric preservation
checks fail**; the unconditional spatial output fails 33. Capacity improvement
passes, preservation/generalization qualification fails for both. All original
source/profile checks and tolerances remain; none is waived.

Blur and motion PSNR regress in both source folders. Lowlight/compound PSNR gains
dominate the aggregate improvement. Cosine regresses in every degraded
source/profile group; SSIM regresses in blur, lowlight and motion for both sources.
Full source/profile/group values: [all_groups.csv](<C:/xampp/htdocs/YEAR 4/Testing/outputs/cctv_dgp_input_selection_v19_r2_analysis/all_groups.csv>).

The selector retains DGP for 101/104 clear cases and 2/104 motion cases. It uses
the spatial output for all blur, lowlight and compound cases and 102 motion cases.
Three clear inputs in `dataset/asian_faces` cross the frozen low-detail branch:

| Case | DGP cosine | Selected spatial cosine |
| --- | ---: | ---: |
| va_asian_02479_clear | 0.97074 | 0.80723 |
| va_asian_01748_clear | 0.97906 | 0.73034 |
| va_asian_03286_clear | 0.95618 | 0.75255 |

All three also lose SSIM. Only the first improves MSE. This exposes a generalization
limit of the ten-face training-calibrated input rule. Keep its failed results;
do not refit its thresholds on this reused development cohort. It is not a face,
pose, covering or insufficient-information classifier.
Decisions: [input_choices.csv](<C:/xampp/htdocs/YEAR 4/Testing/outputs/cctv_dgp_input_selection_v19_r2_analysis/input_choices.csv>).

## Original-cell development review

All five fixed sheets were inspected at original resolution: ten predetermined
identities/fifty synthetic cases/300 cells. This is Codex development review;
all 520 outputs were not individually reviewed, and independent final review
remains pending. Appearance changes visible in these sheets corroborate the
negative preservation checks.

| Original sheet | Observed limitation |
| --- | --- |
| [Clear](<C:/xampp/htdocs/YEAR 4/Testing/outputs/cctv_dgp_input_selection_return_v19_r2/outputs/input_selection_v19_r2/grids/clear.png>) | Retained DGP usually preserves visible appearance with softness. Spatial output changes some eye/mouth detail and adds colour/texture changes. |
| [Blur](<C:/xampp/htdocs/YEAR 4/Testing/outputs/cctv_dgp_input_selection_return_v19_r2/outputs/input_selection_v19_r2/grids/blur_lr24.png>) | Spatial/automatic output adds patchy colour and unstable facial detail; clear glasses become less stable. |
| [Lowlight](<C:/xampp/htdocs/YEAR 4/Testing/outputs/cctv_dgp_input_selection_return_v19_r2/outputs/input_selection_v19_r2/grids/lowlight_lr32.png>) | Exposure rises alongside mottled colour/texture and changed facial detail; monochrome input can acquire colour. |
| [Motion](<C:/xampp/htdocs/YEAR 4/Testing/outputs/cctv_dgp_input_selection_return_v19_r2/outputs/input_selection_v19_r2/grids/motion_lr48.png>) | Eye/mouth appearance and glasses edges lose detail or shift under the spatial output. |
| [Compound](<C:/xampp/htdocs/YEAR 4/Testing/outputs/cctv_dgp_input_selection_return_v19_r2/outputs/input_selection_v19_r2/grids/compound_lr24.png>) | Brightening leaves coarse/patchy faces and weak or changed eye/mouth/glasses detail. Heavy degradation retains little usable facial information. |

The declared pretrained prior baseline also produces fragmented/changed features
on several degraded previews. This observation concerns the specific frozen
comparison path. The review does not calibrate an input rejection rule or infer
exact hidden appearance. Review receipt:
[visual_review_original_v19_r2.json](<C:/xampp/htdocs/YEAR 4/Testing/outputs/cctv_dgp_input_selection_v19_r2_analysis/visual_review_original_v19_r2.json>).

## Separate saved-output brightness diagnostic

A bounded 17.23s local analysis performs zero neural calls/training. One fixed
decomposition adds the mean observed luminance difference between the spatial
candidate and canonical DGP to the DGP raw, uniformly across RGB, before the
unchanged clipping/PNG composition. Candidate computation uses no target, label
or identity. Targets only score it; no coefficient search or selector refit.

On the same 416 degraded cases, this yields PSNR 18.388dB and SSIM 0.62427,
with 41.94% MSE reduction against retained DGP. It reproduces 83.59% of the full
spatial candidate's mean MSE reduction in these cases. **13 MSE/SSIM checks still
fail**, including clear/blur/motion groups. Fresh recognizer values were not
computed; full original qualification is unavailable. This is exploratory
component analysis, not a new accepted restorer or independent validation.

The result supports separating photometric correction from spatial synthesis.
The invalid assumption was that aggregate PSNR/capacity gain would preserve
visible facial structure. A uniform component accounts for much of that pixel
gain without establishing reconstruction of lost facial information. Future
work must retain clear/blur/motion appearance and test insufficient-information
handling; adding spatial detail or scaling the old recipe alone is unjustified.

Diagnostic: [uniform_luminance_diagnostic.json](<C:/xampp/htdocs/YEAR 4/Testing/outputs/cctv_dgp_input_selection_v19_r2_analysis/uniform_luminance_diagnostic.json>).
Reproducible source: [analyze_cctv_dgp_input_selection_v19_r2.py](<C:/xampp/htdocs/YEAR 4/Testing/scripts/analyze_cctv_dgp_input_selection_v19_r2.py>).
The analysis's earlier manifest-filename startup error is retained separately;
it occurred before any helper import, destination creation or neural call.
No original frozen source/package was modified to run the analysis.

## Milestone decision and remaining goal

Close V19 r2 as executed, fully audited and negative for preservation. Do not
repeat the frozen job, lower its gates, select its decoder for the app, or retune
the selector on these development results. Keep the original V19 stop, confirmed
normalization diagnostic, V18 preservation failures and all original checkpoints.
No new training pilot or VM action was performed by the assistant.

The next design must isolate exposure correction, preserve visible structure
and appearance, and justify any separate finite VM training from training-only
evidence. Pretrained restoration remains a declared comparison. The required
DGP-led app/override, usable native CCTV outputs, insufficient-input requests,
all covering families and independent final/Playwright review remain incomplete.
Native 24/reserved 32 are untouched by this experiment. The full goal stays active.
