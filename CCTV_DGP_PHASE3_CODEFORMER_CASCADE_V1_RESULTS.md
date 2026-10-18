# Phase 3 DGP output followed by CodeFormer — 15 October 2026

[Open the interactive comparison](http://127.0.0.1:61014/cascade_v1/comparison.html).
[Persistent gallery](outputs/cctv_dgp_checkpoint_comparison_v1/cascade_v1/comparison.html).
[Your four examples](outputs/cctv_dgp_checkpoint_comparison_v1/cascade_v1/sheets/cascade-01.png).

The requested sequence is tested on all 88 fixed development cases:
**original input → `checkpoints/dgp_zamboanga_final.pth` saved PNG → CodeFormer w=1**.
The page compares this sequence with Phase 3 alone, CodeFormer alone and resizing.
Model choice remains with the user; this result selects no checkpoint or app default.

## What the images show

The sequence has mixed effects. In the two severely degraded `va_asian_01005`
examples supplied by the user, it reduces several severe texture distortions
from direct CodeFormer. The results remain soft or incomplete and differ from
the known paired reference. In `va_ffhq_37228_blur_lr24`, the sequence removes
glasses that are present in the reference and changes hair shape. The low-light
case of that same face also loses glasses. Coherent-looking detail is therefore
not sufficient evidence of preservation.

On native QMUL-SurvFace crops, several dark inputs produce near-black cascade
outputs with colored edge noise. Many tiny inputs still lack reliable facial
structure. On ChokePoint, the sequence often darkens and smooths the face;
some frames lose or weaken visible glasses. Clearer frames retain more structure,
with appearance differences between the paths. Native inputs have no aligned
clean reference, so these observations do not establish recovered identity.
Insufficient inputs retain the request for a clearer crop. Strong side views
remain visible in this development comparison but do not qualify the requested
frontal or mildly turned workflow.

## Paired measurements

These are arithmetic means for delivered native PNGs on **32 degraded synthetic
pairs**, with clear-image change measured separately on **8 clear controls**.
Higher PSNR/SSIM and lower clear change indicate closer reference similarity.
They support visual review; they do not rank native CCTV accuracy or establish
useful detail by themselves. The resize row is retained as a control.

| Processing path | PSNR (dB) | SSIM | Clear change, MAE / 255 ↓ |
|---|---:|---:|---:|
| Input / resize | 18.2263 | 0.5922 | 0.0000 |
| Phase 3 DGP alone | 16.6052 | 0.4990 | 12.1429 |
| CodeFormer alone, w=1 | 18.0292 | 0.5375 | 4.3034 |
| Phase 3 PNG → CodeFormer, w=1 | 16.6076 | 0.5106 | 12.0380 |

Compared with direct CodeFormer, the sequence has higher SSIM in 16/32 degraded
pairs and lower SSIM in 16/32; PSNR improves in 5/32 and declines in 27/32.
Mean differences are −1.4217 dB PSNR and −0.0270 SSIM. It changes all 8 clear
controls more than direct CodeFormer, with mean extra MAE of 7.7345 / 255.
These results do not demonstrate a general improvement from adding Phase 3 first.

Sources are also reported separately; source names do not imply ethnicity:

| Processing path | `dataset/asian_faces`: PSNR / SSIM | `dataset/thumbnails128x128`: PSNR / SSIM |
|---|---:|---:|
| Phase 3 DGP alone | 17.7562 / 0.5735 | 15.4541 / 0.4246 |
| CodeFormer alone | 19.8625 / 0.6400 | 16.1960 / 0.4351 |
| Phase 3 PNG → CodeFormer | 17.7756 / 0.5842 | 15.4396 / 0.4369 |

Each paired source contributes 16 degraded pairs and 4 clear controls. Native
QMUL-SurvFace (24 crops, 24 identities) and ChokePoint (24 crops, 12 identities)
are unpaired and have no PSNR/SSIM values. They are not pooled into this table.

## Execution and verification

The second stage receives the **exact saved Phase 3 native PNG**, including its
original padding. This is explicitly a PNG-intermediate test, not a float-RAW
cascade. CodeFormer receives the same weights, fidelity w=1 and AdaIN setting as
the direct control. Its existing wrapper uses 512×512 internally and returns
256×256. No extra face alignment, sharpening or completion is added. Final RAW,
model PNG and display PNG are retained separately; display policy is based on
the original degraded input. No grayscale policy activates in this cohort.

Local CPU inference made 88 new CodeFormer calls, reused all Phase 3 outputs,
and took 524.24 seconds including bookkeeping. There were zero new DGP calls,
gradient queries or optimizer updates. Timing is local CPU evidence, not a VM
training or deployment estimate. Both checkpoint states remain unchanged.

An independent agent's saved-output audit verified 941 bindings, 792 artifacts
for the three neural paths, exact intermediate linkage, output conversions,
padding and unchanged state receipts. Its independent metric recomputation
matches all 792 shared rows; 1,440 numeric values match exactly. All 27 shared
aggregate groups match within 3.56e-15. Resize-control rows are covered by the
separate measurement hash audit, outside that three-path recomputation.

The root assistant inspected all 22 planned sheets (88 cases). This is an
assistant development review, not an independent human final review. Bundled
inline Playwright verified four sources, both output views, A/B selection,
navigation, nine measurement views and exact PNG/CSV downloads. Widths 375,
768 and 1280 pass without page/console errors or horizontal overflow. These
checks verify the comparison page, not full application or model qualification.

## Retained evidence

- [Frozen protocol](outputs/cctv_dgp_checkpoint_comparison_v1/cascade_v1/plan.json), SHA-256 `726bdc381eb37d8d46575340f8a5c41b50691d3c109cb124b76126dac16ce877`; [execution receipt](outputs/cctv_dgp_checkpoint_comparison_v1/cascade_v1/receipt.json).
- [All measurements](outputs/cctv_dgp_checkpoint_comparison_v1/cascade_v1/metrics.csv), [per-case differences](outputs/cctv_dgp_checkpoint_comparison_v1/cascade_v1/cascade_vs_direct_summary.json) and [measurement hash audit](outputs/cctv_dgp_checkpoint_comparison_v1/cascade_v1/measurement_hash_audit.json).
- [Independent saved-output audit](outputs/cctv_dgp_checkpoint_comparison_v1/cascade_v1/independent_saved_output_audit_v1.json) and [independent metric crosscheck](outputs/cctv_dgp_checkpoint_comparison_v1/cascade_v1/independent_metric_crosscheck_v1.json).
- [All 22 sheets](outputs/cctv_dgp_checkpoint_comparison_v1/cascade_v1/visual_sheet_manifest.json), [visual observations](outputs/cctv_dgp_checkpoint_comparison_v1/cascade_v1/visual_review.json) and [browser verification](scratch/phase3-codeformer-cascade-browser-audit.json).
- [New sequence outputs](outputs/cctv_dgp_checkpoint_comparison_v1/cascade_v1/results/phase3_then_codeformer); original controls, provenance and terms remain in the [parent comparison](CCTV_DGP_CHECKPOINT_COMPARISON_V1_RESULTS.md).

Phase 3 checkpoint SHA-256:
`b6376f56c4161ef1f26a8f9efb0bb309b3014f4ecdeaca5399a47c1c4607586c`.
CodeFormer checkpoint SHA-256:
`1009e537e0c2a07d4cabce6355f53cb66767cd4b4297ec7a4a64ca4b8a5684b7`.

No reserved-final pixels are used. Prior training overlap is not fully excluded
for every checkpoint; the previous provenance/terms limitations still apply.
No ethnicity or Zamboanga performance is inferred. This test neither trains a
model nor changes the application. The restoration-first goal and separate
completion work remain incomplete; further model selection belongs to the user.
