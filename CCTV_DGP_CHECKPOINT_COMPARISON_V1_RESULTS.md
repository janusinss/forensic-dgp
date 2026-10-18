# Restoration checkpoint comparison — 15 October 2026

**All model results are provided below. The user will choose which model to continue.**
No checkpoint has been selected, replaced or trained by this comparison.
The gallery contains all 12 checkpoint states and the resize control.
Native CCTV detail and paired measurements are available for the user's review.

[Open the interactive comparison](http://127.0.0.1:61014/comparison.html).
[Local gallery file](<C:/xampp/htdocs/YEAR 4/Testing/outputs/cctv_dgp_checkpoint_comparison_v1/comparison.html>).

## What was compared

All nine `checkpoints/dgp*.pth` files, the current research DGP checkpoint,
the mapped pretrained starting weights and the separately acquired official
CodeFormer restoration weights. All 12 tensor states and complete output
signatures are distinct. Basic bilinear resizing is a thirteenth control.
Completion and mask detection are bypassed. Historical failed experimental
states remain preserved with their gate failures and are outside this root-checkpoint inventory.

There are 88 fixed development inputs: 24 QMUL-SurvFace crops, 24 ChokePoint
crops (12 source identities, two frames each), and 40 paired synthetic cases
from eight validation references (four per source, all five profiles).
Selection was frozen before inference. Source-specific results remain separate.
No reserved-final pixels were read. No training, gradients, optimizer updates,
app code changes, checkpoint replacement or VM execution occurred.

## Paired development measurements

Means below use **32 degraded synthetic cases**. Clear-change MAE uses eight
additional clear controls; lower is better. These are development results,
with previously exposed identities and incomplete historical/pretraining overlap exclusion.

| Checkpoint | PSNR dB ↑ | SSIM ↑ | Clear change / 255 ↓ | Degraded SSIM above resize |
| --- | ---: | ---: | ---: | ---: |
| Current research DGP | 18.75 | 0.6368 | 5.73 | 27/32 |
| Resize control | 18.23 | 0.5922 | 0.00 | 0/32 |
| Mapped starting weights | 18.22 | 0.5909 | 1.53 | 18/32 |
| dgp_epoch_1.pth | 18.04 | 0.5875 | 2.67 | 6/32 |
| CodeFormer w=1 | 18.03 | 0.5375 | 4.30 | 5/32 |
| Phase 3 final | 16.61 | 0.4990 | 12.14 | 9/32 |
| dgp_improved_epoch_20.pth | 15.97 | 0.4755 | 13.30 | 5/32 |
| dgp_improved_epoch_19.pth | 15.96 | 0.4742 | 13.34 | 4/32 |
| dgp_epoch_5.pth | 15.88 | 0.4637 | 23.34 | 2/32 |
| dgp_transfer_epoch_5.pth | 14.41 | 0.4373 | 23.17 | 0/32 |
| dgp_epoch_17.pth | 14.42 | 0.4354 | 23.37 | 0/32 |
| dgp_improved_epoch_10.pth | 11.44 | 0.2536 | 31.14 | 0/32 |
| dgp_improved_epoch_2.pth | 11.18 | 0.2354 | 27.33 | 0/32 |

The current research checkpoint exceeds resize SSIM on 27/32 degraded cases.
It improves mean PSNR by 0.52 dB over resize and 2.14 dB over Phase 3.
Its 5.73/255 mean clear-image change remains an issue, despite being lower
than Phase 3's 12.14/255. Aggregate scores do not qualify native restoration.

## What needs improvement first

**Visible detail under blur and low resolution is the immediate target.**
Current DGP gains are concentrated in low-light and compound cases.
On the pure-blur profile, mean SSIM is 0.6472 versus resize 0.6497.
The current model still changes clear faces. Better brightness scores alone
do not resolve the structure/preservation problem.

| Profile (eight cases each) | Resize SSIM | Current DGP SSIM |
| --- | ---: | ---: |
| clear | 1.0000 | 0.9142 |
| blur_lr24 | 0.6497 | 0.6472 |
| lowlight_lr32 | 0.5679 | 0.6350 |
| motion_lr48 | 0.7064 | 0.7098 |
| compound_lr24 | 0.4449 | 0.5553 |

The user will choose the next training starting point after reviewing all
models. Retain every original checkpoint. Any later restoration pilot must
follow that choice and have a distinct, evidence-based design with finite limits.
Preserve eyes, brows, nose, mouth, jaw and visible appearance, while keeping
clear-face controls and all existing structure/preservation gates binding.
This comparison does not recompute or pass the historical 1% feature gate,
prepare a training recipe or authorize an unchanged failed optimizer resume.

## Native CCTV visual review

All 22 complete sheets (1,144 model/case thumbnails) and six fixed 256-pixel
focus sheets (78 additional model/case views) were viewed at original resolution.
This is assistant development review. Independent final review is still pending.

QMUL-SurvFace retains seven insufficient, eight weak, eight coarse and one partial
input-only labels; pose restrictions remain binding within those categories.
No tested checkpoint convincingly resolves anatomy in the unreadable crops.
Request a clearer/less-covered crop when required. ChokePoint shows the current
DGP retaining rough features better than the older smoothing checkpoints, but
fine facial edges, hair and wrinkles still soften. Epoch1 and mapped weights
preserve more of the observation while showing limited useful restoration.

CodeFormer often produces sharper plausible faces; paired cases demonstrate
mouth/expression and hair/contour changes. It remains a comparison baseline.
Those observations apply to the declared common-crop implementation; they
do not rank every model configuration or the publisher's complete pipeline.

## Reproducibility and limits

Every arm receives the identical retained RGB 256 PNG, converted with NumPy
float32/float32(255). Native faces use neutral center-padding and bilinear
resizing; there is no per-model alignment, denoising, sharpening or tuning.
The three 352-entry checkpoints strictly load their original 6017d97 decoder;
the archived and adapted forward-function ASTs are exact. Only pretrained
initialization downloads and an unused loss import are adapted. Current
622-entry checkpoints use stored evaluation normalization with disposable
InstanceNorm copies. CodeFormer fidelity is fixed at 1.0, adain enabled,
512 internal pixels returned to 256; no publisher alignment pipeline is used.
Mapped starting weights run in our graph, so this arm is not a full publisher
DeblurGAN benchmark. All parameters and buffers remain unchanged.

Native float RAW, floor-converted PNG with padding restored, and separate app
grayscale-display PNG are retained. No selected input activates the grayscale
policy, so the two PNG views are identical here. Metrics mask observed pixels;
SSIM uses Gaussian windows on support eroded by 11 pixels. Edge MAE is a
diagnostic and is not the historical feature-structure gate. FFHQ canonical
targets use the mixed protocol's explicit canonical hashes, retaining its
older thumbnail hash fields as history. No clean target is invented for CCTV.

QMUL-SurvFace and ChokePoint have no aligned clean reference in this cohort;
native PSNR/SSIM fields are null. Dataset folder names do not establish
ethnicity. No Zamboanga performance or recovered hidden identity is claimed.
Source notices/acquisition provenance are retained in `source_terms/`.

The execution completed 1,056 CPU forwards in 915.74 seconds, within its
finite inference limits. Informal per-model CPU timings vary with local load
and are not a controlled hardware-speed ranking. The separate saved-output
auditor checks 405 input/source/weight bindings and all 3,432 RAW/PNG/display
measurement rows; 11 separately calculated pixel scans crosscheck within 1e-10.

The preview-server image-connection failure is retained in scratch. A larger
listen queue and HTTP/1.1 resolve it. Bundled inline Playwright passes 375,
768 and 1280 layouts with no overflow or page/console errors, eight source/view
scenarios, nine measurement views and three exact PNG/CSV downloads. These
checks verify the report interface; they do not qualify the thesis app/model.

## Files

`outputs/cctv_dgp_checkpoint_comparison_v1/` contains the frozen `plan.json`,
all `results/<checkpoint>/<case>/{raw.npy,native.png,display.png}`,
`metrics.json`, `metrics.csv`, `saved_output_audit.json`,
`visual_sheet_manifest.json`, `visual_review.json` and `comparison.html`.

Current research checkpoint SHA256 (one of the compared models):
`646fbb11674e8882ea41e251d6e6e7d468c963979b48c93132def8292834e24b`.
Root checkpoints, source splits, research caches and failed runs remain retained.
The initial preparation stop is retained separately in
`outputs/cctv_dgp_checkpoint_comparison_v1_prepare_failure/`.
