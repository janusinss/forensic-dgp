# DGP app v3: covering review and completion-context comparison

5 October 2026. The current research app works, but **automatic covering detection
and full covering-family usefulness remain unqualified**. Its trained DGP also
changes/softens visible appearance. Running DGP after completion does not solve
these problems and is **not adopted**. No training or cloud action occurred.

## Fixed development scope

The 36 cases reuse 18 previously inspected photographs and their synthetic
degradations from `outputs/varied_covering_practical_protocol_v1/protocol.json`
(SHA256 `15f6ca20827c4221f7e12630f7d3fab54d7eceae51235eb9401c6f529563382e`).
These are development photographs, **not native CCTV**. Their legacy `_native`
suffix means original photo. Previously exposed training/development sources
cannot establish independent performance. No ethnicity, Zamboanga or local CCTV
performance is inferred.

Both input sheets were inspected before new inference; the new frozen plan
records input-only decisions and 151 source/checkpoint/input/mask bindings.
The difficult three-quarter mask/glasses case is excluded. Nearly hidden hands
require a less-covered/clearer input. Both conditions of each case reject in all
three modes: 12 rejected requests, no unexpected rejection.

The other 32 cases use the existing frozen operator removal footprints with
Off/On/Auto. Automatic proposals are evaluated separately; none was approved for
generation. There is no automatic-generation quality claim and no change to
historical footprints, splits, candidate gates or app selection thresholds.

## Actual inference and independent saved-output checks

| Check | Result |
| --- | --- |
| Current route, finite CPU inference | 108 requests in 428.18s; 600s worker budget/650s outer limit |
| Model forwards | 36 detector, 84 completion, 49 DGP; all three states unchanged |
| Saved-output audit | 6.15s; 151 source and 336 output bindings; 96 exact PNG compositions and 32 exact Auto aliases |
| Preservation/composition | All Off pixels outside removal exact; all On completed pixels equal Off; no post-review mask expansion |
| Training/evaluation limits | Zero optimizer/backward/training calls; no native CCTV or reserved 32 crops used |

The selected main restorer remains our trained identity-v2 DGP, SHA256
`646fbb11674e8882ea41e251d6e6e7d468c963979b48c93132def8292834e24b`.
Separate pretrained CodeFormer inpainting estimates the reviewed hidden area.
Raw float arrays are audited separately from floor-to-PNG, grayscale display and
mask composition. The original Phase 3 and rejected V18/V19/r2 remain preserved.

All six current-route output sheets were inspected: 96 generated outputs and
four rejected cases. The 1608×1764 sheets were displayed at 1504×1649; this is not
an original-resolution claim. Detailed case observations and image hashes are
in `outputs/dgp_app_covering_review_v3/visual_review.json`.

## Automatic proposals versus assisted completion

Mean overlap below is against approximate **operator footprints**, not expert
segmentation truth. Each condition includes 18 cases, including controls and
excluded inputs. Empty counts therefore include appropriate empty controls.

| Proposal | Original-photo mean IoU / empty | Degraded-photo mean IoU / empty | Status |
| --- | --- | --- | --- |
| Current retained app detector | 0.214 / 5 of 18 | 0.191 / 14 of 18 | Substantial misses/false marks; unqualified |
| Cached existing91 candidate | 0.730 / 3 of 18 | 0.746 / 3 of 18 | Cached comparison; historical failures remain |
| Cached varied133 candidate | 0.748 / 3 of 18 | 0.745 / 3 of 18 | Cached comparison; historical failures remain |

Better footprint overlap alone does not qualify a detector. The cached candidates
were not rerun, adopted or excused from previous preservation/usefulness failures.

| Covering or control | Assisted visual result and remaining limitation |
| --- | --- |
| Cloth/pink masks and hand over mask | Main covering removed with plausible lower-face estimates; some boundary remnants, unstable hidden appearance and DGP softening remain |
| Sunglasses and strong glare | Reviewed regions largely replaced; stylized eye estimates or rectangular joins; strong-glare frame mostly retained, with no hidden accuracy reference |
| Hands and obstructing hair | Hand-eye case retains fingers/poor joins; hair case has inconsistent estimated versus visible gaze; DGP softens visible eyes |
| Scarves and other objects | Some plausible central estimates; scarf/glove texture remains around chin/cheek; leaf hand/edge remains outside footprint; distinguish clothing outside the face from failed removal |
| Uncovered/clear-glasses controls and exclusions | Empty-mask Off preserves all 65,536 pixels; On changes visible appearance. Three-quarter/near-hidden cases reject through operator policy, not a calibrated classifier |

**Protected-mask clarification:** every generated case with a protected-region
file has an empty protected mask. Only the excluded three-quarter case has 60
protected pixels. Zero changed-pixel records in the original audit are vacuous
and do not prove DGP On eyewear/hair preservation. The original receipt is intact;
`protected_support_clarification.json` records support counts. Off's full visible
pixel checks and the four whole-image controls are nonvacuous preservation evidence.

## Paired synthetic evidence and fixed context comparison

The 16 supported degraded cases have their original **covered** photo as the
paired reference. Metrics evaluate known pixels outside the frozen removal
footprint; background or unmarked covering can remain. SSIM uses valid 7×7 windows
outside the footprint and border. There is no aligned clean hidden-face reference,
hidden-region PSNR/SSIM, native paired metric or fresh identity score.

On worsens mean nonremoved pixel error versus Off in **all 11 reported family
groups**. SSIM improves in three groups (hands, hand over mask, hair) and regresses
in eight. Auto selects On for all 16 supported degraded cases; among original
photos it selects Off for 15 and On for the strong-glare case. Exact alias checks
establish deterministic selection, not beneficial selection.

A separate fixed, unfitted ablation feeds the **saved rendered Off completion**
to the same DGP, then retains identical completed pixels. This tests whether
unremoved covering texture in the DGP input explains visible degradation.
Its pre-inference plan caps 32 forwards/180s and changes no app code or thresholds.
It finishes in **17.07s**, with unchanged DGP state and zero detector/completion/
training calls. A separate **4.81s** saved-array audit verifies 32 exact PNG
compositions, all recorded synthetic metrics and four exact empty-mask raw controls.

Compared with current covered-input On, completed context improves nonremoved MSE
in 9/16 cases and SSIM in 10/16. Compared with Off, only **1/16 MSE and 3/16 SSIM**
improve. All eight 1072×1192 comparison sheets were inspected at original resolution.
They show similar softness and no convincing structural gain; the unchanged
completion preserves existing gaze, finger and scarf defects. **Do not adopt or
refit this variant on the exposed cohort.**

## Browser flow and disposition

Bundled inline Playwright uses the verified running URL
`http://127.0.0.1:8000/` and saves only local `scratch/` artifacts. In **30.25s** it
passes 375/768/1280 overflow/error checks; real-model glare, hands and hair flows;
mask-import review invalidation; one 256 output; exact saved inference PNGs; PNG
and bundle downloads; and operator clearer-input rejection without generation.
There are zero page/console errors. The saved-download audit independently
matches three PNGs and the bundle's input, masks, raw DGP and result.
Earlier 34 regressions remain the verified unchanged app baseline; they were not
rerun solely for these analysis scripts/docs. All 22 previous integration record
bindings are checked again when this milestone is recorded.

The browser proves functional behavior, not useful restoration or family-wide
completion. Development visual review by the implementing assistant is not an
independent final assessment. No reserved/final outputs were exposed.

Keep the full goal active. Current native core outputs remain soft without useful
structural improvement. The next model decision must address demonstrated visible
appearance/structure loss and completion boundary/gaze failures; this comparison
does not justify scaling or repeating a failed recipe. No new VM pilot is launched
or prepared here. Training remains L4 VM only, with verified transfers and exact
pasteable commands provided for user execution when a justified pilot is ready.

Evidence roots: `outputs/dgp_app_covering_review_v3/`,
`outputs/dgp_app_completion_context_v3/`; browser/download receipts
`scratch/dgp-v3-family-flow.json`, `scratch/dgp-v3-family-download-audit.json`.
