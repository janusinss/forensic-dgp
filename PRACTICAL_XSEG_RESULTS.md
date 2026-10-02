# Pretrained XSeg covering proposals — 2 October 2026

XSeg1 finds substantially more masks, hands and objects than the retained detector,
but this conversion also marks background, ordinary hair and clear glasses. It
wrongly rejects a usable scarf source. It is **not selected by the main app**.
This is a segmentation-only comparison; no completion-quality or training gain
is claimed.

Windows root: `C:\xampp\htdocs\YEAR 4\Testing\`; intended Linux counterpart after
transfer: `~/forensic-dgp/`. Model/provenance are under Windows
`C:\xampp\htdocs\YEAR 4\Testing\outputs\xseg_pretrained_v1\` (Linux
`~/forensic-dgp/outputs/xseg_pretrained_v1/`). Acquisition is local only.

## Official model and contract

The official FaceFusion asset is XSeg1, 70,324,286 bytes. Published CRC32
`f207afe3` matches; observed SHA256 is
`c4d1498b8a03b5fe2a3a5d2ef2a0402ab03bd51edaf5b2d8d5fb764702a97dd3`.
FaceFusion's model metadata attributes it to DeepFaceLab, year 2021, GPL-3.0.
Official source snapshots and the DeepFaceLab GPL-3.0 text are retained beside
`acquisition.json`; these are observed master snapshots, without an invented
commit identity. [Official model metadata and consumer](https://raw.githubusercontent.com/facefusion/facefusion/master/facefusion/face_masker.py),
[official hash implementation](https://raw.githubusercontent.com/facefusion/facefusion/master/facefusion/hash_helper.py),
[DeepFaceLab license](https://github.com/iperov/DeepFaceLab/blob/master/LICENSE)

ONNX Runtime 1.29.0 verifies NHWC float32 BGR/255 input `[batch,256,256,3]` and
output `[batch,256,256,1]`. The output represents **visible face**, not a covering
class. Our independent research adapter does not import FaceFusion code. Before
the first forward, freeze a conversion: visible probability below 0.5 inside a
fixed frontal-crop ellipse centered `(127.5,137)`, radii `(83,104)`, then 3-pixel
dilation constrained to that ellipse. No target-dependent routing, smoothing,
threshold sweep or label changes occur. This conversion is a development
hypothesis; its inferred contour is not a measured landmark hull.

## Execution and numerical repair

The first run saves 23 rows, then stops on a finite probability
`1.0000001192092896` (one float32 ULP above 1). V1 protocol/results remain intact.
One diagnostic forward records the overshoot. V2 clips finite probabilities with
at most `1e-6` saturation error, consistent with the official consumer's `[0,1]`
clipping. Spatial policy, cases and selection criteria stay unchanged.

V2 completes all 36 cases using 24 cached probabilities plus 12 new forwards.
All 36 baseline proposals are verified caches, so no extra baseline call occurs
in V2. V1 used 24 XSeg and 20 retained-detector forwards; the one diagnostic is
separate. Totals across both versions/diagnostic: **37 XSeg +20 detector forwards**,
zero completion/restoration forwards and zero optimizer updates. V2 runs in
4.67 CPU seconds excluding model loading; V1 took 18.60 seconds. Neither run
trains or modifies a checkpoint.

The fixed gallery is 36 native/degraded cases from 18 previously inspected sources.
It includes empty uncovered/clear-glasses controls and two nearly hidden inputs.
Two three-quarter inputs remain difficult diagnostics, not first-version pose
coverage. Scoring uses fixed operator proposals, including V3 footprint corrections;
there is no hidden-face ground truth or pristine population holdout.

| Evidence | Windows local | Linux after transfer | SHA256 |
| --- | --- | --- | --- |
| V2 protocol | `C:\xampp\htdocs\YEAR 4\Testing\outputs\xseg_mask_comparison_v2\frozen_protocol.json` | `~/forensic-dgp/outputs/xseg_mask_comparison_v2/frozen_protocol.json` | `9fd70d1edef468e6fb6fe951e3080290e9e63bef0fc65c5fffb99a17fadb9516` |
| V2 results | `C:\xampp\htdocs\YEAR 4\Testing\outputs\xseg_mask_comparison_v2\results.json` | `~/forensic-dgp/outputs/xseg_mask_comparison_v2/results.json` | `f844ec2740352f1ec5ce15fb32b06bc280fe72e6f41fea8d6b7e51f53da518df` |
| Independent audit | `C:\xampp\htdocs\YEAR 4\Testing\outputs\xseg_mask_comparison_v2\independent_verification.json` | `~/forensic-dgp/outputs/xseg_mask_comparison_v2/independent_verification.json` | `580c05c0116c9790dd16a85f424626ca4b351e472b8d6613f851cf3ee96ab344` |

The independent audit verifies all 36 probabilities, 108 binary masks, frozen
assets, cached equality, threshold/dilation/domain conversion, reference metrics,
aggregates and conditional visibility checks. New forward counts come from
execution records, not an independent replay. All six preview sheets (36 rows)
were inspected. Existing immutable reports are not rewritten by this comparison.

## Result and decision

| Native group | Sources | Retained reference recall | XSeg reference recall |
| --- | ---: | ---: | ---: |
| Mouth masks, including one difficult turn | 3 | 31.45% | 75.67% |
| Sunglasses | 2 | 0% | 79.02% |
| Hands | 2 | 15.27% | 80.34% |
| Obstructing hair | 1 | 0% | 40.57% |
| Other objects | 2 | 1.64% | 74.71% |

Recall improvements do not establish useful removal masks. The fixed ellipse
creates a ring outside smaller faces and includes hairstyle/background. XSeg
also leaves a partial mirror-lens hole and much of the obstructing hair. Native
scarf mean reference IoU decreases from 0.57023 to 0.52883 despite higher recall.
The masked three-quarter clear-glasses source is poor and outside the initial
demonstrated pose scope.

Empty controls expose scope violations: ordinary clear glasses receive 10,918
marked pixels native and 10,227 degraded (16.66%/15.61% of the crop). The uncovered
control receives 5,088/4,653 pixels (7.76%/7.10%). The retained native uncovered
control also has false detections (11,683 pixels); that existing defect is reported,
not hidden. Both reviewed-control masks are empty.

XSeg correctly triggers the nearly hidden rejection for both hand-covered inputs,
but falsely triggers it for the usable knit-scarf native/degraded pair. The main
app therefore keeps its current review/correction flow. Do not invert raw XSeg
probabilities into new training labels or call this model a scope-specific detector.

Follow-up status, 3 October: the returned direct-occlusion head comparison is now
complete and independently audited on 36 native/degraded cases. Its covering
target avoids XSeg's raw complement, but camera degradation fragments coverage,
obstructing hair remains empty and both nearly hidden cases miss rejection.
Its historical synthetic-retention failure remains unchanged. Report:
`C:\xampp\htdocs\YEAR 4\Testing\PRACTICAL_DIRECT_DETECTOR_RESULTS.md` ↔
`~/forensic-dgp/PRACTICAL_DIRECT_DETECTOR_RESULTS.md` after transfer.

Next: run the separately frozen three-arm camera/source diagnostic in
`C:\xampp\htdocs\YEAR 4\Testing\REAL_CAMERA_VM.md` (isolated VM counterpart
`~/forensic-dgp/real_camera_vm_bundle/REAL_CAMERA_VM.md`). Package/input audits
are complete; CUDA preflight is pending. All actual training remains on the L4
VM. XSeg stays unselected and its frozen protocols, outputs and failures remain.
