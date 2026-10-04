# Local face workflow — status updated 3 October 2026

## DGP-first integration preparation

The current combined app still uses CodeFormer for visible restoration. A strict
256×256 DGP adapter and post-pilot native review plan are now prepared in
`C:\xampp\htdocs\YEAR 4\Testing\DGP_INFERENCE_READINESS.md`
↔ `~/forensic-dgp/DGP_INFERENCE_READINESS.md` after source/document transfer.
Seven adapter checks and four candidate-selection checks pass; eight native DGP
forwards exactly reproduce the audited Phase3 baseline without state changes or
training. This does not establish improved output or select an app default.

The useful-candidate review, 256 output/composition policy, qualification of tiny
or severely degraded crops and DGP-led browser verification remain pending.
The existing decoder's32-pixel minimum and original-size/50% blend behavior below
describe the current historical route, not the final CCTV contract. UI and live
route behavior remain unchanged during this inference-preparation milestone.

## Current combined route

The existing main application now provides upload, automatic removal-area preview,
paint/erase correction, one face estimate and PNG/ZIP downloads. Hidden facial
features are plausible estimates. Automatic detection uses the retained development
baseline and needs review; corrected-mask results must not be reported as automatic
detector accuracy. The broad-family review is complete: assisted hand, hair, scarf
and object estimates are useful in several inspected cases; one scarf remains
partial and automatic proposals miss most native coverings. See
`PRACTICAL_BROAD_OUTPUT_RESULTS.md` under the Windows/VM repository roots below.

A separate four-case assisted context comparison now reduces scarf texture in one
case, with an unresolved boundary line and no established general-default benefit.
It is audited and documented in `COMPLETION_CONTEXT_RESULTS.md` under the same
Windows/VM repository roots. The main app still uses the existing route; the
grayscale detector VM return and full practical evaluation remain pending.

| Component | Windows local | Linux VM repository after transfer |
| --- | --- | --- |
| Repository | `C:\xampp\htdocs\YEAR 4\Testing\` | `~/forensic-dgp/` |
| Main app | `C:\xampp\htdocs\YEAR 4\Testing\app.py` | `~/forensic-dgp/app.py` |
| Inference policy | `C:\xampp\htdocs\YEAR 4\Testing\face_workflow.py` | `~/forensic-dgp/face_workflow.py` |
| Active palette extension | `C:\xampp\htdocs\YEAR 4\Testing\face_workflow_palette.py` | `~/forensic-dgp/face_workflow_palette.py` |
| Input color policy | `C:\xampp\htdocs\YEAR 4\Testing\face_color_policy.py` | `~/forensic-dgp/face_color_policy.py` |
| API | `C:\xampp\htdocs\YEAR 4\Testing\face_workflow_web.py` | `~/forensic-dgp/face_workflow_web.py` |
| UI | `C:\xampp\htdocs\YEAR 4\Testing\templates\face_workflow.html` | `~/forensic-dgp/templates/face_workflow.html` |

## Start in the existing local environment

```powershell
Set-Location 'C:\xampp\htdocs\YEAR 4\Testing'
.\venv\Scripts\python.exe -m uvicorn app:app --host 127.0.0.1 --port 8000
```

Open `http://127.0.0.1:8000`. The tested project environment currently uses CPU
PyTorch, despite the computer having NVIDIA hardware. The UI displays the actual
inference device. First use loads the separately pinned models; the initial
256-pixel mask completion browser request took 7.8 seconds including lazy loading.
This is one measured request, not a latency guarantee. No local training occurs.

The historical `POST /reconstruct` DGP benchmark and original template/script
are retained. The main `/` page serves the new workflow in the same terminal
layout and palette. It does not display fabricated identity confidence, FAN-based
identity verification, converged-epoch claims or a sub-100ms processing promise.

## Required model files

| Purpose / environment override | Windows default | Linux counterpart after transfer |
| --- | --- | --- |
| Retained detector / `FACE_DETECTOR_CHECKPOINT` | `C:\xampp\htdocs\YEAR 4\Testing\outputs\downloaded_completion\outputs\completion_pilot\epoch_2.pth` | `~/forensic-dgp/outputs/downloaded_completion/outputs/completion_pilot/epoch_2.pth` |
| Completion / `FACE_COMPLETION_CHECKPOINT` | `C:\xampp\htdocs\YEAR 4\Testing\checkpoints\codeformer_inpainting.pth` | `~/forensic-dgp/checkpoints/codeformer_inpainting.pth` |
| Visible restoration / `FACE_RESTORATION_CHECKPOINT` | `C:\xampp\htdocs\YEAR 4\Testing\outputs\codeformer_restoration_pretrained_v1\codeformer.pth` | `~/forensic-dgp/outputs/codeformer_restoration_pretrained_v1/codeformer.pth` |
| Historical Phase 3 DGP | `C:\xampp\htdocs\YEAR 4\Testing\checkpoints\dgp_zamboanga_final.pth` | `~/forensic-dgp/checkpoints/dgp_zamboanga_final.pth` |

All overrides select a path, not an unreviewed model. Restricted loading and
fingerprint checks reject missing/incompatible checkpoints; no random generator
fallback is supplied. The detector accepts only the retained baseline fingerprint
until a separately reviewed candidate is explicitly selected in code. Models are
lazy, frozen and serialized for inference. Checkpoints remain Git-ignored where
already ignored; `git pull` alone does not transfer these files.

| File | Expected SHA256 |
| --- | --- |
| Detector | `c2d4cd180226413af63bcfc2a3e17461cfa95f87aff9a8998b893fc3afc42e93` |
| Inpainting | `b0d1b868c3dacf75d637bbcc0d4b3dd4ce2a064a338dfb4d35c740d3b4ae8797` |
| Restoration | `1009e537e0c2a07d4cabce6355f53cb66767cd4b4297ec7a4a64ca4b8a5684b7` |

CodeFormer source is pinned at `b33cc7d639d6545bfcccc7e0bc6ae51f24e79c2b`;
S-Lab License 1.0 is retained in `third_party/codeformer/LICENSE` under either
repository root. The restoration fingerprint was observed at acquisition, not
published as a checksum. Provenance is retained with the experimental outputs.

## Processing and review

1. Upload one PNG/JPEG crop, 32–4096 pixels per side and no more than 16 million
   pixels. The backend applies EXIF orientation. Non-square crops receive neutral
   padding for inference, then return at the original dimensions.
2. Review the green proposal. The detector threshold is 0.5, with a 3-pixel margin
   at 256 scale. Paint/erase, undo, clear, reset or import a matching mask. Clear
   glasses and hair outside facial features should remain outside the removal area.
3. Choose Auto/Off/On visible restoration and confirm the removal area was reviewed.
   The editor uses at most 512 pixels on its longest side and exports a binary mask
   at the EXIF-oriented original dimensions using nearest-neighbor scaling.
4. Generate one estimate. The submitted mask gets no additional dilation. Completion
   changes only that area. Restoration uses the separate face-restoration model at
   fidelity 1.0, with a 50% visible-region blend after completion. Generated-region
   pixels are preserved through restoration. PNG quantization occurs at export.
   Active policy `reviewed-face-workflow-v2` also keeps near-grayscale inputs gray:
   only generated pixels are projected when Off; restored visible pixels are also
   projected when restoration is applied. Color inputs remain unchanged by this
   extension. Selection uses only visible uploaded RGB with a development threshold.
5. Inspect original, binary removal mask and estimate. Download the final PNG or
   ZIP containing `original.png`, `removal-mask.png`, `estimate.png`, `processing.json`
   and `README.txt`. The original is decoded/oriented, not the original JPEG bytes.

Auto uses only uploaded visible pixels: smoothed 256-scale Laplacian variance below
24 or a robust noise estimate of at least 8/255 on low-gradient visible samples.
Mask boundaries are excluded from the signal. These are developmental thresholds,
not a universal calibrated quality classifier; use the override when necessary.
The earlier comparative report blends saved PNG intermediates, while this runtime
blends model floats before final PNG quantization. Its full workflow is separately
checked; the report's 23.2% gain describes that fixed benchmark, not every upload.

The visibility check uses approximate frontal-crop geometry: an ellipse and central
eye/nose/mouth bands. It rejects a mask covering at least 80% of that ellipse, nearly
all four central bands or 85% of the whole crop, before generator loading/inference.
It is conditional on a correctly reviewed mask, not a reliable face landmark or
occluder detector. If detection misses a covering, manual review must supply it;
if little face is visible, use a less-covered image. This limitation is explicit.

## Verified so far

Fifteen inference/adapter/palette unit checks pass: reviewed support, post-restoration hole
preservation, non-square empty bypass, pre-inference visibility rejection, invalid
mask/output rejection, input-only blur/noise signals and orientation/bundle handling.
Inline bundled Playwright verifies upload, mask import, keyboard paint/erase/undo,
review gating, Auto/Off/On, one output, PNG/ZIP download and nearly hidden rejection.
No accidental horizontal overflow or unexpected page/console errors occurs at
375/768/1280 pixels before and after generation. The rejection produces the expected
HTTP 422, separately recorded from unexpected errors.

Downloaded ZIP pixels are independently checked: reviewed mask matches exactly
and zero pixels change outside it with restoration off. The clear-glasses Auto
download has zero observed-pixel changes; forced On changes visible pixels as
advertised. Evidence is local git-ignored `scratch/`; intended receiving counterpart
is `~/forensic-dgp/scratch/` if explicitly transferred, not through Git.

The broad comparison independently verifies 28 outputs and 32 detection masks.
All seven assisted degraded cases improve known-visible MAE; the active palette
route reduces their mean by 19.55%. This is exposed development evidence, not
hidden-identity accuracy. Both reviewed nearly hidden cases reject; automatic
proposals miss both, so source review holds their outputs separately. Twenty-eight
cached palette compositions and one real browser result also pass independent
verification. Grayscale browser pixels match the cached V2 result exactly.

The subsequent 36-case XSeg and returned direct-detector comparisons are complete.
XSeg marks ordinary clear glasses/background and wrongly rejects a usable scarf;
the direct detector misses obstructing hair and loses coverage under degradation.
Neither candidate is selected by this app. See `PRACTICAL_XSEG_RESULTS.md` and
`PRACTICAL_DIRECT_DETECTOR_RESULTS.md` under the repository roots above.

The three-arm VM camera/source diagnostic is complete and audited. Camera
degradation improves practical masks/glare/hands, but retention/clear checks fail,
hair stays undetected and both nearly hidden cases miss automatic rejection.
Four follow-up estimates still contain missed cloth/finger remnants. Report:
`C:\xampp\htdocs\YEAR 4\Testing\REAL_CAMERA_RESULTS.md` ↔
`~/forensic-dgp/REAL_CAMERA_RESULTS.md` after a future transfer.

The subsequent varied-covering V2 VM comparison and36-case local practical mask
review are now audited. New data improves exposed family fit and some hand/scarf
transfer, but old/reflection retention fails, gallery hair remains undetected and
nearly hidden automatic rejection still fails. Both returned models preserve the
four practical clear controls; the retained app-default cached mask has a native
uncovered-control error. The new 20-estimate downstream comparison is complete,
independently reconstructed and visually reviewed. Central coverings are often
replaced, but hair remains, covering edges persist and scarf/flower estimates
contain conspicuous artifacts. These automatic outputs do not establish full-scope
readiness; optional manual correction remains a separate assisted path. See
`C:\xampp\htdocs\YEAR 4\Testing\VARIED_COVERING_RESULTS.md` ↔
`~/forensic-dgp/VARIED_COVERING_RESULTS.md` after future transfer.

A subsequent saved-pixel/cached-output diagnosis confirms that reviewed footprints
improve hair/hand/scarf/object completion and trigger near-hidden rejection on the
fixed examples. The automatic detector remains the main observed limitation;
generated scarf texture and stylized anatomy also remain imperfect.

Next: run the separately prepared matched RGB/grayscale detector pilot on the VM
after preflight. Commands: `C:\xampp\htdocs\YEAR 4\Testing\GRAY_COVERING_VM.md` ↔
`~/forensic-dgp/gray_covering_vm_bundle/GRAY_COVERING_VM.md` after extraction.
No new app checkpoint is qualified; the current local workflow and pinned models
remain unchanged. Goal active; no local training or actual new CUDA execution.
