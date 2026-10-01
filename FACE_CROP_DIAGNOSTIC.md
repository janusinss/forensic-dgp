# Fixed face-size diagnostic — 2 October 2026

The reflection return verifies procedural learning but no real validation glare
recovery and failed synthetic retention. This experiment tests spatial scale
without new fitting. It does not change the application or original quality gates.

Windows root: `C:\xampp\htdocs\YEAR 4\Testing\`.
VM repository: `~/forensic-dgp/`; completed VM return source:
`~/forensic-dgp/coverage_vm_bundle/outputs/reflection_coverage_vm/`.
This is a local CPU diagnostic; no new VM job or optimizer is created.

## Fixed membership and model state

Use all 73 reviewed real training cases (47 covered, 26 clear) and all 280
immutable reflection training fixtures (224 covered, 56 clear). Exactly two real
training reflections receive lens-only counts with V2 mouth-mask pixels removed.
Validation/test images and the 400-case benchmark receive no forward passes.

The two frozen states are audited source 30 and final reflective 42. The latter
is the predeclared advancement model; source 30 only diagnoses ancestry. Do not
choose an epoch or source model based on this diagnostic. Source 30 SHA256 is
`9ba74a30719f01bfafd7ee9c060dc090c507c26eeeecb117b4cb342a8bf4df42`;
reflective 42 is
`a51f20e8fa12cee7dec574195debc5ada9b8cfeeb289b12bad11a6d39a9acf25`.
Original training masks are reused only after their exact CPU reproduction and
file/input/model hash bindings verify. New inference must preserve every tensor.

## One input-only transform

Use the existing SCRFD face-box model `det_10g.onnx`, SHA256
`5838f7fe053675b1c7a08b633df49e7af5495cee0493c7dcf6697200b85b5b91`,
at 256 input, CPU only, confidence 0.6. Exactly one finite reliable face box is
required, its center inside the input and shorter dimension at least 16 pixels.
No landmark, target mask, crop-score search or identity reference chooses a box.

Square side is `ceil(1.25 * max(face_width, face_height))`, centered on the box,
at least 32 and strictly smaller than the longest input dimension. The affine
maps pixel centers uniformly to 256 pixels. RGB is bilinear; padding is 96.
Missing/multiple/unreliable/no-zoom detections preserve the original mask and
perform no additional forward pass. Face detector failure is not a clear label.

Threshold remains 0.5. Crop masks map back with nearest-neighbor interpolation;
retain the original predictions outside the crop ROI. Targets map only after
crop selection for diagnostic visualization/counts. Whole-image real metrics and
original valid-support fixture metrics use the mapped predictions, not cropped
targets. Neither discarded labels nor padding can artificially improve scores.

## Predeclared training signal and stop rule

Reflective 42 must improve the two-case lens-only recovered-pixel total, retain
whole real-training IoU and fixture IoU, and not increase real visible FP or
real/fixture clear-case errors. Every condition must hold. These are diagnostic
criteria, not replacement quality gates or an eligible checkpoint claim.

If this fixed rule fails, stop this crop experiment without another margin,
confidence, threshold or epoch sweep. If it passes and crop geometry is reviewed,
one separately declared evaluation on the unchanged original validation/benchmark
may test deployment eligibility. No held-out crop fitting or generator training.
The full goal still requires original safeguards and reviewed end-to-end output.

## Execution and preserved evidence

Tests: eight geometry/inference counterexamples plus three membership/signal
contracts. Protocol preparation freezes code, data and model hashes before any
crop outputs. The runner refuses existing protocol/result directories.

```powershell
Set-Location 'C:\xampp\htdocs\YEAR 4\Testing'
venv/Scripts/python.exe -m unittest tests.test_face_crop_diagnostic tests.test_face_crop_protocol
venv/Scripts/python.exe scripts/run_face_crop_diagnostic.py --prepare
venv/Scripts/python.exe scripts/run_face_crop_diagnostic.py
```

Local protocol: `C:\xampp\htdocs\YEAR 4\Testing\outputs\face_crop_protocol_v1\protocol.json`.
Local diagnostic: `C:\xampp\htdocs\YEAR 4\Testing\outputs\face_crop_diagnostic_v1\`.
Existing local face-box model:
`C:\Users\janus\.insightface\models\buffalo_l\det_10g.onnx`.
No corresponding VM upload is needed for this CPU experiment. Hidden facial
features remain plausible estimates; crop interpolation adds no real detail.
