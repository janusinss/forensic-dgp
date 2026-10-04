# V10: face-prior feasibility before new training

Status,4 October2026: all case images/embeddings were saved, but the original
inference run failed while rendering its first preview grid. Pillow's paste
received a NumPy array instead of an Image. The frozen source/run/failure are
preserved. A separately pinned postprocessing recovery reconstructs reviewable
grids and metrics from those saved bytes with **zero neural forwards**; the
original run is not restarted or marked successful. Its missing final model-state
and forward-count receipt cannot be recovered from PNGs. Three meaningful grid
regressions now pass, including exact preservation of every original cell.

This does not train a new prior adapter, select a checkpoint or change the main
application. All training remains on the L4, stopped after the V9 return.
Original native24/24 was logged at765seconds, within the1,200-second cap;
this is progress-log timing, not a completed original terminal receipt.
Recovery plan SHA256
`4302708d31b8e466f5daac6ab71bfabe197b7d606ae369f77d7582dcef629d15`.
Recovered review: `C:\xampp\htdocs\YEAR 4\Testing\outputs\cctv_dgp_face_prior_recovery_v10_r2\review\`
↔ intended `~/forensic-dgp/outputs/cctv_dgp_face_prior_recovery_v10_r2/review/`
after explicit report sync, not claimed transferred. The independent recovery audit and all ten original-cell visual reviews are complete. The cascade is rejected as an adopted upgrade; see `CCTV_DGP_FACE_PRIOR_RESULTS_V10.md` beneath the Windows root.

The user authorizes a pretrained face-generating prior with our own trained
conditioning layers if it produces better reviewed output. V9's improved pixel
error with appearance regressions motivates a controlled feasibility comparison,
not another unchanged pixel-training run. A frozen CodeFormer chain is an
explicit pretrained comparison, not a renamed new DGP model.

## Frozen comparison

Ten existing validation references, five per source, selected by ascending
SHA256 of `face-prior-v10:reference_id`; all five unchanged profiles per reference
give **50 paired photographic-proxy cases**. No metric filtering or role changes.
All24 existing native CCTV development crops are retained. The input-reviewed
six frontal/mild coarse-structure cases form a separate core review; seven
insufficient cases require a clearer crop. Reserved32 images are not forwarded
or rendered. Native CCTV has no paired clean truth or verified identity accuracy.

| Paired control / arm | Meaning |
| --- | --- |
| Exact input | Existing quantized256 pixels |
| Cached V2 | Retained baseline PNG from audited V9 |
| Cached V9 epoch20 | Failed snapshot, diagnostic preprocessing only |
| CodeFormer(input), w1 | Official frozen restoration baseline |
| CodeFormer(V9 PNG), w1 | Diagnostic prior with DGP signal preprocessing |

Native controls reuse audited Phase3 and CodeFormer raw floats, composited on
the same observed support. New native arms are V9 epoch20 diagnostic output and
CodeFormer of its delivered PNG. A failed checkpoint is not production `best.pth`.

Exact RGB256 input geometry remains unchanged. Native RGB is center-padded with
RGB128 before Pillow bilinear256; masks use nearest. No extra alignment, denoise,
CLAHE, sharpening or target-dependent model conditioning. CodeFormer internally
uses bilinear512 normalized[-1,1], `w=1`, `adain=True`, then bilinear256.
The chain receives floor-quantized DGP output with original padding restored.
Raw floats and exported PNG/composite are saved separately.

Official checkpoint SHA256:
`1009e537e0c2a07d4cabce6355f53cb66767cd4b4297ec7a4a64ca4b8a5684b7`.
Source revision `b33cc7d639d6545bfcccc7e0bc6ae51f24e79c2b`;
retained S-Lab License1.0 and NOTICE. Authors describe the quality/fidelity tradeoff
and cropped/aligned-face usage. This controlled unaligned CCTV comparison is
not a reproduction of their aligned benchmark protocol.
[Official CodeFormer repository](https://github.com/sczhou/CodeFormer).

## Budget and review decision

CPU, four threads, **124 CodeFormer forwards,24 DGP forwards,260 recognizer
forwards**, zero backward calls/optimizer updates. Loading is separate; execution
cap is1,200seconds. At the first five paired cases, measured time is62.71seconds
and projected execution867.61seconds, within the cap. Wrong bytes, nonfinite
outputs, changed states, excessive timing or wrong counts stop the run. Retain
partial/failure evidence; no automatic resume or repeated execution.

Inspect five original-cell 10-reference profile grids and all24 native outputs.
Preserve contour, apparent feature layout and expression before sharpness; reject
generic added anatomy and strong color/artifact changes. Report paired source/profile
PNG MSE/SSIM and fixed-affine ArcFace similarity separately. Native input change
is not ground-truth error. Feasibility requires coherent detail improvement on
reviewed eligible inputs; no automatic model selection or application promotion.
An independent arithmetic audit and assistant visual review follow execution.
Final independent human review remains outstanding even if feasibility improves.

## Paths, evidence and execution

Windows root: `C:\xampp\htdocs\YEAR 4\Testing\`; VM root: `~/forensic-dgp/`.
This local inference experiment is not claimed transferred or executed on the VM.

| Artifact | Windows | Intended VM counterpart after explicit sync |
| --- | --- | --- |
| This runbook | `C:\xampp\htdocs\YEAR 4\Testing\CCTV_DGP_FACE_PRIOR_V10.md` | `~/forensic-dgp/CCTV_DGP_FACE_PRIOR_V10.md` |
| Frozen comparison | `C:\xampp\htdocs\YEAR 4\Testing\outputs\cctv_dgp_face_prior_feasibility_v10\frozen_plan.json` | `~/forensic-dgp/outputs/cctv_dgp_face_prior_feasibility_v10/frozen_plan.json` |
| Output/evidence | `C:\xampp\htdocs\YEAR 4\Testing\outputs\cctv_dgp_face_prior_feasibility_v10\run\` | `~/forensic-dgp/outputs/cctv_dgp_face_prior_feasibility_v10/run/` |
| Runner | `C:\xampp\htdocs\YEAR 4\Testing\scripts\compare_cctv_dgp_face_prior_v10.py` | `~/forensic-dgp/scripts/compare_cctv_dgp_face_prior_v10.py` |
| Auditor | `C:\xampp\htdocs\YEAR 4\Testing\scripts\audit_cctv_dgp_face_prior_v10.py` | `~/forensic-dgp/scripts/audit_cctv_dgp_face_prior_v10.py` |

Plan SHA256 `8393e67abeec1e21428fb56c97e7d6be8d3657a38bd1767fd89e127ce805b250`;
297 pinned source/weight/control/input/cache files. Five meaningful role/path/
quantization/fingerprint boundary tests pass. Their first two attempts failed
because Python3.13 mode700 temp folders denied restricted-token access; the
test harness now uses inherited-access project scratch directories. Original
failed test receipts remain preserved; no experiment outputs existed then.

These commands document the original failed local experiment; do not repeat them. They are not a
request to repeat it:

```powershell
Set-Location -LiteralPath 'C:\xampp\htdocs\YEAR 4\Testing'
.\venv\Scripts\python.exe -X utf8 -u scripts/compare_cctv_dgp_face_prior_v10.py --prepare
.\venv\Scripts\python.exe -X utf8 -u scripts/compare_cctv_dgp_face_prior_v10.py --run
.\venv\Scripts\python.exe -X utf8 -u scripts/audit_cctv_dgp_face_prior_v10.py
```

Next: complete the fixed comparison/audit/visual inspection. Choose a distinct
finite conditioning pilot only if the outputs justify it; otherwise diagnose
alignment, fidelity or representation limits before more training. Historical
V1–V9 protocols, failed guards, reserved data and app defaults remain intact.
