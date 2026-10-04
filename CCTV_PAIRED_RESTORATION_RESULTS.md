# Paired CCTV-style restoration regression — 3 October 2026

The retained DGP does not establish improvement across this fixed regression.
It smooths recoverable features and darkens low-light cases. CodeFormer produces
more apparent detail but also changes expressions, facial structure and, in one
case, introduces a hand-like object absent from the reference. Neither result
justifies a new app default or checkpoint promotion. These findings support a
bounded DGP training diagnostic with structure and clear-input safeguards.

The app Goal is active with the revised DGP-first objective. No local training,
optimizer updates, application changes, Git push or VM run occurred here.

## Fixed protocol and inherited split

The **original** Phase 4 split is available at
`C:\xampp\htdocs\YEAR 4\Testing\outputs\downloaded_phase4\outputs\phase4_with_progress\split.json`.
Its original VM path is `~/forensic-dgp/outputs/phase4_with_progress/split.json`.
SHA256: `5c71bc358a351d50e3c0a7d76abe749cb4412aca80d2c7eb4de5fb33dbd29071`.
It contains 76,000 training and 4,000 validation paths: 3,500 FFHQ-thumbnail and
500 Asian-source validation images. The path partition is exact after separator
normalization. Reuse is a historical development regression, not proof of an
untouched final identity test or non-overlap with pretrained models.

SHA ordering selects four references per source before outputs. Five fixed
profiles produce 40 cases: clear, blur/LR24, low-light/LR32, motion/LR48 and
compound low-light/LR24. Native RGB is center-padded with gray 128, then bilinear
resized to 256 without stretching portraits. Stress affects the observed rectangle;
padding outside it remains unchanged. The profile's short dimension follows
source aspect; LR24 denotes the **long edge**, not a guaranteed 24×24 face.

The reference photographs themselves are processed images. FFHQ examples are
128×128 thumbnails; Asian-source originals range from 119×166 to 168×211.
Enlargement adds no captured detail. Some Asian references are dim/grainy before
the synthetic stress. An input-only review fixes six uncovered frontal/mild
references (`ref_01`, `ref_02`, `ref_04`, `ref_05`, `ref_06`, `ref_08`) before
restoration outputs. The sunglasses reference (`ref_03`) and strong profile
(`ref_07`) remain in all-case diagnostics. A covered photograph is not an
uncovered-face reference, and the profile is outside the initial app scope.

Both frozen models receive identical RGB256 inputs. Phase 3 DGP uses retained
`dgp_zamboanga_final.pth`; CodeFormer uses the distinct official restoration
weights, fidelity 1 and AdaIN, internal 512 then returned 256. Raw floats and
floor(float×255) PNGs are saved without display enhancement or face alignment.

PSNR/MAE exclude padding. SSIM averages its seven-pixel-window map on a
three-pixel-eroded observed mask. Perfect clear-input PSNR is represented by
`null` plus `perfect_match=true`, not a finite score. The mask includes source
hair/background; these are reference-agreement metrics, not a direct measure
of facial identity or perceived usefulness.

One SCRFD request per reference fixes five landmarks and a 112px similarity
transform. Exactly one detection at confidence≥0.6 with eye span≥8 is required.
The same transform crops reference, input and both raw outputs. Seven references
qualify; the strong profile has no detection. All arms share eligibility, including
all six input-selected in-scope references. Frozen ONNX cosine is a development
diagnostic, not independent identity accuracy or recovered hidden truth.

## Findings

For the **24 degraded cases from the six input-selected in-scope references**:

| Arm | Mean PSNR dB | Mean SSIM | Mean fixed-reference cosine |
| --- | ---: | ---: | ---: |
| Degraded input / resizing control | 19.2331 | 0.6029 | 0.3271 |
| Retained DGP raw | 17.3306 | 0.4926 | 0.2268 |
| CodeFormer restoration raw | 19.1314 | 0.5634 | 0.1945 |

Clear controls are kept separate. Their input is exactly the target; DGP's mean
SSIM is 0.7763 and cosine 0.8212, while CodeFormer's are 0.8907 and 0.8999. The
observed smoothing supports clean-image anchors and an eventual automatic
bypass for inputs that do not need restoration. It does not establish a bypass
threshold; the user override remains required.

Every degraded profile has lower mean fixed-reference cosine after either
restorer than after resizing in the six-reference subset. In low light, DGP's
PSNR falls from 15.0969 to 13.1253 and SSIM from 0.5745 to 0.4022. That is a measured
stress-domain failure alongside the code-derived absence of explicit exposure
augmentation; it does not prove a particular new augmentation will fix it.

Source results are separate. Over 12 degraded cases per source, FFHQ-thumbnail
cosine is 0.3326(input), 0.1858(DGP), 0.2528(CodeFormer); Asian-source cosine is
0.3217, 0.2679, 0.1363. Source membership is not an inferred ethnicity label.
Per-source/profile scores and all 40-case results remain in the bound JSON files.

All 40 cases, all four columns and the ten-row preview are reviewed. DGP repeatedly
softens eyes/skin and changes mouth appearance; clear inputs also lose detail.
CodeFormer can look coherent and sharper while generating different smiles,
hair/skin detail and severe compound/profile artifacts. `ref_02` has a generated
hand/knuckle-like object near the mouth in blur/low-light outputs, absent from
the reference. Clear glasses/head coverings outside facial features should remain;
no hidden-eye truth is scored for the sunglasses target. This assistant review is
development evidence; independent final human review remains pending.

## Execution and audit bindings

The run executes 40 DGP and 40 CodeFormer CPU forwards in 229.91 seconds after
loading, plus eight reference-detection requests and 112 frozen recognition
forwards. Reference geometry takes 1.36 seconds. There are zero optimizer updates
and unchanged model states. The independent audit reconstructs 136 PNGs, verifies
80 float stages and reconstructs 105 saved embedding cosines without model
forwards. It checks the fixed split, weights, source bytes, geometry, metrics,
source/profile summaries and time/forward guards. Sharing the fingerprinted
degradation helper is not an independent physical-sensor implementation.

| Evidence | SHA256 |
| --- | --- |
| Frozen protocol | `788437c9a6708faab6a737c6175d574939c9d0c028844e2ec32863f2fd0ba2a4` |
| Results | `afe601078153d133f68870c9ef120029c270099ae9def034efa40357dcd44a30` |
| Independent audit | `8a05eb9ec6559e80f9ced6930c4fb8c69d2bfd185d9464c4f698af71d1371964` |
| Input-selected in-scope analysis | `c4845ac83c294529cbcb68de3de84ac7c5e79a7f6437db6259f091a1e7fa9fbb` |
| Development visual review | `8203c3e53983851f88db35e03b4eb36198b704b514baf28f133cb119d8aa7a7b` |

Windows artifacts:
`C:\xampp\htdocs\YEAR 4\Testing\outputs\cctv_paired_regression_v1\`.
Linux counterpart after explicit transfer:
`~/forensic-dgp/outputs/cctv_paired_regression_v1/`.
The executable runners are under the corresponding `scripts/` directories:
`run_cctv_paired_regression.py` and `audit_cctv_paired_regression.py`.
Completed runners refuse overwrite/repetition; create a new version for a new run.

## Candidate training-pool audit and next gate

A deterministic candidate pool from the inherited split contains 1,024 training
references (512/source) and 128 validation references (64/source), including the
eight preceding paired development references. All 1,152 source files fully decode
and have recorded bytes/format/dimensions. No selected train/validation exact-byte
overlap is found, but two duplicate groups occur **within Asian-source training**.
Preserve this pool; deduplicate in a derived manifest before training.

All 576 FFHQ candidates are 128px. Among 512 Asian training candidates, median
width/height is 175.5/200; only one has both dimensions≥256. None of the 64 Asian
validation candidates does. The 32 predetermined previews include grayscale
photographs stored asRGB, scans/noise, dim originals, sunglasses, closed eyes/
partial covering and strong profiles. Readability/resolution does not qualify
every image as a clean uncovered target. Only these 32 examples were visually
reviewed; no all 1,152 quality-label claim is made.

Pool artifacts:
`C:\xampp\htdocs\YEAR 4\Testing\outputs\cctv_dgp_pilot_pool_v1\`
↔ `~/forensic-dgp/outputs/cctv_dgp_pilot_pool_v1/` after transfer.
The data-audit SHA256 is
`422d0646b2685664a42e3eca593acd10ac2bc2a58505381699f3fdb5b10dc79f`;
the independent source-decode/hash audit is
`0c9bb2573e18fa2687515505fe8ea43b29e30d51a501417cf7f658f2a421433f`.
Neither checks all 80,000 historical files or same-person identity duplication.
At this candidate-pool milestone it was **not training-ready**. Subsequent
deduplication, gate V1/V2 and input review are now complete in separate immutable
versions. V1 rejected tight crops for contextual padding despite visible facial
features; V2 measures feature-core/landmark support and masks unsupported identity
context consistently. It admits 917 training and 110 validation references; two
additional input-reviewed training exclusions and equal-source truncation yield
902 training references. The original raw pool is not itself a qualified manifest.
No high-resolution targets were downloaded or all-cohort pristine quality claimed.

The executable matched two-arm pilot is packaged and independently audited at
`C:\xampp\htdocs\YEAR 4\Testing\outputs\cctv_dgp_pilot_protocol_v1\`
↔ `~/forensic-dgp/outputs/cctv_dgp_pilot_protocol_v1/` after explicit report transfer.
The transfer bundle extracts as `~/forensic-dgp/cctv_dgp_vm_bundle/` from
`C:\xampp\htdocs\YEAR 4\Testing\outputs\cctv-dgp-vm-bundle.tar.gz`.
It freezes902 training references, 550 validation cases, two epochs / 226 updates
per arm and 90 minutes total. All1,012 target canvases, 2,354 prepared inputs and
5,419 archive entries reconstruct or match. Eight tests and a two-image CPU
forward/encoder-parity check pass without state changes/backward/optimizer updates.
The converted encoder maximum absolute error is 3.219e-6 against ONNX on those
two inputs. Actual GPU gradients/VRAM and trained output remain pending.

See `C:\xampp\htdocs\YEAR 4\Testing\CCTV_DGP_VM.md`
↔ `~/forensic-dgp/cctv_dgp_vm_bundle/CCTV_DGP_VM.md` for exact commands and
`CCTV_DGP_NEXT_PILOT.md` for the recipe. Its metrics use exported PNGs and
`ArcFace_observed_fixed`, with MSE-derived aggregate PSNR; they are a new protocol,
not directly interchangeable with this earlier float/unmasked regression.

## Research basis and limits

[Real-ESRGAN's primary paper](https://arxiv.org/abs/2107.10833) motivates varied
synthetic degradations for practical restoration. Our five-profile RGB proxy
does not reproduce that model, its full degradation pipeline or its published
results. [Unprocessing](https://arxiv.org/abs/1811.11127) addresses sensor noise
and camera processing; [the authors' code](https://github.com/timothybrooks/unprocessing)
also distinguishes RAW noise modeling from processed RGB. The approximate
exposure/shot/read-noise stress here is a development hypothesis without RAW
sensor calibration, tone-map inversion, Bayer modeling, H264 video or Zamboanga
camera metadata. Keep its paired scores separate from native CCTV evidence.

[Official FFHQ documentation](https://github.com/NVlabs/ffhq-dataset) distinguishes
128px thumbnails from 1024px aligned images and records per-image licenses/
attribution. It also states the dataset is not intended for improving facial
recognition technologies. Keep this work focused on face restoration and the
frozen recognizer as a development diagnostic; do not train a recognizer or
present these experiments as identity/attendance decisions. Terms/provenance
need to accompany any future source upgrade; no source upgrade is presumed here.

The 32 reserved native CCTV crops remain unviewed/unforwarded. No real Zamboanga
sample exists yet. The useful full app output and independent final review gates
remain open; a completed benchmark does not complete the Goal.
