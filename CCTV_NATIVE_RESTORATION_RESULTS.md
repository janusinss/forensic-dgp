# Native CCTV restoration development comparison — 3 October 2026

The first native CCTV comparison is complete and its saved pixels are independently
audited. The retained Phase 3 DGP often smooths weak features; CodeFormer sometimes
produces a coherent sharper face but also adds details that the input cannot verify.
Neither establishes reliable improvement across this gallery. No checkpoint or
application change is selected, and no training ran locally.

## Updated Goal and evidence locations

The current objective puts our own DGP restoration first, with completion supporting
covered regions. The replacement app Goal paragraph is in
`C:\xampp\htdocs\YEAR 4\Testing\SYSTEM_WORKFLOW_AND_GOAL.md` ↔
`~/forensic-dgp/SYSTEM_WORKFLOW_AND_GOAL.md` after transfer. The preceding milestone
read the app Goal as `paused`; the user has since applied the revised objective
and resumed it. A fresh read now verifies the complete DGP-first Goal as `active`.
The earlier control limitation is retained in the historical handoff milestone.

| Evidence | Windows local | Linux counterpart after explicit transfer |
| --- | --- | --- |
| Native selection and input review | `C:\xampp\htdocs\YEAR 4\Testing\outputs\cctv_native_development_v2\` | `~/forensic-dgp/outputs/cctv_native_development_v2/` |
| DGP/CodeFormer comparison and independent audit | `C:\xampp\htdocs\YEAR 4\Testing\outputs\cctv_native_comparison_v1\` | `~/forensic-dgp/outputs/cctv_native_comparison_v1/` |
| Signal preprocessing | `C:\xampp\htdocs\YEAR 4\Testing\outputs\cctv_signal_preprocessing_v1\` | `~/forensic-dgp/outputs/cctv_signal_preprocessing_v1/` |
| Alignment feasibility | `C:\xampp\htdocs\YEAR 4\Testing\outputs\cctv_alignment_feasibility_v2\` | `~/forensic-dgp/outputs/cctv_alignment_feasibility_v2/` |
| Independent processing audit | `C:\xampp\htdocs\YEAR 4\Testing\outputs\cctv_processing_evidence_v1.json` | `~/forensic-dgp/outputs/cctv_processing_evidence_v1.json` |

## Source-native selection

Use the official QMUL-SurvFace release documented in `CCTV_BENCHMARK_STATUS.md`.
Select cases deterministically by a fixed hash ordering before inspecting model
outputs. The shorter native side defines four diagnostic ranges: ≤15, 16–23,
24–39 and ≥40 pixels. These are sampling ranges, not automatic rejection thresholds.

There are 24 development crops, six per range, from 24 labeled published training
identities. There are 32 reserved crops, eight per range, from separate labeled
published gallery identities. The 56 native copies match archive bytes exactly;
selected person IDs do not overlap and selected byte hashes are distinct.
The reserved crops receive header/hash checks only: no contact sheet, visual
inspection, restoration inference or optimization. Do not tune on these cases.
Published split separation does not prove separation from older model-training data.

The initial V1 selector incorrectly assumed every `.jpg` member contained JPEG
data. It stopped with an incomplete selection and is preserved. A bounded header
diagnostic confirmed valid PNG content under many `.jpg` filenames; every ≥40-pixel
case in that diagnostic sample was PNG. V2 accepts the detected JPEG/PNG formats,
retains original bytes/names and uses the same seed, size ranges and quotas.
The selected 56 cases contain 24 PNG and 32 JPEG encoded images. This is not a
format census of the entire archive or an individual source-country attribution.

Subset SHA256:
`c063985b258b808efaa897c88593cbdbcee2848d686d3d056219f8a8e555a98e`.
Input-only review SHA256:
`41912033f5070547e120a6b384024fa929eeaf3a7ecca408c0c67c53d37d36e2`.
The source review records tiny/dark insufficient-information cases and full
profiles separately. All 24 remain in the comparison. Six cases have coarse
readable structure and an approximately frontal/mild pose according to the
assistant's pre-output review; these judgements are developmental, not final labels.

## Controlled model comparison

Convert native RGB to a square canvas by center-padding the shorter axis with
RGB128, then resize with Pillow bilinear to 256×256. Both models receive the
same exact quantized input. No alignment, denoising, synthetic degradation or
additional crop is applied in this first comparison.

| Arm | Behavior observed in development review |
| --- | --- |
| Bilinear common input | Retains the degraded observation; resizing supplies no new observed detail |
| Phase 3 DGP raw | Often smoother, with weak eyes/mouth structure and darker low-light appearance; useful detail recovery is not established across the gallery |
| DGP with legacy display only | More local contrast, with amplified patterned/grain-like texture; enhancement is not evidence of recovered anatomy |
| CodeFormer restoration raw, fidelity 1 | Some coherent sharper faces, but also washout, textures and resolved anatomy beyond what the input can validate |

Keep `checkpoints/dgp_zamboanga_final.pth`, SHA256
`b6376f56c4161ef1f26a8f9efb0bb309b3014f4ecdeaca5399a47c1c4607586c`.
The comparison uses the separately fingerprinted restoration CodeFormer weights,
not the inpainting weights. CodeFormer operates internally at 512 and returns
256; it remains a comparison baseline, not the DGP thesis contribution.

The display-only arm reproduces the current top-1 illumination calibration,
CLAHE and unsharp step. It does **not** reproduce the full legacy `/reconstruct`
denoising/alignment/top-k pipeline. Raw float stages and PNGs remain separate.
Review all four six-row sheets; the ten-row preview contains the first ten fixed
cases, not a representative or selectively favorable replacement for the gallery.

Runtime is 141.08 CPU seconds after loading: 24 DGP and 24 CodeFormer forwards,
zero optimizer updates. Both model state fingerprints remain unchanged. The
independent auditor verifies 56 archive copies, 48 raw float stages and exact
reconstruction of 96 input/raw/display PNGs. No clean paired reference exists:
PSNR/SSIM are intentionally absent. Input-change MAE is diagnostic only and is
not a model-quality, identity-preservation or checkpoint-selection score.

| Binding | SHA256 |
| --- | --- |
| Comparison protocol | `b5d177d0601048d8ef85fc3a7029508c54ccc3887b03626ca561f9cea642bb30` |
| Comparison results | `3be73af7b0cdae85f54adc8173fd8f48b42511a604fcd60244e3b1d2565a4c34` |
| Independent comparison audit | `c7801b90a817f20403abc3bd99743e8d0f8e782003b1e1d938bfff05be38671a` |
| Assistant visual review | `3402e2e6920e47ffa361e054d539114a224daf1dde53f3e1ac02fbde2ac59433` |

## Processing findings before another VM run

The signal audit runs the existing native-space deinterlace → mosaic smoothing
→ adaptive denoise sequence on all 24 development cases, without alignment or
restoration. Interlacing/mosaic triggers are zero; 23 inputs are unchanged.
Only `dev_le15_03` changes, on 151 native pixels. Signal filtering therefore does
not explain the general smoothing in the controlled raw-model comparison, which
used unfiltered inputs. The preprocessing audit takes 0.40 seconds after import.
Its trigger values do not prove the source contains those physical degradations.

The alignment feasibility probe uses exactly the six coarse frontal/mild cases
selected from the **pre-output input review**. It compares the unchanged legacy
geometry function on native RGB with the already padded 256×256 RGB input.
The initialization uses existing fingerprinted FAN/S3FD weights and eager
`compile=False`, avoiding a compilation warmup. No restoration or optimization runs.

Native alignment returns zero canonical transforms out of six requests. Five
detector calls raise `RuntimeError: ... Output size is too small`; the legacy
helper silently catches these errors and center-crops instead. Using the padded
256 input removes these size errors and returns three canonical transforms out
of six. The remaining three still fall back. Passing a geometric gate does not
prove that the predicted landmarks are correct or restoration improves.

The first alignment probe is preserved as incomplete: its combined guard aborted
before saving caught-error findings. V2 changes reporting to retain those errors
as explicit diagnostic fallbacks; time/state failures remain fatal. It takes
12.70 seconds after loading, with 13 S3FD and six FAN attempted forwards,
12 alignment requests, no optimizer updates and unchanged model states.
Saved transforms visibly alter framing/tilt on some sources; no universal
alignment default is adopted. A focused native-size error reproduction is also
preserved in `outputs/cctv_alignment_failure_diagnostic_v1.json`.

An independent processing audit reconstructs all 96 signal-stage PNGs and all
12 alignment PNGs, checks source spaces, geometry gates, matrices, original
code fingerprints, unchanged states and the five captured detector exceptions.
It executes zero model forwards. Processing-audit SHA256:
`0bd4b61f6fa2e50a04f20d769f31efca9b050fe5e9400198b548a3c8e7306a32`.

Code inspection also finds a training-domain gap: the current curriculum includes
24–32-pixel, 48–64-pixel and 128–256-pixel inputs, while this native gallery has
smaller crops. It adds fog/noise/codec effects but no explicit low-light exposure
step. This is a code-derived mismatch hypothesis, not proof that changing the
recipe will improve restoration. Do not train on hopeless cases to encourage
invented structure, or use a degraded real crop as an uncovered clean target.

## Aligned restoration comparison — 3 October 2026

The three canonical successes from the input-selected feasibility probe now have
a frozen common-frame restoration comparison. The runner reuses the audited
256-space affine matrices and pixels; it does not run another landmark detector.
It compares restoration after alignment with each cached unaligned model output
warped into that same frame. A nearest-neighbor source-observation mask excludes
reflected/padded pixels from input-change diagnostics. The extra resampling of
the cached control is an explicit confound; the three cases cannot establish a
universal preprocessing default.

There are three DGP and three CodeFormer CPU forwards, taking 17.06 seconds after
loading, with zero optimizer updates and unchanged model states. The independent
audit reconstructs all 18 PNGs and verifies 12 float stages, observed masks and
the saved geometry. The five-column, three-row grid has been reviewed. DGP stays
soft; alignment changes some eye/detail/framing appearance without establishing
a reliable gain across the cases. CodeFormer adds unverified detail and still
shows uncertain pose/framing. No alignment default or checkpoint is selected.

Artifacts: `C:\xampp\htdocs\YEAR 4\Testing\outputs\cctv_alignment_restoration_comparison_v1\`
↔ `~/forensic-dgp/outputs/cctv_alignment_restoration_comparison_v1/` after transfer.

| Binding | SHA256 |
| --- | --- |
| Aligned protocol | `4bf6a1a2f776cba9d9504c0a81a6fd80a29a9e69056e8e1214be91fe29906cde` |
| Aligned results | `b7bee3e3b1da6f4a932ec24ece9aa6b493ebfaabad7918046234a331a1f7ec79` |
| Independent audit | `22978b2ece744304caca040b9a2827657583517269d4047129af8dc3f57eaae2` |
| Development visual review | `f359841c69d025e3baf25e8a0592d7312b676a601cf23fe5d8f75934ca941304` |

## Next execution

1. Transfer the independently verified camera/clear-anchor pilot package in
   `CCTV_DGP_VM.md`; the derived manifest has902 training and 110 validation
   references. Original pool, gate V1 failure and gate V2 remain immutable.
2. Run only its guarded VM preflight and matched two-arm pilot,452 updates total
   and 90 minutes maximum. GPU behavior and new output quality remain pending;
   synthetic reference agreement does not establish CCTV identity accuracy,
   hidden-face truth or Zamboanga performance.
3. Audit the returned checkpoints/pixels/cohorts and compare qualified candidates
   on this24-case native development gallery. Original Phase 5 full GPU return
   remains absent; do not launch it or historical detector bundles unchanged.
4. Integrate the validated DGP route and insufficient-information behavior in the
   existing app, with conditional completion as a separate component.
5. Evaluate the frozen candidate under independent final review and verify the
   app with bundled inline Playwright. The 32 reserved crops, real Zamboanga data
   and requested covering families cannot be claimed validated by these diagnostics.

The completed runners refuse to overwrite their evidence. For a new experiment,
create a new protocol/output version rather than rerunning `prepare` or `run`.
The executable scripts are under `C:\xampp\htdocs\YEAR 4\Testing\scripts\`
↔ `~/forensic-dgp/scripts/` after explicit transfer. Existing app, checkpoints,
historical protocols and native archive remain intact.
