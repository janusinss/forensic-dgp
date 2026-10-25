# R2, U96 and S96: fixed development comparison

21 October 2026. This comparison answers the user request to check the three models objectively. It does not change the preferred model, promote an app default or qualify forensic identity recovery. All original states and the interrupted historical run remain intact.

**Measured PNG reconstruction order on all 36 paired DEV observations: U96 > S96 > R2**, where better means lower image MAE to the saved clean photograph. The SSIM order is U96 > S96 > R2. These are explicitly metric-specific orders, not a universal face-quality or identity ranking.

## Fair input and execution contract

The exact previously frozen 68 inputs are retained: 36 paired filter-stress observations from four DEV reference photographs, 24 native ChokePoint crops and eight native QMUL-SurvFace crops. R2, U96 and S96 receive identical original-upload geometry, observed support, prepared RGB256, w=1/AdaIN and 256 delivery. No context blur, brightness enhancement, sharpening, alignment or model-specific input processing was added. The 40 input-only refusal controls remain excluded. Borderline filter cases and coarse QMUL cases remain explicit diagnostic probes, not usable-input admissions.

Thirty exact outputs from the ten previously audited cases were reused. The174 new CPU forwards completed in988.8seconds. The remaining 174 frozen CPU forwards ran on a separate loopback server, without occupying the interactive comparison server. All 204 output bundles/RAW/PNG/provenance records were checked from saved files with exact source bindings. Counts, Content-Length bodies, ZIP CRC/member sets, model file hashes, RAW-to-PNG floor quantization and exact outside-support preservation were verified. No training, gradient, optimizer, VM or checkpoint mutation occurred. The new audit is an author saved-only readback using the retained independently implemented metric helper, not an independent blinded clinical/forensic study.

## Paired results

Values below are arithmetic means of per-observation scores. The four photographs each have nine profiles; these are 36 observations, not 36 independent subjects. RAW and PNG scores are separately retained in JSON/CSV. SSIM uses the unchanged uniform7x7/sample-covariance RGB definition; it is not a facial-landmark or recognition score. Adjacent and centered residual errors are diagnostics, not the historical structure gate.

### All36 paired DEV observations

| Model | PNG MAE ↓ | PSNR dB ↑ | SSIM ↑ | Adjacent residual error ↓ | Centered residual MAE ↓ |
|---|---:|---:|---:|---:|---:|
| R2 | 0.047526 | 23.985 | 0.789688 | 0.020161 | 0.036542 |
| U96 | 0.044855 | 24.412 | 0.795592 | 0.019715 | 0.035489 |
| S96 | 0.047224 | 24.024 | 0.791194 | 0.020135 | 0.036462 |

U96 reduces mean PNG MAE by5.6209% versus R2; S96 reduces it by0.6358%. U96 wins MAE on36/36 observations and SSIM on34/36. This is not a percentage improvement in facial identity accuracy or the historical1% structure gate.

### Readable degraded inputs only

Nine input-admitted degraded observations from three reference photographs; clear controls, borderline probes and the appearance-flagged reference are not mixed into this table.

| Model | PNG MAE ↓ | PSNR dB ↑ | SSIM ↑ | Adjacent residual error ↓ | Centered residual MAE ↓ |
|---|---:|---:|---:|---:|---:|
| R2 | 0.043437 | 24.395 | 0.833079 | 0.017686 | 0.030038 |
| U96 | 0.040153 | 24.904 | 0.840855 | 0.017238 | 0.028786 |
| S96 | 0.043067 | 24.442 | 0.835100 | 0.017660 | 0.029971 |

U96 reduces mean PNG MAE by7.5597% on the nine readable degraded observations. This small subset does not cover every CCTV degradation.

### Clear controls

| Model | PNG MAE ↓ | PSNR dB ↑ | SSIM ↑ | Adjacent residual error ↓ | Centered residual MAE ↓ |
|---|---:|---:|---:|---:|---:|
| R2 | 0.026736 | 28.648 | 0.919853 | 0.013840 | 0.023676 |
| U96 | 0.024260 | 29.386 | 0.926355 | 0.013278 | 0.022483 |
| S96 | 0.026479 | 28.711 | 0.921121 | 0.013797 | 0.023553 |

This table includes the separately flagged08025 reference. Passing a relative R2 comparison does not resolve historical clear/normal appearance failures or qualify automatic restoration.

### Source/appearance stratification

| Stratum | Observations | PNG MAE order, lowest first |
|---|---:|---|
| all32degraded | 32 | U96 > S96 > R2 |
| input_readable12 | 12 | U96 > S96 > R2 |
| input_borderline24 | 24 | U96 > S96 > R2 |
| appearance_unflagged27 | 27 | U96 > S96 > R2 |
| appearance_flagged9 | 9 | U96 > S96 > R2 |
| reference_va_ffhq_00383 | 9 | U96 > S96 > R2 |
| reference_va_ffhq_01093 | 9 | U96 > S96 > R2 |
| reference_va_ffhq_08025 | 9 | U96 > S96 > R2 |
| reference_va_ffhq_09056 | 9 | U96 > S96 > R2 |

### Direct paired wins

| Pair | PNG MAE better/equal/worse | PNG SSIM better/equal/worse |
|---|---:|---:|
| spatial_u_vs_r2 | 36/0/0 | 34/0/2 |
| spatial_s_vs_r2 | 36/0/0 | 32/0/4 |
| spatial_u_vs_spatial_s | 36/0/0 | 34/0/2 |

The exact-zero equality band is numerical bookkeeping only; a tiny metric win is not a meaningful visible improvement. One photograph contributes nine correlated observations. Repeated DEV exposure and possible unknown pretrained/subject overlap remain limitations.

## Native CCTV: unpaired evidence

| Capture source | Crops | Clean target | U96 difference from R2, mean RGB8 levels | S96 difference from R2, mean RGB8 levels |
|---|---:|---|---:|---:|
| native_ChokePoint | 24 | None | 2.0532 | 0.2488 |
| native_QMUL-SurvFace | 8 | None | 0.6916 | 0.0948 |

Output distance measures change, not correctness. Native PSNR/SSIM and a clean-target winner are intentionally null. Do not rank generated native details by apparent sharpness or invent target accuracy. Source labels do not establish ethnicity or Zamboanga performance. All17 planned sheets were personally inspected; findings are bound in the root review receipt.

## Prior training evidence remains separate

[The retained independent TRAIN review](<C:/xampp/htdocs/YEAR 4/Testing/outputs/codeformer_spatial_continuation_return_review_v1/INDEPENDENT_NUMERIC_REVIEW.md>) reports U96 moderate/strong PNG MAE reductions of8.8875%/4.8837% versus R2, compared with S96 reductions1.0739%/0.6705%. S96 lowers centered feature L1 more (0.6890% versus U96 0.1296%). These fitted/repeated TRAIN measurements do not replace the new DEV comparison. Small TRAIN guards passed, but inherited full-development preservation failures remain unresolved.

## Visual evidence and limits

All sheets paste exact saved 256 PNGs beside inputs and paired targets/native notes. Coded A/B/C panels reduce model-label emphasis, but root review is not an independent blinded study. The panel key is retained. No screenshot enhancement or resampling is used in sheet construction.

Coded findings were recorded before opening the key: A=R2, B=U96, C=S96. R2 and S96 look extremely similar. U96 mainly changes tone/local contrast and is modestly darker on several cases. All three share smoothing, altered eye/mouth/glasses detail and persistent local-blur/mosaic defects. The review did not establish a material new facial-structure winner. Coarse QMUL probes can produce weakly supported or malformed features with all three models; input uncertainty remains in force.

Root personally reviewed all17 sheets/204 model PNG cells. The nine final paired sheets are byte-identical to the reviewed coded previews. Paired visual review preceded the full numerical score readout; the remaining native review followed the paired scores, so coding does not imply fully blinded review. [Root visual observations](<C:/xampp/htdocs/YEAR 4/Testing/outputs/codeformer_r2_u96_s96_objective_v1/root_visual_review_v1.json>).

| Sheet | Cases / outputs |
|---|---|
| [01_filter_DEV.png](<C:/xampp/htdocs/YEAR 4/Testing/outputs/codeformer_r2_u96_s96_objective_v1/visuals/01_filter_DEV.png>) | 4 / 12 |
| [02_filter_DEV.png](<C:/xampp/htdocs/YEAR 4/Testing/outputs/codeformer_r2_u96_s96_objective_v1/visuals/02_filter_DEV.png>) | 4 / 12 |
| [03_filter_DEV.png](<C:/xampp/htdocs/YEAR 4/Testing/outputs/codeformer_r2_u96_s96_objective_v1/visuals/03_filter_DEV.png>) | 4 / 12 |
| [04_filter_DEV.png](<C:/xampp/htdocs/YEAR 4/Testing/outputs/codeformer_r2_u96_s96_objective_v1/visuals/04_filter_DEV.png>) | 4 / 12 |
| [05_filter_DEV.png](<C:/xampp/htdocs/YEAR 4/Testing/outputs/codeformer_r2_u96_s96_objective_v1/visuals/05_filter_DEV.png>) | 4 / 12 |
| [06_filter_DEV.png](<C:/xampp/htdocs/YEAR 4/Testing/outputs/codeformer_r2_u96_s96_objective_v1/visuals/06_filter_DEV.png>) | 4 / 12 |
| [07_filter_DEV.png](<C:/xampp/htdocs/YEAR 4/Testing/outputs/codeformer_r2_u96_s96_objective_v1/visuals/07_filter_DEV.png>) | 4 / 12 |
| [08_filter_DEV.png](<C:/xampp/htdocs/YEAR 4/Testing/outputs/codeformer_r2_u96_s96_objective_v1/visuals/08_filter_DEV.png>) | 4 / 12 |
| [09_filter_DEV.png](<C:/xampp/htdocs/YEAR 4/Testing/outputs/codeformer_r2_u96_s96_objective_v1/visuals/09_filter_DEV.png>) | 4 / 12 |
| [10_native_ChokePoint.png](<C:/xampp/htdocs/YEAR 4/Testing/outputs/codeformer_r2_u96_s96_objective_v1/visuals/10_native_ChokePoint.png>) | 4 / 12 |
| [11_native_ChokePoint.png](<C:/xampp/htdocs/YEAR 4/Testing/outputs/codeformer_r2_u96_s96_objective_v1/visuals/11_native_ChokePoint.png>) | 4 / 12 |
| [12_native_ChokePoint.png](<C:/xampp/htdocs/YEAR 4/Testing/outputs/codeformer_r2_u96_s96_objective_v1/visuals/12_native_ChokePoint.png>) | 4 / 12 |
| [13_native_ChokePoint.png](<C:/xampp/htdocs/YEAR 4/Testing/outputs/codeformer_r2_u96_s96_objective_v1/visuals/13_native_ChokePoint.png>) | 4 / 12 |
| [14_native_ChokePoint.png](<C:/xampp/htdocs/YEAR 4/Testing/outputs/codeformer_r2_u96_s96_objective_v1/visuals/14_native_ChokePoint.png>) | 4 / 12 |
| [15_native_ChokePoint.png](<C:/xampp/htdocs/YEAR 4/Testing/outputs/codeformer_r2_u96_s96_objective_v1/visuals/15_native_ChokePoint.png>) | 4 / 12 |
| [16_native_QMUL-SurvFace.png](<C:/xampp/htdocs/YEAR 4/Testing/outputs/codeformer_r2_u96_s96_objective_v1/visuals/16_native_QMUL-SurvFace.png>) | 4 / 12 |
| [17_native_QMUL-SurvFace.png](<C:/xampp/htdocs/YEAR 4/Testing/outputs/codeformer_r2_u96_s96_objective_v1/visuals/17_native_QMUL-SurvFace.png>) | 4 / 12 |

Full478 historical/filter coverage, reserved independent FINAL, formal preservation gates and user restoration acceptance remain outstanding. This fixed68 subset can support a metric-specific measured preference; it does not prove an overall qualified replacement. R2 remains a protected permanent reference regardless of the descriptive numeric order. No app default or current training priority is changed automatically.

Evidence: [frozen plan](<C:/xampp/htdocs/YEAR 4/Testing/outputs/codeformer_r2_u96_s96_objective_v1/comparison_plan_v1.json>), [saved output audit + all metrics](<C:/xampp/htdocs/YEAR 4/Testing/outputs/codeformer_r2_u96_s96_objective_v1/saved_output_audit_v1.json>), [paired RAW/PNG CSV](<C:/xampp/htdocs/YEAR 4/Testing/outputs/codeformer_r2_u96_s96_objective_v1/paired_metrics_v1.csv>), [metric sanity fixtures](<C:/xampp/htdocs/YEAR 4/Testing/outputs/codeformer_r2_u96_s96_objective_v1/metric_fixture_check_v1.json>), [coded panel key](<C:/xampp/htdocs/YEAR 4/Testing/outputs/codeformer_r2_u96_s96_objective_v1/visuals/panel_key.json>).

The batch server was stopped after completion. The previous interactive process was later found absent; its cause was not established. The unmodified interactive launcher was restarted on61815 and its zero-forward status verified. [Server lifecycle](<C:/xampp/htdocs/YEAR 4/Testing/outputs/codeformer_r2_u96_s96_objective_v1/server_closure_v1.json>).
