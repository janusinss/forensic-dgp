# Color/detail counterfactual V1: no processing remedy qualified

10 October 2026. Four predeclared arithmetic controls inspect the two returned
full-decoder arms with raw structure gain above 1%. No model, gradient,
parameter update or learned output selector is run locally. Both fitting pools
and all 100 exposed paired TRAIN cases are retained, giving 400 transformed
outputs. Aligned targets are used for scoring only; the transforms use the
retained baseline and failed candidate arrays. Native and final pixels are not
decoded. No app change or model qualification follows.

The constant-RGB control removes the observed-support mean of candidate minus
baseline. The low-band control subtracts a fixed sigma2, 13tap Gaussian-filtered
difference, adding the remaining difference to the baseline. This sigma is
inherited from the existing detail measurement; no parameter sweep or fitting
occurs. Values are clipped to [0,1], converted to float32, retain camera pixels
outside observed support, and use the original floor-to-PNG policy. These are
counterfactual processing diagnostics, not raw neural outputs or recommended
delivery processing.

| Pool | Fixed control | Raw feature gain | PNG feature gain | Clear raw MSE multiplier | Raw/PNG pixel preservation failures | Result |
| --- | --- | ---: | ---: | ---: | ---: | --- |
| 0 | remove_constant_RGB_delta | 1.955282% | 1.964808% | 4.007483× | 27/27 | Fail |
| 0 | retain_base_low_band_sigma2 | 0.874326% | 0.898383% | 1.000282× | 26/24 | Fail |
| 1 | remove_constant_RGB_delta | 1.255331% | 1.263470% | 3.490134× | 27/27 | Fail |
| 1 | retain_base_low_band_sigma2 | 0.553217% | 0.576301% | 0.993884× | 25/25 | Fail |

The independent V1 implementation reconstructs 400 transformations, 800 metric
records and all 400 exact saved PNGs, checking 209 source bindings. Transform
replay is exact; its independently implemented high-pass filter differs by at
most 8.67e-19. MSE/SSIM thresholds remain 1e-12/1e-6 and the necessary early
filter-gain requirement remains 1%. All four controls fail these necessary
pixel requirements. ArcFace is not recomputed and these new transformed images
have no visual-usefulness review; the original neural-return gallery is the
separate completed 64-sheet review. Therefore no full preservation or visual
qualification claim is possible.

A subsequent precision review corrects PNG-detail conversion to float64 from
saved uint8, matching the original frozen PNG metric. V1's float32 conversion
had compared slightly different filter arithmetic with that baseline. Retain
V1 and its receipt. R2 changes only the 400 PNG-detail values and their derived
group/gain summaries. Maximum changed per-case detail error is 3.86e-11;
independent OpenCV versus SciPy discrepancy is at most 1.31e-18. All raw metrics,
all other PNG metrics, failure locations and decisions remain exact. No image,
scientific threshold or model changes. The table uses R2.

The initial preparation's incorrect assumption of per-case raw NPY files also
remains recorded. Returned paired arrays actually use five-case lossless NPZ
packs; both revised decoders verify each recovered float32 array against its
original stored pixel hash. Candidate XOR storage changes no raw value.

These findings reject a display-only correction as the sufficient remedy for
these two failed arms. They do not establish that every possible training-time
color constraint is ineffective or identify a unique cause. The selected next
investigation is active reconstruction supervision before the first update,
with finite raw/PNG acceptance and source/profile protection still required.

[Original frozen arithmetic plan](<C:/xampp/htdocs/YEAR 4/Testing/outputs/cctv_dgp_multiscale_color_detail_v1/plan.json>),
[Independent V1 transformation audit](<C:/xampp/htdocs/YEAR 4/Testing/outputs/cctv_dgp_multiscale_color_detail_v1/independent_audit.json>),
[R2 metric correction and all decisions](<C:/xampp/htdocs/YEAR 4/Testing/outputs/cctv_dgp_multiscale_color_detail_v1_r2/results.json>).
