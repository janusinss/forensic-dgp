# V10 face-prior comparison: 4 October 2026

The direct DGP → CodeFormer cascade is **not adopted**. It sometimes produces
sharper faces, but paired examples show invented glasses, facial hair and changed
eyes, mouths and expressions. Brightness recovery does not establish faithful
facial restoration. No new conditioner was trained or checkpoint promoted.

Windows root: `C:\xampp\htdocs\YEAR 4\Testing\`; intended report counterpart:
`~/forensic-dgp/CCTV_DGP_FACE_PRIOR_RESULTS_V10.md` after explicit sync.
This experiment ran locally. The L4 remains stopped after V9.

## Evidence and execution limitation

The frozen comparison used 50 existing paired proxy cases: ten distinct
validation references, five per source, with all five fixed profiles. It also
used all 24 native CCTV development crops. The 32 reserved native images were
not forwarded or rendered. Native data has no aligned clean reference.

The original runner saved all case outputs, then failed while constructing its
first preview: Pillow received a NumPy array instead of an Image. Preserve that
run and its failure. It did **not** write its final model-state or forward-count
receipt. Saved artifacts cannot establish those missing checks.

A separately frozen recovery copied all 781 original files byte-for-byte and
rebuilt metrics and ten grids with no model forwards. Its independent audit
passed: 370 PNG measurements/compositions, 148 raw floats, 250 cosine values,
222 cached controls/composites, 70 summary groups and 470 original grid cells.
Recovery took 16.07 seconds; its audit took 15.01 seconds. The original log
recorded all native outputs saved at 765 seconds; this is a progress timestamp,
not a completed execution receipt. Original forward counts are artifact-implied.

| Evidence | SHA256 |
| --- | --- |
| Original comparison plan | `8393e67abeec1e21428fb56c97e7d6be8d3657a38bd1767fd89e127ce805b250` |
| Separate recovery plan | `4302708d31b8e466f5daac6ab71bfabe197b7d606ae369f77d7582dcef629d15` |
| Recovered results | `183657baa27ab9397a2ce6182ffce5e342bd32ab385e5a1280c21fa31a4b9866` |

Original run: Windows `outputs\cctv_dgp_face_prior_feasibility_v10\run\`.
Recovered review: Windows `outputs\cctv_dgp_face_prior_recovery_v10_r2\review\`.
Their intended VM locations are the same relative paths below `~/forensic-dgp/`
after explicit report transfer; they are not claimed transferred. The recovery
directory includes `independent_audit.json` and `assistant_visual_review.json`.

## Paired proxy results

These are the **40 degraded cases** in the 50-case subset. They are not directly
comparable to V9's full 520-case aggregate. ArcFace is fixed-affine recognizer
similarity to photographic references, not identification accuracy.

| Arm | PSNR, dB | SSIM | ArcFace similarity |
| --- | ---: | ---: | ---: |
| Exact input | 14.2664 | 0.59165 | 0.36057 |
| Cached V2 retained baseline | 15.7509 | 0.62591 | 0.35853 |
| V9 epoch20 diagnostic | 21.6204 | 0.66109 | 0.32986 |
| CodeFormer(input), w1 | 14.1953 | 0.52275 | 0.23590 |
| CodeFormer(V9 PNG), w1 | 20.8840 | 0.59894 | 0.23318 |

The cascade raises pixel quality relative to direct CodeFormer, but loses
structure and recognizer similarity relative to its DGP input. The same pattern
appears in both source aggregates. On clear cases, V9's preservation is better
than either prior arm. No numerical improvement overrides visual failures.

## Original-cell development review

All five paired profile grids and all four native galleries were inspected at
their original 256-pixel cells, plus the ten-row native core/insufficient preview.
Each paired profile contains ten different references; these are still
photographic proxies, not real Zamboanga CCTV.

- Blurred pairs include invented glasses on `va_asian_09121`, changed expressions
  on `va_asian_09042` and altered eyes/mouths on several other references.
- Compound pairs add facial hair absent from references, including
  `va_ffhq_01093` and `va_ffhq_13682`. Low-light and motion cases also show
  painted texture and feature changes.
- The six eligible native core crops have mixed outcomes. Some faces appear
  coherent, but several have missing eyes, texture stripes or changed features.
  Without clean native truth, apparent sharpness is not verified fidelity.
- Seven input-reviewed insufficient crops still require a clearer crop. Other
  development rows include strong profiles and obstructions outside the first
  frontal/mild restoration scope. They are reported separately from core cases.
- Independent final human review is pending. The review rejects this cascade as
  the next adopted DGP upgrade; it does not rule out a properly trained prior.

## Next implementation

Prepare a separate feature/code-conditioning prototype. Keep the pretrained
codebook and renderer frozen and identified; train only our declared conditioning
layers. Supervise face-code prediction from clean **training** references rather
than backpropagating through the existing inference wrapper and hard top-1 lookup.
First prove a zero-initialized conditioner reproduces the frozen baseline and
that the chosen prior can represent the training targets. Any backward preflight
and actual pilot must run on the L4 with finite updates, timing and unchanged
validation safeguards. Do not restart V10, repeat V9 unchanged or change the app
default before useful outputs are demonstrated.

The official CodeFormer training procedure separates learned codebook,
code-prediction and controllable-module training. It motivates this interface;
our CCTV usefulness remains to be demonstrated.
[Author training procedure](https://github.com/sczhou/CodeFormer/blob/master/docs/train.md).
