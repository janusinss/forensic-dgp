# DGP processing diagnostics — 5 October 2026

Both isolated inference variants are closed without app adoption. Neither result
qualifies useful native CCTV restoration or the covering-family scope. The app's
retained trained DGP, confirmed removal footprints, automatic selection and
overrides remain unchanged. No local training, VM/cloud action or reserved
evaluation was performed.

The fixed six-pixel **conditioning** suppression test uses the same exposed 36
photographic development cases as the current route. It erases a wider area only
from the completion model's input, discards generated ring pixels and keeps the
operator's final removal footprint. This is not a change to the removal margin.
One zero-radius parity control and 28 nonempty cases require 29 completion
forwards, taking 151.44s under the frozen 300s worker/350s external limits. Four
empty-mask controls bypass completion, and four cases retain their input-only
pose/insufficient-face exclusions. There are 96 outputs and 32 exact Auto aliases.
The completion state is unchanged. Existing DGP visible pixels are reused; no
new DGP or detector forward occurs.

An independent 5.12s saved-output audit checks 360 source and 191 artifact
bindings, the zero-radius raw control, exact conditioning dilations, all 96
compositions and original Off/cached On visible pixels. All eight 1340×1192
comparison sheets were viewed at original resolution. Scarf/glove texture is
reduced inside the removed area, but sharp colour/texture boundaries appear
across cloth masks, sunglasses/glare, hands, hair and the leaf case. A universal
six-pixel setting therefore fails the frozen material-regression criterion.
Cropping a prediction made with wider erased context back to the smaller output
area is a plausible explanation for the join mismatch; it is an inference, not
a proven latent cause. No per-case setting was fitted and no global output-mask
expansion is adopted. Nonremoved paired metrics are unchanged by construction;
there is no aligned clean hidden-face target.

A separate DGP diagnostic compares stored evaluation statistics with per-image
statistics that pass no running buffers to the normalization operator. Identical
trained parameters are used on two **existing training-role clear controls**;
four CPU forwards take 7.07s under 120s worker/180s external limits. All five
normalization layers are instrumented; learned parameters and stored buffers
remain unchanged. No optimizer, backward call or statistics writeback occurs.
The prospective stop requires no clear-control MSE increase beyond 1e-12 and no
valid-window SSIM decrease beyond 1e-6 before any broader/native evaluation.

| Training reference / source folder | Stored PSNR | Per-image PSNR | Stored SSIM | Per-image SSIM | Stop |
| --- | ---: | ---: | ---: | ---: | --- |
| tr_asian_00048 / dataset/asian_faces | 32.98744 | 35.94814 | 0.969057 | 0.968097 | SSIM loss |
| tr_ffhq_00084 / dataset/thumbnails128x128 | 29.22293 | 29.70442 | 0.877165 | 0.867938 | SSIM loss |

These are paired clear photographic controls, not native CCTV or generalization
results. Folder names record provenance and do not establish ethnicity. The
independent 0.31s audit reproduces all four raw-to-PNG compositions, observed
MSE/PSNR and valid-window SSIM, both stop failures and eight exact sheet cells.
It checks consistency of the 20 saved activation records, but does not replay
neural activations or checkpoint state. The 1072×608 sheet was viewed at original
resolution: tone changes do not establish a coherent structural gain, and the
second face/glasses remain soft. Both SSIM stops fire, closing this variant
before broader or native inference. This does not prove the overall cause of
DGP softness; it rejects this particular inference substitution under its
declared preservation guard.

Evidence is in
[completion_margin_full_v3](<C:/xampp/htdocs/YEAR 4/Testing/outputs/completion_margin_full_v3/visual_review.json>)
and
[dgp_eval_statistics_diagnostic_v1](<C:/xampp/htdocs/YEAR 4/Testing/outputs/dgp_eval_statistics_diagnostic_v1/saved_output_audit.json>).
Original plans, outputs, failed decisions and earlier app/pilot records are
preserved. The three prior mutable milestone documents are snapshotted in
`outputs/dgp_processing_diagnostics_milestone_v3/before_docs/`; all 22 bindings
in the prior app integration record remain exact. Independent human final
review, useful native outputs and full automatic/assisted family acceptance
remain incomplete. Manual transfer files and pasteable VM commands only remain
the authorized execution path.
