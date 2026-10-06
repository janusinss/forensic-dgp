# V20: fixed face-prior RGB difference fails motion preservation

**5 October 2026: close this exact recipe before constructing a learned
conditioner or preparing a VM training pilot.** A ten-case training-only,
clean-target-informed feasibility intervention completes in 61.30 seconds.
An independent 1.99-second saved-output audit reproduces four predeclared
preservation failures. Both original-resolution sheets are reviewed; motion
outputs have conspicuous colour and texture artifacts. No model is adopted.

## Why this inference diagnostic was run

The native ChokePoint comparison in `CCTV_NATIVE_SOURCE_EXTENSION_V1.md`
shows retained DGP smoothing already visible structure. Earlier learned RGB
corrections improve photographic synthetic pixel error but fail appearance
preservation. Inference normalization and fixed completion-context changes
also fail their stated processing hypotheses. Preserve every earlier failure,
checkpoint, source, split and gate.

This separate hypothesis asks whether expressing a correction through a fixed
face generator, while anchoring the output to retained DGP pixels, can retain
structure. It changes neither normalization nor learned weights. Before spending
VM training on a conditioner, the diagnostic supplies the clean training target
to the prior and checks the resulting prescribed difference. The question is
whether this particular target-code correction is feasible, not whether all
possible latent-space architectures are impossible.

## Frozen scope and declared prior

Use the first fixed training-preview reference from each provenance source in
the immutable V16 r2 parent: `tr_asian_00048` and `tr_ffhq_00084`. Both have
training roles and five existing synthetic profiles: clear, blur_lr24,
lowlight_lr32, motion_lr48 and compound_lr24. Ten cases total; no new data
generation, validation, native CCTV or reserved identity use. Source folders
are provenance groups, not ethnicity labels. Targets are the inherited
photographic proxies, not native CCTV clean ground truth.

The declared prior is the official clean-image VQGAN teacher, acquired earlier
from CodeFormer's official release. Teacher checkpoint SHA256:
`4d1c6741b3cffcbdc2cd1a12b2c3c2442282e042d5de66909cb643d4fa31b20f`.
S-Lab License 1.0 remains with the pinned source. Encoder, nearest-vector
quantizer and full RGB generator stay frozen. No CodeFormer restoration
transformer, fitted code head, input-feature fidelity fusion or ADAIN is used.
No pretrained module is relabeled as our learned contribution.

Every arm uses the same 256×256 camera input. The retained DGP uses canonical
NumPy float32/255 normalization and unchanged stored evaluation statistics.
Teacher input is bilinear 512 normalized to [−1,1]; generator output is clamped
and bilinearly returned to 256. Let `G(Q(E(image)))` denote that fixed VQ
reconstruction. The prospective composition is:

```text
oracle = clamp(DGP(camera) + VQ(clean_target) - VQ(camera), 0, 1)
```

The subtraction is evaluated before adding to DGP. Raw float32 arrays are saved
separately from floor-quantized PNGs; only unsupported padding is restored in
delivery. There is no sharpening, display colour correction, output-specific
coefficient or fit. The clean target is unavailable to real restoration, so
this oracle cannot count as a learned output or demonstrate native usefulness.
It is not a mathematically optimal upper bound over every possible conditioner.

Five saved arms are resize, retained DGP, input VQ, clean-target VQ oracle and
the DGP-anchored prior-difference oracle. The input VQ images expose the textures
being subtracted; the target VQ images expose representation loss and estimated
appearance even when the clean photograph is available.

## Prospective stops, actual execution and independent audit

Before any inference, the plan requires exact raw/PNG DGP parity on both clear
controls and no MSE or valid-window SSIM loss against DGP on **each** degraded
case (tolerances 1e-12 and 1e-6). Any failure stops before a conditioner or VM
pilot. No motion exception or revised coefficient is introduced after review.

CPU inference uses four threads, a 240-second worker cap and 300-second external
timeout. Actual 61.30 seconds includes loading and sheets: ten DGP, twelve teacher
encoder, twelve quantizer and twelve generator calls. All reported DGP/teacher
state hashes remain unchanged. Zero optimizer construction, backwards or updates.

The independent auditor imports no neural model. It verifies 39 source/106 output
bindings, independently reconstructs ten exact prior-difference arrays, forty
raw-to-PNG compositions and fifty paired metric rows. Both clear raw/PNG controls
are exact DGP. Every cell in the two original 1340×1504 sheets matches saved PNGs.
This audits saved arithmetic and reported counts/states; no neural or CUDA replay.

Both motion cases fail MSE and SSIM preservation: four failures total. The other
six degraded cases pass those numeric guards under the clean-target intervention.
That success depends on unavailable target information and is not a restoration
result or a reason to waive the failed motion guards.

| Paired synthetic profile, two training cases each | DGP PSNR dB / SSIM | Target-informed difference PSNR dB / SSIM |
| --- | ---: | ---: |
| Clear | 30.709 / 0.92311 | 30.709 / 0.92311; exact DGP |
| Blur_lr24 | 21.151 / 0.62235 | 22.348 / 0.67668 |
| Lowlight_lr32 | 15.270 / 0.60017 | 20.831 / 0.68476 |
| Motion_lr48 | 23.782 / 0.71450 | 22.490 / 0.66459; both cases fail |
| Compound_lr24 | 12.544 / 0.47968 | 19.752 / 0.68806 |

PSNR is derived from mean observed MSE; SSIM averages valid observed windows.
These are pooled two-photograph training proxies, separate from unpaired native
CCTV evidence. No recognizer/identity metric or independent final review occurs.

## Original-resolution visual findings and decision

Both sheets are actually viewed at 1340×1504. Clear oracle outputs are exactly
DGP. Target VQ reconstruction itself changes some eye/lip/expression detail.
On blur, lowlight and compound inputs, target-informed differences add clearer
face/glasses features but leave patchy highlights, skin and lip mismatch. Motion
produces conspicuous mottling and green/pale artifacts around eyes, nose and
mouth on both sources. Subtracting an input-prior reconstruction with different
estimated appearance exposes these mismatches; that is a mechanism suggested by
the saved decomposition, not proof that all generative priors fail.

No conditioner, trainer, VM transfer archive or launch command is created.
Do not repeat this exact recipe, add output-based profile exceptions or relax its
guards. Keep the current app, all historical checkpoints/splits and both reserved
native identity sets intact. Future training requires a separately justified
recipe that addresses the demonstrated structure/appearance limitation.

Under `AGENTS.md`'s three-attempt circuit breaker, model-path changes stop while
one diagnostic user review clarifies whether the shown native DGP softness is
useful. That feedback cannot rewrite the frozen experiment's failed gates.
The app remains functionally integrated under earlier regressions/Playwright;
this diagnostic makes no new browser-test claim. Useful native own-DGP output,
automatic/assisted covering-family acceptance and independent final review remain
incomplete. Goal active; all actual training stays manual on the existing L4 VM.

Evidence: `outputs/cctv_dgp_latent_delta_feasibility_v20/plan.json`, `results.json`,
`independent_saved_output_audit.json`, `visual_review.json`, raw arrays, PNGs and
two sheets. Reproducible frozen source:
`scripts/check_cctv_dgp_latent_delta_feasibility_v20.py` and
`scripts/audit_cctv_dgp_latent_delta_feasibility_v20.py`.
