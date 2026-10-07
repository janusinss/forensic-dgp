# V28 final-state diagnostic and the V29 hypothesis

The downloaded diagnostic is complete: 100 VM component-gradient queries,
zero optimizer updates, zero backwards and no new checkpoint. Its archive is
211,650,756 bytes with SHA256
`7ed57b598096e2bf52302b5acba45626a12ec66f17ca3d94f565e704d170cc75`.
The separate R1 independent audit verifies 322 regular files, all ten gradient
matrices and their aggregate (54,848,970 values), parameter displacement,
component norms, Gram matrices, cosines and per-tensor statistics. CPU replay
uses 50 original-DGP, 50 final-DGP and 100 recognizer forwards. All 50 fresh VM
initial/final raw and embedding comparisons have maximum error zero; final
delivered PNGs match exactly. Original, final and recognizer states remain fixed.

The original prospective local audit stopped at its exact float64 norm equality:
two Linux/Windows norms differed by one ULP, at most 5.55e-17. Its source and
failure remain in `outputs/cctv_dgp_v28_preservation_diagnostic_v1_audit_repair_r1`.
The distinct repair auditor uses relative tolerance 1e-12 and absolute 1e-14
only for derived arithmetic. Saved gradient arrays, parameter displacement,
archive contents and delivered PNGs retain their exact checks. Four regression
tests reject meaningful/nonfinite/shape changes and retain those exact checks.
No model-quality threshold changes.

At the measured final V28 state, the original seven-term objective has gradient
norm 1.476055252. The improvement and preservation gradient cosine is
-0.246421193. The derivative of the diagnostic RGB-mean penalty along the plain
negative original-objective gradient is +1.998307315. Along that same direction,
clear SSIM and ArcFace hinge penalties decrease (-0.104978540 and -0.133036285).
Preservation gradients are present and active; this does not support claiming
they were absent or that all weights were simply too small. These measurements
are local first-order evidence, not a reconstruction of AdamW's trajectory or
a unique historical cause.

V28 remains rejected: its final training landmark-HF gain was 18.0595%, but it
failed clear-source SSIM, clear-source ArcFace and the brightness-fraction gate.
The fixed post-training RGB-mean control retained 18.0237% structure gain but
failed five preservation gates. Neither is adopted. The diagnostic itself is
not a restoration-quality pass.

The single V29 hypothesis is that removing the RGB-mean delta **inside** the
decoder optimization allows its spatial weights to adapt under that restriction.
For each image, the same-input frozen original DGP provides the baseline. The
candidate-minus-baseline observed RGB mean is subtracted before unchanged losses
and final clamping. All inputs use the same path; no clean target, profile label,
source or identity enters inference. The centering has no learned parameters,
strength setting or surrogate derivative. Clipping can reintroduce mean shift,
so the original delivered-output brightness gate still decides acceptance.

This is one architecture-path change, starting from the original checkpoint.
It keeps the selected 12 decoder tensors, frozen encoder/FPN/inactive head4 and
normalization buffers, original seven loss weights, 800-update schedule, AdamW
settings, 1% early structure stop and all final structure/source/preservation
gates. It must pass fresh initial output/gradient proof before any optimizer.
No acceptance of an earlier checkpoint, unchanged historical rerun or automatic
follow-on is allowed. A standalone candidate checkpoint does not include this
baseline-conditioned projection; later deployment would require both original
and learned DGP plus the fixed path.

Local forward-only projection and return-safety regressions pass. Two local
unissued test setup failures are retained: the VM guard's wrong arity and a
float32 analytic-equality assumption; then a copied test's incorrect auditor
filename. These did not start VM work or query local derivatives. The transfer
packet must be independently verified before release; actual V29 training is
pending human execution through the steps in
[CCTV_DGP_MEAN_CENTERED_DECODER_V29_VM.md](CCTV_DGP_MEAN_CENTERED_DECODER_V29_VM.md).

All 50 cases are paired photographic TRAIN capacity evidence, with source
labels reported separately. Dataset names do not establish ethnicity or native
CCTV capture. No new native or reserved-final pixels are opened. DGP native
usefulness, all seven automatic/assisted covering families, full application
verification and independent final review remain required. No Zamboanga or
exact hidden-identity claim; the overall goal remains active and incomplete.
