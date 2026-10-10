# DGP learning review after the independently audited V32 r2 stop

The selected approach diagnoses the current learning before another model change.
The user's exact instruction was "apply the best approach and do research also if
needed". The required three-attempt discussion is satisfied. No failed gate is
waived, and actual training remains the user's manual existing-L4 workflow.

V32 r2's complete3,905-case delivered-PNG snapshot improved degraded structure by
0.9706721697%, below the unchanged1% early requirement. All17 group-average
preservation checks passed. All50 previews were reviewed across eyes, nose,
mouth, outline and visible appearance; useful additional whole-face clarity was
not established. The stop, original checkpoints, source receipts and splits stay
binding. There is no800-update result or application promotion.

The additional loss review uses only the already frozen50 TRAIN previews:
10 clear controls and40 synthetic degradations. None was optimized by update50,
but all were used in preflight/normalization and are not unseen evaluation.
The source groups are dataset/asian_faces and dataset/thumbnails128x128. They
do not establish ethnicity, native CCTV or Zamboanga performance. No new native
or reserved-final pixels were opened.

| Saved50-case observation | Original | Stopped50 | Implication |
| --- | ---: | ---: | --- |
| Degraded raw feature-error reduction | — | 0.968684% | Weak before PNG rounding |
| Degraded delivered-PNG reduction | — | 0.991558% | Rounding does not explain the weak preview gain |
| Mean total raw objective, all50 | 1.2999999802 | 1.3002432080 | Does not improve on this fixed TRAIN cohort |
| Reduction in the three restoration terms | — | 0.0121946314 | A small improvement |
| Added preservation-term value | 0 | 0.0124378591 | Slightly exceeds that restoration-term reduction |

Loss values are not gradient magnitudes. This comparison does not prove that
preservation gradients dominate an AdamW step or identify a unique cause.
It shows a measured tradeoff that the initial all-zero preservation gradients
cannot describe. Zero initial values follow exact initial baseline equality.

At stopped50, the raw identity penalty activates on8/40 degraded previews and
1/10 clear controls. The clear baseline anchor activates on all10 clear cases.
One clear case, v9_tr_asian_00180_clear, has3.802615% more raw observed-pixel MSE.
No raw SSIM penalty activates on these50 cases. Per-case regressions can coexist
with the passing group-average delivered gates; those original gates remain
unchanged, and the measured per-case protection is retained.

The largest preview raw fixed-recognizer cosine decrease is0.06551696 on
v9_tr_ffhq_01210_compound_lr24; its PNG cosine also decreases0.06725246.
This is a similarity measurement to a paired photographic TRAIN target, not
evidence of recovered identity. The corresponding degraded input was already
reviewed as needing clearer information for useful fine structure.

Regional raw high-pass reductions over the40 degraded previews remain uneven:

| Fixed target-landmark area | Relative error reduction |
| --- | ---: |
| Eye0 / eye1 patches | 1.337294% / 0.891919% |
| Nose patch | 0.460435% |
| Mouth0 / mouth1 patches | 0.615022% / 1.101575% |
| Observed interior outside the five patches | 0.476032% |

The last area is not an anatomical outline annotation. The qualitative review
still includes the face outline and all visible appearance. These small numerical
gains do not override the soft results. The mean raw change on degraded previews
is0.0075848322 in RGB[0,1]. Their mean final boundary-channel fraction is0.080401%.
That endpoint fraction does not reveal internal tanh/clamp saturation or gradient
availability. The forward path has both original residual clipping and the
subsequent mean-centering/clipping constraint; changing it requires evidence.

The forward-only local review makes100 frozen recognizer image calls and zero
DGP forwards, gradient calls, backwards or optimizer updates. It finishes in
45.94s against its180s cap. A separate40.53s checker verifies387 source bindings,
all50 cases,400 feature/region quantities and100 raw recognizer images. It uses
an independent float64 filter/SSIM calculation and loss-term assembly. Maximum
differences are7.41e-9 pixel MSE,9.82e-10 high-pass MSE,1.80e-5 float64 SSIM,
zero recognizer cosine difference and2.17e-7 term value. Float64/float32 readback
tolerances do not change any VM or PNG quality gate.

The first preparation used the wrong metadata field, reference, instead of
source_person_or_reference. Its plan and failure receipt remain intact. A
separate pinned R1 review corrects only that lookup and evidence/output routing.
It does not change the original model, objective or training data.

Primary research helps constrain the next investigation. Rahaman et al. describe
frequency-dependent learning in their studied networks; that supports measuring
fine structure but does not prove the cause here.
[On the Spectral Bias of Neural Networks, ICML2019](https://proceedings.mlr.press/v97/rahaman19a.html).
Jiang et al. use a complex-spectrum distance with detached adaptive weights to
emphasize harder frequencies. The current model already rewards a fixed luma
high-pass error; substituting focal frequency loss remains an untested hypothesis.
[Focal Frequency Loss, ICCV2021](https://arxiv.org/html/2012.12821v3).

AdamW combines per-coordinate moments with separate weight decay. Endpoint
weights and initial gradients cannot reconstruct its unsaved moment history;
the present saved-gradient/parameter cosine is not a causal diagnosis.
[Decoupled Weight Decay Regularization](https://arxiv.org/abs/1711.05101),
[PyTorch2.9 AdamW algorithm and optimizer state](https://docs.pytorch.org/docs/2.9/generated/torch.optim.AdamW.html).
The author DGP progressively adapts a generative prior for an image, while this
project's own trained reconstructor uses a feedforward MobileNet/FPN residual path.
The names do not establish equivalence or transferable performance claims.
[Pan et al. DGP](https://xingangpan.github.io/projects/DGP.html),
[Kupyn et al. DeblurGAN-v2](https://arxiv.org/abs/1908.03826).

The justified next experiment is a **separate zero-update stopped-V32 gradient
diagnostic**, not another800-update retry or a guessed learning-rate increase.
It measures the seven existing loss gradients at original and stopped states
on two50-case TRAIN cohorts. Both have five references from each photographic
source, each with a clear control and four degradations. The first uses the first
five touched references per source in the frozen first50 batches. The other uses
all50 fixed previews, matched by source/profile, which were not optimized by50.
Selection uses the pre-existing metadata and fixed preview set, not loss rankings.

The distinction from the earlier V31 diagnostic is substantive: the stopped
V32 model has actually changed both fusion and decoder, and the new cohorts
include every fixed preview with the raw-loss findings above. It measures
component conflict, whole-objective directional derivatives, partition magnitudes
and sampled batch coherence. It retains all seven terms, all23 selected tensors,
all frozen state and all1%/10% preservation gates. Gradients guide the next design
decision; they do not qualify useful finite outputs or authorize a continuation.

The independently verified packet is54,322 bytes with three files. It allows
280 finite gradient queries,0 optimizer updates,0 epochs and no checkpoint writer.
Six adversarial metadata/return regressions pass, Python3.10 source syntax passes,
Windows differentiation stops before neural imports, and the read-only Bash
syntax check passes. The first Bash audit was denied by the Windows sandbox's
signal-pipe restriction; the exact successful parser check and a separate R1
readback retain that failure without changing packet code or protocol.

Use an idle existing L4/g2-standard-4 with6GiB free. Worker cap600s, external900s
plus30s grace, export300s with external330s plus30s grace, allocated VRAM20GiB,
uncompressed return1.5GiB. Estimated execution is1–5 minutes plus1–3 minutes export,
based on earlier280-query measurements. Source/state mismatch or nonfinite
values stop and retain evidence. No automatic historical or follow-on launch.

[Five manual upload/install/tmux/launch/download steps](<C:/xampp/htdocs/YEAR 4/Testing/CCTV_DGP_V32_LOSS_GRADIENT_V1_VM.md>)
[Independent packet verification](<C:/xampp/htdocs/YEAR 4/Testing/outputs/cctv_dgp_v32_loss_gradient_v1_preparation/independent_packet_audit.json>)
[Saved losses](<C:/xampp/htdocs/YEAR 4/Testing/outputs/cctv_dgp_post_v32_saved_losses_v1_r1/analysis.json>)
[Independent loss readback](<C:/xampp/htdocs/YEAR 4/Testing/outputs/cctv_dgp_post_v32_saved_losses_v1_r1/independent_audit.json>)

Returned results must be independently audited before selecting another finite
training recipe. A measured conflict may justify a different constrained update
design while preserving the actual visible-face protection. Healthy gradients
with weak finite updates may justify a separately measured optimizer calibration.
An architectural revision remains available if the current path is demonstrably
limiting. None is predetermined by this diagnostic, and neither protection nor
whole-face scope is removed to make a score pass.

The own-DGP application checkpoint, selector,256 processing and design remain
unchanged. The separate converted-MAT comparison remains unqualified; automatic
and assisted seven-family completion still require separate useful-output review.
Original checkpoints, splits, failure records and research caches are preserved.
Useful native restoration and independent final review remain required. Goal
active/incomplete; VM diagnostic not launched by this agent.
