# Multiscale DGP calibration V1: execution completed, quality rejected

10 October 2026. The downloaded packet is hash-verified and independently audited.
All 64 planned saved-PNG comparison sheets were actually inspected. Every one
of the twelve independent trials fails the unchanged quality requirements.
Keep the retained app DGP; do not resume or promote any calibration state.
The complete restoration and seven-family completion goal remains active.

## What ran

The user manually ran the finite diagnostic on the existing NVIDIA L4. Twelve
independent arms each start from the same retained checkpoint and make one Adam
update after balanced accumulation of ten reference batches. Total: 12 optimizer
updates, 120 backwards, 280 component-gradient queries, 600 fitting exposures,
zero complete epochs. Model work took 362.287 seconds, approximately six minutes.
The trainer exited successfully; the experiment's quality outcome is rejection.
Export completion establishes evidence packaging, not model qualification.

The two trainable partitions are the repaired deepest branch (3 tensors/147,456
elements) and the complete original RGB decoder (15 tensors/609,219 elements).
The MobileNet/FPN feature extractor, five stored normalization layers, original
checkpoint and fixed recognizer stay frozen. This is not whole-model training.

All arms evaluate the same 100 historically exposed paired photographic TRAIN
cases from 20 references, covering clear, blur, low light, motion and compound
profiles. These cases are not held-out validation. All 24 frozen ChokePoint C1
native development crops are reported separately as unpaired evidence. They
have no aligned clean face target. Final identity pixels enter neither fitting
nor tuning. Source namespace names do not establish ethnicity, capture country
or performance in Zamboanga City.

The return archive contains 1,639,351,615 bytes; SHA256:
`61cfd317f18b0fd99b6a05a48c03b67c5fbf42e751c6b31cef9c3762de511769`.
Protocol SHA256:
`c0239c2eebab3891cd4f42d34b93a5c7c2b07c8597bf93247df2cc16239dd491`.
The complete GZIP stream passes its integrity check.

## Identical-input results

Structure gain measures the reduction of the frozen landmark high-frequency
error against paired TRAIN targets. It is not an identity-recovery percentage
or a useful-image rating. Raw floats and delivered PNGs remain separate.
Failure counts below are group/metric entries, not counts of faces.

| Partition | Learning rate | Fitting pool | Raw structure gain | PNG structure gain | Raw/PNG preservation failures | Result |
| --- | ---: | ---: | ---: | ---: | ---: | --- |
| deep3 | 1e-05 | 0 | 0.00097542% | 0.00217429% | 9/14 | Fail |
| deep3 | 1e-05 | 1 | 0.00062821% | 0.00101858% | 9/16 | Fail |
| deep3 | 0.0001 | 0 | 0.00960443% | 0.01307640% | 10/14 | Fail |
| deep3 | 0.0001 | 1 | 0.00631960% | 0.01235666% | 9/11 | Fail |
| deep3 | 0.001 | 0 | 0.09611715% | 0.10184920% | 12/13 | Fail |
| deep3 | 0.001 | 1 | 0.06391209% | 0.08468126% | 9/10 | Fail |
| decoder15 | 1e-05 | 0 | 0.03986256% | 0.04341209% | 12/12 | Fail |
| decoder15 | 1e-05 | 1 | 0.03152609% | 0.05348834% | 15/15 | Fail |
| decoder15 | 0.0001 | 0 | 0.39080488% | 0.41326437% | 16/16 | Fail |
| decoder15 | 0.0001 | 1 | 0.29915469% | 0.30585931% | 17/17 | Fail |
| decoder15 | 0.001 | 0 | 1.88430004% | 1.90484848% | 36/36 | Fail |
| decoder15 | 0.001 | 1 | 1.17009119% | 1.16295033% | 37/37 | Fail |

Only the complete decoder at 0.001 exceeds the 1% early structure threshold.
Its clear-image raw pixel error rises from 0.00067381223 to 0.01668709704 in
pool0 (24.765 times) and 0.02124276589 in
pool1 (31.526 times). Clear-image SSIM and
fixed observed-face embedding similarity also worsen. The original 1% early,
10% later capacity, source/profile appearance and 20% brightness-share limits
stay unchanged. A zero brightness-only fraction when overall pixel error
worsens does not demonstrate preserved color or recovered detail.

## Independent audit and retained checker failure

R1 stopped on derived brightness-share arithmetic: 0.9264908100902142 versus
0.9264908100876813, a difference of 2.5328628e-12 above its 2e-12 allowance.
Its original source, traceback and unsuccessful execution receipt remain.
R2 permits 5e-11 only for that derived ratio and its recorded failure-value
field. Pixel, SSIM, embedding, structure thresholds, failure group names and
all Boolean decisions stay unchanged; gates recalculated from stored rows
are checked exactly. Nine negative controls reject changes to scientific
thresholds, preservation metrics, gate decisions and excessive ratio drift.

The successful R2 audit verifies 2,276 manifest files, 1,300 paired raw/PNG
records, 312 unpaired native records, 280 saved gradient vectors and all twelve
full model/optimizer/scheduler/RNG states. Ten fresh CPU inference replays
across two arms differ in raw pixels by at most 2.03e-6. Initial saved-gradient
accumulation and first-Adam arithmetic are independently checked. Local audits
make no new gradients or optimizer updates. This is independently implemented
arithmetic/integrity checking; the goal's independent final reviewer is still
required.

[Independent return receipt](<C:/xampp/htdocs/YEAR 4/Testing/outputs/cctv_dgp_multiscale_calibration_v1_independent_audit_r2.json>).

## Completed visual review

All 40 paired TRAIN and 24 native development sheets were inspected: 64/64,
2,280 unscaled 256×256 RGB cells. A separate gallery audit verifies every
cell byte-for-byte against its saved PNG and binds every page hash. Sixteen
explicit visual batches bind page IDs, case IDs and observations.

The deep branch and lower-rate full decoder add little convincing whole-face
clarity. Severe synthetic profiles remain soft around eyes, noses and mouths.
The highest full-decoder rate produces blue/purple or gray/green washes,
lifted skin/backgrounds and changed highlights, without useful corresponding
facial definition. Clear glasses and ordinary hair generally remain recognizable;
that does not override changed visible appearance or preservation failures.
Native outputs remain soft, with visible tonal changes in the high-rate arms.
No native reference PSNR/SSIM, recovered identity or local-CCTV performance
claim follows.

All 24 native inputs retain their original input-only usable labels. Model
failure cannot relabel a usable input as insufficient. Clearer-crop requests
remain tied to the frozen input review, rather than which treatment looks best.
This implementing-assistant development inspection is distinct from the
independent final visual review required by the full goal.

[Complete 64-sheet visual ledger](<C:/xampp/htdocs/YEAR 4/Testing/outputs/cctv_dgp_multiscale_calibration_v1_return_review_v1/visual_review.json>).

## Demonstrated learning limitation

The saved-step analysis replays 3,360 signed products: each of twelve actual
parameter changes against twenty reference derivatives and fourteen signals.
All 120 pre-update regression-barrier values are exactly zero. Because every
trial starts with candidate output equal to its retained anchor, the ReLU
regression penalty contributes no preservation gradient to that first update.
Actual clear-image failures show that the soft barrier does not protect the
first finite change. This is a demonstrated limitation, not proof of a unique
cause of all earlier failures.

The current negative combined gradient also predicts increasing clear-image
pixel error. Consequently an optimizer-name change alone is not a supported
solution. First-order products do not predict later epochs, finite clipping,
Adam curvature or delivered-image preservation.

[Saved-step arithmetic audit](<C:/xampp/htdocs/YEAR 4/Testing/outputs/cctv_dgp_multiscale_calibration_v1_first_step_analysis/independent_audit.json>).

## Additional diagnostic closure

Four fixed color/detail controls process 400 returned paired TRAIN outputs
without models, new gradients or training. Removing constant RGB differences
retains over 1% filter gain but still gives approximately 3.5–4 times the clear
pixel error and fails preservation. Retaining the baseline low-frequency band
reduces raw filter gain to 0.553–0.874%, with remaining preservation failures.
They do not establish useful corrected images, native performance or a new
restorer. The separate report keeps processing distinct from trained output.

The fixed active-anchor analysis tests the current objective and two added
direct RGB reconstruction terms using only saved derivatives. Clear/blur
fitting-mean slopes become favorable, but some reference-level slopes worsen.
No coefficient search, model assignment or local optimizer occurs. This
supports investigating active preservation supervision in a finite matched
test; it does not select qualified weights or demonstrate an optimal recipe.

## Next action and full-goal boundary

[The next training design](<C:/xampp/htdocs/YEAR 4/Testing/CCTV_DGP_POST_MULTISCALE_TRAINING_REVIEW.md>)
tests active clear/blur reconstruction supervision in a separate copy of the
current DGP's complete original decoder. Keep the failed rates, states,
source roles, exact baseline, full-TRAIN gates and raw/PNG evidence. More
epochs of the rejected objective are not justified by this diagnostic.
Additional epochs 1/2/5 remain conditional on a finite recipe passing
preservation, useful native development review and the original capacity gates.
No new executable VM packet or next training launch is claimed here.

The retained app checkpoint SHA256 remains
`646fbb11674e8882ea41e251d6e6e7d468c963979b48c93132def8292834e24b`;
original Phase3 checkpoint, research caches, splits, provenance and failures
remain. Current app design/model selection is unchanged. All five milestones,
seven automatic/assisted covering families, independent final review and
bundled inline Playwright full-flow verification remain required. No automatic
historical pilot or failed-state resume is permitted. Direct VM access remains
maintenance-only; all actual training stays under manual transfer/tmux workflow.
The latest storage/workload values are historical, not fresh VM availability.
