# Pretrained continuation results — 1 October 2026

The VM pilot completed420 additional updates, reaching630 cumulative model
updates. Synthetic mask accuracy improves, but all four candidates still fail
the original retention guard. No `best_detector.pth` is selected. End-to-end
restoration/completion improvement is not established and the application,
completion generator and Phase3 restoration baseline remain unchanged.

## Outcome on the fixed validation sets

| Global epoch | Cumulative updates | Real IoU (25 cases) | Synthetic IoU (400 cases) | Real/synthetic gates |
| --- | ---: | ---: | ---: | --- |
| Source10 |210|0.82283|0.84378|Previously ineligible |
|11|231|0.80428|0.84662|Pass/fail |
|15|315|0.83367|0.88010|Pass/fail |
|20|420|0.79326|0.89038|Pass/fail |
|30|630|0.81739|0.92039|Pass/fail |

The real gate compares against the original common parent's IoU0.06065, visible
false positives and empty/clear-case counts. Passing it does not imply improvement
over source10. Epoch30 real IoU is slightly below its starting source; human-only
IoU is0.81997 and the known mannequin is reported separately at0.79113. Real
clear false-positive cases remain0/10. These25 reused cases cannot establish
generalization or statistical significance.

The synthetic gate still requires the original parent:IoU at least0.97469,
missed fraction at most0.01648, visible false-positive fraction at most0.001333,
at most1 empty covered case and0 false-positive clear cases. Epoch30 achieves
0.92039/0.06648/0.002098 respectively, with0 empty covered cases and1/80 clear
false-positive cases. Selection rules and threshold0.5 were unchanged.

## Return verification

The373,872,188-byte archive SHA256 is
`035777e5aa22ca97c12b07ce96fa2abb076eea5806c408dd7f8d6d6f2bf84028`.
Executed code and source inventories match the sent package. Every logged batch
matches the fixed420-step order. Both update counters, candidate losses and
selection decisions were reconstructed. All2,915 saved masks were independently
recounted against verified targets.

Strict CPU loading verifies all four checkpoint states and preserved metadata.
Each has92 frozen reference-head/BatchNorm tensors unchanged from source10;
60 encoder,30 decoder and2 new-head tensors change. Source bytes are exact.
The final optimizer snapshot binds to epoch30 SHA256
`9ba74a30719f01bfafd7ee9c060dc090c507c26eeeecb117b4cb342a8bf4df42`.
All92 parameter states covering14,328,209 elements have finite moments, correct
shapes/settings and fresh step420. This was weight continuation with a fresh
optimizer; the original210-update optimizer moments were unavailable.

CPU inference compares all2,915 returned masks. Three pixels differ:one in the
copied source's training prediction, one in epoch15 synthetic validation and one
in epoch15 real training. All epoch30 saved masks match exactly. Independent CPU
selection agrees with the VM for every candidate. The cause of the three runtime
differences was not established. No local optimizer updates were performed.
The original parent/generator's remote invariance remains an executed-code
assertion/log claim because its VM tensors were not returned.

NVIDIA L4/PyTorch2.9.1+cu129 reports99.16 seconds for the run, excluding archive
compression, with1,047,887,872 peak allocated CUDA bytes and1,130,364,928 reserved.
The ten-row validation grid was inspected, including clear faces, difficult cloth
covering, the mannequin and the missed lens-reflection case.

## Fit diagnostic

The final detector was inferred on all638 pinned cached training replay cases
on CPU in batches of8; state tensors remained unchanged. Original parent and
source10 confusion counts were reused after source/cache/membership/target-count
and re-aggregation checks. Their unchanged models were not reinferred on all638.

| State | Unique training replay IoU | Synthetic validation IoU | Training missed fraction |
| --- | ---: | ---: | ---: |
| Original parent |0.96904|0.97469|0.01582 |
| Source10 |0.84827|0.84378|0.09736 |
| Epoch30 |0.93217|0.92039|0.05735 |

Schedule-exposure weighting gives epoch30 IoU0.93427 over840 repeated replay
exposures, representing638 unique cases. The fit deficit persists on seen cases;
it is not exclusively a validation-transfer problem. This does not prove the
cause is capacity, loss weighting or insufficient updates.

Degraded irregular coverings remain weakest:training IoU0.84717 and validation
0.81573. Source-pool training/validation IoU is0.93794/0.93291 for Asian-source
cases and0.92618/0.90771 for FFHQ-source cases. Pool names do not infer ethnicity.
Real training IoU rises to0.91719 while real validation is0.81739.
Combined epoch loss falls from0.25990 at epoch11 to0.07460 at epoch30;
separate supervised/teacher terms were not logged, so their individual trends
cannot be inferred.

The same ten source10-ranked training failure cases were held fixed for a
before/after grid. Rectangular boundaries and the hat false positives improve;
the degraded triangle still contains large missing regions and a goggles region
is predicted in a procedural clear target. Final training clear errors are1590
and1765; validation clear error90 contains409 false pixels. Native sources
`69560.png`,`11915.png`,`08368.png` were inspected:the goggles sit above visible
eyes, the hat shades the face, and a flower is beside the cheek. Anatomical
context and annotation scope deserve attention; no source was relabeled or
moved across splits from these error inspections.

## Reflection pixels, measured separately

Whole-mask IoU on the two glare-containing training images is0.84064. One also
contains an11,381-pixel medical mask, which dominates that aggregate. Reflection
fit is therefore measured using the previously accepted V3-minus-V2 additions.
The V2 parent hash, identical source images/splits and preservation of every old
masked pixel were verified. Mouth-mask pixels cannot count as reflection recovery.

| Case | Reviewed reflection pixels | Source10 recall | Epoch30 recall | Cumulative training exposures at30 |
| --- | ---: | ---: | ---: | ---: |
| Train15:uncovered_05.png |1345|52.64%|60.97%|27 |
| Train46:new_covered_48.png |686|0.00%|56.56%|30 |
| Validation23:new_uncovered_16.png |491|0.00%|0.00%|0 |

Training reflection recall increases from34.86% to59.48% when weighted by the
2,031 added pixels. The single validation example remains completely missed at
every checkpoint. This is evidence of incomplete training fit and failure on
that particular validation case; one case cannot estimate population transfer.
Visible false pixels are reported separately. Masks are approximate assistant
polygons accepted by the user, without independent expert adjudication.
No test predictions were scored in these diagnostics.

The reflection-only zoom grid was inspected. Two new diagnostic test modules
provide six passing tests for pixel/area aggregation, the1% target-area boundary,
invalid counts and exclusion of the original mouth mask from glare recovery.
The preceding return checker has17 passing tests (25 with existing regressions).

## Evidence locations and next action

Local root:`C:\xampp\htdocs\YEAR 4\Testing\`.
VM root:`~/forensic-dgp/coverage_vm_bundle/`.
Returned archive:
`C:\xampp\htdocs\YEAR 4\Testing\outputs\face-occlusion-continuation-results.tar.gz`;
VM archive:
`/home/janusdominic0/forensic-dgp/coverage_vm_bundle/face-occlusion-continuation-results.tar.gz`.

Local extracted model root:
`C:\xampp\htdocs\YEAR 4\Testing\outputs\downloaded_face_occlusion_continuation\outputs\face_occlusion_continuation_vm\`.
VM model root:`~/forensic-dgp/coverage_vm_bundle/outputs/face_occlusion_continuation_vm/`.
Local audit:`outputs/face_occlusion_continuation_validation/{results,reproduction,members}.json`.
Local fit:`outputs/face_occlusion_continuation_fit/results.json`,
`synthetic_preview.png`,`glare_preview.png`,`clear_error_preview.png`.
Local reflection audit:`outputs/face_occlusion_continuation_glare/results.json`
and`preview.png`. These diagnostic outputs were produced locally, not on the VM.

Checkers:`scripts/audit_face_occlusion_continuation_results.py`,
`scripts/diagnose_face_occlusion_continuation_fit.py`,
`scripts/audit_face_occlusion_glare_pixels.py`.
The VM runner/specification/package remain immutable and previously completed;
its default output cannot be rerun into the existing directory.

Follow-up: one matched, bounded VM intervention targets underfitted small
reflection regions and degraded irregular boundaries while retaining clear-face
specificity. The verified final model/optimizer give equal starting conditions.
The existing real/synthetic guards, source membership and held-out exclusions
remain fixed. Its single change, matched control, budget, checks and stopping
point are registered before execution.
The follow-up is now prepared as the matched region-weighting pilot in
`FACE_OCCLUSION_FOCUS.md`; `FACE_OCCLUSION_FOCUS_VM.md` has exact VM commands.
It starts both arms from verified epoch30 weights and saved optimizer moments,
with210 new updates per arm and unchanged gates. Local tests, source forward and
all training-only maps pass; new CUDA execution/output quality remain unverified.
Do not promote a checkpoint from training fit.

Only an eligible detector advances to reviewed end-to-end restoration/completion.
Track1 retains`checkpoints/dgp_zamboanga_final.pth`; full Phase5 identity training
is pending. External FaceExtraction FFHQ overlap and identity separation remain
unverified, and development cases have been repeatedly inspected. Hidden facial
features remain plausible estimates. The overall improvement goal remains open.
