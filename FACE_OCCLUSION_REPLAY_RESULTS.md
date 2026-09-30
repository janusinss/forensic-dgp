# Direct occlusion replay fit — 30 September 2026

The pretrained detector lacks sufficient fit on the synthetic replay examples it
trained on. Validation failure is not confined to new benchmark images. This
supports a bounded additional-fitting test; it does not prove that more updates
will satisfy the original safeguards or fix glare.

| Detector | Training replay IoU (638 unique cases) | Validation IoU (400 cases) | Training missed fraction | Training clear false-positive cases |
| --- | ---: | ---: | ---: | ---: |
| Original completion parent | 0.96904 | 0.97469 | 0.01582 | 2/262 |
| Random initialization, epoch10 | 0.40017 | 0.40058 | 0.56572 | 262/262 |
| Pretrained initialization, epoch10 | 0.84827 | 0.84378 | 0.09736 | 4/262 |

Exact schedule-exposure weighting gives pretrained training IoU0.84747 versus
parent0.96973, with840 replay exposures:420 covered and420 clear. Those are
repeated exposures of638 unique cases, not840 different images. Weighting does
not remove the fit deficit.

## Error strata

All four covered kinds fall behind the parent on both cached training and
validation. Degraded irregular coverings are weakest; eye-band and object
predictions also spill into visible pixels, including otherwise clear examples.
The fixed0.5 threshold and original gates remain unchanged.

| Kind | Pretrained clean training IoU | Pretrained degraded training IoU | Pretrained clean validation IoU | Pretrained degraded validation IoU |
| --- | ---: | ---: | ---: | ---: |
| Lower face | 0.94965 | 0.83972 | 0.93944 | 0.85339 |
| Eyes | 0.90430 | 0.84880 | 0.87573 | 0.85270 |
| Object | 0.89136 | 0.84691 | 0.87545 | 0.85457 |
| Irregular | 0.83704 | 0.73761 | 0.80871 | 0.71069 |

Source-stratified training IoU is0.85968 for the Asian source pool and0.83644
for FFHQ; corresponding validation IoU is0.85782 and0.82944. These name the
dataset pools, not inferred ethnicity of an individual. Both pools have a fit
deficit. Training uses321 Asian-source cases and317 FFHQ-source cases; validation
has200 from each pool.

The diagnostic ten-row grid contains prediction-ranked **training** cases:
lowest IoU for each covered kind/degradation pair, most false pixels for clear
pairs, lowest case ID for ties. It is explicitly a failure sample, not a
representative average or independent validation. Visual inspection shows
rounded rectangle boundaries, thin stroke omissions, an irregular triangle with
large missing areas, and false hat/goggle regions. Clear glasses/headwear that
do not hide facial regions are negative examples under the existing labels.
No examples or labels were changed or moved into training from validation/test.

## Evidence and scope

`scripts/diagnose_face_occlusion_replay.py` verifies frozen cache/protocol hashes,
the audited returned checkpoints/masks and original benchmark targets. It
reuses previously measured parent confusion counts from
`outputs/replay_fit_comparison/results.json` after checking cache/checkpoint
identity, exact case/metadata membership, target pixel counts and re-aggregation.
Parent inference is not repeated. Source report hash is recorded in the new
output. Both trained detectors are inferred on all638 cached cases on CPU in
batches of8; their states remain unchanged. Zero optimizer updates.

Three tests verify pixel aggregation rather than mean image IoU, source strata,
schedule-exposure weights and duplicate/invalid weight rejection. All pass.
Evidence:`outputs/face_occlusion_replay_fit/results.json` and `preview.png`.
Mask files are retained for both detectors under `train_masks/`.
The previous CPU/VM comparison recorded five thresholded pixel differences in
other inference configurations; no cross-runtime bitwise identity is assumed.
FaceExtraction FFHQ pretraining overlap remains unresolved. These are reused
development data and cannot establish final generalization.

Local root:`C:\xampp\htdocs\YEAR 4\Testing\`.
Underlying VM source:`~/forensic-dgp/coverage_vm_bundle/outputs/face_occlusion_pilot_vm/`.
Training cache local:
`outputs/downloaded_coverage/outputs/coverage_training_vm/replay_pixels.pth`;
VM:`outputs/coverage_training_vm/replay_pixels.pth`.

## Fixed next intervention

Prepare one additional-fitting experiment from the verified pretrained epoch10
state, with20 additional epochs/420 updates. Keep architecture, original source
membership, masks, losses, learning rates, class balance and selection gates.
Use a fresh AdamW because optimizer state was not saved; describe this as a
weight continuation with an optimizer reset, not exact training-state resumption.
Pre-register candidate checks and a hard stopping point before VM execution.

This directly tests whether the current representation can improve replay fit
without discarding its real-mask gain. Do not change architecture, add a new
loss, relax gates or start a sweep in the same experiment. If it fails, inspect
the resulting learning/precision evidence before a different intervention.
Strong glare remains an explicit unresolved requirement. Generator/restoration
baseline and application behavior remain unchanged; the goal remains open.
