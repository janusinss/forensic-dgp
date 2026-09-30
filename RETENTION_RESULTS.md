# Retention pilot result — 30 September 2026

Decision: reject this candidate for application use. No identical rerun, added
epochs or relaxed gates is justified by the returned result.

The VM completed21 scheduled batches with18 accepted updates across42 trials.
Independent log audit matches every batch to the frozen schedule, verifies trial
order, separate class ceilings, counters and stop reason. Executed source matches
the sent bundle. All425 saved masks were independently recounted, including
human/mannequin/glare subsets, and both selection failures were reproduced.

| Metric | Parent | Retained final |
| --- | ---: | ---: |
| Real mask IoU | 0.060648 | 0.058751 |
| Real missed fraction | 0.931603 | 0.935093 |
| Real visible false positives | 0.018086 | 0.014833 |
| Synthetic IoU | 0.974690 | 0.974644 |
| Synthetic missed fraction | 0.016476 | 0.017281 |

Human-only IoU also falls0.064293 ->0.062615; mannequin IoU falls0.020287
->0.016914. The single glare validation case remains empty. Six of15 covered
real cases remain empty; one of10 clear real cases has false positives.
Synthetic clear false positives remain0/80 and empty covered cases remain1/320.

Initial tensors exactly equal the pinned parent. All generator tensors remain
unchanged,50 segmenter tensors differ, and final tensors are finite. The ten-row
preview was inspected: surgical/respirator coverings are often absent or reduced
to fragments; patterned masks remain largely missed. This is not improved
completion input. No completion output or model promotion was performed.

## Interpretation and next step

Constraining training replay loss permits some updates and keeps synthetic IoU
close to the parent, but does not deliver useful real-mask validation learning.
It also does not guarantee the original synthetic safeguards: missed fraction
worsens despite satisfying logged training-loss ceilings. The original guards
remain intact. The small IoU difference is not claimed statistically significant;
the candidate nevertheless lacks the required evidence for promotion.

Next: reproduce final-checkpoint predictions on the25 real validation images
and measure real **training** fit against the parent, using inference only. This
will distinguish lack of training adaptation from poor transfer in this bounded
pilot before selecting any further intervention. Do not infer that outcome from
mixed loss on different batches. No new VM training recipe is approved by this
result; a longer constrained run is not the default next step.

Limitations: checkpoint-to-mask inference and final replay loss have not yet been
independently reproduced. Trial losses are log-verified, not recomputed from
intermediate states; intermediate optimizer states were not archived. Scalar
rollback tests verify helper behavior, not every historical VM update.

## Evidence

Local root: `C:\xampp\htdocs\YEAR 4\Testing\`.
VM root: `~/forensic-dgp/coverage_vm_bundle/`.
VM outputs: `outputs/retention_training_vm/`; downloaded local copy:
`outputs/downloaded_retention/outputs/retention_training_vm/`.

Archive SHA256:
`e3411d7ff196060e448f9aa6e59e703c759b439105e32eb8f6af2db5f2e2f923`.
Final checkpoint SHA256:
`6ecbd889ebbee97d2dad54defa04acdab131ba558b80e4c3669c1c8c50a3e727`.

Independent evidence: `outputs/retention_validation/results.json` and
`outputs/retention_validation/preview.png`.
Recount/state script: `scripts/evaluate_retention_results.py` (refuses existing
output directory). Log audit: `scripts/audit_retention_logs.py` (read only).
