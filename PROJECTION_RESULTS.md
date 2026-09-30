# Projection comparison results — 30 September 2026

**Neither arm qualifies for promotion at any epoch.** The fixed one-sided
projection modestly improves synthetic retention but reduces real-mask overlap;
it does not solve the existing output-quality problem. No unchanged rerun.

| Final epoch10 metric | Ordinary | Projected | Original baseline |
| --- | ---: | ---: | ---: |
| Real IoU | 0.32442 | 0.31973 | 0.06065 |
| Human-only IoU | 0.34495 | 0.34024 | 0.06429 |
| Synthetic IoU | 0.95082 | 0.95268 | 0.97469 |
| Synthetic missed fraction | 0.038027 | 0.036927 | 0.016476 |
| Synthetic visible FP | 0.001725 | 0.001604 | 0.001333 |
| Glare IoU | 0 | 0 | 0 |

Projection activated on113/210 updates. Reported first-order replay-ascent steps
decreased from95/210 ordinary to71/210 projected. Of the113 projected updates,
36 still have an actual AdamW parameter step opposing the current replay gradient.
This confirms that the implemented raw-gradient constraint does not constrain
the optimizer's actual step. It does not prove those individual steps increased
finite-step replay loss or caused held-out retention failure.

## Verification

Archive108,001,754 bytes, SHA256
`c815e4aa5d825fa41ab9a21e07ea61adde9109275f4fa3ab5c95cc3b4cf97ceb`.
8,593 archive entries safely extracted to `outputs/downloaded_projection/`.
Executed script/helper hashes match local versions. Full inventory, protocol,
replay hash and initial-state hash checked; initial tensors equal the parent.
All420 logged batches match the fixed schedules. Gradient decomposition errors
are<=1.640e-6. Model-generator tensors unchanged in all20 checkpoints.

All8,500 saved masks recounted; aggregate, human/mannequin/glare subgroup metrics
and every selection decision agree. Ordinary final aggregate metrics match the
prior extended coverage arm; intermediate metrics are not all bit-identical.
The prior run is context, not the matched treatment control. Ten-row preview inspected:
fragmented surgical masks, dark-patterned omissions and background predictions
remain in both arms. No visual completion gain established.

Final checkpoint SHA256s:
ordinary `3aee99fa9d5219705bb8510ae47755adb07ceb4af77a1c28679ecce8ae216078`;
projected `d81b917ca1a1abc217a2dcc6bb0556bff5ce4bf5219bfef1291cb745faa071c7`.
Step alignment numbers are verified-script VM measurements; full per-step
gradient/parameter vectors were not returned. This turn did not reproduce final
checkpoint inference; saved masks and checkpoint state invariants were checked.

Evidence: `outputs/projection_validation/results.json`, `preview.png`.
Recount script: `scripts/evaluate_projection_results.py`.

## Next decision

Do not commission another scalar/learning-rate/projection sweep. Review whether
an optimizer-step constraint is justified versus separating domain adaptation
from the retained detector, accounting for already failed feature-head experiments.
First quantify the magnitude (not merely sign) of logged replay-ascent steps and
how much real gradient was removed. This uses existing artifacts, zero training,
and can reject a negligible-effect explanation before any new VM proposal.
All original gates stay fixed; current application/generator baseline unchanged.

Local root: `C:\xampp\htdocs\YEAR 4\Testing\`.
VM root: `~/forensic-dgp/coverage_vm_bundle/outputs/projection_training_vm/`.
