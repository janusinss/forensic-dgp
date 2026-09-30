# Full real-training fit — 30 September 2026

All73 real training images evaluated using parent/control/extended checkpoints.
CPU inference only; zero optimizer updates. Actual frozen schedule exposures and
per-image masks are recorded. Control did not train on the five additions.

| Training category | Count | Control IoU | Extended IoU | Extended missed fraction |
| --- | ---: | ---: | ---: | ---: |
| Original non-glare masks | 41 | 0.38258 | 0.36032 | 0.60741 |
| Original strong glare | 2 | 0.18118 | 0.19918 | 0.78996 |
| Added opaque lenses | 2 | 0 | 0.000352 | 0.99963 |
| Added profile respirator | 1 | 0.13937 | 0.18969 | 0.79871 |
| Added hand-over-mask | 1 | 0.10236 | 0.14512 | 0.85038 |

The26 clear controls have8 false-positive cases for control and9 for extended
(parent9). Visible false-positive area fractions are0.00924/0.00970, respectively.
Empty target plus empty prediction has undefined per-image IoU (`null`), not
failure. Aggregate zero-union convention is retained only for consistency with
the existing recount function.

Extended exposures: original masks5–10, original glare9–10, opaque lenses7–9,
profile/hand10 each, clear controls13–18. Control's old covered examples receive
9–10 exposures; the extra transparent control receives0. Fixed total real slots
spread across a larger pool, so individual exposure is not matched by design.

## Interpretation

Poor training fit extends beyond the added eyewear: original-mask median
per-image IoU is0.27512 control versus0.25383 extended. This is not solely a
generalization failure on new covering types. The data addition did not resolve
the original segmenter's real-domain fitting problem under this constrained
mixed recipe. It also did not establish that model capacity is insufficient:
short exposure, optimizer settings and replay constraints remain confounded.

No new threshold, selection gate or model promotion. These are training metrics;
they do not replace the previously failed validation or demonstrate better
face completion. No architecture decision can be proved from this audit alone.

Evidence: `outputs/coverage_full_fit/results.json` and219 saved prediction masks.
Source/checkpoint hashes verified; reproducer `scripts/diagnose_coverage_full_fit.py`.
Five added cases are consistent with the preceding targeted diagnostic.

Next: reconcile earlier real-only/high-learning-rate fit diagnostics with these
new coverage results before defining another VM run. Avoid repeating previously
failed settings. If those results do not isolate the cause, a bounded VM diagnostic
should measure real/replay optimization conflict rather than assume more epochs
or more labels will solve it. No new training command is justified yet.

Local root: `C:\xampp\htdocs\YEAR 4\Testing\`.
Related VM root: `~/forensic-dgp/coverage_vm_bundle/`.
