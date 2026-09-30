# Fixed projection hypothesis — 30 September 2026

The mixed-gradient audit motivates one controlled test, not an established
improvement. No optimizer has been run for this experiment.

Let `r` be the actual batch-weighted real supervised gradient, and `s` the
batch-weighted replay supervised gradient plus weight1 teacher-consistency
gradient. Ordinary arm uses `r+s`. Treatment uses:

```
if dot(r,s) < 0 and dot(s,s) > 0:
    r_projected = r - dot(r,s) / dot(s,s) * s
else:
    r_projected = r
gradient = r_projected + s
```

This is one-sided replay-referenced projection, not the symmetric randomized
PCGrad algorithm. Float64 dot products reduce cancellation; parameter gradients
retain their original dtype. Replay itself is not modified. No tunable projection
weight, validation threshold sweep or changed selection gate is introduced.

Implementation: `coverage_projection.py`. Seven combined projection/decomposition
tests passed, including conflict, aligned/orthogonal/zero references, input
immutability, invalid values, numerical alignment and actual-step sign reporting.
Only small synthetic tensors were used; no local model training.

## Matched VM runner requirements

Both arms must use the extended73 training dataset, pinned original parent, same
cached638 replay cases, and extended arm's frozen210-update schedule. Keep
AdamW1e-5, weight decay1e-4, loss weights, clipping1 and all original gates fixed.
Archive the same initial state for both arms. Use a newly instrumented ordinary
arm; the previous run provides context but lacks actual-step diagnostics.

For each update record projection occurrence, dot products before/after, removed
real norm, clip scaling and `g_replay dot (theta_after-theta_before)`. Positive
last value indicates predicted first-order replay-loss ascent. AdamW momentum,
preconditioning and weight decay can invalidate the raw-gradient property;
projection does not guarantee finite-step loss reduction or validation retention.
Neither an alignment statistic nor improved training fit is a selection criterion.

Next: integrate these helpers into a separate VM-only runner with forward-only
preflight, verified initial/cache/data hashes, unchanged-generator assertions,
per-epoch masks and actual-step logs. Package only after runner checks pass.
Existing coverage runner/bundle and application weights stay unchanged.

Local: `C:\xampp\htdocs\YEAR 4\Testing\`.
VM data/checkpoints: `~/forensic-dgp/coverage_vm_bundle/`.
