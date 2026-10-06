# V25 saved-state gradient diagnostic — finite, zero updates

6 October 2026. V25's audited0.0282235% delivered structural gain misses the1%
early stop, despite active own-DGP feature projections. Its corrected objective
rewards degraded improvements, but changes remain visually too small across all50
training cases. Diagnose actual gradient scales and competing terms before another
recipe. The user-selected own-DGP spatial/feature direction remains in effect.

This diagnostic reads both saved0/50 heads from the failed V25 folder. It does
not rerun its trainer, change its objective/quality thresholds, select a checkpoint
or create a new trained weight file. Original DGP weights, corrected normalization,
recognizer, inputs, masks,250 frozen input-conditioned FPN arrays, splits and failed
gates stay immutable. No data or weights need uploading again.

Packet **15,259 bytes**, four regular files/two executable assets:

```text
Archive: cctv-dgp-v25-gradient-diagnostic-v1-execution.tar.gz
Archive SHA256: 669626a5cbf480edb38b0c6787dcd59da6ae8d935a65789011e3102f0f45d5c5
Protocol SHA256: 884412f5e3dab571e630cdadb4f962046c8da309ebe6b62eac4d5af0e29d281a
VM folder: ~/forensic-dgp/cctv_dgp_v25_gradient_diagnostic_v1_vm
Required existing parent: ~/forensic-dgp/cctv_dgp_spatial_features_vm_v25
```

Both states use exactly the same50 exposed TRAINING cases. Ten sequential batches
of five per state produce20 head forwards and140 component autograd.grad calls.
Each component batch mean is divided by10 to give its full50-case mean gradient.
All seven original terms are retained: degraded landmark detail, degraded observed
detail, degraded pixel error, clear baseline anchor, pixel regression, SSIM regression
and fixed ArcFace regression. No optimizer/backward()/weight-update call exists.

The original frozen ArcFace computes50 target and50 baseline embeddings, plus20
prediction batches:120 recognizer forwards total. The original DGP has zero forwards
because the audited250 saved feature maps are reused. Targets affect losses only;
they never condition the restoration head. Gradient storage is7x53,781 float64 per
state. Save norms, Gram matrix, cosines, total-gradient norm, exact26-parameter
layout, per-parameter component norms and per-batch component values/norms.

Before neural work, verify the original235 assets, original protocol and114 saved
input hashes, retained update50 failure and250 cached feature file hashes. Compile
only the original verified head class, host guard and five metric/objective functions
from source AST. Do not import or execute the V25 training entry point. Require the
existing named Linux VM, g2-standard-4 metadata, idle NVIDIA L4 and paths under
~/forensic-dgp. A competing GPU task is rejected without stopping it. An idle tmux
shell is permitted. Windows rejects execution before model imports/output creation.

Worker is capped at420s; external supervisor480s plus30s termination grace. Export
is capped at30s internally/60s externally plus10s grace. Require1GiB free disk and
at most20GiB torch-allocated VRAM. Stop on nonfinite gradients, wrong counters,
changed parent evidence/state, raw parity failure, missing original asset, competing
GPU process or timing/storage failure. Preserve the partial return; no automatic
retry, training follow-on, gate increase or app promotion. Shell exports failure
evidence as well as successful measurements.

Each fixed-state batch must match original saved raw outputs within the inherited
head2e-6 replay bound. Saved head hashes must match before/after, recognizer must stay
frozen, cached features must remain gradient-free, and head .grad buffers remain
empty because autograd.grad returns disposable component gradients. Parent evidence
and all cached/source hashes are checked again at the end. No new checkpoint exists.

The packet, Python3.10 syntax, actual Windows guards and Bash syntax are verified;
nine unsafe-packet/source/matrix regressions pass. The sandbox Bash signal-pipe
restriction is retained as preparation evidence; the same read-only -n check passes
outside the sandbox. No script commands, neural gradients or VM actions run locally.

On return, strict archive/sidecar/export/source verification precedes interpretation.
Independently recompute saved gradient norms/Gram/cosines/per-parameter statistics
from the two finite float64 arrays, check counts/timing/no-update/frozen-state receipts,
compare recorded component sums with the existing fixed CPU saved-loss audit and
retain the original V25 failure. CPU/CUDA scalar compatibility is prospectively
bounded at2e-5 for nonidentity terms and5e-4 for the fixed ArcFace regression term;
the latter derives from the original5e-5 cosine bound twice, multiplied by the fixed
penalty5. These numerical bounds do not relax a scientific or visual quality gate.
Zero-norm cosine entries are marked by their zero norms; do not infer agreement
from a placeholder cosine0. Negative cosine indicates opposing fixed-state terms;
it does not prove that cancellation alone caused the failed optimizer trajectory.

This measures saved-state gradients rather than AdamW moments or its step trajectory.
It cannot prove useful restoration, generalization, local CCTV performance, native
identity or any covering-family acceptance. Preserve all original protocol limits,
checkpoint states and exposure boundaries. All visible facial features remain
required together. Any candidate app integration and independent final review remain pending.
Goal active/incomplete.

[Manual tmux/upload/download steps](<C:/xampp/htdocs/YEAR 4/Testing/CCTV_DGP_V25_GRADIENT_DIAGNOSTIC_V1_VM.md>) ·
[Packet verification](<C:/xampp/htdocs/YEAR 4/Testing/outputs/cctv_dgp_v25_gradient_diagnostic_v1_preparation/independent_packet_audit.json>) ·
[V25 audited closure](<C:/xampp/htdocs/YEAR 4/Testing/CCTV_DGP_SPATIAL_FEATURES_V25_RESULTS.md>)
