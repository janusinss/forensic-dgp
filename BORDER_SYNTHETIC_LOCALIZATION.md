# Synthetic training subset localization — 30 September 2026

`scripts/audit_border_synthetic_local.py` ran CPU inference only. The interrupted
local cache contains 77 of its originally expected 400 examples. Exactly 46 inputs
match expanded-run RGB hashes: 16 clear, 15 object and 15 irregular examples.
No eye/lower-face cases match the revised fixed-arm texture recipe. This is an
availability-limited subset, not representative full-training evaluation.

Verified original training membership through the expanded manifest loader,
source file hashes, saved input/mask hashes, feature metadata and encoder digest.
Regenerated expanded inputs and targets match each included cached example.
Both checkpoint hashes/protocols match the returned border experiment.
All 184 saved prediction masks independently recounted. Evidence:
`outputs/border_synthetic_local/results.json`, `verification.json`, `preview.png`.

| Gated FP pixels | Control near / far | Border2 near / far |
|---|---:|---:|
| Clean object (8) | 372 / 33 | 427 / 53 |
| Degraded object (7) | 2,363 / 570 | 2,917 / 817 |
| Clean irregular (8) | 1,258 / 27 | 1,403 / 42 |
| Degraded irregular (7) | 773 / 986 | 1,190 / 1,221 |

Near means outside the original target but within an 8-pixel Chebyshev band; far
is the remainder. This is reporting only, not relaxed scoring. All 16 clear subset
cases are empty after gating, while ungated heads produce small false positives.
These are CPU-cache results; full VM synthetic per-case equivalence is not claimed.

Six degraded cases with largest treatment distant-FP counts were visually reviewed.
Spurious regions appear on background and skin, especially the scene behind the
red irregular covering. Degraded irregular shapes also retain missing boundary
regions. Thus the synthetic problem is not solely real polygon annotation
precision, nor merely the presence gate. Border weighting is still rejected.

Next: prepare an architecture-level, bounded refinement comparison using the
existing image and frozen semantic features, rather than another weight sweep.
First establish a zero-initialized residual refinement head that exactly preserves
the parent at initialization and inspect its image-input/provenance path. Compare
image-conditioned refinement with an equal-budget semantic-only control before
claiming benefit. This is a hypothesis, not an established superior architecture.
All fitting remains on VM, original targets/gates remain fixed, and known presence
and glare transfer failures remain separate requirements before any promotion.

Local root: `C:\xampp\htdocs\YEAR 4\Testing\`.
VM cached data: `~/forensic-dgp/expanded_feature_bundle/outputs/expanded_feature_training/`.
