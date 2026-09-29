# Expanded synthetic retention attribution — 29 September 2026

Gate correction alone cannot qualify either expanded model. A diagnostic using
ground-truth presence to retain all covered predictions and suppress all clear
ones gives IoU 0.95218 fixed / 0.93899 anatomical, below the unchanged 0.97469
baseline. This label-informed oracle is not an inference method or selected model.
Missed pixels and visible false positives still exceed baseline under this oracle.

The real presence gate adds 19,092 missed pixels to 112,348 segmentation misses
for fixed placement, and 22,704 to 171,183 for anatomical placement. Thus most
remaining misses already exist before gating. All four fixed / five anatomical
gate-rejected covered validation cases are irregular occlusions.

Degraded inputs account for about 92.4% fixed / 92.0% anatomical raw missed target
pixels. This is pixel-weighted attribution, not a per-image causal estimate: mask
sizes and dilation differ across strata. Fixed degraded irregular cases contribute
48,430 raw misses, degraded lower 30,230 and degraded eyes 16,810. Corresponding
anatomical values are 58,861, 42,471 and 45,153. The current failures therefore
involve degraded segmentation as well as the gate, not just real glare coverage.

Script `scripts/audit_expanded_retention.py` recomputed every saved gate decision
and aggregated all 400 synthetic cases per arm from previously recounted pixels.
No thresholds, epochs, labels, models or selection criteria changed. No new
inference/training ran. This reuses development validation to diagnose a failure;
it is not independent test evidence or a basis for hand-tuning individual cases.

Next independent work: inspect the existing **training** irregular-covering cases
under the predeclared camera degradation and measure training fit by kind and
degradation. Use the existing final checkpoint/caches where available. Do not
pick augmentation constants from validation errors or repeat training yet.
The glare scope question remains pending; ambiguous small-highlight proposals
stay disabled regardless of this independent diagnostic.

Local evidence: `C:\xampp\htdocs\YEAR 4\Testing\outputs\expanded_retention_audit\results.json`.
Parent VM results: `~/forensic-dgp/expanded_feature_bundle/outputs/expanded_feature_training/`.
Application and generator baselines remain unchanged; the full goal is unmet.
