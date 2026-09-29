# Presence architecture comparison results — 29 September 2026

Decision: spatial presence is a better experimental component; neither complete
detector qualifies for deployment. Application and generator checkpoints unchanged.

Both VM arms completed 20 epochs with identical 468 training cases, batch order,
optimizer settings and fixed threshold 0.5. Returned script hash matches local
source, final state protocols match, and training metadata is identical across
arms. Spatial training classification has zero errors; global has 26 missed
covered cases and three positive clear cases. No local optimizer updates.

## Fixed validation

| Measure | Global | Spatial |
|---|---:|---:|
| Real composed-mask IoU (25 cases) | 0.83710 | 0.83710 |
| Real missed covered cases | 1/15 | 1/15 |
| Real clear classifier false positives | 0/10 | 0/10 |
| Synthetic composed-mask IoU (400 cases) | 0.76566 | 0.87321 |
| Synthetic missed covered cases | 52/320 | 5/320 |
| Synthetic clear classifier false positives | 14/80 | 4/80 |
| Synthetic clear nonempty composed masks | 12/80 | 3/80 |
| Synthetic visible false-positive fraction | 0.00800 | 0.00801 |

Both pass the original real aggregate safeguard; both fail synthetic retention.
The sole real glare validation case is rejected by both. Spatial source-stratified
synthetic IoU: Asian 0.87991 (one missed covered case), FFHQ 0.86618 (four misses).
These source comparisons do not establish a demographic cause.

The unchanged original synthetic baseline remains IoU 0.97469, visible FP 0.00133,
one missed covered case and zero clear cases marked. The spatial result still
falls materially short; no gate or target was weakened after observing results.

## Verification and visual inspection

`scripts/evaluate_presence_comparison.py` evaluated both final checkpoints with
the verified validation embeddings and exactly the same raw pixel predictions.
Each raw mask was recounted against its target before reuse. Independent review
script recounted all 850 saved composed masks (425 per arm). Two gate-arithmetic
tests pass. An initial evaluation stopped on RGB-encoded synthetic target PNGs;
targets now explicitly convert to grayscale, matching the dataset loader. Partial
outputs are retained in `outputs/presence_comparison_validation_partial_rgb_error`;
the completed result is `outputs/presence_comparison_validation/results.json`.

Ten-row `preview.png` was inspected: one real glare failure, three recovered
synthetic masks, three remaining spatial misses and three clear false positives.
It is diagnostic selection, not a representative sample. Some irregular masks
are recovered, but blurred inputs still trigger false regions; a small real glare
region and several synthetic coverings are erased by the gate.

Spatial head has 4097 parameters versus 257 for global. This is evidence for the
tested spatial architecture, not an isolated causal proof that location alone
explains improvement. Both heads use CPU inference on cached features; prior
encoder device parity checked 20 variants from two source images only. Repeated
use of this development validation set limits claims of unbiased generalization;
independent evaluation will still be needed before thesis-level final claims.

## Next justified action

Keep spatial head as a research candidate, not the app default. Audit raw pixel
errors against mask boundaries and covering/degradation strata, using existing
outputs first, to distinguish coarse boundary errors from false object regions.
Use training-only evidence to specify a bounded pixel-head change on the VM.
Do not repeat the global gate recipe, tune the 0.5 threshold on validation, or
claim the generator improved. Sparse glare labels remain an explicit limitation.
Any candidate still needs unchanged full-detector safeguards and end-to-end
restoration/completion visual and metric improvements.

Returned inputs: `C:\xampp\htdocs\YEAR 4\Testing\outputs\downloaded_presence_comparison\`.
VM source: `~/forensic-dgp/feature_vm_bundle/outputs/presence_architecture_comparison/`.
