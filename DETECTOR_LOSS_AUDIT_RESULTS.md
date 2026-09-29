# Detector loss/gradient audit — 29 September 2026

Measured the existing objective without optimizer steps or model changes. All68
real training images were used for per-image output-logit gradients. Six fixed,
balanced training batches (two covered/two uncovered) were used for parameter
gradients at the original completion epoch2 detector and consistency epoch10.
No validation/test gradients or synthetic replay gradients were evaluated.

The measured terms are BCE, soft Dice, and the actual weighted hard-visible
penalty (0.25 times highest-error10% negative-pixel BCE). The decomposition matches
the production objective. A strict elementwise floating-point comparison initially
failed; checking relative norm error gave 3e-6 to1.4e-5, below the declared1e-4
audit tolerance. This was an audit numerical check, not a change to training gates.

| Parameter-gradient measurement across six batches | Initial | Consistency epoch10 |
|---|---:|---:|
| Median penalty / combined BCE+Dice norm | 1.536 | 0.343 |
| Median penalty versus BCE+Dice cosine | -0.186 | 0.065 |
| Batches with opposing directions (cosine <0) | 4/6 | 3/6 |
| Median uncovered / covered contribution norm | 0.122 | 0.306 |
| Median total norm before clipping | 38.43 | 29.65 |

The penalty has exactly zero direct gradient on covered pixels, as intended. It
can still oppose learning through shared network parameters. Uncovered examples
do not consistently dominate: their relative parameter norm varies widely by
batch (initial0.0007–1.564; trained0.033–1.583). The penalty is not uniformly
dominant either. Every audited total norm exceeds the training clip threshold1;
clipping scales the combined direction, not each loss independently.

## Interpretation and next experiment

This identifies an early loss-balance hypothesis, not a proven root cause or a
new optimal weight. Gradient norm is not an AdamW update magnitude; cosine
conflict alone does not establish worse validation. Six real-only batches do not
represent the full mixed-data trajectory. Earlier real-only experiments also
showed that removing the penalty can worsen visible false positives.

Next controlled comparison: keep replay, initialization, lr1e-5, data, seed and
budget fixed, comparing hard-visible weight0 versus0.25. Measure real training fit,
corrected real validation and fixed synthetic validation throughout. Keep all
selection gates unchanged and do not deploy either arm if visible false positives
or synthetic retention regress. This mixed-replay penalty ablation has not yet
been run; earlier zero-penalty real-only training is not its matched control.
Do not simultaneously change learning rate, penalty and data augmentation.

Local evidence: `c:\xampp\htdocs\YEAR 4\Testing\outputs\detector_loss_audit\results.json`.
Reproduction script: `outputs/run_detector_loss_audit.py` under the same root.
Corresponding VM root is `~/forensic-dgp/`; these ignored artifacts need explicit
transfer. The interrupted second-checkpoint audit was completed without rerunning
the saved initial-checkpoint portion. No model weights or production code changed.
