# Training-only detector diagnostic — 28 September 2026

Completed two local CPU runs from the original completion epoch 2 detector.
Each run repeatedly used the same four **training** images at native 256px:
`dev_003` (patterned covering), `dev_004` (dark respirator), `new_covered_52`
(pleated mask), and `uncovered_00` (negative control).
No validation/test examples were used. No weights were saved or deployed.

Both arms used seed 42, batch four, 80 updates, AdamW, gradient clipping and the
existing BCE + Dice + 0.25 hard-visible penalty. Only learning rate differed.
The generator stayed tensor-identical in both runs. Synthetic replay/teacher loss
was intentionally absent: this diagnoses basic supervised learnability, not the
full mixed-data training recipe.

| Training metric | Initial | 80 updates, lr 1e-5 | 80 updates, lr 1e-4 |
|---|---:|---:|---:|
| Mask IoU | 0.17920 | 0.89130 | 0.98975 |
| Missed covered pixels | 81.17% | 8.88% | 1.00% |
| Visible-pixel false-positive rate | 1.1014% | 0.4809% | 0.0046% |
| Negative cases with false masks | 0/1 | 0/1 | 0/1 |

Predeclared fit criterion: IoU >=0.90, missed fraction <=0.10, visible false-positive
rate <=0.01, and no false masks on the negative control. The higher-rate arm passed
by update 20 and at update 80; the current-rate arm narrowly missed the IoU
criterion at update 80. These thresholds are for this diagnostic only and do not
replace deployment selection gates.

The final overlay confirms nearly complete training masks in the higher-rate arm.
This rules out an absolute inability of this architecture/objective to fit these
three coverings. It does not establish generalization, an optimal learning rate,
retention of synthetic detection, or improved face completion. Repeated exposure
is much higher here than in the full pilot; that also limits comparison.

## Artifacts and next step

Local root: `c:\xampp\htdocs\YEAR 4\Testing\`; VM root: `~/forensic-dgp/`.
Results currently exist only locally:

- `outputs/detector_overfit_diagnostic/results.json`: hashes, exact membership,
  criterion, measured curves and generator integrity assertions.
- `outputs/detector_overfit_diagnostic/training_masks.jpg`: input, label, initial,
  low-rate and high-rate predictions.
- `outputs/run_detector_overfit_diagnostic.py`: reproducible bounded diagnostic;
  refuses to overwrite an existing output directory.

Next experiment should measure both training fit and validation during a bounded
learning-rate comparison on the full training split with synthetic replay. Keep
source membership, update budget and selection gates fixed; compare 1e-5 against
1e-4 rather than adopting the higher rate outright. Use corrected V2 validation
and report the known mannequin separately. Stop or reject candidates if synthetic
retention or visible-region behavior regresses. That experiment is not yet run;
this diagnostic alone does not justify another full GPU training launch.
