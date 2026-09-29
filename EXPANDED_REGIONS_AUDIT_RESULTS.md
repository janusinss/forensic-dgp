# Region audit results — 30 September 2026

Archive `outputs/expanded-regions-audit-results.tar.gz` SHA256:
`423e63c23fa7227672a134a41999a312ade76c1d41f507f00f0ca5a3eb64c99d`.
Evidence: `outputs/downloaded_expanded_regions_audit/` (`results.json`,
`verification.json`, `review.png`, and binary masks).

All 5,632 binary masks (1,408 cases, four masks each) were independently read and
recounted. Core/border/outside counts reproduce the report and earlier per-case
TP/FP/FN. Script, prior-audit digest, checkpoint references and protocol match.
The executed VM script additionally verifies regenerated input hashes and target
equality against its cached features. This was inference only, with zero updates.

| Degraded irregular training metric | Fixed | Anatomical |
|---|---:|---:|
| Core missed / core pixels | 0.5863% | 1.0111% |
| Border missed / border pixels | 18.8162% | 22.8076% |
| Border share of all missed pixels | 96.5838% | 95.2086% |
| Outside-target false-positive pixels | 36,365 | 19,305 |

Each arm contains 352 degraded irregular cases; core pixels total 2,415,836 and
added-border pixels total 2,128,063. The border is 46.83% of the target area, so
its much larger error share is not explained solely by its area. This localizes
the observed training deficit; it does not prove which loss or architecture will
generalize better. Other covering types and glare remain separate requirements.

The six predetermined preview rows were inspected. Most show narrow missing
boundary strips; row 0093 has substantial missing border around a brown covering.
Row 0103 includes false-positive background regions. Thus expanding every predicted
mask indiscriminately is not justified. Preserve targets and visible-region gates.

## Next matched VM experiment specification (not yet executed)

Use the fixed-placement final checkpoint as a shared initialization; this choice
does not promote it. Compare unchanged pixel BCE+Dice (control) against border-
weighted BCE+unchanged Dice. Give pixels in the existing degraded synthetic target
outside its original geometry weight 2 instead of 1; normalize BCE by total
weights per image, then average images. All other pixels, including real/clear
cases, retain weight 1. No target editing, dilation change or threshold tuning.

Freeze the same presence head in both arms to isolate pixel learning. Use identical
frozen caches, full source membership, sample order, seed 42, optimizer state reset,
AdamW learning rate 1e-4, weight decay 1e-4 and gradient clip 1.0. Predeclare a
bounded 10 epochs x 80 updates per arm, batch 12 with the existing six-group
balanced scheduler. Save final checkpoints only; do not select an intermediate
epoch from repeated validation. The equal-budget control is necessary because
additional updates alone could improve the existing model.

Before execution, implement/test region-weight construction and ensure regenerated
targets match cache rows. Actual optimization runs only on the VM. Report raw and
gated training core/border/outside errors plus the unchanged real/synthetic
validation safeguards, mannequin separately, clear-image behavior and glare.
Neither improved training fit nor improved aggregate IoU alone qualifies a model.
End-to-end completion review remains required before any application promotion.

Local root: `C:\xampp\htdocs\YEAR 4\Testing\`.
VM root: `~/forensic-dgp/expanded_feature_bundle/`.
