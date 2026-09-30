# Border experiment: real-training error localization

30 September 2026. Local CPU inference only on the 68 reviewed training images;
no optimizer updates. `scripts/audit_border_training_localization.py` checks V2/V3
training membership and image hashes, reuses image features, and loads current V3
targets. Both new checkpoint hashes and encoder identity match the returned run.
All real raw/gated aggregate counts match the VM report exactly; this does not
claim feature bitwise identity across CPU/GPU.

Evidence: `outputs/border_training_localization/results.json`, `verification.json`,
`preview.png`. Independently recounted 272 saved masks against current targets and
recomputed region counts. Six largest treatment gated-FP training cases inspected.

Diagnostic region: outside the target but within Chebyshev distance 8 pixels of it
(17x17 dilation minus target). This splits error reporting only; it is not a new
target, tolerance for passing, inference postprocess or threshold.

| Gated false-positive pixels | Control | Border weight 2 |
|---|---:|---:|
| Within 8 pixels of target | 8,128 | 8,893 |
| Beyond 8 pixels | 31 | 68 |
| Total | 8,159 | 8,961 |

Boundary-proximate errors dominate both arms (>99%). Weighting adds 765 near and
37 distant FP pixels. All 25 clear training cases remain empty after gating;
ungated control marks 80 pixels across 3 clear images and treatment 119 across 4.
Do not generalize clear-training performance to validation or deployment.

The six inspected examples show mask-boundary differences and thin strap handling.
Some targets are coarse polygons. The pictures do not justify declaring every
disagreement a model error or a label error, nor do they authorize relabeling to
match predictions. Glare training fit remains distinct from failed glare transfer.

Next: check synthetic training false-positive localization using existing local
caches, only after verifying exact training-source and generated-image provenance.
This real-only result does not establish the source of synthetic retention failure.
Keep original evaluation gates and targets unchanged; no further weight sweep.

Local root: `C:\xampp\htdocs\YEAR 4\Testing\`.
VM run: `~/forensic-dgp/expanded_feature_bundle/outputs/expanded_border_training/`.
