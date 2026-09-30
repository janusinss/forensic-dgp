# Refinement comparison — 30 September 2026

Neither candidate qualifies. RGB conditioning did not improve the semantic-only
control under this fixed recipe. No generator or application promotion.

Archive `outputs/refinement-results.tar.gz` SHA256:
`b47875a0dd295c4a016dbc37d1f4645e1e434c057021b2dbe6d6e62667359111`.
Both arms report 800 updates and 3,540 examples; combined VM runtime 560.5 seconds.
Executed code hashes, protocol, schedule, parent/checkpoint hashes checked. Parent
and gate tensors in both final checkpoints equal their frozen originals exactly.

Important provenance limit: independent local seeded reconstruction does NOT match
the reported initial-state digest (local torch 2.13.0+cpu vs VM 2.9.1+cu129).
The executed VM code asserts equal initial states in both arms, but initial tensors
were not archived. Runtime differences are a hypothesis, not a proven explanation.
The mismatch remains explicitly recorded in evaluation output; it was not treated
as a passing provenance check. Predictions/quality can still be measured directly.

Local evaluation `scripts/evaluate_refinement.py`: 25 real plus 400 synthetic
development cases per arm, fixed targets/thresholds. All 1,700 saved raw/gated masks
independently recounted; ten failure-focused preview rows inspected. Evidence:
`outputs/refinement_validation/results.json`, `verification.json`, `preview.png`.

| Gated validation metric | Semantic only | RGB |
|---|---:|---:|
| Real IoU | 0.82118 | 0.82060 |
| Real excluding mannequin IoU | 0.81895 | 0.81829 |
| Mannequin IoU | 0.84340 | 0.84366 |
| Glare IoU | 0 | 0 |
| Synthetic IoU | 0.95683 | 0.95501 |
| Synthetic missed fraction | 0.02009 | 0.01877 |
| Synthetic visible FP fraction | 0.00355 | 0.00404 |
| Synthetic empty covered cases | 4 | 4 |
| Synthetic clear cases marked | 1 | 1 |

Both pass real aggregate safeguards but fail original synthetic retention (IoU
0.97469, missed fraction 0.01648, visible FP 0.00133, empty <=1, clear FP 0).
Raw synthetic IoU also fails (0.96080 / 0.95885). The gate is not the sole problem.
The preview retains patterned-mask gaps, eye-band boundary spill, object/background
false positives, a gate-rejected irregular covering and absent glare masks.
Compared with the unrefined parent, missed pixels improve but false positives grow.
This does not establish reviewed end-to-end completion improvement.

Next: no-training original-runtime initialization audit, then reassess the failed
architectural assumptions before spending more training time. Do not launch a
learning-rate/width sweep on this repeatedly inspected development set.
Upload `scripts/export_refinement_initial_vm.py` to VM home and run:

```bash
cd ~/forensic-dgp/expanded_feature_bundle &&
source ../feature_vm_bundle/.venv/bin/activate &&
python -u ~/export_refinement_initial_vm.py --root "$PWD"
```

Download `/home/janusdominic0/forensic-dgp/expanded_feature_bundle/refinement-initial-audit.tar.gz`
to `C:\xampp\htdocs\YEAR 4\Testing\outputs\refinement-initial-audit.tar.gz`.
This exports recreated tensors and reports either digest match or mismatch without
training. It cannot retroactively prove the unavailable historical initial tensors.
