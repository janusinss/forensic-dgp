# Pixel architecture comparison — 29 September 2026

Decision: neither candidate qualifies for deployment. Context improves synthetic
segmentation relative to the matched pointwise arm but reduces real validation IoU.
No application or generator checkpoint changed; no local training occurred.

## Execution and verification

Both VM arms completed 20 epochs / 1600 updates on the same 468 cases. Returned
protocol and checkpoint hashes match the local training script, parent pixel
checkpoint, source manifest, V3 labels and frozen spatial gate. Initial logit
difference on the checked GPU inputs was exactly zero. Parameter counts differ:
16,513 pointwise versus 147,585 context, so capacity and context are not separated.

425 fixed validation cases evaluated per arm. Cached real feature hash and image
order were verified; synthetic input hashes and encoder identity were checked.
Presence probabilities reproduce the previous spatial-head evaluation within
1e-6. Independently recounted all 1,700 saved raw/gated masks, including gating
composition. Ten diagnostic preview rows inspected. These checks establish
evaluation consistency, not unbiased generalization after repeated validation use.

## Results at unchanged thresholds

| Gated metric | Pointwise | Context |
|---|---:|---:|
| Real validation IoU | 0.82858 | 0.81811 |
| Real visible false-positive fraction | 0.01162 | 0.00978 |
| Real missed target fraction | 0.10339 | 0.12536 |
| Synthetic validation IoU | 0.88015 | 0.92223 |
| Synthetic visible false-positive fraction | 0.00952 | 0.00652 |
| Synthetic missed target fraction | 0.06286 | 0.03691 |
| Synthetic empty covered masks | 5/320 | 5/320 |
| Synthetic clear cases marked | 4/80 | 3/80 |

Both pass original real aggregate safeguards, but neither meets original synthetic
retention (IoU 0.97469, visible FP 0.00133, missed fraction 0.01648, empty 1/320,
clear cases marked 0/80). Context raw synthetic IoU is 0.92295, so gate removal
alone cannot meet the target. The frozen spatial gate still rejects the real glare
case and five synthetic covered cases; a pixel-only change cannot undo that.

Context training IoU is 0.95748 real / 0.97074 synthetic after gating, considerably
above validation 0.81811 / 0.92223. This gap is consistent with limited training
diversity or overfitting, but does not by itself prove a single cause. The current
synthetic training set has only 40 source images, with ten fixed variants each.

Preview selection: real glare and mannequin, two largest real error regressions,
and worst context error in six synthetic strata. It is failure-focused, not
representative. Context masks remain fragmented on some real patterned masks,
miss straps, and mark visible regions on blurred faces. Some synthetic boundaries
improve. No improved reconstructed-face output is claimed from mask metrics.

## Next justified action

Stop expanding architectures on this small fixed training pool. Audit available
training-source diversity and reviewed real coverings, keeping validation/test
membership untouched. Prepare a larger, balanced training-source manifest with
exact overlap checks, and assess training-only coverage of glare, patterns and
degradation. Do not relabel validation errors to improve scores or repeat the
same 40-source recipe. Any data expansion should be tested with the architecture
and thresholds fixed, so its effect can be distinguished from architecture changes.
More real glare annotations and independent final evaluation remain required.
Training stays on the VM after the data audit; baseline remains unchanged.

Local evidence: `outputs/pixel_comparison_validation/results.json`,
`verification.json`, `preview.png`; inputs `outputs/downloaded_pixel_comparison/`.
VM source: `~/forensic-dgp/feature_vm_bundle/outputs/pixel_architecture_comparison/`.
