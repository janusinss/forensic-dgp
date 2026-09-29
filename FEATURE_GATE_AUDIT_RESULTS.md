# Presence gate and device parity audit — 29 September 2026

Decision: the checked CPU/GPU encoder differences do not explain the failed
presence gate. Proceed to a bounded architecture comparison on the VM, not a
repeat of the current training recipe. No model has been promoted.

## Verified evidence

Returned archive `outputs/feature-mixed-audit-results.tar.gz` SHA256:
`3e86bcf2c114af6d9013f89ff0289e44df34caa0deebd3ef1ca66518db4e1555`.
Checkpoint matches the previously evaluated final mixed pilot. All 400 training
case indices and source/kind/degradation metadata match its cache manifest.
Independent recount confirms all grouped classifier counts. Detailed evidence:
`outputs/downloaded_feature_mixed_audit/results.json` and `verification.json`.

| Training covering | Covered cases | Rejected by presence gate |
|---|---:|---:|
| Lower face | 80 | 0 |
| Eyes | 80 | 3 |
| Irregular | 80 | 11 |
| Object | 80 | 18 |

32/320 covered cases rejected. Asian-source cases account for 27/160 and FFHQ
5/160; source differences are not proof of demographic causation. Clean/degraded
misses are 15/160 and 17/160. Masks covering 5–20% of the image account for 30/233
misses; masks above 20% account for 2/86. Only one case is at most 5%, so the audit
does not establish small-occlusion coverage.

Five clear cases have positive classifier decisions, but only three produce
nonempty composed masks: two have empty pixel predictions. This explains the
audit's five classifier false positives versus the training log's three final
mask false positives; the measurements have different definitions.

## Device check

20 preselected inputs: first case for each source/kind/degradation combination.
This covers only two source identities, ten variants each, not 20 independent
people. CPU versus CUDA embeddings produce zero presence-decision flips, maximum
probability difference 0.0003445 and at most two differing mask pixels per image
(out of 65,536). Saved GPU versus fresh GPU embeddings match exactly on these
20 cases. Both heads are evaluated on CPU for this comparison, so it isolates
encoder differences, not every cross-device head-kernel difference.

The sample supports using the existing validation embeddings for diagnosis;
it does not certify numerical equivalence over every input or machine.
The 32 missed fitting examples independently establish a learned-gate problem.

## Next experiment specification

Hypothesis, not established cause: averaging the full embedding map before a
linear decision loses localized occlusion evidence. Compare a compact spatial
presence head against a freshly initialized global linear control on exactly the
same 468 training examples. Freeze the encoder and current pixel head, hold the
seed, six-group sampler, update budget, optimizer and 0.5 threshold fixed. Use
final checkpoints only, with training metrics separated by real/synthetic source.
No validation-driven threshold or epoch search. Training must run on the VM.

This tests the presence architecture only. Even perfect presence gating cannot
repair the current raw pixel arm's synthetic IoU of 0.87391 versus the original
0.97469 baseline, so raw segmentation remains a separate unresolved requirement.
Any new head must pass unchanged real/synthetic safeguards and glare inspection;
then evaluate full completion images before considering deployment. No claim of
improved generated facial detail follows from this audit.

Next implementation: prepare the matched VM comparison with checkpoint/provenance
checks and CPU-refusal safeguards. It has not been implemented or trained yet.
