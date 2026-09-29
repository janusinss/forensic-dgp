# Mixed feature detector VM pilot — 29 September 2026

Decision: reject for deployment. Training completed, but synthetic retention fails.
No generator or application checkpoint was replaced. No local training occurred.

## Provenance and execution

Returned archive `outputs/feature-mixed-vm-results.tar.gz` SHA256
`1ae7a968840df118b89c44fafc445e34e66093b56a818e9b4413285bd0669e1e`.
Extracted separately into `outputs/downloaded_feature_mixed_vm/` to preserve the
stopped local attempt. Its bundle inventory exactly matches the sent inventory.
Final checkpoint SHA256
`eaa16229f88d416ad09d813a6f08ca0ae72bba214cb8266648bed382603864a4`.
20 epochs completed on NVIDIA L4, PyTorch 2.9.1+cu129, Python 3.10.12.
Two heads trained on 68 real + 400 synthetic cases; SAM encoder frozen.

Final training IoU: real raw/gated 0.8811/0.8820; synthetic raw/gated
0.9085/0.8453. The gate erased 32 covered synthetic training cases and one real
training case. Thus the gate problem already exists on fitting data.

## Fixed validation

25 real and 400 synthetic cases evaluated at unchanged 0.5 thresholds.
An independent verifier recounted all 850 raw/gated saved masks and checked gate
composition. No validation fitting or threshold search was performed.

| Metric | Real raw | Real gated | Synthetic raw | Synthetic gated |
|---|---:|---:|---:|---:|
| IoU | 0.83357 | 0.83710 | 0.87391 | 0.72867 |
| Missed target fraction | 0.09812 | 0.09927 | 0.07207 | 0.23303 |
| Visible false-positive fraction | 0.01160 | 0.01076 | 0.00909 | 0.00773 |
| Empty covered masks | 0/15 | 1/15 | 0/320 | 69/320 |
| Uncovered cases with false positives | 7/10 | 0/10 | 48/80 | 14/80 |

Real aggregate safeguard passes against original V3 baseline. Excluding the
known mannequin, gated IoU is 0.83084. The sole glare validation case is erased
(presence probability about 0.021). Aggregate success does not establish glare
success or identity-disjoint generalization.

Synthetic original baseline IoU is 0.97469, missed fraction 0.01648, visible FP
0.00133, empty covered 1/320, false-positive uncovered 0/80. Both raw and gated
arms fail retention. Removing the gate alone is not a qualified fix.
Asian-source gated IoU is 0.61906, empty 59/160; FFHQ-source is 0.84218,
empty 10/160. These are dataset-stratified results, not causal demographic claims.
Compared with the prior real-only gate, mixed training raises synthetic IoU from
0.41404 to 0.72867, but this remains below the unchanged acceptance baseline.

## Visual evidence and limitations

`outputs/feature_mixed_validation/diagnostic_preview.png` has ten diagnostic rows:
glare, mannequin, and the largest pixel-error example in eight named synthetic
strata. It is intentionally failure-selected, not representative sampling.
Visible raw covering detections are discarded by the gate; some clear controls
retain large false-positive regions. The mannequin mask misses strap boundaries.
The preview was inspected locally; no browser was used.

Validation uses existing verified CPU SAM feature caches, whereas fitting used
GPU float32 embeddings. This is recorded, not assumed numerically identical.
Before attributing all error to architecture, compare a fixed small set of identical
inputs across CPU/GPU embedding paths on the VM without optimizer updates.
Training-set gate failures independently show that this is not solely an unseen
validation problem. Neither arm qualifies for completion-output promotion.

## Next bounded action

Prepare a VM diagnostic that uses saved training features to audit presence errors
by covering kind, degradation and target area, plus a fixed CPU/GPU inference
parity check. No new fitting or threshold tuning in that diagnostic. If parity is
acceptable, use training evidence to specify a spatial presence classifier
comparison against the current globally averaged linear gate; training remains
VM-only. Do not repeat this recipe or merely increase epochs. Keep the original
real/synthetic acceptance criteria and require end-to-end completion improvement.

Local root: `C:\xampp\htdocs\YEAR 4\Testing\`.
VM artifacts: `~/forensic-dgp/feature_vm_bundle/outputs/feature_mixed_training/`.
