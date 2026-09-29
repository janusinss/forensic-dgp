# Expanded placement results — 29 September 2026

Neither final candidate qualifies for promotion. Both pass the original real
aggregate safeguards but fail unchanged synthetic retention. No generator or
application checkpoint changed. No local optimizer updates occurred.

## Evidence and execution

Returned `outputs/expanded-feature-results.tar.gz` SHA256:
`a063b28cfb11c92f419dbd22d74acca6a9a6d9c8c7de5e7ea809d6d36b353bdb`.
Both arms completed 20 epochs / 1,600 updates on NVIDIA L4, torch 2.9.1+cu129,
Python 3.10.12; reported combined runtime about 34.5 minutes. Each saw all 3,540
real-plus-synthetic cases. Inventory, source manifest, checkpoints, initial-head
digest and reconstructed schedule match the predeclared experiment.

All 7,080 returned cache records were checked for source membership/provenance;
2,180 corresponding real/generic row digests match across arms. Feature arrays
remain on the VM, so returned digests alone do not independently verify their
contents. Local validation uses previously verified frozen feature caches with
image/encoder provenance checks, not training features.

425 fixed development-validation cases evaluated per arm: 25 real, 400 synthetic.
All 1,700 raw/gated saved masks independently recounted against target pixels.
Ten failure-focused preview rows visually inspected. Repeated use of this
development validation does not constitute an untouched final evaluation.

## Results at unchanged thresholds

| Gated metric | Fixed placement | Anatomical placement |
|---|---:|---:|
| Real IoU | 0.82150 | 0.81148 |
| Real visible false-positive fraction | 0.00774 | 0.00823 |
| Real missed target fraction | 0.13358 | 0.14134 |
| Real excluding mannequin IoU | 0.82204 | 0.80922 |
| Mannequin IoU | 0.81614 | 0.83425 |
| Glare IoU | 0 | 0 |
| Synthetic IoU | 0.94662 | 0.93227 |
| Synthetic visible false-positive fraction | 0.00222 | 0.00158 |
| Synthetic missed target fraction | 0.03911 | 0.05769 |
| Synthetic empty covered masks | 4/320 | 5/320 |
| Synthetic clear cases marked | 2/80 | 1/80 |

Original synthetic safeguards remain IoU 0.97469, visible FP 0.00133, missed
fraction 0.01648, empty covered 1/320, clear marked 0/80. Neither arm meets them.
Raw synthetic IoU is 0.95069 fixed / 0.93759 anatomical: gate removal alone cannot
meet the requirement. Fixed placement is stronger on this fixed-geometry benchmark;
that does not establish superiority for all real occlusions. Real metrics also
favor fixed placement in this run, apart from the separately reported mannequin.

Training gated IoU: fixed real 0.94397 / synthetic 0.95704; anatomical real 0.94655 /
synthetic 0.94607. Synthetic training masks differ between arms, so these training
scores are not directly comparable evidence of generalization. Differences from
older runs also combine data expansion and trainable presence heads.

## Visual findings and next action

Both miss all 491 labeled pixels in the validation glare case before gating.
The raw masks contain only 7 or 10 false-positive pixels; gate probabilities are
0.000161 / 0.000173. This is not solely a gate rejection. Do not lower thresholds
using this example or alter its label to improve results.

The ten-row preview includes mannequin, glare, two largest real regressions and
six synthetic strata. Patterned masks remain fragmented or undersized; some blur
boundaries extend into visible skin. Anatomical masks lose portions of the covered
eye band in a difficult synthetic case. Both gates reject a covered irregular
case. These are diagnostic examples, not a representative visual-quality score.

Next: inspect existing final heads on the two reviewed **training** glare cases
and their clear controls, using local inference only. Determine whether they fit
the labeled glare at all before choosing additional data or model work. Expanded
ordinary-face data did not solve glare. More independently reviewed training glare
examples and an independent final evaluation remain needed. No repeat of this
same VM run, threshold tuning, or generator training is justified by these results.
End-to-end completion improvement remains unproven; baseline stays unchanged.

Local evidence: `C:\xampp\htdocs\YEAR 4\Testing\outputs\expanded_feature_validation\`
(`results.json`, `verification.json`, `provenance_verification.json`, `preview.png`).
Returned input: `outputs/downloaded_expanded_feature/` under the same workspace.
VM originals: `~/forensic-dgp/expanded_feature_bundle/outputs/expanded_feature_training/`.
