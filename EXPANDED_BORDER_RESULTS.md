# Matched border-weight experiment — 30 September 2026

Decision: reject border weighting as a model improvement. Neither final checkpoint
qualifies for promotion. Keep generator/application baselines unchanged.

Returned archive `outputs/expanded-border-results.tar.gz` SHA256:
`fcc4cb4875391c474246bd6669be4f73c5efb52b937eafe09765734aff7d4e29`.
Both VM arms completed 800 updates, all 3,540 training examples, approximately
731.7 seconds combined. Script, parent checkpoint, initial heads, schedule digest,
protocol, final checkpoint hashes and frozen presence tensors were verified.
Feature arrays remain on VM; reported training counts are not independently
recounted from predictions for these new heads.

Local inference: `scripts/evaluate_expanded_border.py`; evidence
`outputs/expanded_border_validation/results.json`, `verification.json`, `preview.png`.
Evaluated unchanged 25 real and 400 synthetic development-validation cases per arm.
All 1,700 saved raw/gated masks independently recounted. Ten failure-focused preview
rows inspected. This repeatedly used development set is not an untouched test set.

| Gated validation metric | Control | Border weight 2 |
|---|---:|---:|
| Real IoU | 0.82306 | 0.82282 |
| Real excluding mannequin IoU | 0.82122 | 0.82117 |
| Mannequin IoU | 0.84138 | 0.83921 |
| Glare IoU | 0 | 0 |
| Synthetic IoU | 0.95096 | 0.94910 |
| Synthetic missed fraction | 0.03077 | 0.02724 |
| Synthetic visible FP fraction | 0.00283 | 0.00367 |
| Synthetic empty covered cases | 4 | 4 |
| Synthetic clear cases marked | 1 | 1 |

Both pass original real aggregate gates but fail synthetic retention. Original
synthetic requirements remain IoU >=0.97469, missed fraction <=0.01648, visible FP
<=0.00133, empty covered <=1/320, clear marked 0/80. Raw synthetic IoU also fails:
0.95456 control / 0.95225 border2. Fixing the presence gate alone is insufficient.

Degraded-irregular training: border misses decrease from 281,457 control to
236,685 border2, while outside-target FP increases from 61,632 to 90,967. Core
misses are nearly unchanged (11,279 / 11,198). This matches the intended loss
effect but demonstrates the specificity cost. Additional unweighted updates also
improved fit relative to the parent; that cannot be credited to border weighting.

Preview findings: patterned mask boundaries remain incomplete, lower-face and
object masks spill into visible/background pixels, degraded eye coverings retain
boundary artifacts, glare remains absent, and a covered irregular case is rejected
by the frozen gate. No reviewed end-to-end completion gain has been established.

Next bounded diagnostic: local inference on already cached training examples to
locate false-positive regions (near target boundary versus distant regions) and
check whether degradation/content affects them. Use training membership and cache
provenance checks, original thresholds and fixed case membership. This is needed
before selecting a representation/head change; do not launch a loss-weight sweep,
raise dilation, tune validation thresholds or repeat this recipe unchanged.

Local root: `C:\xampp\htdocs\YEAR 4\Testing\`.
VM experiment: `~/forensic-dgp/expanded_feature_bundle/outputs/expanded_border_training/`.
