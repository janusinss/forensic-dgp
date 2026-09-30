# Ordinary epoch1 versus constrained pilot — 30 September 2026

Decision: stop this retention-pilot branch. Neither candidate qualifies for
promotion, and repeating the same constrained recipe has no demonstrated benefit.

Inference on the identical73 real training inputs, no optimizer updates. Ordinary
epoch1 checkpoint verified byte-for-byte against the pinned returned projection
archive. Parent, protocol, inventory, replay cache, nominal LR, weight decay,
clipping and loss-weight metadata agree between runs. Existing parent/retained
inference results reused after checking image order; their report hash is recorded.

| All73 real training images | Parent | Ordinary epoch1 | Constrained final |
| --- | ---: | ---: | ---: |
| Mask IoU | 0.071212 | 0.075673 | 0.066046 |
| Missed covered fraction | 0.920358 | 0.916640 | 0.927646 |
| Visible false-positive fraction | 0.017457 | 0.014980 | 0.014085 |
| Empty covered cases | 14/47 | 15/47 | 16/47 |
| Mean supervised loss | 2.907424 | 2.677319 | 2.777603 |

Ordinary training makes a small aggregate coverage gain at this budget, whereas
the constrained pilot regresses. Both remain poor fits. The ordinary run took21
full-rate updates; the constrained pilot accepted18 updates (12 full-rate, six
reduced-rate) from42 trials. They share21 scheduled batches but are not matched
on accepted update dose or compute. Do not claim a clean causal estimate of the
constraint independent of these mechanisms, or claim that all adaptation is
impossible. Earlier four-example fitting already demonstrated limited capacity
to learn masks, while longer ordinary runs failed retention safeguards.

Ordinary epoch1's previously recounted validation realIoU0.072506 passes the
real gate; syntheticIoU0.974844 alone improves, but synthetic missed fraction
0.016647 exceeds parent0.016476, so its synthetic gate fails. The constrained
pilot fails both gates. Neither is a deployable replacement. This comparison
does not justify choosing one metric while ignoring the others.

## Next decision

No further LR, penalty, projection or backtracking variant should be launched
from these results. The repeated original-segmenter adaptations and earlier
frozen-feature heads have not produced complete real masks while preserving the
synthetic safeguards. Consolidate the tested representation/data approaches before
choosing one materially different segmentation initialization or pretraining
strategy. Review primary-source methods and available weights against the actual
mask/glare scope and dataset provenance; do not repeat the already tested SAM2
frozen-head variants under a new name. A new candidate should first receive a
fixed, inference-only benchmark where applicable, then a bounded VM pilot only
if evidence supports it. This is the next research/implementation decision, not
an instruction to train or a claim that any particular replacement will work.

The automatic single-image detector remains part of the goal. Supplying manual
masks may isolate completion quality during evaluation but does not satisfy the
automatic detector requirement or justify marking the project complete.

Evidence: `outputs/retention_epoch1_comparison/results.json`;
script: `scripts/compare_retention_epoch1.py`.
Ordinary checkpoint SHA256:
`9a3e14c77f606debf767531941f16639841eec84bafb4ee99f43abfbf0300d34`.
Local root: `C:\xampp\htdocs\YEAR 4\Testing\`.
VM source: `~/forensic-dgp/coverage_vm_bundle/outputs/projection_training_vm/ordinary/epoch_1.pth`.
No generator/application change, new training, threshold tuning or gate relaxation.
