# Earlier replay/consistency reconciliation — 30 September 2026

Checked original returned run.json/metrics.jsonl files and checkpoint hashes,
plus the later corrected-label evaluation. No training, inference or selection
changes occurred. Evidence: `outputs/detector_decision_review/legacy_reconciliation.json`.

The replay and consistency reports agree on 200 source paths/hashes, initial
completion epoch-2 checkpoint, original manifest, split, benchmark, seed42,
10 epochs x21 updates, batch8, LR1e-5 and background penalty0.25. Consistency adds
weight1.0 Bernoulli-KL preservation of the frozen initial segmenter on synthetic
replay. Identical reported settings do not independently prove historical batch
order or optimizer state; original initial tensors were not exported.

| Final epoch10 | Replay | Consistency |
|---|---:|---:|
| Real IoU, original labels | 0.31997 | 0.31523 |
| Synthetic IoU | 0.95141 | 0.95705 |
| Synthetic missed fraction | 0.03905 | 0.03327 |
| Synthetic visible FP | 0.001475 | 0.001487 |
| Synthetic clear cases marked | 0 | 1 |
| V2 corrected real IoU (saved evaluation) | 0.32044 | 0.31568 |

No epoch was selected in either returned run. Both fail original retention.
Consistency improved synthetic overlap at this recipe but did not preserve the
baseline and slightly worsened real-mask overlap. This rejects an unchanged rerun;
it does not establish that every possible distillation method fails.

Parent hash: `c2d4cd180226413af63bcfc2a3e17461cfa95f87aff9a8998b893fc3afc42e93`.
Verified final checkpoint hashes match the V2 evaluation records:
replay `9765e5ba0489101f0046b24a6a83e4da05042f60230385bc90ab69aeb9693519`,
consistency `5db0085b5864e610569f49eb83d129107aafda8e88873e2746393de6a59368f6`.

## Label/ancestry limits

V1 -> V2 changes only validation `new_covered_03.png`; images, membership and
training masks match. V2 -> V3 changes two train masks (`uncovered_05`,
`new_covered_48`), one validation mask (`new_uncovered_16`) and one test mask
(`covered_04`) for glare scope. Membership/image hashes still match.
The legacy original segmenter and later SAM2 feature heads are different model
families with different training targets/budgets. Cross-family score differences
must not be presented as a matched architecture ablation. Legacy runs do not test
the later glare scope; later training-only overfit results show the original
segmenter can fit several real examples, but do not establish generalization.

## Next decision

Do not repeat legacy replay/consistency or extend the unsuccessful head/loss series.
Shift preparation to training-data coverage: audit available real-occlusion source
membership and duplicates before preparing a broader, explicitly reviewed training
annotation queue. The current 68-image real training pool and two glare examples
provide limited variation, while observed failures include patterned coverings,
boundaries, background confusion and glare transfer. This motivates a coverage
hypothesis, not a claim that data quantity alone will solve it.

Protect original validation/test sources and benchmark membership. Keep ambiguous
tiny-highlight proposals disabled; use the already agreed strong-reflection scope.
Any new training labels need image-based review; do not copy candidate predictions
as truth. Preserve existing labels/gates and compare a future coverage experiment
against an equal-budget control. No training package is ready from this review.
Full goal still requires all existing safeguards and reviewed completion gains.

Local root: `C:\xampp\htdocs\YEAR 4\Testing\`.
Legacy VM root: `~/forensic-dgp/outputs/detector_consistency_vm/`.
