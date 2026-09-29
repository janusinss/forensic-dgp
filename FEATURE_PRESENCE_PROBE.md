# Image-level presence gate — 29 September 2026

V3 re-scoring did not qualify the existing heads: all-image-Dice training IoU
0.79075/negativeFP4 of25; nonempty-Dice0.86034/negativeFP9 of25. Image hashes/order
were verified against the cached V2 features; only targets changed. No retraining
or retroactive V2 claim. Evidence `outputs/glare_policy_review/feature_v3_training.json`.

## Bounded training-only probe

Hypothesis: an image-level presence classifier can suppress local false masks
without the coverage loss caused by forcing all pixel probabilities toward zero.
Fixed encoder and nonempty-Dice pixel head; trained only a linear256->1 classifier
on globally averaged SAM2 embeddings with per-image channel layer normalization.
All68 V3 training images,43 covered/25 uncovered; seed42,200 full-batch AdamW
updates,lr0.01,weight decay0.001,equal covered/uncovered mean BCE. Both classifier
and pixel threshold0.5; no threshold search, validation or test fitting.

Final training classifier errors0/68. Applying its gate reduces false-mask cases
9->0, leaves missed covered pixels8.534% unchanged, gives IoU0.86307 and zero empty
covered cases. Pixel-head tensor equality verified. This passes preliminary
training-fit criteria only;68 examples can be memorized and do not establish
generalization. Glare masks remain approximate and sparse.

## Frozen validation completed

Launched one evaluation on all25 V3 validation images using the saved final
classifier and frozen pixel head. No updates, threshold tuning, proposal selection
or target-derived prompts. Report all cases, known mannequin separately and glare
stratum; compare to original detector's V3 baseline with unchanged real gates.
If real gates pass,400-case synthetic retention still must pass before an offline
completion-output comparison. No application changes or promotion.

Local root `c:\xampp\htdocs\YEAR 4\Testing\`; VM counterpart `~/forensic-dgp/`
requires explicit artifact transfer. Local artifacts `outputs/feature_presence_probe/`
(protocol,results,diagnostic classifier) and `outputs/feature_presence_validation/`
(live progress, predicted masks; final results/features on completion). Scripts
`outputs/probe_feature_presence.py` and `outputs/evaluate_presence_v3.py`.
No duplicate launch while process is live.

Process exited successfully;25 cases complete. Gated real validation IoU0.84645,
missed9.580%,visible FP0.9657%,empty1/15,negativeFP0/10. This passes the existing
real gate against the original V3 baseline; raw head IoU0.84267/negativeFP3.
Mannequin-excluded gated IoU0.84399. Ten-row preview inspected, including the
mannequin, glare case and uncovered controls. Mask bodies improve but straps,
edges and patterned regions still have missed pixels/holes.

The only glare validation case is rejected by the presence gate (probability0.0518).
Raw glare IoU0.32986 becomes0 after gating. Thus aggregate success does not prove
the new glare requirement is met. No threshold adjustment was made after seeing
this result. Only2 glare training cases exist, one also has a face mask.

Next: benchmark this same frozen combination on the400 synthetic cases to measure
retention before choosing an improvement. Glare rejection must also be addressed
with training evidence and broader coverage, not a one-case validation tweak.
No deployment and no end-to-end completion gain claimed.

## Synthetic retention launched

`outputs/evaluate_presence_synthetic.py` evaluates the same frozen encoder, pixel
head and presence classifier on all400 original cases. Both thresholds remain0.5;
no fitting or policy changes. Logs both raw and gated masks, pooled metrics and all
ten occlusion/degradation groups; applies original retention gates. Manifest and
model hashes are checked before inference. Expected CPU duration approximately
12minutes. Never duplicate a live process.

Artifacts `outputs/feature_presence_synthetic/`: progress/final JSON, `raw/`,
`gated/`, and exact FP32 `features/` with source-image and encoder hashes. These
features belong exclusively to validation and must never be mixed into training
or replay. Caching supports reproducible evaluation only. About1.6GB of features;
available disk space checked before launch. No VM run or application change.

## Synthetic benchmark complete — candidate rejected

Process exited0; all400 saved raw/gated masks independently rescored and matched
reported metrics. IDs, groups, source counts and unchanged model hashes verified.
All five synthetic retention safeguards fail. No promotion or completion-output
comparison is justified for this candidate.

| Metric | Original baseline | Raw feature head | Presence gated |
|---|---:|---:|---:|
| IoU | 0.97469 | 0.63395 | 0.41404 |
| Missed covered pixels | 1.648% | 34.752% | 57.533% |
| Visible-pixel FP | 0.1333% | 0.4298% | 0.3776% |
| Empty covered cases | 1/320 | 5/320 | 139/320 |
| Uncovered false-mask cases | 0/80 | 22/80 | 12/80 |

Each source contributes200 cases (160 covered/40 uncovered). Asian-source raw
IoU0.64507 -> gated0.26179,empty2->111; FFHQ raw0.62300 -> gated0.56408,empty3->28.
This is a dataset-specific generalization failure, not proof of its cause or an
individual-demographic performance estimate. These datasets differ in more than
demographic composition. Raw masks miss blur margins; the gate adds complete
rejections. Disabling the gate alone cannot qualify.

Evidence: `outputs/feature_presence_synthetic/results.json`, `verification.json`
and fixed first-source ten-case `first_source_preview.jpg`. The real25-case gate
passed, but synthetic retention and glare coverage did not. Goal remains unmet.

Next: prepare one bounded mixed-training recipe for both heads, using V3 real data
plus independent synthetic TRAINING sources balanced across FFHQ/Asian datasets
and covered/uncovered cases. Verify hashes and exclude all benchmark sources first.
Never fit on the400 cached validation embeddings or tune thresholds against them.
Broader glare coverage is still needed; two real training glare examples cannot
establish robust glare completion. No job remains live.
