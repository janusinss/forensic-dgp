# Expanded training-fit audit — 30 September 2026

The returned inference-only audit confirms a training-fit weakness on degraded
synthetic coverings, especially irregular shapes. Neither candidate is eligible
for promotion. No optimizer updates occurred in this audit.

Archive: `outputs/expanded-fit-audit-results.tar.gz`, SHA256
`125f79f9662acad8f7609a8cb7b7c3a08673136beaa50554e2e4425cadd60053`.
Extracted evidence: `outputs/downloaded_expanded_fit_audit/results.json` and
`verification.json`. The execution script matches the local prepared script.
Both checkpoint hashes and the full protocol match the original training return.
All 7,080 case records match the corresponding original cache rows. Every group
was independently summed from case counts; real/synthetic aggregate counts agree
exactly with the original VM training report. Probability gating and count bounds
were checked. Feature arrays and predicted training masks remain on VM: these
checks establish returned-data consistency, not independent pixel-level recounts.

## Training IoU at unchanged 0.5 thresholds

| Group | Fixed placement | Anatomical placement |
|---|---:|---:|
| Real, 68 cases | 0.94397 | 0.94655 |
| Synthetic, 3,472 cases | 0.95704 | 0.94607 |
| Clean irregular, 352 cases | 0.96622 | 0.96789 |
| Degraded irregular, 352 cases | 0.90155 | 0.88405 |
| Degraded eyes, 340 cases | 0.95344 | 0.94650 |
| Degraded lower face, 340 cases | 0.95185 | 0.93703 |
| Degraded object, 352 cases | 0.94859 | 0.94049 |

Raw degraded-irregular IoU is 0.90177 / 0.88515: the presence gate explains
little of this group's deficit. Gated missed-target fractions are 0.09124 /
0.11219. This is evidence against treating the problem solely as transfer to
unseen faces or solely as presence rejection. It does not identify whether the
cause is target construction, representation, head capacity, loss or update budget.
Training geometry differs between arms; their fit scores are not a controlled
comparison on identical targets. Training and validation populations also differ.

## Next bounded diagnostic before another training recipe

Inspect training-only target observability and spatial resolution first.
`expanded_feature_data.py` dilates every degraded target by a 17x17 kernel,
while the head predicts 64x64 logits interpolated to 256x256. Separate original
occluder core from added target border in a fixed, reproducible training sample;
inspect image/geometry/target overlays and target downsample/upsample behavior.
These are diagnostic transformations, not new labels or validation gate changes.
A label-derived reconstruction is not a deployable detector or a proven capacity
ceiling. Keep the original target and evaluation policy until evidence warrants
a separately versioned, documented experiment.

If errors chiefly occupy observable core, investigate representation/head fitting;
if they cluster in the added border, test the border hypothesis in a matched VM
experiment. Do not infer either outcome from aggregate counts. Any actual training
remains VM-only. Glare transfer remains a separate unresolved issue. Existing real
and synthetic safeguards plus end-to-end visual improvement are still required.

Local root: `C:\xampp\htdocs\YEAR 4\Testing\`.
VM root: `~/forensic-dgp/expanded_feature_bundle/`.
