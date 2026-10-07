# V30 sampling-gradient diagnostic: independently audited results

The diagnostic completed 280 component-gradient queries in 34.729 seconds on the
existing L4. It performed zero optimizer updates, zero backwards and zero epochs,
created no new checkpoint, and preserved the original DGP, stopped V30 candidate
and fixed recognizer states. This is diagnostic evidence, not improved restoration.
The V30 training failure at 50 updates remains: 0.805717% structure gain against
the unchanged 1% early requirement. No app candidate was promoted.

Return: `outputs/cctv-dgp-v30-sampling-gradient-v1-results.tar.gz`,
401,300,114 bytes, SHA256
`f7a8e3d02ac17f61d4e2a48772bb4eb72a1641049f220c66638c08ea44dfa49e`.
All 756 regular, allowlisted returned files passed hash verification. The separate
local audit replayed 200 outputs across original and stopped states using CPU
inference, without autograd or training. Maximum raw disagreement was
0.000002593 (limit 0.00001); PNG disagreement was at most one byte; fixed
recognizer-vector disagreement was 0.0000005253 (limit 0.00005). The audit took
244.704 seconds. Saved gradient sums, partitions, norms, cosines and scalar loss
receipts were recomputed independently from returned arrays.

Authoritative audit: `outputs/cctv_dgp_v30_sampling_gradient_v1_independent_audit.json`.
Arithmetic analysis: `outputs/cctv_dgp_v30_sampling_gradient_v1_analysis/analysis.json`.
Both are bound into the next packet. The audit executes trusted local source;
returned Python and shell files are compared by hash rather than executed.

## Measured gradient behavior

The exposed and unexposed cohorts each contain 50 TRAIN cases with matched
reference repetition. Unexposed means not selected in V30's first 50 updates;
these are approved training photos, not independent evaluation identities.

| State / cohort | Improvement norm | Preservation norm | Improvement/preservation cosine | Batch coherence |
| --- | ---: | ---: | ---: | ---: |
| Original / exposed | 0.369662 | 0 | 0 | 0.618197 |
| Original / unexposed | 0.420801 | 0 | 0 | 0.715432 |
| Stopped50 / exposed | 0.329337 | 0.858172 | -0.343047 | 0.394894 |
| Stopped50 / unexposed | 0.427563 | 5.320449 | 0.099382 | 0.644302 |

All 12 selected original decoder tensors have finite nonzero improvement
gradients in every state/cohort aggregate. Original improvement gradients across
cohorts have cosine 0.800380; stopped-state cosine is 0.646043. The observations
do not support a broken decoder-gradient path or severe initial cohort cancellation.

At stopped50, the infinitesimal negative total-objective direction increases
the whole-observed-detail term in both cohorts (directional derivatives
+0.008154 and +0.067068), while decreasing the landmark-detail term. This is a
measured local objective tradeoff. It does not reproduce historical AdamW updates
or establish a unique cause of V30's failed early requirement.

## Preservation has a measured purpose

Five unexposed cases activate the raw pixel-regression hinge. Four are clear
controls. The largest is `v9_tr_asian_00343_clear`: raw observed pixel MSE rises
from 0.000362100 to 0.000421020, a 16.2719% relative increase. The other clear
increases are 5.9943%, 2.6285% and 2.1719%; one motion case increases 0.7204%.
None of the 50 exposed cases activates that pixel hinge. These are raw float
objective measurements; they are separate from delivered PNG preservation gates
and paired synthetic PSNR/SSIM. All 17 V30 delivered-output preservation groups
still pass at stopped50. Averaged gates do not erase these individual drifts.

Weakening the preservation losses would discard a demonstrated safeguard.
The next single-variable hypothesis instead pairs each reference's clear control
with its four degraded versions in every optimization batch. V29's small TRAIN
capacity experiment used this pairing; its failed development review remains
retained and it is not eligible for app use. V31 tests the pairing across the
full existing 781-reference TRAIN corpus, starting from the original DGP.

## Next bounded experiment

V31 changes batch formation only. Decoder architecture, selected 12 tensors,
original initialization, seven loss weights, fixed initial50 normalizers,
AdamW settings and numerical preservation/structure gates stay fixed. The first
781 updates cover all 3,905 TRAIN cases once; the remaining 19 batches cover 19
distinct references. Every batch contains one clear control and four declared
degradations. It stops at 50 updates unless the existing 1% structure requirement
passes; the maximum remains 800 updates with finite timing, disk and memory caps.
The frozen V30 case permutation determines reference first-appearance order.

Actual VM execution remains the user's manual transfer/tmux workflow. Returned
results require an independent audit and review of all 50 TRAIN preview cases.
Only a passing capacity result justifies repeating the fixed 520 paired DEV
cases and 24 unpaired native development crops. Reserved final identities remain
unopened. A trained/exported archive alone cannot qualify the local application.

No native CCTV or reserved final image participated in this diagnostic. Source
names describe acquisition groups, not ethnicity. No real Zamboanga CCTV samples
or local-performance claims exist. Useful native restoration, all seven covering
families with separate automatic/assisted review, independent final review and
the complete bundled inline Playwright application flow remain outstanding.
