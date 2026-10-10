# V34 preservation-gradient diagnostic and V35 finite image probe

V34 completed its manual L4 diagnostic and passed the prospective independent
return audit. It measured a missing constraint direction; it did not train a
replacement model. V35 R1 is prepared for manual execution and has not run.
Useful native CCTV restoration and full covering-family qualification remain
unfinished.

The returned archive is `cctv-dgp-group-guard-grad-v34-results.tar.gz`,
1,017,922,963 bytes, SHA256
`0a9a6a2b1dffcedf623c47e6e7d30c711785b28714a591a6d48c4c9e22d968c7`.
Its sidecar, export receipt, regular-file paths and all230 members match.
The VM performed300 per-case queries in52.1923seconds:100 matched photographic
TRAIN cases, original DGP only,23 own-DGP tensors/978,243 values. There were
zero optimizer updates, parameter updates, backwards, epochs or checkpoints.
The external receipt records55.0645seconds within630s plus30s grace; peak
allocated VRAM was1,119,268,352 bytes, within20GiB. Export took59.37seconds.

The independent audit took375.15seconds. All100 new raw outputs exactly equal
the previously audited original-state V33 outputs. Raw guard-value readback
errors are0.0000316911 and0.0000154412, within the prospective0.0001 tolerance.
It separately replayed280 pinned frozen CPU outputs: maximum raw difference
0.000002414, PNG difference one byte, embedding difference0.000001349 and
component difference0.000003875, within the original bounds. The original DGP
and recognizer states were restored; local gradients and optimizer updates
were zero. No return-supplied code was executed.

[Independent return audit](outputs/cctv_dgp_group_guard_grad_v34_independent_audit.json)

## What the diagnostic establishes

The original-state four preservation hinges have zero gradients, as retained
in V32/V33. V34 measures non-hinged observed-support raw MSE, one-minus-SSIM
and one-minus-batchmatched-fixed-ArcFace functions instead. The original
five-case batch context remains. These are diagnostic derivatives, not changed
training losses and not substitutes for delivered PNG metrics.

| TRAIN subset | Nonzero source/profile guard rows | Predicted worsening under its old original-state restoration displacement |
|---|---:|---:|
| Exposed during V32's first50 updates | 51 | 12 |
| Unexposed during those updates | 51 | 12 |

Every positive predicted change above is an ArcFace error direction. Raw MSE
and SSIM directions improve in all17 groups. The aggregate ArcFace error also
worsens: predicted increase0.000645885 in exposed and0.000218131 in unexposed.
Thus this run shows an explicit identity-surrogate conflict, rather than an
example where a good aggregate conceals every bad subgroup. Some source/profile
directions improve while others worsen; all102 rows are kept independently.

Both subsets are TRAIN design data now that their diagnostics guide a proposal.
The word unexposed describes the first50 V32 updates only. It does not mean
held-out development or final evaluation. Source labels do not establish
ethnicity, Asian generalization or Zamboanga performance.

## Changed proposal and limits

The analysis freezes one symmetric proposal: the arithmetic mean of the two
audited original-state V33 fresh AdamW displacements. It is not AdamW applied
to an average gradient. It projects this displacement against all102
non-hinged preservation rows and the six nonzero original restoration-loss
rows, three from each subset. The four zero hinges remain in actual output
measurements; no nonzero row is discarded.

The full108-row certificate passes direct stationarity, nonnegative dual
weights, feasibility and complementary-slackness checks. A separate primal
solve agrees to0.000000000000000044 relative L2 error. Independent group
reassembly from the300 saved case derivatives also passes. Its displacement
norm is0.02340017 versus0.02361228 before projection, retaining99.1017% of
the magnitude. This is a statement about parameter geometry, not restored
facial detail. Nine constraints have positive dual weights. The66-dimensional
Gram rank reflects overlapping groups; every original constraint is verified
after reduction.

Float32 parameter copies produce tiny positive linear guard changes in three
to six rows, maximum0.000000014073 across the four fixed scales. These are
reported without declaring exact nonincrease after quantization. Finite neural
curvature and PNG rounding can produce larger regressions. The original PNG
preservation, source, brightness and capacity gates remain necessary.

GEM motivates projecting an update against protected loss gradients while
explicitly relying on local linearity and representative memory. Its reported
classification experiments do not validate forensic face restoration. This
proposal adapts that mathematical idea to this saved-array diagnostic and
requires actual images to assess it.
[Primary GEM paper](https://proceedings.neurips.cc/paper_files/paper/2017/file/f87522788a2be2d171666752f97ddebb-Paper.pdf)

[Independent mathematical readback](outputs/cctv_dgp_group_guard_grad_v34_analysis_v1_r1/independent_analysis_audit.json)

## Retained implementation failures

The first local analysis computed the arrays but failed JSON serialization:
Python `sum` over NumPy comparisons returned `int64`. Its source, empty summary,
projection and arrays are retained. A distinct R1 explicitly serializes the
count as a builtin integer and encodes JSON before opening the destination.
Independent readback confirms every numeric artifact is hash-identical between
the failed attempt and R1. No geometry or quality threshold changed.

The initial unrun V35 draft used two replay directories that differed from the
pinned V33 basis. Packet review detected this before transfer or model execution.
The draft archive, source, runbook and failure record remain. V35 R1 restores
the validated V27 parent and V28 active module directories. Its recipe,
direction, scales and gates are unchanged. Only R1 is intended for execution.

## Finite manual V35 R1 experiment

The9-file,12,146,657-byte packet contains8 assets plus protocol. Its independent
audit verifies230 V34 return bindings,19 local bindings, all108 projected rows,
Python3.10 syntax for four sources, read-only Bash syntax, three invalid
host/platform/root rejections, Windows rejection before neural work or writes,
eight unsafe-return rejections and the exact R1 root/replay directories.
Geometry stationarity error is0.00000000000000000119. No VM call or local
neural/gradient/optimizer operation occurred in packet verification.

V35 R1 uses the same original DGP, two50-case TRAIN subsets and five-case batch
context. It creates100 original baseline outputs and400 trial outputs at
scales1,1/2,1/4,1/8, resetting the disposable copy between all four trials.
It performs zero optimizer updates, new gradient queries, committed training
updates or epochs. It saves raw outputs separately from unenhanced PNGs and
exports every17-group/source/brightness decision. It creates no checkpoint and
does not resume V32, run an old pilot, select a model or launch follow-on training.

All17 group checks remain: MSE tolerance0.000000000001, SSIM/ArcFace tolerance
0.000001, nonnegative per-source structural gains and brightness fraction at
most0.2. Full-corpus1% at50 and10% at800 requirements remain. This single-step
subset probe cannot satisfy those training-capacity requirements, regardless
of its numbers. Useful eyes, nose, mouth and remaining visible appearance
still need actual review; some softness is acceptable, changed structure is not.

Require the existing running L4/g2-standard-4, idle GPU and6GiB free. The new
probe is estimated at2-6minutes plus1-3minutes export and has not been timed.
Worker900s/external930s plus30s grace, export300s/external330s plus30s grace,
allocated VRAM20GiB and768MiB uncompressed export caps are enforced.

Protocol SHA256:
`ba359d8aa6b3cd6c32b3e8f1c5d459b0af40751414d9f55a9261f3055d44b68b`

Execution archive SHA256:
`954d81811e6881612787608086f27cceb96fa86888a49ee6f52537ca4d3b7ca3`

[Five exact manual steps](CCTV_DGP_GROUP_GUARD_PROBE_V35_R1_VM.md)
[Independent packet audit](outputs/cctv_dgp_group_guard_probe_v35_r1_preparation/independent_packet_audit.json)

## Remaining thesis goal

The existing own-DGP remains the primary local restorer; all14 app/model
bindings remain unchanged. No new native CCTV, paired DEV or reserved final
pixels were opened. V34 is paired photographic TRAIN diagnostic evidence,
separate from the24 frozen ChokePoint C1 unpaired development crops and520
paired synthetic DEV cases. Reserved45 labeled identities/58 crops remain
unopened; no real Zamboanga CCTV sample exists.

Previous functional Auto/override, mask review, PNG/bundle and bundled inline
Playwright evidence remains. The negative automatic and assisted full-family
completion reviews remain binding. Masks, sunglasses, strong lens glare,
hands, obstructing hair, scarves and objects all remain in scope, with clear
glasses/non-obstructing hair and visible appearance preserved. Request clearer
or less-covered input where insufficient information remains. No hidden identity
is claimed. Useful native restoration, full covering-family quality and
independent final review are pending. Goal active/incomplete.
