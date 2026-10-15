# Frozen bank-path return and training diagnosis — 11 October 2026

The downloaded diagnostic **completed successfully**. Its saved outputs were
independently checked, but **none of the seven variants qualifies as a model**.
Removing the 128-scale readout produces the strongest preview detail score and
still misses detail and preservation requirements. More epochs are not yet justified.

## Returned run and independent audit

The L4 diagnostic took **162.77 seconds** (about 2 minutes 43 seconds)
and used **1.114 GiB peak allocated VRAM**.
It made 312 B forwards and 140 fixed-recognizer forwards, with **zero gradients,
backwards, optimizer updates and epochs**. Archive export took 24.86 seconds.
The 529,617,760-byte archive has SHA256
`2aa1829b56cd9483d0bd4d5f912eae8f4b683966dddc82114a6ebf127ef1d532`. All **1,192 returned files** and
**5,852 retained-parent files** were hash checked. Export success is packaging
evidence and is separate from useful restoration.

Independent saved-array checks covered **700 paired raw/PNG compositions**,
100 baseline cases and **168 native compositions**. The enabled and disabled
controls replayed **200 paired RGB/embedding cases and 48 native raw/PNG outputs
exactly** against the parent L4 run. Zeroing all five readouts equaled disabling
the bank exactly on two source batches. A separate frozen CPU replay covered
every variant on those two batches, **70 compositions**; CPU/CUDA tolerances
apply only to that numerical replay, never to scientific preservation gates.
The rejected B50 state and all buffers remained unchanged. No checkpoint was created.

All **26 planned sheets** were displayed at original resolution, with actual
observations and **1,216 independently checked source cells**. Clearer native
faces retain texture better than original DGP, consistent with the earlier user
feedback. Soft, dark and turned faces still lack eye, nose and mouth detail.
Individual readout removals mostly produce subtle tone changes. No single removal
demonstrates convincing whole-face clarity improvement. Clear glasses and ordinary
visible hair remain; completion was not tested. The implementing assistant's
review is not independent final review. Input-only usable labels stay fixed;
model softness cannot remove a usable DEV case from the review.

## Seven fixed readout comparisons

These are the **same 100 paired photographic TRAIN previews** (20 references,
ten per source, five profiles each), not the full 3,905-case pool. Nine of the
twenty references were exposed in the earlier 50-update run. Unexposed TRAIN
references remain TRAIN evidence. The detail metric is **landmark high-frequency
MSE**, not recognizer feature MSE or recovered identity. Failures overlap groups
and metrics, and are not counts of failed people.

| Frozen variant | Raw / PNG detail error reduction | Raw / PNG group-metric failures | Raw / PNG brightness-only contribution |
|---|---:|---:|---:|
| enabled | 0.617096% / 0.627306% | 7 / 8 | 43.758849% / 43.848320% |
| disabled | 0.716408% / 0.739873% | 10 / 10 | 76.247268% / 76.646960% |
| drop16 | 0.616203% / 0.636225% | 7 / 8 | 40.930767% / 40.991765% |
| drop32 | 0.621167% / 0.622779% | 9 / 9 | 55.078814% / 55.309093% |
| drop64 | 0.652311% / 0.658820% | 8 / 8 | 46.223275% / 46.342573% |
| drop128 | 0.739311% / 0.746189% | 6 / 6 | 23.166817% / 22.335946% |
| drop256 | 0.554659% / 0.571488% | 12 / 12 | 73.531135% / 73.982539% |

All fail the unchanged partial preview criteria. The early detail requirement is
**1%**, with a **20% maximum** brightness-only contribution and retained source/
profile preservation requirements. Drop128 reaches **0.739311% raw / 0.746189%
PNG**, but brightness contribution remains **23.166817% / 22.335946%** and six
overlapping group-metric failures remain. Clear detail error worsens by about
**21.74%** relative to enabled, with worse clear MSE, SSIM and
similarity. Dropping this path is not selected for app adoption or training.
The older parent's full-pool gain of 0.680385% uses a different evaluation scope;
it must not be confused with these preview-subset scores.

## Isolating the learned decoder and bank parameters

Path removal alone did not identify a qualified architecture. A fixed-weight
cross-composition separated the **15 learned current-decoder tensors** from the
**22 learned style/spatial/fusion tensors**; all **755 frozen tensors** remained
exact. It uses the pinned initial B state and rejected B50 state, with no learning.
Initial equals the retained baseline, joint equals the saved enabled model, and
trained-decoder/initial-bank equals the already audited disabled control.
Only the missing **initial-decoder/trained-bank** composition was computed locally.

Frozen CPU inference took **193.76 seconds**, with 50 B forwards,
20 recognizer forwards and zero gradients or updates. Six fresh control batches
cover ten cases per initial/joint/decoder control across the two sources, bounding
CPU/CUDA raw differences at 1e-4; this is not full fresh control replay. All 100
fixed target embedding rows stayed identical to the controls. The new bank-only
outputs are CPU inference while saved controls are L4 output, with that limited
parity evidence declared. Saved pixel/PNG/metric checks covered all **100 new
paired cases and 24 native cases**. All **26 cross-composition sheets** were viewed
and **720 cells** independently checked.

| Frozen cross-composition | Raw / PNG detail error reduction | Raw / PNG group-metric failures | Raw / PNG brightness-only contribution |
|---|---:|---:|---:|
| joint | 0.617096% / 0.627306% | 7 / 8 |
| decoder_only | 0.716408% / 0.739873% | 10 / 10 |
| bank_only | 0.169222% / 0.182174% | 20 / 19 |

Bank-only gives **0.169222% / 0.182174%** detail gain and **20 / 19** overlapping
failures. Its pooled degraded MSE worsens from 0.027559660 to 0.028247220; its
zero brightness fraction is not success when total MSE does not improve.
Decoder-only reaches **0.716408% / 0.739873%**, but brightness contributes
**76.247268% / 76.646960%** and ten failures remain. The bank helps some clear
preservation when combined with decoder learning, but the full learned combination
still misses the retained requirements. Visually, isolating bank changes follows
the original smoothing, while joint/decoder retain more clear texture; blurred
features remain soft. These are fixed-weight counterfactuals, **not independently
trained decoder-only or bank-only experiments**, and do not prove either path's
eventual training capacity.

## Selected next diagnostic before new training

The evidence supports reviewing **loss directions and their interaction** before
changing another architecture or allowing extra epochs. Large combined gradients
and changed weights do not establish useful detail learning. The exact cause is
not proven by these output comparisons. Freeze a manual L4 diagnostic on the same
20 TRAIN references at the **initial and joint B50 states**, with no DEV/final
model inputs, no parameter updates and no optimizer construction.

It separately measures the seven retained objective components (MSE, SSIM,
recognizer, detail, anchors, preservation barrier and brightness guard), plus
clear and degraded MSE reference gradients. It records norms and pairwise products
for the decoder, style, spatial and five fusion scales. A mathematical direction
uses the retained Adam/clipping/weight-decay configuration and saved moments
without mutating them. It is a **hypothetical one-reference first-order direction**,
not the actual accumulated five-reference optimizer update, and cannot prove
finite-step image quality or convergence. Diagnostic use of rejected B50 gradients
is not failed-state training continuation.

Finite scope: **360 gradient queries**, 80 model/ 40 recognizer forwards, zero
backwards/optimizer updates/epochs, 20 TRAIN references, 100 distinct TRAIN cases,
two measured states, no native/final pixels and no checkpoint. Retain full gradient
arrays and mathematical directions for **four reference-state batches**, one per
source per state, so those summaries can be independently recomputed; the other
36 summaries have a narrower audit scope. Do not claim all 40 raw gradients were
independently replayed. No automatic follow-on training or hyperparameter search.

The **255,429-byte packet** passed independent source, archive,
all 5,852 parent hashes, LF checksum, readonly Bash syntax, Python3.10 hashing
compatibility and actual Windows rejection before neural/gradient/output work.
**100 cached metric rows** passed scalar loss-arithmetic checks; fixed synthetic
arrays passed independent NumPy direction/Gram checks. These are arithmetic checks,
not actual L4 gradient verification. Sampled raw-gradient upper bound is
**555,120,096 bytes**, including
metadata reserve. Actual CUDA gradients, runtime and results remain **unrun**.

Require **4 GiB free after installation**, idle L4 and manual tmux. Unmeasured
estimate: **3–10 minutes measurement plus 1–5 minutes export**. Worker cap 15 minutes,
measurement stop 10 minutes checked between reference batches, export 5 minutes;
external caps15.5/ 5.5 minutes with 30-second kill grace. Projection uses the first
two batches with 1.5 safety factor. VRAM 20 GiB; output/reserve 1 GiB each. No current
VM free space, power state or cloud balance is claimed.

[Five exact manual upload/tmux/download steps](<C:/xampp/htdocs/YEAR 4/Testing/CCTV_DGP_BANK_LOSS_ALIGNMENT_V1_VM.md>).
[Returned path audit](<C:/xampp/htdocs/YEAR 4/Testing/outputs/cctv_dgp_bank_path_ablation_v1_analysis/independent_audit.json>);
[Path visual closure](<C:/xampp/htdocs/YEAR 4/Testing/outputs/cctv_dgp_bank_path_ablation_v1_analysis/visual_review.json>);
[Cross-composition audit](<C:/xampp/htdocs/YEAR 4/Testing/outputs/cctv_dgp_bank_partition_v1_analysis/independent_audit.json>);
[Cross-composition visual closure](<C:/xampp/htdocs/YEAR 4/Testing/outputs/cctv_dgp_bank_partition_v1_analysis/visual_review.json>);
[Prepared transfer audit](<C:/xampp/htdocs/YEAR 4/Testing/outputs/cctv_dgp_bank_loss_alignment_v1_transfer_audit.json>).

## Goal and evidence boundaries

The current app checkpoint remains
`646fbb11674e8882ea41e251d6e6e7d468c963979b48c93132def8292834e24b`.
It is the permanent fine-tuned DeblurGAN-v2-compatible baseline. The custom bank
replacement learns our conditioning/reconstruction with a declared external frozen
StyleGAN2 bank; it is not an own-trained StyleGAN or an exact published-DGP replica.
No incumbent is accepted here. Extra epochs and failed-state continuation remain
unqualified. No app/model binding, checkpoint, research cache, split or historical
stop record changed. No agent VM connection, cleanup, training, gradient or power
action occurred; local frozen inference and audits are allowed.

Native ChokePoint DEV remains unpaired, with no native clean-reference PSNR/SSIM.
Paired photographic TRAIN measurements stay separate. Source labels do not establish
ethnicity or Zamboanga performance; no real Zamboanga samples exist. Final identities
remain outside tuning, and independent final review is pending. All five milestones
remain required, including useful reviewed outputs, retained baseline/pretrained/
resize comparisons, qualified primary DGP Auto/override integration and bundled
inline Playwright app verification preserving the existing design.

Completion remains separate for masks, sunglasses, strong glare, hands, obstructing
hair, scarves and objects. Require automatic removal-area preview/optional correction,
preserved clear glasses/ordinary hair/visible appearance, a documented small margin,
original/mask/one plausible estimate, PNG and optional bundle downloads, clearer or
less-covered requests when information is insufficient and separate automatic versus
assisted review. None of these seven covering families is qualified by this work.
Goal **ACTIVE and incomplete**. Reported credits/migration notice and November 2026
deadline remain governed by the canonical plan; no live financial claim is added.
