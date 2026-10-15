# Source/profile R1 measurements and next finite comparison — 12 October 2026

**The manual diagnostic completed and its saved evidence passed independent
audits. It measured learning conflicts; it did not train or improve a checkpoint.**
Runtime: **118.28 seconds**. Scope: 20 paired photographic
TRAIN references, 100 cases, 32 distinct source/profile loss groups, 320 gradient
queries, 40 B and 20 fixed-recognizer forwards. There were zero backwards calls,
optimizer updates, parameter displacements or epochs. Peak allocated CUDA
memory was 3.051 GiB and peak RSS
2.935 GiB. The diagnostic reported unchanged initial
model/buffer, original-restorer and recognizer states. No new checkpoint exists.
Those state claims are checked against saved receipts; no local neural replay
or new local gradient query was performed.

The **227,356,876 byte** archive, SHA256
`f9cccc2fd05e1a52c6bb5713a60e458a711caa4c685d812e2b217f4adfea3d03`, has 89 verified
file bindings. All 5,852 parent bindings, 100 byte-identical raw controls, 32
aggregate-vector norms/dots/loss means, 320 query-scope bindings and four saved
sample vectors passed checks. Two clear-MSE sample derivatives also agreed
with prior measured normalized derivatives after conversion to raw loss units.
All320 neural derivatives were not independently recomputed. No new delivered
PNG quality or visual result is inferred from the frozen control arrays.

[Independent measurement audit](<C:/xampp/htdocs/YEAR 4/Testing/outputs/cctv_dgp_bank_group_gradients_v1_r1_analysis/independent_audit.json>).

## Demonstrated limitation

At this initial model state, the old ordinary direction predicts increasing
loss in **10 of32 groups**; its two-reference projected version
does so in **7 of32**. The conflicts include low-light and
compound-degradation groups. For example, the projected direction's raw-MSE dot
is +3.5166 for `dataset/asian_faces/lowlight_lr32`, and +13.1256 for
`dataset/thumbnails128x128/compound_lr24`. Positive dots mean an initial loss
increase for the old additive direction. These are mathematical predictions;
the earlier finite R2 outputs already independently failed preservation.

The old design constrained a small reference average. That average did not
represent every source/profile. More epochs following the same direction are
not justified by this diagnostic. Dataset labels describe source partitions;
they establish neither ethnicity nor performance on Zamboanga CCTV.

## Changed design and its limits

A minimum-L2 direction was derived from all32 saved group vectors. It targets
fractional structure descent for each source and a positive preservation margin
for every one of30 source/profile MSE, SSIM-loss and ArcFace-loss groups. Raw
losses and unit-gradient normalization are declared separately. A dual NNLS
solve, KKT checks, a separate zero-initialized primal SLSQP solve and exact
float32 direction/dot reconstruction passed. The smallest normalized-Gram
eigenvalue is 7.2369185e-05; the independent primal
direction discrepancy is 3.1163931e-07.

Research on [multi-objective learning](https://proceedings.neurips.cc/paper_files/paper/2018/hash/432aca3a1e345e339f35a30c8f65edce-Abstract.html)
motivates checking competing task gradients. Our inequality-constrained
direction is a project-specific design, not that paper's implementation or a
convergence guarantee. All derived gains are local first-order predictions.
Finite network behavior, quantization and visible quality still need checking.

[Independent saved-array direction audit](<C:/xampp/htdocs/YEAR 4/Testing/outputs/cctv_dgp_bank_group_gradients_v1_r1_analysis/common_direction/independent_audit.json>).

## One prepared manual forward comparison

| Additive step | Minimum predicted structure gain per source | Minimum predicted preservation-loss reduction per group |
|---:|---:|---:|
| 0.005 | 0.5% | 0.05% |
| 0.01 | 1% | 0.1% |
| 0.02 | 2% | 0.2% |
| 0.04 | 4% | 0.4% |

These are predictions and step coefficients, not completed epochs. The direction
L2 norm is 0.63478916; the largest prospective
displacement is 0.02539157,
0.04030%
of the selected initial-parameter L2 norm. No model was displaced locally.

The fresh **2,838,507 byte** packet is verified and **UNRUN**.
It uses the same20 references/100 inputs, exact initial controls, affines,
frozen original/prior/recognizer and PNG policy. It changes the derived
direction and its four step coefficients; original used packets and failures
remain intact. Every proposal resets to initial state. No optimizer, autograd,
training epoch or saved replacement checkpoint is planned.

It plans 100 B/160 recognizer forwards, 100 exact controls, 400 raw outputs,
400 PNGs, saved raw/PNG embeddings and20 eight-column review sheets. The retained
all17-group MSE/SSIM/ArcFace preservation gates and brightness limit remain
unchanged. A new additional requirement prevents one sampled source's structure
gain from hiding a weak source: each source must reach1% on raw and PNG outputs.
This strengthens the probe; it does not relax any historical failed gate.
Saved-gate fixtures confirm that a passing average cannot hide a failed source.

Estimate **2–5 minutes** from the previous same-size54-second forward run plus
verification/export; this exact new direction is unmeasured. Require **4 GiB**
free after installation; recheck at launch. Worker1200s/measurement900s,
external1230s; export300s/external330s plus30s kill grace. VRAM20GiB/RSS12GiB,
output768MiB/reserve1GiB. Explicit startup storage failures are retained before
any neural import. Scientific failure, nonfinite state, mismatched controls or
resource limits retain a stop; no automatic retraining or historical relaunch.

[Exact five manual steps](<C:/xampp/htdocs/YEAR 4/Testing/CCTV_DGP_BANK_GROUP_DIRECTION_FORWARD_V1_VM.md>).
[Sealed transfer audit](<C:/xampp/htdocs/YEAR 4/Testing/outputs/cctv_dgp_bank_group_direction_forward_v1_transfer_audit.json>).

## Goal status and next decision

Goal ACTIVE. Neither diagnostic completion nor this direction qualifies more
epochs or a new incumbent. The next manual return must pass saved-output audits
and every planned visual sheet review before a finite training design is
considered. Full TRAIN, audited native CCTV DEV and independent reserved final
identity review remain separate requirements. Synthetic paired PSNR/SSIM are
not unpaired CCTV evidence.

The permanent original checkpoint SHA256
`646fbb11674e8882ea41e251d6e6e7d468c963979b48c93132def8292834e24b`
is retained. Its DeblurGAN-v2-compatible origin and the replacement's own
condition/reconstruction path with externally pretrained frozen StyleGAN2 prior
remain declared. Pretrained restoration models remain comparison baselines.
No random-weight replacement, app promotion or exact hidden-identity claim.

All five milestones, preserved app design, DGP primary/Auto/override, insufficient
input handling, independent final/app review and all seven covering families
remain required. Masks, sunglasses, strong glare, hands, obstructing hair,
scarves and objects need separate automatic and assisted results; clear glasses,
ordinary hair and visible appearance must be preserved. Preview editable removal
area, original/mask/one plausible estimate, PNG and optional bundle downloads
remain required. The fresh three-width browser layout audit is partial evidence;
the complete model/mask/download flow and quality scope are not newly qualified.

All actual gradients, parameter proposals and training remain human-manual L4
tmux work. Agent VM access remains authorized inventory-first, hash-bound
maintenance only. No agent launched a new VM neural job. Credits/migration
notifications and the November2026 deadline stay as previously agreed.
