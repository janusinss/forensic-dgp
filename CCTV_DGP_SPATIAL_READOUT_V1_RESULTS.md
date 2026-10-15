# Spatial reconstruction V1 returned results — 14 October 2026

**USED / QUALITY FAILED.** Both routes completed 50 manual L4 updates, but
neither meets the unchanged structure and preservation requirements. There is
no accepted development update, full-corpus extra epoch, new incumbent or app
promotion. All 20 planned comparison sheets / 500 cells were directly inspected.
The original app checkpoint remains `646fbb11674e8882ea41e251d6e6e7d468c963979b48c93132def8292834e24b`.

## Actual endpoint results

| Route | RAW structure gain | Delivered PNG gain | Failed RAW / PNG group–metric checks | Decision |
|---|---:|---:|---:|---|
| A | 0.115752% | 0.132699% | 8 / 8 | Rejected |
| B | 0.095199% | 0.111965% | 18 / 18 | Rejected |

The early requirement is ≥1% globally and in each source, with all 17 RAW and
17 PNG groups preserved. These percentages measure relative reduction of the
fixed facial high-frequency error; they are not recognition accuracy or proof
of recovered identity. Failed checks are overlapping group–metric records,
not counts of distinct bad faces. Neither RAW nor delivered PNG qualifies.

| Route | Source | RAW gain | PNG gain |
|---|---|---:|---:|
| A | Asian-source replay | 0.158301% | 0.169444% |
| A | HQ FFHQ counterparts | 0.105081% | 0.123480% |
| B | Asian-source replay | 0.443368% | 0.440975% |
| B | HQ FFHQ counterparts | 0.007887% | 0.029419% |

These are 20 already exposed photographic TRAIN references / 100 synthetic views,
ten references per source. `dataset/thumbnails128x128` is the historical key;
the actual targets are the audited HQ FFHQ counterparts. Asian-source replay
is reported separately, without ethnicity or capture-location inference. No
native CCTV, unexposed DEV, final identity or completion evidence was used.
The 50 updates per route equal 2.5 passes through this small sample, **zero full-
corpus additional epochs**. Archive completion cannot change those roles.

## Location, cause and next action

Location: the route endpoints in returned `outcome_A.json` and `outcome_B.json`;
the supervised training phase exits 2 intentionally for the retained quality stop.
Cause of rejection: insufficient global and source structure gain plus actual
preservation regressions. This is a model-quality failure after computations,
not a missing-file, disconnected-gradient or CUDA-loading traceback.
Next action: retain both failed states and review the saved learning path before
choosing a new finite training change. Do not rerun this recipe or add epochs.

A regresses low-light RGB errors in both sources and some compound-view RGB,
SSIM and fixed-recognizer checks. B adds clear-image and blur/motion regressions;
its delivered clear aggregate MSE worsens
7.288%.
B's RAW/PNG brightness-control ratio is
1.004843 /
1.004555, above the 0.2 limit.
The constant-channel-shift counterfactual accounts for approximately 100% of
its aggregate MSE gain; values slightly above 1 mean that counterfactual improves
the score more than the candidate. This is not a percentage of all image changes.
A's aggregate brightness ratio is 0, but its other failures remain.

All 15 A and 25 B selected tensors changed. Every tensor has a finite nonzero
derivative in the recorded manual preflight, and all five B kernels acquired
nonzero neighboring coefficients. Fixed observation features, normalization,
conditioning, external StyleGAN2, anchor readouts and recognizer stay unchanged.
The bank remains an external pretrained face prior; this is not an own-trained
GAN or an exact reproduction of a published DGP method.

All 50 updates in each route hit the common original-relative distance boundary
`0.040184702871782822`. A projection scales range
0.579484–0.723629;
B scales range0.283042–0.601460.
B spends 52.381% of final squared parameter displacement on the new spatial
connections and 47.619% on the original decoder. This demonstrates an active
constraint and differing allocation, not proof that relaxing it would be safe
or sufficient. The scientific bound and appearance gates remain unchanged.

Saved-array evaluation of the declared weighted training formula gives mean
objective5.200000 initially,
5.113825 for A and5.872913 for B.
A lowers this objective on 20/20 references, with much of the improvement in
the clear RGB anchor, while the structure term barely improves. B is worse on
19/20 references, with added anchor, regression and photometric costs. A lower
weighted objective is therefore insufficient evidence of the actual gate.
This uses fixed saved arrays and 15-image embeddings; training used 5-image
embedding batches. No new GPU objective or derivative was replayed locally.

## Visual evidence and audit coverage

A is very close to the original in all 20 sheets. B mainly shifts tone; degraded
eyes, nose, lips and teeth stay diffuse. Clear forced restorations often retain
less texture than the input/target. Neither route demonstrates a convincing
useful new whole-face detail gain. Severe compound views remain insufficient
for readable features; request a clearer crop in that setting. Existing visible
glasses, ordinary hair, cloth and hands remain in these restoration comparisons;
their presence does not qualify automatic covering removal or hidden completion.

The2,107,676,784-byte archive SHA256 is
`3490fd2955cd9895f7dba1e6a198c72272d7e0910f80eb7840bd9767cd140765`. Import verifies 781 archive members,
all 15 declared packet files plus protocol and 203 parent assets. The prospective
local checker recomputes 500 RAW/PNG compositions, all source/profile reductions,
original/initial parity, exact observed exterior pixels, both17-group gate
decisions, all 100 projection formulas, final vectors, frozen states and
optimizer receipts. Every 500 gallery cells exactly matches its pinned source.
The original checkpoint and documents stayed hash-identical throughout auditing.

The manual worker reports 258.21 s including cache/parity/preflight/endpoints;
actual fits are 30.45 s for A and 86.04 s for B. Manual fresh CUDA replay verifies
20 cases in 4.14 s. Local saved-output audit took 133.03 s. These are finite measured
times, not a promise for a larger study. Independent CUDA gradient replay and
independent final human review remain absent; the local work performs zero
learned-model construction, neural forwards, gradients or optimizer updates.

At the authorized maintenance inventory,14 October 13:31:39 UTC
(21:31:39 Asia/Manila), the VM had 84.343 GiB free. Cleanup was unnecessary;
no files were deleted and no agent launched training. Training was already
running under the human's manual workflow, later completed/exported, and the
human downloaded this return. Original checkpoints, caches, splits, states,
provenance, masks and every quality stop remain retained.

[Independent saved-output audit](<C:/xampp/htdocs/YEAR 4/Testing/outputs/cctv_dgp_spatial_readout_v1_analysis/independent_audit.json>).
[Parameter and source attribution](<C:/xampp/htdocs/YEAR 4/Testing/outputs/cctv_dgp_spatial_readout_v1_analysis/learning_attribution.json>).
[Fixed-array objective review](<C:/xampp/htdocs/YEAR 4/Testing/outputs/cctv_dgp_spatial_readout_v1_analysis/saved_objective_audit.json>).
[Every directly reviewed sheet](<C:/xampp/htdocs/YEAR 4/Testing/outputs/cctv_dgp_spatial_readout_v1_analysis/visual_observations.json>).
[Next evidence review](<C:/xampp/htdocs/YEAR 4/Testing/CCTV_DGP_POST_SPATIAL_READOUT_REVIEW_V1.md>).

The goal remains active/incomplete: useful native DEV, untouched final review,
qualified primary integration, app-flow regressions and all seven automatic/
assisted covering families remain required. No current model is declared best
possible or successful merely because this finite comparison completed.
