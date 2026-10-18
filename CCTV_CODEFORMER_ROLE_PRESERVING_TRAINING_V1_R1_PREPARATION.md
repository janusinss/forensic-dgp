# Historical R1 commands: stopped before training — 18 October 2026

**R1 FAILED INITIALIZER LOGIT PARITY: ZERO NEW UPDATES / NO NEW CHECKPOINT.**
This document is retained as historical evidence. Do not rerun the failed folder
unchanged. [Actual return review](CCTV_CODEFORMER_ROLE_PRESERVING_TRAINING_V1_R1_RETURN_REVIEW.md).
Use the separately prepared [R2 five-step guide](CCTV_CODEFORMER_ROLE_PRESERVING_TRAINING_V1_R2_VM.md); R2 is unrun and
keeps the restoration objective and quality requirements. Its initializer
numerical logit test changes explicitly; see the new guide before uploading.

---

# CodeFormer restoration training R1 preparation — 18 October 2026

**Sealed and independently verified source transfer; manual L4 run pending.**
[Five upload/tmux/download steps](CCTV_CODEFORMER_ROLE_PRESERVING_TRAINING_V1_R1_VM.md). [Original V1 return review](CCTV_CODEFORMER_ROLE_PRESERVING_TRAINING_V1_RETURN_REVIEW.md).

Original V1 failed its exact candidate/full-coarse initializer forward comparison
before optimizer construction. Its actual archive has 460 regular members and
37,688,585 payload bytes, including 100 fresh HQ-reference cache sets. Independent
CPU auditing matched all 200 HQ arrays to retained parent bytes and canonical
array payload hashes. The failure records **zero new restorer updates, zero Adam
constructions, zero backward/gradient calls and no new checkpoint**. It therefore
provides no new restoration-quality result. Original source, return, checkpoints
and failure evidence remain preserved.

R1 repairs the comparison instrumentation without changing the training objective.
Candidate and frozen-parent initializer parity uses the same input, eval/no-grad
context, RNG and helper. Separately captured unscoped training-mode outputs are
compared with the gradient-isolated training route. Arrays, module modes and RNG
are saved before the exact comparison gates. The extra two training-capture
forwards are explicitly budgeted. Source and CPU checks support this repair;
its numerical/gradient behavior remains unverified until the user's L4 run.

The pilot trains CodeFormer's actual restoration network: 515 parameter tensors,
350 trainable and 165 frozen. It starts from the retained full-coarse research
checkpoint with fresh Adam; it does not append to original V1's zero updates or
load historical optimizer/RNG/sampler state. Normal slots use untouched-model full
RGB L1/token KL behavior at weights 2/0.1. Dark slots retain clean token CE,
latent, full-coarse, eligible pixel/LPIPS terms at 0.5/1/4/1/1, plus frozen-parent
RGB/token retention at 2/0.1. Roles select TRAIN losses; inference uses only the
face crop. There is no tone head, normal brightness-centering, runtime brightness
switch or weight search. These choices are prospective, not a promise of improvement.

Fresh Adam retains base rates 0.00001 for encoder/token parameters and 0.00002
for fusion, with a new 50-update linear warmup. Maximum 50 committed updates,
200 microbatches/backward calls, five gradient queries, 1,399 neural forwards
and 1,854 reservation events. Evaluation parity and real normal/dark gradient
connectivity must pass before Adam construction. Initial/final or stopped
checkpoints, optimizer/RNG/sampler/counters and actual training records are retained.

Unchanged acceptance gates compare all 61 clear and 53 normal-blur development
cases with untouched CodeFormer in RAW and delivered PNG, at mean/worst
regression limits 0.01/0.03. Initial parent failures stay recorded. At update50,
both 53-case dark validation cohorts must retain or improve parent mean PNG
MAE and SSIM. All 406 DEV outputs require independent review. Native CCTV is
unpaired and receives no clean-reference score. No quality threshold is relaxed.

CPU preparation closes 3,362 immutable physical file bindings: 1,434 selected
sources/assets/data, 1,624 retained RAW/PNG outputs and 304 HQ/runtime/metric
witnesses. Data scope remains 200 training observations from 100 selected TRAIN
references, 100 direct clean512/HQ target sets and 406 DEV inputs/targets/support
masks. Historical metadata retains the 391 TRAIN/53 validation split. Source and
photograph identifiers do not prove independent human identities. FINAL pixels
remain unused. Source-bound CPU and independent stage/transport reviews are
recorded in the preparation folder; no local neural training or VM run occurred.

Upload **1,353,807 bytes**, SHA-256 `318f466d5738bdd51fea25a0e0aafd4a00b645f84cb4fcee5cb498bd1b6091ad`.
Protocol `4edacae4f1e61f06b21fb1308423398d5b9f842def160c02f74f8edc0975fc40`. All **32 staged regular members**
passed independent physical single-frame gzip/USTAR readback and source binding.
Require 16 GiB free after installation and retain a 4 GiB reserve. Worker cap:
3,000 seconds; initialization600, witness180, optimization900 and each snapshot600.
Preflight300/export600, plus30 termination grace. Source-derived storage bounds
remain prospective; measured VM timing/resources and actual state are audited
after return. Manual launch remains the user's next action.

The full goal remains useful 256x256 restoration, visible-appearance preservation,
insufficient-input refusal, user acceptance, independent FINAL review and the
existing app Auto/override/PNG/Playwright flow. Credit CodeFormer accurately;
preserve untouched CodeFormer, original DGP, trained states, caches, splits,
provenance, terms and failures. Keep paired synthetic and source-separated native
unpaired evidence distinct; no ethnicity or Zamboanga performance claim. Separate
covering detection/mask review/seven-family plausible completion remains deferred
until restoration acceptance. Storage offload is complete; this preparation
does not complete model qualification or the research goal.
