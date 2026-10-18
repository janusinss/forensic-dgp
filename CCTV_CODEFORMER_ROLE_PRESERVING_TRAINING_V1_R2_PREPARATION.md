# Historical R2 training guide — actual 50-update run retained

**R2 ALREADY RAN: GENUINE TRAINED MODEL / PRESERVATION QUALITY STOP.**
Do not rerun this historical training folder unchanged or append epochs.
Use [the local research testing guide](<C:/xampp/htdocs/YEAR 4/Testing/CCTV_CODEFORMER_R2_LOCAL_TEST_GUIDE.md>) and
[the actual independently audited return review](<C:/xampp/htdocs/YEAR 4/Testing/CCTV_CODEFORMER_ROLE_PRESERVING_TRAINING_V1_R2_RETURN_REVIEW.md>).
Dark retention passed, preservation failed; the trained model remains experimental.

---

# CodeFormer restoration training R2 preparation — 18 October 2026

**Sealed and independently verified source transfer; manual L4 run pending.**
[Five upload/tmux/download steps](CCTV_CODEFORMER_ROLE_PRESERVING_TRAINING_V1_R2_VM.md). [Latest R1 return review](CCTV_CODEFORMER_ROLE_PRESERVING_TRAINING_V1_R1_RETURN_REVIEW.md).

Both original V1 and R1 stopped before Adam construction: **zero new restorer
updates, zero gradients/backwards and no new checkpoint**. V1 mismatched
evaluation/training comparison contexts. R1 corrected the contexts and exported
467 regular files / 46,082,287 payload bytes, preserving both actual initializer
captures. Independent auditing found native512, RAW256, encoder latent and all
256 selected tokens exactly equal. Only logits differ: 223,123/262,144 values,
maximum absolute difference 0.00000762939453125. All 200 fresh HQ target arrays
match their historical parent bytes and canonical payload hashes. Neither run
reached the restoration objective or a development quality bank. Original V1/R1
sources, returned arrays/contexts/RNG and failures remain preserved.

R2 changes only the prospective initializer **numerical test policy**. Float32
logits must satisfy declared PyTorch 2.9 defaults, atol0.00001 and rtol0.0000013;
all 256 argmax indices and saved tokens must be identical, and every parent
top-two margin must exceed twice the corresponding maximum logit error.
Native/RAW/latent/token arrays, model state, contexts and RNG stay exact.
The initializer gate therefore changes explicitly; restoration-quality gates
do not change. This does not establish the unknown low-level numerical cause.
[PyTorch 2.9 assert_close](https://docs.pytorch.org/docs/2.9/testing.html#torch.testing.assert_close).

The candidate/parent comparison retains the same input, eval/no-grad context,
RNG and helper. Training-mode unscoped/scoped route checks remain separate;
their existing exact loss/array requirements are retained. Compared arrays,
actual contexts, RNG and differences are saved before rejection. No additional
neural forward, standalone diagnostic, rate search or automatic retry is added.
The source/CPU-tested policy and actual neural gradient behavior remain
unverified until the user's manual L4 training run.

The pilot trains CodeFormer's actual restoration network: 515 parameter tensors,
350 trainable and 165 frozen. It starts from the retained full-coarse research
checkpoint with fresh Adam; it does not append to either failed run's zero updates or
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

Upload **1,424,772 bytes**, SHA-256 `c10957cd804d0844e67ecc4ee7990c810677d9b172e069631443654f0f5709ad`.
Protocol `82fcd34cada1eed238ef65a341c22580d12146e3ee30c2ff3094312d3ba1c85a`. All **43 staged regular members**
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
