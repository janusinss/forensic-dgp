# Historical V1 commands: stopped before training — 18 October 2026

**ORIGINAL V1 FAILED INITIALIZER PARITY: ZERO NEW UPDATES / NO NEW CHECKPOINT.**
This document is retained as historical evidence. Do not rerun the failed folder
unchanged. [Actual return review](CCTV_CODEFORMER_ROLE_PRESERVING_TRAINING_V1_RETURN_REVIEW.md).
Use the separately prepared [R1 five-step guide](CCTV_CODEFORMER_ROLE_PRESERVING_TRAINING_V1_R1_VM.md); R1 is unrun and
keeps the original restoration objective and quality requirements.

---

# Direct restoration training preparation — 18 October 2026

**Sealed and independently verified; user manual L4 launch pending.**
[Five upload/tmux/download steps](CCTV_CODEFORMER_ROLE_PRESERVING_TRAINING_V1_VM.md).

The last fixed-state measurement established unwanted darkening on all 100 fitted
dark and 32 unused-reference dark observations. Root reviewed all 24 sheets.
[Exact returned evidence](CCTV_CODEFORMER_FIXED_STATE_RESPONSE_V1_RETURN_REVIEW.md).
Its scalar head changes tone and cannot repair spatial facial detail. A lower
combined loss concealed worse dark behavior; neither more head epochs nor a
brightness threshold is justified by that result.

The new pilot changes the actual restoration objective. It initializes from
the retained full-coarse CodeFormer model (515 tensors; 350 trainable and 165
frozen) and creates fresh Adam state. Both normal slots use only untouched-model
RGB L1 and token KL targets, at weights 2 and 0.1. Dark slots retain clean token
CE/latent/coarse/eligible pixel/LPIPS supervision, weights 0.5/1/4/1/1, plus
frozen-parent RGB/token retention at 2/0.1. Role labels select TRAIN targets and
losses; inference receives only the face crop. There is no tone head, normal
brightness-centering term, runtime brightness switch, weight search or old
optimizer/RNG installation. This is a prospective design, not a proven unique
cause or a guarantee of improvement.

Fresh Adam uses the retained base rates 0.00001 for encoder/token parameters and
0.00002 for fusion, with a new 50-update linear warmup. The pilot commits at most
50 updates and 200 microbatches; it does not append epochs automatically. Genuine
initial/final checkpoints, Adam, RNG, sampler, rates, reserved/committed counters,
gradient witnesses, 200 actual training observations/targets and complete 406-case
development outputs are retained. Numerical forward/target parity and real
normal/dark gradient connectivity are checked before optimizer construction.

Normal preservation still compares all 61 clear and 53 normal-blur cases, in RAW
and delivered PNG, against untouched CodeFormer. The unchanged mean/worst
regression limits are 0.01/0.03. Initial full-coarse failures are recorded without
pretending they are the original baseline. At update50, both dark validation
cohorts (53 lowlight and 53 compound) must retain or improve their parent's mean
PNG MAE and SSIM. Native CCTV has no clean-reference metric. No quality threshold
was relaxed, and a trained/exported model is not automatically accepted.

CPU preparation closed 3,362 immutable physical file bindings: 1,434 selected
source/assets/data, 1,624 retained RAW/PNG outputs, and 304 HQ/runtime/metric
witnesses. Actual CPU decoding checked 200 training inputs, 100 direct clean512
targets/HQ array sets and 406 development inputs/targets/support masks. Source
metadata retains the original 391 TRAIN/53 validation split; only 100 selected
TRAIN source pixels/HQ targets are needed for this finite run. Photograph/source
IDs do not prove separate human identities. FINAL images are unused.

Separate source-bound CPU receipts contain 61 tests run, 60 passed and one
unsupported Windows physical-hard-link check skipped. Four additional sealer
fixtures checked canonical gzip/USTAR and rejected an extra zero record, second
gzip frame and bad CRC. Independent model, data, worker, transport and actual
stage reviews found no unresolved issue. These are preparation checks: no local
neural forward, autograd, optimizer, training or VM connection occurred.

Upload: 1,487,646 bytes, SHA-256 `ec0ad9b3a815c8e9ee76e52be60f929a99a24a7a5791c0ff3d4180ee5a561fd8`.
Protocol: `67d71dbb38de308b3f0b9299bc9f34844c51c8445149ce35cdac8e98004f1afb`. All 40 staged members and physical
single-frame archive boundaries passed independent readback. Require 16 GiB
free after installation and retain 4 GiB reserve. The source-derived ledger
bounds payload at about 3.6 GB within a 4 GiB cap and archive within 4.25 GiB;
it is a conservative projection, not measured VM peak usage. Actual worker
limits are documented in the guide and remain authoritative.

The full goal stays useful 256x256 restoration with visible appearance preserved,
clearer-input refusal when insufficient, credited untouched CodeFormer and old
DGP/checkpoints/caches/splits/failures preserved, audited source terms/overlap,
native unpaired and paired synthetic evidence separated, independent FINAL and
user review, then full existing-app Auto/override/PNG/Playwright verification.
No ethnicity, Zamboanga performance or exact hidden identity is established.
Separate covering detection, reviewable removal masks and seven-family plausible
completion remain deferred until restoration acceptance. Storage offload is
already complete. This preparation does not complete the research goal.
