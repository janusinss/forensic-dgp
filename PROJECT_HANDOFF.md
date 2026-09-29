# Forensic DGP: workspace handoff and training runbook

## Latest pixel audit — 29 September 2026

`PIXEL_BOUNDARY_AUDIT.md` and `outputs/pixel_boundary_audit/results.json` document
570 cases: 68 real training, 77 verified partial synthetic training, and all 425
validation cases. Local inference only. Synthetic validation missed pixels are
81.85% near a fixed four-pixel boundary band; 60.28% of false positives are far
from boundaries. Errors are not solved by simply expanding masks. Training
evidence supports testing local context in the pointwise pixel head.
Next: implement an initialization-matched 1x1 versus 3x3 pixel-head VM comparison,
with unchanged loss/data/budget and frozen spatial presence gate. Architecture
implementation, GPU execution and quality improvement remain unproven. Existing
model/application defaults remain unchanged; all fitting must use the VM.

## Latest validation — 29 September 2026: spatial presence improves, deployment still rejected

Presence comparison archive returned. Both 20-epoch final heads evaluated on
25 real + 400 synthetic cases with unchanged pixel predictions and threshold 0.5.
Spatial lowers synthetic misses from 52/320 to 5/320 and raises composed-mask IoU
0.76566 to 0.87321. Independent recount verified all 850 masks. Both still fail
synthetic retention and reject the sole real glare validation case. No deployment.
See `PRESENCE_COMPARISON_RESULTS.md` and `outputs/presence_comparison_validation/`.
Next: analyze raw pixel boundary/region errors before specifying VM-only pixel
training. Spatial remains a research component; no further global-gate repeat.
All local work was inference/testing, with no local optimizer updates.

## Latest preparation — 29 September 2026: matched presence comparison ready

`scripts/compare_presence_heads_vm.py` is ready to upload and run on the existing
VM bundle; exact commands are in `PRESENCE_COMPARISON_VM.md`. Global 1x1 versus
spatial 4x4 pooled linear heads, fresh initialization, identical fixed training
recipe and 468 examples. Parameter counts differ (257 versus 4097), so this is
an architecture comparison rather than a capacity-controlled causal test.
Encoder and pixel detector unchanged; all training VM-only. Three tests pass,
Python compilation passes, no local optimizer updates. GPU run remains pending.
Return `~/forensic-dgp/feature_vm_bundle/presence-comparison-results.tar.gz` to
`C:\xampp\htdocs\YEAR 4\Testing\outputs\` for evaluation. Existing raw segmentation
still fails retention; this experiment alone cannot qualify the full pipeline.

## Latest diagnostic — 29 September 2026: gate failure confirmed

VM audit received and verified; see `FEATURE_GATE_AUDIT_RESULTS.md`.
400 synthetic training cases: gate misses 32/320 covered, concentrated in object
(18/80) and irregular (11/80) coverings. Five classifier false positives on clear
images correspond to three nonempty composed masks, consistent with training.
CPU/GPU encoder parity on 20 variants from two source images: zero presence
decision flips, maximum probability delta 0.0003445, at most two changed mask pixels.
Saved/fresh GPU embeddings match on those samples. No optimizer updates in audit.

Next: implement a matched VM comparison of spatial versus global presence heads,
with frozen pixel/encoder weights, identical training data and fixed thresholds.
The current raw segmentation also fails synthetic retention; improving presence
alone cannot qualify this candidate. No application promotion or local training.
Evidence under `C:\xampp\htdocs\YEAR 4\Testing\outputs\downloaded_feature_mixed_audit\`;
VM original evidence under `~/forensic-dgp/feature_vm_bundle/outputs/feature_mixed_audit/`.

## Latest result — 29 September 2026: VM pilot evaluated, candidate rejected

Read-only next diagnostic prepared: `scripts/audit_feature_mixed_vm.py`.
Upload this one file to the VM home directory and run with the bundle's `.venv`
Python and `--root ~/forensic-dgp/feature_vm_bundle`. It audits all 400 saved
synthetic training features at the unchanged 0.5 gate threshold, grouping errors
by source, covering kind, degradation and mask area. It also compares CPU/CUDA
encoder paths for 20 cases selected by source/kind/degradation before inspecting
predictions, including saved-GPU versus fresh-GPU reproducibility. No fitting.
Two arithmetic tests passed; Python compilation passed. GPU run pending.
Return `~/forensic-dgp/feature_vm_bundle/feature-mixed-audit-results.tar.gz`.
The real 68 feature embeddings were not persisted by the training runner, so
this cached training audit is explicitly synthetic-only, not a full real audit.

The returned `outputs/feature-mixed-vm-results.tar.gz` completed 20 epochs on an
NVIDIA L4. Extracted into `outputs/downloaded_feature_mixed_vm/`; sent/returned
bundle inventories match. Fixed local inference evaluated 25 real + 400 synthetic
cases; independent saved-mask recount verified all 425 cases. No local training.
Gated real IoU 0.83710 passes aggregate safeguards, but glare still fails.
Gated synthetic IoU 0.72867, 69/320 covered masks empty, 14/80 clear cases falsely
marked: synthetic retention fails. Raw predictions also fail retention. Do not
promote this checkpoint or change application/generator defaults.

Evidence: `FEATURE_MIXED_VM_RESULTS.md`,
`outputs/feature_mixed_validation/results.json`, `verification.json`, and
`diagnostic_preview.png`. Next: VM training-feature error audit plus a small
CPU/GPU inference parity check before specifying another training experiment.
VM remains `~/forensic-dgp/feature_vm_bundle/`; local remains
`C:\xampp\htdocs\YEAR 4\Testing\`. Earlier pending-training notes below are history.

## Latest update — 29 September 2026: mixed detector pilot packaged for VM

**Ready for VM preflight, not yet GPU-tested or trained.** Portable package:
`C:\xampp\htdocs\YEAR 4\Testing\outputs\feature-mixed-vm-bundle.tar.gz`
(155,184,930 bytes, 299 archive members), SHA256
`8033ea88f353a182a96d9b1e85c64120b4bd3868eb40aae0382e043222ab766a`.
Extract into `~/forensic-dgp/feature_vm_bundle/`. Exact upload, setup, tmux,
training and result-return instructions are in `FEATURE_VM_RUN.md` in this
Windows workspace and the VM bundle. No git pull is required for this isolated
package, and no commit/push was performed.

Four runtime tests passed (CPU refusal, low VRAM refusal, balanced complete
sampling, runner refusal before data reads). Both Bash scripts passed syntax
checks; Python compilation passed. All packaged file hashes were checked after
archiving. No local optimizer updates were run. VM preflight verifies CUDA,
bundle/source/split integrity and one forward pass. Full GPU execution is pending.
The run recomputes all 468 embeddings on CUDA, trains only two heads for 20 epochs,
and exports `feature-mixed-vm-results.tar.gz`. Local evaluation follows after the
user returns that archive; original validation safeguards and application baseline
remain unchanged. This is not yet evidence of better completed faces.

**User execution constraint: training must run on the Google Cloud VM, not locally.**
Stopped local mixed-feature session during feature caching, before optimizer
updates. No training results or final checkpoint were produced. Preserve partial
cache for provenance; do not resume this training runner locally. Local work may
prepare code/data and evaluate results. Next: package a portable VM runner and
preflight instructions for `~/forensic-dgp/`; user pastes SSH commands because
this workspace has no configured VM connection. Earlier running notes below are
historical and superseded. The user requested continuation; the goal tool currently
reports paused, and agents cannot change that tool status to active.

**Historical, stopped local attempt — not currently running:**40 verified training sources
(20Asian/20FFHQ),4196 excluded hashes,400 fixed training variants plus68 V3 real
examples. Frozen encoder; train pixel/presence heads for20epochs/1600updates,
six balanced groups per batch. No validation feature reads or threshold tuning.
Recipe/provenance `FEATURE_MIXED_TRAINING.md`; local artifacts
`c:\xampp\htdocs\YEAR 4\Testing\outputs\feature_mixed_training\`, VM counterpart
`~/forensic-dgp/outputs/feature_mixed_training/` after transfer only. Check live
process before restarting. Next: finish fixed budget, evaluate final checkpoint
against unchanged real/synthetic safeguards; glare and completion still unproven.

**Presence synthetic benchmark completed; candidate rejected:** all400 cases
independently verified. Raw IoU0.63395; gated0.41404,empty139/320,negativeFP12/80;
all five original synthetic safeguards fail. Gated Asian-source IoU0.26179 versus
FFHQ0.56408. No model promoted. Real gate success does not establish overall
success or glare handling. No job remains live. Details `FEATURE_PRESENCE_PROBE.md`;
local `outputs/feature_presence_synthetic/verification.json`, VM mapping after
transfer. Next: prepare one bounded, source-balanced mixed TRAINING-data recipe
for both feature heads; verify no validation-source/cache leakage first.

**Frozen presence synthetic benchmark running:**400 original validation cases,
same encoder/pixel/presence weights and thresholds0.5; hashes verified. Local
artifacts `c:\xampp\htdocs\YEAR 4\Testing\outputs\feature_presence_synthetic\`
(VM `~/forensic-dgp/outputs/feature_presence_synthetic/` only after transfer).
Exact cached features are VALIDATION ONLY, never training/replay inputs. Check the
live session before restarting; no duplicate launch. See `FEATURE_PRESENCE_PROBE.md`.
Glare remains unresolved despite real aggregate gate passing; no promotion.

**Presence-gated real validation passed, scope still incomplete:**25 V3 cases
finished; IoU0.84645,visible FP0.9657%,empty1/15,negativeFP0/10; mannequin-excluded
IoU0.84399. Ten-row preview inspected. The single glare case is incorrectly
rejected (presence0.0518), so this is not completion of the glare requirement.
Next: frozen400-case synthetic retention; address glare using training evidence,
not validation-specific threshold changes. No job currently running, no promotion.
Report `FEATURE_PRESENCE_PROBE.md`, artifacts `outputs/feature_presence_validation/`.

**Presence-gate feasibility passed; frozen real validation running:** V3 cached
image hashes verified; previous pixel heads still failed. A separate linear
image-level classifier trained on68 V3 training embeddings removes9 negative
false-mask cases with no covered case rejected; training IoU0.86307. Pixel head
and encoder unchanged. This is training fit, not generalization. One frozen25-case
V3 validation launched; no threshold tuning. Details `FEATURE_PRESENCE_PROBE.md`,
local `outputs/feature_presence_validation/` under `c:\xampp\htdocs\YEAR 4\Testing\`;
VM equivalent under `~/forensic-dgp/` after transfer. Synthetic retention and
completion-output comparison remain required; no promotion.

**V3 finalized and original baseline re-evaluated:** user accepted all4 marked
reflection types. Manifest SHA256
`e36ce5cf04c858d61885c2a0187d0182099ada9852eb03d21d645f5bdbb3ea18`;
standard manifest checks passed. Train43/25 covered/uncovered, validation15/10,
test4/3. Original detector V3 validation IoU0.060648, empty6/15, negativeFP1/10;
single glare-validation image missed. No model training or promotion. Pending notes
below are historical. Next: re-evaluate saved feature heads with V3 targets before
choosing another experiment; reuse embeddings only after verifying image hashes.
Local `outputs/glare_policy_review/v3_baseline.json`, VM equivalent under
`~/forensic-dgp/` after transfer. Full policy/provenance: `OCCLUSION_POLICY_V3.md`.

**V3 full source review complete:**100 crops inspected,14 glasses cases enlarged;
96 unchanged proposals and4 glare additions (2train/1validation/1previously-seen
test). All hashes, split assignments, binary masks and preservation of V2 masked
pixels checked. Preview `outputs/glare_policy_review/proposals.jpg` under local
`c:\xampp\htdocs\YEAR 4\Testing\` (VM `~/forensic-dgp/` after transfer).
User clarification pending on dark scene reflections/blue glare versus white glare;
proposal labels remain disabled for training. See `OCCLUSION_POLICY_V3.md`.

**User policy decision — strong lens glare included:** estimate facial regions
obscured by strong lens glare; preserve transparent areas and visible detail.
`OCCLUSION_POLICY_V3.md` defines the version transition. Created a separate100-case
pending review queue at local `dataset/detector_glare_review_v3/audit_queue.json`
(VM `~/forensic-dgp/` mapping after transfer). Training is disabled for this queue;
V2 labels/splits are unchanged. Next: consistent full-dataset glare review and
proposed masks, then versioned baseline reevaluation. Do not retroactively count
V2 false positives as successes or relabel only failed examples.

**Uncovered-case audit complete:** all26 training negatives checked for both
feature heads; verified4/10 false-mask cases. Treatment areas1–737 pixels/65536,
including high-confidence eyeglass reflections plus teeth/chin/clothing/background.
All10 treatment failures visually inspected. V2 labels unchanged. The user
clarification is resolved by the V3 decision above; full review remains pending.
Details/artifacts in `FROZEN_FEATURE_PROBE.md`.

**Matched loss comparison completed:** control training IoU0.79353/negativeFP4,
nonempty-Dice treatment IoU0.86296/negativeFP10 (26 uncovered cases). Both fail
fixed training-fit criteria. Coverage improves at the cost of more false masks;
no promotion or validation run. Next: inspect uncovered-face error locations,
areas and confidence before another model change. No diagnostic job remains live.

**Loss audit and comparison setup (historical):** frozen-head covered/uncovered
gradients oppose in6/6 balanced batches; empty-target Dice dominates uncovered BCE.
Testing one change with identical cached features, initial weights, batches and
20epoch budgets: all-image Dice versus zero Dice on empty targets (BCE retained).
No validation/test fitting. Local artifacts `outputs/feature_empty_dice_comparison/`
under `c:\xampp\htdocs\YEAR 4\Testing\`; VM counterpart `~/forensic-dgp/` requires
transfer. Do not relaunch a live process. See `FROZEN_FEATURE_PROBE.md` for evidence
and protocol; this diagnostic does not change application models or selection gates.

**Frozen-feature diagnostic completed:** final training IoU0.73410, missed
pixels24.949%, visible FP0.3217%, zero empty masks/42 covered and3 false masks/26
uncovered. Failed the predeclared IoU>=0.80 and negativeFP<=2 training-fit criteria.
All68 saved masks independently re-scored; fixed10-row preview inspected. No
validation or deployment. Next: training-only loss-component audit of late recall
decline while loss decreased; no immediate repeat training. Details in
`FROZEN_FEATURE_PROBE.md`. Process exited successfully; no feature job remains live.

**Diagnostic setup (historical):** frozen SAM2 image embeddings with a small prompt-free
occlusion head on all68 V2 training images only.20 fixed epochs/220updates; encoder
and application models unchanged. Three head tests passed. Protocol, hashes and
predeclared training-fit criteria are in `FROZEN_FEATURE_PROBE.md`. Artifacts under
local `c:\xampp\htdocs\YEAR 4\Testing\outputs\frozen_feature_probe\`; VM counterpart
`~/forensic-dgp/outputs/frozen_feature_probe/` exists only after transfer. Check the
live process before relaunching. Validation/test fitting is excluded; even a
successful training fit does not establish retention or completion-output gains.

**Final SAM2 result:** all400 synthetic cases completed, zero execution failures;
IDs, strata and pixel counts verified. Refined synthetic IoU0.74929 versus original
baseline0.97469, missed pixels17.243% versus1.648%, visible FP1.5362% versus0.1333%,
uncovered false masks1/80 versus0/80. Four of five synthetic safeguards fail.
Real V2 IoU improved to0.43854, but combined selection fails. Inspection confirms
both missed blur margins and whole-face/person selections. No promotion, no app
change, no generator training. Goal remains active. Detailed evidence and next
training-only pretrained-feature feasibility hypothesis: `SAM2_DETECTOR_BENCHMARK.md`.
Local `c:\xampp\htdocs\YEAR 4\Testing\outputs\sam2_synthetic_validation\summary.json`;
VM equivalent `~/forensic-dgp/outputs/sam2_synthetic_validation/summary.json` requires
transfer. The earlier progress notes below are historical; the process exited0.

**SAM2 benchmark setup (historical):** implemented and tested image/detector-only prompt
adapter `detector_refinement.py` (not connected to app);20 focused tests passed.
Official SAM2.1 tiny source/weights are pinned by revision/hash in
`SAM2_DETECTOR_BENCHMARK.md`. Isolated CPU dependencies under outputs; no project
PyTorch replacement. Real V2 validation25: raw consistency IoU0.31568 -> refined
0.43854, visible FP1.398% ->1.141%, empty masks2/14 unchanged, uncovered FP1/11
unchanged. Wrong-object prompts remain serious failures; no promotion. Frozen
400-case synthetic validation launched locally and saves per-case progress under
`c:\xampp\htdocs\YEAR 4\Testing\outputs\sam2_synthetic_validation\` (VM equivalent
`~/forensic-dgp/outputs/sam2_synthetic_validation/` only after transfer). Check its
live process/session before resuming; no duplicate launch. Goal stays active;
synthetic retention and completion-output review are outstanding.
Prepared `outputs/summarize_sam2_validation.py` to require all400 unique expected
case IDs, matching strata and valid pixel counts before writing `summary.json`.
It uses pooled confusion counts (the same metrics as detector training), reports
each occlusion/degradation group and applies the original synthetic baseline gates.
Run it only after the live benchmark writes `results.json`; partial cases are not
selection evidence. Script and artifacts currently exist locally only.

**Probability audit complete:**99 thresholds on all68 real training images only,
no validation tuning. Best unconstrained training IoUs: initial0.10300,
no-penalty-epoch4 0.33929, consistency-epoch10 0.43505; corresponding visible FP
rates2.963%,10.883%,5.901%. Under initial training FP/empty-case constraints,
no-penalty has no qualifying grid threshold and consistency reaches only0.35682.
No deployed threshold changed. This supersedes the pending probability audit below.
See `DETECTOR_PROBABILITY_AUDIT_RESULTS.md` in local
`c:\xampp\htdocs\YEAR 4\Testing\` / VM `~/forensic-dgp/`; local evidence under
`outputs/detector_probability_audit/`. Next distinct candidate: pretrained SAM2
boundary refinement using input/detector-derived prompts only, with abstention and
unchanged validation safeguards (now in progress as reported above). Manual or
ground-truth-derived prompts must not be presented as automatic performance.
Goal stays active; no detector or completion model has been promoted.

**Active-goal penalty ablation complete:** mixed replay at lr1e-5, penalty0 versus
verified0.25 control, same four-epoch/84update budget and source membership.
Penalty0 final real-validation IoU0.28058 versus0.20196 control, but visible false
positives3.146% versus1.377%; synthetic IoU0.96869 versus0.97244. All four new
epochs failed both gates; no promotion. Saved four diagnostic checkpoints locally
and verified generator tensors unchanged. Application baseline retained. See
`DETECTOR_PENALTY_COMPARISON_RESULTS.md` at both repository-root mappings and local
`c:\xampp\htdocs\YEAR 4\Testing\outputs\detector_penalty_comparison\` (VM counterpart
`~/forensic-dgp/outputs/detector_penalty_comparison/` only after transfer).
Next bounded work: training-only probability separation/precision-recall audit,
without validation threshold tuning, to distinguish calibration from representation
failure before further training. Goal remains active; end-to-end improvement is
not yet demonstrated. This supersedes the pending penalty comparison below.

**Loss audit now complete:** evaluated output-logit gradients on all68 training
images and parameter gradients on six balanced training batches at initial and
consistency-epoch10 checkpoints, without optimizer steps. Weighted visible penalty
opposed combined BCE/Dice gradients in4/6 initial and3/6 trained batches; median
relative gradient norms were1.536 and0.343. Uncovered examples did not consistently
dominate. This is a loss-balance hypothesis, not proof of the bottleneck. Details:
`DETECTOR_LOSS_AUDIT_RESULTS.md` in both repository-root mappings. Local artifacts:
`c:\xampp\htdocs\YEAR 4\Testing\outputs\detector_loss_audit\`; VM counterpart
`~/forensic-dgp/outputs/detector_loss_audit/` only after explicit transfer.
Next proposed bounded test: mixed replay at lr1e-5 with hard-visible weight0 versus
0.25, same data/seed/budget and unchanged real/synthetic validation gates. No new
VM training or production-model change. This supersedes the pending audit below.

**Follow-up real-only comparison also complete:** matched lr1e-4,84updates and exact
real-image batch subsequences against the completed mixed replay control. Final
training/real-validation/synthetic IoUs were0.33008/0.23944/0.60377 real-only versus
0.38737/0.26627/0.85874 replay. All real-only epochs failed synthetic retention;
generator tensors were unchanged and no weights were saved. Starting checkpoint,
V2 manifest and baseline metrics matched. Removing replay also increases the real
contribution to the batch-averaged loss, so this does not isolate gradient conflict.
See `DETECTOR_REAL_ONLY_COMPARISON_RESULTS.md` at the local/VM repository roots.
Local evidence: `c:\xampp\htdocs\YEAR 4\Testing\outputs\detector_real_only_comparison\`;
VM counterpart `~/forensic-dgp/outputs/detector_real_only_comparison/` requires
explicit transfer. Keep replay; no further unchanged VM run. Next proposed step:
training-only loss/gradient contribution audit before changing the objective.
This supersedes the pending real-only comparison below.

The local comparison continued after the interrupted conversation and completed
both four-epoch arms; it was not restarted. All 68 real training examples were
seen each epoch, with identical synthetic replay sources, initialization, seed and
update budget. Original Phase 4 split hash and replay-source membership were now
independently checked locally. Corrected V2 validation and separate mannequin
reporting were used; test cases were not used.

Final lr 1e-5: training IoU 0.24240, real-validation IoU 0.20196, synthetic IoU
0.97244. Final lr 1e-4: 0.38737 / 0.26627 / 0.85874 respectively. The higher-rate
real-validation peak was epoch 2 at 0.35362, but synthetic IoU was only 0.86137.
All eight candidates failed synthetic retention. Generator integrity checks passed;
no model weights were saved or deployed. Do not send either recipe to the VM as a
proven improvement. This supersedes the earlier pending-learning-rate comparison.

Details: `DETECTOR_LR_COMPARISON_RESULTS.md` in local
`c:\xampp\htdocs\YEAR 4\Testing\` (VM counterpart `~/forensic-dgp/`). Local evidence:
`outputs/detector_lr_comparison/results.json`, `PROTOCOL.md`, and
`validation_masks.jpg`; VM copies exist only after explicit transfer. Next proposed
diagnostic: a matched real-only versus replay comparison to isolate training-fit
limitations; no further GPU launch yet. Current production baseline is retained.

## Latest update — 28 September 2026: continue in this workspace

**Training-only overfit diagnostic completed locally:** two 80-update runs on three covered training examples plus one uncovered training control, native 256px, original completion epoch 2 initialization. Current lr 1e-5 reached training IoU 0.89130; diagnostic lr 1e-4 reached 0.98975 (initial 0.17920). Higher-rate final missed coverage was 1.00%, visible false positives 0.0046%, uncovered false masks 0/1. Generator tensors stayed unchanged; no weights saved, validation/test images used, or deployment performed. This proves fit capability on these examples only, not generalization or synthetic retention. Details in `DETECTOR_OVERFIT_RESULTS.md`; local artifacts `c:\xampp\htdocs\YEAR 4\Testing\outputs\detector_overfit_diagnostic\` (VM counterpart `~/forensic-dgp/outputs/detector_overfit_diagnostic/` only after transfer). Next: a bounded full-training-split learning-rate comparison with replay, training-fit monitoring, corrected validation and unchanged retention gates; do not simply adopt the higher rate for production. This supersedes the pending-overfit recommendation below.

**Label-v2 comparison completed:** created `dataset/detector_expanded_review_v2/` under local `c:\xampp\htdocs\YEAR 4\Testing\` (VM counterpart `~/forensic-dgp/dataset/detector_expanded_review_v2/` only after transfer). Preserved all 100 images/splits and every training/test mask; refined one validation mask and flagged the known mannequin for separate reporting. The original label already excluded most cheek skin, contrary to the earlier thumbnail-based description; the actual defects were boundary inaccuracies and upper-edge overshoot. Both label versions remain approximate assistant annotations.

Re-evaluated initial, real-only epoch 9, replay epoch 10 and consistency epoch 10 on the same 25 validation inputs. V2 IoUs: 0.06078 / 0.33786 / 0.32044 / 0.31568. Excluding the one known mannequin: 0.06444 / 0.34771 / 0.33835 / 0.33475. Rankings are unchanged; covered-pixel misses remain above 61% in each adapted model. No checkpoint is promoted and no training was run. Details: `DETECTOR_LABEL_V2_RESULTS.md` at either repository root; local JSON/overlays under `outputs/detector_label_v2_review/`. Next: a small training-only overfit diagnostic to distinguish learnability from generalization failure before another VM run. This supersedes the older pending-label-correction recommendations below.

**Follow-up implementation:** audited all 42 covered training labels and 14 covered validation labels. `new_covered_03.png` validation polygon includes exposed cheek through a mask cutout; `new_covered_40.png` is a mannequin. Original labels/splits remain fixed for comparability. See `~/forensic-dgp/DETECTOR_LABEL_AUDIT.md` ↔ `c:\xampp\htdocs\YEAR 4\Testing\DETECTOR_LABEL_AUDIT.md` for limitations and commands.

**Consistency experiment now completed and reviewed:** the frozen-original-detector Bernoulli KL term (weight 1) did not produce a qualified checkpoint. All ten CUDA epochs finished; all synthetic retention gates failed. Do not rerun the launcher unchanged. Original implementation checks passed 50 focused tests and Bash syntax; smoke exports remain separate from this full run.

Received `c:\xampp\htdocs\YEAR 4\Testing\outputs\detector-consistency-results.tar.gz`, SHA-256 `21cb7a5d828aabcc8e3cbdf35fe26891d3b34f45c9ed274eb95bb8f5122d5557`. VM `~/forensic-dgp/outputs/detector_consistency_vm/` is extracted locally to `c:\xampp\htdocs\YEAR 4\Testing\outputs\downloaded_detector_consistency\outputs\detector_consistency_vm\`. Review artifacts are at local `outputs/detector_consistency_review/` (VM counterpart only after explicit transfer).

| Epoch-10 comparison | Plain replay | Consistency |
|---|---:|---:|
| Real IoU | 0.31997 | 0.31523 |
| Real missed covered pixels | 64.82% | 65.34% |
| Completely missed real masks | 1/14 | 2/14 |
| Synthetic IoU | 0.95141 | 0.95705 |
| Synthetic uncovered false-mask cases | 0/80 | 1/80 |

The modest synthetic IoU gain does not offset poor real coverage or meet the original synthetic baseline (0.97469). There is no selected best_detector.pth and no demonstrated completion improvement. Keep existing application weights and manual region correction. Next: correct the documented label defect in a versioned dataset, distinguish mannequin cases in reporting, and rerun the existing baselines on that fixed evaluation before choosing a further training intervention. Do not simply increase the consistency weight or number of epochs; this comparison does not establish either as beneficial.

The user confirmed work continues here at `c:\xampp\htdocs\YEAR 4\Testing\`, with GPU training at `~/forensic-dgp/`. Maintain this document at meaningful implementation, training and review milestones; no workspace transfer is currently required. The older handoff instructions below remain available as a runbook, not an instruction to move work.

The mixed detector replay GPU run is complete. Received archive: `c:\xampp\htdocs\YEAR 4\Testing\outputs\detector-replay-results.tar.gz`, SHA-256 `75b1584b0debdbb96001e7979c2cf47f0db59f03adbc98cb4563e6bb6254afdd`. VM run `~/forensic-dgp/outputs/detector_replay_vm/` was safely extracted to `c:\xampp\htdocs\YEAR 4\Testing\outputs\downloaded_detector_replay\outputs\detector_replay_vm\`.

| VM validation metric | Initial detector | Replay epoch 10 |
|---|---:|---:|
| Real mask IoU | 0.06039 | 0.31997 |
| Real covered pixels missed | 93.18% | 64.82% |
| Real visible-pixel false-positive rate | 1.814% | 1.404% |
| Synthetic mask IoU | 0.97469 | 0.95141 |
| Synthetic uncovered cases with false masks | 0/80 | 0/80 |

All ten epochs completed on CUDA (21 updates/epoch, batch 8, learning rate 1e-5). **No epoch passed strict synthetic retention; no best_detector.pth was selected.** Epoch 10 is a diagnostic candidate, not an approved replacement. Replay reduced forgetting relative to the earlier real-only epoch 9 (synthetic IoU 0.91828, false masks 9/80), but does not solve real-mask coverage.

Local checks verified the recorded initial checkpoint, reviewed-label manifest and synthetic benchmark hashes; all ten checkpoints preserve generator tensors exactly. All 200 unique replay source byte hashes match local files and do not overlap the known benchmark or reviewed-real source/crop hashes. Full original-split membership is recorded by the VM split hash and was not independently revalidated locally in this review. The seven previously inspected real test cases were not reused for this run's selection.

Review artifacts: `c:\xampp\htdocs\YEAR 4\Testing\outputs\detector_replay_review\` (audit JSON and ten-row validation mask comparison); VM equivalent `~/forensic-dgp/outputs/detector_replay_review/` exists only if explicitly transferred. Phase 3 remains the restoration baseline; Phase 5 full GPU results are still not supplied.

**Next step:** inspect real-validation label consistency and incomplete predicted-mask coverage, then design a bounded detector experiment that addresses those errors while preserving synthetic behavior. Do not repeat the same run, relax selection thresholds merely to obtain a best file, or train the completion generator on raw masked images as clean targets. A frozen-original-detector consistency term on training-only synthetic replay is a candidate to test, not an implemented or proven improvement. Completion quality and a fresh independent holdout remain required before deployment.

**Historical snapshot: 27 September 2026; superseded where noted by the latest update above.** Local workspace: `c:\xampp\htdocs\YEAR 4\Testing\`; training VM: `~/forensic-dgp/`. The following execution section remains a reference; do not repeat the now-completed replay run.

**Work continues here per the latest user instruction.** Keep this handoff current for future use. Read the latest update before running older pilot commands. Use Playwright whenever inspecting or testing the web interface.

## September 27 workspace state and dataset inventory

| Track | Implementation and current decision | VM path ↔ Windows local path |
|---|---|---|
| 1: Restoration | Phase 3 baseline retained. Phase 4 epoch 27 improved PSNR but reduced identity similarity. Phase 5 differentiable ArcFace identity loss and EMA are implemented; full GPU pilot and output review remain pending. | `~/forensic-dgp/checkpoints/dgp_zamboanga_final.pth` ↔ `c:\xampp\htdocs\YEAR 4\Testing\checkpoints\dgp_zamboanga_final.pth` |
| 2: Completion/inpainting | Separate gated U-Net generator and covering detector; custom two-epoch pilot completed without a selected best model. CodeFormer inpainting benchmark completed. Real-mask detection remains the immediate bottleneck; train the detector with real + synthetic replay while freezing the generator. | `~/forensic-dgp/COMPLETION_TRAINING.md` ↔ `c:\xampp\htdocs\YEAR 4\Testing\COMPLETION_TRAINING.md` |
| Current detector recipe | Implemented replay trainer, 47 focused tests passed, one-update local mechanics run and Bash syntax check passed. Full replay VM run remains pending. No production checkpoint replacement. | `~/forensic-dgp/DETECTOR_REPLAY_TRAINING.md` ↔ `c:\xampp\htdocs\YEAR 4\Testing\DETECTOR_REPLAY_TRAINING.md` |

All repository-relative paths in commands and historical sections resolve against these two roots: Linux `~/forensic-dgp/`, Windows `c:\xampp\htdocs\YEAR 4\Testing\`. Slash-separated suffixes have the same meaning on both machines; do not pass Windows paths to Linux. Git transfers code/docs, not ignored datasets, weights or run artifacts. Verify `git status --short` and `git rev-parse HEAD` on each machine; an old revision elsewhere in this document is historical, not the current commit.

| Dataset | Verified local inventory | VM directory | Windows local directory | Source/use |
|---|---:|---|---|---|
| GREATGAMEDOTA FFHQ | 70,000 files | `~/forensic-dgp/dataset/thumbnails128x128/` | `c:\xampp\htdocs\YEAR 4\Testing\dataset\thumbnails128x128\` | Kaggle `greatgamedota/ffhq-face-data-set`; clean-face baseline |
| Asian Demographic Prior | 10,000 JPG images | `~/forensic-dgp/dataset/asian_faces/` | `c:\xampp\htdocs\YEAR 4\Testing\dataset\asian_faces\` | Hugging Face `hiennguyen9874/face-age-gender-asian`; demographic-prior source, not proof of Filipino representativeness |
| Real Occlusion Review | 1,510 JPG images; zero TXT files remaining | `~/forensic-dgp/dataset/real_occlusion_review/` | `c:\xampp\htdocs\YEAR 4\Testing\dataset\real_occlusion_review\` | YOLO TXT annotations purged; raw photos alone are not pixel masks or clean reconstruction targets |
| Face Mask Dataset Candidate | Source designation; not an additional verified image count | Same real-occlusion review directory above | Same real-occlusion review directory above | [Kaggle hughiephan/face-mask](https://www.kaggle.com/datasets/hughiephan/face-mask/data), user-provided provenance; designated for real-world occlusion training/evaluation to bridge the synthetic-to-real gap |

Counts above were rechecked locally during the handoff update; VM inventory must pass Stage 0. Do not count the candidate source as a second independent dataset. License/consent suitability and identity-disjointness are not established by these counts.

Reviewed detector labels: `~/forensic-dgp/dataset/detector_expanded_review/manifest.json` ↔ `c:\xampp\htdocs\YEAR 4\Testing\dataset\detector_expanded_review\manifest.json`. There are 100 images: train 68 (42 covered/26 uncovered), validation 25 (14/11), earlier test 7 (4/3). Polygon annotations are approximate assistant-reviewed labels. The seven test images have now been inspected; do not call them untouched or reuse them for checkpoint selection.

Measured completion status:

- Initial custom pilot: epoch 2 predicted-mask hole MAE 0.0867; neither epoch selected. Do not repeat unchanged.
- CodeFormer benchmark: 400 cases, zero failures, hole MAE 0.07355 and visible MAE 0.02413. Preprocessing comparison reported 0.06344 and 0.00886 respectively; inspect its specific arm/configuration before attributing the gain to weights.
- Real-only detector: visible-penalty epoch 9 selected on real validation, IoU about 0.337. Synthetic IoU fell from 0.97469 to 0.91828, with false masks on 9/80 uncovered cases. Not promoted.
- Next controlled experiment: 10 epochs × 21 updates, batch 8, two examples from each real-covered/real-uncovered/synthetic-covered/synthetic-uncovered group. Starts from original completion epoch 2, not the forgetting-prone epoch 9. Synthetic replay uses 200 original-training source images, excluding known benchmark, validation/test and reviewed-real hashes.

Review evidence lives locally at `c:\xampp\htdocs\YEAR 4\Testing\outputs\detector_balanced_review\REPORT.md`; its VM counterpart would be `~/forensic-dgp/outputs/detector_balanced_review/REPORT.md` **only after explicit artifact transfer**. The VM run is `~/forensic-dgp/outputs/real_detector_balanced_vm/`; its downloaded local counterpart is `c:\xampp\htdocs\YEAR 4\Testing\outputs\downloaded_detector_balanced\outputs\real_detector_balanced_vm\`.

## Autonomous `/goal` execution specification

Copy the following specification into the receiving agent. This is a proposed goal, not a claim that a goal or cloud job has been started. VM access is currently through user-pasted SSH commands; if the receiving agent has no configured VM connection, provide the relevant block for the user rather than pretending it ran remotely.

```text
/goal Continue Forensic DGP from the September 27 snapshot in PROJECT_HANDOFF.md.
Local workspace: c:\xampp\htdocs\YEAR 4\Testing\; VM workspace: ~/forensic-dgp/.
Execute the four stages below sequentially and record commands, revision, hashes,
metrics and failures. Stage 0: verify GPU, dependencies, datasets and preserved
split; run isolated one-batch smoke checks. Stage 1: run the bounded Phase 5
restoration pilot and mixed real/synthetic detector replay pilot sequentially
inside tmux, preserving the generator and existing baseline. Stage 2: apply each
track's selection gates and inspect fixed previews; no automatic deployment or
extra epochs when a candidate fails. Stage 3: export artifacts with checksums,
update evidence-based docs and synchronize the receiving local workspace.
Do not train on real validation/test images or treat masked photos as clean face
targets. Hidden facial structure is a plausible estimate. Use Playwright for web
testing. Report the concrete next step at each stopping point.
```

### Stage 0: pre-flight verification

Run in VM SSH (local equivalent root is `c:\xampp\htdocs\YEAR 4\Testing\`). Push reviewed local code first; stop on Git conflicts, missing artifacts or failed checks.

```bash
cd ~/forensic-dgp
git status --short
git pull --ff-only origin main
if [ -d venv ]; then source venv/bin/activate; fi
python3 -m pip check
nvidia-smi --query-gpu=name,memory.total,memory.free --format=csv
python3 - <<'PY'
import json
from pathlib import Path
import torch
assert torch.cuda.is_available(), 'CUDA unavailable'
print('PyTorch', torch.__version__, 'GPU', torch.cuda.get_device_name(0))
free,total=torch.cuda.mem_get_info()
print('VRAM free/total GiB:', free/2**30, total/2**30)
roots={'dataset/thumbnails128x128':70000,'dataset/asian_faces':10000,
       'dataset/real_occlusion_review':1510}
for folder,expected in roots.items():
    root=Path(folder)
    assert root.is_dir(), folder
    count=sum(p.suffix.lower() in ('.jpg','.jpeg','.png') for p in root.rglob('*') if p.is_file())
    assert count==expected,(folder,count,expected)
assert not list(Path('dataset/real_occlusion_review').rglob('*.txt'))
split=json.loads(Path('outputs/phase4_with_progress/split.json').read_text())
a,b=split['train'],split['validation']
assert len(a)==76000 and len(b)==4000, 'Wrong split; do not substitute smoke split'
norm=lambda paths: {str(Path(p.replace('\\','/')).resolve()) for p in paths}
ta,tb=norm(a),norm(b)
assert len(ta)==len(a) and len(tb)==len(b) and not ta & tb
assert all(Path(p).is_file() for p in ta|tb), 'Missing split source files'
for p in ('checkpoints/dgp_zamboanga_final.pth','outputs/completion_pilot/epoch_2.pth',
          'dataset/detector_expanded_review/manifest.json',
          'outputs/completion_pretrained_vm/manifest.json'):
    assert Path(p).is_file(),p
print('Preflight passed. Replay additionally excludes content hashes across known held-out data.')
PY
tmux new-session -A -s dgp_training
```

Inside tmux, run the isolated dry runs below. Their outputs map to local `c:\xampp\htdocs\YEAR 4\Testing\outputs\handoff_phase5_smoke\` and `...\outputs\handoff_completion_smoke\` after transfer. Use fresh names if they already exist. Phase 5 launcher prepares the landmark cache before training; this can scan all 80,000 images even for a one-batch smoke. A one-batch run does not establish full-batch VRAM capacity or output quality.

```bash
cd ~/forensic-dgp
if [ -d venv ]; then source venv/bin/activate; fi
OUTPUT_DIR=outputs/handoff_phase5_smoke bash scripts/run_phase5_gcp.sh --dry_run --batch_size 1 --num_workers 0
OUTPUT_DIR=outputs/handoff_completion_smoke bash scripts/run_completion_gcp.sh --dry_run --batch_size 1 --num_workers 0
```

### Stage 1: training execution

Use `tmux new-session -A -s dgp_training` to enter the session. Run one GPU job at a time. If a named full run already exists, inspect it first; do not overwrite it or silently repeat training. Track 1 outputs map from `~/forensic-dgp/outputs/phase5_identity/` to `c:\xampp\htdocs\YEAR 4\Testing\outputs\phase5_identity\` when transferred.

```bash
cd ~/forensic-dgp
if [ -d venv ]; then source venv/bin/activate; fi
DATA_DIR="dataset/thumbnails128x128,dataset/asian_faces" OUTPUT_DIR=outputs/phase5_identity bash scripts/run_phase5_gcp.sh
```

Track 2's actionable pilot is **real-mask integration into the detector**, with synthetic replay and the completion generator frozen. It uses the already-uploaded reviewed 100-image package and existing CodeFormer benchmark. The generic completion launcher does not accept real-mask supervision; do not append the raw masked-photo directory to its clean-target data. The original full custom generator pilot is already complete and should not be repeated unchanged.

```bash
cd ~/forensic-dgp
if [ -d venv ]; then source venv/bin/activate; fi
bash scripts/run_detector_replay_gcp.sh
```

Track 2 exports `~/forensic-dgp/outputs/detector_replay_vm/` ↔ `c:\xampp\htdocs\YEAR 4\Testing\outputs\detector_replay_vm\` after transfer. The recipe and prerequisite paths are in `~/forensic-dgp/DETECTOR_REPLAY_TRAINING.md` ↔ `c:\xampp\htdocs\YEAR 4\Testing\DETECTOR_REPLAY_TRAINING.md`. Detach with Ctrl+B then D; reattach with `tmux attach -t dgp_training`.

### Stage 2: validation and guardrails

| Candidate | Selection and review criteria |
|---|---|
| Track 1 `outputs/phase5_identity/best.pth` | Higher PSNR than selected best; overall SSIM and fixed-alignment ArcFace at least baseline; unchanged nonzero eligible identity-pair count; no per-source PSNR/SSIM/identity regression. Check `best_selection.json`: epoch 0 means retained baseline, not trained success. |
| Original Track 2 `outputs/completion_pilot/best.pth` | Lower predicted-mask hole MAE; no baseline visible-error regression, no aggregate segmentation IoU regression, no >85% predicted-coverage cases, and no reported source/condition visible-error regression. No best file was selected in the completed pilot. |
| Replay `outputs/detector_replay_vm/best_detector.pth` | Real IoU improves without worse real visible FP, empty detections or negative-case FP; synthetic IoU, missed fraction, visible FP, empty detections and negative-case FP retain baseline. Missing best file is a valid rejection, not a crash. This detector gate does not measure completion hole MAE. |
| Completion output acceptance | Re-run the identical fixed completion benchmark with the selected detector/configuration; require hole MAE reduction without visible-region error regression before claiming completion improvement. Inspect ten fixed rows: input, known/edited mask, known-mask completion, predicted-mask completion, target where available. Real masks without uncovered references cannot provide hidden-face MAE. |

Every output suffix above uses both root mappings defined at the top of this document. Inspect the full ten-row completion preview grid, including lower-face, eyes, irregular/object and uncovered controls under clear/degraded conditions; smoke previews alone are insufficient. Check seams, remaining mask material, generated anatomy and changes to visible features. Detector replay does not automatically produce a ten-row completion grid: generate/review it using the existing completion benchmark flow after selection. Keep the prior baseline if either metrics or visual review fails. Do not relabel the previously inspected seven images as a fresh final test; collect a separate reviewed holdout for final claims.

### Stage 3: workspace sync protocol

On the VM, package complete runs including metrics, configuration and selection records, rather than only a file called best. Include every epoch when no candidate qualifies. The commands below assume both Stage 1 runs finished; omit a missing run explicitly rather than archiving unrelated smoke output.

```bash
cd ~/forensic-dgp
mkdir -p outputs/handoff_environment
git rev-parse HEAD > outputs/handoff_environment/git-revision.txt
python3 -m pip freeze > outputs/handoff_environment/pip-freeze.txt
nvidia-smi > outputs/handoff_environment/nvidia-smi.txt
tar -czf ~/dgp-handoff-results.tar.gz outputs/phase5_identity outputs/detector_replay_vm outputs/handoff_environment
sha256sum ~/dgp-handoff-results.tar.gz > ~/dgp-handoff-results.tar.gz.sha256
printf '%s\n' "$HOME/dgp-handoff-results.tar.gz" "$HOME/dgp-handoff-results.tar.gz.sha256"
```

Download both printed paths through Google Cloud SSH's Download File action to `c:\xampp\htdocs\YEAR 4\Testing\outputs\`. Do not put weights/datasets into Git. Review archive member paths before extraction; use a fresh staging directory to preserve local outputs. In receiving Windows PowerShell:

```powershell
Set-Location 'c:\xampp\htdocs\YEAR 4\Testing'
Get-FileHash 'outputs/dgp-handoff-results.tar.gz' -Algorithm SHA256
Get-Content 'outputs/dgp-handoff-results.tar.gz.sha256'
tar -tzf outputs/dgp-handoff-results.tar.gz
New-Item -ItemType Directory -Path outputs/downloaded_handoff_20260927
tar -xzf outputs/dgp-handoff-results.tar.gz -C outputs/downloaded_handoff_20260927
```

Compare the checksum strings before extracting; stop on a mismatch or unexpected absolute/parent-traversal archive paths. Extracted VM `outputs/phase5_identity/` maps to `c:\xampp\htdocs\YEAR 4\Testing\outputs\downloaded_handoff_20260927\outputs\phase5_identity\`; replay has the analogous suffix. Record actual evidence in this document without overwriting the September 27 historical snapshot.

On the workspace that owns the reviewed documentation changes (Windows commands below; VM equivalent begins `cd ~/forensic-dgp`), explicitly stage docs, review the diff and publish:

```powershell
Set-Location 'c:\xampp\htdocs\YEAR 4\Testing'
git status --short
git add PROJECT_HANDOFF.md DETECTOR_REPLAY_TRAINING.md
git diff --cached --check
git diff --cached --stat
git commit -m "docs: update September 27 training handoff"
git push origin HEAD
```

Confirm the pushed branch is `main` before the receiving workspace runs `git pull --ff-only origin main`; otherwise merge the reviewed branch through the normal project workflow first. This documentation edit itself does not execute commit/push. If Antigravity opens the same local directory, files are already shared; do not reclone. In its terminal:

```powershell
Set-Location 'c:\xampp\htdocs\YEAR 4\Testing'
git status --short
git pull --ff-only origin main
git rev-parse HEAD
venv/Scripts/python.exe -m unittest tests.test_detector_replay tests.test_detector_training
```

Preserve uncommitted work and resolve conflicts explicitly; never reset/clean to force a pull. **Next action at handoff:** publish any unpushed code/docs, run VM Stage 0, then the bounded pilots. Quality review precedes any further generator training or deployment.

---

## Historical setup and restoration evidence (23–25 September)

**Scope update, 25 September 2026:** the user confirmed single-image restoration plus completion of facial regions hidden by masks or other objects. Hidden features are plausible estimates; visible degraded regions may also be restored. Read [FACE_COMPLETION_RESEARCH.md](FACE_COMPLETION_RESEARCH.md) for the researched plan and benchmark sequence, and the current runbook linked above for the implemented baseline. The Phase 5 launcher below remains restoration-only. Historical results and VM instructions below remain relevant.

## 1. Start here

The goal is better face-restoration output for a Philippine school setting, especially fidelity to the actual person. Higher sharpness or PSNR alone is insufficient. The user wants to train again after reviewing and implementing improvements.

**Current decision:** retain `checkpoints/dgp_zamboanga_final.pth` as the application baseline. Phase 4 completed, but its identity metric declined. Phase 5 is implemented and passed local tests; full GPU training and output-quality verification remain outstanding.

| Item | Verified state at handoff |
|---|---|
| Repository | https://github.com/janusinss/forensic-dgp |
| Local project | `C:\xampp\htdocs\YEAR 4\Testing` |
| Existing VM project directory | `~/forensic-dgp` |
| Local code revision before this document | `67a0ba1` (`changes`); remote push status not independently verified |
| Existing training hardware | Screenshot reported NVIDIA T4 and about 16 GB host RAM; exact machine type, project ID and zone are unverified |
| Local training runtime | Windows virtual environment, CPU-only PyTorch; not a substitute for GPU verification |
| Existing tmux session | `dgp_training` |

Read this document, [Phase 5 research](PHASE5_RESEARCH.md), and the actual training code before continuing. [The original understanding document](PROJECT_UNDERSTANDING_AND_IMPROVEMENTS.txt) provides historical context but includes outdated claims. This handoff records verified limitations explicitly.

## 2. What the project actually does

The restoration model is a feedforward residual generator with a MobileNetV2 feature pyramid, compatible with the project's DeblurGAN-v2-derived weights. It accepts RGB tensors in `[0,1]`, normalizes internally, and produces 256×256 restoration outputs. It is not a newly implemented diffusion model.

| File | Responsibility |
|---|---|
| `models/dgp_synthesizer.py` | Restoration architecture |
| `dataset.py`, `dataloader.py` | Images, synthetic degradation, landmarks and data splitting |
| `train.py`, `evaluation.py`, `training_state.py` | Phase 4 training, evaluation and checkpoint state |
| `train_phase5.py`, `phase5_utils.py`, `models/identity_loss.py` | Phase 5 identity supervision, EMA and selection |
| `app.py` | FastAPI application, preprocessing, restoration and display postprocessing |

Training creates degraded inputs from reference faces. The application additionally uses preprocessing, alignment/cropping, contrast adjustments and sharpening. Evaluate raw model output separately from these display changes.

Corrections to earlier descriptions:

1. FAN supervision is heatmap MSE, not an explicit landmark Euclidean constraint or a guarantee against hallucination.
2. A recurrent second restoration pass is not implemented merely because an old document describes one.
3. Application candidate scores are hardcoded display values, not measured loss, confidence or identity probabilities.
4. A restored face is an estimate. Current tests do not establish forensic admissibility or recovery of details absent from the input.
5. The application prioritizes `checkpoints/dgp_zamboanga_final.pth`. Saving `outputs/.../best.pth` does not automatically deploy it; its epoch fallback search also does not include epoch 31.

## 3. Completed work and measured results

### Historical training

The earlier project history describes Phase 1 as epochs 1–10, Phase 2 as 11–20, and Phase 3 as 21–26. Phase 3's named output is `dgp_zamboanga_final.pth`. Older reported Phase 2 metrics were training-batch measurements and must not be compared directly with the later held-out validation.

Phase 4 continued from Phase 3 through epochs 27–31. Changes included deterministic degradation, 35% heavy primary blur, a fixed 5% validation split, validation progress reporting, gradient clipping, complete training-state saves, and PSNR-based best-checkpoint selection. The existing component, VGG, color, FAN, Sobel and FFT losses remained active.

The actual cloud output directory was `outputs/phase4_with_progress`, although the Phase 4 launcher defaults to `outputs/phase4`.

### Full Phase 4 cloud validation

| Checkpoint | PSNR | SSIM | ArcFace similarity | Valid identity pairs |
|---|---:|---:|---:|---:|
| Phase 3 baseline | 20.2795 | 0.6603 | 0.3568 | 3886/4000 |
| Epoch 27 | 20.6097 | 0.6700 | 0.3487 | 3887/4000 |
| Epoch 28 | 20.0884 | 0.6541 | 0.3410 | 3904/4000 |
| Epoch 29 | 20.4924 | 0.6659 | 0.3461 | 3893/4000 |
| Epoch 30 | 20.1700 | 0.6532 | 0.3505 | 3881/4000 |
| Epoch 31 | 20.4048 | 0.6645 | 0.3357 | 3911/4000 |

`best.pth` contains exactly the same tensors as epoch 27. File hashes differ because serialization can differ. The full `last_state.pth` records epoch 31 and is not a dry run.

**Interpretation:** epoch 27 improved pixel/structural metrics, but every Phase 4 epoch had lower average ArcFace similarity than the starting model. “Best” meant highest validation PSNR, not proven best identity fidelity.

The run used 76,000 training images and 4,000 validation images. Training contained 66,500 FFHQ-source and 9,500 Asian-source images; validation contained 3,500 and 500 respectively. Recorded paths do not overlap. Earlier phases may already have seen these validation images, and identity-level separation was not established.

### Local checkpoint review completed 22 September

The downloaded archive is `outputs/phase4-results.tar.gz`. Its extracted run is at `outputs/downloaded_phase4/outputs/phase4_with_progress/`.

A paired check used the same degraded inputs for all models on 64 FFHQ validation images, selected with seed 20260922, degradation seed 42 and heavy-blur probability 0.35. All 64 were common valid identity pairs.

| Model | PSNR | SSIM | ArcFace, shared pairs |
|---|---:|---:|---:|
| Phase 3 | 19.4734 | 0.6376 | 0.3081 |
| Phase 4 best / epoch 27 | 19.7414 | 0.6454 | 0.3048 |
| Phase 4 epoch 31 | 19.5833 | 0.6413 | 0.2933 |

There were also 60 successful model/mode inference cases using 10 synthetic and 5 wild images, including sub-32 mode on the wild images. Visual changes were modest; severe blur remained soft. These real images have no verified pristine target here.

Review artifacts, which Git excludes:

- `outputs/phase4_review/REPORT.md` and `results.json`.
- `wild_comparison.png` and `wild_sub32_comparison.png` in that directory.
- `synthetic_comparison.png` and `paired_comparison.png` in that directory.
- Individual raw outputs and scene composites in that directory.
- `outputs/verify_phase4.py` and `outputs/write_phase4_report.py`, the local comparison helpers.

### Checkpoint meanings

| File | Meaning and intended use |
|---|---|
| `checkpoints/dgp_zamboanga_final.pth` | Phase 3 inference weights; retained deployed baseline and Phase 5 starting point |
| Phase 4 `best.pth` | Epoch 27 weights selected by PSNR; comparison candidate |
| Phase 4 `dgp_improved_epoch_31.pth` | Last Phase 4 weights; not the selected best |
| Phase 4 `last_state.pth` | Complete Phase 4 continuation state; not an inference-only weight file |
| Phase 5 `last_state.pth` | Phase 5 raw model, EMA, optimizer, scheduler, configuration, selection and random states |

Known SHA-256 values:

```text
Phase 3:      b6376f56c4161ef1f26a8f9efb0bb309b3014f4ecdeaca5399a47c1c4607586c
Phase 4 best: 90fd34c1f4b531cd4bb5ee02f6d6995af97f2d43122208e79412de30760a7f09
Phase 4 ep31: 64b7bf98f90cbcc8ef48697d44758dd73f0f4e8f8b929c023657528f8ca2b97a
ArcFace ONNX: 4c06341c33c2ca1f86781dab0e829f88ad5b64be9fba56e56bc9ebdefc619e43
```

## 4. Phase 5: implemented, awaiting full training

Phase 5 starts a separate **two-epoch pilot numbered 1–2**, from Phase 3. It does not continue the Phase 4 optimizer or imply epochs 32–33.

| Implemented change | Purpose |
|---|---|
| Frozen differentiable ArcFace cosine loss | Penalize identity-feature changes while gradients reach restored pixels |
| Same reference-landmark alignment for output and target | Avoid changing the crop according to the generated face |
| GPU preparation and reusable landmark cache | Avoid repeating landmark detection each epoch |
| Exponential moving average (EMA) | Validate and export smoothed model weights |
| Overall and per-source selection gates | Prevent a PSNR-only win from replacing a baseline with worse measured identity |

The objective adds `0.1 * mean(1 - cosine_similarity)` to the existing restoration loss. ArcFace weights stay frozen. Invalid reference landmarks contribute no identity loss. The cache key includes resized RGB content, shape and detector version.

Defaults: both datasets, original Phase 4 split, batch size 8, two workers, seed 42, head learning rate `1e-5`, backbone learning rate `2e-6`, EMA decay `0.999`, heavy-blur probability `0.35`. These are experiment settings, not a demonstrated optimum. No dataset reweighting or mixed-precision training was introduced.

Phase 5 reports `ArcFace_fixed`, which is **not directly comparable** to Phase 4's detection-based ArcFace values. It uses fixed reference alignment and reports each dataset source separately. Source is not an ethnicity label.

`best.pth` initially contains the starting baseline. Replacement requires higher PSNR than the selected best, overall SSIM and identity at least as good as baseline, the same nonzero eligible-pair count, and no baseline regression in each source's PSNR/SSIM/identity. `best_selection.json` records the decision; epoch 0 means no trained candidate has qualified.

Local verification already completed: 11 unit tests, Python and launcher syntax checks, conversion parity against ONNX Runtime (maximum embedding difference `1.31e-6`), nonzero input gradients with frozen recognition weights, and real CPU smoke training/save/resume. Smoke tests used one batch per nominal epoch and **are not full training or quality evidence**. `outputs/phase5_smoke` must never be presented as a production training result.

Research rationale and primary references are in [PHASE5_RESEARCH.md](PHASE5_RESEARCH.md). Full GPU compatibility, runtime and restoration benefit are still unverified.

## 5. Continue on the existing VM

Commands in this section run in the VM's Linux SSH terminal. First commit/push any intended local changes; `git pull` cannot retrieve files that have not been pushed. Preserve any VM-local edits if Git reports a conflict.

### Update and inspect

```bash
cd ~/forensic-dgp
git status --short
git pull --ff-only origin main
if [ -d venv ]; then source venv/bin/activate; fi
python3 -m pip install -r requirements-phase5.txt
python3 -m pip check
nvidia-smi
python3 -c "import torch; print(torch.__version__, torch.cuda.is_available()); assert torch.cuda.is_available()"
ls checkpoints/dgp_zamboanga_final.pth outputs/phase4_with_progress/split.json
```

Do not reinstall working GPU drivers or replace the existing PyTorch environment merely to follow the fresh-VM section. If `venv` is absent, the historical environment may be in user site-packages; confirm the printed interpreter and CUDA check before proceeding.

### Start the pilot inside tmux

```bash
tmux new-session -A -s dgp_training
```

Inside that session, confirm no other training command is running before starting another:

```bash
cd ~/forensic-dgp
if [ -d venv ]; then source venv/bin/activate; fi
DATA_DIR="dataset/thumbnails128x128,dataset/asian_faces" bash scripts/run_phase5_gcp.sh
```

The launcher checks ArcFace conversion/gradients, prepares landmarks, runs baseline validation, and performs two training/validation epochs. It requires the Phase 4 split and both datasets. Use a fresh output directory for a new experiment:

```bash
OUTPUT_DIR=outputs/phase5_identity_trial2 bash scripts/run_phase5_gcp.sh
```

Detach with **Ctrl+B**, release, then **D**. Reattach with `tmux attach -t dgp_training`. tmux survives an SSH disconnect; it does not keep training alive through a stopped or rebooted VM.

### Resume an interrupted pilot

```bash
cd ~/forensic-dgp
if [ -d venv ]; then source venv/bin/activate; fi
bash scripts/run_phase5_gcp.sh --resume_state outputs/phase5_identity/last_state.pth
```

Resume starts after the last completed epoch; partial-epoch work is repeated. Keep original parameters, output directory and data membership. If a custom output directory was used, set `OUTPUT_DIR` to that directory and point `--resume_state` to its `last_state.pth`. If no complete checkpoint exists, use a new output directory to restart. A completed two-epoch run reports completion; increasing epochs is a new experiment rather than an identical resume.

## 6. Build a fresh Google Cloud VM

### Provisioning

These are proposed setup choices, not a record of the old VM's exact configuration. No new VM was provisioned while writing this handoff.

1. Select your Google Cloud project and enable Compute Engine with billing and GPU quota in the chosen zone.
2. Create an N1 VM with one NVIDIA T4; `n1-standard-8` is a reasonable starting host configuration for preprocessing, subject to budget and availability.
3. Select Ubuntu 22.04 LTS and a 100 GB persistent balanced disk as an initial allocation; allow more space for duplicate downloads, caches or higher-resolution data.
4. Use standard provisioning for the first pilot and the GPU-required terminate-on-maintenance policy. Keep training access through SSH; the training job does not require a public web port.
5. Record project ID, zone, instance name, disk size and image version before connecting with SSH.

Follow [Google's N1/T4 creation guide](https://docs.cloud.google.com/compute/docs/gpus/create-gpu-vm-general-purpose) for supported combinations and regional availability. The existing screenshot hostname was `forensic-dgp-thesis`; do not infer the project ID or zone from that hostname. If using Secure Boot, follow the signed-driver procedure in the driver documentation.

### OS dependencies and GPU driver

```bash
sudo apt-get update
sudo apt-get install -y git tmux python3-venv python3-dev build-essential curl unzip libgl1 libglib2.0-0
nvidia-smi
```

If the image already supplies a working NVIDIA driver, skip driver installation. On a fresh plain Ubuntu VM without one, Google's installer procedure is:

```bash
cd ~
curl -fL https://storage.googleapis.com/compute-gpu-installation-us/installer/latest/cuda_installer.pyz --output cuda_installer.pyz
sudo python3 cuda_installer.pyz install_driver --installation-mode=repo --installation-branch=prod
```

If an Ops Agent is collecting GPU metrics, stop it before installation as described by Google. The installer can reboot the VM; reconnect and rerun the same installation command if instructed, then verify `nvidia-smi`. Restore a previously stopped Ops Agent afterward. Secure Boot requires the additional signing steps, not just the command above. [Official driver instructions](https://docs.cloud.google.com/compute/docs/gpus/install-drivers-gpu).

### Repository and Python environment

Run this clone only when `~/forensic-dgp` does not already exist:

```bash
cd ~
git clone https://github.com/janusinss/forensic-dgp.git
cd ~/forensic-dgp
python3 -m venv venv
source venv/bin/activate
python3 -m pip install --upgrade pip setuptools wheel
python3 -m pip install numpy cython
python3 -m pip install torch torchvision --index-url https://download.pytorch.org/whl/cu126
python3 -m pip install -r requirements.txt -r requirements-phase5.txt
python3 -m pip install kagglehub pyarrow pillow
python3 -m pip check
python3 -c "import torch; print(torch.__version__, torch.version.cuda); assert torch.cuda.is_available(); print(torch.cuda.get_device_name(0))"
```

The CUDA 12.6 wheel channel is documented by [PyTorch](https://pytorch.org/get-started/previous-versions/). Packages in the main requirements file are unpinned, so this is a fresh installation recipe, not a byte-for-byte recreation of the historical environment. Check Python/driver compatibility using the [official selector](https://pytorch.org/get-started/locally/) if installation fails. The `CUDA Version` printed by `nvidia-smi` is not the installed PyTorch wheel version. Run the project's conversion and gradient checks before committing GPU time to training.

Record the working environment after installation:

```bash
mkdir -p outputs/environment
python3 -m pip freeze > outputs/environment/pip-freeze.txt
git rev-parse HEAD > outputs/environment/git-revision.txt
nvidia-smi > outputs/environment/nvidia-smi.txt
```

### Model assets

Confirm the Phase 3 weights were obtained with the repository:

```bash
ls -lh checkpoints/dgp_zamboanga_final.pth
sha256sum checkpoints/dgp_zamboanga_final.pth
```

The required ArcFace file is normally `~/.insightface/models/buffalo_l/w600k_r50.onnx`. Restore that cache from the old machine or initialize the existing InsightFace package to download its model pack:

```bash
python3 - <<'PY'
from insightface.app import FaceAnalysis
app = FaceAnalysis(name='buffalo_l', providers=['CPUExecutionProvider'])
app.prepare(ctx_id=-1, det_size=(256, 256))
PY
```

This CPU initialization downloads assets; Phase 5 converts the recognition network to PyTorch and uses CUDA for training. FAN and VGG pretrained assets also download on first use. Keep internet access for initial preparation or restore the corresponding `~/.cache/torch` assets. The landmark preparation and baseline stages can therefore include first-use downloads.

## 7. Datasets: install, verify and preserve the split

Both datasets were used in Phase 4 and are retained for the controlled Phase 5 comparison. Two datasets are not automatically better than one: quality, representation, split integrity and measured outcomes matter. The current mixture is about 87.5% FFHQ-source and 12.5% Asian-source, with no balancing sampler.

| Local relative directory | Downloader source | Expected previous-run count |
|---|---|---:|
| `dataset/thumbnails128x128` | Kaggle `greatgamedota/ffhq-face-data-set` | 70,000 |
| `dataset/asian_faces` | Hugging Face `hiennguyen9874/face-age-gender-asian`, first 10,000 extracted records | 10,000 |

Prefer transferring the exact existing dataset directories for reproducibility. To download into a fresh VM:

```bash
cd ~/forensic-dgp
source venv/bin/activate
python3 scripts/download_ffhq.py dataset/thumbnails128x128
python3 scripts/download_asian_faces.py dataset/asian_faces
```

The FFHQ downloader treats 5,000 existing images as sufficient to skip downloading, although the completed run used 70,000. **Do not treat its success message alone as proof of a complete dataset.** Restore the complete old directory if counts or file paths differ. Access/authentication requirements of the upstream hosts can change; never put tokens into Git.

Check counts and decode every image:

```bash
python3 - <<'PY'
from pathlib import Path
from PIL import Image
for name, expected in [('thumbnails128x128', 70000), ('asian_faces', 10000)]:
    paths = sorted(p for p in (Path('dataset') / name).rglob('*')
                   if p.is_file() and p.suffix.lower() in {'.jpg', '.jpeg', '.png'})
    print(name, len(paths), 'expected', expected, flush=True)
    assert len(paths) == expected, 'Dataset count differs from the completed run'
    for path in paths:
        with Image.open(path) as image:
            image.verify()
    print(name, 'all files decoded', flush=True)
PY
```

Copy `outputs/phase4_with_progress/split.json` from the old VM. It is excluded from Git. When restoring the downloaded archive to a fresh VM, upload your own `phase4-results.tar.gz` to the home directory and extract into a separate staging directory:

```bash
cd ~/forensic-dgp
mkdir -p outputs/restored_phase4
tar -xzf ~/phase4-results.tar.gz -C outputs/restored_phase4
ls outputs/restored_phase4/outputs/phase4_with_progress/split.json
```

Use this restored location explicitly:

```bash
SPLIT_FILE=outputs/restored_phase4/outputs/phase4_with_progress/split.json bash scripts/run_phase5_gcp.sh
```

Run from the repository root and retain original relative image paths. The trainer checks that the manifest includes all images exactly once with no overlap. Do not silently create a different split to bypass a mismatch. If the old manifest contains machine-specific absolute paths, map only the root while preserving membership and document that migration before training.

The current FFHQ targets are 128px thumbnails resized to 256px; resizing creates no additional reference detail. The Asian-source dataset is not a verified Filipino-school benchmark. [FFHQ's official documentation](https://github.com/NVlabs/ffhq-dataset) also specifies usage restrictions, including exclusion of facial-recognition development. Freezing ArcFace for restoration does not itself establish that the intended school identification use is permitted. Dataset/model permissions and a representative evaluation set remain unresolved deployment requirements.

## 8. Expected progress, files and troubleshooting

Validation measures outputs without updating the restoration model. Training updates its weights. A new run performs baseline validation, epoch training, epoch validation, checkpoint saving, and repeats. The phase number in a one-batch smoke test did not prove that real epochs had already trained.

Historical Phase 4 baseline validation took about 43 minutes for 4,000 images. Epoch 27 took about 11 hours 31 minutes plus about 43 minutes of validation. Phase 5 runtime has not been measured; cache preparation is extra first-run work.

Inspect a running VM from a second SSH session:

```bash
nvidia-smi
ps -eo pid,etime,time,pcpu,stat,args | grep '[p]ython'
```

High CPU usage with low GPU utilization can indicate CPU detection or preprocessing; a single GPU snapshot cannot establish that a job is stuck. In Phase 4, InsightFace used CPU execution and lacked visible validation progress initially. ArcFace was not removed to fix that visibility problem.

Phase 5 output directory contents:

- `best.pth` plus `best_selection.json`: selected inference weights and selection reason.
- `epoch_1.pth`, `epoch_2.pth`: EMA inference weights, including candidates that failed selection.
- `last_state.pth`: full state for resuming, not an inference-only checkpoint.
- `metrics.jsonl`, `baseline.json`, `epoch_*.json`: aggregate and per-image measurements.
- `config.json`, `split.json`, `baseline.png`, `epoch_*.png`: experiment settings, membership and comparisons.

| Location | Cause | Fix |
|---|---|---|
| CUDA preflight | CPU wheel, wrong environment or missing driver | Check interpreter, `nvidia-smi`, and PyTorch CUDA availability before launching |
| Split/data check | Missing files, partial download or different layout | Restore exact data and original manifest; verify counts and paths |
| ArcFace preflight | Missing ONNX or conversion mismatch | Restore/download the required asset; run conversion checker and inspect its error |
| Existing output/resume check | New run reuses an output directory or changes configuration | Resume its own full state with original settings, or start a separately named experiment |
| FAN/identity validation | Undetected faces or no eligible reference pairs | Review coverage and images; isolated warnings can occur, but zero baseline identity pairs stops Phase 5 |

For an out-of-memory error, inspect GPU processes first. A smaller batch is a new run setting and cannot silently replace the batch size of an existing resume. Preserve failed-run logs when creating the replacement experiment.

## 9. Download results and move to another workspace

### Export a completed cloud run

Run on the VM after training has finished so metrics and checkpoint state are consistent:

```bash
cd ~/forensic-dgp
tar -czf /tmp/phase5-results.tar.gz outputs/phase5_identity
sha256sum /tmp/phase5-results.tar.gz
```

In SSH-in-browser choose **Download file** and enter `/tmp/phase5-results.tar.gz`. For Phase 4 the equivalent path was `/tmp/phase4-results.tar.gz`, containing `outputs/phase4_with_progress`.

On local Windows, save Phase 5's archive as `outputs/phase5-results.tar.gz`. From the project root in PowerShell:

```powershell
Get-FileHash outputs/phase5-results.tar.gz -Algorithm SHA256
New-Item -ItemType Directory -Force outputs/downloaded_phase5
tar -xzf outputs/phase5-results.tar.gz -C outputs/downloaded_phase5
```

Compare the local hash with the VM's hash. Extract only the archive you exported into a new staging directory; do not overwrite the deployed checkpoint. Its nested run will be `outputs/downloaded_phase5/outputs/phase5_identity/`.

### Transfer checklist

1. Push/clone tracked project code, including this handoff and `PHASE5_RESEARCH.md`; record the commit hash.
2. Transfer `outputs/phase4-results.tar.gz`, `outputs/phase4_review/`, comparison helpers, and any completed Phase 5 run archive separately.
3. Transfer the two dataset directories and original split for exact reproduction, or use the documented fresh-download checks.
4. Preserve required pretrained weights and optionally the landmark/Torch/InsightFace caches; retain checkpoint hashes and environment records.
5. Copy local `AGENTS.md` guidance if the next workspace needs it; recreate its virtual environment instead of copying Windows `venv` to Linux.

`outputs/`, `dataset/`, `weights/`, `venv/`, `docs/`, and agent configuration directories are ignored. Most checkpoint files are ignored too, with specific named exceptions. A successful `git pull` does not bring back all research evidence. Do not include credentials, `.env` secrets or account tokens in a shared archive.

## 10. Verification in the next workspace

The prior test results are recorded above; this document does not imply tests were rerun on a fresh VM. From the project root with the correct environment:

```bash
python3 -m unittest discover -s tests -p test_phase5.py
python3 scripts/check_phase5_identity.py --model ~/.insightface/models/buffalo_l/w600k_r50.onnx --device cuda
```

For a local Windows CPU check, use `venv/Scripts/python.exe` instead of `python3` and `--device cpu` for the conversion checker. Unit tests and conversion checks validate mechanics, not trained output quality.

A separately named local smoke run is optional when validating a rebuilt environment:

```powershell
venv/Scripts/python.exe train_phase5.py --data_dir dataset/thumbnails128x128 --batch_size 1 --num_workers 0 --dry_run --output_dir outputs/phase5_smoke_new_workspace
```

That command requires existing model assets and images. Never mix its artifacts with a complete training run. To reproduce the previous Phase 4 comparison, transfer its ignored helper and inputs first; `outputs/verify_phase4.py` uses fixed archive/report paths and rewrites its review outputs when rerun.

To inspect the application locally using the retained baseline:

```powershell
venv/Scripts/python.exe -m uvicorn app:app --host 127.0.0.1 --port 8000
```

Open `http://127.0.0.1:8000`. Do not interpret the application's hardcoded candidate scores as validation results.

## 11. Next work, in order

1. Complete the two-epoch Phase 5 GPU pilot with the original data split and preserve all outputs.
2. Compare selected and per-epoch EMA weights against Phase 3 and Phase 4 epoch 27 on identical inputs, using raw images and display outputs separately.
3. Run legacy detection-based evaluation and an independent identity/visual assessment. The recognizer used in the new training loss also supplies `ArcFace_fixed`, so improvement on that metric alone is not independent evidence.
4. Run a controlled identity-loss ablation with a new output directory and `--lambda_identity 0`; retain the same data and remaining settings. This isolates identity supervision from EMA and reduced learning rates.
5. Based on measured errors, plan higher-resolution reference data, representative permitted Filipino test data, and camera-matched degradation experiments. These are proposals, not implemented or proven improvements.

Do not automatically extend training because more epochs are available. Preserve the Phase 3 baseline until the output review supports a deliberate deployment decision. Record selected epoch, checkpoint hash, configuration, data split, coverage, runtime, visual comparisons and limitations for each new result.

### Instruction for the next assistant/workspace

Read `AGENTS.md` if present, this handoff, `PHASE5_RESEARCH.md`, and the current training scripts. Inspect Git status and transferred artifacts before claiming anything is complete. Phase 4 has measured results; Phase 5 has implementation and local mechanical tests only. Verify whether new cloud results have arrived since this handoff, retain the user's focus on actual restored output, and update this document with evidence rather than assuming an improvement.
