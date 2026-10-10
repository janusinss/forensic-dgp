# Method clarification — 10 October 2026

The user's latest answer, “apply the best approach here,” selects an explicit
generative-prior interpretation for the replacement comparison. The existing
app model is a fine-tuned DeblurGAN-v2-compatible conditional restorer; keep its
checkpoint as the permanent comparator. Its historical DGP class name does
not establish a separate generative face prior.

Develop our own trained input conditioning and256 reconstruction using useful
retained weights and a separately declared frozen face-generating bank.
Describe this as an adapted hybrid generative-prior restorer, not an exact
reproduction of published DGP or a generator trained entirely by us.
Compare against corrected current-DGP training with active clear/blur target
reconstruction. Full random-weight generator training is not selected without
a separate feasibility case. Pretrained restoration models remain comparators.

The standalone StyleGAN2 component has passed independent local mechanics
checks; conditioned restoration, GPU feasibility, useful CCTV quality and
completion remain unverified. This method decision changes neither the
preservation requirements nor the manual L4 training boundary. The full
comparative goal below stays active.
[CCTV_DGP_GENERATIVE_PRIOR_METHOD_REVIEW_V2.md](<C:/xampp/htdocs/YEAR 4/Testing/CCTV_DGP_GENERATIVE_PRIOR_METHOD_REVIEW_V2.md>)
records the evidence, terms, limits and next executable prerequisite.

---

# Current goal — comparative DGP improvement, 10 October 2026

This is the canonical project goal following the user's request to update it
and continue improving the model. It supersedes conflicting preparation
directions below, including a commitment to only the current decoder or an
automatic next eight-arm diagnostic. Historical protocols, gates, commands and
results retain their original meaning and are not reopened by this update.
The execution plan is
[CCTV_DGP_COMPARATIVE_IMPROVEMENT_PLAN_V2.md](<C:/xampp/htdocs/YEAR 4/Testing/CCTV_DGP_COMPARATIVE_IMPROVEMENT_PLAN_V2.md>).

Develop and verify our own trained DGP as the primary model for useful 256×256
restoration of degraded CCTV face crops for the thesis “Forensic Deep Generative
Prior Face Reconstruction for Degraded CCTV Video in Zamboanga City,” in
`C:\xampp\htdocs\YEAR 4\Testing`. Follow this goal and PRACTICAL_OUTPUT_SCOPE.md;
later user decisions take precedence over historical directions and the outdated
manuscript. Preserve all visible facial structure and appearance, accepting some
softness. Request a clearer or less-covered crop when usable information is
insufficient. Support one already cropped frontal or mildly turned face in the
existing local application, preserving its design.

The first improvement comparator is the retained app DGP,
`outputs/cctv_dgp_face_code_fit_vm_v12_r2/weights/dgp_v2.pth`, SHA256
`646fbb11674e8882ea41e251d6e6e7d468c963979b48c93132def8292834e24b`.
Keep it as the permanent starting baseline even after a better development
candidate is accepted. Its lineage is original Phase3 training plus two selected
identity-v2 fine-tuning epochs/226 branch updates; the complete lifetime epoch
count is unconfirmed. Rejected pilots do not add epochs to these app weights.

Improve the model through a finite comparative training programme. First review
the demonstrated loss, data-coverage, numerical and reconstruction-path
limitations. Compare a materially corrected training recipe for an isolated copy
of the current DGP against one evidence-justified replacement DGP reconstruction
design. Retain useful learned weights where compatible; document inherited,
frozen, newly initialized and trained components. Do not default to resetting
every weight or substituting a pretrained restorer. Training an entire model from
random weights requires a separate data, feasibility and finite-budget case.
Published restoration architectures are design references and declared
comparison baselines; a generic replacement must not be called DGP without a
documented generative-prior role and thesis-method justification.

Freeze the two recipes, identical input preparation, evaluation roles, training
exposures, measured compute limits and acceptance checks before manual VM runs.
Declare differences that prevent a strict causal architecture comparison. Use
small diagnostics only to answer a named prerequisite, with a decision point;
avoid an indefinite chain of head, loss or learning-rate changes. A candidate
must show useful visible-detail improvement over the current incumbent on
development review while passing the retained structure and appearance checks.
Training loss, a 1% proxy score, PSNR, extra epochs or export completion alone is
insufficient. Compare raw floats and delivered PNGs separately and report mixed,
negative and insufficient-information outcomes honestly.

After independently audited development acceptance, retain the accepted version
as the next incumbent and test another justified finite continuation against
both that incumbent and the permanent original baseline. Preserve every version
and rollback choice. Additional epochs are permitted when useful learning and
preservation justify them. The proposed first longer study remains at most five
additional epochs, with checkpoints at 1, 2 and 5 unless a new protocol justifies
other checkpoints prospectively; these numbers are an assistant proposal, not
an exact user-selected schedule. Later extensions need their own finite limits,
validation points and stop/export rules. Plateau, regression, resource limits
or three consecutive unsuccessful attempts require a diagnosis/design decision,
not an unchanged failed-state resume. Every iteration aims to improve quality;
success and a universally best model are not guaranteed. This is a bounded
programme within the available resources and thesis schedule, not an endless
completion requirement.

Use audited public native CCTV development crops for input-only and visual
review. No real Zamboanga CCTV samples exist yet. Prioritize Asian capture
sources where available, report each source separately and infer neither
ethnicity nor Zamboanga performance. Reuse audited genuine high-resolution clean
training references and approved replay; acquire new data only for demonstrated
coverage/detail gaps with audited terms, provenance, native resolution and
overlap/exposure limits. Treat CCTV without aligned clean references as unpaired
evidence. Keep paired synthetic PSNR/SSIM separate. Track repeated development
exposure and keep reserved final identities outside training, recipe choice and
tuning. Freeze the selected recipe/checkpoint before independent final review;
do not recycle final results into a development selection loop.

Complete the five milestones:

1. Freeze public native CCTV development and separate labeled final evaluation
   identities, provenance, terms, resolution, overlap/exposure limits and
   input-only review criteria.
2. Compare resizing, retained original Phase3 DGP, the retained app DGP, each
   declared candidate/incumbent and pretrained restoration baselines on identical
   prepared 256×256 inputs. Separate raw/display outputs, visible-structure
   preservation and insufficient-information cases. Improvement means beating
   our retained DGP with preservation; do not claim superiority to every external
   baseline or on every crop without evidence.
3. Fix demonstrated limitations and execute justified finite manual VM
   comparisons/continuations, preserving original checkpoints, splits, caches,
   protocols, full stopped states and failed gates. Choose reviewed useful
   checkpoints rather than the latest epoch.
4. Integrate the development-qualified DGP as primary with automatic restoration
   selection and a user override, preserving the existing design. A stronger
   model on training cases alone is not an app-promotion decision.
5. Independently review useful development and reserved final outputs, meaningful
   regressions and the full app flow through bundled inline Playwright. Update
   PROJECT_HANDOFF.md at comparison decisions, accepted versions and milestones.

Separately train/evaluate and qualify plausible completion of masks, sunglasses,
strong lens glare, hands, obstructing hair, scarves and other objects. Restoration
and completion may share components, but require distinct examples and acceptance
evidence. Show the automatic removal area for optional correction before
generation. Preserve clear glasses, non-obstructing hair and visible appearance,
with only a small documented margin. Request a less-covered image when too little
face remains. Return one plausible estimate alongside original and mask, with
PNG and optional original/mask/result bundle downloads. Do not claim exact
hidden identity. Report automatic and assisted results separately; restoration
improvement does not qualify completion or its seven covering families.

Inference and audits may run locally. All actual training, gradient diagnostics
and optimizer updates remain manually launched inside tmux on the existing
NVIDIA L4 g2-standard-4 at `~/forensic-dgp`, until the user identifies a verified
replacement VM. Provide verified transfer files and exact Windows Google Cloud
SDK upload/download and VM install/verify/tmux/launch commands. Use one remote
source per gcloud SCP download. Independently audit returned artifacts. Direct
connection to the existing VM is authorized only for inventory-first, hash-bound
storage maintenance preserving research assets, workloads, checkpoints, splits
and failures. No automatic training, historical-pilot launch, unchanged failed
recipe, stop bypass or gate weakening is authorized.

Every run needs finite updates/epochs, measured timing/projection, wall-time,
VRAM, storage reserve/output and export limits despite no overall GPU-hour cap.
The previously reported USD34 balance is not a current billing reading; the user
will notify near USD5 for migration export. Verify a local backup of model,
optimizer/scheduler/RNG/schedule state, next epoch/update, architecture code,
environment, data/split/provenance hashes, logs and failures before migration.
Distinguish exact resume from weights-only fine-tuning. Submission/defense is
around or after 20 November 2026; reserve time for app/final review and reporting.

Complete only when reviewed DGP-led app outputs meet useful restoration and all
seven covering-family requirements with the five milestones verified. Beating
the old DGP is a necessary improvement objective, not sufficient project
completion. The saved goal is updated; no new training recipe, executable packet,
successful model or promotion is implied. Earlier document bytes are backed up
under `outputs/cctv_dgp_goal_update_20261010_comparison_v2/before/`; preceding
status entries below are historical where superseded by this section.

---

**Latest verified status — 10 October 2026: multiscale return audited; all64 visual sheets reviewed; all12 treatments rejected. Full goal incomplete.**

The manual L4 diagnostic completed12 independent one-update arms,600 fitting
exposures and zero complete epochs. Full original-decoder LR0.001 gives
1.170–1.884% raw structure gain, but changes visible appearance and increases
clear pixel error24.8–31.5 times. Lower rates/deep-only learning do not supply
enough useful detail. Every treatment fails unchanged quality requirements;
no stopped state is resumed or promoted. Current checkpoint/app design remain.

Independent R2 integrity/arithmetic checks preserve the original checker failure
and every scientific gate decision. All64 planned sheets/2280 exact saved PNG
cells were inspected; native24 ChokePoint DEV cases remain separately unpaired
and retain their input-only usable labels. This implementing-assistant review
is not the goal's independent final review. Final identity pixels remain outside
tuning. Report: CCTV_DGP_MULTISCALE_CALIBRATION_V1_RESULTS.md.

400 fixed color/detail controls also fail necessary pixel checks. Six fixed
saved-gradient contrasts support examining active clear/blur RGB reconstruction
supervision before the first update, while retaining adverse reference slopes.
These are arithmetic diagnostics, not trained or qualified models. Selected
next design: CCTV_DGP_POST_MULTISCALE_TRAINING_REVIEW.md. No new executable VM
packet or additional-epoch run is prepared here; the epoch1/2/5 study stays
conditional on a passing finite recipe and useful native development review.

All five milestones and seven covering families remain required. Current model,
original checkpoints, caches, splits, provenance and failed gates are preserved.
Training remains manual through verified transfers/tmux on the existing L4;
direct VM access remains maintenance-only. Current guest storage/workload are
not verified by this local return audit. Historical ready/pending statuses and
closed commands below are superseded, not authorization to rerun them.

---

**Latest maintenance status — 10 October 2026: backup-verified VM cleanup complete; multiscale diagnostic still awaits the user's manual run. Full goal incomplete.**

Two obsolete Head4 capacity home transfer archives were removed after complete local SHA-256/GZIP backup checks and fresh workload checks. Observed recovery: 3.746 GiB. Fresh guest inventory: 11.181 GiB free; conservative projection after the new packet upload/install: 10.776 GiB, above the retained 7 GiB requirement.

Independent checks preserve 468,473 research metadata entries, 4,111 critical/evidence byte hashes, 346 local bindings and all251 current packet assets. Scientific caches were not deleted; their full bytes were not all rehashed. The deletion proof is exact regular nlink1 files outside the research tree, with unchanged research metadata and critical hashes. All unpacked failed-pilot outputs, states, checkpoints, splits and provenance remain.

The existing verified L4 instance is RUNNING, with an idle GPU and no tmux session at the final snapshot. Maintenance started no VM, model, gradient or training job and did not stop the VM. Upload, install and launch remain manual: CCTV_DGP_MULTISCALE_CALIBRATION_V1_VM.md. Recheck storage/workload before launching; this snapshot is not a future availability guarantee. The previous quality failures and all five restoration/completion milestones remain unchanged. Report: CCTV_DGP_MULTISCALE_STORAGE_CLEANUP_V1.md. Evidence: outputs/cctv_dgp_multiscale_archive_cleanup_v1/.

---

**Latest verified status — 10 October 2026: failed Head4 capacity retained; balanced multiscale/rate diagnostic prepared and independently checked. Manual VM run pending; full goal incomplete.**

The downloaded 50-update Head4 pilot remains rejected: raw structure gain
0.00608187% versus 1%, two blurred-source group MSE regressions and 53.2186%
brightness-only share versus the 20% maximum. Its checkpoint, full stopped state,
original assertions, all saved outputs and failure remain; no failed-state resume.

The saved-gradient review independently checks 1680 products with zero neural,
gradient or optimizer calls. A favorable averaged initial direction did not
validate the per-reference recipe: 8/20 individual directions predict worse
detail in at least one source/cohort group. This is initial TRAIN evidence,
not a unique causal explanation or a forecast across fifty updates/epochs.
See CCTV_DGP_POST_HEAD4_CAPACITY_DESIGN_REVIEW.md.

The new initializer preserves all100 current-DGP CPU outputs exactly; ten cases
are independently replayed. The frozen finite calibration compares deep3 with
the complete original decoder15, three rates and two balanced exposed TRAIN
pools. Twelve independent arms have one update/50 fitting exposures each,
280 preceding gradient queries, separate raw/PNG results and all24 native
development crops as unpaired review evidence. No final identity pixels enter
fitting, rate choice or tuning. All64 planned comparison pages require review.
Archive/member/source/data/launch guards pass. Actual L4 gradients, finite
outputs, useful native appearance and longer training are still unverified.

The existing VM was API-verified stopped; guest disk space is unknown. Manual
start/upload/tmux/download commands are in CCTV_DGP_MULTISCALE_CALIBRATION_V1_VM.md.
Require7GiB after installation, runtime projection, a1GiB reserve and2.5GiB
output cap. Model work stops at1800seconds and export at600seconds; external
supervision is also bounded. Local work used no new gradients or updates.

Original1%/10% and appearance requirements remain. Current app checkpoint/model
selection is unchanged. The proposed additional-epoch1/2/5 study, five milestones,
independent final review and allseven completion families remain incomplete.
Native and paired synthetic evidence remain separate; no ethnicity, hidden
identity or Zamboanga performance claim follows. Historical statuses below are
superseded; closed pilot commands must not be automatically rerun.

---

**Latest verified status — 10 October 2026: Head4 capacity V1 returned and independently audited; 50 updates failed the unchanged quality requirement. Full goal incomplete.**

The manual L4 training pilot completed all 50 planned optimizer updates (zero
complete additional epochs; 6.402% of one 781-batch epoch). All three reconnected
trainable pieces receive positive gradients and their weights change. Raw
structure gain is 0.00608187% and delivered PNG gain is 0.00578294%, below
the retained 1% requirement. Both output stages slightly increase pixel error in
two blurred TRAIN-source groups. Brightness shifts explain 53.2186%
raw/54.6191% PNG of the measured pixel-error improvement, exceeding
the unchanged 20% maximum. The stop is an intentional quality rejection after
evaluation; archive completion is not model qualification.

The unchanged independent checker verifies the 3,586,556,555-byte archive
(da008028d730db86b7d5dfe92a718474bfee1fdb45e63fb48c5782757800fd3d), all 7,810 case records across
baseline/candidate snapshots, frozen weights, full optimizer/scheduler/RNG state
and two fresh CPU replays. Original checkpoints, raw/PNG artifacts and the
failure remain. Local auditing uses zero gradients/updates. No failed-state
resume, extra epochs, new training, model/app promotion or gate change follows.
The current V1 command guide is closed: do not rerun or resume this failed pilot.
Its previous ready status, disk values and commands below are historical.
This is paired photographic TRAIN capacity evidence; native development and
reserved final identities were not evaluated. Two planned diagnostic sheets/10
TRAIN comparisons were inspected; comprehensive visual review remains pending.
See CCTV_DGP_HEAD4_CAPACITY_V1_RESULTS.md and
outputs/cctv_dgp_head4_capacity_v1_independent_audit.json. The earlier prepared
status below is historical and superseded by this returned failed pilot.

---

**Latest verified status — 10 October 2026: repaired-head4 gradient return audited; backed-up archive cleanup closed; first manual training stage verified. Full goal incomplete.**

The downloaded L4 gradient diagnostic is complete with zero optimizer updates or
epochs. Independent R2 checks327 members,160 saved vectors,480 component-part
norms,100 initial raw/PNG/embedding records,40 losses and two fresh CPU replays.
Allthree repaired connected pieces have finite nonzero improvement gradients in
both exposed TRAIN cohorts; original equivalents have zero gradients. This is
learning-path evidence, not restored-image usefulness. The original norm-summary
checker failure remains; R2's1e-14 allowance affects derived L2 arithmetic only,
with exact zero/nonzero decisions and unchanged image-quality requirements.

The new independent training initializer preserves all100 CPU outputs exactly,
the full original fusion tensor and620 other original state entries. Three
independently owned pieces/147,456 elements are planned trainable. All stored
normalization, other original parameters and the retained checkpoints remain.
The selected positive reconstruction loss weights are prospectively reviewed
against saved TRAIN gradients, including first-Adam weight decay. Those arithmetic
predictions do not prove finite-step, PNG or native quality improvement.

The new self-contained packet has5505 assets including all5467 unchanged TRAIN
assets,781 canonical targets and3905 cases. Independent packet audit verifies
5506 archive members,90 local source bindings, canonical/clear-target equality,
50 fixed reference batches,100 initial-parity cases, lossless raw storage, safe
incoming boundaries, Windows training refusal, killed-worker failure retention
and allfive gcloud transfer commands. No training has been launched by the agent.

Authorized maintenance removes15 exact regular nlink1 home transfer archive
copies, each matching a complete CRC/hash-verified local backup. It reclaims
6.210GiB observed. All461,169 research filesystem metadata
entries and3,973 protected byte hashes remain unchanged. Cache bytes are not all
rehashed; their retention is additionally proven by disjoint home-only deletions,
independent-link checks and unchanged full research metadata. No checkpoint,
split, failure, unpacked training folder, cache, runtime or current new packet is
deleted. Fresh post-cleanup inventory reports15.027GiB free and an idle GPU.
This snapshot is not a guarantee for a future launch. The guide's earlier stopped
API observation is superseded by the fresh running-VM inventory.

Next manual action: **CCTV_DGP_HEAD4_CAPACITY_V1_VM_CURRENT.md**. The original
prepared guide remains. Current commands pin the verified public SSH host key
for the VM's new address. Its445,352,976-byte
execution archive SHA256 is
`c7651ee72f14b3ac1b2e44f2671982b880f21b88172a87d4a03561dc8c32c8b4`;
protocol SHA256 is
`3d6faf634d45867dc2edd7589aa0b5450790e307342f1897f8ea4b859a646532`.
Require14GiB free after installation. Worker2400s/external2430s,
each snapshot900s, fitting300s, export600s/external630s, VRAM20GiB,
return6GiB and free reserve1GiB are enforced. Manual tmux is mandatory.

This first test has a50-update ceiling and250 fitting exposures, checking all3905
TRAIN outputs at0/50. It completes0 full epochs (50/781 of an epoch). Both raw
and delivered PNG must meet the retained1% structure and preservation requirements;
all outputs/embeddings are retained losslessly. Full model/optimizer/scheduler,
RNG/schedule position, code, environment and failed-gate state are exported.
Logical full-state portability is distinguished from unproven bitwise CUDA resume.
Do not repeat a stopped recipe or resume a failed checkpoint.

Additional epochs1,2,5 with maximum5 remain the proposed longer study, requiring
audited preservation and useful native development gains. This50-update stage
does not test or waive the later10% capacity requirement, qualify a model, select
final identities or promote an app checkpoint. All native CCTV evidence remains
unpaired; synthetic paired metrics remain separate. Ancestral training epochs
and full historical subject overlap remain unconfirmed.

Allfive milestones and allseven completion families remain binding: masks,
sunglasses, strong lens glare, hands, obstructing hair, scarves and objects.
Preserve visible appearance/clear glasses/ordinary hair and existing design;
request clearer or less-covered input when insufficient. Show automatic masks
for correction, provide original/mask/one plausible estimate and PNG/bundle
downloads, and report automatic versus assisted results separately. No ethnicity,
hidden-identity or real Zamboanga performance claim is supported. DGP-led Auto
plus override, independent native/final visual review, meaningful regressions
and bundled inline Playwright app-flow verification are still required.

Current evidence: CCTV_DGP_HEAD4_REACTIVATION_V1_RESULTS.md,
CCTV_DGP_HEAD4_TRAINING_V1_DESIGN.md,
outputs/cctv_dgp_head4_capacity_v1_preparation/independent_packet_audit.json and
outputs/cctv_dgp_head4_archive_cleanup_v1/independent_audit.json.
Historical status entries below retain their original bytes and prior decisions.

---

**Latest verified status — 10 October 2026: original deepest-path numerical limitation isolated; exact-preserving initialization and manual L4 gradient packet verified. Full goal incomplete.**

The current DGP's deepest head is zero on100 exposed TRAIN inputs. Its two kernels
and connected fusion slice are near6.305e-40 and are bitwise unchanged from Phase3.
Other heads remain active. This is one demonstrated branch limitation, not a unique
explanation of all prior failures or proof of image-quality improvement.

A separate fixed initializer copies our current trained adjacent head/fusion and
uses six frozen anchors. All100 initial CPU outputs match the current DGP exactly;
the branch responds on100/100. Independent fresh replays and619 untouched full
state entries pass. No local gradients, optimizer updates, new epochs or app
adoption occurred. Original checkpoints, normalization, thresholds and failures remain.

The full input/target audit now covers3,905 TRAIN cases,781 canonical targets and24
unpaired native development inputs. All5467 TRAIN asset hashes are preserved.
HQ canonical hashes match; legacy thumbnail-hash mismatches are documented, not
corruption. Full-corpus motion cases have limited geometric overlap with native
eye spacing. This does not establish equivalent resolved detail or usable routing.

Next manual action: use **CCTV_DGP_HEAD4_REACTIVATION_V1_VM.md**. The verified
183,353,481-byte packet is a matched original/repaired gradient diagnostic only:
160 queries, zero optimizer updates/epochs,600s worker/630s external stop;3GiB
free required after install. It has not run on the VM. A finite nonzero connected
gradient pass does not waive the1%-at50/10%-at800 or preservation requirements.
Additional epochs1/2/5 require a justified changed training recipe and later
independent capacity/development review. No historical failure is resumed.

All five milestones and seven completion families remain binding. Native CCTV
stays unpaired; no ethnicity or Zamboanga performance claim is made. Reserved-final
identities remain outside tuning. Preserve visible appearance and existing design,
show completion masks for correction, request clearer/less-covered inputs when
insufficient, and retain original/mask/plausible-result downloads and independent
final/inline-Playwright requirements. Direct VM connection remains maintenance
only; all autograd/training is manually launched by the user in tmux.

Current evidence: CCTV_DGP_HEAD4_REACTIVATION_V1_DESIGN.md,
CCTV_DGP_FULL_TRAINING_COVERAGE_V1_RESULTS.md and
outputs/cctv_dgp_head4_reactivation_v1_preparation/independent_packet_audit.json.
The preceding status entries below are retained historical evidence.

---

**Latest verified status - 10 October 2026: finite-guard R1 returned and fully reviewed; conversion and input-coverage diagnostics closed. Full goal incomplete.**

The manual L4 study accepts zero parameter changes and zero epochs. Its frozen
return checker passes1,629 members,400 raw/PNG records,64 saved aggregate vectors,
24 comparisons and80 CPU replays. All20 exact-cell pages/400 model outputs have
primary-assistant development visual review. The proposals remain soft; highest
PNG structure gain is0.043497%, below1%. Every raw preservation comparison passes,
but delivered ArcFace regressions reject allthree proposals. Original checkpoints,
model states, split roles, caches, provenance, terms and failed gates remain.

A separately frozen local conversion check verifies400 rounded PNGs and800 metric
rows. Nearest rounding approximately halves conversion error, but qualifies no
proposal, retains recognition failures and introduces negative source gains.
Historical floor decisions are reproduced. No policy or model is adopted. New
rounded images are numerically audited, not separately visually qualified.

The input-only coverage review measures100 exposed TRAIN and all24 usable native
development crops. Sampled degraded TRAIN geometric eye spacing is5.38-21.40 grid
pixels; native annotation spacing is23.02-47.04 capture pixels. These are geometry,
not equivalent resolved detail or a full3,905-case corpus audit. Existing Auto
selects all10 asian_faces clear cases and none of10 FFHQ clear cases, so it cannot
guarantee a clear-image bypass. All24 native usable labels and source separation
remain. The older transitive-quality-source omission and preparation failure are
retained; current sources are prospectively bound. There are zero local DGP or
completion forwards, gradients, parameter updates or training epochs in these
two diagnostics; only the conversion audit uses frozen recognizer inference.

Next design: broaden resolution/framing/degradation coverage from genuine audited
HQ TRAIN references and investigate input-conditioned spatial learning in a
separate current-DGP copy. This is an evidence-based design hypothesis, not an
executable training packet. More epochs of the rejected recipe are not justified.
The current checkpoint remains Phase3 plus two selected fine-tuning epochs; its
total ancestral count is unestablished. Additional epochs1/2/5 remain a proposed
five-epoch study after finite qualification, preserving existing quality gates.

All actual training/autograd studies remain manually launched inside tmux on the
existing L4 until the user names another VM. This turn did not connect, start,
clean or train on the VM. Returned guest receipts show the human's completed run;
current power/free-space state was not checked. Earlier stopped-VM statements are
historical. Direct VM authorization remains inventory-first, hash-bound maintenance
preserving research and active work. No historical pilot is resumed automatically.

All five milestones remain required: native provenance/terms/overlap/input-only
criteria; same-input resize/Phase3/DGP/pretrained comparisons; justified finite
training; primary DGP/Auto/override integration preserving the existing design;
useful development outputs, independent final review, regressions and bundled
inline Playwright. Native CCTV remains unpaired, separate from paired synthetic
PSNR/SSIM. Prioritize Asian capture sources without ethnicity or Zamboanga claims.
All visible facial features remain in scope; request clearer/less-covered crops
when information is insufficient. Masks, sunglasses, strong lens glare, hands,
obstructing hair, scarves and objects require separate plausible completion,
automatic-area preview/correction, visible-appearance/clear-glasses/ordinary-hair
preservation, original/mask/result and PNG/bundle downloads, and separate automatic
versus assisted evidence. Exact hidden identity is not claimed. Goal active/incomplete.

Read CCTV_DGP_FINITE_GUARD_V1_R1_RESULTS.md,
CCTV_DGP_FINITE_GUARD_QUANTIZATION_V1_RESULTS.md,
CCTV_DGP_POST_FINITE_GUARD_INPUT_COVERAGE_V1_RESULTS.md and
CCTV_DGP_POST_FINITE_GUARD_TRAINING_REVIEW.md. This status supersedes historical
pending/prepared/stopped statements below without deleting their evidence.
Exact preceding bytes: outputs/cctv_dgp_finite_guard_v1_r1_closure/before/.


**Latest status - 10 October 2026: group-conflicts and covering-score reviews closed; finite-guard R1 prepared/verified, not run.**

All 12 group-conflicts quality decisions fail despite a certified initial
direction. All 400 unique outputs were reviewed: small changes remain soft,
larger changes regress finite preservation. Unchanged direction/epoch extensions
are not justified. The user selected the best approach after the diagnostic
question. A separate finite R1 mechanics study recomputes 64 objectives from
both sampled TRAIN cohorts and admits at most three accepted training changes,
nine proposals and 960 gradient queries. These are actual training changes,
not zero-update probes or completed epochs. Former cross-check TRAIN now fits;
no DEV/final role is changed. Original1% and all preservation thresholds remain.
Microchange acceptance does not qualify a model or resume any stopped recipe.
Manual commands: CCTV_DGP_FINITE_GUARD_V1_R1_VM.md. No automatic continuation.

The user chose the existing L4 VM. The last API check reports it stopped;
guest free space is unknown. No startup, deletion or training occurred. Require
7GiB free after installation and live inventory/hash-bound preservation before
maintenance. All learning stays manual in tmux; inference/audits can be local.
Original checkpoints, splits, gate failures and research assets are preserved.

The covering-score diagnostic is fully audited and all36 exposed development
inputs reviewed. Automatic proposals remain unqualified; 28 post-hoc fixed
score witnesses cannot be corrected by a single pointwise threshold. No
completion generator runs or app weights change. See the two latest results
reports and PROJECT_HANDOFF.md for numerical, visual and audit limits.

The full own-trained DGP-led 256x256 goal remains active/incomplete: all five
milestones, useful native DEV outputs, independent final identities/reviewer,
paired synthetic evidence kept separate, all seven automatic/assisted covering
families and bundled inline Playwright remain required. The goal and later user
decisions retain precedence over historical wording. Exact preceding bytes:
outputs/cctv_dgp_group_conflicts_v1_closure/before/.


**Latest diagnostic status — 10 October 2026: loss-balance return reviewed; group-conflicts diagnostic verified, not launched.**

All nine reviewed loss-balance trials fail preservation. Some exceed1% structure
gain on two small photographic TRAIN cohorts, but clear appearance regresses and
outputs remain soft. All1,000 unique outputs are visually reviewed. The mean
recognition-protection assumption failed; further epochs under that recipe are
not justified. V42's full-corpus failure and all original preservation gates remain.

The user requests the best evidence-based approach before training. A separate
manual diagnostic measures32 source/profile objectives before any optimizer.
There are160 finite VM-only gradient queries, zero optimizer updates and zero
epochs. Only a certified improving direction permits three reset trials. A
failed certificate completes the diagnostic without training and proves no
global infeasibility. Independent transfer verification passes; the packet is
prepared and verified, not run. See CCTV_DGP_GROUP_CONFLICTS_V1_VM.md and
CCTV_DGP_GROUP_CONFLICTS_V1_DESIGN.md for exact commands and finite stop rules.
The declared SSIM surrogate does not replace scientific evaluation; saved group
vectors are audited without locally replaying individual autograd queries.

Current/Phase3 checkpoints, splits, research caches and failed gates are preserved.
No failed pilot is resumed and no automatic training or app promotion occurs.
All five milestones, native DEV usefulness, independent final identities, seven
automatic/assisted covering families and bundled inline Playwright remain required.
The full goal is active/incomplete. This status supersedes historical pending
statements below without changing the goal or manual execution boundary.
Exact preceding bytes: outputs/cctv_dgp_original_loss_balance_v1_closure/before/.


**Latest diagnostic status — 9 October 2026: original-feature return verified; loss-balance diagnostic prepared and verified, not launched.**

All nine original-feature trials fail preservation, including the decoder trial
that exceeds 1% structure gain on two small photographic TRAIN cohorts. All
1,000 outputs have received assistant development visual review. This does not
solve V42's full-corpus 0.00692364% failure or establish useful native restoration.
The retained current DGP and original Phase 3 checkpoint are unchanged.

A distinct finite diagnostic tests original-decoder structure descent combined
with original-feature identity descent, using audited saved derivatives. It has
zero new gradient queries, zero optimizer updates and zero epochs. Its independent
packet audit verifies all transferred members, source bindings, nine formulas,
local trial denials and 100 initial-output parity cases. Manual tmux instructions:
CCTV_DGP_ORIGINAL_LOSS_BALANCE_V1_VM.md. No VM execution is launched automatically.
Scientific requirements remain unchanged; every failed trial is retained. More
epochs require a justified recipe and full-corpus, DEV and native review.

Read CCTV_DGP_ORIGINAL_FEATURE_PROBE_V1_RESULTS.md and the latest PROJECT_HANDOFF.md.
All five restoration milestones, all seven automatic/assisted covering families,
independent final review and bundled inline Playwright remain required. The full
goal is active/incomplete. This execution status supersedes pending-return
statements below without changing the agreed goal or manual training workflow.
Exact preceding bytes: outputs/cctv_dgp_original_feature_probe_v1_closure/before/.


**Latest verified execution — 9 October 2026: V42 failed at 50 updates; the full goal remains active.**

The downloaded V42 return is independently audited. Its proposed five-epoch
study stops before completing one epoch: delivered structure improves only
0.00692364% against the retained1% requirement, and compound appearance
preservation fails in both sources. Raw outputs fail too. Only the added spatial
decoder trained; the current app DGP and its normalization remain unchanged.

This execution status supersedes the prepared-only or not-launched status of
that first study recorded below. It does not alter the full goal, dataset/final
boundaries, scientific thresholds or manual VM workflow. More epochs require a
distinct evidence-justified recipe; V42 must not be resumed unchanged or have its
failed stop bypassed. The architecture diagnostic question required by AGENTS.md
is pending. No answer is inferred from elapsed time.

Read CCTV_DGP_V42_RESULTS.md and CCTV_DGP_POST_V42_ARCHITECTURE_REVIEW.md for the
verified failure and proposed design discussion. All five restoration milestones,
all seven automatic/assisted completion families, useful native development
review, independent final review and bundled inline Playwright remain required.
No new learning recipe or app promotion occurs in this milestone. Previous
canonical bytes are retained in outputs/cctv_dgp_v42_return_review_v1/before/.


# DGP restoration of degraded CCTV faces: execution Goal

## Current goal — user decisions, 9 October 2026

Develop and verify our own trained DGP as the primary model for useful 256×256 restoration of degraded CCTV face crops for the thesis “Forensic Deep Generative Prior Face Reconstruction for Degraded CCTV Video in Zamboanga City,” in C:\xampp\htdocs\YEAR 4\Testing. Follow SYSTEM_WORKFLOW_AND_GOAL.md, PRACTICAL_OUTPUT_SCOPE.md and the latest decisions recorded in CCTV_DGP_CURRENT_TRAINING_IMPROVEMENT_PLAN.md; later user decisions take precedence over historical documents and the outdated manuscript.

Start improvement from an isolated copy of the DGP currently used by the main local application: outputs/cctv_dgp_face_code_fit_vm_v12_r2/weights/dgp_v2.pth, SHA256 646fbb11674e8882ea41e251d6e6e7d468c963979b48c93132def8292834e24b. Its tensors match the retained identity-v2 epoch-2 checkpoint: original Phase3 training plus two selected fine-tuning epochs and 226 updates in that selected branch. The complete ancestral epoch count is unconfirmed. Rejected Phase4 epochs 27–31, later failed pilots and zero-update diagnostics do not extend the current app weights.

Investigate longer training where evidence justifies it, alongside the demonstrated supervision and reconstruction-path limitations. Verify the reconstruction objective, trainable spatial/feature path, normalization and learning-rate design before freezing a new executable recipe. The initial proposed continuation compares additional epochs 1, 2 and 5 under one finite protocol, with a five-epoch maximum for that first study; this is a design target, not a prepared or launched run. Longer finite extensions require independently audited preservation and useful development evidence. Preserve existing scientific thresholds and every original failure; do not bypass failed stops, weaken gates, rerun immutable historical pilots or repeat failed recipes unchanged. More epochs, lower training loss, a higher PSNR or archive completion alone do not establish useful restoration.

Preserve all visible facial structure and appearance, accepting some softness; request a clearer or less-covered crop when usable information is insufficient. Support one already cropped frontal or mildly turned face and preserve the existing local application's design. Use the already audited native CCTV development crops to guide review. Reuse audited genuine high-resolution clean training references and approved source replay before unnecessary acquisition; acquire additional public references only for demonstrated gaps, with audited provenance, terms, resolution and overlap limits. No real Zamboanga CCTV samples exist yet. Prioritize Asian capture sources where available, report sources separately and infer neither ethnicity nor local performance. Native CCTV without an aligned clean reference remains unpaired evidence, never a fabricated clean supervision target. Keep paired synthetic PSNR/SSIM separate. Reserved final identities remain outside optimization and development tuning.

Complete five milestones:
1. Freeze public native CCTV development and separate labeled final evaluation identities, provenance, terms, resolution, overlap/exposure limits and input-only review criteria.
2. Compare basic resizing, retained original Phase3 DGP, the current retained DGP and declared pretrained restoration baselines on identical prepared 256×256 inputs; separate raw model outputs from display processing, structure preservation and insufficient-information cases. Pretrained restorers are comparison baselines.
3. Fix demonstrated processing/learning limitations and prepare justified finite VM pilots or continuations with preserved original checkpoints, splits, source caches and gate failures. Select checkpoints by preserved structure, source/profile results and useful native development outputs rather than their epoch number.
4. Integrate the qualified DGP as the primary local restorer with automatic restoration selection and a user override, preserving the application's design.
5. Independently review useful development and final outputs, run meaningful regressions and verify the complete app flow through bundled inline Playwright. Update PROJECT_HANDOFF.md at milestones.

Separately qualify plausible completion for masks, sunglasses, strong lens glare, hands, obstructing hair, scarves and other objects. Show the automatic removal area for optional correction before generation; preserve clear glasses, non-obstructing hair and visible appearance, allowing only a small documented margin. Request a less-covered image when too little face remains. Return one plausible estimate alongside the original and mask, with PNG and optional original/mask/result bundle downloads. Do not claim exact hidden identity. Report automatic and assisted results separately.

Inference, metadata review and independent audits may run locally. All actual training remains manually launched inside tmux on the existing NVIDIA L4 g2-standard-4 VM at ~/forensic-dgp until the user identifies a replacement VM and its environment/data bindings are verified. Prepare exact pasteable Windows Google Cloud SDK upload/download commands and VM install/verify/tmux/launch commands with verified transfer files; use one remote source per gcloud SCP download command. Independently audit returned artifacts. Direct connection to the existing VM is authorized for inventory-first, hash-bound storage maintenance, preserving research assets, active workloads, original checkpoints, splits and failed gates; this does not authorize automatic new training.

The user reports USD 34 remaining, chooses no additional monetary cap per experiment, and will notify us near USD 5 to export before changing VMs. Finite epochs/updates, measured timing/projection, wall-time, cache, inference, VRAM, storage and export limits still apply despite no fixed overall GPU-hour cap. Before migration, verify a local backup of weights, optimizer/scheduler/RNG state, next epoch/update and schedule position, architecture code, environment versions, data/split/provenance hashes, logs and failures. Distinguish exact resume from a newly declared weights-only fine-tune; never automatically resume a scientifically failed recipe. Thesis submission or defense is around or after 20 November 2026; reserve time for independent final review, app verification and evidence-based reporting.

Complete only when the DGP-led local workflow works and reviewed outputs meet the agreed restoration and seven-covering-family scope. Acquisition, a finished training run, a diagnostic or a successful archive is not completion.

This current goal supersedes conflicting historical directions below. The initial
five-epoch study is a proposed design, not a prepared packet or a launched run.
The full goal remains active/incomplete. Current checkpoint evidence and the
preparation rationale are recorded in CCTV_DGP_CURRENT_TRAINING_IMPROVEMENT_PLAN.md.

The exact preceding document is archived in
outputs/cctv_dgp_goal_update_20261009_v1/before/SYSTEM_WORKFLOW_AND_GOAL.md.
Its complete original bytes are retained below as historical workflow evidence.


# DGP restoration of degraded CCTV faces: execution Goal

**Latest application milestone - 7 October 2026: imported masks require fresh review before generation.**

A reproduced frontend race allowed generation with the previously reviewed mask
while a replacement mask loaded, then displayed the old-area result after review
was cleared. Import now locks generation immediately and clears approval. Invalid
imports recover controls without approval; a superseded import cannot alter or
unlock a newer face upload. Three inline Playwright ordering regressions pass.

The real local app completed 11 generation and 11 proposal requests in 66.29 seconds
across all seven covering families, uncovered/clear-glasses controls and On/Off/Auto
selection. Two insufficient/unsupported input decisions submit zero generation
requests. All 11 PNG downloads match the prior audited neural processing exactly;
the independent audit also verifies 1,521,399 visible Off bytes and a complete
original/mask/result/raw-DGP bundle. Responsive 375/768/1280 checks have no overflow,
page exceptions or console errors. The existing design and backend are preserved.

This verifies review ordering and functional development flow. The cases are
previously exposed photographs, not native CCTV or independent final evidence.
Their legacy native suffix means original photo. Covering generation uses reviewed
operator masks; automatic-family quality remains unqualified. The prior finger,
scarf, gaze and DGP visible-softening defects are unchanged. No hidden-identity,
ethnicity or Zamboanga-performance claim is made. No training occurred.

The original frontend is archived at
outputs/dgp_mask_review_race_fix_v1/before/static/face_workflow.js.
Historical app22 source/evidence bindings remain verified against that archived
source where required; other bindings stay at their original locations. The
current frontend is separately bound. Original checkpoints, splits, scientific
caches, failure records, previous documents and the actual Windows backup remain.

The user's V31 launch then failed before training: the schedule helper in the
packet root was missing from Python's script import path. A read-only VM status
check retained two terminal tracebacks, confirmed no V31 process/log/outputs and
an idle GPU. A second read-only check verified the command-only PYTHONPATH fix:
all 3,905 cases/781 references pass transfer verification with zero neural,
gradient or optimizer calls. No training was started by the agent and no pilot
file was changed. The packet/protocol/gates are unchanged and no return is
present locally. The corrected launcher supersedes only step4 of the old guide.
Actual new training remains the user's manual tmux workflow on the existing L4 VM.
Useful native restoration, automatic/assisted covering quality and independent
final review remain outstanding. Goal active/incomplete.

[Application ordering fix and verification](<C:/xampp/htdocs/YEAR 4/Testing/CCTV_DGP_MASK_REVIEW_RACE_FIX_V1.md>)
[V31 five manual steps](<C:/xampp/htdocs/YEAR 4/Testing/CCTV_DGP_PROFILE_BATCHES_V31_VM.md>)
[Corrected V31 tmux launch](<C:/xampp/htdocs/YEAR 4/Testing/CCTV_DGP_PROFILE_BATCHES_V31_LAUNCH_IMPORT_V1.md>)

The complete previous document body is preserved below. Its unchanged-app-source
statements refer to the archived pre-fix frontend; neural processing is unchanged.


**Latest research milestone - 7 October 2026: sampling diagnostic audited; V31 paired-batch pilot ready.**

The human returned the distinct V30 sampling-gradient diagnostic. The independent
local audit verifies all 756 files, 280 saved component-gradient queries and 200
CPU inference outputs across original/stopped states. VM diagnostic runtime was
34.729 seconds, with zero optimizer updates, zero backwards and no new checkpoint.
All 12 selected decoder tensors have finite nonzero improvement gradients.
The audit used no local autograd or training. Original models and V30's failed
50-update/0.805717% structure requirement remain preserved.

The stopped candidate drifts on four unexposed clear TRAIN controls: the largest
raw observed-pixel MSE increase is 16.2719%. One motion case also activates the
pixel-regression hinge. Preservation gradients therefore protect a measured
regression. The stopped negative total-objective direction increases the
whole-observed-detail term in both measured cohorts. These are local gradient
and raw-loss observations, not a reconstruction of AdamW steps or a unique causal
proof. The cohorts are TRAIN data; source labels are not ethnicity. Raw objective
drift is separate from delivered PNG gates and paired synthetic PSNR/SSIM.

V31 changes batch formation only: each approved reference's clear control and
four degradations share one batch. The full 781-reference/3,905-case TRAIN corpus,
original DGP initialization, mean-centered decoder path, 12 trainable tensors,
seven losses, fixed initial50 normalizers, AdamW and all numerical gates stay
fixed. The frozen V30 permutation determines reference order. The first 781
updates cover every TRAIN case once; 19 further batches complete the finite
800-update maximum. The unchanged early gate stops at50 unless structure gain
reaches1%; final gates require10%, all17 preservation groups, nonnegative source
gains and mean-only fraction<=20%. No V30 continuation or unchanged failed rerun.

The nine-file 751,432-byte transfer packet is independently verified, including
10 regressions, Python3.10 syntax, read-only Bash syntax and rejection of local
training before model imports. Protocol SHA256:
ae91d25066bcb532a8a88bdc48fe401c4b18d73e8d348b47b5d012d5df8e485e.
Execution archive SHA256:
bdf79346b10376b75328dfd2fab3ab3902935982cb9910b4478b401c0c55b9b8.
Actual V31 VM training remains unstarted. Follow the five manual upload/install/
tmux/launch/download steps on the existing L4/g2-standard-4 VM at ~/forensic-dgp.
Require6GiB free; timing, VRAM and export stops are enforced. Historical storage
measurements are not a fresh free-space reading. The local research-cache backup,
original checkpoints, splits, provenance and all failures remain retained.

Returned V31 results require an independent audit and all50 TRAIN preview review.
Only passing capacity justifies repeating the fixed520 paired DEV cases and24
unpaired native development crops. V29 development/native failures remain binding;
reserved final pixels remain unopened. No new native or evaluation images were
used in this diagnostic or packet. No real Zamboanga CCTV evidence exists yet.

The DGP-led app is unchanged. Useful native structure, all seven covering families
with separate automatic/assisted review, independent final review and the full
bundled inline Playwright app flow remain required. Preserve clear glasses,
non-obstructing hair and all visible facial appearance; request a clearer or
less-covered crop when usable information is insufficient. Goal active/incomplete.

[Audited diagnostic findings](<C:/xampp/htdocs/YEAR 4/Testing/CCTV_DGP_V30_SAMPLING_GRADIENT_V1_RESULTS.md>)
[V31 five manual steps](<C:/xampp/htdocs/YEAR 4/Testing/CCTV_DGP_PROFILE_BATCHES_V31_VM.md>)

The complete earlier document body is preserved below. Its diagnostic-unrun
language is historical and superseded by this audited human return.


**Latest research milestone - 7 October 2026: V30 independently audited; early structure gate failed.**

The human ran V30 on the existing NVIDIA L4 and returned the archive. The local
independent R1 audit verifies 27,622 files, both complete 3,905-case TRAIN
snapshots, raw/display separation, source and checkpoint provenance, saved
gradients and CPU inference replay. V30 stopped after 50/800 optimizer updates:
structure gain 0.805717% against the unchanged 1% early requirement. No
800-update result exists. Preserve the stopped checkpoint, logs and failed gate;
do not resume, rerun unchanged or promote V30 into the app.

All 17 fixed group preservation checks pass at update50. Source-group gains are
1.072549% for dataset/asian_faces/degraded and 0.742743% for
dataset/thumbnails128x128/degraded. This is paired synthetic photographic TRAIN
evidence; source labels do not establish ethnicity, real CCTV performance or
Zamboanga performance. No new native, development or reserved-final cases were
opened. All 50 fixed preview rows/250 exact comparison cells were visually
reviewed: visible appearance remains broad, but the outputs are soft and show
little useful whole-face clarity gain. All visible facial features remain in scope.

The original local checker failed at its last early-receipt equality comparison
because CPU/VM aggregate arithmetic differed by 1.1102230246251565e-16. Its source
and failure logs remain unchanged. A separate R1 checker allows 1e-12 receipt
arithmetic difference, while retaining the exact 1% threshold and both failed
gate decisions. This checker repair does not change the V30 recipe or qualification.

After the required V28-V30 architecture discussion, the user selected 'Apply the
best approach'. Before another training recipe, a distinct metadata-selected
sampling-gradient diagnostic compares two 50-case matched TRAIN cohorts at the
original and stopped update50 states. It has 280 gradient queries, zero optimizer
updates, a 600-second worker limit, a 900-second external limit and finite export
limits. These subsets are not held-out evaluation. Selection uses frozen schedule
and source/profile metadata only; reference repetition is matched one to one.
No new loss, training recipe or checkpoint is
created. The independently verified packet remains unrun and requires the human
upload/install/tmux/download steps on the existing L4/g2-standard-4 VM. All new
actual training continues under the user's manual command workflow.

The prior authorized cleanup reclaimed 36.75 GiB and measured 43.13 GiB free
before V30 ran; those are historical measurements, not a fresh disk reading.
The actual Windows research-cache backup remains preserved. Research markdown,
provenance, original checkpoints, splits and all prior failed gates remain intact.

The DGP-led app remains unchanged. V29 development/native qualification failures
remain binding. Useful native restoration, automatic and assisted evidence for
all seven covering families, an independent final review and the full bundled
inline Playwright app flow remain required. Preserve clear glasses and visible
appearance; request a clearer/less-covered crop when structure is insufficient.
Goal active/incomplete.

[V30 results](<C:/xampp/htdocs/YEAR 4/Testing/CCTV_DGP_BROADER_MEAN_V30_RESULTS.md>)
[Next manual diagnostic steps](<C:/xampp/htdocs/YEAR 4/Testing/CCTV_DGP_V30_SAMPLING_GRADIENT_V1_VM.md>)
[Independent V30 audit](<C:/xampp/htdocs/YEAR 4/Testing/outputs/cctv_dgp_broader_mean_v30_independent_audit_r1.json>)

The complete earlier document body is preserved below. Its V30-not-started
language is historical and superseded by this audited human run.


**Latest maintenance â€” 7 October 2026: inactive VM caches removed after full local backup verification.**

The latest explicit user authorization permits direct maintenance connection to
the existing forensic-dgp-thesis/us-central1-a VM. Fresh inventory confirms the
original instance and 100 GB boot-disk identities, an idle NVIDIA L4 and no active
cache readers. The offered SSH key matches the previously trusted fingerprint;
the initial uncached-IP stop is retained. Linux restricted the first added
process-file check; a separate read-only privileged inspection established the
safe correction. The stopped verifier and original source remain preserved.
Exact file unlinks still run as the original VM user, without root deletion.

All 4,431 actual Windows cache files/39,448,585,279 bytes pass fresh full SHA256
verification against the frozen source inventory. All 4,431 VM hashes match those
copies. Cleanup removes only 4,429 inactive .bin/.npz files:39,442,892,400 logical
bytes. These are four expanded-feature binary files and 4,425 closed V16 r2 cache
files. Their two original state.json metadata files remain on the VM, together
with original DGP checkpoints, datasets, splits, source, research markdown,
provenance, logs and failed gates. The Windows actual cache copies remain intact.
Closed historical recipes must stay closed; restore a required historical cache
from the verified local files before a justified later read-only audit.

Initial live free space is 6,905,155,584 bytes (6.43 GiB).
The removal receipt measures 39,455,641,600 additional free bytes
(36.75 GiB); final live free space is46,312,128,512 bytes
(43.13 GiB). Independent VM verification confirms every planned
deletion, 210,274 retained file hashes,
388 retained tensor stamps and all 5,757 current
V30 dependency bindings, including 5,467 TRAIN files. The existing CUDA runtime
remains available. No active workload is killed, VM stopped or model promoted.

V29 remains rejected for application promotion. V30's distinct broader TRAIN
coverage packet and finite 800-update protocol remain unchanged and verified.
V30 is not installed or started. Actual training remains the user's manual
upload/install/tmux workflow on the existing L4/g2-standard-4 at ~/forensic-dgp.
Use the five exact steps in CCTV_DGP_BROADER_MEAN_V30_VM.md, including separate
PuTTY-compatible downloads. The unchanged 6 GiB requirement is satisfied.
Returned results still require independent audit, paired synthetic and unpaired
native evidence kept separate, useful native structure review and preservation
gates. Useful DGP restoration, all seven covering families, independent final
review and full bundled inline Playwright app verification remain required.
Goal active/incomplete.

[Cleanup evidence](<C:/xampp/htdocs/YEAR 4/Testing/outputs/cctv_dgp_vm_storage_cleanup_20261007_v2/closure_manifest.json>) Â·
[V30 manual steps](<C:/xampp/htdocs/YEAR 4/Testing/CCTV_DGP_BROADER_MEAN_V30_VM.md>) Â·
[Actual Windows cache backup](<C:/xampp/htdocs/YEAR 4/Testing/outputs/cctv_dgp_local_research_cache_backup_20261007_v1/complete.json>)

Earlier complete document bodies remain preserved history. Current cache
locations and live free space above supersede earlier storage readings; prior
restoration findings, quality failures and research scope remain unchanged.


**Latest maintenance — 7 October 2026: actual research caches backed up locally and audited.**

The user chooses a Windows backup in preparation for a possible later Google Cloud VM.
All three historical caches are now copied as actual files: 4,431 files /
39,448,585,279 bytes (36.74 GiB), including both expanded-feature caches and the
V16 r2 cache. Every reconstructed file matches its frozen VM SHA256; a separate
read-only verifier re-reads all 4,431 local files and passes. The source caches
retain their original sizes, inodes and timestamps. No source cache is removed.

The initially selected cloud snapshot was an agent destination error. It is deleted;
independent live checks confirm its absence and the original running VM/disk.
The earlier migration recovery ZIP contains metadata only and its cloud restore
instructions are obsolete; its original bytes and the rollback evidence remain.
The actual local files and current restore guide supersede that route.

The installed Windows SDK cannot carry the interactive binary control stream
through its automatic PuTTY stdin reply. The corrected transfer uses the SDK-generated
console SSH connection with cached host-key checking. A demonstrated Windows
temporary-path limit was fixed using short names; completed blocks were retained
through an owned local-transfer pause. The early failures, original partial and
source revisions remain. Only 147 redundant new Windows transfer blocks are
removed after the independent full-file audit, reclaiming 36.74 GiB locally.

This is a three-cache backup, not a boot-disk image or proof that every separate
checkpoint, dataset and environment is bundled. Existing model, split, provenance
and gate-failure files remain. The GPU is idle; latest source free space is
6.43 GiB, still above the unchanged 6 GiB V30 requirement.
Actual training remains human/manual on the existing L4 at ~/forensic-dgp.
No new VM, optimizer, pilot, application change or source shutdown occurs.
V29 remains rejected for application promotion; useful DGP restoration, all
covering families, independent final review and the full app flow remain required.
Goal active/incomplete.

[Local backup and restore guide](VM_CACHE_BACKUP_RESTORE_20261007_V1.md) ·
[Independent local audit](outputs/cctv_dgp_local_research_cache_backup_20261007_v1/independent_local_cache_audit.json) ·
[Source/resource audit](outputs/cctv_dgp_local_research_cache_backup_20261007_v1/source_final/independent_source_resource_audit.json)

Earlier complete document bodies remain preserved history.


**Latest maintenance — 7 October 2026: VM cleanup complete; V30 remains ready for manual VM execution.**

The user explicitly authorizes direct connection and removal of unnecessary
VM files to reclaim at least1.3GiB, or more, before the6GiB V30 space check.
The existing forensic-dgp-thesis/us-central1-a connection is used for storage
maintenance only. Actual training remains manual VM execution on the existing
NVIDIA L4/g2-standard-4 at ~/forensic-dgp; no pilot or optimizer is launched.

Ten duplicate top-level transfer/result archives and four redundant pretrained
recognizer copies in closed V22/V23/V24/V25 packets are removed only after full
Windows backups and VM SHA256/stamp verification:1,915,161,691 logical bytes.
The four recognizer copies match the retained active V27 VM weight and their
original local packet manifests. V26's two-link copy stays; unlinking a shared
copy would not reclaim that storage. The initial conservative inventory stop
and the diagnostic that established this distinction are both retained.
Original trained DGP checkpoints, data, splits, logs, source, all failed gates
and all Windows recovery files remain. Historical V22–V25 recognizer copies
can be recovered from the exact Windows paths in the plan; do not rerun those
immutable failed pilots automatically. Research markdown files are preserved.

The pip download-cache scan yields0 eligible files.
Independent live and Windows audits verify
the exact deletion ledger, 210,274 protected file hashes,
4,817 retained tensor stamps and5757 current
dependency bindings, including all5467 V30 TRAIN assets. The three historical
scientific caches remain intact:4431 files/39,448,585,279 bytes. All11 tmux
sessions were at shell prompts before maintenance; the L4 remains idle and the
existing CUDA runtime is available. No process is killed or VM stopped.

The removal receipt measures 1,915,215,872 additional free bytes
(1.78GiB). Final live free space is 6,907,056,128 bytes
(6.43GiB), satisfying the unchanged6GiB V30 requirement.
V30 has not been uploaded or installed on this VM. Use the existing five manual
steps; its packet, protocol, finite800-update schedule and quality gates are
unchanged. Cleanup does not qualify V29 restoration or promote a model.

All previous5934 research bindings, the deeper371/1041/586/50/668/cleanup60/
309/66/692/299/697/513 history, complete document bodies and app22 bindings
remain preserved. Useful DGP native restoration, all seven covering families,
independent final review and the full local app flow remain required.
Goal active/incomplete.

[Cleanup evidence](<C:/xampp/htdocs/YEAR 4/Testing/outputs/cctv_dgp_vm_storage_cleanup_20261007_v1/closure_manifest.json>) ·
[V30 manual steps](<C:/xampp/htdocs/YEAR 4/Testing/CCTV_DGP_BROADER_MEAN_V30_VM.md>)

Earlier complete document bodies below remain preserved history. Their research
findings and quality failures retain their original scope; the current verified
free space above supersedes earlier storage readings.


**Latest research milestone — 7 October 2026: V29 TRAIN capacity audited;520 paired DEV cases and24 native faces reviewed; broader-coverage V30 prepared for manual VM execution.**

The downloaded V29 return matches SHA256
ecf88626b31dee6b0ceb65f6a2a76e534cea8079183e0efd143c6bcb16fb5ae2,
294,561,124 bytes. The original frozen independent checker passes837 files,
38,394,279 gradient values, all four checkpoints and500 forward-only CPU calls.
All800 VM updates complete. Final50-case TRAIN structure gain20.0551%, all17
preservation groups and the original brightness gate pass. All50 TRAIN faces
are viewed. Original checkpoint, encoder/head4, buffers, splits and failures
remain. TRAIN capacity is necessary evidence and does not qualify restoration.

The separately frozen single-crop inference path passes seven contract checks
and50 CPU/VM parity cases: raw error<=2.355e-6, PNG<=one byte, exact padding.
It requires original baseline plus candidate plus the fixed mean projection.
The existing app model/design remains unchanged; this is a benchmark candidate.

All520 paired photographic DEV cases are retained and independently audited.
Aggregate degraded landmark structure gain is1.3516%; asian_faces source
worsens3.4051%, thumbnails128x128 improves2.3437%. There are21 fixed group/metric
regressions,17 in ArcFace. ArcFace is a preservation diagnostic, not identity
accuracy. All50 fixed DEV preview faces are actually viewed: sharper edges
can coexist with changed eyes, mouth or expression. Reject V29 app promotion.
Paired synthetic MSE/PSNR/SSIM do not become native CCTV metrics.

All24 frozen ChokePoint C1 native development faces are independently audited
and actually reviewed against resizing, retained Phase3, original DGP and
declared CodeFormer on identical256 inputs. V29 retains broad face arrangement
but lacks convincing useful clarity over resize. The previously useful02_t033
still needs clearer eyes/nose/lips. Usable inputs are not reclassified from
model softness. Source capture country is unspecified in acquired metadata;
no ethnicity or Zamboanga performance is inferred. No final reserved pixels
are opened. The separate58 cases /45 namespaced final identities remain reserved.

V30 changes optimization coverage from10 to the already approved781 TRAIN
references /3905 existing cases, retaining800 updates /4000 sample exposures.
The first781 batches cover every case once;19 predetermined second-epoch
batches follow. Original initialization, same mean-centered decoder path,
selected12 tensors, seven losses, fixed50 normalizers, AdamW and numeric gates
stay unchanged. This tests the coverage hypothesis, not a proven unique cause.
No V9 trained weights, V29 continuation or unchanged historical pilot is run.
The104 DEV references remain outside optimization; person/pretrained overlap
remains unknown. Every original checkpoint and failed gate remains preserved.

The eight-file746,356-byte packet uploads no images or weights. Protocol:
b7b6e26bef102b5ca873d4ff01cd4c4df6bcabfb898e56bd899757ac99b65aa1.
Archive:7a3d58db1e6d14a1c5a589269ec6922cd8fbe8d0f1834f4c242fb1d2cf72ac39.
Independent metadata/packet/source checks pass5467 TRAIN files, exact50-case
source parity, Python3.10 parsing, Windows pre-neural training rejection,
seven return-boundary regressions, readonly Bash syntax and five manual steps.
The original local Bash checker failed because the process sandbox blocked
its signal pipe; the source and failure remain. The distinct R1 checker uses
the separately observed readonly external Bash check. No packet, loss or gate
changed. This is a local setup repair, not a VM/model success.

Actual V30 training has not started. Human gcloud upload/SSH/tmux is required.
Require idle existing L4/g2-standard-4 and6GiB free, without deleting research
assets. Initial proof300s /70 queries, cache900s, fit3600s, worker4500s,
external4800s+30s grace, export900s/external930s+30s grace, allocatedVRAM<=20GiB.
Snapshots0/50/400/800 include all3905 TRAIN outputs and mean controls; only50
raw previews are exported. Early50>=1% structure, final800>=10%, both sources
nonnegative, all17 groups and brightness<=20% remain. Final800 only, no resume,
overwrite, unchanged retry, competing-task termination or automatic promotion.
The prospective independent return checker checks all saved outputs/gates,
the initial proof and frozen partitions, every final3905 case and checkpoint
previews through CPU inference only, with7200s audit cap.

All prior371 bindings and1041/586/50/668/cleanup60/309/66/692/299/697/513 history,
three complete document bodies and app22 bindings are preserved. All seven
covering-family automatic/assisted requirements, useful native restoration,
full app flow and independent final review remain unfinished. Goal active/incomplete.

[V29 results](<C:/xampp/htdocs/YEAR 4/Testing/CCTV_DGP_MEAN_CENTERED_DECODER_V29_RESULTS.md>) ·
[V30 five manual steps](<C:/xampp/htdocs/YEAR 4/Testing/CCTV_DGP_BROADER_MEAN_V30_VM.md>)

Earlier bodies below are preserved history. V29 pending/not-started language is
superseded by its audited return and failed development qualification. Only
the distinct V30 coverage pilot is ready for manual execution; no failed
historical pilot is automatically rerun.


**Latest research milestone — 6 October 2026: V28 final-state diagnostic audited; distinct mean-centered V29 finite pilot verified for human VM execution.**

The downloaded211,650,756-byte zero-update diagnostic matches SHA256
7ed57b598096e2bf52302b5acba45626a12ec66f17ca3d94f565e704d170cc75.
The R1 independent audit verifies322 regular files and54,848,970 saved gradient
values, displacement and per-component/per-tensor arithmetic. CPU replay uses
50 original-DGP,50 final-DGP and100 recognizer forwards with no local derivatives.
All50 fresh VM initial/final raw and vector parity maxima are exactly zero;
final PNGs match exactly. The VM performs100 gradient queries in30.042s,
zero optimizer updates/backwards/epochs and no new checkpoint. All states remain.

The original local audit stopped on two one-ULP Linux/Windows norm differences
(max5.55e-17). Its original source and failure are retained. The distinct R1
checker allows rtol1e-12/atol1e-14 only for derived arithmetic. Saved arrays,
displacement, archive/source hashes and PNGs still require exact matches.
Four repair regressions pass. No model-quality gate changes.

At the final V28 state, the existing seven-term objective's plain negative
gradient increases the measured RGB-mean penalty (directional derivative
+1.998307315), while clear SSIM and ArcFace hinge penalties decrease.
Preservation gradients are active. This is local first-order evidence, not a
finite-step/AdamW guarantee or a unique historical cause. V28's18.0595% TRAIN
structure gain still fails three gates; the post-training mean control still
fails five. Both remain rejected, with no app promotion or earlier selection.

V29 tests one changed spatial path: subtract the observed RGB mean of candidate
minus frozen same-input original-DGP baseline BEFORE unchanged losses and final
clamping. All profiles use that path without target/source/identity routing.
No learned projection parameters, strength selection, loss-weight change or
surrogate derivatives. Clipping can reintroduce mean shift; the original
brightness and all17 preservation groups still decide acceptance. Original
selected12 decoder tensors train from the original checkpoint in a separate
copy; encoder/FPN/inactive head4 and all evaluation buffers stay frozen.

The schedule/optimizer remain800 updates/80 epochs, AdamW lr0.00003/WD0.01,
clip1, snapshots0/50/400/800, selected12 initial70-query proof before optimizer.
Early50 structure gain>=1%, final>=10%, both source gains>=0, all17 groups and
brightness fraction<=20% remain. Final800 only. No unchanged historical run,
resume, gate weakening, competing-task termination or automatic follow-on.

V29's seven-file30,293-byte packet uploads no images or weights. Protocol:
77565ba437959305f22cff4dd967fc6c3caadbf9dbd4ac91abf8a366577fa72f.
Archive:de158e52c44494638c0477cd7fca2a67ed12f18ad40a9fc38e9ff45cdac67e4d.
It verifies246 original assets plus11 closed V28 and9 diagnostic dependencies.
Python3.10 parsing,17 forward-only/return regressions, all50 exact initial
projection cases, float64 reference arithmetic(max5.96e-8), actual Windows
pre-neural/gradient rejection, readonly Bash syntax and five manual steps pass.
Two local unissued test setup failures remain recorded; no VM work or local
gradients occurred. Actual V29 training has not started. Human gcloud upload,
SSH and tmux remain the only training execution path. Preflight300s, fit1500s,
worker1800s, external2100s+30s grace, export120s/external150s+30s grace;
idle existing L4/g2-standard-4,3GiB free, allocatedVRAM<=20GiB.

All prior1041 bindings and deeper586/50/668/cleanup60/309/66/692/299/697/513
history, three full document bodies and app22 bindings remain preserved.
No new native or reserved-final pixels are opened. These50 paired photographic
TRAIN cases do not establish native CCTV, ethnicity or Zamboanga performance.
V29 return audit and all50-face review must precede separately frozen native/app
qualification. Useful DGP restoration, all seven automatic/assisted covering
families, full app flow and independent final review remain required. The
existing app model/design stays in place. Goal active/incomplete.

[Diagnostic results](<C:/xampp/htdocs/YEAR 4/Testing/CCTV_DGP_V28_PRESERVATION_DIAGNOSTIC_V1_RESULTS.md>) ·
[V29 five manual steps](<C:/xampp/htdocs/YEAR 4/Testing/CCTV_DGP_MEAN_CENTERED_DECODER_V29_VM.md>)

Earlier bodies below are retained history. Diagnostic-ready/pending wording is
superseded by this audited return; only the distinct V29 pilot is ready for
manual execution. V28 and the fixed mean control remain rejected.


**Latest research milestone — 6 October 2026: V28 completed800 but failed acceptance; all50 final faces reviewed; distinct zero-update preservation diagnostic verified for manual VM execution.**

The downloaded291,926,255-byte V28 archive matches
3196e8ab797763e7ced75a57411b973afc3b1b12654632b7662fab3e2e168aac.
Independent audit passes838 regular files,246 original assets,38,394,279
saved gradient values and every checkpoint/raw/PNG/metric at0/50/400/800.
CPU replay performs50 original-DGP,200 candidate-DGP and250 recognizer
forwards without local derivatives or updates. All selected12 initial CUDA
parity/gradient requirements pass. The VM completes800 updates/80 epochs:
fit91.850s, worker124.967s, terminal125.789s, peak allocated1,240,671,744 bytes.
The original checkpoint, encoder/FPN, inactive head4 and evaluation buffers
remain unchanged. The historical R2 all14 failure remains failed.

Final delivered degraded landmark-HF MSE improves18.0595%; both photograph
source gains are nonnegative. V28 still fails the fixed requirements: one
clear-source group losesSSIM and frozen ArcFace similarity, and a constant
RGB-mean shift explains71.3988% of the degraded pixel-MSE gain against20% max.
This fraction refers to pixel-MSE gain, not recovered identity or a percentage
of structural gain. necessary_capacity_pass remains false. Final800 only;
no earlier checkpoint selection, unchanged rerun or weakened gate is allowed.

All50 final TRAIN faces are viewed in ten exact-size sheets;200 saved cells
match their source arrays. Several degraded faces have clearer coarse eyes,
nose and mouth boundaries, with unresolved or altered finer eyes, glasses,
gaze and expression in strong degradation. Training capacity and this primary
assistant review do not establish generalization or independent final review.
Source labels do not imply ethnicity. Native unpaired evidence remains separate.

One fixed target/profile-independent RGB-mean control retains18.0237% landmark-HF
gain and reduces brightness fraction to1.6955%, but fails five preservation
checks. Independent readback verifies all50 arrays/PNGs/recognizer vectors and
17 groups. Only two control examples are visually inspected; no complete
control visual acceptance is claimed. This processing control is insufficient
and is not adopted. All original raw outputs, checkpoints and failures remain.

The next distinct VM diagnostic measures the final V28 original seven loss
gradients plus RGB-mean and clear-only SSIM/ArcFace diagnostic components.
It permits100 queries in ten fixed5-case batches, zero optimizer updates/epochs
or backwards, and no new model checkpoint. Extra components are measurements,
not a new training objective or selected weights. First-order derivatives do
not identify a unique trajectory cause. No actual VM run has occurred yet.
The300s worker/330s external+30s grace and120s export/150s external+30s grace
are finite; idle existing L4/g2-standard-4,2GiB free and allocatedVRAM<=20GiB.
Every source/state/parity/finite/time failure is retained; no resume or follow-on.

The28,332-byte three-file packet uploads no images or weights. It reuses307
closed V28 evidence bindings and246 original assets. Protocol SHA256:
a270f4631f00c55e8c0377f082bc5ec7d593b633c53f91126078a5452e3cf5b0.
Archive SHA256:d700acbb0bf6cdc03de9b6389132e77830937af518bc6ae563fee67698f11a51.
Python3.10 parsing, nine malformed-return/no-training regressions, actual
Windows rejection before neural/gradient work, read-only Bash syntax and five
manual command steps pass. The initial local AST save-check failure occurred
before packet creation/VM work; its source/evidence is preserved. The corrected
check specifically rejects Torch checkpoint/optimizer calls. Human transfer,
SSH and tmux remain the execution path; no assistant cloud action occurs.

Previous586 bindings and deeper50/668/cleanup60/309/66/692/299/697/513 history,
three full document bodies and app22 bindings remain intact. The existing app
model/design stays in place. Original checkpoints, splits, scientific caches,
logs and all prior gate failures remain. No new native or reserved-final pixels
are opened. Useful native restoration, all seven automatic/assisted covering
families, meaningful full app verification and independent final review remain
required. No Zamboanga or exact hidden-identity claim. Goal active/incomplete.

[Audited V28 results](<C:/xampp/htdocs/YEAR 4/Testing/CCTV_DGP_ACTIVE_ORIGINAL_DECODER_V28_RESULTS.md>) ·
[Next five manual diagnostic steps](<C:/xampp/htdocs/YEAR 4/Testing/CCTV_DGP_V28_PRESERVATION_DIAGNOSTIC_V1_VM.md>)

Earlier bodies below are retained history. V28-ready/pending wording and its
training commands are superseded by the rejected final return; only the distinct
zero-update diagnostic is ready for manual execution.


**Latest research milestone — 6 October 2026: R2 all14 failure audited; inactive head4 traced; distinct V28 active-decoder pilot verified for manual execution.**

The downloaded93,557,954-byte R2 archive matches
dfd8e661126d144e87875f236a7eedb339f7803a65f57b92aa0fd128110d4e3e.
Safe import verifies172 regular files. Independent audit checks46,909,863 saved
gradient values,246 original assets,50 original-DGP CPU forwards,50 recognizer
forwards and50 cohort rows. All70 L4 queries ran, then the unchanged all14 gate
failed at line263. Head4's two tensors are exactly zero in every component and
every batch; the other12 aggregate improvement gradients are nonzero. The four
initial preservation terms/gradients are correctly zero. No optimizer, updates,
epochs or new checkpoint occurred. Worker25.232s/supervisor26.416s; no timeout.
Export completion is packaging only. R2 remains failed and must not rerun.

Forward-only trace: every110,592 head4 kernel value is nonzero float32 subnormal
(maximum about6.31e-40). The fourth map is nonzero, but head4 outputs exactly zero
in all50 cases. Separate float64 calculations give2.49e-76–4.29e-76 maxima, below
float32's smallest positive value1.40e-45. Independent NumPy readback verifies
350 activation arrays/100 convolutions. No local derivatives or model changes.
No unique checkpoint-history cause or explanation of all prior failures is claimed.
The fourth map still contributes through the FPN top-down path to other heads.

The actual app encoder divides in NumPy before device transfer; R2 used the older
Torch division after transfer. Both are exactly equal on all256 byte values and
all50 current CPU inputs/raw/PNGs: no CPU normalization defect or gain. CUDA
encoding equivalence was not measured. Earlier canonical wording is corrected
in scope without editing old receipts. V28 ships the exact actual app encoder AST.

The selected separate-original-decoder direction continues as distinct V28:
train head1–head3,smooth,smooth2,final only; keep head4, the encoder/FPN and all
stored buffers frozen. The unchanged original forward has498,627 trainable
parameters in12 tensors. Initial actual-app raw/PNG parity passes all50 CPU
inputs; original state remains unchanged. This does not waive or accept R2 all14.
V28 first requires70 fresh same5-case CUDA gradient queries: every selected12
improvement gradient finite/connected/nonzero, all initial preservation values
and gradients exactly zero, exact raw/PNG baseline parity before any optimizer.

V28 is bounded800 updates/80 epochs; AdamW fixed0.00003, weight decay0.01 and
clip1. Snapshots0/50/400/800; stop at50 if structure gain<1%. Final gain>=10%, both
source nonregression, original17-group MSE/SSIM/ArcFace bounds and brightness
fraction<=20% remain. No checkpoint selection before final800; even capacity
pass requires all50-face review and separately frozen broader/native/app work.
Require idle existing L4/g2-standard-4 and3GiB free. Preflight300s, fit1500s,
worker1800s; supervisor2100s+30s grace; export120s/external150s+30s grace;
torch allocated VRAM<=20GiB. Preserve every failure; no automatic follow-on,
overwrite, resume, failed-recipe rerun or threshold search. Training remains
human transfer/SSH/tmux only. No assistant VM/cloud action occurs.

The29,527-byte eight-file packet uploads no original data or weights.
Archive SHA256:e1484a675ad4330e4615d2f58a70f66ba8a8ad287b78cae66f8555ec4c4b1614.
Protocol SHA256:27c140430673880f1aaab47a9df9af9b33758cc5d8adec53822cd9e05b13bbc8.
Python3.10 parsing, source/packet/app-encoder contracts, actual Windows pre-neural
rejection, unchanged historical shell deadlines, Bash syntax and nine malformed
return/frozen-partition/group/quality regressions pass. A prospective complete
return audit is prepared. Its initial CPU replay was exercised on genuine closed
R2 evidence:50 DGP/50 recognizer/50 cohort rows; this is not a V28 VM result.
The missing CPU device argument found before release was corrected; preliminary
auditor source/checks and failure evidence remain. Actual V28 training is pending.

Previous50 preparation bindings, deeper668/cleanup60/309/66/692/299/697/513,
the three complete document bodies and app22 bindings remain intact. Cleanup
remains previously verified14 duplicate archives/1.64GiB; last measured7.48GiB
free, not a new live disk claim. Original checkpoints, scientific caches, splits,
logs and all earlier failed gates remain. No native or reserved-final pixels are
opened. Photograph source labels remain separate and do not imply ethnicity.
Useful whole-face native restoration, all seven automatic/assisted covering
families, canonical app flow/regressions and independent final review remain
required. No Zamboanga or hidden-identity claim follows. Goal active/incomplete.

[R2 results and branch diagnosis](<C:/xampp/htdocs/YEAR 4/Testing/CCTV_DGP_ORIGINAL_DECODER_GRADIENT_V1_R2_RESULTS.md>) ·
[Five manual V28 steps](<C:/xampp/htdocs/YEAR 4/Testing/CCTV_DGP_ACTIVE_ORIGINAL_DECODER_V28_VM.md>) ·
[App-input clarification](<C:/xampp/htdocs/YEAR 4/Testing/CCTV_DGP_DECODER_APP_NORMALIZATION_V1.md>)

Earlier bodies below remain preserved history. R2-ready/pending and its launch
commands are superseded by the closed failed return; only distinct V28 is ready.


**Latest research milestone — 6 October 2026: selected original-decoder review verified; finite zero-update L4 proof R2 ready.**

The user selects review of a separate copy of the original DGP reconstruction
decoder after V25–V27 failed the unchanged early structure gate. V27 is fully
audited and visually reviewed: 0.221088578% gain versus 1%; its failed50/800
endpoint remains closed. No earlier recipe, checkpoint, split or gate changes.

The separate copy uses the unchanged DGPSynthesizer forward and enables only
head1–head4,smooth,smooth2,final:609,219 parameters in14 tensors. The2,703,488
remaining encoder/FPN parameters and every stored buffer stay frozen/evaluation.
All original shared FPN aliases remain internal; no tensor is shared with the
original model. Exact fresh-reference initial raw/PNG parity passes all50 exposed
TRAIN cases. CPU review21.631s:50 original/51 candidate forwards; no derivatives,
backwards, optimization or checkpoint writes. Actual local differentiation and
invalid-input attempts reject before neural work; partial support is preserved.
Independent251-binding source/layout readback passes. Clamp saturation0–3.1993%
of RGB component values is forward evidence, not a derivative or failure cause.

Only the new original-decoder gradient proof V1 R2 is ready for manual execution.
It is not a training pilot:10 five-case batches,70 component gradient queries,
zero optimizer construction/updates/epochs and no new checkpoint. Require finite
connected nonzero improvement gradients at all14 original decoder tensors;
all four initial preservation terms and every gradient must be exactly zero.
Same-batch fresh canonical baseline/candidate output must match exactly. Legacy
cached evidence remains separate. The seven objective formulas, original CPU-
built fixed filter and all retained capacity/preservation gates remain unchanged.
Save all ten7×609,219 matrices and their sum; no automatic follow-on or app promotion.

Unissued V1/R1 drafts are preserved. Source review corrected GPU kernel creation
and an erroneous shell substring deadline before release. R2's full AST/shell
comparison, Python3.10 parsing, actual Windows neural rejection, Bash syntax and
eight malformed scientific-return regressions pass. Synthetic arrays are not VM
evidence. The released13,816-byte/four-file packet uploads no data or weights.
Archive SHA256:f21634d909e880a26b8ae90e0e6393ff32fc7f2758ebb5312faf59707d82900d.
Protocol SHA256:81127e45a205c44ef685646a4f82911d02b2355dfe24c63772c23e62e66a41f2.
Require2GiB free/idle existing L4; worker600s, supervisor660s+30s grace,
export90s/external120s+10s grace; at most20GiB torch-allocated VRAM. Five pasteable
Google Cloud SDK/SSH/tmux steps include separate PuTTY-compatible downloads.
An independent safe importer/full-matrix/CPU forward/vector/cohort audit is
prepared before execution. It never executes returned source or local derivatives.

Actual L4 proof and returned audit remain pending. No new training protocol is
defined before that proof; no further assistant VM/cloud action occurs. Direct
VM cleanup was separately authorized and verified:14 backed-up duplicate archives
removed/1.64GiB; last verified free7.48GiB. Original scientific caches, sources,
checkpoints, outputs, failures and current V27 archives remain protected.

Previous668 research bindings, cleanup60, deeper309/66/692/299/697/513 histories,
three full document bodies and app22 bindings remain intact. The original DGP
remains primary in the existing app; pretrained restorers are comparisons. No
native/reserved-final pixels, ethnicity or local/hidden-identity claim enter this
TRAIN-only diagnostic. Useful native outputs, all visible facial features,
canonical app flow/regressions, all seven automatic/assisted covering families
and independent final review remain required. Goal active/incomplete.

[Original-decoder review](<C:/xampp/htdocs/YEAR 4/Testing/CCTV_DGP_ORIGINAL_DECODER_REVIEW_V1.md>) ·
[Five manual R2 VM steps](<C:/xampp/htdocs/YEAR 4/Testing/CCTV_DGP_ORIGINAL_DECODER_GRADIENT_V1_R2_VM.md>) ·
[Verified cleanup](<C:/xampp/htdocs/YEAR 4/Testing/VM_STORAGE_CLEANUP_20261006_V2.md>)

Earlier bodies below remain preserved history. Their V27-pending, architecture-
question and prior pilot-launch text is historical; only the new R2 proof is ready.


**Latest research milestone — 6 October 2026: V27 return independently audited/reviewed; structure failure closed.**

The325,777,145-byte archive matches SHA256
1bd4aec2bdc10cd6aca4455f24875b006bf4bca2d1a4f1cae8f4932400b386ab.
Safe import verifies631 regular files. Independent CPU replay checks246 original
assets,100 raw/PNG pairs and metric rows,100 head/110 recognizer forwards and250
own-DGP feature arrays. All original source/state/numerical/cohort/timing checks
remain. No local gradients/backwards/optimizer updates or model changes occur.

V27's delivered degraded structure gain is0.2210885780%, below the unchanged1%
requirement. It stops at50/800 updates with51 backwards; final800 never runs.
All five new feature-readout gradients are active before optimization and the
original five projection gradients are active at update2. Initial corrected
identity value/all36 gradients are exactly zero for all ten batches. Original
DGP and recognizer states remain frozen. Worker30.127s, fit9.615s, supervisor
36.006s and allocated VRAM1,843,821,568 bytes pass original timing/memory bounds.
This is a quality stop, not a transfer, CUDA, timing or absent-gradient failure.

All ten original-detail sheets/50 paired photographic TRAIN cases are reviewed;
all200 source cells are exact. All50 PNGs change, with median degraded correction
1.0231445 byte units/max8. Small tonal changes do not convincingly add eye, nose,
mouth, outline and overall visible-appearance structure together. At stopped50,
saved arithmetic has no original17-group pixel/SSIM/identity violation. This does
not establish final capacity, native generalization or independent human acceptance.

V25/V26/V27 gains are0.0282235%/0.0335636%/0.2210886%. Stop model modifications for
the architecture discussion required by the frozen plan and workspace circuit
breaker. The invalid assumption is that these finite small-head changes on frozen
own-DGP pixels/features would provide sufficient whole-face structure. No unique
optimization cause or impossibility of all heads is proved. The user selects
review of a separate candidate of the original DGP reconstruction decoder;
the direct reply is preserved. Review its source/initial parity before a distinct
finite protocol. No next
pilot/protocol is created, no failed recipe reruns and no gate is relaxed.

The user-authorized direct VM cleanup is complete and independently verified:
14 backed-up archive duplicates/1.64GiB removed;8,036,728,832 bytes(7.48GiB) free;
207,967 protected hashes,4,817 scientific tensor stamps and490 current research
bindings preserved. The39,448,585,279-byte scientific caches/current V27 archives
remain. CUDA available/L4 idle; VM left running. Direct access was maintenance
only; actual training remains manual on the existing L4/g2-standard-4 with verified
transfers and pasteable commands. The exact restated user goal remains recorded.

Completed cleanup60 bindings, previous309 and deeper66/692/299/697/513 histories,
five complete document bodies, app22 bindings and every original checkpoint,
split and failed gate remain preserved. Own-trained DGP remains primary and
pretrained restoration models comparisons. Native CCTV stays unpaired; paired
photographic TRAIN evidence stays separate. Reserved-final identities remain
unviewed. No new native/covering pixels, ethnicity or Zamboanga/hidden-identity
claim enters this milestone. Previously useful inputs remain usable. Useful
native outputs, canonical app flow/regressions, all seven automatic/assisted
covering families and independent final review remain required. Goal active/incomplete.

[Audited V27 results](<C:/xampp/htdocs/YEAR 4/Testing/CCTV_DGP_FEATURE_SKIPS_V27_RESULTS.md>) ·
[Architecture discussion](<C:/xampp/htdocs/YEAR 4/Testing/CCTV_DGP_POST_V27_ARCHITECTURE_REVIEW.md>) ·
[Verified VM cleanup](<C:/xampp/htdocs/YEAR 4/Testing/VM_STORAGE_CLEANUP_20261006_V2.md>)

Earlier bodies below are preserved history. Their V27 install/launch steps and
pilot-pending statements are historical. V27 is closed; do not rerun those steps.


**Latest authorization and maintenance — 6 October 2026: VM cleanup complete; full DGP goal active.**

The user explicitly authorizes direct connection/storage cleanup before continuing
the DGP workflow. The existing gcloud connection to forensic-dgp-thesis in
us-central1-a is verified. This maintenance exception supersedes the prior
no-connection restriction for cleanup only; actual training remains manual on
the existing NVIDIA L4/g2-standard-4 under ~/forensic-dgp. No historical pilot,
instance start/stop, model training, process termination or app promotion occurs.

Fourteen exact duplicate top-level transfer/result archives are removed after
complete retained Windows copies and VM SHA256/stamp checks: 1,756,556,632 bytes
(1.64 GiB). The apply receipt measures 1,756,598,272 additional free bytes. The final
live check reports 8,036,728,832 bytes (7.48 GiB) free. Separate live
and Windows audits verify the deletion ledger, 207,967 protected
file hashes, 4,817 scientific tensor stamps
and 490 current V26/V27 bindings. Original unpacked checkpoints, sources, inputs,
splits, outputs and failed gates remain. All three scientific caches stay intact:
4,431 files/39,448,585,279 bytes. The existing CUDA runtime is available; L4 idle.
Current V27 execution/return archives and all Windows recovery copies stay intact.

V27 is reported stopped at50/800 updates for 0.221088578% structure improvement,
below the unchanged1% early requirement. Its complete 325,777,145-byte return,
checksum and export receipt are present on Windows, with reported SHA256
1bd4aec2bdc10cd6aca4455f24875b006bf4bca2d1a4f1cae8f4932400b386ab.
Cleanup confirms transfer hash only; independent result replay and visual review
are the next research actions. Export completion does not imply quality success.
Do not rerun V27 unchanged or relax its stop. Following the third spatial-path
failure, stop model modifications for the required architecture discussion.

All previous309 research bindings, deeper66/692/299/697/513 histories, three full
document bodies, earlier maintenance and app22 bindings remain preserved. The
user's restated goal takes precedence over historical manuscript/permission text.
Own-trained DGP remains primary; pretrained restoration models are comparisons.
All visible facial features, native unpaired evidence, separate paired synthetic
metrics, all seven automatic/assisted covering families, exact local app flow and
independent final review remain required. No new native/reserved-final pixels or
ethnicity/Zamboanga/hidden-identity claims enter maintenance. Goal incomplete.

[Cleanup evidence](<C:/xampp/htdocs/YEAR 4/Testing/outputs/cctv_dgp_vm_storage_cleanup_20261006_v2/closure_manifest.json>) ·
[Cleanup report](<C:/xampp/htdocs/YEAR 4/Testing/VM_STORAGE_CLEANUP_20261006_V2.md>)

Earlier document bodies below are preserved history. Their earlier pilot-launch
instructions and no-connection statements are historical; current authorization
and the retained quality failures above govern the next action.

**Exact goal restated by the user on6 October2026:**

Develop and verify our own trained DGP as the primary model for useful 256×256 restoration of degraded CCTV face crops for the thesis “Forensic Deep Generative Prior Face Reconstruction for Degraded CCTV Video in Zamboanga City,” in C:\xampp\htdocs\YEAR 4\Testing. Follow SYSTEM_WORKFLOW_AND_GOAL.md and PRACTICAL_OUTPUT_SCOPE.md; later user decisions take precedence over the outdated manuscript. Preserve visible facial structure, accepting some softness; request a clearer crop when usable structure is insufficient. Support one already cropped frontal or mildly turned face in the existing local application, preserving its design. No real Zamboanga CCTV samples exist yet: acquire and audit public native CCTV data before choosing new training, prioritize Asian capture sources where available, report sources separately and do not infer ethnicity or local performance. Proceed through five milestones: (1) freeze native CCTV development and separate labeled evaluation identities, provenance, terms, resolution, overlap limits and input-only review criteria; (2) compare basic resizing, retained Phase 3 DGP and declared pretrained restoration baselines on identical 256×256 inputs, separating raw outputs from display processing, structure preservation and insufficient-information cases; (3) fix demonstrated processing limitations and prepare justified finite VM pilots only when needed, preserving original checkpoints, splits and gate failures; (4) integrate DGP as the main local restorer with automatic restoration selection and a user override; (5) verify useful development outputs, independent final review, meaningful regressions and the full app flow with bundled inline Playwright, updating PROJECT_HANDOFF.md at milestones. Treat real CCTV without an aligned clean reference as unpaired evidence; keep paired synthetic PSNR/SSIM separate. Pretrained restoration models are comparison baselines. A separate completion component may estimate regions hidden by masks, sunglasses, strong lens glare, hands, obstructing hair, scarves or other objects. Show the automatic removal area for optional correction before generation; preserve clear glasses, non-obstructing hair and visible appearance, allowing a small documented margin. Request a less-covered image when too little face remains. Return one plausible estimate alongside the original and mask, with PNG and optional original/mask/result bundle downloads; do not claim exact hidden identity. Report automatic and assisted results separately. Inference and audits may run locally; all actual training must run on the existing NVIDIA L4 g2-standard-4 VM at ~/forensic-dgp. No configured SSH connection exists: prepare exact pasteable commands and verified transfer files, then independently audit returned results. Every pilot needs finite updates/epochs, timing and stop rules despite no fixed GPU-hour cap. Do not run immutable historical pilots automatically or repeat failed recipes unchanged. Complete only when the DGP-led local workflow works and reviewed outputs meet the agreed restoration and covering-family scope; archive acquisition or training completion alone is insufficient.

The explicit cleanup request permits current maintenance access; future training uses verified transfers and pasteable commands.


**Latest research milestone — 6 October 2026: corrected V26 gradient return audited; V27 direct own-feature pilot independently prepared.**

The 1,665,230-byte diagnostic return matches SHA256
`bb07f90ef10496a8b464d61d9e446826e4f8d824c12478f35765b3d3b160c201`.
Ten regular files, 240 original assets, 118 saved inputs, 250 frozen own-DGP feature
arrays and two 7×53,781 gradient matrices pass independent source/state/statistic/
batch/count/timing readback. All 752,934 saved gradient values are checked. The L4
measurement takes 19.089s, with 140 gradient queries and zero optimizer updates.
Its snapshot50 replays the old stopped V26 head; no new training occurs. Initial
identity value/all26 gradient tensors are exactly zero in all ten batches.
Preservation/improvement norm ratio at stopped50 is 0.3701499, cosine −0.7547647;
the improvement gradient remains nonzero. No preservation term or margin changes.

Independent pure-array path arithmetic verifies 269 sources/84 term-family rows
and 250 input-feature summaries. Direct RGB carries 98.4251552% of stopped50 landmark
squared-gradient magnitude. This depends on parameterization and does not prove
full optimizer trajectory causality. V26 remains a closed structure failure:
0.0335635587% delivered gain versus the unchanged1% early stop, with no convincing
whole-face gain in50 TRAIN cases. Do not rerun V26 or the completed diagnostic.

Within the user's selected own-DGP spatial direction, V27 adds five independent
normalized zero-initialized 1×1 RGB readouts from frozen own-DGP feature levels.
They shorten the feature path into the same bounded residual, retaining every
original decoder/RGB tensor, frozen DGP/recognizer, corrected seven-term loss,
seed,50 exposed TRAIN cases and original schedule. Add1743 parameters for55524 in
36 groups. All28 original VM state tensors must match V26 initialization exactly.
No fitted statistics, clean target, person/source label or pretrained restoration
output conditions inference. All visible facial features remain in scope together.

The 51,049-byte/seven-file thin packet reuses240 original VM assets with hash-bound
hardlinks and uploads no data or weights. Original files are never modified; require
3GiB free and preserve partial work. Installation60s; neural preflight300s; matched
identity proof180s; fit1500s; worker1800s; external2100s plus30s grace; export120s/
external150s plus30s grace; peak allocated VRAM20GiB. Finite800 updates/80 epochs.
Original update20 timing projection,1% early/10% final structure, all17 group pixel/
SSIM/identity bounds, source gains, brightness limit and final800 selection remain.
All36 initial identity gradients must be exactly zero; all five new skip weight
gradients must be positive before an optimizer. The original five projection
gradients must remain positive at update2. No automatic follow-on or unchanged retry.
If this third spatial-path capacity attempt fails, stop model modifications for
an architecture discussion before another attempt.

Full AST comparisons retain original head/trainer/guard/supervisor and prospective
safe importer/CPU/group/capacity/state/feature/cohort/timing audits except declared
new connections, counts, names, initial proofs and hardlink receipt. Python3.10
parsing,14 meaningful source/archive/receipt regressions and actual Windows transfer/
neural-install rejection pass. All247 bundle files remain unchanged by guard checks.
The shell is byte-equivalent to the original after restoring its worker filename.

Local inference verifies exact initial raw/PNG parity for50 cases and all28 shared
CPU tensors. Five predetermined, unfitted forward controls prove wiring without
clean targets, fitting or derivative estimates; independent NumPy/OpenCV arithmetic
checks their responses. Partial support preserves outside pixels; empty support
is rejected. There are61 valid CPU head forwards, zero DGP/recognizer forwards and
zero local derivatives/backwards/optimizer updates in the new interface audit.
No probe checkpoint/image is created. Actual L4 proof and trained capacity remain
pending. Five pasteable gcloud/SSH/tmux steps include separate PuTTY downloads.
The human performs every VM/cloud action; none is performed by this agent.

Previous66 bindings and deeper692/299/697/513 histories, seven complete document
bodies, app22 bindings and concurrent completed VM maintenance are preserved.
No candidate is promoted and the existing app design/checkpoint is unchanged.
Native CCTV stays unpaired; paired photographic TRAIN metrics remain separate.
Reserved-final identities remain unviewed. No new native/covering pixels, ethnicity
or Zamboanga performance claim enters this milestone. Previously useful native
inputs remain usable despite failed model corrections. Useful native development
output, candidate app parity/full flow, input-only insufficient-information handling,
all seven automatic/assisted covering families and independent final review remain
required. **Goal active/incomplete.**

[Verified diagnostic](<C:/xampp/htdocs/YEAR 4/Testing/CCTV_DGP_V26_GRADIENT_DIAGNOSTIC_V1_RESULTS.md>) ·
[V27 frozen design](<C:/xampp/htdocs/YEAR 4/Testing/CCTV_DGP_FEATURE_SKIPS_V27_PLAN.md>) ·
[Five manual VM steps](<C:/xampp/htdocs/YEAR 4/Testing/CCTV_DGP_FEATURE_SKIPS_V27_VM.md>) ·
[Milestone](<C:/xampp/htdocs/YEAR 4/Testing/outputs/dgp_v26_gradient_return_and_feature_skips_v27_milestone/milestone.json>)

Previous bodies below are preserved history. Their diagnostic-pending statements
and V22–V26/diagnostic commands are historical. Launch only the new V27 protocol
once under its current manual runbook; all earlier failed/completed runs stay closed.


**Latest research milestone — 6 October 2026: corrected V26 scalar objective audited; finite zero-update L4 gradient measurement verified.**

The unchanged corrected V26 objective decreases from 1.2999999568 to 1.2997388024
between saved states 0/50. Degraded structure/pixel terms improve slightly; clear
controls contribute a small preservation cost. Two degraded cases incur SSIM
costs and one incurs an identity cost at stopped 50. Scalar penalty sizes do not
establish their active gradient strength or the optimizer trajectory's cause.
Do not remove preservation terms, relax gates or invent a new recipe from these
scalar observations. The original V26 failure remains closed: 0.0335635587% versus
the unchanged 1% early requirement, with no convincing whole-face gain in 50 cases.

The 54.273-second local diagnostic uses 30 frozen-recognizer CPU forwards and 20
saved prediction replays, zero head/DGP forwards and zero local derivatives or
optimization. The actual corrected loss/metric function ASTs and five-case batch
contexts are retained. Independent arithmetic checks 280 source bindings, 30 vector
arrays, 100 case states and 34 group states. All seven term assemblies agree exactly.
This is saved-scalar/source evidence, not a GPU derivative or SSIM-field rerun.

The next packet measures actual corrected V26 component gradients at the two
saved heads: 20 head batches, 140 gradient queries, 70 recognizer forwards, zero
DGP forwards and zero optimizer updates/epochs. Its 16,047-byte transfer has four
regular files and uploads no data or weights. Original 240 assets, 118 saved-input
bindings, the prior identity proof and 250 frozen feature arrays are checked.
Every initial batch must retain exactly zero identity value/all 26 gradients.
Worker 420s; supervisor 480s plus 30s grace; export 30s internally/60s externally
plus 10s grace. Require an idle existing L4 and 1 GiB free disk; preserve any stop.
No new checkpoint, unchanged training retry or automatic follow-on is allowed.

Eleven source/archive/matrix tamper checks, Python 3.10 parsing, Windows transfer/
neural-gradient rejection and Bash syntax pass. Full diagnostic-worker AST and
supervisor behavior retain the original measurement except declared objective,
proof, counter and name changes. Prospective safe import/full matrix audit preserve
original guards and require exact-zero corrected identity proof. Actual L4
measurement and independent returned-matrix audit remain pending; no VM/cloud
action occurs. Manual gcloud transfers and tmux launch are in the new five-step
runbook, with three separate PuTTY download calls. No new training recipe exists.

Original V26 and earlier failures/checkpoints/splits, previous 692 bindings and
deeper 299/697/513 evidence, four full document bodies, app 22 bindings and concurrent
completed VM maintenance remain preserved. The user-selected own-DGP spatial path
and all visible facial features together remain required. No app candidate is
promoted. Native CCTV stays unpaired; paired photographic TRAIN evidence stays
separate. No native/reserved-final/new covering pixels or ethnicity/Zamboanga
performance claims enter this milestone. Previously useful inputs remain usable
despite model softness. Useful native output, candidate app parity/full flow,
input-only insufficient-information handling, all seven automatic/assisted
covering families and independent final review remain required. Goal active/incomplete.

[Corrected objective evidence](<C:/xampp/htdocs/YEAR 4/Testing/CCTV_DGP_V26_CORRECTED_OBJECTIVE_REVIEW.md>) ·
[Finite measurement design](<C:/xampp/htdocs/YEAR 4/Testing/CCTV_DGP_V26_GRADIENT_DIAGNOSTIC_V1_PLAN.md>) ·
[Five manual VM steps](<C:/xampp/htdocs/YEAR 4/Testing/CCTV_DGP_V26_GRADIENT_DIAGNOSTIC_V1_VM.md>) ·
[Milestone](<C:/xampp/htdocs/YEAR 4/Testing/outputs/dgp_v26_corrected_objective_and_gradient_diagnostic_milestone/milestone.json>)

Previous bodies below are preserved history. V26 is a closed training failure;
the new commands measure saved states only. Historical training/diagnostic commands
are not instructions to repeat immutable failed recipes or earlier measurements.


**Latest research milestone — 6 October 2026: V26 return audited; identity-reference correction proved, structure failure retained.**

The downloaded 325,764,761-byte V26 archive matches SHA256
`9ea6afad52b70a27681630b8b09f96dfe560f6756664b627037198fe9e47ae53`.
Safe import verifies 631 regular files. Independent CPU inference and saved-source/
arithmetic checks pass: 240 assets, two snapshots at 0/50, 100 raw/PNG pairs and
metric rows, 50 original own-DGP forwards and 250 feature arrays. All original
numerical and quality bounds remain unchanged. Twenty-five tamper regressions pass.

Actual L4 preflight confirms exact zero matched identity penalty and all 26 matched
parameter-gradient assertions on all 50 cases, before optimization. The legacy
discrepancy is reproduced. The reference-processing fix therefore works, but it
does not resolve the structure failure: **0.0335635587% delivered degraded gain
against the unchanged 1% requirement at update 50**. Training stops after 50 updates/
51 backwards; final 800 never runs. Export complete:true packages the retained
failure and does not imply training success. Do not repeat V26 unchanged or relax
the stop. Original V22–V25 failures and the separate processing evidence stay intact.

All ten original-size sheets/50 paired photographic TRAIN cases are reviewed;
200 exact cells are independently checked. Eyes, nose, mouth, face outline and
overall visible appearance show no convincing V26 gain. All five own-feature
projection gradients are active and all 26 learned tensors change. Corrections
reach output, but the median degraded raw change is only 0.1348669090 byte level.
Raw degraded structure gain is 0.0151795995%, below even the delivered PNG gain;
quantization alone does not explain the failure. Worker 29.077s, fit 9.197s and
supervisor 35.001s pass timing bounds. Torch allocated peak VRAM is 1,806,989,312
bytes. There is no time-cap or CUDA failure behind the reported structure stop.

The corrected identity calculation is insufficient to establish useful restoration.
Review the corrected objective and spatial response at saved states before choosing
another training recipe. The complete optimizer-trajectory cause is not proved.
No V27 packet, unchanged retry, local gradients/optimization or assistant VM/cloud
action occurs. The user-selected own-DGP spatial/feature direction remains active.

The previous 299 bindings, deeper 697/513 bindings, twelve complete document bodies,
app 22 bindings and concurrent completed VM maintenance are preserved. No candidate
is promoted. Native CCTV stays unpaired; exposed paired TRAIN metrics are separate.
No native/reserved-final/new covering pixels or ethnicity/Zamboanga performance
claim enters this milestone. Previously useful native inputs remain usable despite
model softness. Useful native development output, input-only insufficient-information
handling, candidate app parity/full flow, all seven automatic/assisted covering
families and independent final review remain required. Goal active/incomplete.

[V26 verified result](<C:/xampp/htdocs/YEAR 4/Testing/CCTV_DGP_BATCHMATCHED_IDENTITY_V26_RESULTS.md>) ·
[Milestone](<C:/xampp/htdocs/YEAR 4/Testing/outputs/dgp_batchmatched_identity_v26_audit_milestone/milestone.json>)

Previous bodies below are preserved history. Their V26-pending statements and manual
V22–V26/diagnostic launch commands are historical. V26 is closed as a failure;
do not use those commands to repeat immutable historical pilots.


**Latest research milestone — 6 October 2026: V25 gradient return audited; V26 reference-processing correction verified for manual L4 execution.**

The 1,695,519-byte fixed-state diagnostic return is verified and safely imported:
ten regular files, two 7×53,781 gradient matrices, 14 component rows and 26 parameter
groups per state. Independent arithmetic/source/state/count/timing checks pass.
The L4 measurement takes 18.535 seconds with 140 gradient queries and zero optimizer
updates. This readback checks saved derivatives; it does not independently recompute
L4 derivatives or establish the cause of the entire optimizer trajectory.

All 50 initial raw outputs equal their cached own-DGP baseline exactly. The legacy
identity reference nevertheless incurs a 1.3932586e-7 mean penalty and a gradient
1.45139 times the landmark-detail gradient. Baseline scores are computed one case
at a time without gradient tracking, while predictions use five-case gradient
calls. The zero-margin penalty responds to that numerical difference. A demonstrated
reference-processing correction is justified; the original V25 structure failure
(0.0282235213% against 1% at update 50) remains a failure. Real preservation penalties
are retained, and useful restoration is still unproven.

V26 changes only the identity reference calculation: baseline and prediction share
one frozen recognizer call, with the baseline embedding detached. The original
zero margin, coefficient 5, other six terms, own-DGP spatial head, 50 exposed TRAIN
cases, schedule, optimizer, quality gates and finite limits remain unchanged.
The L4 preflight must reproduce the legacy discrepancy and prove exactly zero
matched penalty and all 26 matched parameter gradients on every initial baseline
batch before creating an optimizer. No epsilon or preservation waiver is added.

The new transfer is 49,537 bytes, seven regular files and five new source assets.
Its guarded 60-second installer verifies and copies 235 original VM assets into a
fresh directory, preserving failed V25. No original data or weights are reuploaded.
The actual proof has a 180-second limit inside the original 300-second preflight.
The pilot retains at most 800 updates/80 epochs, the 1% early structure gate at 50
and the 10% final structure gate at 800, all 17 preservation groups and the brightness
limit. Fit 1,500 seconds; worker 1,800; supervisor 2,100 plus 30 seconds grace.
Any proof, timing or quality stop is retained; no unchanged failed recipe is rerun.

Eleven saved-matrix/archive/reference contracts, Python 3.10 parsing, actual Windows
rejection guards and Bash syntax checks pass. Tests use an arithmetic mock recognizer;
no local learned-model forwards, gradients, backwards or optimization occur. No
assistant VM/cloud action occurs. Actual V26 L4 proof/training, independent return
audit and whole-face review remain pending. Human execution uses the five manual
upload/install/tmux/run/download steps; PuTTY downloads use three separate calls.

Prior 697 evidence bindings and ten full document bodies are preserved, along with
the app's 22 bindings and concurrent completed VM maintenance. No app promotion.
The user-selected own-DGP spatial/feature direction and all visible facial features
together remain required; previous useful native inputs stay usable despite model
softness. Native CCTV remains unpaired, and paired photographic TRAIN metrics remain
separate. No native/reserved-final/new covering pixels, ethnicity or Zamboanga
performance claim is introduced. Useful native outputs, candidate app parity/full
flow, insufficient-information handling, all seven automatic/assisted covering
families and independent final review remain required. Goal active/incomplete.

[Audited diagnostic](<C:/xampp/htdocs/YEAR 4/Testing/CCTV_DGP_V25_GRADIENT_DIAGNOSTIC_V1_RESULTS.md>) ·
[Finite V26 design](<C:/xampp/htdocs/YEAR 4/Testing/CCTV_DGP_BATCHMATCHED_IDENTITY_V26_PLAN.md>) ·
[Current five manual steps](<C:/xampp/htdocs/YEAR 4/Testing/CCTV_DGP_BATCHMATCHED_IDENTITY_V26_VM.md>) ·
[Milestone](<C:/xampp/htdocs/YEAR 4/Testing/outputs/dgp_v25_gradient_audit_and_batchmatched_identity_v26_milestone/milestone.json>)

Previous bodies below are preserved history. Diagnostic-pending statements and old
V22–V25/diagnostic launch commands are superseded by the completed diagnostic audit
and the guarded fresh V26 correction. Do not repeat immutable historical pilots.


**Latest research milestone — 6 October 2026: V25 audited failure; fixed-state gradient diagnostic verified, manual L4 measurement pending.**

The325,762,305-byte V25 return is verified and safely imported; independent replay
checks628 returned files/235 assets,100 raw/PNG pairs, both0/50 snapshots,50 original
DGP CPU forwards and250 frozen feature arrays. All original replay/quality bounds
stay unchanged. The trainer correctly stops at50 updates/51 backwards:
**0.0282235213% delivered degraded structure gain, below the required1%.** Final800
never runs. complete:true packages the retained failure; it does not accept a model.

All ten original-cell sheets/50 paired photographic TRAINING cases are reviewed;
200 exact cells are independently checked. No convincing whole-face gain is visible.
All26 spatial-head tensors change, all five projection gradients are active and
the stopped correction now reaches the output without the old final filter
attenuation. The typical degraded correction is still0.12994 of one byte level.
The saved corrected loss decreases through degraded cases, with a clear-preservation
cost; the V23 clear-reward mismatch does not explain this small saved improvement.
Scalar evidence does not establish GPU gradient competition or optimizer causality.

The next transfer is a **zero-update fixed-state gradient diagnostic**, not a new
training recipe or an unchanged V25 retry. It uses the two saved heads, same50
TRAIN cases, original seven objective terms and250 frozen own-DGP feature arrays.
The15,259-byte packet is independently checked:20 head batches/140 gradient calls/
120 recognizer forwards, zero DGP forwards/optimizer updates. Worker420s, external
480s plus30s grace; export30s internally/60s externally plus10s grace. Require1GiB
free disk and an idle L4; preserve any stop. Actual L4 gradients and independent
returned-matrix audit remain pending before choosing another training recipe.

Thirteen V25 auditor regressions and nine new diagnostic packet/source/matrix
guards pass. Python3.10/actual Windows rejection/Bash syntax/frozen mask and affine
geometry checks pass. Preparation failure evidence is retained. No local gradients,
backwards or optimization and no assistant VM/cloud action occur. The user-selected
own-DGP spatial/feature direction and all visible facial features together remain
in scope. Exact manual upload/install/tmux/launch/download steps are in the new
diagnostic runbook; PuTTY downloads remain three separate remote-source calls.

Original checkpoints/splits/failed gates, previous513 milestone bindings, app22
bindings and concurrent completed VM maintenance are preserved. The app and its
historical34 regressions/bundled inline Playwright checks remain unchanged. No
candidate is promoted. Native CCTV stays unpaired; paired TRAINING metrics remain
separate. No native/reserved-final/new covering pixels, ethnicity or Zamboanga
performance claims are introduced. The previously useful CCTV crop stays usable
despite model softness. Useful native output, candidate app parity/full flow,
insufficient-information handling, all seven automatic/assisted covering families
and independent final review remain required. Goal active/incomplete.

[V25 audited result](<C:/xampp/htdocs/YEAR 4/Testing/CCTV_DGP_SPATIAL_FEATURES_V25_RESULTS.md>) ·
[Finite diagnostic design](<C:/xampp/htdocs/YEAR 4/Testing/CCTV_DGP_V25_GRADIENT_DIAGNOSTIC_V1_PLAN.md>) ·
[Manual diagnostic commands](<C:/xampp/htdocs/YEAR 4/Testing/CCTV_DGP_V25_GRADIENT_DIAGNOSTIC_V1_VM.md>) ·
[Milestone](<C:/xampp/htdocs/YEAR 4/Testing/outputs/dgp_spatial_features_v25_audit_and_gradient_diagnostic_milestone/milestone.json>)

Previous bodies below are preserved history. Their pending-return/manual-V25-launch
statements are superseded by the audited V25 failure and diagnostic-only next step.
Old V22–V25 training commands are historical; do not repeat those failed recipes.


**Latest research milestone — 6 October 2026: selected own-DGP spatial path; V25 transfer verified, manual L4 run pending.**

The downloaded V24 failure is independently audited and all50 exposed training
cases reviewed. Its0.0033227502% delivered structure gain misses the unchanged1%
early stop. Preserve/close V22–V24; no unchanged rerun or failed-gate waiver.
After the architecture discussion, the user selects Route A: revise our DGP's
spatial/feature path. All visible facial features remain required together.

V25 introduces a53,781-parameter decoder using our frozen DGP's five FPN maps
and the full-resolution camera image. Original DGP weights/normalization remain
frozen. It removes the failed final output Gaussian high-pass, retaining observed
RGB mean removal/support masking. This is a new own-model spatial-path capacity
hypothesis, not proven learning, a pretrained restorer replacement or app adoption.

Packet218,140,185 bytes,235 assets/237 regular archive members; SHA256
`7652fa82a95d218de18c31d943114124bfaf9299e6492e8c79da710b59831034`;
protocol`ed43355b1fa0bd768d78e4e2b9ed3bd32cc046fe6221629605899a6a9b2e3175`.
All221 original assets, same50cases/10references/800updates/80epochs, optimizer,
schedule, corrected V24 objective, quality guards and finite timing stops stay
exact. All50 fresh CUDA baseline comparisons, four CPU feature comparisons,
250 frozen feature arrays and actual update2 gradients are prospectively required.
Intermediate CPU/CUDA numerical bounds do not relax quality or qualify the app.

Local interface verification uses4 frozen DGP/5 head forwards, no backward or
optimization: initial fresh CPU baseline is exact and disposable smooth-camera/
feature links reach output. A separate saved-array audit passes. Twelve feature
failure checks and ten safe-return transfer tests pass. Independent archive/
source/Python3.10/Bash/actual Windows transfer and training-host guards pass.
Preparation failures are retained. Real L4 features/gradients/timing/learning and
independent returned-output audit/whole-face review remain pending. complete:true
export packages evidence; it does not certify training success or usefulness.

Manual upload/install/tmux/launch/download commands are frozen in the V25 runbook;
Windows PuTTY downloads use three separate remote-source calls. No assistant VM/
cloud action or local training occurs. Prior434 V24 closure bindings and app22
bindings, original checkpoints/splits/failures and the concurrent completed VM
storage-maintenance entry remain preserved. Historical34 app regressions/bundled
inline Playwright checks are retained; the unchanged app requires no new run.

Native remains unpaired; these paired photograph TRAINING metrics stay separate.
No validation/reserved-final/native/new covering pixels enter V25; no ethnicity
or Zamboanga performance claim. Useful native development output and canonical
local-app inference, insufficient-information handling, all seven automatic/
assisted covering families and independent final review remain required. The
previously useful CCTV crop stays usable despite softness. Goal active/incomplete.

[V25 finite design](<C:/xampp/htdocs/YEAR 4/Testing/CCTV_DGP_SPATIAL_FEATURES_V25_PLAN.md>) ·
[V25 manual tmux commands](<C:/xampp/htdocs/YEAR 4/Testing/CCTV_DGP_SPATIAL_FEATURES_V25_VM.md>) ·
[Milestone](<C:/xampp/htdocs/YEAR 4/Testing/outputs/dgp_spatial_features_v25_preparation_milestone/milestone.json>)

Previous bodies below are preserved history. Their pending-return/no-new-packet
statements are superseded by the audited V24 closure and selected V25 preparation;
old V22–V24 launch commands remain historical, not instructions to repeat them.


**Latest research milestone — 6 October 2026: V24 failure audited/reviewed; architecture discussion before another pilot.**

All three downloaded return files are verified:76,492,526-byte archive, SHA256
`1d758362366f3e1f17ddf8239b0f41940b6b7c6ebf349a86ed76e0abdf63dd15`.
Safe import371 members and independent43.42s replay pass all221 frozen assets,
100 raw/PNG pairs,110 recognizer calls, original cohort scalars and execution/
gradient/timing/state receipts. No tolerance or scientific gate changes.
Training correctly stops at update50:0.0033227502% delivered degraded structure
gain against the original1%; raw gain is0.0018034721%. Final800 never executes.
Export complete:true packages the failure; it is not a training success receipt.

All ten original-cell sheets/50 exposed TRAINING cases are visually reviewed;
all200 cells independently verified. No convincing whole-face gain over frozen
own DGP. Typical degraded correction is0.011075 of one byte level. The known-
source50-case stopped-layer trace and fixed saved-objective decomposition are
verified without local backwards/optimization. The V24 clear-reward correction
works, but degraded learning remains far below useful structure. Keep the stop.

Close the V22–V24 small filtered-head sequence. The user's “all are important”
requires eyes/nose/mouth/outline/visible appearance together. A concrete new
architecture review recommends revising our DGP spatial/feature reconstruction
path; the previously permitted declared face-prior extension is an alternative.
The user selects our DGP spatial/feature path. No fourth recipe, executable training
packet or model fix is prepared. Existing training commands for these stopped
recipes are historical; do not rerun or waive them. No further V24 download is
needed. Future training/gradient preflight remains manual on the existing L4.

The original documents, concurrent completed-storage-maintenance entry, original
checkpoints/splits/failures, prior664 research bindings and app22 bindings remain
preserved/verified. The original18 auditor regressions and historical34 app/
bundled inline Playwright checks are retained; no changed app requires a rerun.
Native evidence remains unpaired; paired photographic TRAINING metrics stay
separate. No native/reserved/new covering pixels, local training, assistant VM/
cloud action, app promotion, ethnicity or Zamboanga-performance claim occurs.
The previously useful native input remains usable despite model softness.
Useful native output, canonical app parity, all seven automatic/assisted covering
families and independent final review remain required. Goal active/incomplete.

[V24 audited result](<C:/xampp/htdocs/YEAR 4/Testing/CCTV_DGP_DEGRADED_DETAIL_V24_RESULTS.md>) ·
[Architecture discussion](<C:/xampp/htdocs/YEAR 4/Testing/CCTV_DGP_POST_V24_ARCHITECTURE_REVIEW.md>) ·
[Milestone](<C:/xampp/htdocs/YEAR 4/Testing/outputs/dgp_degraded_detail_v24_audit_and_architecture_milestone/milestone.json>)

Previous bodies below are preserved history, including superseded pending-return
and pending-training notices. They are not current instructions to rerun V24.


**Latest research status — 6 October 2026: V24 user-reported early structure failure; return audit pending.**

The pasted L4 log reports successful preflight, exact four-case original-DGP
CUDA parity and 50 initial cached-DGP outputs. At update50, delivered degraded
feature error changes from 0.0019965976532523044 to 0.001996531311299121:
0.0033227502% improvement, below the unchanged 1% early requirement. Trainer
exit1 retains that stop. Export exit0 and complete:true confirm failure packaging;
run_results_present:false/failure_present:true. No final800 capacity test executes.

Reported return: 76,492,526 bytes, SHA256
`1d758362366f3e1f17ddf8239b0f41940b6b7c6ebf349a86ed76e0abdf63dd15`.
The three return files are not yet present locally. Actual source/data/gradient/
timing/state/cohort/metric audit and every-case image review remain pending.
Do not infer visible usefulness from the tiny metric improvement or console log.
Use step5 in the V24 runbook: three separate Windows gcloud downloads.
Steps1–4 are historical for this stopped recipe; do not rerun them or relax gates.

V22, V23 and V24 have missed the same early structure requirement. The assumption
that the tested small high-frequency correction on frozen own-DGP outputs can
deliver enough facial structure has not held in these finite pilots. Stop this
head-recipe sequence, audit returned V24 evidence, and review architecture/data/
loss before another recipe. No fourth pilot or training fix is prepared.
The user confirms “all are important”: eyes, nose, mouth, face outline and overall
visible appearance remain in scope together. No region-only acceptance or focus.

The V24 return importer/auditor is prepared with 18 passing tamper/schema/gate
regressions. The unchanged full old-auditor logic and replay tolerances are
checked by AST comparison; new cohort checks independently validate 50 baseline
rows, ten clear controls, forty degraded rows, policy and exact float32 means.
A fixed CPU contract uses100 high-pass calls, no head/DGP/recognizer prediction,
backward or optimization. These are checker tests, not real V24 audit evidence.
An initial auditor-only stale-bundle path failure and its correction are preserved;
the VM runtime, protocol, original checkpoints and all quality gates are unchanged.

Original documents and the concurrent completed-storage-maintenance entry are
preserved before this update. The DGP-led app/design remains unchanged; no new
local training, assistant VM/cloud action, app promotion or browser rerun. Full
native usefulness, canonical app parity, all seven automatic/assisted covering
families and independent final review remain required. Native remains unpaired;
paired TRAINING metrics are separate. No ethnicity or Zamboanga performance claim.
The previously useful CCTV input remains usable despite model softness. Goal active.

[V24 downloads](<C:/xampp/htdocs/YEAR 4/Testing/CCTV_DGP_DEGRADED_DETAIL_V24_VM.md>) ·
[V24 audit preparation](<C:/xampp/htdocs/YEAR 4/Testing/outputs/cctv_dgp_degraded_detail_v24_return_audit_preparation/plan.json>)

Previous entries below are preserved history; V24 pending-training statements
are superseded by this reported stopped run, not by an independent quality pass.


**Current milestone — 6 October 2026: V23 failed run independently audited; V24 objective experiment prepared.**

The downloaded V23 archive matches 76,481,832 bytes/SHA256
`b71639cabd17476d5989dfd4582d47de9b747b8bf2a611232e5f62cc67bf9fd9`.
All 370 returned files, 209 original assets, 100 raw/PNG pairs/metrics, two complete
50-case snapshots, stopped state and source-bound timing/gradient/count receipts
pass independent audit. The L4 completed 50 updates/51 backwards, then correctly
failed its unchanged 1% early gate: delivered degraded feature error worsens
0.00223306%. Nonzero direct gradients and changed tensors are verified. All ten
original-resolution sheets/50 cases show no convincing visible structure gain.
Seven original preservation checks diagnostically fail at 50; the unexecuted
800-update final gate is not evaluated. Close V23 without rerun or app adoption.

The exact old objective is decomposed on both saved raw states, independently
arithmetic-checked, and decreases overall through clear-case gains while the
degraded cohort worsens. Clear cases contribute 101.4223% of its net reduction.
This is a demonstrated loss/goal mismatch, not proof of GPU gradient causality.
V24 changes that objective prospectively: degraded-cohort HF normalizers, no
clear target reward and clear baseline preservation controls. Same 4,613-parameter
head, 50 exposed training cases, 800 updates/80 epochs, optimizer, schedule,
appearance/brightness gates, timing budgets and early stops. No failed gate is
waived; original V22/V23 sources, protocols, archives and stopped states remain.

V24 transfer: 218,041,986 bytes, 221 assets/223 regular members. Independent
archive/source/Python3.10/Bash/Windows guard and cohort contracts pass, with eight
loss invariants. All 50 fixed baseline errors match the prior decomposition;
100 fixed CPU filters, no new head/DGP/recognizer prediction in preparation.
Actual CUDA gradients, learning and capacity are pending. A comparator-only
field exclusion correction and sandbox Bash initialization failure are preserved;
neither changes the pilot or quality conditions. If this distinct objective trial
fails, stop blind head-recipe changes and revisit architecture/data/loss.

Prior 644 and older 560 milestone bindings, all 22 current app bindings and prior
document bodies are verified/preserved. The DGP-led Auto/On/Off application,
design, checkpoints and splits are unchanged; historical 34 regressions and
bundled inline Playwright are not rerun for this evidence/transfer-only milestone.
Manual existing L4/g2-standard-4 under ~/forensic-dgp only: verified transfers and
exact pasteable gcloud/tmux/separate-download commands. No assistant VM/cloud
action or local backward/optimizer update. Earlier maintenance entries are
historical; the user has subsequently returned an L4 run, with no new VM polling.

The user's useful-case feedback still requires clearer visible structure; its
usable input is not relabeled insufficient. No native/reserved/covering/COFW-test
pixels enter V24. Native CCTV remains unpaired; these paired TRAINING metrics are
separate. No ethnicity or local Zamboanga inference. Useful native outputs,
canonical app parity, all seven automatic/assisted covering families and independent
final review remain open. Preserve clear glasses, non-obstructing hair and visible
appearance; show the removal area for optional correction, request a less-covered
crop when evidence is insufficient, and return one plausible estimate with the
original/mask and PNG/optional bundle. No exact hidden identity claim. Goal active.

[V23 audited result](<C:/xampp/htdocs/YEAR 4/Testing/CCTV_DGP_DETAIL_SKIP_V23_RESULTS.md>) ·
[V24 plan](<C:/xampp/htdocs/YEAR 4/Testing/CCTV_DGP_DEGRADED_DETAIL_V24_PLAN.md>) ·
[V24 commands](<C:/xampp/htdocs/YEAR 4/Testing/CCTV_DGP_DEGRADED_DETAIL_V24_VM.md>)

Previous entries below are preserved history and are superseded by this milestone.


**Latest 6 October 2026 — V23 user-reported early structure stop; download/audit pending:**
The pasted L4 log reports successful source/data/CUDA preflight, exact four-case
original-DGP parity and50 initial cached-DGP checks. Training reaches update50.
Delivered-PNG degraded feature error increases from0.0019965976532523044 to
0.0019966422385438174, approximately0.00223306% worse, instead of the required1%
improvement. The unchanged early condition stops the worker (trainer exit1).
Export exits0 and packages failure evidence; its complete flag is not model success.

Reported return:76,481,832 bytes, SHA256
`b71639cabd17476d5989dfd4582d47de9b747b8bf2a611232e5f62cc67bf9fd9`.
The return is not yet present locally. Independently verify hashes, sources,
timing/gradient/state/metric receipts and all images before diagnosing the
underlying model failure. Do not repeat the unchanged V23 launch, relax its gate,
or prepare another recipe from this console log alone. Final800 capacity gate
was not executed; no app acceptance or visible-quality verdict is claimed.

Proceed to step5 in the V23 runbook: three separate Windows gcloud downloads.
Original transfer/protocol/runtime and historical failure gates are unchanged.
Prior document bytes are retained under
`outputs/cctv_dgp_detail_skip_v23_reported_stop_v1/before_docs/`.
No assistant VM action/local training or app change occurs. Goal remains active;
full native restoration, covering-family scope and independent review stay open.
Runbook: [V23](<C:/xampp/htdocs/YEAR 4/Testing/CCTV_DGP_DETAIL_SKIP_V23_VM.md>).


**Latest 6 October 2026 — V22 R1 stopped run independently closed; different V23 transfer prepared:**
The returned archive/source hashes,196 assets/366 returned members,100 raw/PNG
pairs/metrics and both50-case snapshots pass the independent local audit. The
original L4 run stops after50 updates/51 backwards:0.0000322991% delivered feature
gain versus its unchanged1% early gate. Ten original-resolution sheets/all50 cases
show no visible structure improvement. The too-small correction through the deep
route/detail projection is traced and independently saved-evidence checked.
Nine original preservation checks diagnostically fail at update50; the unexecuted
800-update final gate is not evaluated. V22 is closed without adoption or rerun.
Failed auditor fixtures/source and explicit initializer-only compatibility
correction are preserved; replay/quality tolerances remain unchanged.

V23 is a new4613parameter full-resolution detail bypass plus shallow branch.
Same ten exposed training photographs/50 cases, objective,800 updates/80 epochs,
batch5, appearance/brightness gates, budgets and early stops; no new data or
CodeFormer RGB/features.209 assets/211 members, Python3.10/Bash/Windows guard,
50 exact initial raw/PNG cases and fixed sensitivity/padding contracts pass.
These verify implementation/transfer, not gradients, learned structure or quality.
Actual one-batch gradient check and finite fitting remain manual on the existing
NVIDIA L4/g2-standard-4 under ~/forensic-dgp. The packet exports fuller timing,
step/VRAM/component counts and supervisor receipts; failed partial outputs remain.
No assistant VM/cloud action or local training occurs. User execution preference
remains verified transfer files and exact pasteable upload/tmux/download commands.

Prior560 milestone and22 app bindings pass; original workflow/runbook bytes are
preserved. The DGP-led app/design, original checkpoints, splits and historical
failures remain unchanged; historical app regressions/Playwright are not rerun.
The shown native crop remains useful and input-usable with clearer structure
required. Full native usefulness, canonical app parity, all seven automatic/
assisted covering families and independent final review remain unqualified.
Native remains unpaired; paired exposed-training metrics are separate. No real
Zamboanga CCTV samples, inferred ethnicity or local-performance claims. Final
reserves/COFW test pixels remain unopened, reviewers/cohort unassigned. Goal active.
Commands: [V23](<C:/xampp/htdocs/YEAR 4/Testing/CCTV_DGP_DETAIL_SKIP_V23_VM.md>).
Result: [V22 R1](<C:/xampp/htdocs/YEAR 4/Testing/CCTV_DGP_DETAIL_PRIOR_V22_R1_RESULTS.md>).
Evidence: `outputs/dgp_detail_prior_v22_r1_audit_and_skip_v23_milestone/`.


**Latest 5 October 2026 — structure feedback answered; finite V22 VM packet prepared:**
The user finds the shown native own-DGP output useful but needs clearer facial
structure. The already usable crop remains usable; positive feedback does not
qualify all outputs or waive preserved failures. The pending review is answered.

The V21 saved-array trace confirms raw input-aligned eye-detail attenuation on
all24 exposed native development cases (median own-DGP slope0.704009), with exact
raw/PNG delivery. Native is unpaired; captured high frequencies may be noise.
A fixed detail control fails15 paired TRAINING preservation checks and all ten
original-resolution sheets show no convincing degraded structural gain. A more
permissive target-informed restricted band-gate bound yields at most5.44026%
paired-MSE gain versus its frozen10% condition. Both routes close; original
results, failed checks and separate two-count metadata correction remain intact.

Prepare V22 as the justified finite own-model training capacity step: a45443-
parameter new detail head conditioned only on input and own-DGP pixels, high-pass/
zero-mean correction before clamp, no CodeFormer generator features or RGB.
Same ten exposed training photographs/50 cases,800 updates/80 epochs, batch5;
no native, reserved or covering pixels. Prospectively require >=10% feature-HF
MSE reduction, existing source/profile pixel/SSIM/ArcFace safeguards, and <=20%
brightness-only gain. Keep all earlier historical gate outcomes unchanged.

Exact initial/finite-kernel/padding contracts, source and schedule audits,
Windows training rejection, Bash syntax,195 assets and197 archive members pass.
These are preparation checks; the actual CUDA preflight, learning and output
review remain pending. User runs only existing L4/g2-standard-4 under
~/forensic-dgp. Preflight300s/fit1500s/worker1800s/external2100s/export120s,
with update20 timing projection and update50 insufficient-structure stop.
Manual verified transfers and pasteable commands only; no assistant VM action.

Active execution revision R1 replaces the newer checksum API with streaming
SHA256 for Python3.10 in the VM trace; original unexecuted V22 packet remains
archived. Exact head/core functions, data, schedule, objective and gates are
unchanged. R1 verifies196 assets, Python3.10 grammar and198 safe archive members;
Bash syntax and Windows training rejection pass. Use the R1 runbook commands.

After return, independently audit hashes, source, budgets, saved metrics and
local head/recognizer replay; review every training case. Capacity success is
not app qualification. Freeze any broader development/native stage separately;
reserved final review remains unexecuted. App source/design, checkpoints,
splits, old failed gates and all covering-family requirements stay intact.
Previous52 milestone and22 app bindings pass; old documents are preserved.
Historical tests/Playwright remain verified history, not reruns. Goal active.
Runbook: [V22](<C:/xampp/htdocs/YEAR 4/Testing/CCTV_DGP_DETAIL_PRIOR_V22_VM.md>).
Report: [V21](<C:/xampp/htdocs/YEAR 4/Testing/CCTV_DGP_STRUCTURE_TRACE_V21.md>).
Evidence: `outputs/dgp_structure_and_detail_pilot_milestone_v21_v22/`.

**Earlier 5 October 2026 — final-review preparation audited; execution not ready:**
Blank independent-review forms cover 58 reserved native cases/45 namespaced
release person labels, 116 input rows, 580 blinded output rows, 54 covering
slots and 34 five-milestone requirements. A separate 0.03s preparation audit
checks source metadata, maps, blank verdicts and notices. Original V1 is restored
exactly; the expanded blinding clarification remains separate R1. Reviewers are
unassigned; final candidate/cohort bindings are empty and native reserve pixels
remain unopened.

The existing verified COFW archive supplies 507 publisher-test metadata rows and
64 deterministic input candidates, 16 per sparse-flag stratum. Preparation and
independent audit take 1.46s/1.26s; only three metadata arrays and image-object
shapes/types/names are read. No test pixels, models, training admission or final
cohort selection. Sandbox failure and HTTP 403 refresh are preserved; terms rely
on the exact acquired CC BY 4.0 publisher snapshot. The registry's 42 COFW rows
are author-training-only; person overlap and complete historical exposure remain
unknown. Sparse flags are not covering-family labels or masks. This is separate
occlusion-photo evidence, not CCTV or aligned clean hidden-face references.

Useful own-DGP native and full covering-family acceptance remain unqualified.
Model changes remain stopped for the pending single softness diagnostic review;
no frozen failure is waived. All 160 preceding milestone and 22 app bindings pass;
old workflow documents are preserved before this update. App/model unchanged;
no neural/training/VM action or regression/Playwright rerun. Final execution needs
a justified candidate/cohort, independent reviewers and full app verification.
Goal active; manual L4 verified transfers/pasteable commands remain the only
training execution path.
Report: [Independent review readiness](<C:/xampp/htdocs/YEAR 4/Testing/INDEPENDENT_REVIEW_READINESS_V1.md>).
Protocol: [V1 R1](<C:/xampp/htdocs/YEAR 4/Testing/INDEPENDENT_REVIEW_PROTOCOL_V1_R1.md>).
Evidence: `outputs/dgp_independent_review_readiness_milestone_v1/`.

**Earlier 5 October 2026 — V20 feasibility fails; no new VM pilot:**
A separate ten-case training-only, clean-target-informed VQ-prior difference
check completes in 61.30s: ten DGP and twelve each teacher encoder/quantizer/
generator calls, all states unchanged. Independent 1.99s saved-array audit
passes 39 source/106 output bindings, 40 raw/PNG compositions, 50 paired metric
rows, ten exact differences and both exact clear-DGP controls. Both original
1340×1504 sheets are reviewed. Motion cases fail MSE and SSIM on both provenance
sources (four predeclared failures) and show colour/texture artifacts. Other
synthetic gains use unavailable clean-target information and are not restoration.

Close this exact recipe before conditioner construction or VM pilot; no output-
based profile exception or gate relaxation. No optimizer, backward, local training,
native/reserved use, app change or VM/cloud action. Model-path changes stop under
the workspace three-attempt circuit breaker while one user diagnostic review
clarifies whether the shown native DGP softness is useful. Feedback cannot erase
frozen failed gates. Current output acceptance remains unqualified.

The preceding native milestone remains: ChokePoint24 reviewed development crops,
12/13 labeled development/reserved identities; 72 identical-input CPU forwards
and all six sheets reviewed. Own DGP still softens visible details; CodeFormer
is a clearer declared baseline with unverifiable fine detail. Source/capture/
overlap limits and unpaired evidence remain explicit. QMUL32 and ChokePoint13
reserved sets stay unviewed. Existing app integration/34 regressions/Playwright
remain historical verified checks, not reruns. Useful DGP native outputs, full
automatic/assisted covering-family acceptance and independent final review remain
required and incomplete. Goal active; actual training stays manual on the L4.

Report: [V20 feasibility result](<C:/xampp/htdocs/YEAR 4/Testing/CCTV_DGP_LATENT_DELTA_FEASIBILITY_V20.md>).
Evidence: `outputs/cctv_dgp_latent_delta_feasibility_v20/` and
`outputs/cctv_dgp_latent_delta_feasibility_milestone_v20/`.
Native report: [CCTV_NATIVE_SOURCE_EXTENSION_V1.md](<C:/xampp/htdocs/YEAR 4/Testing/CCTV_NATIVE_SOURCE_EXTENSION_V1.md>).
Verified transfer files/pasteable commands-only execution preference remains.

**Earlier 5 October 2026 — native source/split/comparison audited; own DGP still unqualified:**
Native CCTV extension: publisher-linked ChokePoint original 800×600 frames are
acquired and independently release-audited (6,876 headers/72 annotation XMLs).
A source-specific split freezes 12 development/13 reserved person labels. All 24
development face crops (93×112–189×227) pass input-only rough-structure review;
26 reserved metadata cases remain undecoded. Geometry/split audit reproduces all
selections and 24 inputs. Cross-source/historical person overlap and capture
country remain unknown; no ethnicity or Zamboanga inference.

Matched resize/Phase 3/retained own DGP/CodeFormer and current Auto comparison
completes 72 CPU forwards in 177.31s, all states unchanged. Independent 3.05s
saved-output audit passes 102 source/226 artifact bindings and exact compositions,
Auto choices and all 120 sheet cells. All six original-resolution sheets reviewed:
retained DGP still softens visible glasses/eyes/mouths relative to resize; CodeFormer
is clearer but its fine detail is unverified. Auto selects 11 DGP/13 resize cases.
No native paired metrics, pretrained-main substitution or useful-DGP qualification.
Source terms/notices and model contributions remain explicit.

Separate larger-QMUL input scan finds three cases within the frozen 16,384-member
cap; all are out of scope, with shortfalls preserved and no model runs. The fixed
six-pixel completion-context trial exposes patch boundaries; per-image DGP
statistics fail two training-control SSIM guards. Both diagnostics are audited,
reviewed and closed without adoption. Original failures and app bindings remain.
No app source/threshold changes or new VM pilot. Earlier 34 regressions/Playwright
remain historical verified checks; they were not rerun for this evidence-only work.
Useful DGP native output, full automatic/assisted covering scope and independent
final review remain incomplete. QMUL32 and ChokePoint13 reserved sets stay unviewed.
All training stays on the L4; user runs verified transfer files/pasteable commands.

Reports: [Native source/comparison](<C:/xampp/htdocs/YEAR 4/Testing/CCTV_NATIVE_SOURCE_EXTENSION_V1.md>)
and [processing diagnostics](<C:/xampp/htdocs/YEAR 4/Testing/CCTV_DGP_PROCESSING_DIAGNOSTICS_V3.md>).
Evidence: `outputs/cctv_chokepoint_native_comparison_v1/`,
`outputs/cctv_chokepoint_native_development_v1/`,
`outputs/cctv_native_structure_extension_v1/`,
`outputs/dgp_processing_diagnostics_milestone_v3/`. Goal active.

**Earlier 5 October 2026 — covering review complete; context change unhelpful:**
36 reused photographic cases/108 requests complete in 428.18s with unchanged
detector/completion/DGP states. Independent 6.15s audit reproduces all 96 PNGs,
32 Auto aliases and Off visible preservation. Twelve expected requests reject
through input-only operator review. Automatic proposals miss many coverings;
assisted outputs retain finger/scarf/boundary defects and inconsistent gaze.
All six route sheets reviewed at displayed resolution. Generated protected masks
have zero support; separate clarification prevents vacuous On-preservation claims.
Paired synthetic nonremoved MSE regresses in all 11 family groups with DGP On;
SSIM improves in three. No native or clean-hidden reference metrics.

Separate 32-forward/17.07s DGP-on-completed-context comparison and 4.81s saved-array
audit pass composition/state/control checks. All eight original-resolution sheets
are reviewed. Only 1/16 MSE and 3/16 SSIM improve over Off; the variant is not
adopted. Unchanged app passes bundled inline Playwright glare/hands/hair/downloads,
renewed mask review and near-hidden operator rejection in 30.25s, with zero errors/
overflow at 375/768/1280. Three PNGs and bundle input/mask/raw/result are exact.
Useful native output, full covering scope and independent final review remain
incomplete. No reserved native use, local training, new VM pilot or cloud action.
Manual VM commands-only preference and all historical failures remain intact.
Report: [DGP v3 covering results](<C:/xampp/htdocs/YEAR 4/Testing/CCTV_DGP_APP_V3_COVERING_RESULTS.md>).

**Earlier 5 October 2026 — DGP-led research app functionally verified:**
the main route uses retained trained identity-v2 DGP at 256×256, observed-support
Auto/On/Off, required operator input/removal-area review and raw/display-separated
exports. 34 regressions pass. Eight real CPU DGP forwards in 4.56s exactly match
the six frozen native core raws; states stay unchanged. All six native rows
remain soft, without useful structural improvement. Bundled inline Playwright
passes actual-model flows, downloads/rejection paths and 375/768/1280 layout/error
checks. Assisted mild-turn cloth completion and exact clear-glasses Off control
work; automatic cloth proposals fail. Three-quarter mask/glasses case remains
an out-of-scope diagnostic with residue. No full-family or native acceptance,
automatic structure classifier, reserved/final review, local training or VM action.
Report: [DGP app v3](<C:/xampp/htdocs/YEAR 4/Testing/CCTV_DGP_APP_V3_INTEGRATION.md>).
Keep the full goal active, failed gates intact, and manual VM commands preference.

**Earlier 5 October 2026 — V19 r2 executed, audited/reviewed; preservation fails:**
successful 1,507,038,572-byte return matches sidecar/receipt/hash and all 189 frozen
assets. L4 completes 520 development cases in 175.04s; full export 356.31s, zero
updates/backwards and unchanged states. Independent Windows audit 97.61s passes
5,957 artifacts, 2,080 metrics/cosines, 1,040 raw compositions, 50 fresh training
parity cases, exact 520 V15 DGP PNGs, all aliases and 24 cached CPU decoder replays.
The normalization correction works; original V19 failure/diagnostic stay preserved.

Automatic synthetic degraded PSNR gains 3.027dB/50.19% MSE reduction, but SSIM
0.61930→0.60458 and cosine 0.33011→0.22346 regress. Thirty automatic preservation
checks fail; 33 unconditional checks fail. All five original-cell sheets reviewed
show patchy colour/detail and unstable visible features/glasses. Three clear
development inputs wrongly take the spatial branch. No gate or threshold changes.

Separate 17.23s saved-output luminance analysis reproduces 83.59% of the mean MSE
reduction with zero neural/training calls, but still fails 13 MSE/SSIM checks;
no fresh recognizer/full qualification or promotion. Close R2 commands. Isolate
exposure correction and explicit structure/appearance preservation before another
separately justified finite VM pilot; no unchanged repeat or development refit.
Report: [V19 r2 results](<C:/xampp/htdocs/YEAR 4/Testing/CCTV_DGP_INPUT_SELECTION_V19_R2_RESULTS.md>).
This is repeatedly used photographic synthetic development evidence, with no
native/reserved/independent-final/Zamboanga claim. DGP-led app/override, input
usability, useful native outputs, covering families and full final/Playwright
review remain incomplete. Goal active; manual VM execution preference persists.

**Earlier 5 October 2026 — normalization cause confirmed; V19 r2 prepared:**
the returned 7,380,842-byte diagnostic passes transfer and independent saved-array
audit. Eight frozen L4 DGP forwards demonstrate NumPy-before-GPU recovers the
V15 development PNGs exactly; CUDA scalar division instead preserves V18's
training path exactly. Tiny float differences produce a few one-byte PNG changes.
Original V19 failure and the diagnostic are closed and preserved.

Separate immutable V19 r2 keeps both conventions on identical RGB inputs,
adds one canonical DGP forward per development case, and saves raw/PNG evidence
before parity stops. Eight regressions, independent transfer audit and actual
frozen-package source/data preflight pass. Archive 87,831,525 bytes/191 members;
all 146 original V19 members remain byte-identical. Checkpoints, 104 development
identities/520 synthetic cases, splits, selector thresholds and scientific gates
remain unchanged. Inference only; 20min inference/30min overall finite limits.
No R2 VM launch, local training or model adoption is claimed.
Manual commands: [V19 r2 runbook](<C:/xampp/htdocs/YEAR 4/Testing/CCTV_DGP_INPUT_SELECTION_V19_R2_VM.md>).
Evidence and limits: [V19 r2 plan](<C:/xampp/htdocs/YEAR 4/Testing/CCTV_DGP_INPUT_SELECTION_V19_R2_PLAN.md>).
Returned audit and all-five-sheet development review must precede the next model
decision. Native/reserved/final evidence and the full DGP-led app/covering-family
scope remain incomplete; the full goal stays active.

**Earlier5 October2026 V19 failure closure:** downloaded167,982,054-byte failure
matches receipt/hash and all144 original assets. The first development case
stops at exact V15 DGP PNG equality after50 fresh training parity cases passed.
No completed development predictions or final state/count receipt. Independent
5.59s partial audit verifies100 unchanged training raw/PNG compositions,728 exact
camera/target/support images,104 target embeddings and six initial states.
Zero local neural/backward/optimizer calls. No model-quality/adoption conclusion.

Two different normalization paths are a source-backed processing hypothesis;
the failed raw/PNG was not saved. Separate21KB three-case/eight-forward DGP-only
diagnostic is prepared/transfer-audited, with four regressions and120s worker/
240s overall limits. It preserves original V19 and all scientific gates, saving
both raw outputs before checking. No training/resume or assistant cloud launch.
Manual commands:[V19 parity diagnostic](<C:/xampp/htdocs/YEAR 4/Testing/CCTV_DGP_INPUT_SELECTION_V19_PARITY_DIAGNOSTIC_VM.md>).
Report:[V19 failure](<C:/xampp/htdocs/YEAR 4/Testing/CCTV_DGP_INPUT_SELECTION_V19_FAILURE.md>).
Original V19 launch commands are closed; next audit the separate diagnostic
return before any correction or next pilot. Full thesis/app goal remains active.

**Earlier5 October2026 V19 preparation:** a separate input-only training control
retains DGP for ten clear cases and fixed V18 update600 for forty degraded cases.
Runtime uses only pixels/canvas support, with fixed Laplacian thresholds and no
label/target/output score. Separate50-case saved-output audit2.53s verifies exact
aliases/raw compositions and unchanged training guards; ten regressions pass.
V18's failed unconditional guards stay unchanged. No generalization/adoption claim.

Finite inference-only V19 now frozen for unchanged104 development identities/
520 synthetic cases, preceded by50 fresh parity checks. Five comparison arms,
including resizing/retained DGP/declared pretrained baseline; no optimizer,
backward, threshold refit, native/reserved use or best checkpoint. Inference20min,
overall30min, finite timing stop and unchanged scientific gates.80,388,709-byte
package/146 members/798 data files independently verified in2.39s. No VM launch.
Manual commands:[CCTV_DGP_INPUT_SELECTION_V19_VM.md](<C:/xampp/htdocs/YEAR 4/Testing/CCTV_DGP_INPUT_SELECTION_V19_VM.md>).
Full evidence limits:[V19 plan](<C:/xampp/htdocs/YEAR 4/Testing/CCTV_DGP_INPUT_SELECTION_V19_PLAN.md>).
Returned audit/all-five-sheet review precede any next training or model decision.
Repeated-use photographic development evidence is not independent final/native
CCTV/Zamboanga acceptance. The full goal remains active/incomplete.

**Completed5 October2026 V18 closure:** successful481,723,919-byte return matches
hash/size/sidecar/receipt and all21 frozen assets. Independent CPU audit60.29s
passes250 replays/metrics/cosines,50 starting DGP parity cases,eight fresh VM
parity cases,600 updates/6,000 exposures and550 original grid cells. All ten
sheets are reviewed. Separate arithmetic-only corrections preserve original
float32/log10 equality failures; coefficients, scientific gates and checkpoint/
protocol/source remain unchanged. Zero local training/backwards.

Degraded synthetic training PSNR15.5527→23.5201dB,SSIM0.62517→0.70649 and84.03%
MSE gain show capacity on ten fitted photographs. Four clear preservation guards
still fail; no trained snapshot qualifies for generalization under V18. Soft/
patchy compound features and weak glasses remain. No held-out/native/reserved/
teacher/selection/adoption or local/Zamboanga performance claim. V18 commands
are closed; no repeat. Next is a separately documented input-only clear-input
preservation/automatic-selection processing diagnosis before another protocol.
[V18 result report](<C:/xampp/htdocs/YEAR 4/Testing/CCTV_DGP_STRUCTURE_V18_RESULTS.md>).
The full restoration/completion/app/final-review goal remains active/incomplete.
Historical preparation and maintenance snapshots below are superseded by this
return closure; all frozen scientific evidence stays intact.

**Latest 5 October 2026 storage maintenance:** the user completed V18 upload
steps1/2/3 and explicitly authorized direct gcloud connection and removal of
unnecessary files before continuing.74 remote archives match preserved Windows
copies;104 pip-cache files are disposable. Cleanup recovers9.0GiB; separate
`df -h /` confirms87% used/13GiB available. All10,480 protected hashes and4,739
scientific tensor records match before/after; both large feature caches remain.
Three V18 upload hashes are verified. No training launch or VM shutdown.
Receipts: `outputs/cctv_dgp_vm_storage_cleanup_20261005/`. This authorizes storage
maintenance; the existing manual V18 launch/return workflow remains in force.

**Earlier V18 milestone,5 October2026 — prepared, training pending at that time.** A separate
spatial decoder retains the audited DGP pixel base and learns a correction from
input/DGP RGB plus declared frozen CodeFormer features. Our R2 code head stays
frozen. Two fresh training-only CPU inputs pass exact initial DGP parity and
unchanged state checks in13.60s; independent eight-artifact audit0.1064s and
thirteen meaningful regressions pass with zero local training/backwards.

The immutable9,063,630-byte VM package has23 separately verified safe members,
21 source/assets and70 selected data files. Ten training photographs/fifty
cases;600 updates/6,000 exposures maximum; fixed0/50/200/600 snapshots; update50
full-cohort fitting stop; unchanged structure/appearance requirements.
Cache300s/fit900s/audit300s/overall1,800s and bounded export. No validation/native/
reserved use, selection or promotion. This is synthetic photographic capacity
evidence only; initial zero residual is parity, not trained improvement.
Protocol e3a7567e9a9c53a1668464ab5c95edbf094c73e9a7c8e550635dcf8d8ef04adb.

Use [CCTV_DGP_STRUCTURE_V18_VM.md](<C:/xampp/htdocs/YEAR 4/Testing/CCTV_DGP_STRUCTURE_V18_VM.md>)
for the exact manual gcloud/tmux/return commands. The user's transfer-files-only
preference remains authoritative for training. The separate user-authorized
storage maintenance above is complete. No assistant V18 launch or independently
confirmed live V18 run; CUDA gradient checks and trained-quality review are
pending. Returned results must pass separate raw/PNG/trace/cache audits and CPU
head replay, followed by all-original-cell development review. Rationale/limits:
[CCTV_DGP_STRUCTURE_V18_PLAN.md](<C:/xampp/htdocs/YEAR 4/Testing/CCTV_DGP_STRUCTURE_V18_PLAN.md>).
The full native restoration, covering-family and app/final-review goal remains
active and incomplete.

**Earlier V17 processing control closed, 5 October 2026.** The separate training-only
12-case local check uses original encoder feature connections with fidelity w1
and statistics omitted. Inference 139.99 seconds; zero training updates. Its
2.70-second saved-output audit and all three original-cell reviews are complete.
Clear previews improve versus w0, but degraded PSNR/SSIM and visible fragments
worsen; default w1 is rejected as a universal repair. No fresh-image head,
validation/native/reserved use, selection or app promotion. R2 history and
unchanged preservation failures remain retained. Report:
`CCTV_DGP_FIDELITY_SPOTCHECK_V17_RESULTS.md`. Before more training, design an
explicit DGP/input-structure-preserving path and verify initial parity. A finite
VM capacity pilot must test learned benefit before broader training or adoption.
At that review, no next VM package was prepared. V18 preparation above now
addresses the interface/capacity prerequisite; its training remains pending.
The Goal remains incomplete.

**V16 r2 audited/reviewed; preservation failed,5 October2026.** The downloaded
839,574,170-byte failure archive matches the pasted hash/receipt. Safe import
preserves the original failure; a separate240s bounded auditor corrects only
stale timing tuples and passes in73.05s. All3,128 updates/31,280 exposures,
1,710 PNGs/cosines,300 raw previews,100 fresh parity cases,781 teacher arrays,
4,425 cache bindings and600 original grid cells are verified with zero local
neural/backward/optimizer calls. Original protocol/source/checkpoints stay unchanged.

Epoch8 paired degraded PSNR improves16.027→17.402dB, but SSIM0.6193→0.5788
and fixed ArcFace0.33011→0.19280 regress. Clear appearance also regresses;
both trained snapshots fail unchanged preservation guards. All ten original
256-cell sheets show visible-feature changes/fragments, including training
previews. This is not an adopted upgrade or native CCTV/Zamboanga evidence.
Report: `CCTV_DGP_BROADER_CODES_V16_R2_RESULTS.md`. Next: bounded inference-only
appearance-preserving rendering controls before more training. No unchanged
failed-recipe repeat, checkpoint selection, app promotion or assistant cloud
operation. The Goal remains active and incomplete.

**Earlier V16 r2 report and audit preparation,5 October2026:**
The user reports3,128 updates/epoch8 completed in1,055.125922286s, followed by
`Timing stop receipt differs` in the arithmetic audit. The pinned R2 checker
still expects20 cache timing references; the pinned R2 protocol requires30.
This migration defect is locally reproduced. A separate checksum-bound audit
correction reads timing counts/caps from the unchanged design; seven regressions
pass in0.124s with zero neural/backward/optimizer calls. Preserve the completed
training, all frozen sources/protocols/checkpoints, the original failed audit and
the failure export. No retraining or cap relaxation. Collection and independent
full corrected audit are pending; useful restoration is not established.
`CCTV_DGP_BROADER_CODES_V16_R2_AUDIT_RECOVERY.md` describes the separate240s local
audit and `CCTV_DGP_BROADER_CODES_V16_R2_VM.md` has exact manual download commands.
No assistant cloud execution. The Goal remains active and incomplete.

**Earlier V16 failure audit and R2 preparation:** The
downloaded failure contains0 backward/optimizer updates and20 train-only teacher
arrays. Export/hash, original23 members and exact bootstrap recovery provenance
are verified. The startup-inclusive timing formula's31.24-minute estimate triggered
the unchanged15-minute gate; actual steady cost was not measured separately.
`CCTV_DGP_BROADER_CODES_V16_FAILURE_AUDIT.md` records the failure and preserved head.

V16 r2 records startup once and samples cache timings by source/role, excluding
four first-reference warmups from steady rates while counting their actual cost
once. Same datasets/splits, seeded head, optimizer schedule,3,128 updates, quality
guards and finite caps; separate immutable root/protocol/archives. Eight regressions
pass with zero local neural/backward/optimizer calls. New archive1,098,502 bytes,
27 independently verified members;116 parent/6,195 data/2,804 baseline assets checked.
Protocol4f64cf2204458f8fadcab1824fc4bc293c65803e123fdebb304f7b5bd3a502a0.
Exact manual VM commands: `CCTV_DGP_BROADER_CODES_V16_R2_VM.md` ↔ intended VM copy
after separate transfer. No assistant cloud execution or local training occurred;
the R2 training report above awaits independent return audit. No useful restoration
or application adoption is established. Original V16 failure and
native24/reserved32 remain preserved. The Goal stays active; manual r2 return audit,
visual review, DGP-led app flow, covering families and independent final review remain.
<!-- V16 r2 current status end -->

## Earlier V16 report

**V16 manually launched; cache timing gate rejected,5 October2026.** The user
selected transfer files and pasteable VM commands only, superseding automatic
gcloud execution instructions in historical paragraphs below. The1,088,480-byte
execution archive has23 independently verified members. Fourteen boundary checks
passed with zero local backward/optimizer updates; preparation verified116 parent,
6,195 data and2,804 baseline assets. The reset code-only pilot has a frozen
3,128-update/eight-epoch budget, separate104-reference development validation,
unchanged preservation guards, training-only teacher labels, finite timing/epoch4
fit stops and no selection/promotion. No completed optimizer training, backward
preflight result or useful restored output is claimed. Exact files/commands/return audit:
`C:\xampp\htdocs\YEAR 4\Testing\CCTV_DGP_BROADER_CODES_V16_VM.md` ↔ intended
`~/forensic-dgp/CCTV_DGP_BROADER_CODES_V16_VM.md` after separate transfer.
Protocol4331c28c97a659846eab76ee0dee9150811a00da3c6139d8d24b7905173a6681.
Two manual launch reports stopped before training: an outer tmux session triggered
the idle guard, then the original bootstrap's captured preflight child returned1
after extraction. Local reproduction proves a string/Path caller bug. A separate
preflight recovery r1 is prepared with ten passing regressions and zero local
neural/backward/optimizer calls. It preserves the original package/failed directory,
verifies the actual VM error before correction, and retains the original finite
protocol and unchanged supervisor. Recovery SHA256:
489e85d001b3cca74fdeca3c9fbd50edc2e5bf1fa7aa30c2b45db36bce99bb17.
The user subsequently reported recovery preflight success and detached tmux launch,
then trainer failure `Cache projection exceeds900s` at the20-reference estimate.
That gate precedes backward preflight and optimizer updates; counters and timings
await independent return review. Collect the failure archive/checksum/export
receipt and inspect timing and recovery provenance before any new recipe or cap.
The startup-inclusive timing estimate is a demonstrated processing concern, not
grounds to rerun the failed protocol unchanged. No success result or adoption.
The user-reported20-reference timing is22.754411s elapsed and1,874.529446s projected
against900s. Local arithmetic matches the frozen formula exactly; independent
serialized failure audit and steady cache-rate measurement remain pending.
The trained DGP baseline, negative historical experiments, native24/reserved32
and existing app remain preserved. Returned-result audit, useful outputs,
DGP-led app integration, covering-family readiness and independent final review
remain unfinished. The Goal stays active.
<!-- V16 preparation current status end -->

**V15 generalization probe closed — negative, 4 October 2026:** all50
fresh-image outputs matched the audited ten-face cache; fixed104-reference/
520-case development validation then exposed severe overfitting. Degraded PSNR
13.987dB versus retained DGP16.027dB; fixed ArcFace0.07662 versus0.33011. Clear
PSNR14.081 versus DGP31.132dB. Both unchanged preservation guards fail; embedding
similarity regresses in every degraded source/profile group. All five fixed
original-cell grids show distorted/changed features. No useful upgrade/adoption.
L4 inference212.12s; full audit/export296.54s; peak VRAM1,002,894,336bytes.
Independent local audit55.49s rebuilt1560 PNGs/2080 cosines,150 raw previews,
50 fresh/cached parity renders and250 grid cells, with zero neural/training calls.
All741 execution/data/head assets and116 parent assets reverified on VM.
Zero optimizers/backwards/updates; no teacher/native/reserved/selection/promotion.
Windows C:\xampp\htdocs\YEAR 4\Testing\CCTV_DGP_GENERALIZATION_V15_RESULTS.md ↔
intended ~/forensic-dgp/CCTV_DGP_GENERALIZATION_V15_RESULTS.md after sync.
Local outputs\cctv_dgp_generalization_return_v15\ ↔ VM
~/forensic-dgp/cctv_dgp_generalization_vm_v15/outputs/generalization_v15/.
All returns collected; fresh idle check passed; Cloud **TERMINATED**,
stop2026-10-04T08:21:58.016-07:00. No docs/git publication or app integration.
**Next:** separately prepare a reset code-only head on781 existing training
references, balanced source/profile exposure and finite VM-only budget. Do not
resume the memorized head. Broader data is a hypothesis; clear appearance also
needs a DGP-preserving output path before adoption. Retain the baseline and guards.
Protocol6c0e1de5bcb56b755b82a3289adb4905c591f49908f02365ad6ad5fbd151f9da.
<!-- V15 current status end -->

Confirmed with the user on 2 October 2026 after three workflow question rounds.
This document records intended behavior and completion criteria, not evidence
that the application or models already meet them.

## Current Goal control — 4 October 2026

**V14 r2 capacity diagnostic closed, 4 October 2026:**1,000 updates/2,000
exposures completed on the L4 in97.27 seconds; training/audit/export163.34 seconds,
peak allocated VRAM1,187,559,424 bytes. The587,564,966-byte return is independently
audited (450 renders,150 code/stat probes,1,000 traces, zero local neural/backward/
training calls). All five original256-cell grids are reviewed. Direct prediction
now produces coherent faces on the ten training photographs. Degraded no-stat
PSNR13.522→24.272dB, code accuracy2.02→98.95%; this is fitting, not generalization.
Observed statistics retain dark/noisy degradation; predicted statistics cause
color/brightness drift and clear PSNR24.731→23.572dB. Do not adopt the stat head.
Original zero-update V14 failure and separate numeric-only r2 correction remain
preserved; all100 starting parity checks and frozen-state/quality gates passed.
No validation/native/reserved use, best.pth, selection or production promotion.
Windows C:\xampp\htdocs\YEAR 4\Testing\CCTV_DGP_DIRECT_CODES_V14_R2_RESULTS.md
↔ intended ~/forensic-dgp/CCTV_DGP_DIRECT_CODES_V14_R2_RESULTS.md after sync.
Local outputs\cctv_dgp_direct_codes_return_v14_r2\ ↔ VM
~/forensic-dgp/cctv_dgp_direct_codes_vm_v14_r2/outputs/cctv_dgp_direct_codes_v14_r2/.
All11 execution sources/116 parent assets reverified. Returns collected; idle
checks passed and Cloud confirms **TERMINATED**, stop2026-10-04T07:54:24.806-07:00.
No git/doc publication. The active Goal and DGP-led app integration are incomplete.
**Next:** separately freeze a no-training fresh-image parity/generalization probe
on the unchanged104-reference/520-case development validation cohort. Freeze
no-stat rendering from training evidence; compare retained DGP and starting prior,
report clear and all source/profile groups; keep native/reserved data untouched.
Broader training requires a later justified finite VM protocol.
Protocol d7d5dcac64202c24a7c85658e6b660bdc806bf7d8c16b09542c53170f40a1da2.

**V13 rendering control closed, 4 October 2026:** all 50 cases completed in
52.21 seconds on the L4, with 162 frozen-generator forwards and zero training
updates. The 152,609,077-byte return is verified. Independent local audit took
10.98 seconds and checked 200 PNGs, 160 distinct raw renders and 350 grid cells
without neural calls. All five grids are reviewed. Clean teacher codes render
coherent faces; predicted codes remain structurally wrong, while observed
degraded AdaIN statistics carry dark/noisy texture. Removing statistics alone
worsens several predicted-code outputs. Both prediction and clean rendering
statistics need repair. Oracle arms use target codes and prove no learned
upgrade. No checkpoint/selection/promotion, validation, native or reserved use.
Windows report `C:\xampp\htdocs\YEAR 4\Testing\CCTV_DGP_FACE_CODE_CONTROLS_V13_RESULTS.md`
↔ intended `~/forensic-dgp/CCTV_DGP_FACE_CODE_CONTROLS_V13_RESULTS.md` after sync.
Audited local return `outputs\cctv_dgp_face_code_render_controls_return_v13\`
↔ VM `~/forensic-dgp/cctv_dgp_face_code_render_controls_vm_v13/outputs/cctv_dgp_face_code_render_controls_v13/`.
All returns are collected; idle checks passed and Cloud confirms **TERMINATED**,
last stop `2026-10-04T06:55:04.228-07:00`. No git/doc publication occurred.
**Next:** prepare our direct code-prediction and clean-statistics conditioning
layers with starting parity and a finite VM-only train-cohort capacity pilot.
Keep the retained DGP and declared pretrained renderer frozen; useful training
outputs are required before a separately versioned held-out experiment.

**Current V9 closure:** all7,820 updates completed on the L4; unchanged guards
rejected all trained snapshots. Epoch20 gains5.1719dB degraded PSNR but reduces
fixed ArcFace similarity in all eight degraded source/profile groups. Return,
113.84-second independent local audit and all five original-cell preview reviews
are complete. `best.pth` retains V2, epoch0; no production promotion or trained
native forwards occurred. The assistant restored the prior stopped state after
collection and idle checks; Cloud confirms TERMINATED. Detailed evidence:
`C:\xampp\htdocs\YEAR 4\Testing\CCTV_DGP_MIXED_V9_RESULTS.md` ↔ intended
`~/forensic-dgp/CCTV_DGP_MIXED_V9_RESULTS.md` after document sync.

**Latest architecture authorization:** the user permits adding a pretrained
face-generating prior with our own trained conditioning layers if reviewed output
improves. Declare both contributions; a pretrained baseline alone is not our
trained DGP. First run a separately versioned inference-only feasibility comparison
before selecting another finite VM training recipe. This later user decision
extends the original baseline-only architecture scope; it does not change frozen
V1–V9 experiments. All training remains VM-only, reserved32 remain unused and
the full Goal stays active. Configured Cloud start/stop/SSH/SCP access supersedes
the old no-SSH sentence in the Goal-tool objective.

**V10 closed as a negative cascade comparison:** all 50 paired and 24 native
development outputs were saved, but the original runner failed at preview rendering
before final model-state/count receipts. A separate no-forward recovery and
independent arithmetic/grid audit passed; all ten grids were reviewed. Direct
DGP → CodeFormer sometimes adds detail but invents glasses, facial hair or
expressions. No adoption, checkpoint selection or app promotion. Reserved 32
remain unused; the VM stays stopped. See `C:\xampp\htdocs\YEAR 4\Testing\CCTV_DGP_FACE_PRIOR_RESULTS_V10.md`
↔ intended `~/forensic-dgp/CCTV_DGP_FACE_PRIOR_RESULTS_V10.md` after explicit sync.
Next: a separately versioned feature/code-conditioning prototype, zero-conditioner
parity and prior-capacity checks before a finite L4 training pilot.

**V11 interface/capacity check complete:** our untrained 455,072-parameter
conditioner starts at zero and reproduces the frozen CodeFormer baseline exactly
on two artificial RGB cases. Frozen states/counts are verified; 161.12-second
local inference and 5.39-second independent no-forward audit passed. Both
five-reference training-source grids are reviewed. A clean-code oracle forms
coherent faces but changes some features; it does not restore CCTV or establish
identity fidelity. Next: a bounded VM-only fitting diagnostic, with gradient/VRAM
preflight and no checkpoint promotion. No validation or native data was used.
Runbook: `C:\xampp\htdocs\YEAR 4\Testing\CCTV_DGP_FACE_CODE_V11.md`
↔ intended `~/forensic-dgp/CCTV_DGP_FACE_CODE_V11.md` after explicit source sync.

**V12 r2 fitting closed, 4 October 2026:** the L4 completed 300 updates / 600
exposures on ten training faces. Trainer 125.22 seconds; complete VM
training/audit/export 177.28 seconds; peak allocated VRAM 1.43 GB. CUDA
zero-conditioner parity and gradient preflight passed. The 468,092,580-byte
return and all 116 VM source/asset bindings are verified. A 21.64-second local
no-forward audit rebuilt all 300 renders, 150 code probes and 500 grid cells.
All ten original-cell grids were reviewed. **No useful upgrade:** degraded
code accuracy stays near 2%, clear accuracy falls 21.35% to 13.78%, and facial
artifacts persist. No best.pth, selection or production promotion; no validation,
native development or reserved access. The original Python3.10 loader failure
(zero updates) and its separate r2 compatibility correction remain preserved.

Current report: `C:\xampp\htdocs\YEAR 4\Testing\CCTV_DGP_FACE_CODE_FIT_V12_RESULTS.md`
↔ intended `~/forensic-dgp/CCTV_DGP_FACE_CODE_FIT_V12_RESULTS.md` after doc sync.
Audited local return: `C:\xampp\htdocs\YEAR 4\Testing\outputs\cctv_dgp_face_code_fit_return_v12_r2_verified\`
↔ executed VM results below `~/forensic-dgp/cctv_dgp_face_code_fit_vm_v12_r2/outputs/cctv_dgp_face_code_fit_v12/`.
The earlier sandbox-blocked partial extraction is retained separately. Receipts,
learning diagnosis and grid review are linked in the report. All return files
are local; idle checks passed and Cloud confirms **TERMINATED**. Source/doc/git
publication has not occurred. The active Goal remains incomplete.

**Next:** freeze an inference-only training-cohort control that separates code
prediction from original-input AdaIN/fidelity rendering. Clean teacher-label
arms are declared oracle diagnostics, not learned results. Do not repeat this
recipe or scale it up before identifying a specific repair.

**Historical V9 running snapshot:** source-only review accepted 390 of 451 original Asian
training candidates (28 coverings/33 quality exclusions), combined with 391
reviewed HQ FFHQ references. All 3,905 prepared training cases and unchanged
520 validation inputs are independently checked. The executable, return auditor,
VM-only guard and 7,820-update/20-epoch finite schedule are frozen; ten runtime
boundary tests passed and all 6,232 archive members match pinned hashes. Trainer
cap is 40 minutes; complete training/audit/export cap is 60 minutes. Starting
V2 baseline, clear/degradation replay and unchanged per-source/profile identity
selection safeguards remain. The assistant started the VM through configured CLI;
idle/CUDA checks, transfer and all 6,232 asset hashes passed. Dedicated tmux
`dgp_mixed_v9` is running with the existing CUDA versions unchanged. CUDA
preflight passed; update-32 timing projects 1,492.51 seconds within the 40-minute
trainer cap. A separate inference loader preserves the same five frozen
normalization layers; 14 tests and four retained-checkpoint CPU forwards passed
with unchanged state and matching artificial-signal outputs. These checks do
not establish useful restoration. Completion,
return audit and reviewed useful outputs remain pending. See `C:\xampp\htdocs\YEAR 4\Testing\CCTV_DGP_MIXED_V9.md`
↔ intended `~/forensic-dgp/CCTV_DGP_MIXED_V9.md` after document sync. Full Goal stays
active; all actual training remains VM-only and native reserved cases remain unused.
Later execution receipts supersede this snapshot. The V8 paragraph below describes
the earlier completed diagnostic and its prior VM stopped state.

**Previous V8 execution:** V8 isolated blur fitting completed on the existing
L4 after independent offline preparation and exact VM asset verification.
Frozen budget: 1,000 updates, snapshots 20/100/1,000, trainer cap 600 seconds,
complete training/audit/export cap 900 seconds. This is training-only fitting
of two existing V7 blurred pairs with collateral evaluation of all ten training
pairs. It completed 1,000 updates in 45.95 seconds; execution/audit/export took
67.87 seconds. Return audit checked 40 PNGs/raw predictions, 50 embeddings,
1,000 traces and three changed states, with all four grids reviewed. Fixed final
blur error fell about 80% on the two fitted images; collateral faces show strong
texture/contrast and color regressions. This is fitting capacity, not a useful
released restorer. The assistant started the previously stopped VM through
configured CLI access and restored it after collection and competing-job checks;
Cloud status at V8 closure was `TERMINATED`.
See `C:\xampp\htdocs\YEAR 4\Testing\CCTV_DGP_BLUR_FIT_V8.md` ↔ intended VM
`~/forensic-dgp/CCTV_DGP_BLUR_FIT_V8.md` after document sync. No validation/native
use or production promotion is permitted by this fit diagnostic. The full Goal
and VM-only training instruction remain active. Next: source-only review of the
451 original Asian training references, then freeze a broader finite mixed-source
pilot with clear/degradation replay and unchanged validation safeguards. The
coverage audit verifies 1,353 original Asian assets, 451 target pixel hashes and
1,684 existing camera inputs, preserving original roles and the 104 validation
references. V6's controlled HQ target experiment omitted Asian training examples;
Asian cases remained validation sentinels. All 451 replay candidates have native
minimum edges below 256 and need a separate source-quality/covering review. No
direct train/validation source-hash collision was found; identity disjointness and
generalization remain unestablished. No next executable training protocol exists
yet. Evidence: `C:\xampp\htdocs\YEAR 4\Testing\outputs\cctv_dgp_training_coverage_v9_audit.json`
↔ intended VM copy after preparation, not claimed synced. Later paragraphs are
historical experiment snapshots where execution status differs.

**Latest user override: use the VM for training.** After learning that the assistant
can start the existing Google Cloud instance through the configured CLI, the user
withdrew the short-lived permission for small local training. All actual training
now uses the existing L4 at `~/forensic-dgp/`; local data preparation, inference and
audits remain allowed. Start the VM when a justified finite experiment is ready,
verify its state and competing processes, collect the result, and restore a VM
started solely for that experiment to its prior stopped state when safe. No local
CUDA installation or device-fit experiment is needed. The full DGP-first objective,
split safeguards, quality requirements and finite experiment budgets remain active.

Local hardware check: NVIDIA GeForce RTX 3050 Laptop GPU, 4,096 MiB VRAM; current
`C:\xampp\htdocs\YEAR 4\Testing\venv\` has PyTorch `2.13.0+cpu`, CUDA unavailable.
The prior matched eight-image V5 preflight recorded 4,398,557,696 allocated CUDA
bytes (4.10 GiB). This is historical hardware evidence; training location is now
the user's VM instruction, regardless of pilot size. The V6 package
was prepared locally with zero model forwards/backward calls/updates: 391 training
references, 104 validation references, 520 validation cases, 196 proposed updates,
and a 1,200-second pilot cap. Independent local preparation audit passed in 33.67
seconds: all 1,302 input PNGs, 391 reduced targets and 444 canonical sources were
rebuilt. Standalone L4 preflight passed in 27.40 seconds with zero updates. The
matched pilot completed 196 updates in 259.18 seconds; no trained epoch passed
the unchanged selection guards. The return, independent local output/execution
audits and all five paired preview reviews are complete. Integrity and training
completion do not establish useful output. See `CCTV_DGP_TARGETS_RESULTS_V6.md`.
The VM initially reported `TERMINATED`. It has now been started for the authorized
bounded comparison; L4/CUDA and an idle GPU are verified. The new external IP's
SSH key matches six previously trusted instance keys and is explicitly pinned.
The slow full upload was cancelled and its partial preserved. The 63.6-MB transport
passed locally but stopped on VM camera-library pixel drift before training. A
100.1-MB exact-byte recovery restored all 2,709 original assets. A separate portable
driver binds the independent local derivation receipt and invokes the unchanged
V6 verifier, trainer and output auditor; no model/loss/selection rule changed.

All 510 genuine higher-resolution FFHQ counterparts and 256×256 targets are
acquired and independently audited, preserving the 451/59 historical roles.
The audit checked thumbnail/source/target hashes and all 32 source-only pages
in 61.84 seconds. A read timeout left three files missing; bounded recovery
fetched only 4.1 MiB in 42.18 seconds, preserving the original failure receipt.
All acquisition/audit process handles are terminal. All 32 source pages and 38
ambiguous 1024 sources are now reviewed under the fixed rubric; a separate frozen
ledger accepts 444 references (391 training-role, 53 validation-role), excludes
60 coverings and excludes six unsuitable references. The 510 source/target hashes,
original roles, page bindings and initial 16 sample decisions pass the freeze check.
This assistant source review is not independent final human/output review.
The finite V6 pilot trained both matched arms on the L4 after evaluating their
identical starting model on the common approved targets. Its output review led to
a separate ten-pair V7 training-fit diagnostic: 200 L4 steps in 35.15 seconds,
with matched pixel-only versus retained objectives. Both fail the fixed final
blur/motion fit criterion. The independent local audit and final grids are complete;
pixel-only blur outputs show eye/color artifacts. This does not prove the model
cannot fit a face; each example had only 20 exposures among other tasks. Next:
freeze an isolated-blur diagnostic with 1,000 updates, snapshots at 20/100/1,000,
a 600-second trainer cap and a 900-second complete execution cap,
then execute on the VM. V7 is not a full-cohort/native improvement or a release.
After collecting both experiments and verifying no other jobs, the assistant
restored the originally stopped VM; current Cloud status is `TERMINATED`.
The Goal stays active, and the VM can start when the next verified pilot is ready.
See `CCTV_DGP_CAPACITY_V7.md` and the current `PROJECT_HANDOFF.md` milestone.
Preparation used zero model
forwards/backward calls/updates. Source criteria
are fixed before any new model evaluation. Evidence:
`C:\xampp\htdocs\YEAR 4\Testing\CCTV_DGP_HQ_TARGETS_STATUS.md`
↔ intended VM `~/forensic-dgp/CCTV_DGP_HQ_TARGETS_STATUS.md` after transfer.
The full objective, earlier safeguard failures and native reserved set remain
intact; the latest VM-only training instruction above now applies.

A fresh Goal-tool read confirms `active` with the unchanged DGP-first objective
below. The manual VM-support pause recorded in `PROJECT_HANDOFF.md` is historical.
Direct read-only Google Cloud API/SSH access is now verified using the installed
CLI and command-scoped Windows trust bundle, with TLS verification enabled.
The objective's earlier "no configured SSH" statement records the former
capability limit. Current instance zone is `us-central1-a`. The user's existing
V5 process has now finished: 452 updates in 348.84 seconds. Its export is local,
and the user's independent audit passed. All four trained epochs fail the frozen
selection safeguards; both best files retain the V2 starting tensors. The prior
live-process PID snapshot is historical. Do not restart that failed recipe.
The corrected VM return and checksum have now arrived in
`C:\xampp\htdocs\YEAR 4\Testing\outputs\` from
`~/forensic-dgp/cctv_dgp_vm_bundle/`. The independent audit and fixed24-case native
development comparison are complete. Identity epoch2 passes paired synthetic
guards; native visual review still finds softness/color mismatch and does not
recommend a default. The32 reserved native crops remain untouched.

The matched V3 comparison has now completed452 updates in332.78 seconds and passed
the independent return audit. Allfour changed checkpoints fail the unchanged
selection safeguards; both best files retain the starting V2 tensors. The five
paired preview grids still show unresolved blur/compound detail. No V3 candidate
is eligible for native forwarding or default promotion. Detailed outcomes are in
`C:\xampp\htdocs\YEAR 4\Testing\CCTV_DGP_PERCEPTUAL_RESULTS.md` (intended VM
`~/forensic-dgp/CCTV_DGP_PERCEPTUAL_RESULTS.md` after explicit document transfer).

V4 has now completed on the L4: 40 training cases, 50 autograd traversals,
21.75 seconds and zero updates. The user's local receipt reproduces exactly.
Blur and low-light reconstruction gradients oppose identity in both sources;
this is training-only diagnostic evidence, not a model improvement or Adam
prediction. Outcomes: `C:\xampp\htdocs\YEAR 4\Testing\CCTV_DGP_OBJECTIVE_RESULTS.md`
↔ intended VM `~/forensic-dgp/CCTV_DGP_OBJECTIVE_RESULTS.md` after transfer.

V5 compared identity weight 0.4 against two-objective PCGrad at weight 0.1, retaining
the V2 start, original 902/110 split, postactivation VGG, optimization and guards.
PCGrad raised aggregate degraded PSNR to 16.9750 dB, but source/profile errors and
identity regressions remain. All five paired preview grids were reviewed; severe
blur/compound detail is unresolved. All 550 baseline PNGs match V2 identity epoch 2.
No V5 candidate was sent to native development data or promoted.

A header/hash audit of all 1,012 references found 896/902 training originals below
256 in both axes; FFHQ targets all enlarge 128×128 originals. Next: verify a bounded
sample of genuine higher-resolution counterparts and target suitability, preserving
split roles/provenance, before defining another finite VM pilot. Resolution is a
demonstrated supervision limitation, not proof of cause or a guaranteed fix.
Results: `C:\xampp\htdocs\YEAR 4\Testing\CCTV_DGP_CONFLICT_RESULTS.md`
↔ intended `~/forensic-dgp/CCTV_DGP_CONFLICT_RESULTS.md` after document transfer.
One independent local audit is required for each new returned experiment; the
successful unchanged V5 receipt is reused rather than rerunning the long command.
The assistant can handle future audits after new return files arrive.
Separately, the input-only development audit demonstrated limitations in the
32-pixel decoder minimum and padding-sensitive quality signals. The observed-only
helper is tested but not deployed; no structural gate was calibrated. See
`C:\xampp\htdocs\YEAR 4\Testing\CCTV_INPUT_POLICY_STATUS.md`
↔ intended VM `~/forensic-dgp/CCTV_INPUT_POLICY_STATUS.md` after transfer. The full
local DGP-led workflow, covering-family output and independent final review
requirements remain active.

## Active Goal objective — updated 5 October 2026

The user has applied the revised objective and resumed the app Goal. A fresh
`get_goal` read verifies status `active` with the DGP-first objective below.
The preceding milestone's paused-control limitation is historical. Preserve the
full objective and continue through its milestones; it is not complete.

```text
Develop and verify our own trained DGP as the primary model for useful 256×256 restoration of degraded CCTV face crops for the thesis “Forensic Deep Generative Prior Face Reconstruction for Degraded CCTV Video in Zamboanga City,” in C:\xampp\htdocs\YEAR 4\Testing. Follow SYSTEM_WORKFLOW_AND_GOAL.md and PRACTICAL_OUTPUT_SCOPE.md; later user decisions take precedence over the outdated manuscript. Preserve visible facial structure, accepting some softness; request a clearer crop when usable structure is insufficient. Support one already cropped frontal or mildly turned face in the existing local application, preserving its design. No real Zamboanga CCTV samples exist yet: acquire and audit public native CCTV data before choosing new training, prioritize Asian capture sources where available, report sources separately and do not infer ethnicity or local performance. Proceed through five milestones: (1) freeze native CCTV development and separate labeled evaluation identities, provenance, terms, resolution, overlap limits and input-only review criteria; (2) compare basic resizing, retained Phase 3 DGP and declared pretrained restoration baselines on identical 256×256 inputs, separating raw outputs from display processing, structure preservation and insufficient-information cases; (3) fix demonstrated processing limitations and prepare justified finite VM pilots only when needed, preserving original checkpoints, splits and gate failures; (4) integrate DGP as the main local restorer with automatic restoration selection and a user override; (5) verify useful development outputs, independent final review, meaningful regressions and the full app flow with bundled inline Playwright, updating PROJECT_HANDOFF.md at milestones. Treat real CCTV without an aligned clean reference as unpaired evidence; keep paired synthetic PSNR/SSIM separate. Pretrained restoration models remain declared comparison baselines. Following the user decision on4 October2026, a pretrained face-generating prior with our own trained conditioning layers is also permitted if reviewed output improves; document pretrained and learned contributions separately. A separate completion component may estimate regions hidden by masks, sunglasses, strong lens glare, hands, obstructing hair, scarves or other objects. Show the automatic removal area for optional correction before generation; preserve clear glasses, non-obstructing hair and visible appearance, allowing a small documented margin. Request a less-covered image when too little face remains. Return one plausible estimate alongside the original and mask, with PNG and optional original/mask/result bundle downloads; do not claim exact hidden identity. Report automatic and assisted results separately. Inference and audits may run locally; all actual training must run on the existing NVIDIA L4 g2-standard-4 VM at ~/forensic-dgp. Earlier Google Cloud CLI access was verified, but the user selected verified transfer files and exact pasteable VM commands only. Do not automatically connect, upload, launch, stop or clean the VM. Independently audit user-returned results. All actual training remains manual on the existing L4 VM. Every pilot needs finite updates/epochs, timing and stop rules despite no fixed GPU-hour cap. Do not run immutable historical pilots automatically or repeat failed recipes unchanged. Complete only when the DGP-led local workflow works and reviewed outputs meet the agreed restoration and covering-family scope; archive acquisition or training completion alone is insufficient.
```

## Primary outcome clarification — 3 October 2026

The user reconfirmed that the main thesis outcome is a useful face restoration
model for degraded CCTV imagery in Zamboanga City. Facial covering removal remains
part of the existing Goal and supports that outcome; it must not displace the
restoration work. Keep the confirmed local application, optional mask correction,
one-image result and VM-only training constraints while clarifying CCTV-specific
requirements. This is a priority correction, not cancellation of the existing Goal.

The manuscript at `C:\xampp\htdocs\YEAR 4\Testing\docs\THESIS MANUSCRIPT.docx`
(intended `~/forensic-dgp/docs/THESIS MANUSCRIPT.docx` after transfer) was read for
its title and historical objectives. Its title is "Forensic Deep Generative Prior
Face Reconstruction for Degraded CCTV Video in Zamboanga City." The user states
that the manuscript is outdated and developments since 19 September take
precedence. Historical video input, police-user and multiple-output descriptions
do not override later user-confirmed first-version decisions.

The first CCTV clarification round confirmed:

| Decision | User-confirmed requirement |
| --- | --- |
| Input | Keep one already cropped CCTV face first; full-frame/video upload is outside the first version |
| Local CCTV data | No real Zamboanga CCTV samples are available yet; do not describe public or synthetic examples as locally validated footage |
| Main restoration model | Improve our trainable DGP; pretrained restoration models are comparison baselines |
| Degradation scope | Address all requested CCTV degradations, with degraded-face restoration as the main outcome; the user has not ranked individual degradations or specified input face sizes |

The second CCTV clarification round confirmed:

| Decision | User-confirmed requirement |
| --- | --- |
| Visible appearance | Preserve the apparent facial structure of uncovered regions, accepting some softness rather than sharper generic facial features |
| Insufficient information | Request a clearer crop when extreme blur or tiny face size leaves insufficient usable facial structure |
| Evaluation before training | Obtain a public real-CCTV benchmark before choosing new training |
| Output resolution | Keep 256×256 for the first controlled DGP comparison |
| Review | The assistant reviews development outputs; independent reviewers assess final thesis results |

The final round permits a separate pretrained completion component for covered
regions while DGP remains the main restorer. Prioritize Asian CCTV capture sources
where available, include broader data, and report sources separately. Capture
location/source is not an individual ethnicity label. No Zamboanga-specific
performance or population-representation claim can be established without
appropriate local data. The CCTV workflow questions are resolved sufficiently
to proceed with benchmark acquisition; do not invent input face-size thresholds
or infer actual hidden appearance from these answers.

Evaluate restoration on degraded uncovered faces as its own primary track.
Evaluate degraded covered faces separately through restoration and completion.
Existing covering protocols and their recorded failures remain intact. Obtain
the real-CCTV benchmark, record provenance/terms/splits and freeze the restoration
comparison before choosing new training. The announced broader context-margin
experiment has not been prepared or run; it is supporting work after the primary
CCTV restoration comparison, not the next main-model experiment. Any already
produced detector return may still be audited without selecting a new model.

The first native public CCTV release is now acquired and structure-audited;
see `C:\xampp\htdocs\YEAR 4\Testing\CCTV_BENCHMARK_STATUS.md` (intended
`~/forensic-dgp/CCTV_BENCHMARK_STATUS.md` after transfer). QMUL-SurvFace V1 count
discrepancies and unlabeled distractors are recorded. The 24-case development
comparison and 32-case reserved subset are now frozen; model-output and processing
audits are complete in `CCTV_NATIVE_RESTORATION_RESULTS.md`. The three-case aligned
comparison and 40-case paired camera-stress regression are now independently
audited. The latter finds clear-input smoothing and low-light/structure failures;
neither the retained DGP nor CodeFormer is promoted. Findings and source/profile
metrics are in `CCTV_PAIRED_RESTORATION_RESULTS.md`.

The original candidate pool remains immutable: 1,024 inherited training and 128
validation references, two within-training exact duplicate groups and imperfect
photographs. The derived input gate V2 and independent reconstruction are now
complete. V1's full 112-pixel context assumption is preserved as failed evidence; V2
checks observed facial features. A separate input review excludes two further
training previews with an obstructing hand or facial watermark. Equal-source
truncation produces 902 training references (451/source); all 110 qualified
validation references remain, producing 550 fixed cases. Not all references were
visually reviewed, and these are processed photographs rather than pristine HQ.

The self-contained pilot package and independent archive/pixel audit are complete
at `C:\xampp\htdocs\YEAR 4\Testing\outputs\cctv_dgp_vm_bundle_v1\`
↔ `~/forensic-dgp/cctv_dgp_vm_bundle/` after extraction. It freezes two matched
camera/clear-anchor arms with identity coefficient 0 versus 0.1, 226 updates each
and a 90-minute total runtime cap. Eight contract tests and a bounded two-image
CPU forward check pass, with unchanged states and zero local backward/optimizer
updates. Exact commands are in `C:\xampp\htdocs\YEAR 4\Testing\CCTV_DGP_VM.md`
↔ `~/forensic-dgp/cctv_dgp_vm_bundle/CCTV_DGP_VM.md` after transfer. The package
was the original guarded VM preparation. Its separately versioned InstanceNorm
correction and452-update L4 run have since completed and passed independent audit;
see `CCTV_DGP_PILOT_RESULTS.md`. The historical original failure/package remain
intact. No checkpoint or app default has been promoted.

| Sequential milestone | Current verified state |
| --- | --- |
| 1: Public native CCTV data and review subsets | Acquired/audited; 24 development and 32 reserved crops with disjoint selected labeled identities; reserved outputs untouched |
| 2: Controlled comparisons | Native, alignment and paired camera comparisons audited; V2 trained-candidate native run/audit/visual review complete, without demonstrated useful native restoration |
| 3: Justified model pilot | Historical pilots through V19 r2 audited and closed. Normalization parity fixed; 32-case completion-context ablation audited/reviewed and not adopted. Appearance/preservation failures remain; no new VM pilot or live job confirmed. |
| 4: Primary DGP app integration | DGP-led 256 research route, observed-support Auto/On/Off and operator input/removal review functionally verified. Useful-native qualification and calibrated automatic usability remain open. |
| 5: Final output/workflow verification | 34 app regressions remain verified; 108 covering requests/96 PNGs/32 aliases and the context comparison independently audited/reviewed. Bundled inline Playwright family/rejection/layout/download flow passes. Native usefulness, full automatic/assisted covering scope and independent final review remain incomplete. |

Goal remains active. Archive preparation and training completion do not satisfy
final output quality, covering-family behavior or Zamboanga validation.

Local inference preparation is documented in
`C:\xampp\htdocs\YEAR 4\Testing\DGP_INFERENCE_READINESS.md`
↔ `~/forensic-dgp/DGP_INFERENCE_READINESS.md` after source/document transfer.
Seven adapter checks and four native-candidate selection checks pass, with eight
native forward comparisons exactly matching the Phase3 baseline and zero local
backward/optimizer updates. This is fidelity evidence, not a newly useful model.
The frozen24-case native candidate plan excludes the32 reserved images and caps
post-return forwards at48. The V2 trained-candidate native run used24 forwards;
its independent audit and separate visual ledger are now complete. Its output
does not qualify for the default. The main application and frozen original bundle
remain. V3 execution is audited and rejected; V4 is audited diagnostic evidence.
The separate additive V5 comparison is running on the L4; final audit/review is pending.

| Location | Windows local workspace | Linux training VM |
| --- | --- | --- |
| Repository | `C:\xampp\htdocs\YEAR 4\Testing\` | `~/forensic-dgp/` |
| This specification | `C:\xampp\htdocs\YEAR 4\Testing\SYSTEM_WORKFLOW_AND_GOAL.md` | `~/forensic-dgp/SYSTEM_WORKFLOW_AND_GOAL.md` after transfer |
| Ongoing status | `C:\xampp\htdocs\YEAR 4\Testing\PROJECT_HANDOFF.md` | `~/forensic-dgp/PROJECT_HANDOFF.md` after transfer |
| Product scope | `C:\xampp\htdocs\YEAR 4\Testing\PRACTICAL_OUTPUT_SCOPE.md` | `~/forensic-dgp/PRACTICAL_OUTPUT_SCOPE.md` after transfer |
| Evidence | `C:\xampp\htdocs\YEAR 4\Testing\outputs\` | `~/forensic-dgp/outputs/` or the named experiment bundle |

## Objective

Develop and verify our trained DGP as the primary model for useful 256×256 face
restoration from degraded CCTV crops, targeting the Zamboanga City thesis use case.
Address low resolution, blur, compression, low light/noise and other CCTV
degradations while preserving visible facial structure; some softness is
acceptable. Request a clearer crop when usable facial information is insufficient.
Obtain a public real-CCTV benchmark before choosing new training, prioritizing
Asian capture sources where available and recording broader sources separately.

Deliver this restoration in the existing local application for school staff and
thesis researchers, using one already cropped frontal or mildly turned face.
When facial regions are covered, preview the automatically detected removal
area and allow manual correction before conditional completion. A separate
pretrained completion component is permitted; pretrained restoration models are
comparison baselines for the DGP contribution. Target masks, sunglasses, strong
lens glare, hands, obstructing hair, scarves and other objects over the face.
Preserve ordinary clear glasses, non-obstructing hair and visible appearance.
Permit a small, documented surrounding skin margin for complete removal and
request a less-covered image when too little face remains.

Select blur/noise restoration automatically using the input, with a user override.
Show the original, removal area and one final output. Provide a final-image download
and an optional bundle containing the original, removal mask and result. Prioritize
useful output over speed. Hidden features are estimates; no reference photograph
of the same person is required and exact hidden anatomy is not an acceptance test.

## Execution boundaries

1. Run inference and the application locally. Small pilots estimated at 3–5
   minutes on the L4 may train locally when local timing/memory make that practical.
   Larger pilots around 30 minutes use the user's existing NVIDIA L4 /
   g2-standard-4 VM (4 vCPUs, 16 GB RAM); tell the user when the VM is needed. There is no
   fixed GPU-hour limit in the user's answer; every experiment still needs a
   concrete update/epoch budget, timing and an evidence-based stop rule.
2. Start with existing data. Add suitable public research datasets when necessary;
   record sources, terms, duplicates and split membership before use. Asian-source
   inclusion does not establish Philippine population coverage.
3. Preserve the main application's design. First-version input is a cropped face,
   with one output. Full-scene detection, multi-face/video processing and full
   profiles are outside this first-version workflow.
4. Preserve old experiment files, hashes, data splits and gate outcomes. Version
   the new practical review protocol before evaluation. A deliberate removal
   margin is separate from the raw occlusion label and detector score.
5. Use bundled inline Playwright for browser verification. Keep milestone notes
   in `PROJECT_HANDOFF.md`. There is no configured SSH connection in this chat;
   prepare exact VM commands and verified transfer files when training is needed,
   then audit the returned results before selecting a checkpoint.

## Five sequential milestones

### 1. Freeze a practical review gallery

Start with a public benchmark of native CCTV face crops; audit its original
camera/source provenance, access terms, references and published split before
inference. Keep public CCTV development and final evaluation identities separate.
Do not train on the final benchmark or rename inspected development samples as
an unseen holdout. Without an aligned clean reference, real CCTV cannot supply
ground-truth restoration PSNR/SSIM; synthetic paired cases remain a separate
quantitative regression track. Same-person photos from another capture are
appearance/identity references, not exact clean pixel targets.

Audit existing native examples by covering family and pose without changing their
labels or splits. Select a fixed development/review gallery with clear/degraded
inputs and uncovered/transparent-glasses controls. Include every requested covering
family before declaring whole-scope readiness. Record missing families explicitly;
prepare additional data if needed. Keep previously trained or inspected cases
marked as such; they cannot become a pristine holdout by renaming them.

Freeze case IDs, source hashes, pose, covering type, allowed removal margin and the
five review criteria below before generating new comparison outputs. Save a source
contact sheet and machine-readable coverage report. Do not infer hidden-face
ground truth from a covered photograph or invent paired clean targets.

### 2. Identify the cause of poor output

Compare unmodified degraded CCTV, a basic resize baseline, the retained DGP and
declared pretrained restoration baselines on identical frozen crops at the
agreed comparison resolution. Inspect structure, useful clarity, visible
appearance, artifacts and insufficient-information cases before choosing a
training recipe. Record raw model output separately from display/postprocessing.

On the same frozen cases, compare current automatic masks with reviewed/corrected
removal masks through the pretrained completion baseline. Compare restoration
off/on only on declared degraded cases and controls. Inspect covering remnants,
plausible facial structure, visible appearance and seams. Report automatic versus
assisted completion separately. Save masks and outputs so every conclusion has
inspectable evidence. Select changes that address the observed failure: detection,
mask construction, alignment, completion or visible restoration.

### 3. Apply justified improvements

Implement needed processing fixes first. Make restoration routing depend only on
the uploaded input and user override. Mask expansion must follow a frozen policy
and remain distinct from raw detector output. Do not apply visible restoration to
clear inputs indiscriminately or assume it improves generated regions.

When inference fixes do not address an observed model failure, prepare a bounded
VM pilot with fixed splits, one-batch preflight, baseline comparison, visible-face
regression checks and explicit checkpoint-selection rules. Train only on the VM.
Use reviewed real masks as detector supervision; completion targets must come
from clean images with valid synthetic occlusions or genuine paired data. Do not
train a completion model to reproduce a real covering as hidden facial truth.
Benchmark pretrained completion before committing to generator retraining.

The existing native-lens pilot archive is an immutable candidate, not a command
to run automatically: assess its relevance after the practical baseline comparison.
Do not repeat an unchanged failed experiment. Export results with an LF checksum;
independently verify source, budget, metrics and images on return.

### 4. Integrate the validated workflow in the existing application

The primary restoration path must use the validated DGP checkpoint. Conditional
completion may use a separately identified pretrained component. The existing
CodeFormer visible-restoration route is comparison evidence, not evidence that
the DGP model has improved; switch the combined app only after the DGP-focused
comparison supports the change.

Connect covering detection and completion to the main app. Preserve the current
design while adding review/correction of the removal area before generation,
automatic restoration with an override, one output and the requested downloads.
Provide a clear request for a less-covered image when facial evidence is inadequate.
Confirm the local runtime works: installed RTX 3050 hardware currently coexists
with CPU-only project PyTorch, so CUDA inference is not yet configured. Measure
actual local processing time and use a compatible runtime rather than claiming
the GPU is being used merely because it exists.

### 5. Verify usefulness and record readiness

Review the frozen public CCTV restoration gallery as the primary output test.
The assistant reviews development outputs; independent reviewers assess final
thesis results. Record incomplete independent review explicitly rather than
claiming it has occurred. Preserve facial structure even when sharper alternatives
look more photorealistic, and test requests for clearer crops separately from
the existing near-total-covering guard.

Run meaningful processing/API regression checks and Playwright at 375, 768 and
1280 pixels, including the upload, mask correction, restoration override, generate
and download flow. Inspect a ten-row preview grid and the complete fixed gallery.
Record useful/failed cases per covering family and restoration setting, including
manual corrections. Keep the Phase 3 restoration baseline unless a new candidate
has measured benefit and no material visible-appearance regression.

Update the handoff with chosen artifacts, hashes, reproduction commands, runtime,
remaining limitations and the next step. Mark the Goal complete only when the
agreed existing-app workflow works locally and reviewed outputs demonstrate its
usefulness across the fixed requested-family gallery. A training run finishing,
a lower aggregate mask error or a checkpoint named `best.pth` does not meet this
completion condition by itself.

## Primary restoration review, frozen before inference

1. Preserve the apparent visible eyes, nose, mouth, face contour and appearance;
   sharper invented features do not qualify as a DGP improvement.
2. Improve useful clarity on inputs with readable structure. Keep insufficient
   information cases explicit and request a clearer crop in the product workflow.
3. Report native dimensions, pose and source limits. Frontal/mild crops are the
   first supported scope; full profiles remain diagnostics.
4. Save raw model output separately from enhancement. Native unpaired CCTV has
   no aligned clean target; do not derive clean-reference PSNR/SSIM from its input.
5. Use development review to choose experiments and independent reviewers for
   final thesis results. Do not tune on the reserved evaluation subset.

The native selection, input-only review and frozen comparison are recorded under
`C:\xampp\htdocs\YEAR 4\Testing\outputs\cctv_native_development_v2\` and
`outputs\cctv_native_comparison_v1\` (intended Linux counterparts under
`~/forensic-dgp/outputs/` after transfer). The partial V1 selection remains intact;
V2 corrects the discovered `.jpg` filename/PNG-content assumption without changing
the size ranges or original image bytes. Reserved sources have header/hash audits
only and remain outside development inference.

## Supporting covering review, frozen before inference

1. The covering is substantially removed within the intended facial area.
2. Generated features are plausible; exact hidden appearance is not required.
3. Visible features retain their appearance outside the documented removal margin.
4. The estimated region joins the surrounding face without conspicuous covering
   remnants, blur patches or seams.
5. Clear faces and ordinary transparent glasses avoid unnecessary generation;
   detection failures permit manual correction and near-total coverage requests
   a less-covered image.

Report these judgements per case and covering family. Record whether the result
used an automatic or manually corrected mask. Review of already trained/inspected
examples is developmental evidence, not an unbiased accuracy estimate. Preserve
earlier numerical gate failures and report new regression metrics alongside the
practical review. Do not label the system broadly ready when a requested family
has no reviewed examples or has only failing outputs.

## Historical covering-track starting state — 2 October 2026

This historical snapshot predates the combined app integration and the confirmed
CCTV-first priority. Current status is at the top of `PROJECT_HANDOFF.md`;
the next primary action is the native DGP comparison, not the older action below.

`app.py` provides the existing Phase 3 restoration app; `completion_web.py`
provides a separate experimental mask/completion page. Pretrained CodeFormer
inpainting and Phase 3 weights exist locally. They have not yet been integrated
into the confirmed combined workflow. Prior native mask experiments do not prove
all-covering output quality. The supported reviewed dataset has 115 records
(83 train / 25 validation / 7 previously inspected test); 68 training records lack
a covering-family tag. Untagged does not mean the covering is absent.

The descriptive 83-source training inventory and source contact sheets are complete
without label/split changes. A ten-source native comparison is frozen and audited;
see `PRACTICAL_NATIVE_OUTPUT_RESULTS.md`. Reviewed masks help several mouth/hand
cases, but dark sunglasses and white glare still fail. Full-family/degraded
coverage and the agreed main-application integration remain incomplete.

Generic Places2 LaMa and face-specific AOT-GAN comparisons are now audited in
`PRACTICAL_LAMA_RESULTS.md` and `PRACTICAL_AOT_RESULTS.md`. Neither replaces the
CodeFormer baseline. Reviewed-mask results still have difficult eyewear/glare and
mask-edge remnants; approximate labels do not guarantee a complete removal area.

Historical action: source-review full covering footprints and freeze revised
operator proposals in a new version before a targeted CodeFormer/AOT comparison.
Preserve old labels and results. Prepare missing-family public data and compare
visible restoration on declared degraded inputs. No local training.
