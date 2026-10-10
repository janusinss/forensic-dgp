# Latest transfer fix — 10 October 2026: checksum line endings

The user reported both bank-comparison uploads completed. Step 3 stopped at
`sha256sum` because the Windows-produced checksum ended in CRLF; GNU Linux
read the CR as part of the archive filename. The `&&` chain stopped before
extraction. This is an installation-format failure, not a training/gate failure.
VM execution and installation success have not been independently observed.

The local checksum now contains exact ASCII LF bytes (111 bytes instead of 112).
The builder explicitly writes LF; the transfer checker now validates raw bytes
rather than Python's normalized text. Step 3 filters CR from the uploaded checksum
stream before verification, preserving that original uploaded file. No archive
rebuild, training-recipe change, optimizer/gradient call or VM launch occurred.
Archive SHA256 ece56acfdf440fc1ced9f5d9d53ba39db8c3a4da48fa0b57bee5346d265297a0
and protocol SHA256 0e4f9645abc99f6d980f3b20240b196f65f3e130472795faa2f964df1a5a3b3e
remain unchanged. Preserve original source/checksum/doc bytes and the old transfer
audit in outputs/cctv_dgp_bank_checksum_lf_repair_v1/before and the existing audit
paths. The original audit's successful content checks did not prove LF format.

Manual continuation: replacement step 3 in CCTV_DGP_BANK_COMPARISON_V1_VM.md;
steps 4–5 remain manual tmux. CUDA/learning checks and returned-results audits
remain pending. The full goal is active and unqualified.

---

# Latest maintenance — 10 October 2026: verified local output backup and VM storage cleanup

The user requested VM space and a local backup before a later VM change. The
existing L4/g2-standard-4 instance was freshly identified; no VM was switched,
started, resized or stopped. New actual training remains manual in tmux.

Four hash-matched redundant home archives and 76,973 singly-linked inactive image
output copies from seven historical runs were removed after complete Windows
backup audits. Net increase in free space during maintenance: 13.170 GiB.
Fresh guest free space: 20.641 GiB, from initial 7.471 GiB.
Conservative free space after the ready bank-packet installation reservation:
19.045 GiB; its 16 GiB requirement is satisfied.
The current bank packet is not installed and has not run. This net space reading
includes storage used by new plans and receipts, not just deleted allocations.

All checkpoints/full states, dedicated research/training caches, inputs/splits,
source/instructions and gate/failure/log records remain on the VM. 82 non-image
gradient/optimizer files were explicitly retained and separately hashed.
The independent audit compared 394,076
retained metadata entries and 4,289
critical/evidence byte hashes. Only direct output-parent directory times/size
changed besides the exact planned image files. Scientific cache bytes were not
all rehashed; unchanged metadata and disjoint singly-linked removal are recorded.
Local app/packet bindings and the permanent checkpoint SHA256
646fbb11674e8882ea41e251d6e6e7d468c963979b48c93132def8292834e24b are unchanged.

Portable backup: outputs/cctv_dgp_vm_output_offload_20261010_v1_r2/
dgp-image-output-recovery-20261010-v1.tar, 12,579,983,360 bytes,
SHA256 5fe41326fa0ccd6399babf18f4f45582b0fa7d9815d506b998ca29fa637a4c1c.
All 76,973 payload members and 127,334 full-return bindings passed independent
verification. Complete local returns remain too. The earlier 4,431-file / 36.739 GiB
cache backup remains present with its frozen sizes and prior full-hash audit.
This output archive is not a complete boot-disk/environment backup.

Evidence: outputs/cctv_dgp_bank_archive_cleanup_v1/independent_audit.json;
outputs/cctv_dgp_vm_output_offload_20261010_v1_r2/independent_backup_audit.json,
independent_cleanup_audit.json, tail_recovery_v1/remote_receipts and inventory_after.json.
Recovery: VM_IMAGE_OUTPUT_BACKUP_RESTORE_20261010_V1.md.
Manual next run: CCTV_DGP_BANK_COMPARISON_V1_VM.md, unchanged verified packet.
Use a freshly identified/verified destination before any later VM migration.

The original 900-second removal task stopped during retained-file hashing after
all deletions. Its failed receipt/traceback remain. A separate finite audit-only
task verified the complete deletion ledger and exact retained state with zero
additional removals. The independent local audit passed; the original execution
remains recorded as timed out. Immediate pre-removal free bytes were not saved,
so only ledger allocation and measured interval/net space changes are claimed.

The original local selector failure was retained before the corrected preparation;
the user chose to store complete verified image packs locally. V41 optimizer
step NPZ files were identified from their actual contents and stayed on the VM.
No historical pilot was relaunched, gate changed or model adopted. Maintenance
ran zero model/gradient/training calls. The full goal remains ACTIVE and
incomplete; useful restoration and all seven completion families still require
their agreed independent reviews. Earlier handoff sections remain historical.
Exact pre-maintenance handoff bytes are in this offload's documents directory.

---

# Latest preparation — 10 October 2026: matched generative-bank comparison ready for manual VM preflight

The active comparative goal and the user's "apply the best approach here"
choice are implemented as two independent finite routes. No quality gain or
model adoption is claimed. The application and permanent checkpoint remain
unchanged: SHA256646fbb11674e8882ea41e251d6e6e7d468c963979b48c93132def8292834e24b.

A trains the parity-preserving repaired current reconstruction with active
clear/blur anchors:15 selected tensors/609,219 elements. B retains that path
and adds our input/style and spatial conditioning/fusion around the declared
frozen standalone StyleGAN2 prior:37 tensors/3,364,643 trainable elements.
The external generator is not our own trained generator, and no pretrained
restoration encoder/checkpoint is substituted. Its FFHQ person overlap remains
unexcluded. The bank's executed restoration feature path stops at256.

Completed local evidence:230.35s initialization proof on100 paired TRAIN plus
24 native unpaired DEV inputs, exact original/A/B output parity, zero updates.
The20.34s independent audit verified480 files,11 sources,124 raw/PNG
compositions,66 sheet cells, mean-style recomputation and four fresh CPU model
triplets. All three sheets/66 cells were viewed; inherited softness remains.
Internal conditioned features vary with the input; this is not learned quality.
Both local gradient-enabled model APIs were refused.

Prepared archive: outputs/cctv-dgp-bank-comparison-v1-execution.tar.gz,
809,058,558 bytes, SHA256ece56acfdf440fc1ced9f5d9d53ba39db8c3a4da48fa0b57bee5346d265297a0.
Protocol SHA2560e4f9645abc99f6d980f3b20240b196f65f3e130472795faa2f964df1a5a3b3e.
The84.28s independent transfer check verified5,611 archive members/5,610
assets, actual zero-neural transfer CLI and five adverse protocol refusals.
Original embedded return checker is preserved. The portable local checker adds
only bounded metadata roundoff allowances; scientific gate AST is exact and
recomputed original decisions must agree. Real returned outputs remain pending.

Manual run only: NVIDIA L4/g2-standard-4, existing retained venv, exact new root
~/forensic-dgp/cctv_dgp_bank_comparison_v1_vm, human tmux. Require16GiB free
after install; fresh current guest space/GPU state was not inspected here.
Each route starts independently,50 updates/250 reference exposures/1,250
profile-image exposures,0 full epochs.125 TRAIN references from each source.
Full baseline/A50/B50 checks cover3,905 paired TRAIN outputs plus24 separate
unpaired native DEV outputs. Bank-disabled ablation is fixed100 TRAIN+24 DEV.
Final pixels are excluded. Source labels are not ethnicity/capture claims.

Before optimizer: CUDA initialization parity, saved CPU/CUDA prior fixture,
true derivative and timing/resource projections. B conditioning starts with
expected zero gradients under zero fusion; after update1, nonzero finite
conditioning gradients are required before update2. Worker90min, fit10min per
route, snapshot20min, preflight5min, export10min,20GiB peak,1GiB reserve,
6GiB output cap, independent external deadlines. Failed gates/states are kept;
no failed resume, extra epochs, automatic continuation or app promotion.

Commands: CCTV_DGP_BANK_COMPARISON_V1_VM.md (Windows gcloud upload, VM install,
tmux launch and three separate PuTTY-compatible SCP downloads).
Evidence: outputs/cctv_dgp_conditioned_bank_initialization_v1_independent_audit.json,
outputs/cctv_dgp_conditioned_bank_initialization_v1_visual_review.json,
outputs/cctv_dgp_bank_comparison_v1_transfer_audit.json and
outputs/cctv_dgp_bank_comparison_v1_portable_checker_review.json.
Return audit: scripts/audit_cctv_dgp_bank_comparison_return_v1_portable.py.
Review: scripts/review_cctv_dgp_bank_comparison_v1.py;20 paired+6 native sheets,
all available planned sheets must actually be viewed. Missing stopped outputs
are declared. No VM connection, gradient, backward or training occurred here.

Goal service remains ACTIVE. All five milestones, independent final review,
seven separate covering families, automatic/assisted completion evidence,
full app/inline Playwright qualification and useful reviewed outputs remain
required. This preparation does not complete a quality milestone or the goal.
Exact pre-update handoff bytes are retained in
outputs/cctv_dgp_bank_comparison_v1_documents/PROJECT_HANDOFF.before.md.

---

# Latest decision — 10 October 2026: explicit generative-prior route selected

The latest human method answer is “apply the best approach here.” The active
/goal already requires a justified generative-prior role and a two-route
comparison; this decision specifies that role in the canonical goal/scope/plan.
The goal service was verified ACTIVE in this continuation. The older paused
goal-service statement below is historical.

Current app/checkpoint is unchanged: SHA256
646fbb11674e8882ea41e251d6e6e7d468c963979b48c93132def8292834e24b.
Fresh code/state inventory confirms a DeblurGAN-v2-compatible MobileNet/FPN
conditional restorer,181 unique parameter tensors/3,312,707 elements,622 state
entries including aliases/buffers. There is no separate face-generating prior
in its executed forward. The class name remains for compatibility.

Route A: repaired current-DGP copy with always-active clear/blur RGB target
anchors. Route B: retain useful camera feature/reconstruction weights and
train our conditioning/reconstruction around a declared frozen continuous
StyleGAN2 face bank. This is a DGP adaptation, not a published-DGP reproduction
or an entirely own-trained generator. No pretrained restoration encoder is
selected. Original gates and failed recipes are retained.

The official standalone prior was downloaded in26.68s; source/licence/release
metadata and204,535,545 bytes are retained under
outputs/cctv_dgp_generative_bank_source_v1.
SHA25605f5d33d79b32a3355cae3ede30e7ee06a90e56c60b1c2efe4ddd0d0e5a2959f.
Its external FFHQ source overlaps our FFHQ TRAIN source; exact pretrained
identity overlap is unexcluded. It is not an Asian CCTV capture source.

A24.32s local probe used six prior forwards on two generated seeds plus a
repeat. Zero CCTV/clean photographic/final inputs, gradients, backwards or
optimizer updates. The7.56s independent checker verified31 outputs/11 sources,
14 NumPy operator fixtures, exact CPU generator/five-feature replay, original
source bodies and seven protected hashes. Both cells of the generated-seed
sheet were inspected at original512x256. These are generated faces, not CCTV
restoration, own-trained output or hidden-identity evidence.

Next prerequisite: implement the conditioned replacement copy and matched
returned-output checker; verify initializer/raw/PNG parity, bank contribution,
copied/frozen/trainable paths and finite L4 resource projections before allowing
any optimizer. Then freeze matched manual comparison/transfer/tmux commands.
No new executable training packet or ready launch command exists yet. No VM
connection, training, app adoption or historical pilot launch occurred here.

Full milestone/covering-family/final qualification remains incomplete.
Report: CCTV_DGP_GENERATIVE_PRIOR_METHOD_REVIEW_V2.md.
Selection/evidence: outputs/cctv_dgp_method_comparison_v2/selection.json and
outputs/cctv_dgp_generative_bank_probe_v1_independent_audit.json.
Exact pre-update documents and prefix verification are retained in
outputs/cctv_dgp_method_comparison_v2/document_update/.

---

**Goal revision — 10 October 2026: bounded comparative DGP improvement is now explicit. No new training or model promotion.**

The user requested a goal update implementing the best evidence-supported
approach and continued improvement beyond the current comparison model. The
preceding goal already allowed evidence-justified training; it did not explicitly
require a corrected-current versus replacement-design comparison and an accepted
incumbent for subsequent improvements. These decisions are now canonical at the
start of SYSTEM_WORKFLOW_AND_GOAL.md and aligned with PRACTICAL_OUTPUT_SCOPE.md.
The current preparation plan is
[CCTV_DGP_COMPARATIVE_IMPROVEMENT_PLAN_V2.md](<C:/xampp/htdocs/YEAR 4/Testing/CCTV_DGP_COMPARATIVE_IMPROVEMENT_PLAN_V2.md>).

The starting app checkpoint has been freshly hash-verified as
`646fbb11674e8882ea41e251d6e6e7d468c963979b48c93132def8292834e24b`.
Keep this permanent baseline and compare future accepted development incumbents
against it. The replacement route preserves useful weights where compatible and
must document its DGP role; training every weight from scratch is a separate
feasibility decision. Pretrained restorers remain declared comparators.

The latest executed multiscale study remains rejected: twelve independent
one-update arms, zero complete epochs, all64 sheets reviewed. The full decoder
can improve the TRAIN filter proxy by1.170–1.884%, but clear pixel error worsens
24.8–31.5times and visible tones change. Current learning evidence justifies
reviewing active reconstruction supervision; it neither proves the architecture
irreparable nor validates a replacement. Read the unchanged
CCTV_DGP_POST_MULTISCALE_TRAINING_REVIEW.md as prior evidence, not an automatic
eight-arm launch. Existing packet hashes, scientific failures and stops remain.

Next prerequisite: finish a comparison-design review that addresses loss,
realistic degradation coverage, numerical learning paths and one viable new DGP
reconstruction route; freeze a finite matched protocol and independent return
checks before verified transfers/manual commands. No replacement architecture,
new executable training recipe or transfer packet has been completed by this
goal edit. Additional epochs remain conditional on passing finite checks and
useful development preservation;1/2/5 is an assistant-proposed schedule.

Accept useful preserved development improvements before changing the incumbent;
retain rejected versions and rollback choices. Continue justified finite stages
within resource/deadline limits, with plateau/regression stops and a design
decision after three unsuccessful attempts. No unchanged failed-state resume,
automatic historical pilot or lowered gate. Final identities stay outside tuning.
Independent final review, five milestones, DGP-primary app flow, seven automatic
and assisted completion families and inline bundled Playwright remain incomplete.

Byte-exact before copies and fingerprints are under
`outputs/cctv_dgp_goal_update_20261010_comparison_v2/`.
The prior training/review reports are protected and unchanged. This revision
uses no neural, gradient or optimizer calls and makes no VM connection or app
change. The goal service was read as paused; its tool has no objective-edit or
resume operation. This is a saved project-goal revision, not a claim that the
desktop goal card was rewritten or resumed. Later human decisions and the
canonical files govern subsequent authorized work.

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


**DGP milestone - 10 October 2026: group-conflicts return fully audited/reviewed; finite-guard R1 verified for manual L4 execution. Full goal incomplete.**

The group-conflicts returned archive (608,243,422 bytes; SHA256
701cfd773d48f69cdbf5a933fd453db11bcd9ac3839de0fc88f7b50592e30dfb) passes
its frozen prospective checker: 1,624 regular members, 400 raw/PNG records,
32 saved aggregate gradients, three reset placements and 12 unchanged scientific
decisions. Eighty fresh CPU inference cases agree within declared tolerances
(raw maximum 2.272427e-6; PNG one byte; embedding 2.831221e-7). All 20 exact-cell
pages and 400 unique outputs were actually visually reviewed; the independent
pixel checker verifies 600 cells and 20 observations. This is primary-assistant
development visual review, not independent final review. Earlier audit flags
stating visual_review_pending are closed by the later bound visual review.

All 12 quality decisions fail. Smallest delivered gains are 0.034730% and
0.018856% on two exposed photographic TRAIN cohorts; the cross-cohort has four
delivered preservation failures. Larger steps worsen some finite losses despite
negative initial derivative predictions for every sampled group. The largest
loses structure in both cohorts and introduces tone/canvas-edge defects. These
findings justify finite-step checks and TRAIN coverage, not unchanged recipes
or more epochs. See CCTV_DGP_GROUP_CONFLICTS_V1_RESULTS.md. The 1% early quality
requirement and V42's full-corpus 0.00692364% failure remain binding. No model is
promoted and no trained epoch or parameter update occurred in that diagnostic.

The AGENTS circuit-breaker question was asked after three unsuccessful designs;
the user selected the best approach. The new finite-guard R1 study recomputes
64 separate cohort/source/profile objectives at each state, including both
50-case TRAIN cohorts (100 total). The formerly TRAIN cross-check cohort now
also guides fitting; its history is retained and no independent-generalization
claim is allowed. At most three actual accepted training changes, nine tested
placements and 960 autograd queries are permitted. Original and preceding-state
raw AND delivered preservation/structure/source/brightness checks determine
mechanics acceptance. The original 1% requirement is measured separately for
every state; microchange acceptance is not model qualification or an epoch.
Rejected outputs/decisions are retained. No resume, automatic continuation or
failed-pilot launch. A later 50-update/full-corpus experiment requires separate
justification and the existing preservation gates.

R1 is prepared and verified, not run. It contains 171 regular archive members,
53 local source bindings and 20 Python3.10-parsed sources. Four CPU parity cases
have maximum raw error 2.205372e-6; original/copy/buffer states are exact. Four
worker modes and three candidate local derivative/trial guards reject execution;
pure arithmetic and preservation-rejection fixtures pass. The prospective
return checker is frozen before launch. Individual autograd queries/aggregation
are not replayed locally; saved group-vector/certificate arithmetic, decisions,
trained-state contents and fresh CPU outputs will be independently checked.

Protocol: efdd62759213136114a56f0aaa256278753cac8bac4bc6c6a52f1f8b7550598f.
Execution: 190,367,690 bytes; SHA256
4a620dd41d96cedfef25b0ed7bbd82ac362f0e867c3ab4ca656642bb1c74870c.
Guide: CCTV_DGP_FINITE_GUARD_V1_R1_VM.md.
Design: CCTV_DGP_FINITE_GUARD_V1_R1_DESIGN.md.
Audit: outputs/cctv_dgp_finite_guard_v1_r1_preparation/independent_packet_audit.json.
The earlier V1 packet is superseded: a redundant derived-PNG storage projection
exceeded the 3GiB cap. Its packet, sources, audit and storage failure are retained.
R1 reconstructs only the preceding-anchor mean-only diagnostic PNG from exact
saved raw pixels; all primary raw/PNG outputs and both scientific anchor metrics
remain. R1's guide-publication filename collision is also retained; a distinct
guide was published without changing the frozen packet or prior guide.

Estimated R1 runtime 10-20 minutes plus 1-6 minutes export, extrapolated from
prior L4 timings; new 64-group timing is unmeasured. Require 7GiB free after
installation. Enforced worker1800s/external1830s, per-state derivatives240s,
solver120s/proposals180s, cache120s, export360s/external390s, 30s kill grace,
20GiB allocated VRAM, 3GiB retained return and 512MiB reserve. Dynamic timing
and conservative storage stops remain mandatory. No cleanup is bundled.

The user selected the existing NVIDIA L4 g2-standard-4 VM. A read-only Google
Cloud API check verifies instance identity in us-central1-a and reports
TERMINATED. No guest disk/workload inventory is available, no VM was started,
no files were deleted and no training was launched. Maintenance remains
authorized only through a live inventory and exact retained-backup/hash-bound
plan preserving research assets, active work, original checkpoints, splits and
gate failures. Start the existing VM manually before transfer; if space is
insufficient, obtain a live inventory before any deletion or launch. All actual
learning remains the user's manual tmux/verified-transfer workflow.

Separate covering-detector score review is also closed. The 36 review-mask calls
on 18 photographs and their fixed synthetic degradations run only the retained
segmenter; no DGP/completion-generator forwards or training occur. All 36 current
proposals are exact, 174 source bindings and 153 artifacts are audited, and all
nine pages/36 cases are actually reviewed. Four input-only exclusions stay held.
Post-hoc score witnesses show an inversion between one fixed covering-core pixel
and one protected-appearance pixel in each of 28 eligible covering fixtures.
A single pointwise threshold cannot fix those witnesses. Approximate assisted
footprints are not expert ground truth; the '_native' fixture suffix means
original photograph, not native CCTV. Automatic proposals remain unqualified;
no detector, threshold, margin or completion generator is promoted. See
CCTV_DGP_AUTOMATIC_PROPOSAL_SCORE_V1_RESULTS.md. The recorded numerical bounds
and later review audit close creation-time pending flags, not quality failures.

Retained DGP/Phase3 checkpoints and app design remain exact. Native QMUL evidence
remains unpaired: no convincing useful current-DGP gain on its six core cases;
seven insufficient cases remain insufficient and 32 reserved final pixels are
unviewed. Photographic/synthetic TRAIN findings are not native CCTV or Zamboanga
performance. Source labels do not establish ethnicity. All five milestones,
paired DEV preservation, useful native DEV outputs, independent final review,
seven automatic/assisted covering families and bundled inline Playwright app
flow remain required. Preserve visible appearance, clear glasses and ordinary
hair; request clearer/less-covered crops when insufficient; return original,
mask and one plausible estimate with PNG/bundle downloads and no hidden-identity
claim. Thesis deadline around/after 20 November 2026 and the user's manual VM/
credit-migration decisions remain unchanged. Full goal active/incomplete.
This prefix supersedes historical pending-return statements below; exact prior
bytes and status hashes are retained in outputs/cctv_dgp_group_conflicts_v1_closure/.


**DGP diagnostic milestone — 10 October 2026: loss-balance return audited and visually reviewed; group-conflicts diagnostic verified for manual execution.**

The downloaded original-loss-balance return is verified:869,859,156 bytes,
SHA256 a18bf64a1db7abac2aaa7885aa8882463f59415841681d273722fe45433d0e2b.
The independent checker verifies all4,030 regular members,1,000 raw/PNG metric
records, nine saved-vector trial formulas and36 unchanged scientific decisions.
Two hundred fresh CPU inference cases meet the prospectively declared bounds:
raw2.712012e-6, PNG1 byte, embedding3.091991e-7. All60 exact-cell pages and
1,000 unique outputs are visually reviewed as primary-assistant development
evidence; a separate checker verifies1,800 exact256-pixel cells and20 notes.
This is not independent final review. Earlier creation-time pending flags are
closed by the later hash-bound visual_review.json.

All nine trials remain unqualified. The strongest ratio-2 trial has delivered
structure gains1.386497% and1.099287% in the two small photographic TRAIN cohorts,
but13 and12 delivered preservation failure records. Clear delivered MSE worsens
about9.67% and8.73%, with clear SSIM/recognition regressions too. Balanced outputs
remain soft; strong feature-only steps wash out clear detail. Failure records
count group/metric entries, not people or identification mistakes. Raw failures
persist, so PNG processing alone does not explain the limitation. All100 baseline
raw/PNG outputs and six matched decoder portions equal the prior pure-decoder
controls. Mean identity protection does not protect every source/profile output.

V42's full3,905-case0.00692364% structure failure remains binding. These100 TRAIN
cases do not establish useful native CCTV, paired DEV or final performance.
Current and original checkpoints retain their exact hashes; no optimizer,
epoch, trained checkpoint, app change or completion generation is added.

The user selected the evidence-based approach before training. A distinct
group-conflicts diagnostic measures32 source/profile losses with160 bounded
manual-VM gradient queries and zero optimizer updates/epochs. It conditionally
tests three reset displacements only if every sampled group has a certified
local descent direction. No certificate means export without trials, not proof
of global infeasibility. All17 raw/PNG source/profile scientific groups retain
the existing structure, MSE, SSIM, recognition and brightness requirements.
The SSIM derivative is a declared surrogate; finite scientific checks are not
replaced. Aggregate gradients and metadata are exported, but individual query
vectors/autograd are not independently recomputed locally; saved-vector
arithmetic and fresh forward outputs are audited on return.

Manual guide: CCTV_DGP_GROUP_CONFLICTS_V1_VM.md.
Design: CCTV_DGP_GROUP_CONFLICTS_V1_DESIGN.md.
Reviewed result: CCTV_DGP_ORIGINAL_LOSS_BALANCE_V1_RESULTS.md.
Packet audit: outputs/cctv_dgp_group_conflicts_v1_preparation/independent_packet_audit.json.
Protocol: bc0dcbd92b876c53ebe4f484cc5743fad7c8e7cdce6ca8222e04de3d485ec4b2.
Execution archive:220,267,830 bytes; SHA256
cfe829808d9280744d45633bcdfb9bcdd0dbc6a70549d6f86d0a73acf2f96ff5.
Independent preparation verifies174 archive members,39 source bindings,
19 Python3.10 AST files, four mathematical fixtures,100 cached loss checks and
four initial CPU parity cases. All four local worker modes and three local
candidate derivative/trial guards refuse execution. States/buffers stay exact.
The first loss target-rounding parity failure is preserved; the corrected check
matches the retained target definition without changing scientific thresholds.

This new packet is prepared and verified, not run. Require4GiB free after
installation; estimated diagnostic8–20 minutes and export1–5 minutes. Worker1500s,
external1530s plus30s grace, cache120s, gradients480s, solver120s, trials300s,
export300s/external330s plus30s grace,20GiB allocated VRAM,1.5GiB uncompressed
return and512MiB reserve are enforced. There is no bundled cleanup or follow-on
training. Launch manually inside tmux and download one remote source per gcloud
command. No VM connection or actual training occurs during preparation/closure.

The native QMUL comparison remains unpaired development evidence: no convincing
useful current-DGP gain on its six predeclared core cases; seven insufficient
cases remain insufficient and32 reserved final pixels remain unviewed. Source
labels imply neither ethnicity nor Zamboanga performance. A subsequent epoch
recipe still requires full-corpus capacity, paired DEV preservation, useful
native DEV outputs and independent final review. No failed recipe is resumed.
All five restoration milestones, seven automatic/assisted completion families
and bundled inline Playwright app verification remain required. The full goal
is active/incomplete. This status supersedes pending-return statements below
while retaining their exact bytes. Before/status hashes are preserved in
outputs/cctv_dgp_original_loss_balance_v1_closure/.


**Original-DGP diagnostic milestone — 9 October 2026: return audited and all outputs reviewed; a distinct loss-balance packet is verified for manual execution.**

The downloaded original-feature return is present and verified: 1,051,691,000
bytes, SHA256 dbba13dac96fe0bcdf58c03083f6bdfc1e939383b3996b6039ba5c8e93b0f290.
The independent R2 checker verifies all 4,042 regular members, 1,000 raw/PNG
metric records, 30 saved derivative vectors, nine reset displacements and all
36 unchanged scientific decisions. Two hundred fresh CPU outputs meet the
prospective replay tolerances. R1's derived brightness-ratio arithmetic failure
and exact failure log remain preserved; R2 bounds arithmetic propagation only
and changes no scientific threshold or trial decision.

All 60 exact-cell pages and 1,000 unique model outputs are visually reviewed by
the primary assistant as development evidence. The strongest original-decoder
trial reaches 1.34283% and 1.17117% delivered structure gain in the two small
photographic TRAIN cohorts, but has 23 and 18 delivered preservation failure
records. All nine trials are unqualified. Decoder outputs remain soft without
convincing useful clarity; stronger feature/joint steps wash out facial detail.
These 100 cases are not V42's full 3,905-case corpus, native CCTV, DEV or final
identities. V42's 0.00692364% failure remains binding. No quality fix is claimed.

Saved derivatives show that original-decoder structure descent increases mean
identity loss, whereas original-feature identity descent provides a protective
direction. A distinct nine-trial loss-balance diagnostic tests coordination of
these disjoint partitions. First-order mean predictions do not guarantee finite
image or source/profile preservation. The same 1% structure, raw/PNG appearance,
source/profile and brightness requirements apply, with every failure retained.
No failed pilot or epoch trajectory is resumed.

Manual guide: CCTV_DGP_ORIGINAL_LOSS_BALANCE_V1_VM.md.
Design: CCTV_DGP_ORIGINAL_LOSS_BALANCE_V1_DESIGN.md.
Return report: CCTV_DGP_ORIGINAL_FEATURE_PROBE_V1_RESULTS.md.
Packet audit: outputs/cctv_dgp_original_loss_balance_v1_preparation/independent_packet_audit.json.
Protocol: 8e28ab1de4873bca2aff806a21ae173bc039c9463d5b2acf04848b031791d2d2.
Execution archive: 245,615,832 bytes; SHA256
da08ec66d881495efbcfbfbd794ab179df8a1fbff69342b96b3f1e4dd582bede.
Independent preparation verifies 170 archive members, 23 local source bindings,
17 Python 3.10 AST files, all nine saved-vector formulas, both denied local
trial guards, and 100 initial CPU parity cases with unchanged states/buffers.

The new diagnostic reuses the audited derivatives: zero new gradient queries,
zero optimizer updates and zero epochs. It is prepared and verified, not run.
Require 4 GiB free after installation. Estimated diagnostic 5–10 minutes and
export 1–5 minutes; worker/trial/cache/export/storage/VRAM limits are enforced.
All launch commands are manual inside tmux; downloads use one remote source
per gcloud command. No VM connection, cleanup, actual training, app change,
native/final exposure or completion generation occurs during this closure.
Current and original checkpoints retain their exact SHA256 hashes.

QMUL's completed current-checkpoint comparison remains unpaired development
evidence, with no convincing useful current-DGP gain on its six predeclared
core cases. Reserved final pixels remain unviewed. Source labels imply neither
ethnicity nor Zamboanga performance. Any later epoch recipe needs full-corpus,
paired DEV and useful native DEV evidence; independent final review, app flow
and all seven automatic/assisted covering families still require qualification.
The full goal remains active/incomplete. This status supersedes pending-return
statements below while preserving their exact bytes as historical evidence.
Before/status hashes are in outputs/cctv_dgp_original_feature_probe_v1_closure/.


**Current-checkpoint QMUL comparison milestone - 9 October 2026: inference, artifact audit and all-case development visual review complete.**

The earlier QMUL pilot gallery used a different checkpoint. All24 frozen QMUL
DEV crops now compare resizing, original Phase3, current identity-v2 DGP and
CodeFormer fidelity1 on identical256 inputs. Both DGP arms use the retained
frozen evaluation normalization. The current checkpoint stays SHA256
646fbb11674e8882ea41e251d6e6e7d468c963979b48c93132def8292834e24b.
All72 untouched raw arrays,96 delivered images,24 observed masks and four
sheets with exact256 cells are retained. Model states and inputs are unchanged.

Inference finishes in167.20s (172.52s supervised) below480/540s bounds. An
independent checker verifies75 source bindings,200 artifacts,24 geometries,
72 raw arrays,96 PNG compositions and96 gallery cells. All24 current-DGP
CPU replays match exactly (raw error0.0). Phase3/CodeFormer saved arrays and
compositions are audited but are not freshly replayed. No gradients, optimizer
updates, native training, VM call or app change occurs in this milestone.

All24 cases/all four delivered arms are visually reviewed. The six coarse
frontal/mild cases were selected by the unchanged pre-output input review.
None establishes a convincing useful clarity gain from current DGP over resize
in this assistant development review. Broad placement often remains, but eye
and mouth contrast is diffuse. Softness alone is acceptable; the missing useful
gain is the limitation. CodeFormer's finer estimates are unverifiable here.
Seven predeclared insufficient cases still request clearer crops; all controls
remain without replacements. No independent final reviewer or model qualifies.

QMUL evidence is unpaired and remains separate from ChokePoint and paired
synthetic metrics. No native PSNR/SSIM, identity accuracy, per-crop country,
ethnicity or Zamboanga performance is inferred. All32 reserved QMUL cases are
unviewed. Within-release role separation is retained; cross-source and historical
person overlap remain unknown. Original copyright/research notices accompany
outputs. The object-covered profile is not a completion qualification.

V42's0.00692364% structure failure and both-source appearance regressions remain
binding. It trained an added17952-parameter path, not the frozen original DGP.
No threshold is weakened and no failed trajectory is resumed. The verified
original-feature probe remains the next finite manual VM diagnostic before a
distinct training design: CCTV_DGP_ORIGINAL_FEATURE_PROBE_V1_VM.md. No local
return is found during this review; whether the user has run it is unknown.
Its execution archive remains hash-verified, not automatically launched.

Report: CCTV_DGP_CURRENT_QMUL_NATIVE_COMPARISON_V1_RESULTS.md.
Evidence: outputs/cctv_dgp_current_qmul_native_comparison_v1/.
Later visual_review.json supplies the completed review without rewriting the
creation-time pending flags in immutable run/audit receipts. All five restoration
milestones and seven automatic/assisted covering families stay required; the
full goal remains active/incomplete. Exact preceding handoff/status bytes are
archived in the closure folder and retained as suffixes below.


**Original-feature diagnostic milestone — 9 October 2026: packet verified, manual VM run pending.**

The user's request to solve V42's0.0069% structure gain keeps the1% requirement
unchanged. V42 remains failed and is not resumed. The next investigation uses
an isolated current original-DGP copy to compare feature/FPN-only, original
decoder-only and joint structure directions. Earlier V6/V9 joint-training and
V28/V29/V33–V38 preservation failures are reconciled; unfreezing more layers is
not claimed as a proven remedy. The pending architecture preference is not
answered by assumption; the default follows the user's prior best-approach
instruction. This packet diagnoses the next design, not a quality fix or epoch
continuation.

The self-contained183278655-byte execution archive has SHA256
 a76a7be6d7c1c7328f11241eb9c617695e786d3efc0e749f7297cd226a0c2ecf.
The protocol SHA256 is
 ccc30870959c8f4f4f448c1d1e06bb0ae0403d450b81143c6bcb9032d122e94b.
A separate readback verifies all161 archive members, selected158 unique tensors,
all100 initial CPU parity cases and50 historical VM baseline arrays, unchanged
states/buffers, rejected local gradient/trial guards, frozen TRAIN selection,
source bindings and finite launch/export wiring. Two build failures and the
first checker's mistaken legacy reduced-reference pixel hash assumption are
retained. All20 actual targets exactly match frozen V42 target pixels; the
checker records actual pixel bindings without changing targets or thresholds.

Manual commands: CCTV_DGP_ORIGINAL_FEATURE_PROBE_V1_VM.md.
Design: CCTV_DGP_ORIGINAL_FEATURE_PROBE_V1_DESIGN.md.
Preparation review: CCTV_DGP_ORIGINAL_FEATURE_PROBE_V1_PREPARATION_REVIEW.md.
Evidence: outputs/cctv_dgp_original_feature_probe_v1_preparation/.
The proposed diagnostic has30 gradient queries/nine reset finite displacements,
zero optimizer updates/committed trajectories/epochs,4GiB free,1200s worker cap
and300s export cap. It uses50 TRAIN gradient cases and50 different TRAIN probe
cases; no native, DEV or final identity. It cannot qualify full TRAIN capacity,
useful restoration or seven-family completion. Return audit and all trial-face
visual review are required before a distinct subsequent training design.

No VM connection, diagnostic/training launch, app change or native/final pixel
exposure occurs in this milestone. Current checkpoints, splits, caches and failed
stops remain intact. The full five-milestone restoration and seven-family
automatic/assisted completion goal stays active/incomplete. Exact preceding
handoff bytes are preserved in this evidence folder and as the suffix below.


**Native review-boundary milestone — 9 October 2026: frozen metadata and source bytes reconfirmed.**

A current metadata audit and separate readback verify 109 frozen case records,
137 encoded asset hashes, 16 metadata bindings and four raw source files. There
are 51 development/diagnostic cases and 58 reserved cases: original QMUL24/32,
three excluded larger QMUL crops, and ChokePoint24/26. Within each release the
development and reserved labels are disjoint. Cross-source and historical/model
training-person overlap remain unknown; source names are not ethnicity labels.

ChokePoint's 26 reserved entries are metadata only, with no crop/input/mask
files created by the frozen selection. The current audit decodes no image,
views no reserved pixels and runs no inference, gradient, optimizer or VM call.
Creation-time input-review-pending flags are joined to the existing completed
review receipts without rewriting frozen selections. Unsupported and insufficient
cases remain recorded without favorable replacements.

Publisher research terms are rechecked on the official pages; the archived
original notices remain hash-bound. QMUL's aggregate collections include China
and Japan, without per-crop geographic attribution. ChokePoint capture location
is not established by Australian sponsorship. No Zamboanga performance is claimed.
Native observations remain unpaired; no temporal view becomes a clean target.

Report: CCTV_DGP_NATIVE_CONTRACT_AUDIT_20261009.md.
Evidence: outputs/cctv_dgp_native_contract_audit_20261009_v1/.
Final review requires a frozen candidate/processing policy, input-only criteria
and an independent reviewer, with source/temporal grouping and overlap limits.
These cohorts do not qualify seven-family completion. V42 stays failed, original
weights/app stay intact, and the training architecture question remains pending.
The full five-milestone restoration and seven-family automatic/assisted completion
goal remains active/incomplete. Exact preceding handoff bytes are preserved in
this milestone's before/ folder and as the suffix below.


**Original-path ownership milestone — 9 October 2026: metadata review audited; architecture choice pending.**

The retained current DGP has622 serialized state keys,181 unique parameter
tensors/3,312,707 parameters, and171 unique buffer objects. Some state names
alias the same objects. A read-only source/module ownership review finds that
backbone blocks16–18,21 parameter tensors/1,206,080 parameters, are outside the
current FPN forward. The structurally reachable count is160 tensors/2,106,627
parameters, including the two previously diagnosed numerically inactive head4
kernels. Reachability does not prove a nonzero gradient or useful model capacity.

The independent check verifies all622 state names, aliases, numerical inventory,
module ownership and unchanged original state. The two wholly subnormal head4
kernels agree with the retained zero-gradient diagnostic. V28's original-path
preservation/brightness failures remain binding. No checkpoint is pruned or
changed; unused stored layers remain preserved for provenance and migration.

A future original-model learner needs explicit unique parameter ownership,
normalization/initial parity and a finite manual L4 connectivity test. Simply
unfreezing every stored tensor would include parameters outside this forward.
No learning code or packet is implemented while the three-attempt architecture
question is pending. This milestone performs zero neural forwards, gradients,
optimizer updates or VM operations. The app and original weights remain intact.

Report: CCTV_DGP_POST_V42_ORIGINAL_PATH_INVENTORY.md.
Evidence: outputs/cctv_dgp_post_v42_original_path_inventory_v1/.
The full five-milestone restoration and seven-family automatic/assisted completion
goal remains active/incomplete. No native/DEV/final pixels are opened. Exact
preceding handoff bytes are preserved in the evidence directory's before/ folder
and as the suffix below.


**V42 return milestone — 9 October 2026: failed at50 updates; audited, preserved, not promoted.**

The1,156,863,656-byte manual return matches SHA256
308ad100ea6760bf0b6b42621a908d083f8691c3b8e53ab17b132dca0c88d9c2.
The unchanged prospective auditor verifies35,268 archive members, all7,810 PNG
pixel-metric sets and100 stored raw CPU replays. Other raw float pixels were not
retained and no full raw-value audit is claimed. The failed gate is independently
reproduced: delivered structure gain0.00692364%, raw gain0.00709086%, required1%;
both stages have compound ArcFace-score failures in both photographic sources.

All57 added-decoder tensors change, but the original DGP, fixed reference and
recognizer states remain unchanged. There are50 updates/250 photographic TRAIN
exposures and zero complete epochs. This does not add epochs to the current DGP.
The manual worker lasts1,368.48s; export63.20s. No storage stop occurs. Snapshot0
and50 each take about655s; recorded optimization steps total7.75s. The agent
launches no training, resumes no failure and performs no VM operation in this
return-review milestone. Earlier maintenance/launch observations remain dated
history, not evidence of a still-live worker.

All50 unscaled previews are visually reviewed; convincing added structure is
not established. A frozen same-cohort correction-loss comparison improves only
0.0663482% overall; clear controls acquire nonzero corrections. It covers three
reconstruction terms, not the complete combined objective. No gradient or
optimizer is used locally. Original/failed checkpoints, full state, splits,
scientific caches and receipts remain. No app source or selected weight changes.

The V40–V42 added-decoder structure assumption has failed. The AGENTS.md
three-attempt diagnostic question is pending; no new learning code or packet is
created while it is pending. Proposed review: a separate copy of the current
DGP's original reconstruction and feature path, with parity and normalization
checks and explicit reconciliation of previous original-path failures. This is
an unproved design direction, not an approved remedy or automatic VM run.

Reports: CCTV_DGP_V42_RESULTS.md and CCTV_DGP_POST_V42_ARCHITECTURE_REVIEW.md.
Evidence: outputs/cctv_dgp_v42_return_review_v1/ and
outputs/cctv_dgp_residual_epochs_v42_independent_audit.json.
No new native CCTV, development or final pixels are opened. The full five-
milestone restoration and seven-family automatic/assisted completion goal is
active/incomplete. Exact preceding handoff bytes are retained in this milestone's
before/PROJECT_HANDOFF.md and as the suffix below.


**Live maintenance preflight — 9 October 2026: V42 manually active; sufficient storage, no cleanup needed.**

A fresh pinned read-only gcloud inventory confirms the existing instance/L4 and
20.1753GiB free after the V42 installation, above its8GiB requirement. The
expected V42 protocol, manual tmux launcher and GPU workerPID1324 are live.
No file was deleted, VM started, process interrupted or training launched by the
agent. This observation supersedes the earlier prepared-only execution status;
those earlier records remain historical evidence of preparation.

A second inventory at Unix1791547195.1830459 confirms the same worker is still
live, with11,750MiB GPU use and19.9211GiB free. The log shows caching and the
three-component gradient preflight completed and initial snapshot0 reached
1,505/3,905 cases. No terminal result/failure or export receipt exists at that
observation, and no optimizer-update completion is inferred. The measured cache
projection is34.427s against its900s cap. Leave the manual run and original
scientific stops intact. Process presence, logs and storage are not a quality
pass; independently audit returned artifacts before any model or app decision.

The independent storage audit verifies transport hashes, TLS/hostkey settings,
instance identity, protocol, source and18 local research bindings. It observes
active work without modifying it. Original checkpoints, datasets, splits,
caches and failed gates remain. The full five-milestone restoration and
seven-family completion goal stays active/incomplete.

Storage report: CCTV_DGP_V42_VM_STORAGE_PREFLIGHT_V1.md
Live evidence: outputs/cctv_dgp_v42_vm_storage_preflight_v1/
Manual launch/download guide: CCTV_DGP_RESIDUAL_EPOCHS_V42_VM.md
The exact preceding handoff bytes are archived in that live-evidence directory's
before/PROJECT_HANDOFF.md and retained below.


**Training preparation milestone — 9 October 2026: V42 direct correction study ready for manual transfer; no training started.**

The 145-case pre-clamp correction arithmetic audit passes without local training.
Its NumPy/OpenCV checker rechecks1,740 terms and356 source bindings. The distinct
V42 recipe supervises observed, landmark and RGB-pyramid corrections directly,
with zero correction for clear inputs, identity supervision and retained
regression penalties. The current app DGP, stored normalization and fixed initial
decoder remain frozen; only our own17,952-parameter spatial decoder learns.
These are spatial-decoder epochs, not five extra epochs of the frozen encoder.

The prepared study has five complete781-reference epochs/3,905 updates and
19,525 photographic TRAIN exposures, comparing0/50/781/1,562/3,905 snapshots.
Current checkpoint initialization, all original splits and failures remain.
No native/DEV/final input is optimized. Unchanged1% early/10% final structure,
all17 preservation groups, both sources and20% brightness-only limit apply
separately to raw and PNG outputs. Stop and export the first failed requirement.
AdamW3e-4 is reduced to9e-5 after update1,562. Rate/weight choices are declared,
not established optimal; new improvement gradients are checked on the VM before
constructing an optimizer. No failed recipe is resumed or retried automatically.

Independent preparation verifies all5,493 assets/archive contents, five complete
schedules and50 exact initial CPU inference outputs. Every781 reference passes
three scale-support checks. The R1 checker corrects an AST-variable test; the
original checker/failure remain and the packet/recipe/archive do not change.
Read-only Bash syntax passes outside the Windows signal-pipe sandbox restriction;
both outcomes are retained. No local gradient, backward or optimizer call ran.

Manual training requires the idle existing L4/g2-standard-4 inside tmux and8GiB
free after installation. Cache900s, fit6,300s, worker7,200s/external7,230s,
export900s/external930s,30s kill grace,20GiB peak allocated VRAM,3.5GiB return and
512MiB protected disk reserve are enforced. Timing and output/export projections
have1.25 safety factors. Full optimizer/scheduler/RNG/schedule state is exported
at snapshots/stops for a later reviewed migration, never automatic failed resume.

The prospective auditor checks all delivered PNG pixel metrics, raw aggregates,
saved vectors and50 raw CPU replays per snapshot. Other raw arrays are hashed,
so a full raw-value audit is not claimed. Useful native development outputs,
independent final review and all seven automatic/assisted covering families
remain unqualified. The existing app and accepted weights are unchanged.
No VM connection, training or promotion occurred. Full goal active/incomplete.

Manual five-step guide: CCTV_DGP_RESIDUAL_EPOCHS_V42_VM.md
Preparation review: CCTV_DGP_RESIDUAL_EPOCHS_V42_PREPARATION_REVIEW.md
Protocol SHA256: c19ca1790ae9ebe7f99000a81678c8fc756d70ce2114debb6f81691c741a6f2d
Execution archive SHA256: 9988d57e0c3733df0b255c93282b284dd13b98e296fe605de101f3c2972ddf4c
Execution archive bytes: 444,114,939
Independent packet audit: outputs/cctv_dgp_residual_epochs_v42_preparation/independent_packet_audit.json
The exact preceding handoff is archived at
outputs/cctv_dgp_residual_epochs_v42_milestone/before/PROJECT_HANDOFF.md.
Its complete bytes are retained below.


**Goal update — 9 October 2026: current-checkpoint improvement and staged training are part of the canonical goal.**

The human explicitly requested updating the goal to continue work. The current
section at the start of SYSTEM_WORKFLOW_AND_GOAL.md now contains the complete
revised goal, retaining all five milestones and seven-family completion scope.
PRACTICAL_OUTPUT_SCOPE.md is aligned with the latest execution and credit/deadline
decisions. Both prior documents and this prior handoff are archived byte-exact.

Start from the current app DGP's identity-v2 tensors, with original Phase3 training
plus two selected fine-tuning epochs/226 branch updates. Complete lifetime epoch
count is unconfirmed. Review evidence-justified reconstruction-path/supervision
changes before a new finite protocol; compare proposed additional epochs 1, 2 and 5
within that study. Longer extensions require useful independently audited
preservation/development results. All rejected historical pilots and failures stay
preserved; no unchanged rerun, failed-stop bypass or weakened gate is authorized.

Use audited native CCTV development crops for unpaired review and audited clean
training references for paired supervision; final identities remain reserved.
Keep DGP primary locally, preserve visible appearance and app design, request
clearer or less-covered inputs when necessary, and independently qualify the
automatic and assisted seven-family completion and full inline Playwright flow.

The user reports USD 34 and will notify us near USD 5 for a verified full-state
migration export. No additional monetary cap replaces finite timing/update/epoch,
storage, VRAM and export stops. Thesis submission/defense is around or after
20 November 2026. Actual new training stays manual inside tmux on the existing
L4 until the user identifies a verified replacement VM; direct existing-VM
connection remains authorized for inventory-first, hash-bound maintenance only.

This update changes goal documents only. No training packet is prepared, neural
or optimizer call made, VM connected, checkpoint selected or app source changed.
The next technical prerequisite is reconstruction-objective/path verification
before preparing a distinct finite manual training packet. Goal active/incomplete.

Canonical objective text: outputs/cctv_dgp_goal_update_20261009_v1/goal.txt
Objective SHA256, UTF-8 without terminal newline: 19894b32adfb92961a0f48ea7959d158b496b65c40071b008648650aef36d1a3
SYSTEM_WORKFLOW_AND_GOAL.md SHA256: 1e55d2048799385e46ccf37b41135c999e27e823e97c0464d8b61b4ddf416bde
PRACTICAL_OUTPUT_SCOPE.md SHA256: 14076b135779cf1d7ee76d6311493bdfee465336143f32151066c76954d1943f
Exact preceding documents: outputs/cctv_dgp_goal_update_20261009_v1/before/
The complete previous handoff bytes are retained below.


**Training planning milestone — 9 October 2026: current DGP lineage verified; longer-training design scoped.**

The latest user decision starts from the DGP currently used by the main app and
permits evidence-justified training changes. The user selected already audited
native CCTV development crops for review and the best justified reference-data
approach. Their reported credit balance is USD 34; they will notify us near USD 5
to export before moving VMs. No additional monetary cap was selected. Finite
updates/epochs, timing, storage, VRAM and stop rules remain required. Thesis
submission/defense is around or after 20 November 2026, without an exact date.

A new read-only local review confirms all 622 app tensors exactly equal the
selected identity-v2 epoch-2 state. Its branch completed 226 updates; the 452
pilot total includes the other rejected branch. The current state is original
Phase3 training plus two selected fine-tuning epochs, not only two lifetime
epochs. Its weights-only file has no complete ancestral epoch/optimizer history.
Separate Phase4 labels 27–31 and V9's rejected twenty-epoch run do not extend the
current app weights. Original models, splits, caches and failed gates remain.

Existing HQ references include 391 training-role and 53 validation-role images;
reuse them before unnecessary acquisition. Native CCTV is unpaired review
material, not a clean training target. Final identities remain reserved.
Training loss and aggregate V2 development scores improved from epoch 1 to epoch 2,
but blur/motion pixel errors worsened between those two checkpoints. V6/V9 and
the finite-step failures prevent treating more epochs or HQ data alone as a fix.

The preparation direction reviews direct visible reconstruction in our DGP's
spatial/feature path, preserving identity and all existing appearance gates.
A proposed finite continuation would compare additional epochs 1, 2 and 5 under
one new declared recipe. This is a design outline: objective, trainable tensors,
learning rates and numerical resource/stop bounds are not yet frozen. No new
executable protocol, transfer packet, neural call, gradient, optimizer update,
VM connection or app promotion occurred. Actual training stays manual inside
tmux on the existing L4 until the user identifies a replacement. A successful
future migration must export and verify full optimizer/scheduler/RNG/schedule
state as well as weights and provenance; failed recipes are not resumed.

All five goal milestones and separate seven-family automatic/assisted completion
qualification remain required, including useful native review, independent final
review and bundled inline Playwright verification. Goal active/incomplete.

Plan: CCTV_DGP_CURRENT_TRAINING_IMPROVEMENT_PLAN.md
Plan SHA256: bd453ab3bd89885f247c077a028797112dbee8ff464e2feb46b7d36aeb93349f
Review: outputs/cctv_dgp_current_training_review_v1/training_history.json
Review SHA256: b3178c7651a094ae85232f9f2862a9cfc8710103ee2c84be38713aff69876409
The exact preceding handoff bytes are archived at
outputs/cctv_dgp_current_training_review_v1/before/PROJECT_HANDOFF.md.
The complete previous handoff body is preserved below.


**Research milestone — 9 October 2026: final-state return audited; actual-step diagnostic complete, proposals unqualified.**

The human manually ran the final-state tail. Its review/export stages both
completed. The324,974,655-byte archive matches SHA256
4a0037a5029aeb854ca7264e5179b0a06f110c6f6ebfc9c2ba3aa96561516616.
Strict local import checks959 members without executing returned code.
The prospective scientific auditor checks315 raw/PNG/mean-only slots,
828 exact decoded overlap files and15 frozen CPU replays. Persistent states,
original stored-row arithmetic and all categorical decisions remain exact.
One predeclared floored mean-only quotient roundoff allowance is bounded;
both underlying degraded-MSE gates still fail. No model-quality gate changes.

The original storage-stopped run remains failed and preserved. Together the
separate audits cover30 conditions and3,150 unique planned slots:
3,045 +315 -210 repeated controls. CPU replays total160 executions/150 unique
slots. The original full checker is not relabelled as passing. All50 frozen
sheets/250 rows/1,500 exact256-pixel cells have been actually viewed across
the prior and final assistant reviews. The five new sheets add25 rows/150 cells.

Neither recorded nor cone proposals pass preservation on either fixed TRAIN
cohort at any state in raw or PNG form. Current-batch raw passes only at27;
every current PNG fails. The recorded raw landmark term increases in9/10
batches; cone increases in5/10. Degraded eyes, nose and mouth remain soft;
no convincing added whole-face clarity is established. The finite diagnostic
is complete, but it supplies no trained candidate or app qualification.

A separate all145-case saved-array target oracle is completed and independently
audited with zero neural/gradient/optimizer calls. An ideal bounded, mean-centered
correction has96.4534% raw/96.3637% PNG degraded landmark-error reduction.
These are oracle arithmetic, not model performance: clean target access is
unavailable in the app; ArcFace/complete preservation/network capacity are
untested. Mean/amplitude geometry alone does not explain the tiny learned gains.

The selected next design review examines reconstruction-first supervision of
our own spatial DGP decoder before another executable pilot. All earlier
restoration-only/group/PNG-guard failures remain binding. No unchanged retry,
new training protocol, transfer packet, VM launch or app promotion occurs.
Actual training remains manual on the existing L4 under a distinct verified
finite packet with unchanged preservation/structure requirements.

All14 DGP-primary local app bindings and original checkpoint/source/split/failure
assets remain exact. No new native/DEV/reserved-final data are opened. This is
paired photographic TRAIN diagnostic evidence, not native CCTV, covering-family
qualification or independent final review. No ethnicity, Zamboanga performance
or recovered hidden-identity claim is made. Useful native restoration, all
seven automatic/assisted completion families, independent final review and
qualified full app flow remain outstanding. Goal active/incomplete.
The complete preceding handoff, including the cleanup milestone, is archived
and preserved byte-for-byte below. The cleanup free-space reading is historical;
this audit does not report a new live VM free-space measurement.

[Completed diagnostic](<C:/xampp/htdocs/YEAR 4/Testing/CCTV_DGP_ACTUAL_STEP_TAIL_V1_RESULTS.md>)
[Next design review](<C:/xampp/htdocs/YEAR 4/Testing/CCTV_DGP_POST_ACTUAL_STEP_ARCHITECTURE_REVIEW.md>)

**Maintenance milestone — 9 October 2026: backed archive cleanup completed; final-state diagnostic remains manual.**

Removed 35 exact, inactive top-level VM archive copies after fresh inventory,
full local/remote SHA256 backup checks and a pre-apply Windows backup recheck.
Observed free-space increase: 16.298GiB. Separate live audit: 21.644GiB free.
All Windows originals remain retained. The audit confirms 35 paths absent,
382,930 protected research-file hashes unchanged,
3,569 scientific tensor stamps retained and
all 1,095 current diagnostic bindings exact, including
the 828 overlap files required by the frozen final-state packet.
Original checkpoints, data/splits, research caches, provenance, instructions,
sources and failure records remain on the VM. The shared Python runtime is present;
GPU idle. All 14 DGP-primary local app bindings and the manual tail guide remain exact.

No training, inference or diagnostic was launched by maintenance. The user will
manually follow CCTV_DGP_ACTUAL_STEP_TAIL_V1_VM.md; its 2GiB free-space requirement
is met. Preserve the original 3GiB encoded-output protocol stop and all scientific
gate failures. Cleanup is not model qualification. Useful native restoration,
seven-family automatic/assisted completion and independent final review remain
outstanding. Evidence: CCTV_DGP_VM_STORAGE_CLEANUP_20261009_V1.md and
outputs/cctv_dgp_vm_storage_cleanup_20261009_v1/.

**Research milestone — 9 October 2026: actual-step partial return audited and visually reviewed; final state pending.**

The human returned the finite inference diagnostic. Its3,204,580,880-byte archive
matches SHA256 d380ce08cf858eebc334fcad3c6f5590e01d526084239bbf807bd4b465e8c604.
Strict import verifies all9,369 files. The original run stopped at its frozen
3GiB encoded-output limit, preserving29 complete conditions; update45/cone
has66 arrays/130 PNGs but no complete metrics receipt. This is a protocol storage
stop, not a demonstrated full-disk error. No optimizer, gradient, backward or
new trained checkpoint occurs. Preserve the stop; the full3,150-slot diagnostic
is not complete and the original full checker remains import-only.

Separate R1 scientific verification checks every3,045 complete raw/PNG/mean-only
slot and145 frozen CPU replays. All categorical decisions and original
pixel/objective/replay tolerances remain unchanged. The failed first partial
checker is retained. R1 allows only a bounded recomputation of a mean-only
quotient whose denominator is floored at1e-12 and both MSE gates already fail.
Three exceptions have3.469446951953614e-18 group-MSE roundoff; all six corruption/
branch/decision regressions pass. This audit recovery does not repair model gates.

Neither recorded nor cone proposals pass preservation on either fixed TRAIN
cohort at any complete state, in raw or PNG form. Current-batch raw preservation
passes only at27; none of its PNG comparisons pass. The recorded landmark term
increases in9/10 batches; cone finite changes can also increase it slightly.
This rejects the tested local proposal as sufficient evidence for a new recipe;
no unique whole-model cause or PNG-only attribution is established.
All45 fixed sheets/225 rows/1,350 exact256-pixel cells were actually viewed by
the implementing assistant. Proposals remain visibly close to the earlier DGP,
with soft degraded central features and no convincing added facial structure.
This is exposed paired photographic TRAIN diagnostic evidence, not an
independent final, native CCTV or completion-family quality review.

A lossless-storage audit verifies905 repeated original-RGB copies, with at least
570.36MiB potential compressed-payload savings. Nothing is deleted or deduplicated.
The smaller43,992-byte final-state packet remains prepared/unrun. It evaluates
update45 only, repeats210 controls for exact overlap validation and adds105 cone
slots. Its checker must verify315 outputs,828 exact decoded overlap files and
15 frozen CPU replays. Require2GiB free; estimate3–7minutes plus download.
Protocol SHA256: ec6981ce0e6961b765fb5dc1e82f191bc3511db2a1ca6b87e2849e83ea40eef9
Execution archive SHA256: 0436e9477b1ce277856d9c54a1655d2915091c204637ab78e1c1fffe85d68950
All4 transfer members,3 packet assets and260 original parent assets pass;
scope/storage/timing-only source changes have exact inverse proof. Python3.10,
pre-Torch Windows guards and read-only Bash syntax pass. Cache120s, review300s,
worker600s/external630s, export300s/external330s,20GiB VRAM and512MiB returns
are enforced. Manual upload/install/tmux/download steps are prepared. No new
actual training recipe or automatic launch occurs; every historical failure stays.

All14 DGP-primary app bindings remain exact. Original checkpoints, sources,
splits, research caches/local backup and provenance remain retained. No new
native/DEV/reserved-final images are opened and no ethnicity or Zamboanga claim
is made. Clear glasses, ordinary hair and every visible facial feature remain
protected. Useful native restoration, all seven automatic/assisted covering
families, independent final review and qualified full local app flow are still
required. Goal active/incomplete. Prior functional Playwright evidence is
retained; no app change justifies a new browser run for this evidence milestone.
The complete previous handoff is archived and preserved byte-for-byte below.

[Diagnostic findings](<C:/xampp/htdocs/YEAR 4/Testing/CCTV_DGP_ACTUAL_STEP_REVIEW_V1_RESULTS.md>)
[Five manual final-state steps](<C:/xampp/htdocs/YEAR 4/Testing/CCTV_DGP_ACTUAL_STEP_TAIL_V1_VM.md>)

# Forensic DGP: workspace handoff and training runbook

**Research milestone — 9 October 2026: finite actual-step diagnostic prepared; manual VM execution pending.**

V41 remains failed at50/800: structure gain0.0516710943% against the unchanged1%
requirement and one delivered FFHQ compound preservation failure. Its archive,
checkpoint, full audit, source code, splits and failed gates remain retained.
The earlier "apply the best approach" preference and completed applied-step
architecture review select a separate finite diagnostic before another training
recipe. The historical pending question is superseded for this diagnostic
preparation only; original training code remains unchanged.

The separate packet compares ten recorded V41 before-states and three fixed
parameter proposals: unchanged, recorded actual and one saved-array cone step.
Both preselected50-case TRAIN cohorts remain exact; each current five-profile
batch is included. This yields145 unique cases/29 references,30 conditions and
3,150 finite forward slots. Probe states are selected after TRAIN outcomes and
remain diagnostic. Every slot retains raw floats, delivered PNG and a mean-only
control; raw and delivered objectives/preservation are reported separately.
Float32 proposal rounding is retained, including seven proposals with tiny
positive rounded local constraint derivatives. No finite improvement is assumed.

The independently verified261-member transfer archive is197,588,115bytes.
Protocol SHA256: 339c20425b6fa6b4141001aae4ff3010bf9d438acc32eb71ec08f04f001476f5
Execution archive SHA256: ff0ef28764cfb9218f4ec752a5e35dc325c6d896386e78dabc579fe7a4374141
All25 corruption/archive/scope/decision regressions pass. Both Windows worker
and supervisor stop before Torch imports. All100 archived-float fixed-filter
arithmetic checks pass without model construction, gradients or parameter
assignment. All Python sources parse for3.10. A read-only Bash syntax check
passes outside the sandbox after retaining its Windows signal-pipe failure.
Two earlier metadata-only preparation failures and their153-file partial copy
remain preserved; their fix distinguishes provenance paths from runtime assets.
Original metadata, labels and scientific inputs are unchanged.

Manual execution remains on the existing L4/g2-standard-4 at ~/forensic-dgp.
Require6GiB free after installation; estimated10–25minutes plus export.
Cache300s, review1,500s, worker1,800s/external1,830s, export900s/external930s,
20GiB allocated VRAM and3GiB encoded returns are enforced. First315-slot timing
must project within1,500s with a1.25 safety factor. Every failure is retained;
no automatic follow-on, historical rerun or cleanup occurs.
The diagnostic has zero optimizer updates, zero gradient queries and no trained
checkpoint. No VM connection or new neural forward occurs during preparation.
The100 local arithmetic checks use fixed filters, not a neural model.

The prospective independent checker is source-bound before the run. It checks
all3,150 raw/PNG/mean-only slots and150 frozen CPU replays under a2,400s bound.
Numerical aggregate allowances preserve exact categorical decisions and do not
change delivered-image gates. Partial failures receive an import-only receipt,
never a full scientific-output audit. No returned code is executed. This packet
is prepared and unrun; it does not establish full TRAIN capacity or qualify a
restoration model.

All14 DGP-primary app bindings remain exact. No frontend or trained-model change
requires another app-flow test here. The earlier functional Playwright evidence
does not qualify restoration/covering quality. Research caches/local backup,
provenance, original checkpoints and all failure records remain retained.
No new native/DEV/reserved-final images enter this packet. Unpaired CCTV stays
separate from paired photographic metrics; source labels do not establish
ethnicity or Zamboanga performance. Clear glasses, ordinary hair and every
visible facial feature remain protected. Useful native restoration, separate
automatic/assisted review for all seven covering families, independent final
review and qualified app flow remain outstanding. Goal active/incomplete.
The complete earlier handoff is archived and preserved below.

[Five exact manual steps](<C:/xampp/htdocs/YEAR 4/Testing/CCTV_DGP_ACTUAL_STEP_REVIEW_V1_VM.md>)
[Independent packet audit](<C:/xampp/htdocs/YEAR 4/Testing/outputs/cctv_dgp_actual_step_review_v1_preparation/independent_packet_audit.json>)



**Research milestone — 9 October 2026: V41 preservation coverage traced.**

The independent read-only trace verifies all391 FFHQ compound TRAIN cases,
1,173 saved embedding arrays,782 PNGs,391 support masks and20 exact comparison
cells, with2,373 source bindings. The original V41 structure/preservation stop
at50 of800 is unchanged. The27 exposed cases improve mean ArcFace by0.0001794937;
the364 unexposed cases fall by0.0000428180. Their combined group drop0.0000274666
still exceeds the1e-6 allowance. This attributes the saved group score, not
a unique causal mechanism or recovered identity.

Current-batch training preservation uses raw predictions; the final gates use
delivered PNGs over the full TRAIN corpus. Only5 of391 failed-group cases retain
raw floats. PNG changes reach12 channel byte levels. This evidence cannot
attribute the failure exclusively to PNG quantization. Five post-outcome
diagnostic cases/20 unscaled cells are actually viewed; convincing useful added
eye/nose/mouth definition remains unestablished. These are paired photographic
TRAIN cases, not native CCTV or a relabeled independent evaluation set.

The pending design review now requires checking finite actual parameter
proposals on both unchanged preselected50-case TRAIN cohorts and each current
batch, retaining raw and delivered stages for every probe. The proposed review
bound is10 recorded states/three fixed proposals/3,150 forward slots; it is not
a runnable training packet. The circuit-breaker question remains pending.
No model/app/training code, original gate or split changes, no neural/gradient/
optimizer/VM calls and no automatic rerun or app promotion occur here.

The first trace recorder's missing manifest lookup is retained; distinct R1
uses the exact prior prepared-packet bindings. Independent inverse-source and
all-case arithmetic/pixel checking passes in6.751seconds. All14 DGP-primary app
bindings and original/initial/stopped checkpoint hashes remain exact. Research
caches/local backup, provenance, split and failure records remain preserved.
Native CCTV stays unpaired, reserved final identities unopened, and no ethnicity
or Zamboanga performance claim is made. Useful native restoration, all seven
automatic/assisted covering families, independent final review and qualified
full app flow remain outstanding. Goal active/incomplete.
The complete previous handoff is archived and preserved below.

[V41 preservation trace and review requirements](<C:/xampp/htdocs/YEAR 4/Testing/CCTV_DGP_V41_PRESERVATION_TRACE_V1.md>)


**Research milestone — 9 October 2026: V41 failure audited; applied-step design review retained.**

The human returned V41,1,178,179,150bytes, SHA256
01a6e9198e731d81895c0b64cde21f45ed7eb01be2bb48b4c78a9186726c94dc.
Independent audit verifies27,551 files, all50 parameter/moment chains and350
saved component-gradient queries,7,810 delivered PNGs plus7,810 mean-only PNGs
and100 frozen CPU
preview replays. The VM stops at50 of800 updates; its17.37-minute run and
successful export do not imply restoration qualification.

Delivered paired photographic TRAIN structure gain is0.0516710943%, below
the unchanged1% requirement. One FFHQ compound-group ArcFace preservation
failure also remains. All3,905 initial PNGs match V40 exactly. Both preselected
TRAIN cohorts,100 cases/500 unchanged256-pixel cells, are actually reviewed:
useful added eye/nose/mouth/outline definition is not established. No native,
development or reserved-final evaluation is opened or relabeled from outputs.

Saved-step arithmetic shows nine actual AdamW updates opposing the local
landmark-improvement gradient despite its projected direction being nonascending.
This remains after removing weight-decay displacement. It is first-order evidence,
not a unique cause or finite-image guarantee. An independently KKT-checked,
fixed21-iteration cone calculation on all50 saved arrays is a diagnostic proposal
only; no proposed parameter is assigned to a network or trained checkpoint.

The original prospective auditor rejects the exporter's retained V40 archive
prefix before extraction. Its source/log/receipt remain. Distinct R1 changes
only that exact prefix and its receipt filename, imports into the separate V41
folder and passes11 archive-boundary checks. Training scripts, protocol, weights,
image/gate thresholds and all historical failures are unchanged.

The post-V38/V40/V41 circuit-breaker question asks which design diagnostic to
investigate before another training pilot. No new training recipe is modified,
resumed or automatically run. The reviewed next direction is testing actual
optimizer-step preservation before further learning; finite neural evidence is
still required. All actual training and gradient diagnostics stay manual on the
existing L4/g2-standard-4 at ~/forensic-dgp.

All14 DGP-primary app bindings, original/stopped checkpoints, caches/local backup,
data/splits/provenance and failure records remain. Public CCTV is unpaired; paired
synthetic metrics remain separate. Source names do not infer ethnicity or
Zamboanga performance. Reserved45 identities/58 crops remain unopened. Useful
native restoration, seven automatic/assisted covering families, independent
final review and qualified full app flow remain outstanding. Goal incomplete.
The complete previous handoff is archived and preserved below.

[V41 audited result and design review](<C:/xampp/htdocs/YEAR 4/Testing/CCTV_DGP_POST_V41_OPTIMIZER_REVIEW.md>)


**Research milestone — 8 October 2026: human V40 endpoint diagnostic audited; V41 PCGrad pilot prepared.**

The human returned the learning-signal-v1 archive,303,352,790bytes, SHA256
002cd3765d17644b993646a0320c95d72de31e318b62285c93b1bcc7ba478a10.
The unchanged prospective audit passes in94.566s:957 exact files,280 saved
component-gradient queries,200 raw/300 PNG compositions and40 frozen CPU
replay cases. The VM diagnostic completes in75.072s with zero optimizer or
parameter updates/backwards/epochs and no new trained checkpoint. Historical
raw parity is exact and all four model states remain bound. The earlier
stopped-VM observation predates this human run; no fresh guest-space/status
claim or agent VM execution is made here.

Stopped50 identity-preservation gradient norms are5.328730 and7.522134 times
landmark norms in the two exposed TRAIN cohorts. Raw landmark loss improves
only0.009353% and0.019944%; V40's full-TRAIN delivered gain remains0.0106079234%,
below the unchanged1% early requirement. The preview-cohort summed negative
gradient increases observed-detail loss. These are endpoint observations,
not a reconstructed AdamW trajectory or unique cause. All50 newly selected
optimized faces/200 exact cells are visually reviewed on10 unscaled sheets:
no convincing incremental definition over the original DGP. Source names do
not establish ethnicity; these paired photographic TRAIN cases are not CCTV
or independent evaluation. No input-only labels or split are changed.

After primary-source PCGrad research, one fixed saved-array projection rule
yields nonincreasing first-order component directions in all40 batches. The
independent arithmetic also verifies four aggregate sets and200 exact sheet
cells. This motivates V41, not finite-step preservation or model qualification.
The distinct PCGrad treatment combines the same seven weighted gradients;
architecture, initial seed,3,905 TRAIN cases/781 references,800 paired batches,
AdamW, composition and all numerical gates stay fixed. Per-update gradients,
parameter endpoints, scalar losses and AdamW moments are retained for audit.
No old checkpoint or failed gate is overwritten, resumed or automatically run.

V41 packet verification passes:5,493 files/444,567,499bytes; seven corrupted
step records, nine unsafe archives, three wrong scopes, Windows rejection
before neural imports, fixed800 orders, Python3.10 syntax and read-only Bash.
Protocol:46dbeab9515719fdd5571cdbfdf5e52e6673bc78e44ccff84bd2f3f15e69c56e
Archive:0f1cb2e8b6d11dade6fb0483b4ba1f285ed2c926bc0bf9f0442763b73bae0d53
This is prepared only. Use the five verified single-source gcloud upload,
install,tmux,launch and download steps manually on the existing L4/g2-standard-4
at ~/forensic-dgp. Need6GiB free after installation. Maximum800 updates, unchanged
1%-at50/10%-at800 and all17 preservation/source/mean-only gates remain. Cache900s,
fit3600s, worker4500s/external4800s+30s, export900s/external930s+30s,20GiB allocated
VRAM and3GiB uncompressed return stops are enforced. No training starts here.

The original local audit's wrong-Python/PyTorch absence and full import remain;
the same checker passes in the research venv. The separate analysis's exact
derived-ratio failure (8.881784197001252e-16) and source remain; distinct R1 allows
rtol2e-12/atol0 only for that ratio, without changing arrays or decisions. New
prospective V41 arithmetic allowances do not weaken any delivered-image gate.

All14 DGP-primary app bindings, original/stopped checkpoints, research caches,
local backup, provenance, splits and previous failures remain. V38 development
preservation and completion-quality failures remain binding. Native CCTV remains
unpaired and synthetic PSNR/SSIM separate; reserved45 identities/58 crops stay
unopened. No Zamboanga CCTV evidence or hidden-identity claim exists. Useful
native restoration, all seven covering families with separate automatic/assisted
quality, independent final review and qualified full app flow remain outstanding.
The entire previous handoff is archived and preserved below. Goal active/incomplete.

[Audited diagnostic findings](<C:/xampp/htdocs/YEAR 4/Testing/CCTV_DGP_V40_LEARNING_SIGNAL_V1_RESULTS.md>)
[V41 five manual commands](<C:/xampp/htdocs/YEAR 4/Testing/CCTV_DGP_PCGRAD_FIT_V41_VM.md>)


**VM availability — 8 October 2026: existing VM is stopped; diagnostic remains manual and unrun by the agent.**

The fresh read-only Google Cloud API query reports forensic-dgp-thesis in
us-central1-a as TERMINATED, machine type g2-standard-4. The initial maintenance
connection stops at TLS validation. A bounded retry uses the existing Windows
CA bundle and pinned historical host key with TLS enabled, then SSH closes
through IAP. The separate API status query succeeds. Both failures are retained;
no VM start, guest deletion, diagnostic or actual training is performed here.

Fresh disk/GPU/tmux/venv and guest research-file integrity cannot be verified
while stopped. No cleanup or recovered-space claim is made. The next action is
manual VM start when the human is ready, followed by the existing five verified
upload/install/tmux/run/download steps. The supplementary availability guide
includes the exact startup, temporary SDK Shell CA setting and status command.
No global SDK setting or frozen packet/guide/protocol is changed.

The 266-file/239.5MiB endpoint packet remains independently verified. It has
280 component-gradient queries, zero optimizer updates/backwards/epochs, an
existing-L4/idle-GPU/2GiB-free guard,600s worker and finite export/VRAM/size stops.
New actual learning continues under the human's manual VM workflow. The full
prepared-diagnostic closure passes in112.23s and retains578 new bindings plus
the ten prior milestone binding counts. All14 DGP-primary app bindings remain.

V40's failed50-update structure gate, every original/stopped checkpoint, split,
cache/local backup and failure remain locally preserved. There is no new native,
development, reserved-final or completion quality qualification. Useful native
restoration, all seven covering families with separate automatic/assisted
review, independent final review and the qualified DGP-led flow remain required.
The entire previous handoff is archived and preserved below. Goal active/incomplete.

[Manual start and availability evidence](<C:/xampp/htdocs/YEAR 4/Testing/CCTV_DGP_V40_LEARNING_SIGNAL_V1_VM_AVAILABILITY.md>)
[Five verified manual diagnostic steps](<C:/xampp/htdocs/YEAR 4/Testing/CCTV_DGP_V40_LEARNING_SIGNAL_V1_VM.md>)


**Research milestone — 8 October 2026: V40 saved-weight analysis audited; endpoint diagnostic prepared, not run.**

V40's failed50-update structure requirement remains unchanged:0.0106079234%
gain against1%. Every original/stopped checkpoint, failed gate and visual review
remains. No unchanged rerun, resume, weaker gate or new app model is adopted.

Independent saved-state arithmetic verifies all17,952 parameter differences;
all57 tensors changed and the stopped checkpoint equals snapshot50 exactly.
The float64 displacement norm is0.21028403907940274. Initial fixed50 landmark,
observed-detail and pixel gradient alignment with the finite change is weak
(0.0204007062,0.0300057107,0.0598935766); initial preservation gradients are zero.
This establishes weight movement, not useful learning or a unique cause. No
optimizer moments/per-step losses were saved; they are not reconstructed.
Analysis0.792s and independent arithmetic0.079s use no neural/gradient calls.

A distinct endpoint diagnostic is now independently verified but unrun. It
compares original/stopped50 decoder states on100 photographic TRAIN cases:
fixed50 not used by the first50 updates and firstfive optimized references per
source in the frozen schedule, each with clear plus four degraded profiles.
Selection is metadata-only; these cohorts are not held-out evaluation or CCTV.
Source labels do not establish ethnicity. No native or final pixels are opened.

Architecture inverse source/AST, all seven losses, initial50 normalization,
original/fixed decoder and recognizer and every preservation gate stay exact.
There are280 component-gradient queries, zero optimizer/parameter updates,
backwards or epochs. Endpoint norms/conflicts/directions are diagnostic, not an
AdamW trajectory, qualification or the next actual-training recipe. All57 tensor
partitions, all200 raw/300 PNG compositions and40 frozen CPU replay cases have a
prospective independent return checker; it performs no local differentiation.

Protocol SHA256:62d090226e737151083fb388af04c99ee1813aaee37f2e65ec5420e852155071
Execution archive SHA256:90e9d8ee576819602afcc3817ab70b1f5dd53c5d38fb167fe94bcc73526b38c9
Archive 251,114,281bytes / 266regular files.
The verifier checks metadata/portable assets, Python3.10 and read-only Bash,
Windows rejection before neural imports, three wrong scopes, nine unsafe
archives, nine corrupted saved-array cases and five single-source gcloud commands.
Synthetic fixtures are boundary tests, not VM evidence. No VM connection or
diagnostic/training execution occurs here. Human upload/install/tmux/run/download
remains required on the existing L4/g2-standard-4; only its venv is needed.

Require2GiB free. Worker600s/external630s+30s kill grace; export300s/external330s
+30s grace; allocated VRAM20GiB; return512MiB uncompressed. Estimated diagnostic
2–6minutes plus1–3minutes export. Historical free space is not a current reading;
no cleanup occurs. Any diagnostic failure must remain and be downloaded too.
The original V40 failure and completion fusion-off rejection remain binding.

All14 DGP-primary app/checkpoint/design bindings remain exact. Prior documents,
original checkpoints, data/splits/provenance, caches/local backup and all gate
failures remain. The full earlier handoff is archived and preserved below.
V38 development preservation/native-quality failures remain; reserved45
identities/58 crops remain unopened. No native PSNR/SSIM, exact hidden identity,
ethnicity or Zamboanga performance is claimed. Useful native restoration and
all visible features, seven covering families with separate automatic/assisted
quality, independent final review and the qualified DGP-led app flow are still
required. Goal active/incomplete.

[Five manual diagnostic steps](<C:/xampp/htdocs/YEAR 4/Testing/CCTV_DGP_V40_LEARNING_SIGNAL_V1_VM.md>)
[Evidence and diagnostic limits](<C:/xampp/htdocs/YEAR 4/Testing/CCTV_DGP_V40_LEARNING_SIGNAL_V1_REVIEW.md>)


**Research milestone — 8 October 2026: V40 human return independently audited, early structure failed; completion feature-fusion variant also rejected.**

V40 ran manually on the existing L4 and stopped at 50/800 updates. Its unchanged
prospective independent checker confirms 0.0106079234% structure gain against
the retained 1% requirement. All 17 preservation groups pass. This is a learning
quality stop; export exit code 0 packaged failure evidence. No 800 result,
resume, unchanged repeat, development qualification or application promotion.
The original trained DGP remains primary in the unchanged local application.

The 1,138,304,426-byte return, sidecar and export receipt match SHA256:
58706c14674e34a2a60308b94338f02b8f083d02a9fc0b2a94af8d6344745ac6.
Protocol b21597d7c50dc39ee7cb2bd70d7d0ad9f5cd967661b6f5e78413a7e4eb121010.
All 27,451 regular files are safely imported with exact manifest hashes;
returned code is not executed. The 751.24s audit recomputes all 3,905 PNG,
mean-only PNG and vector comparisons per complete snapshot 0/50, all groups and
50 saved raw preview compositions per state. The other 3,855 raw stages per
state are hashed on the VM but not retained; no all-raw audit is claimed.

One hundred frozen CPU replays pass: raw error <=2.2947788e-6, PNG difference
one byte and vector error <=2.7939677e-7 within prospective limits. Original DGP,
fixed decoder and recognizer states stay unchanged; the separate learned decoder
changes. VM has 50 backwards/updates, 1,027.71s total, 492.89s fit including its
update50 snapshot and 9,504,983,552 bytes allocated VRAM within finite limits.
The agent performs no local gradients, optimizer updates, actual training or
VM operation. The stopped checkpoint, original packet and failed gate remain.

All 50 fixed TRAIN previews and 250 exact comparison cells were actually viewed
on 10 sheets at original 256-cell detail. Eyes, nose, mouth, outline and overall
appearance remain in scope. Stopped50 stays near the retained DGP, with slight
tone/texture changes and no convincing new facial definition. No output-based
input relabeling hides the failure. These are TRAIN, not final evaluation.

Fifty paired batches optimize 250 cases from 50 TRAIN references. The 200
degraded optimized cases gain only 0.007422% structure; 2,924 not-yet-used TRAIN
cases gain 0.010873%. Forty degraded fixed previews gain 0.014370%; none of the
50 fixed previews is directly optimized in these updates. Weak learning is not
confined to unseen cases. This does not establish a unique architecture/loss/
optimizer cause. Per-update component losses and optimizer moments are absent;
they cannot be reconstructed from two checkpoints. V39's valid nonzero-gradient
proof does not guarantee useful learning under this recipe. Further learning
needs a separate justified diagnostic, retaining gates and the manual workflow.

The human selected “apply the best approach” for completion architecture review.
A single frozen w=0 ablation disables direct encoder-feature fusion. Initial
w=1 cached parity is exact; the matched neural input, encoder, prediction scores,
codes and quantized latent are identical. Actual candidate fusion calls are
zero. Same two assisted conditioning masks, same final removal support/two-pixel
margin, no output-driven annotation, eligibility, seed/weight or source search.

The completion worker takes 139.81561s (external 145.19614s): 29 frozen forwards,
28 estimates, four exact empty controls and four unchanged input-only exclusions.
There are no DGP/detector, gradient/backward/optimizer or VM calls. All original
states stay unchanged. The 108 artifacts total 164,378,715 bytes within 256MiB;
worker600s/external630s+30s limits remain. Protocol:
53c038ea900c1b8d0c602679435cf794b6770708196183ec6a1773437becf95b.

The original checker stops at replay scores because its saved C-contiguous
input changed the actual channels-last layout. Raw RGB/codes stay exact; scores
differ by up to0.0034141541. A distinct12.05021s two-call frozen diagnostic proves
the layout cause and exact replay with the actual layout. R1 changes only
layout/receipt filename, inverse source/AST exact and every exact rule unchanged.
It passes in13.02887s. Original checker/failure are preserved, without tolerance
changes or replacement estimates. All520 sources/108 artifacts,28 compositions,
160 sheet cells,5,483,088 visible and1,075,314 protected bytes are verified.

All32 completion outputs and32 w=1 counterparts were actually inspected on eight
sheets. Pale eye patches/clouded glare and hand/scarf joins remain; w=0 adds or
strengthens red/bright lower-face patches. It is rejected for adoption. Automatic
outputs:zero; automatic and assisted quality remain separately unqualified.
The legacy native suffix means original photograph, not CCTV. Assistance is
reused for synthetic pairs; pretraining overlap and true hidden appearance are
unknown. Exterior copied fragments are separate from generated defects. Clear
glasses/ordinary hair stay intact. All seven covering families remain required.

The first documentation recorder correctly stops before handoff edits when the
new V40 return begins arriving. Its original source/stop receipt are retained;
the completed human download and independent return audit supersede absence.
Both reports reflect the current evidence. The full previous handoff is archived
and its complete body preserved below, with earlier unrun statements historical.

All14 app/checkpoint bindings/design stay unchanged: original DGP primary,
Auto/On/Off override, input and removal-area review, original/mask/estimate,
PNG and bundle downloads. No new browser-quality claim is made. Checkpoints,
splits, provenance, source terms, research caches/local backup and all failed
gates remain. V38's development failure and weak native clarity remain binding.
Paired synthetic TRAIN and native unpaired CCTV remain separate. No new native
acquisition, hidden identity, ethnicity, native PSNR/SSIM or Zamboanga performance
is claimed. Reserved45 identities/58 crops remain unopened. Useful reviewed
development restoration, all covering families, independent final review and
full qualified DGP-led flow remain outstanding. Goal active and incomplete.

[V40 audited failure](<C:/xampp/htdocs/YEAR 4/Testing/CCTV_DGP_SPATIAL_FIT_V40_RESULTS.md>)
[All50 visual observations](<C:/xampp/htdocs/YEAR 4/Testing/outputs/cctv_dgp_spatial_fit_v40_failure_review/visual_review.json>)
[Completion architecture findings](<C:/xampp/htdocs/YEAR 4/Testing/CCTV_DGP_COMPLETION_FEATURE_FUSION_OFF_V1_RESULTS.md>)


**DGP milestone — 8 October 2026: V39 gradient proof independently passed; full-TRAIN V40 spatial learning packet verified and unrun.**

The human returned V39's78,588,659-byte archive. Its sidecar and export receipt
match SHA25622eb58d246891e946a6b3aeee71ae3e3923f6ff13717e679c460f07b6c1d777e.
The distinct L4 diagnostic takes33.64275s, with70 saved component queries,
zero optimizer/parameter updates, zero backwards and zero epochs. All57 new
decoder tensors receive finite nonzero improvement gradients; norms range from
7.331710e-7 to0.02951934. Initial raw/PNG output is exactly the original on50
exposed TRAIN cases. Four preservation values/gradients are exactly zero.
Original DGP, both decoder copies and recognizer states remain unchanged.
Peak allocated VRAM is2,116,979,200 bytes. No trained checkpoint is created.

The unchanged prospective independent checker takes23.54897s. It verifies273
returned regular files, ten7x17,952 float64 gradient matrices, their exact sum,
scalar sums, norms, Gram and all57 partitions, all50 initial raw/PNG pairs and
normalizer readback. Ten metadata-selected frozen CPU replays cover both sources
and five profiles: maximum raw error2.145767e-6, PNG difference one byte and
target-vector error2.095476e-7 within fixed compatibility limits. Saved VM
gradients are audited without independently differentiating locally. There are
zero local gradients, optimizer updates or VM calls. Connectivity supports a
separate capacity test; no useful-restoration or independent-final pass follows.

V40 learns this SAME spatial decoder across all781 existing TRAIN references
and3,905 cases:1,950 dataset/asian_faces cases and1,955 FFHQ-thumbnail cases.
Source labels are not ethnicity. The original CNN, fixed initial decoder and
recognizer are frozen. Only17,952 parameters in57 new tensors are optimized.
The fixed V31 metadata schedule pairs one clear and four degraded views in each
batch, covering every case once in781 updates and19 additional distinct reference
batches, for800 updates maximum. Same50 normalizer inputs/targets/support bytes,
seven original losses, filter and PNG metric definitions and quality thresholds.
AdamW3e-4/weight-decay0.01/clip1 is fixed for the NEW initialized decoder;
there is no rate sweep, continuation of V38, pretrained-restorer target, display
sharpening or target/source/person/profile-conditioned inference path.

At50, the1% structure requirement and all17 delivered-PNG preservation groups,
both-source nonregression and20% brightness-only limit must pass. Failure stops
and exports evidence. Final800 requires10% structure with unchanged preservation.
The thresholds are retained; checking preservation at50 also provides an earlier
stop. Complete snapshots0/50/800 save all3,905 PNGs, mean-only PNGs and vectors,
plus all50 prospective raw previews. The other3,855 raw stages are hashed but
not retained. The prospective independent return audit covers all PNG metrics,
all17 groups, exact raw compositions for those50 previews and50 frozen CPU
replays per complete snapshot. It does not claim every raw float is retained.
All50 preview outputs require visual review before separately frozen development
work. Necessary TRAIN capacity never overrides development or final failures.

The self-contained V40 packet has5,491 regular files and444,516,831 compressed
bytes, including265,702,338 bytes of fingerprinted input/target/support assets,
original frozen weights, code and the untrained seed. Existing venv only;
no deleted historical worker or cache is required or executed. Protocol SHA256:
b21597d7c50dc39ee7cb2bd70d7d0ad9f5cd967661b6f5e78413a7e4eb121010.
Archive SHA256:
b8113316de65ee965ffe71303adae6caf5446be7f853ab95c11242e776036b0f.

The41.6913s independent packet audit verifies5,490 assets, every tar stream,
all3,905 input files/781 references/800 batches, exact50 proof data, V39 architecture
equivalence except distinct manual root/name, identical filters/losses and AST-exact
PNG metric body. Five synthetic preservation failure cases, nine unsafe-return
archives, three scope rejections, two invalid schedules and actual pre-neural
Windows rejection pass. Python3.10 parsing, read-only Bash syntax and five
single-remote-source gcloud commands pass. Synthetic tests are not learning proof.

The first unverified draft is archived. Review corrected only the learned-stop
checkpoint metadata and nonpreview raw-stage count; recipe/gates did not change.
Git Bash's sandbox signal-pipe failure is retained. A read-only Bash -n parser
outside the sandbox passes in0.542123s without executing the launcher or any VM
operation. There was no automatic approval rejection.

V40 is PREPARED AND VERIFIED, NOT RUN. Follow the five manual upload/install/
tmux/launch/download steps on the existing idle L4/g2-standard-4. Require6GiB
free AFTER installation. Prospective estimate15–40 minutes learning and3–15
minutes export, not measured V40 timing. Cache900s, fit3600s, worker4500s/
external4800s+30s grace, export900s/external930s+30s grace, VRAM20GiB and3GiB
uncompressed return are bounded. Cache and update20 projections use1.25 safety
factors and two measured remaining snapshot allowances. No automatic repeat,
historical pilot, training, VM connection or cleanup is launched here.

V38's one paired-development ArcFace failure and weak native clarity remain.
All previous checkpoint/split/source/terms/provenance/cache/backup and gate
failures remain. The14 app/checkpoint bindings and design are unchanged, with
the original trained DGP primary, Auto/On/Off, mask review, original/mask/result,
PNG and bundle downloads. No new browser or app-quality claim is made. Separate
automatic/assisted completion is still unqualified across masks, sunglasses,
strong lens glare, hands, obstructing hair, scarves and objects. Preserve clear
glasses, non-obstructing hair and visible appearance, with only documented margin.
Request clearer/less-covered input only when usable information is insufficient.

Photographic TRAIN,520 paired development and24 native unpaired CCTV crops remain
separate. No hidden-identity, native PSNR/SSIM, ethnicity or Zamboanga-performance
claim is made. Real Zamboanga CCTV is absent;45 reserved identities/58 crops stay
unopened. Useful reviewed development outputs, independent final review, all
covering families and the full DGP-led app remain required. Goal active/incomplete.

[V39 independent results](<C:/xampp/htdocs/YEAR 4/Testing/CCTV_DGP_SPATIAL_DECODER_V39_RESULTS.md>)
[V40 five manual steps](<C:/xampp/htdocs/YEAR 4/Testing/CCTV_DGP_SPATIAL_FIT_V40_VM.md>)
[V40 independent packet audit](<C:/xampp/htdocs/YEAR 4/Testing/outputs/cctv_dgp_spatial_fit_v40_preparation/independent_packet_audit.json>)

The complete preceding handoff follows; its V39-unrun statements are historical.


**DGP milestone — 8 October 2026: distinct V39 spatial decoder initial proof audited; zero-update manual VM diagnostic prepared.**

The V38 quarter-step development failure remains binding: one FFHQ-thumbnail
compound-profile ArcFace regression on520 paired cases, and no convincing added
clarity in the24 native crops. Smaller steps are not selected from development
outcomes. All original V36/V38 failures and prior capacity stops remain. The
following V38 milestone preserves that evidence and the separate completion
display comparison; neither is an adoption or independent-final quality pass.

The next hypothesis tests a different learned spatial path within our own DGP.
Observed RGB, retained DGP RGB,64 lateral and32 reconstruction feature channels
form102 channels at256. A new16/32-channel two-scale decoder uses local gates,
channel normalization and a spatial shortcut. It subtracts a separately frozen
initial decoder response, mean-centers the half-tanh residual on observed pixels,
adds it to the original DGP and clamps. Outside-support camera pixels are exact.
Original weights/statistics stay frozen. No pretrained restorer weights/targets,
display sharpening or input/source/person/target-conditioned routing is used.

The17,952 parameters in57 new tensors are untrained. A separately transferred
CPU seed asset is initialization, with zero local learning. The46.123-second
initial proof covers all50 exposed V27 TRAIN inputs: exact fresh raw and PNG
parity, partial-support preservation, five pre-neural invalid-input rejections
and actual Windows rejection of learning enablement. It makes101 original DGP
and51 forwards through each decoder copy. No local gradient, optimizer or trained
checkpoint occurs. The9.282-second independent checker verifies90 sources,
203 artifacts, all50 raw pairs/100 PNG compositions and57 seed tensors. Ten
metadata-selected frozen CPU replays are exact; historical cache maximum raw
difference is2.38419e-6 within1e-5. Gradient connectivity is not established.

Primary NAFNet research motivates testing simple gated restoration blocks; it
does not guarantee forensic identity, CCTV usefulness or the cause of our
failures. This is a different experimental spatial decoder within the retained
own-trained feedforward DGP, with disclosed lineage. It uses no NAFNet weights
and makes no published-architecture or transferred benchmark-performance claim.

V39 is a separate70-query, zero-update gradient proof on the same50 photographic
TRAIN cases, ten paired-five-profile batches and seven original loss terms.
Every57 new tensor must have finite nonzero summed improvement gradients.
All four initial preservation values/gradients must be exactly zero; original
DGP, frozen initial decoder and recognizer must remain unchanged. There are no
optimizer/parameter updates, epochs or trained checkpoint writers. A returned
proof and independent audit are needed before a separate finite training recipe.
The1%-at50/10%-at800 and17-group/source/brightness gates remain unchanged.

The self-contained packet has142 regular files,235,212,632 uncompressed bytes
and213,203,706 compressed bytes. It copies verified original weights, code,
untrained seed and existing TRAIN input/target/support/cache assets. Only the
existing VM venv is needed; no deleted historical worker or cache is recreated
or executed. Protocol SHA256:
d1775d67c8c80ddad9372e5af47be6ce987ca3fed7749defcdbdc190fb1153a4.
Execution archive SHA256:
d2093fc136d5305db5071a53be7fbdb0722e5505026051c68501b57959cc29bc.

The4.874-second packet checker verifies141 assets/152 local bindings, every
archive member, all50 cases/10 references/57 tensor layouts, AST-exact original
loss/filter definitions and unchanged quality gates. Three scope rejections,
actual Windows pre-neural rejection, nine unsafe archive cases and seven saved
gradient rejection fixtures pass. Nineteen Python3.10 sources, read-only Bash
syntax and five single-remote-source gcloud commands are checked. Synthetic
fixture arrays are not VM gradient or training evidence.

Two local audit-environment failures remain: Git Bash's sandbox signal pipe
restriction and Python313 TemporaryDirectory access denial. The Bash -n parser
passes without executing the script. A separate5.025-second R1 runner changes
only fixture directory creation to inherited workspace access and partial-log
routing; inverse AST comparison verifies all scientific checks unchanged.
Original checker, packet, protocol, gates and failure receipts are preserved.
The retained fixture directory is explicitly synthetic, separate from VM data.

V39 is PREPARED, NOT RUN. Follow the five manual Google Cloud SDK/SSH/tmux steps
on the existing idle NVIDIA L4/g2-standard-4 at ~/forensic-dgp. Require2GiB free.
Prospective estimate2–6 minutes diagnostic plus1–2 minutes export; this is not
measured V39 VM timing. Worker600s/external630s with30s kill grace, export300s/
external330s with30s grace, allocated VRAM20GiB and uncompressed return192MiB
are bounded. Any stop exports evidence; no automatic follow-on or retry.

The14 app/checkpoint bindings and existing design are unchanged. The original
trained DGP remains primary with Auto/On/Off, mask review, original/mask/result,
PNG and bundle downloads. Prior inline Playwright functional proof is retained;
no new UI or browser claim is made. Completion remains separate and unqualified
for automatic/assisted masks, sunglasses, strong lens glare, hands, obstructing
hair, scarves and objects. Preserve clear glasses/non-obstructing hair and
visible appearance; request clearer/less-covered crops when information is
insufficient, not as an explanation for poor model outputs.

TRAIN photographs, paired520 development and native24 unpaired CCTV are separate.
No native PSNR/SSIM, ethnicity, hidden-identity or Zamboanga-performance claim is
made. There are no real Zamboanga samples. Reserved45 identities/58 crops remain
unopened. Checkpoints, source/split/terms/provenance records, caches/local backup
and gate failures remain. Useful development outputs, independent final review,
all seven covering families and the reviewed DGP-led app remain required. Goal
active/incomplete. No VM connection, cleanup, diagnostic or training was launched
by the agent in this milestone.

[V39 five manual steps](<C:/xampp/htdocs/YEAR 4/Testing/CCTV_DGP_SPATIAL_DECODER_V39_VM.md>)
[V39 spatial design and primary research](<C:/xampp/htdocs/YEAR 4/Testing/CCTV_DGP_POST_V38_SPATIAL_DECODER_REVIEW.md>)
[Independent packet audit and environment correction](<C:/xampp/htdocs/YEAR 4/Testing/outputs/cctv_dgp_spatial_decoder_v39_preparation/independent_packet_audit_r1_execution.json>)

The complete preceding handoff follows. No preceding failure or scope is removed.


**DGP milestone — 8 October 2026: V38 returned and independently audited; frozen quarter-step development check completed.**

The human returned V38's 388,426,632-byte archive, SHA256
c1403d67ba0a66a1200285bcab291cd87dc9d6d7bf0518534f3b07a12ebff876.
The prospective checker verifies all2,135 regular files, all500 raw/PNG/vector
measurements,170 group aggregates,210 projection rows and100 frozen CPU outputs.
Local audit runtime is240.167 seconds inside a241.612-second supervised run.
The111.690-second L4 probe makes four reset trials, zero new gradients, optimizer
updates, committed trajectory updates or epochs, and no new checkpoint.

The quarter and eighth steps pass the unchanged measured PNG/source/brightness
checks on both50-case TRAIN cohorts. Full/half failures remain: six V38 PNG
failures in total, alongside V36's11. The quarter gains are0.068478% and0.116937%,
not recovered-identity percentages. All100 original raw/PNG pairs equal V36.
All20 comparison pages were actually reviewed:100 cases,500 model output cells
and200 input/target cells. A separate checker verifies all700 cells. Visible
eyelids, nostrils, lips and teeth remain weak; passing small steps do not establish
useful additional clarity. TRAIN preservation is insufficient for adoption.

The largest passing TRAIN step,0.25, was fixed before new development inference.
A disposable frozen copy has state
b7aad53d93d4fa58ca826be6162667fff2faf85578da4f421e938b8711bc49ba.
Single-input CPU parity was checked on all100 already-exposed TRAIN cases against
the VM trial. No local fitting, autograd, optimizer or checkpoint writing occurs.
This is a finite displacement, not an800-update trained replacement.

All24 frozen ChokePoint C1 native crops were reviewed on six pages, including
resize, retained Phase3, original DGP, declared CodeFormer and quarter/Auto arms
on identical256 inputs. The saved-output audit verifies all144 cells and every
raw composition/padding/Auto alias. Auto chooses11 quarter and13 resize outputs;
that input-only suggestion is not structure qualification. Quarter remains very
similar to original DGP and adds no convincing native clarity. All24 input-only
usable labels remain; weak model outputs do not make these inputs unusable.
Native evidence is unpaired: no PSNR, SSIM or recovered-identity accuracy.

The separate paired photographic development check retains all520 cases/104
references, with resize and original/quarter DGP. Its runtime is619.479
seconds under1200 internal/1290 external seconds. All520 raw/PNG/vector/metric
rows,17 groups and200 prospective preview cells are independently recomputed.
It records1 preservation failures. Degraded high-frequency gain is
0.097128% overall; source-specific figures remain separately reported.
The actual image review covers50 prospective preview cases;
all520 numeric checks do not imply all520 images were visually reviewed.
No development-selected scale is substituted and no final identity is opened.

The separate completion display comparison changes only the existing two-pixel
margin. It lowers the measured boundary jump in all28 nonempty assisted cases,
but does not fix central glare, eyewear/covering remnants or anatomy. All32
delivered variants on eight pages were actually inspected, with four prior
input-only rejections retained. The independent audit verifies517 sources,
72 artifacts,160 cells,28 old raw compositions and exact core/visible/protected
bytes. Automatic outputs generated here: zero. Automatic and assisted whole-scope
quality remain separately unqualified. The default-Python missing-cv2 audit
failure is retained; only execution in the project venv corrected that runtime.

The14 current app/checkpoint bindings remain unchanged. The original own-trained
DGP is still primary with Auto/On/Off, mask review, original/mask/result and PNG/
bundle downloads. Existing bundled inline Playwright functional proof is retained;
no UI edit or new browser claim is made here. All original checkpoints, splits,
source/provenance/terms records, caches/local backup and failed gates remain.
The1%-at50/10%-at800 capacity gates are unchanged and are not passed by V38.

These are exposed development photographs or native ChokePoint capture, reported
separately. Source labels are not ethnicity. There are no real Zamboanga samples.
Reserved45 identities/58 crops remain unopened. Useful native restoration, all
seven covering families with separate automatic/assisted quality, independent
final review and the complete reviewed DGP-led app remain required. Goal active.
No VM connection, cleanup, new training launch or historical pilot occurs here.

[V38 return findings](<C:/xampp/htdocs/YEAR 4/Testing/CCTV_DGP_DELIVERED_MARGIN_PROBE_V38_RESULTS.md>)
[Frozen quarter development findings](<C:/xampp/htdocs/YEAR 4/Testing/CCTV_DGP_V38_QUARTER_DEVELOPMENT_RESULTS.md>)
[Completion margin findings](<C:/xampp/htdocs/YEAR 4/Testing/CCTV_DGP_COMPLETION_MARGIN_FEATHER_V1_RESULTS.md>)

The complete preceding handoff follows. Its V38-prepared-only language records
the earlier milestone and is superseded by this audited human return.


**DGP milestone — 8 October 2026: V37 diagnostic independently audited; V38 finite PNG-margin probe prepared.**

The human returned V37. Its 43.838-second L4 diagnostic made 300 saved coarse
gradient queries, zero optimizer/parameter updates and no new checkpoint.
The independent local audit verifies all 431 regular files, 100 PNG guard
measurements, 102 group aggregates and 20 frozen CPU outputs. The audit takes
164.784 seconds inside a 165.947-second bounded supervisor. No returned code,
local autograd or optimizer is executed. Packaging completion does not imply
restoration quality, training success or preservation qualification.

All 100 original raw files and 100 original PNG files are byte-identical to V36.
There are no new restoration outputs to visually qualify. The independent saved
array analysis finds the PNG-valued coarse directions predict zero of V36's
11 measured PNG failures. An identity surrogate is not the true derivative of
discrete PNG conversion. V37 changes no adopted loss, decoder or application.

A new finite hypothesis retains every original raw/restoration check and all
65 V36 clearances. It adds 102 PNG coarse checks with 97 positive empirical
margins from all 408 group/metric/scale comparisons of V36's four trials. The
margin is twice the maximum positive scale-normalized finite-minus-coarse loss
discrepancy. This is calibration on TRAIN design data, not a validated error
bound or independent evaluation. All 210 rows and original clearances are
independently reassembled, with scaling/order/infeasibility/invalid-input checks.
The direction's magnitude is 1.08019 times the original proposal, within the
unchanged limit of 2. Its certificate permits a disposable finite probe only.

V38 resets the original DGP separately for scales 1, 0.5, 0.25 and 0.125 and
measures 100 original plus 400 trial outputs. It makes zero new gradients,
optimizer updates, epochs, committed trajectory updates or new checkpoints.
The image/metric/CPU replay definitions are verified unchanged by AST. All 17
PNG groups, source gains, brightness limit and prior capacity/development gates
remain. A future actual training pilot needs separate justification and manual
execution; the finite probe cannot satisfy 1% at50 or 10% at800 training gates.

The nine-file 12,234,380-byte transfer packet passes the independent 32.988-second
audit: 210-row geometry, 230 V34/28 V37/16 V36 saved bindings, original108 rows
and65 margins, 408 calibration observations, Python3.10/Bash syntax, three scope
rejections, eight archive rejection fixtures and five single-source gcloud
commands. The direction differs from failed V36. Protocol SHA256:
caea39469005b7066d7cdbd74e073ba674b9b9b6fb9324f3110fe827a06948ab.
Execution archive SHA256:
6096168f41f4be1fbaae9490340c7bdab70140d744b1db3de7194bab90ea38b0.

V38 is PREPARED, NOT RUN. The five-step guide uses the human's existing L4/
g2-standard-4 VM and tmux. Require 6 GiB free; estimate 3–6 minutes for the
probe plus 1–3 minutes for export. Worker/external caps are900/930 seconds,
export300/330 seconds, with30-second kill grace, VRAM20GiB and uncompressed
return768MiB. Missing or mismatched historical dependencies stop; neither
historical recipes nor new training are automatically launched. Return all
three files, including failure evidence, for independent audit and all100
case visual review. No VM connection or cleanup was performed in this turn;
historical free-space measurements are not a fresh inventory.

The14 current app/checkpoint hashes are unchanged. The original own-trained
DGP remains primary with existing Auto/On/Off selection, mask review and PNG/
bundle downloads. The prior inline Playwright functional checks are preserved;
there was no UI change here. Completion conditioning defects remain separately
recorded, with both automatic and assisted covering quality unqualified.
Original checkpoints, source/splits, caches/local backup and all failures remain.

The two50-case diagnostic cohorts are photographic TRAIN design data; the
historical 'unexposed' name is not a held-out claim. Source labels describe
acquisition, not ethnicity. Native24-crop evidence remains unpaired, paired
development520 cases/104 references remains separate, and reserved45 identities/
58 crops remain unopened. No real Zamboanga CCTV samples exist. Useful native
structure, all seven covering families, independent final review and the
complete reviewed DGP-led workflow remain required. Preserve visible appearance,
clear glasses and non-obstructing hair; request clearer/less-covered crops when
usable information is insufficient. Goal active/incomplete.

[Audited V37 findings](<C:/xampp/htdocs/YEAR 4/Testing/CCTV_DGP_DELIVERED_GUARD_GRAD_V37_RESULTS.md>)
[V38 five exact manual upload/install/tmux/launch/download steps](<C:/xampp/htdocs/YEAR 4/Testing/CCTV_DGP_DELIVERED_MARGIN_PROBE_V38_VM.md>)

The complete preceding handoff follows. Its V37-prepared-only language describes
the earlier milestone and is superseded by this audited human return.


**DGP milestone - 8 October 2026: V36 independently audited and rejected; delivered-PNG path verified; V37 zero-update diagnostic prepared.**

The human's returned V36 archive contains2,135 verified regular files. The
independent local audit verifies all500 saved raw/PNG outputs,170 group receipts,
108 projection constraints and100 frozen CPU replay outputs. Audit runtime is
226.393 seconds; its bounded external supervisor completes in227.784 seconds.
No returned code is executed and no local gradients or optimizer run. VM V36
takes92.611 seconds for four independently reset finite displacements, zero
optimizer updates, zero new gradient queries and no checkpoint or trajectory.

Preservation failures fall from V35's41 to11, but no V36 scale passes both
TRAIN cohorts. At scale1, structure gains are0.277816% and0.410191%; one cohort
has no PNG failures and the other has two. Across scales there are ten delivered
ArcFace and one SSIM group failures. Positive source gains and brightness passes
do not override preservation. All100 V35/V36 original raw and PNG files are
byte-identical. Existing1% at50/10% at800 capacity gates and all development
qualification failures remain. No V36 trial is promoted or resumed.

All100 TRAIN cases and20 pages are actually viewed at original256-pixel cell
detail; a separate audit verifies all700 input/target/output cells. Severe blur
and compound inputs still lack clear eyelid, nostril and lip/tooth structure;
the four trials remain visually close to the original DGP. Softness alone is
not a failure. This is implementing-assistant development review, not independent
final review. Both cohorts are now design data. The historical 'unexposed' label
does not make the second cohort held-out development or final evaluation.

Nine of11 PNG failures have no same-group raw-metric failure; two fail both.
A frozen-output arithmetic diagnostic reproduces all500 actual PNG-valued
forwards exactly, with maximum MSE error4.163336e-17 and SSIM error2.220446e-16.
Its separate7.170-second check verifies1,154 source bindings, all500 encodings,
all11 classifications and four boundary/quantization/filter fixtures. No model,
neural call, gradient, backward, optimizer or VM connection runs for this check.
Primary research and the official floor derivative motivate an explicitly
approximate surrogate; they do not guarantee nonlinear DGP preservation.

After the required architecture discussion, the user selected the best approach
with research where needed. V37 therefore measures the delivered PNG guard path
before any new update design. It keeps the original own-trained DGP, the same
100 photographic TRAIN cases, five-case context and23 tensors. The maximum is
300 coarse autograd gradient queries, zero optimizer/parameter updates, zero
epochs and no new checkpoint. Custom backward machinery is used within grad;
zero backward API calls does not mean zero VM gradient computation. All17 finite
PNG checks, useful-structure gates, source and brightness requirements remain.

The four-file35,166-byte packet is independently verified: Python3.10 syntax,
412 V36 dependency hashes,26 local bindings, Windows/root/instance rejection,
eight unsafe-archive tests, six group tests and read-only Bash syntax. Windows
initially denied Bash's signal pipe; the original checker/failure is retained
and a distinct R1 consumes the successful read-only external Bash-n evidence.
The frozen packet was not changed. Protocol SHA256:
7e3c9ae20fe2f4faa2d0e83ba628294c5b91a8d0f43093ea53bffcc6320d0090.
Execution archive SHA256:
444bea1b2f5f5b79a0d0912b2f06e5a01132a015525e66ff58af30ae38f07503.

V37 is PREPARED, NOT RUN. Actual gradients/training stay under the human's
manual L4/g2-standard-4 tmux workflow. Require6GiB free; expected diagnostic
3-10 minutes, export1-3 minutes. Worker/external stops are900/930 seconds plus
30-second kill grace, export300/330 seconds plus30-second grace, allocated VRAM
20GiB and uncompressed return1.5GiB. A forward mismatch or any finite/state/count/
resource violation stops and exports failure evidence. No historical worker,
follow-on optimizer, cleanup or VM connection is executed by the agent here.

The own-trained DGP remains primary and all14 app/checkpoint bindings are
unchanged. Existing design, Auto/On/Off, mask review and PNG/bundle downloads
remain. The earlier real inline Playwright flow is preserved; no frontend
change requires another browser run here. Completion conditioning improvements
and failures remain separately recorded and neither automatic nor assisted
whole covering-family scope is qualified. Original checkpoints, source/splits,
research caches/local backups, reports and all failure evidence remain.

The24 unpaired native development crops and520 paired synthetic development
cases are unchanged; the45 reserved-final identities/58 crops remain unopened.
No real Zamboanga CCTV samples or ethnicity/local-performance claims follow
from these photographic TRAIN diagnostics. Useful native structure, all seven
covering families and independent final review remain. Goal active/incomplete.

[V36 audited findings](<C:/xampp/htdocs/YEAR 4/Testing/CCTV_DGP_FINITE_CLEARANCE_PROBE_V36_RESULTS.md>)
[PNG-path evidence and primary research](<C:/xampp/htdocs/YEAR 4/Testing/CCTV_DGP_V36_DELIVERED_METRIC_REVIEW.md>)
[V37 five exact manual upload/install/tmux/launch/download steps](<C:/xampp/htdocs/YEAR 4/Testing/CCTV_DGP_DELIVERED_GUARD_GRAD_V37_VM.md>)

The complete preceding handoff follows. Its V36-not-returned language describes
the earlier milestone and is superseded by this audited human return.


**Completion milestone - 8 October 2026: conditioning isolated; several generated remnants reduced; remaining copied boundaries and glare retained.**

The complete 36-case exposed photographic gallery receives a changed, finite
conditioning test. All final reviewed removal masks stay fixed; only the frozen
model's hidden context changes to the union of two existing reviewed masks.
All nine input pages are actually inspected before protocol freezing, with a
separate audit of 32 masks, 18 pair memberships and 132 preview cells. One fresh
current-mask Off call reproduces its cached PNG and both raw stages exactly.

The existing local CPU runtime completes 29 frozen CodeFormer completion
forwards in 152.31 seconds: one parity check plus 28 changed-context trials.
Four empty controls bypass exactly and four input-only exclusions remain.
The unchanged component state is
255e18cbd071e735e5e9789ba83dc010e4872a415048e33611c02c113590afad.
There are zero DGP/detector forwards, gradients, backwards, optimizer updates,
new checkpoints or VM calls. Worker/external stops are 600/630 seconds; artifacts
are 137,774,023 bytes against a 256 MiB cap. All failures and historical recipes
remain; no failed immutable inference or training is repeated.

The separate 6.28-second saved-output audit verifies all actual neural inputs,
28 raw-to-PNG compositions, 32 outputs, 160 page cells, 397 source bindings and
107 artifacts. The 5,483,088 bytes outside final support and 1,075,314 protected
bytes are exact. This includes clear-frame/ordinary-hair context that was hidden
from the model but must still be copied into the delivered image. Display
selection uses the same final support and no new enhancement.

All 32 estimates and their same-final-support baselines are actually viewed at
256-pixel cell detail. Several estimates contain less regenerated dark eyewear,
obstructing hair, petal or green/brown leaf material. Some central hand/scarf
features are cleaner or remain plausible. Strong glare, upper-hand notches,
lower hand/jaw contacts and several peripheral joins remain unqualified. Softness
or unknown hidden appearance alone is not treated as failure. A saved-pixel
recount confirms that both peripheral flower-tip windows are entirely copied;
the upper-hand notch window is mostly copied. These post-output windows are
processing diagnostics, not new covering labels, mask changes or hidden truth.

The union needs TWO historical assisted masks. It is not a validated automatic
single-mask context rule, and no new automatic output is generated. The source
photo assistance is reused for its degraded pair; these are previously exposed
photos, not native CCTV or independent final images. Pretraining overlap and
hidden appearance remain unknown. No new hidden PSNR/SSIM, recovered identity,
ethnicity or Zamboanga-performance claim follows. Neither automatic nor assisted
whole-scope qualification is established; no variant is adopted into the app.

The own-trained DGP remains the main restorer. All 14 current app/checkpoint
bindings and the existing design, Auto/On/Off routing, review-before-generation
and downloads remain. No frontend change requires another browser run here.
V35's preservation failures remain binding. V36's verified changed-direction
packet remains available for manual L4/tmux execution; no V36 return is present
locally at this milestone, and the agent launches no VM work. Useful native DGP
structure, consistent automatic/assisted covering outputs and independent final
review still remain. Goal active/incomplete.

[Completion conditioning findings and all-case evidence](<C:/xampp/htdocs/YEAR 4/Testing/CCTV_DGP_COMPLETION_CONDITIONING_UNION_V1_RESULTS.md>)
[V36 exact manual upload/tmux/download steps](<C:/xampp/htdocs/YEAR 4/Testing/CCTV_DGP_FINITE_CLEARANCE_PROBE_V36_VM.md>)

The complete preceding handoff follows. Its earlier covering observations are
preserved; this diagnostic changes no historical protocol or gate outcome.


**Research milestone - 8 October 2026: V35 R1 audited; all scales fail preservation; changed V36 probe prepared.**

The human returned V35 R1. Its388,519,155-byte archive and2,135 members are
independently verified. The L4 probe completed93.899seconds, export26.162seconds:
four disposable original-state copies, zero optimizer/gradient/backward/epoch
or committed training updates, and no new checkpoint. Independent R2 audit
rederives500 raw/PNG measurements and replays100 frozen CPU outputs in191.532
seconds; all original tolerances pass and states are restored. The prefix-only
checker correction retains the original failed checker/log; no VM rerun needed.

Every scale1,1/2,1/4,1/8 fails preservation in both matched TRAIN subsets.
Scale1 delivered-detail gain is0.277588% exposed and0.418626% other TRAIN,
with4 and8 failed group/metric checks. Across all scales there are40 PNG
ArcFace failures and one clear MSE failure. Every scale also has raw MSE/
ArcFace failures, so PNG conversion alone does not explain the outcome.
Positive source gains and brightness passes do not override preservation.
The subset probe does not satisfy the unchanged1%-at50/10%-at800 capacity gates.

All100 cases/500 model outputs were actually viewed at original256px cell
detail across20 pages. Severe blur and compound inputs retain poorly defined
eyes, nostrils and mouths; trials remain close to the original DGP. Independent
readback verifies all700 input/target/output cells exactly. This is implementing
assistant TRAIN review, not independent final review. A renderer helper-name
collision is retained with its10 partial pages; distinct R1 preserves those
pages exactly and completes review preparation without changing returned data.

V36 changes the direction rather than rerunning the zero-clearance recipe.
All108 rows remain;65 raw MSE/ArcFace rows reserve twice the measured positive
V35 scale1 raw/PNG departure from their linear prediction. SSIM and six
restoration rows retain zero clearance because the saved skimage SSIM differs
from the differentiable VM guard. This is empirical, not a proven nonlinear
bound. Actual17 PNG groups, source, brightness and capacity gates are unchanged.
Both matched subsets now guide TRAIN design; unexposed is not DEV/final.

The array-only primal/dual check and independent full-row KKT/recalibration
pass. Candidate magnitude is100.703574% of the original mean proposal, below
the2x cap; this is not quality gain. Float32 copies retain small reported
clearance residuals. V36 remains an unrun four-scale finite image probe:
100 baselines plus400 trial outputs, no new gradients/optimizer updates,
trajectory, epochs, checkpoints, historical run or app promotion. Its9-file
12,190,362-byte packet passes Python3.10/read-only Bash syntax, all65 empirical
recalibrations,108-row certificate, host/root/platform and unsafe-return tests,
exact export/checker prefix parity and unchanged snapshot AST. Protocol SHA256:
9af4cbf10d7c141e2cbef2248bc282c135a9cc6117f37b47feaef76d71dadd38.
Execution archive SHA256:
ee51b5bc9e888dec32e0dfd90c96a19f9e93e21acd1fd4bef3e8500d12f0a67e.

Use the five manual Google Cloud SDK/tmux steps on the existing idle running
NVIDIA L4/g2-standard-4. Require6GiB free; estimate2-4minutes plus1-2minutes
export. Worker900s/external930s+30s grace, export300s/external330s+30s grace,
VRAM20GiB and uncompressed return768MiB enforce finite work. No VM connection,
start, cleanup or actual new training was performed in this milestone.

All14 current own-DGP Auto/override app bindings, original checkpoints, splits,
research caches, historical gates, instructions and the local backup remain.
No new native development, paired DEV or reserved-final pixels were opened.
These photographic paired TRAIN findings establish neither ethnicity nor
Zamboanga performance. Prior automatic and assisted covering-family failures
remain. Useful native structure, all seven covering families, independent final
review and the full DGP-led scope remain outstanding; request clearer or less-
covered crops when usable structure is insufficient. Goal active/incomplete.

[V35 audited findings and actual review](<C:/xampp/htdocs/YEAR 4/Testing/CCTV_DGP_GROUP_GUARD_PROBE_V35_R1_RESULTS.md>)
[V36 five exact manual steps](<C:/xampp/htdocs/YEAR 4/Testing/CCTV_DGP_FINITE_CLEARANCE_PROBE_V36_VM.md>)

The entire previous handoff follows. Its V35-unrun wording is historical and
superseded by this verified return; earlier quality failures remain binding.


**Research milestone - 8 October 2026: V34 diagnostic audited; V35 R1 finite image probe prepared.**

V34 completed300 per-case original-state non-hinged preservation gradient
queries in52.19seconds on the existing L4. Optimizer/parameter updates, epochs,
backwards and new checkpoints are zero. The independent audit verifies230
archive files and all100 raw outputs: exact parity with audited V33 original
outputs. It separately replays280 frozen CPU outputs within original raw/PNG/
embedding/component tolerances. Audit375.15seconds; no local gradients or
optimizer updates. The original DGP and fixed recognizer states remain intact.

All51 raw source/profile preservation directions are nonzero in each matched
TRAIN subset. The old restoration displacement predicts worsening in12
ArcFace checks per subset, including the aggregate; MSE and SSIM directions
improve. V34 exposes a protected-function conflict absent from the zero-at-
original hinges. These raw batchmatched derivatives are not delivered PNG
preservation, restored identity or native CCTV usefulness.

The independently verified new mathematical direction projects the mean of
the two audited original-state V33 displacements against all102 group guard
rows plus six nonzero existing restoration-loss rows. It retains99.1017% of
parameter-displacement magnitude; this is not a quality gain. Independent
group assembly, full-row KKT and primal checks pass. Float32 copies produce
small positive linear changes, maximum0.000000014073; finite outputs must
still pass the unchanged preservation gates. The mean of displacements is
not an AdamW step on a mean gradient. Both subsets now guide TRAIN design;
unexposed refers only to the first50 V32 updates, not held-out DEV/final.

The first analysis's NumPy-int64 JSON failure and all numeric artifacts are
retained. Serialization-only R1 is verified numerically hash-identical.
The first unrun V35 draft's two stale prospective replay paths are also
retained; R1 restores the validated V27/V28 basis. No recipe or gate changed.

V35 R1 is a new, unrun finite image probe: four disposable directions at
scales1,1/2,1/4,1/8, each reset to original DGP,100 baseline plus400 trial
outputs. Zero optimizer updates, new gradients, committed trajectory updates
or epochs. It creates no checkpoint, resumes no failed50 state and runs no
historical pilot. All17 PNG group, source, brightness and full-corpus1%-at50/
10%-at800 gates remain. A subset probe cannot qualify training capacity.

The9-file12,146,657-byte R1 packet passes independent230 returned/19 local
bindings,108-row geometry, Python3.10/read-only Bash syntax, wrong host/root/
platform, Windows pre-neural and unsafe-return checks. Protocol SHA256:
ba359d8aa6b3cd6c32b3e8f1c5d459b0af40751414d9f55a9261f3055d44b68b.
Archive SHA256:
954d81811e6881612787608086f27cceb96fa86888a49ee6f52537ca4d3b7ca3.
Use only the R1 manual Google Cloud SDK/tmux steps. Require idle running
L4/g2-standard-4 and6GiB free. Estimate2-6minutes plus1-3minutes export;
worker900s/external930s plus30s grace, export300s/external330s plus30s,
allocated VRAM20GiB and uncompressed return768MiB. No VM connection,
start, cleanup or new actual training occurred in this milestone.

The own-DGP-led Auto/override app and all14 current bindings remain unchanged.
No new native, paired DEV or reserved-final pixels were opened. V34/V35 are
paired photographic TRAIN evidence; source labels imply neither ethnicity
nor Zamboanga performance. Native unpaired evidence and synthetic DEV/final
remain separate. Existing functional inline Playwright evidence remains;
negative automatic and assisted covering-family reviews remain binding.
Useful native structure, all seven covering families, independent final
review and the complete DGP-led app scope remain outstanding. Request
clearer/less-covered input when usable structure is insufficient. Goal active.

[V34 findings and retained failures](<C:/xampp/htdocs/YEAR 4/Testing/CCTV_DGP_GROUP_GUARD_GRAD_V34_RESULTS.md>)
[V35 R1 five exact manual steps](<C:/xampp/htdocs/YEAR 4/Testing/CCTV_DGP_GROUP_GUARD_PROBE_V35_R1_VM.md>)

The edited V34 runbook's duplicated checksum-download line is retained. Its
original guide is recovered exactly from its hash-matched generator and bound
separately for historical closure; no live user file was overwritten.

The entire previous handoff follows. Its V34-unrun statements are historical
and superseded by this verified return. Its prior failed restoration and
covering-family quality decisions remain current.


**Research milestone - 7 October 2026: V33 output probe audited; group-preservation diagnostic V34 prepared.**

V33 completed its finite human-run probe in192.75seconds: eight reset first-step
AdamW proposals, zero committed trajectory updates, new gradient queries or new
checkpoints. The independent R2 audit verifies5,914files, all1,400 raw/PNG
measurements, moment/displacement/projection proofs and280 frozen CPU replay
outputs. Maximum raw error0.000002414, PNG difference one byte, vector error
0.000001349 and component error0.000003875 pass the original tolerances. Local
gradients, backwards and optimizer updates are zero. The audit took569.79seconds.

The initial checker stopped on exact float group-dictionary equality; independent
readback measured rounding differences around0.00000000000000017. Its source and
failure are retained. R2 preserves exact saved-row aggregation and checks fresh
group arithmetic with the existing case tolerance, retaining identical failure
keys, source-gain signs and brightness decisions. No qualification gate changed.

No proposal qualifies a training recipe or app replacement. At the original state,
the four preservation terms have zero gradients; cone scale1 therefore equals
the unconstrained proposal and fails15 group/metric checks in each TRAIN subset.
All smaller original-state scales also fail preservation. At stopped50, cone
scale1 improves detail0.37288% in exposed and0.44015% in unexposed matched TRAIN
subsets. The exposed subset passes17 group checks, but the unexposed subset
still fails two ArcFace checks. It repairs one existing blur failure while
introducing a clear-group failure; the compound degradation failure remains.
Neither subset gain replaces V32's full3,905-case0.970672% failed1% gate at50.
Do not resume that checkpoint or repeat the failed projection recipe unchanged.

All four prospectively metadata-selected comparison pages were actually inspected
at256px cell resolution: first reference per source, all five profiles, both
matched TRAIN subsets,20 cases/160 input-target-comparison cells. The difficult
rows remain soft around eyes, nose and mouth, with little convincing all-feature
clarity gain. This is a bounded development review, not review of all1,400 images
or independent final quality. All1,400 measurements were audited separately.
Unexposed means untouched during the first50 V32 updates, not held-out DEV/final.

GEM's locally linear gradient approximation and recent epsilon-constraint work
support investigating the protected functions directly; they do not prove CCTV
usefulness or authorize relaxing preservation bounds. The research rationale
and primary-paper links are in CCTV_DGP_LOSS_CONE_PROBE_V33_RESULTS.md.

V34 measures original-state, non-hinged raw MSE, one-minus-SSIM and one-minus-
fixed-ArcFace derivatives for each of the same100 TRAIN cases. The same five-case
batch context and23 selected own-DGP tensors remain. There are exactly300
gradient queries, zero optimizer/parameter updates and no new checkpoint. Saved
case matrices allow an independent assembly of all17 source/profile group
derivatives, rather than relying on a single average that can conceal opposing
directions. This is a changed diagnostic measurement, not a changed training
loss or an automatic continuation. Finite outputs must still pass the original
PNG, source, brightness and capacity requirements before any later training.

The three-file32,433-byte packet passes independent provenance, Python3.10,
read-only Bash syntax, Windows pre-neural rejection, three wrong-root/instance,
eight unsafe-return and six group-cancellation/reordering/count regressions.
The first verifier's sandbox signal-pipe failure is retained separately; its
read-only Bash syntax check passes outside that restriction. The VM packet and
gates are unchanged. Protocol SHA256:
d03050e18e4fd2369a6dd0803632bebfa0c709cc95cc8f7a6c8776785b0923a1.
Execution archive SHA256:
db822ba0b17f941d8e20783e5ee73ed97586dfdfb0f53a4d4909d392bd4b2656.

V34 remains unrun. Follow the five manual Google Cloud SDK upload/install/tmux/
launch/download steps on the existing running L4/g2-standard-4 VM. Require6GiB
free and an idle GPU. Estimated diagnostic1-6minutes plus export1-3minutes;
worker600s, external630s plus30s grace; export300s/external330s plus30s grace;
VRAM20GiB and uncompressed return1.5GiB stops are enforced. Original assets,
failed gates, splits and research-cache backup remain. No VM start, new training
or cleanup was performed in this milestone. Prior disk/VM snapshots are historical.

The local own-DGP remains the primary restorer with its existing Auto/override
workflow and design. All14 current app bindings remain hash matched. The prior
full app/inline Playwright ordering evidence and negative full-family completion
reviews remain binding. No new native or reserved-final pixels were opened;
public native CCTV development and separate labeled final identities stay frozen.
This new probe is paired synthetic photographic TRAIN evidence, separate from
unpaired CCTV evidence. Source labels imply neither ethnicity nor Zamboanga
performance. Useful native restoration, automatic and assisted covering-family
quality and independent final review remain outstanding. Goal active/incomplete.

[V33 findings and actual review](<C:/xampp/htdocs/YEAR 4/Testing/CCTV_DGP_LOSS_CONE_PROBE_V33_RESULTS.md>)
[V34 five manual steps](<C:/xampp/htdocs/YEAR 4/Testing/CCTV_DGP_GROUP_GUARD_GRAD_V34_VM.md>)
[Independent V33 return audit](<C:/xampp/htdocs/YEAR 4/Testing/outputs/cctv_dgp_loss_cone_probe_v33_independent_audit_r2.json>)

The complete previous handoff follows. Its V33-audit-pending statements are historical and superseded by this verified return. Its covering-family quality failures remain current.


**Completion milestone - 7 October 2026: input-only footprint ablation audited; full-family quality remains incomplete.**

Tighter input-only removal masks preserve more visible context, but they do not
give a consistently useful completion. The complete comparison is retained as
development evidence and is not adopted by the application. Dark eyewear and
hair-like material appear again in estimated regions; glare and several object
joins remain conspicuous. Geometry compliance and exact visible bytes do not
qualify the estimated face.

All36 previously exposed photographic cases were reviewed before annotation:
18 source photographs and their matched synthetic degradations, all seven
covering families and uncovered/clear-glasses controls. The prior strongly
turned pair and nearly hidden pair remain excluded. The other32 cases receive
one fixed comparison against their cached current CodeFormer Off outputs.
The legacy native suffix means original photograph, not native CCTV. There is
no aligned hidden-face reference; pretraining overlap is unknown. No ethnicity,
Zamboanga performance, hidden PSNR/SSIM or exact hidden-identity claim follows.

Fourteen object footprints are traced from inputs inside an explicitly
approximate facial boundary. Their Euclidean two-pixel margin is clipped by
face eligibility and protected visible support. The two clear controls have
empty masks. Clear frames, exposed nose/mouth/gaze, ordinary hair, exterior
hands and clothing are retained. Source annotations are reused on corresponding
degraded photos, so this is assisted evidence, not isolated degraded-only
annotation. All six new overlay pages were actually reviewed before generation.
Five glare-core pixels overlapping a protected clear frame stopped preparation
before image creation; the initial source/failure are preserved. The separate
R1 trace stays below that rim. Soft reflection, thread/strand/petal boundaries
and hidden jaw extent remain uncertain, rather than segmentation truth.

The existing project Python3.13.5/PyTorch2.13.0+cpu environment completed the
fixed comparison in146.67seconds, with28 actual completion forwards, four exact
empty bypasses and four pre-neural exclusions. Internal512 raw tensors and256
float stages are separate from PNG and grayscale display processing. There are
zero detector/DGP forwards, gradients, backwards, optimizer updates or new
checkpoints. An initial system-Python launch lacking Torch failed before model
loading and is retained separately; the R1 uses the same masks, checkpoint,
settings and600-second/630-second external bounds. No dependency installation
or failed immutable execution is repeated.

The independent1.89-second saved-output audit verifies all32 PNGs, all28
internal512-to256 compositions,5,483,088 unchanged visible source bytes and
1,075,314 unchanged protected bytes. Shared visible support also matches the
older outputs exactly. All246 source bindings and104 artifacts are verified;
the95,848,405-byte result is within256MiB. The completion state is identical to
the original route before and after inference. A separate Euclidean-distance
mask audit verifies all32 supports, zero protected/outside-face overlaps and
all source-assisted pair identities. These are arithmetic and preservation
checks, not an independent human final-quality judgment.

All32 estimates/eight comparison pages were actually inspected at256px by the
implementing assistant. Central mouth estimates can remain plausible, and the
scarf/glove example is locally cleaner, but revised dark/tinted eyewear, strong
glare, obstructing curls, upper-hand joins, petal remnants and green leaf-like
staining prevent all-family usefulness. Source covering outside the final
facial overlap is intentionally retained; an approximate boundary can itself
produce an awkward join. Object-like material inside estimated support is also
a generation limitation. These findings are distinct from exact source pixels.
No mask was adjusted using these outputs, no seed search was performed, and
neither automatic nor assisted quality is qualified.

Saved automatic proposals remain a separate diagnostic: they include empty
eyewear/object misses and false marks on an uncovered control. Their overlap
against approximate assisted footprints is not segmentation accuracy; no new
automatic generation is included. The mask ablation changes both conditioning
and final compositing support, so it cannot isolate a unique cause for each
visual change. The next diagnosis should separate those two roles while
retaining the failed historical uniform context6 comparison and keeping the
delivered removal area fixed. Enlarging it to hide visible features is not an
acceptable success criterion.

The own-trained DGP remains the application's primary restorer; all14 current
app/checkpoint/source bindings, design, splits, failure gates and research-cache
backup records remain. Useful native restoration, covering-family qualification
and independent final review are outstanding. Actual training remains manual
on the existing L4 VM. The newly downloaded V33 archive is being independently
audited; this completion comparison does not authorize another optimizer run.

[All-case visual observations](<C:/xampp/htdocs/YEAR 4/Testing/outputs/completion_input_footprints_comparison_v1_r1/visual_review.json>)
[Independent saved-output audit](<C:/xampp/htdocs/YEAR 4/Testing/outputs/completion_input_footprints_comparison_v1_r1/independent_saved_output_audit.json>)
[Original-photo comparison](<C:/xampp/htdocs/YEAR 4/Testing/outputs/completion_input_footprints_comparison_v1_r1/pages/original_photo_3.png>)
[Degraded-photo comparison](<C:/xampp/htdocs/YEAR 4/Testing/outputs/completion_input_footprints_comparison_v1_r1/pages/synthetic_degraded_photo_4.png>)

The complete previous handoff follows. Its V33-unlaunched statements are historical: the human has now supplied a hash-matched return and its independent audit is in progress.


**Latest research milestone - 7 October 2026: V32 loss-gradient return audited; finite V33 output probe ready.**

The human downloaded the completed diagnostic. Archive SHA256
529acd5750c64e7320260c09f55cdd3a115c2f465afd02abc4101009155143e3
matches721,569,318bytes and both sidecars. Independent audit verifies756files,
301,298,844saved gradient values and200frozen CPU restoration replays. Maximum
raw error0.000002563, PNG erroronebyte and vector error0.0000005253 pass unchanged
tolerances. VM diagnostic45.58s/280queries/0optimizer updates; local audit282.12s
with zero gradients, backwards or optimizer calls. No new checkpoint was created.
The sidecar read race and both successful download transports are preserved;
completed canonical sidecars match their staged copies without repeat downloads.

At stopped50, preservation/restoration gradient-norm ratios are7.5746exposed and
4.2689unoptimized, with negative cosines−0.24866/−0.25694. The negative raw
seven-loss sum increases facial-detail loss in both aggregate TRAIN cohorts.
This measures endpoint conflicts, not actual unsaved AdamW trajectories or
curvature. Existing protection counters actual regressions and remains necessary.
V32's50-update/0.970672% versus1% failed structure gate remains; no resume exists.

Own cone arithmetic keeps all seven nonzero gradient constraints. A restoration
proposal descends in all three restoration terms in the stopped aggregates;
projecting the full seven-loss sum can leave a detail derivative at zero. Both
methods on44matrices pass88independent primal checks, analytic/invalid/random
fixtures, with no neural work or parameter assignments. This does not prove
finite face improvement, capacity or useful native output. Primary research and
its mathematical adaptation/curvature/performance limits are recorded separately.

The distinct V33 probe is prepared for manual execution on the existing L4 VM.
It reuses four saved aggregate matrices and makes0new gradient queries. Each
original/stopped state and exposed/unoptimized cohort makes two fresh disposable
AdamW proposal steps, eight total, resetting before every output trial. The
original loss values, normalization, learning rate, clipping, decay and all
quality gates stay. No continuing trajectory, epoch or new checkpoint is created.
The actual restoration AdamW displacement is projected against all seven nonzero
loss gradients, then tested at fixed scales1,1/2,1/4,1/8 beside both controls.
Zero-gradient hinges and float32 rounding can still cause finite regressions.
Actual outputs, not linear projections, decide them. The unconstrained control
cannot qualify a training recipe. No automatic800-update follow-on is permitted.

The42,130byte/five-file packet passes414basis bindings, Python3.10 parsing,
pre-neural Windows update rejection, seven unsafe-import checks, eight independent
primal displacement fixtures and actual read-only Bash syntax. The raw metric
reviewer reproduces the first six components on200saved cases with maximum
error0.00000016393. First syntax/solver/serialization preparation failures remain.
The distinct R1 local reviewer fixes solver coordinates and tightens convergence;
VM packet/protocol, projection and all tolerances remain unchanged. Its source is
frozen separately and supersedes the retained initial return reviewer.

V33 protocol SHA256:ba8d1f87cae38cac8e1b3b893378a185c6126c25151dadbcfa0ebb373a51adee.
Archive SHA256:282931a3b265c834a5736ac7f70c0816e08b74a15948d4c94645679a309eeace.
Require an idle L4/g2-standard-4 with6GiB free. Estimated3–10minute probe plus
1–4minute export; actual V33 timing remains unmeasured. Worker900s,
external930s+30s grace, export300s/external330s+30s grace, VRAM20GiB and return2GiB
are enforced. Save all1,200trial plus200before outputs. Prospective independent
audit checks their composition/metrics, eight actual moment/displacement proofs,
four independent projections and280metadata-selected frozen CPU replays.
The guide gives five exact gcloud upload/install/tmux/launch/download steps.
Agent VM launches, uploads, new training and app changes are zero in this stage.

This is paired synthetic photographic TRAIN evidence. Source labels are not
ethnicity, native CCTV or Zamboanga performance. Native development/reserved
pixels remain unopened. Original checkpoints, splits, caches, failures and actual
Windows backup remain. The own-DGP app checkpoint, selector, override and design
are unchanged. Converted MAT is still unqualified. Useful native restoration,
automatic/assisted quality for all seven covering families and independent final
review remain outstanding. Goal active/incomplete; actual training stays manual.

[Audited gradient review and primary research](<C:/xampp/htdocs/YEAR 4/Testing/CCTV_DGP_V32_LOSS_GRADIENT_RETURN_REVIEW_V1.md>)
[Five exact manual V33 steps](<C:/xampp/htdocs/YEAR 4/Testing/CCTV_DGP_LOSS_CONE_PROBE_V33_VM.md>)

The complete previous handoff is retained below. Its previously unlaunched V32
diagnostic is now completed by the human; the new V33 probe remains unlaunched.


**Latest research milestone - 7 October 2026: stopped-V32 raw-loss review audited; distinct zero-update diagnostic ready.**

The user's "apply the best approach and do research also if needed" answer satisfies
the required design discussion. The selected direction diagnoses current learning
first. Primary research is recorded with explicit limits; it does not establish
this model's failure cause or justify a blind frequency-loss/learning-rate change.

All50 frozen preflight/normalization TRAIN previews have an independently verified
raw-loss review. None was optimized by50; they are not unseen evaluation. The40
degraded previews improve raw feature error0.968684% and PNG error0.991558%.
The full3,905-case PNG stop remains0.970672% against the unchanged1% requirement.
Rounding does not explain the weak preview gain. Eyes/nose/mouth and remaining
observed interior show uneven small improvements, while useful added whole-face
clarity remains unestablished by the earlier full50 visual review.

Per-case protection is active:8 degraded and1 clear preview trigger raw identity
penalties, and1 clear control has3.802615% higher raw pixel error. All10 clear
anchors activate. Mean total raw objective on this fixed50 cohort changes from
1.2999999802 to1.3002432080: added preservation value0.0124378591 slightly exceeds
the restoration-term reduction0.0121946314. Values alone do not prove gradient
dominance or reconstruct unsaved AdamW moments. Passing17 delivered group-average
gates does not erase these per-case findings. No preservation term or gate is removed.

The45.94s local analysis makes100 frozen recognizer image calls and zero DGP,
gradient, backward or optimizer calls. The separate40.53s audit verifies387 bindings,
50 cases,400 regional/filter quantities and100 raw recognizer images. The first
schema lookup error and its plan remain intact; separate R1 review corrects only
the reference field and evidence routing. No original model or loss code changes.

A separate stopped-V32 loss-gradient packet is ready for the human's manual L4
launch. It measures the original and stopped states on two50-case TRAIN cohorts,
five references per photographic source with all five profiles. Exposed references
are the first five touched per source in the frozen schedule. The other cohort
uses all50 fixed previews, source/profile matched and unoptimized by50. Selection
uses existing metadata and the existing fixed set, not output rankings. Both
fusion and decoder actually changed at this endpoint, unlike the earlier V31
diagnostic. The next design depends on returned component conflict, directional
derivatives, partition magnitudes and batch coherence; no recipe is preselected.

The54,322-byte,3-file packet is independently verified against140 TRAIN files and
518 V32 dependencies. Six adversarial boundaries, Python3.10 syntax, pre-neural
Windows differentiation rejection and actual read-only Bash syntax pass. The
sandbox-denied Bash signal-pipe attempt, successful outside-sandbox parse and R1
readback are all retained; the packet and protocol are unchanged by that correction.
Protocol SHA256:61c04aef4ee184adc9830482477bf37ad9c936a8c0247e44d26eefc1a6bf483d.
Archive SHA256:22d76f3637a3ac580bc18fbac79ff5e3d24a61ec0a00ce082fde78c7069c6052.

The diagnostic permits280 gradient queries,0 optimizer updates,0 epochs and no
new checkpoint. Require an idle existing L4/g2-standard-4 with6GiB free. Worker600s,
external900s plus30s grace, export300s/external330s plus30s grace, VRAM20GiB and
uncompressed return1.5GiB are enforced. Estimate1-5 minutes plus1-3 minutes export.
No automatic historical/follow-on launch, cleanup, upload or VM run occurs here.
The guide has exact gcloud upload, tmux launch and three separate PuTTY downloads.
Returned data require independent audit before a justified new finite training pilot.

Source names are not ethnicity. This is paired synthetic photographic TRAIN
evidence, not native CCTV or Zamboanga performance. Native development/reserved
pixels remain unopened in this milestone. The own-DGP app checkpoint, selector,
256 processing and design remain unchanged. Original checkpoints, splits,
research caches, every failure and the actual Windows backup receipt remain.
The converted-MAT comparison remains unqualified; all seven completion families
still need separate automatic/assisted quality, useful native restoration and
independent final review. Goal active/incomplete; actual training remains manual.

[Audited learning and primary research](<C:/xampp/htdocs/YEAR 4/Testing/CCTV_DGP_POST_V32_LEARNING_REVIEW_V1.md>)
[Five manual VM steps](<C:/xampp/htdocs/YEAR 4/Testing/CCTV_DGP_V32_LOSS_GRADIENT_V1_VM.md>)

The complete previous handoff is retained below. Its cause-under-investigation
wording remains valid; this milestone adds measurements and an unlaunched diagnostic.


**Latest restoration milestone - 7 October 2026: V32 r2 independently audited; early structure requirement failed.**

The human ran R2 on the existing L4. The directory repair worked; training reached
50/800 updates and then stopped deliberately. Degraded delivered-PNG structure
error fell from0.001876217941608562 to0.001858006016206468:0.9706721697% against
the unchanged1% early requirement. Export complete:true means packaging only.
The stopped checkpoint, original failure/traceback and all prior gates remain
retained. No800-update result exists. Do not resume, repeat unchanged or promote.

The user authorized direct download after only the R1 files were found locally.
The1,324,635,165-byte R2 archive and two sidecars match the reported SHA256
576b3897a9a9589ddad8896e71c827481086b26ffab6e6cbeadb9a5dbc4032e9.
Separate pinned-host gcloud downloads finish in302.07s with zero VM writes or
agent training launches. The independent return audit verifies27,625 files,
5,467 TRAIN assets,75,324,711 saved gradient values and both complete3,905-case
snapshots:7,810 delivered PNG metric/vector/mean-control rows. Source/protocol,
all23 connectivity and frozen parameters are checked. CPU inference replay covers
50 fixed previews at each snapshot plus initial50, not all3,905 neural outputs.
Maximum raw discrepancy0.0000025034 and one-byte PNG differences stay within
the unchanged audit tolerances. No local gradients/backwards/optimizer updates.

All17 delivered preservation groups pass at50. Photographic-source structure
reductions are0.952033% for dataset/asian_faces and0.975071% for thumbnails128x128;
both are positive but below1%. Source labels are not ethnicity. These are paired
synthetic photographic TRAIN observations, not native CCTV or Zamboanga evidence.
The50 paired batches optimize50 references/250 cases and no full epoch. Worker
time1033.71s and allocated VRAM8,942,142,976 bytes remain within their finite caps.

All50 fixed previews and10 original-detail sheets were actually reviewed across
eyes, nose, mouth, face outline and overall visible appearance. All250 image cells
are verified exact256px copies, with no display processing. Broad expression and
appearance generally remain, including clear glasses, hair, facial hair, cap and
adjacent hands. Fine eyes/nostrils/lips stay soft. R2 shows little added clarity
over V31; a useful incremental whole-face improvement is not established. Severe
compound crops need clearer input. The fixed previews were not optimized by50,
but are preflight/normalization TRAIN, not held-out or independent final evidence.

Saved parameter analysis confirms actual fusion/decoder movements of about
0.339% relative L2, with frozen state unchanged. Initial improvement-gradient dot
products with the actual displacement are negative, but do not reconstruct AdamW
history or isolate a cause. V30-V32 have missed the same structure requirement.
The invalid assumption is that favourable initial gradients and more original
fusion weights were enough under the unchanged finite learning design. The
recommended discussion now examines raw/delivered losses, normalization,
preservation activation and effective updates before another model change.
The user answered the required diagnostic question: "apply the best approach and
do research also if needed". The selected direction diagnoses current learning
first using saved evidence and primary research. The discussion is satisfied;
the cause remains under investigation. No new trainer is prepared here.

The separate converted-MAT comparison is closed:32 assisted cases,28 CPU forwards,
four exact clear-control bypasses,11 adapter/format checks and5,225,232 exact
visible source bytes. All eight pages were reviewed. Hand/glare/hair/scarf defects
prevent consistent usefulness; the converted candidate is retained, not promoted.
Automatic proposals and independent final quality remain unqualified.

The original own-DGP app checkpoint,256 processing, selector, design and current
frontend stay unchanged. R1's separate zero-update failure is fully audited.
Original checkpoints, frozen splits, research caches, all failures, prior
documents and the actual Windows backup receipt remain preserved. No new native
or reserved pixels are viewed. Actual training remains the human's manual
transfer/SSH/tmux workflow on the existing L4; this agent starts no training.
Useful native structure, all seven completion families with separate automatic/
assisted quality and independent final review remain required. Goal active/incomplete.

[Verified R2 results and full50 review](<C:/xampp/htdocs/YEAR 4/Testing/CCTV_DGP_FEATURE_FUSION_V32_R2_RESULTS.md>)
[Post-V32 design discussion](<C:/xampp/htdocs/YEAR 4/Testing/CCTV_DGP_POST_V32_ARCHITECTURE_REVIEW.md>)
[Converted MAT findings](<C:/xampp/htdocs/YEAR 4/Testing/CCTV_DGP_COMPLETION_MAT_MIRROR_V1_RESULTS.md>)

The complete prior handoff is preserved below. Its R2-audit-pending wording is
historical and superseded by this completed independent return audit.


**Completion comparison milestone - 7 October 2026: converted MAT tested and retained as an unsuccessful comparison.**

The pinned third-party FP16 EMA MAT FFHQ512 conversion completed32 eligible
assisted development requests:28 actual CPU forwards and four exact empty-mask
bypasses, with no gradients or training. Its source and published125,280,246-byte
tensor archive are independently checked;465 tensors/all62,612,683 finite values
are verified. This is the converted release and pinned CPU implementation, with
original-author checkpoint/CUDA parity unverified. Original MAT CC-BY-NC4.0 and
ChaiNNer MIT notices remain retained. No general MAT failure or deployment claim.

The frozen comparison keeps the same reviewed masks and current CodeFormer Off
outputs. All seven covering families and uncovered/clear-glasses controls are
included. The32 cases are exposed photographs and synthetic degradations, not
native CCTV. Four prior input-only exclusions stay excluded. Pretraining overlap
is unknown. No aligned clean hidden-face reference, hidden PSNR/SSIM, exact hidden
identity, ethnicity or Zamboanga-performance claim is introduced.

All32 PNG/stage pairs and28 raw512-to256 compositions pass independent readback.
All5,225,232 visible source bytes outside the reviewed masks remain exact;
all four clear controls bypass exactly. One fixed CPU replay matches raw values
with maximum error0.0 and exact PNG. Eleven adapter/tensor-format regressions pass.
The bounded inference completed in201.10s; external observation204.89s is within
1230s. No timeout occurred. Raw512 outputs remain separate from delivered256 PNGs.

All32 cases/eight original-detail comparison pages were actually reviewed by the
implementing assistant as development evidence. Eye/mouth estimates can be
plausible, but hand joins, bright lens glare, hair still obscuring an eye and
scarf/nose artifacts prevent a consistent useful improvement. Retained fingers
outside a reviewed footprint are separate input-mask misses. Zero extra removal
margin is used; no global dilation or favorable-seed selection. This assisted-only
comparison qualifies neither automatic proposals nor independent final quality.
The candidate is retained as a negative comparison and is not promoted.

The original own-trained DGP remains the main local restorer. App source/design,
original checkpoints, frozen splits, all failed gates and the actual Windows
research-cache backup remain preserved. Functional app verification already
passes; useful native DGP structure and all-family quality remain incomplete.

The separate R1 archive is now fully downloaded and independently audited:
180 files,75,324,711 saved gradient values and50 original CPU replay cases, zero
optimizer updates and no new checkpoint. Its pre-optimizer routing failure stays
retained. The user separately launched V32 r2 and reported a50-update early
structure stop. Its1,324,635,165-byte archive and two sidecars are now downloaded
with the reported SHA256576b3897a9a9589ddad8896e71c827481086b26ffab6e6cbeadb9a5dbc4032e9.
Full independent R2 snapshot/metric audit is in progress; no R2 quality claim or
promotion follows from its export. Actual new training remains the human's manual
existing-L4 workflow. No training was launched by the agent. Goal active/incomplete.

[Converted MAT full-family findings](<C:/xampp/htdocs/YEAR 4/Testing/CCTV_DGP_COMPLETION_MAT_MIRROR_V1_RESULTS.md>)
[Independent saved-output audit](<C:/xampp/htdocs/YEAR 4/Testing/outputs/completion_mat_mirror_comparison_v1/independent_saved_output_audit.json>)

The complete prior handoff is preserved below. Its R1 full-download-pending
wording is historical and superseded by the completed independent return audit.


**Latest repair milestone - 7 October 2026: R1 stopped before any optimizer; use the verified V32 r2 routing repair.**

The human-launched V32 r1 completed its initial gradient preflight, then failed
at the cache loader's exact directory guard. Its worker required the R1 directory
but the copied cache still required the original V32 directory. This was an agent
packaging error. It is not a learned-structure gate failure and supplies no improved
output evidence. R1 is retained unchanged; do not rerun that broken packet.

A bounded read-only maintenance SSH fetch verifies the original protocol/source
hashes, failure, gradient-preflight receipt, trainer log/exit, supervision and
export manifest/sidecars. The VM archive hash is
599d84dd2b84fdc05e55b774d115951d251e4013ae777d7f023791bdb60f59e2,
169,708,697 bytes. There were zero optimizer updates, no constructed optimizer,
no new checkpoint and no results.json. Export complete:true means packaging only.
The full archive download and independent audit of its saved gradient values
remain pending; the small original receipts are verified separately. The first
sandbox-denied SDK access is preserved; authorized host access succeeded with
zero VM writes, model/gradient calls or training launch by the agent.

The pre-import R1 path mismatch is reproduced with actual guard AST, using only
simulated platform/home metadata. V32 r2 routes both guards to its distinct root
and calls the same exact cache guard during --verify-transfer and before run.
The cache's Linux/exact-root restrictions are preserved. Eight actual guard
regressions and six actual transfer-dispatch regressions pass, including valid
Linux routing, old/base/nested/outside roots, Windows and incorrect parent.
Transfer dependency checks in those regressions are explicit metadata stubs;
they do not establish real CUDA/data preflight or trained usefulness.

The first local transfer test fixture omitted assets_sha256 and failed at the
status print after valid checks. That verifier/error are retained. A separate
audit corrects only the mock metadata field; the R2 packet remains unchanged.
All11 archive members, source/protocol equivalence, Python3.10 syntax, Windows
pre-neural rejection, prospective return schema and the failed1% receipt pass.
The unchanged candidate/training/shell inherit11 prior core regressions, four
exact raw/PNG CPU parity cases and the original read-only Bash parser result.
No new neural, local autograd or optimizer call is performed for this repair.

V32 r2 protocol SHA256: 87582314eb1313a6914b4940496eeeb246c3023b7b6bf4f84da43e3f5e739d6c
V32 r2 execution archive SHA256: 82824cbf3239bc017a74f13982001b31409c5e77af1dfb1b3bd7b9e9f77d283a
Execution archive:758,053 bytes,11 regular files.

This is the same justified fresh-original fusion/decoder experiment:23 tensors,
978,243 parameters,781 TRAIN references/3,905 cases, frozen schedule, seven losses,
normalizers and AdamW unchanged. All scientific gates remain fixed: maximum800
updates; early50 gain>=1%; final gain>=10%, all17 preservation groups, both source
gains>=0 and mean-only fraction<=20%. Original checkpoint, frozen backbone/head4
and all normalization buffers remain protected. No failed learned recipe or
stopped candidate is resumed. V31's50-update/0.694525% failed gate and every older
checkpoint, split, failure and provenance record remain binding.

Actual new training remains the human's manual upload/SSH/tmux workflow on the
existing L4/g2-standard-4 at ~/forensic-dgp. Require8GiB free and an idle GPU.
Estimate15-35 minutes plus export, with unchanged enforced preflight300s,
cache900s, fit3600s, worker4500s/external4800s+30s, export900s/external930s+30s,
VRAM20GiB and return3GiB limits. This agent has installed/launched no R2 workload.

The original own-DGP app checkpoint/design and actual Windows research-cache
backup remain unchanged. Useful native restoration, all seven completion-family
automatic/assisted quality reviews and independent final review remain pending.
No new development/native/reserved pixels are used. No ethnicity, hidden identity
or Zamboanga performance inference is made. Goal active/incomplete.

[V32 r2 five manual steps and separate R1 failure download commands](<C:/xampp/htdocs/YEAR 4/Testing/CCTV_DGP_FEATURE_FUSION_V32_R2_VM.md>)
[Verified original R1 failure receipts](<C:/xampp/htdocs/YEAR 4/Testing/outputs/cctv_dgp_feature_fusion_v32_r1_failure_v1/host_access_retry/verification.json>)

The complete prior handoff is preserved below. Its R1 transfer recommendation
and unlaunched wording are historical and superseded by this verified repair.


**Latest transfer milestone - 7 October 2026: use V32 r1, with consistent23-tensor protocol descriptions.**

The original V32 packet's runtime correctly selected23 tensors, but four inherited
protocol descriptions still referred to12 selected tensors. That unlaunched
packet, archive, successful checks and original documents are retained unchanged.
Use the separately verified V32 r1 packet below. It corrects initial-parity,
normalization, gradient-policy and optimizer descriptions, clarifies the manual
training boundary and routes a distinct directory/export. This is a metadata
revision of the same new finite fusion-partition experiment, never a rerun of
V31 or a stopped candidate. No loss, learning rate, weight decay, clipping rule,
selected tensor, schedule, forward, normalization buffer or scientific gate changes.

R1's11-file/756,048-byte archive passes exact source/protocol/archive comparison,
Python3.10 syntax, its Windows rejection before neural imports, actual prospective
return gradient-schema verification and failed1% gate receipt verification.
The unchanged candidate, policy, cache, training and shell bytes inherit the
previous11 passing regressions, four exact CPU parity comparisons and read-only
Bash parse. Those are inherited proof, not new neural or parser calls. R1 preparation
and audit perform zero neural, gradient or optimizer calls. No VM action occurs.

V32 r1 protocol SHA256: e25e5721a5474d767fe48710336105a1eaa843ed2e22d6cb03caebd8e3bd10da
V32 r1 execution archive SHA256: 23469b5b79bc093b5b0f064b8a99ec716a750247aaf6e945d36baa6e4b4df40d

The audited feature-fusion diagnostic remains complete:756 files,280 VM gradient
queries,301,298,844 saved values and200 independent CPU inference outputs; zero
training updates. All11 original fusion plus12 decoder tensors are connected in
both measured TRAIN cohorts/states. This supports a finite fresh-copy test, not
an improved-output claim. Keep V31's50-update/0.694525% failed1% requirement,
stopped checkpoint and every earlier failed gate. Source labels are not ethnicity;
native/unpaired evidence stays separate from paired synthetic measurements.

The same experiment enables eight original fusion convolutions/11 tensors beside
12 active decoder tensors,978,243 parameters total. Backbone, inactive head4 and
all buffers remain fixed. Maximum800 updates; stop at50 unless structure gain
reaches1%. Final TRAIN gates retain10%, all17 preservation groups, both source
gains>=0 and mean-only fraction<=20%. Require8GiB free, idle existing L4/g2-standard-4
and the manual tmux flow at ~/forensic-dgp. Preflight300s/cache900s/fit3600s/
worker4500s, external4800s +30s grace, export900s/external930s +30s grace,
VRAM20GiB and return3GiB remain enforced. Preserve every partial/time/gate failure.

The agent has not uploaded, installed or launched either V32 packet, has performed
no local training and has deleted no research files. The app's original own-DGP
primary checkpoint/design are unchanged. Useful native development outputs,
separate automatic/assisted quality for all seven covering families, independent
final review and the full DGP-led goal remain incomplete. Only passing capacity
and all50 TRAIN preview review can justify the fixed520 paired DEV/24 unpaired
native DEV review; reserved-final pixels stay unopened.

[V32 r1 five manual upload/tmux/download steps](<C:/xampp/htdocs/YEAR 4/Testing/CCTV_DGP_FEATURE_FUSION_V32_R1_VM.md>)
[Audited feature-fusion findings](<C:/xampp/htdocs/YEAR 4/Testing/CCTV_DGP_FEATURE_FUSION_GRADIENT_V1_RESULTS.md>)

The complete previous handoff follows. Its original V32 transfer recommendation
is superseded by this separately verified R1 archive and guide.


**Latest research milestone - 7 October 2026: original DGP fusion diagnostic audited; V32 finite pilot ready.**

The human returned the distinct feature-fusion diagnostic. The unchanged
prospective local checker verifies all756 files, 301,298,844 saved gradient
values and200 CPU inference outputs. VM measurement takes40.769s with280
component-gradient queries, zero optimizer updates/backwards/epochs and no new
checkpoint. CPU replay stays within the declared bounds: maximum raw error
2.294778823852539e-6, PNG difference1 byte and vector error5.252659320831299e-7.
No returned code is executed, no gradients are recomputed locally and no native
or reserved-final cases are opened. Both original and stopped states are retained.

All11 original fusion tensors and12 active decoder tensors have nonzero
improvement gradients in both matched TRAIN cohorts at both measured states.
The original fusion-only negative total-objective direction decreases all three
improvement terms in both cohorts. These are first-order observations, not a
prediction of AdamW steps, a unique causal explanation or useful output proof.
Stopped-state preservation terms respond to drift and remain binding. V31 stays
closed at50 updates/0.694525% against its unchanged1% structure requirement.

V32 tests one trainable-partition change on a fresh original-DGP copy: enable
the eight original fusion convolutions (11 tensors/479,616 parameters) alongside
12 active decoder tensors/498,627 parameters. Total978,243 parameters/23 tensors.
Backbone, inactive head4 and every evaluation buffer remain frozen. The full
781-reference/3,905-case TRAIN corpus, paired clear/degraded schedule, original
forward/mean-centering path, seven losses, initial normalizers, AdamW and all
scientific gates are unchanged. No stopped state is resumed or app checkpoint
selected. Source labels remain distinct from ethnicity and native CCTV evidence.

The11-file/754,835-byte packet passes an independent metadata/source/data/transfer
audit,11 meaningful regressions, Python3.10 syntax, read-only Bash syntax and a
Windows rejection before model imports. Four CPU comparisons verify actual23
parameter order, exact starting raw/PNG parity and unchanged buffers even after
train(True). Eight forward calls and zero local gradient/optimizer calls occur.

Protocol SHA256: e022fbc655b1b93caefc0572393171126c3db7c37d0e559732b38bfe8a656ee1
Execution archive SHA256: d4d99b7d6612b0b6a5ffa4f3692cf0e78c1edd06b503367621b510941b93416d

Maximum800 updates (781-batch epoch +19 batches). Stop at50 unless structure gain
reaches1%; final gates require10%, all17 preservation groups, both source gains
>=0 and mean-only fraction<=20%. Require8GiB free for a3GiB uncompressed return,
archive and margin. Enforce preflight300s/cache900s/fit3600s/worker4500s, external
4800s +30s grace, export900s/external930s +30s grace and allocated VRAM20GiB.
Every source, nonfinite, time or numerical stop is retained. Actual V32 training
is the user's manual existing-L4/g2-standard-4 tmux workflow. The agent has not
uploaded, installed or launched it and has deleted no VM or local research file.

An independent return audit and all50 TRAIN preview review are required before
any later fixed520 paired DEV/24 unpaired native DEV pass. Reserved-final pixels
remain unopened. The app still uses its retained original own-DGP checkpoint.
Useful native restoration, separate automatic/assisted covering quality across
all seven families, independent final review and the full goal remain incomplete.
Prior checkpoints, splits, provenance, cache backup and every failed gate remain.

[Audited diagnostic findings](<C:/xampp/htdocs/YEAR 4/Testing/CCTV_DGP_FEATURE_FUSION_GRADIENT_V1_RESULTS.md>)
[V32 five manual upload/tmux/download steps](<C:/xampp/htdocs/YEAR 4/Testing/CCTV_DGP_FEATURE_FUSION_V32_VM.md>)

The complete previous handoff body follows. Its diagnostic-unrun language
describes the prior packet-release milestone and is superseded above.


**Latest diagnostic milestone - 7 October 2026: original DGP feature-fusion measurement ready for manual VM execution.**

V31 remains closed at 50 updates: 0.694525% structure gain against the unchanged
1% requirement. Its independently audited return, 27,623 files, all50 visual
observations, original/stopped checkpoints and numerical failure remain retained.
All17 delivered preservation groups passing does not waive that stop. The app's
original own-DGP primary checkpoint is unchanged; native usefulness is unqualified.

A distinct zero-update diagnostic measures the original FPN feature-fusion path
alongside the active decoder. The separate copy exposes 11 original fusion
tensors (479,616 parameters) and 12 decoder tensors (498,627 parameters) to
gradient queries without changing their values. Backbone, inactive head4 and
evaluation normalization remain preserved. Two source/profile-matched TRAIN
cohorts each contain ten references with one clear and four degraded views.
The unexposed cohort excludes all50 references used by V31's stopped run.
Original and stopped50 states are observations only; neither is resumed.

The three-file 54,462-byte transfer packet passes archive/source/data checks,
Python3.10 syntax, seven metadata/return-boundary regressions, read-only Bash
syntax and rejection of Windows differentiation before neural imports. Four
CPU forward-only comparisons verify the actual23-tensor order and exact starting
raw/PNG parity. Eight model forwards and zero gradient/optimizer calls occurred.
Gradient connectivity through the new fusion partition is still unmeasured.

Protocol SHA256: 5926dea8c477a9d51ff97186e6fec2b0d8eab8e67a7cf7a0ed1a6c716756e0af
Execution archive SHA256: 9b1d3ab30224e24e02e1b00995b3d290bcbb4e037f121aa924d871b80fff4b5f

The worker permits 280 component-gradient queries, zero optimizer updates and
zero epochs, with600s worker/900s external bounds,30s grace,20GiB allocated VRAM,
6GiB required free space and1.5GiB maximum uncompressed return. Export has its
own300s/330s bounds. Any source/state/nonfinite/time failure is retained. No
checkpoint writer, loss/gate relaxation or new training recipe is present.
Actual measurements remain the user's manual existing-L4 tmux workflow. The
agent has not uploaded, installed or launched the new packet on the VM.

The first local Bash parser could not start because the Windows sandbox blocked
its signal pipe. Its failed record/source and passing regression log remain.
The unchanged script passed the same read-only parser outside that sandbox;
this changes no VM recipe or scientific gate.

The prospective local return checker permits only inference and saved-gradient
arithmetic, never returned-code execution or local differentiation. Results must
be audited before choosing any later finite training pilot. No native, development
or reserved-final cases were used by this packet or parity check. The seven
covering families, separate automatic/assisted quality, useful native DGP outputs
and independent final review remain required. The full goal is active/incomplete.

[Five manual upload/tmux/download steps](<C:/xampp/htdocs/YEAR 4/Testing/CCTV_DGP_FEATURE_FUSION_GRADIENT_V1_VM.md>)
[V31 audited findings](<C:/xampp/htdocs/YEAR 4/Testing/CCTV_DGP_PROFILE_BATCHES_V31_RESULTS.md>)

The complete previous handoff body follows. Its unreleased-packet statement
describes the prior return milestone, before the distinct diagnostic above.


**Latest research milestone - 7 October 2026: V31's 50-update structure stop independently audited.**

The returned 1,252,002,625-byte archive matches its original VM export
hash cbb89bb5fb85ca7e8b77203188164ab2472342f8d7f3b3ecd7a07f6b53c6f49b.
All 27,623 regular allowlisted files and 5,467 TRAIN assets are verified.
The original protocol/source, gradient records, stopped checkpoint and failure
remain retained. The independent local checker uses inference only, never returned
code, gradients, optimizer updates or reserved-final pixels.

V31 improves whole-TRAIN degraded landmark detail by 0.694525% at update 50,
below the unchanged 1% requirement; V30's same-baseline gain was 0.805717%.
No continuation or automatic promotion is permitted. All 17 delivered preservation
groups pass at this stopped snapshot. That does not waive the structure stop or
establish native usefulness. The paired-batch change did not qualify this recipe.
All 50 fixed TRAIN previews were viewed in ten original-resolution sheets;
eyes, nose, mouth, outline and visible appearance remain in the review together.
The stopped result remains soft without useful whole-face improvement established.

A static source and checkpoint review confirms that all FPN state entries stayed
fixed while all 12 active decoder tensors changed. The next distinct hypothesis
measures the original five lateral and three top-down convolutions: 11 fusion
tensors with 479,616 parameters. A finite zero-update existing-L4 diagnostic must
test connectivity and preservation tradeoffs before selecting new trainable
weights. This source finding does not prove a unique cause or improved capacity.
Keep the backbone, evaluation statistics, inactive-head finding and all gates.

At 2026-10-07T09:30:56.542780+00:00, the read-only maintenance observation confirms no
V31 worker and an idle GPU; export is complete with the failure preserved.
The agent starts, stops or modifies no VM workload. Two original small export
sidecars were read exactly to verify the user-downloaded archive. Actual new
training remains the manual finite existing-L4 workflow. No unchanged failed
recipe or immutable historical pilot may be launched automatically.

A separate completion source review verifies five official MAT files at commit
d273d891ecdad2e1df106516423a75bc45b2d800, including mask polarity, loading and
runtime prerequisites. No MAT checkpoint is acquired or run. Its usefulness,
dependency/checkpoint provenance and local parity remain unverified. The previous
copied-fragment versus generated-anatomy diagnosis remains binding; a different
prior cannot fix pixels retained outside the reviewed removal area. Automatic
and assisted completion still require separate whole-family review.

The existing app, original own-DGP primary model, checkpoints, splits, research
caches, local backup and failed gates remain preserved. No new training packet
or app candidate is released by this milestone. Useful native restoration, all
seven automatic/assisted covering families and independent final review remain
outstanding. The full goal is active/incomplete.

[V31 audited results](<C:/xampp/htdocs/YEAR 4/Testing/CCTV_DGP_PROFILE_BATCHES_V31_RESULTS.md>)
[Completion source review](<C:/xampp/htdocs/YEAR 4/Testing/CCTV_DGP_COMPLETION_MAT_SOURCE_REVIEW_V1.md>)

The complete previous handoff body follows. Its live V31 observation is historical
and superseded by the terminal snapshot and independently audited return above.


**Latest diagnostic milestone - 7 October 2026: completion pixel origin audited; manual V31 run observed live.**

A separate saved-pixel R1 diagnostic verifies 300 source bindings, all 32 eligible
Off photographic outputs, 28 raw completion compositions and four exact empty-mask
bypasses. Four earlier exclusions retain their later operator input-review
decisions. Eight post-hoc diagnostic rectangles distinguish exactly retained
uploaded pixels from generated pixels. They are not covering truth, facial-region
annotations or independent evaluation. No neural/gradient/optimizer calls, mask
edits, app changes or new pilot are involved; the two failed diagnostic attempts
and their source/receipts remain preserved.

The lower-right hand-join rectangle retains 874/1,656 source pixels (52.8%);
the central scarf/chin rectangle estimates 5,138/5,687 pixels (90.3%). Retained
support includes ordinary visible face, so these are not covering-miss rates.
The earlier context6 variant also retains all source support exactly and cannot
remove a copied fragment outside the reviewed output footprint. Generated
texture/anatomy and the earlier gaze issue need a separate completion-prior
comparison on fixed reviewed support. Global dilation or another unchanged
context recipe is not justified. Both original-detail support sheets are reviewed.
These exposed photographs provide no hidden accuracy, ethnicity, native CCTV or
Zamboanga-performance claim; no hidden PSNR/SSIM is computed.

The read-only VM snapshot at 09:20:06 UTC confirms the human-launched V31 Python
PID 4942 is live on the GPU and evaluating update 50 over all 3,905 TRAIN cases.
The log reaches case 2,750 at that snapshot; no terminal result/export receipt is
observed. This is timestamped progress, not an early-gate or training pass. The
agent starts, stops or modifies no workload. The frozen packet/protocol, original
checkpoints, splits, failed gates and local backup remain retained. New actual
training remains the user's manual existing-L4 workflow. Independently audit the
completed user download before choosing any next DGP experiment or app promotion.

The full goal remains active/incomplete: useful native DGP restoration, all seven
automatic/assisted covering families and independent final review are outstanding.
Earlier app function, mask-review race fix and historical source preservation
remain verified; this diagnostic adds no model-quality qualification.

[Completion pixel-origin findings](<C:/xampp/htdocs/YEAR 4/Testing/CCTV_DGP_COMPLETION_PIXEL_SUPPORT_V1_RESULTS.md>)
[Corrected V31 tmux and download commands](<C:/xampp/htdocs/YEAR 4/Testing/CCTV_DGP_PROFILE_BATCHES_V31_LAUNCH_IMPORT_V1.md>)

The complete previous handoff body is preserved below. Its launch-unstarted
statements are historical and superseded by the timestamped manual-run observation.


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


**Latest 6 October 2026 - VM storage cleanup completed; future VM use preserved:**
The user authorizes direct VM storage maintenance and chooses "pick what doesnt
disrupt the future vm usage." On the existing running forensic-dgp-thesis VM,
seven duplicate archives totaling 2,341,409,251 bytes (2.18 GiB) are removed after
exact Windows backup/hash verification. Final live free space is 9,880,526,848
bytes (9.20 GiB). No pip-cache or scientific-cache file is removed.

Separate live VM and Windows audits verify the exact deletion ledger, all seven
retained recovery copies, 2,630 unchanged protected hashes, 4,817 retained tensor
stamps and 1,139 current V22/V23 bindings. Final runtime/cache inspection confirms
all 4,431 historical cache files (36.74 GiB), Python3.10.12/PyTorch2.9.1+cu129,
NumPy/OpenCV and available CUDA on NVIDIA L4. GPU idle; dgp_detail_skip_v23 tmux
contains Bash. The installed environment remains ready for future sessions.

Stop only the assistant-created incomplete Windows backup process tree; no
remote task is killed. Its 3,205,955,584-byte partial file is retained as
unverified evidence and cannot authorize VM cache removal. The VM stays running;
no instance start/stop, model load, actual training, app change or promotion.
This maintenance authorization is specific to storage; training remains manual.

The V23 results archive, checksum and export receipt are now present locally.
Archive size76,481,832 bytes/SHA256
`b71639cabd17476d5989dfd4582d47de9b747b8bf2a611232e5f62cc67bf9fd9`
match the reported return. The concurrent research milestone below records the
independent V23 audit and prepares a distinct V24 objective experiment. Preserve
that entire entry and its source/result/transfer artifacts. This maintenance
performs no additional model-quality evaluation; the V23 failure stays closed.

Original handoff/report bytes and the newer concurrent research handoff are
preserved under `outputs/cctv_dgp_vm_storage_cleanup_20261006_v1/before_docs/`
and `before_docs_r1/`. The entire current research body remains unchanged.
Evidence: `outputs/cctv_dgp_vm_storage_cleanup_20261006_v1/closure_manifest.json`.
Report: [VM storage cleanup](<C:/xampp/htdocs/YEAR 4/Testing/VM_STORAGE_CLEANUP_20261006.md>).
This current entry supersedes earlier stopped-VM/zero-removal notices below;
their prior evidence is retained. The full DGP restoration/covering goal remains
active and incomplete. V24 training remains on the existing manual VM path.


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


## Current restoration milestone — 6 October 2026: V22 R1 failure audited; different V23 packet ready

The downloaded V22 R1 archive matches76,931,848 bytes/SHA256
`4a734450e9ea8fca8fb240a5e06b55f99a28316a770869938b2bd2aa05ad9cad`.
All366 returned files,196 original assets,100 raw/PNG pairs/metric rows, both
complete50-case snapshots and exact stopped-head state pass the independent
68.30s audit. Fixed CPU replay uses100 head and110 recognizer forwards; no
original-DGP forward, backward or optimization. Ten return-auditor checks pass.
Initial torch-version rounding/Windows temporary-directory failures and their
bounded auditor-only corrections remain preserved; quality tolerances are intact.

The original L4 stop is verified:50 updates/51 backwards,0.0000322991% feature
gain versus the frozen1% early condition. All ten original-resolution sheets/
50 cases are reviewed; no visible structure gain.4,486 pixels change by at most
one byte. A50-case known-layer CPU trace finds a too-small spatial correction
through the deep route/projection. Independent saved evidence checks375 sources
and200 exact review cells. Nine original preservation checks diagnostically fail
at update50; the unexecuted800-update final gate is not called evaluated.
Close V22 without unchanged rerun, relaxed stop or app adoption.

V23 prepares a different4613parameter full256-resolution input-detail bypass
plus shallow branch, preserving data, objective,800 updates/80epochs/batch5,
original preservation/brightness gates and timing stops.209 assets/211 members,
Python3.10 grammar, Bash syntax and Windows training rejection pass. All50 zero
raw/PNG outputs are exact; fixed sensitivity and nonempty-padding contracts pass.
Preparation uses56 new-head/four closed-V22-head fixed CPU forwards; no new local
optimizer/backward or VM execution. Complete timing/VRAM/gradient/count receipts
are added. Learning, capacity and useful outputs remain pending on the manual L4.

The user feedback “its useful but needed clearer structure” remains a useful-case
finding with visible-structure work required. No usable input is relabeled
insufficient to hide model softness. Previous560 milestone and22 current app
bindings pass; prior workflow/runbook bytes are preserved before this update.
App/design, checkpoints, splits and original failed gates remain intact; historical
34 regressions/Playwright are not rerun for this evidence/transfer-only change.

No native/reserved/covering/COFW-test pixels enter V23. Native evidence remains
unpaired, photographic training metrics separate; no inferred ethnicity or local
Zamboanga performance. Useful native DGP restoration, all seven automatic/assisted
covering families, canonical app parity and independent final review remain open.
Goal active. Training execution is verified transfers/pasteable commands only.
The separate maintenance entry below records a stopped VM; it is preserved and
availability is not refreshed here. No assistant SSH/upload/start/cleanup/training.

Commands: [V23](<C:/xampp/htdocs/YEAR 4/Testing/CCTV_DGP_DETAIL_SKIP_V23_VM.md>).
Plan: [V23](<C:/xampp/htdocs/YEAR 4/Testing/CCTV_DGP_DETAIL_SKIP_V23_PLAN.md>).
Result: [V22 R1](<C:/xampp/htdocs/YEAR 4/Testing/CCTV_DGP_DETAIL_PRIOR_V22_R1_RESULTS.md>).
Evidence: `outputs/dgp_detail_prior_v22_r1_audit_and_skip_v23_milestone/`.


## Current maintenance — 6 October 2026: cleanup prepared; VM stopped before deletion

The user authorizes direct storage inspection/removal on the existing VM.
Successful read-only SSH finds7.67GiB free, idle GPU and an idle V22 Bash tmux pane.
Seven older remote archives have exact Windows backups totaling2.18GiB; the
current V22 archive/result/run assets stay protected. Historical expanded fixed/
anatomical and V16 r2 caches occupy approximately36.7GiB. Their full tensors are
not in old result archives; copy and hash them on Windows before any removal.

The VM stops before helper upload. Live API status is TERMINATED, stopped at
00:00:29 on6October Asia/Manila. The earlier TLS failure is fixed with the existing
command-scoped Windows-trusted CA bundle; verification remains enabled.
Temporary restart/return-to-stopped approval is pending. No helper upload, cache
copy, deletion or training occurred. **Files removed:0.**

The exact seven-archive cleanup plan is prepared, SHA256
`0698ddd351c6915f793e55941e655048934424cf715bb806023ad180ed72f987`.
Four maintenance helpers pass Python3.10 grammar checks; the deletion backend
rejects Windows before reading the plan. The original handoff is preserved under
`outputs/cctv_dgp_vm_storage_cleanup_20261005_r2/before_docs/`.
Archive verification/apply and cache backup/removal remain unexecuted; checkpoints,
splits, sources, installed runtimes, logs and all original failed gates remain.
This maintenance authorization does not authorize training/provisioning.
Report: [VM storage cleanup](<C:/xampp/htdocs/YEAR 4/Testing/VM_STORAGE_CLEANUP_20261006.md>).

## Earlier milestone — 5 October 2026: V22 R1 user-reported early stop; Windows download corrected

The user reports successful L4/CUDA preflight and exact four-case retained DGP
parity. Training stops at update50: feature error0.0019965976532523044 becomes
0.001996597008369677, a0.0000322991% improvement versus the frozen1% early gate.
Trainer exit1 and export exit0 are retained; the tensor-to-scalar warning concerns
logging and is not the stopping exception. Do not restart the unchanged recipe.

Expected user-return archive:76,931,848 bytes, SHA256
`4a734450e9ea8fca8fb240a5e06b55f99a28316a770869938b2bd2aa05ad9cad`.
Windows PuTTY rejects multiple remote sources in one gcloud command; corrected
step5 downloads the archive, checksum and export receipt in three separate calls.
The original runbook and this handoff are preserved under
`outputs/cctv_dgp_detail_prior_v22_r1_putty_download_fix_v1/before_docs/`.
Execution source/protocol/archive and all earlier failed gates are unchanged.

These are user-reported execution findings. Download/hash verification, independent
saved-output/head/recognizer audit and image review remain pending. Draft local
return-audit scripts are not yet qualified. No assistant VM action, local training,
app promotion or completed-goal claim. Full restoration/completion scope remains.

Runbook: [V22 R1 corrected download commands](<C:/xampp/htdocs/YEAR 4/Testing/CCTV_DGP_DETAIL_PRIOR_V22_VM.md>).

## Earlier milestone — 5 October 2026: useful-case feedback answered; V22 R1 structure pilot prepared

User: “its useful but needed clearer structure.” Record the shown ChokePoint
case as useful with a remaining visible-feature defect, not full model/scope
qualification. The crop remains input-usable; do not reject it to hide softness.
The pending diagnostic is answered and the resumed Goal is active.

V21 saved trace checks50 exposed paired training cases +24 native development
crops,122 raw/PNG compositions and244 regional measures in5.00s; independent
2.85s audit verifies379 sources. All24 native own-DGP residuals attenuate
input-aligned eye detail (median slope0.704009). This occurs in raw floats;
PNG is not the cause. Captured detail may include noise; native remains unpaired.

One fixed captured-detail control fails15 preservation checks; degraded paired
MSE rises0.08998%. All ten original1072×1504 sheets are reviewed. A restricted
learned-band analytic bound cannot reach its predeclared10% paired-MSE capacity
condition (optimistic ceiling5.44026%). Both routes close without app adoption.
Independent control/bound audits take5.19s/2.97s. Original failures remain;
separate R1 corrects only two duplicated clear-group counts, with metrics intact.

V22 prepares a new45443parameter own-DGP detail head, without CodeFormer features
or RGB, projected to high-pass and zero observed mean before clamp. Same ten
exposed training photographs/50 cases;800 updates/80 epochs, batch5. Structure
capacity uses >=10% landmark-HF error reduction, historical MSE/SSIM/ArcFace
preservation tolerances and <=20% brightness-only gain. Early/timing stops and
all failed/partial outputs are retained; no source/profile exceptions or automatic
follow-on. New prospective structural criterion does not erase old failed gates.

Original preparation verifies195 assets,192 original bindings,80 scheduled epochs and
correct Windows training rejection. Four no-gradient CPU head contract calls
pass; no optimizer/backward or original DGP/recognizer call. Final tested class
AST unchanged; Bash syntax and all197 archive member hashes pass. CUDA gradients,
L4 timing, capacity and outputs remain pending. Active R1 archive207.48MiB, SHA256
6978e423c23909caebff65c7299267ce1a6803d15e2818e38bac6b1f685fdaba.

Active execution revision R1 replaces the newer checksum API with streaming
SHA256 for Python3.10 in the VM trace; original unexecuted V22 packet remains
archived. Exact head/core functions, data, schedule, objective and gates are
unchanged. R1 verifies196 assets, Python3.10 grammar and198 safe archive members;
Bash syntax and Windows training rejection pass. Use the R1 runbook commands.

Actual training is user-run on the existing L4/g2-standard-4 at ~/forensic-dgp:
manual verified gcloud transfers/pasteable commands only. Preflight300s, fit1500s,
worker1800s, external2100s; export120s. No assistant SSH/upload/launch/cleanup.
Returned results need independent source/budget/metrics/head/recognizer audit
and all50-case review before a separately frozen broader development experiment.
No app candidate, native/final output, new completion training or promotion.

All preceding52 milestone bindings and22 app bindings pass; preceding workflow
documents are preserved byte-exact. Existing app design/source/checkpoints,
original splits/gate failures and all unviewed final reserves remain intact.
Historical34 regressions/Playwright are not rerun here. Full useful DGP-native
restoration, automatic/assisted covering families and independent final review
remain required. Goal active, awaiting manual VM execution/returned results.

Runbook: [V22 gcloud/tmux/download steps](<C:/xampp/htdocs/YEAR 4/Testing/CCTV_DGP_DETAIL_PRIOR_V22_VM.md>).
Report: [V21 structure trace and controls](<C:/xampp/htdocs/YEAR 4/Testing/CCTV_DGP_STRUCTURE_TRACE_V21.md>).
Evidence: `outputs/dgp_structure_and_detail_pilot_milestone_v21_v22/`,
`outputs/cctv_dgp_detail_prior_v22_r1_preparation/`,
`outputs/cctv_dgp_detail_prior_vm_v22_r1/`.

## Earlier milestone — 5 October 2026: final-review preparation audited; final execution not ready

Independent-review protocol V1/R1 and blank forms are prepared for 58 reserved
native cases/45 namespaced release person labels: 116 input rows, 580 blinded
output rows, 54 covering-template slots and 34 five-milestone requirements.
The 0.03s preparation audit verifies metadata, arm assignments, empty verdicts
and exact notices. V1 remains byte-exact; the expanded blinding clarification
is separate R1. Reviewers are unassigned, final model/cohort bindings empty,
and reserved native pixels remain unviewed. No final verdict is claimed.

Reuse the existing verified COFW colour archive for metadata-only candidate
preparation: all 507 publisher-test rows and 64 deterministic input candidates
(16 in each of four sparse-flag bins) are independently rederived. Preparation
and audit take 1.46s/1.26s with three metadata-array reads, zero image-value reads
and zero model/training calls. Sandbox reader failure and HTTP 403 refresh are
retained; the existing CC BY 4.0 notice is copied exactly. The registry's 42
COFW records are author-training-only; this does not prove person disjointness
or complete historical exposure. Flags do not label covering families/masks.
COFW is photographic occlusion evidence, not CCTV or hidden-face ground truth.
No test RGB, training admission or final covering-cohort selection occurs.

Useful own-DGP native restoration and full automatic/assisted covering acceptance
remain unqualified. Model-path changes remain stopped for the pending single
native-softness diagnostic review; frozen failed gates remain intact. All 160
preceding milestone bindings and 22 app source/evidence bindings are checked;
the preceding three workflow documents are preserved exactly before this update.
No app/model change, neural call, VM action or regression/browser rerun.
Final candidate/cohort/reviewers and final app verification remain required.
Goal active; training remains manual L4 transfers/pasteable commands only.

Report: [Independent review readiness](<C:/xampp/htdocs/YEAR 4/Testing/INDEPENDENT_REVIEW_READINESS_V1.md>).
Protocol: [V1 R1](<C:/xampp/htdocs/YEAR 4/Testing/INDEPENDENT_REVIEW_PROTOCOL_V1_R1.md>).
Evidence: `outputs/dgp_independent_review_protocol_v1/`,
`outputs/cofw_final_candidate_metadata_v1_r1/`,
`outputs/dgp_independent_review_readiness_milestone_v1/`.

## Earlier milestone — 5 October 2026: V20 motion guards fail; model changes stopped for diagnostic review

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

## Earlier milestone — 5 October 2026: structured native CCTV comparison remains negative for own DGP

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

## Earlier milestone — 5 October 2026: full development covering review; context comparison negative

The unchanged DGP research route completes 36 photographic development cases/
108 requests in 428.18s: 36 detector, 84 completion and 49 DGP forwards, all model
states unchanged. Twelve expected operator exclusions (three-quarter pose and
near-total hands, both conditions/all modes) leave 96 generated outputs. A 6.15s
independent saved-output audit verifies 151 source/336 output bindings, exact
compositions, 32 Auto aliases, all Off visible bytes and identical completed pixels.
This is reused original/degraded photographic evidence, not native CCTV.

Both input sheets were reviewed before inference. All six route-output sheets
are reviewed, displayed at 1504×1649 from 1608×1764 sources. Automatic proposals
miss sunglasses, glare, hands and hair; degraded masks are empty in 14/18 cases,
including appropriate controls. Cached existing91/varied133 comparisons remain
unqualified, with original gate failures intact. Assisted cases include plausible
estimates but finger/scarf remnants, poor joins and inconsistent estimated gaze.
Protected-file support is zero in every generated case: original zero-change
records do not prove On eyewear preservation. The separate clarification retains
the old receipt; full Off visible checks/four whole-image controls remain valid.

Paired synthetic nonremoved MSE worsens with On in all 11 family groups;
SSIM improves in three and regresses in eight. Auto is exact but chooses On in
all 16 supported degraded cases. No clean hidden reference/identity/native metric.
Separate fixed completion-context comparison: 32 CPU DGP forwards/17.07s,
unchanged state, no training; 4.81s saved-array audit/32 exact PNGs/four exact
empty-mask raw controls. All eight original-resolution sheets reviewed. Only
1/16 MSE and 3/16 SSIM improve versus Off; no convincing structural gain.
The context variant is not adopted; app sources/thresholds remain unchanged.

Bundled inline Playwright passes glare/hand/hair assisted flows, renewed review,
one output/PNG/ZIP and near-hidden operator rejection in 30.25s, with no page/
console errors or overflow at 375/768/1280. Three downloads and bundle original/
mask/raw/result exactly match saved inference. Earlier 34 app regressions remain
the unchanged verified baseline. Previous integration record bindings stay intact.

Native usefulness and full automatic/assisted-family acceptance remain negative
or unproven; independent final review is incomplete. The 32 reserved native crops
remain untouched. No local training, VM/cloud action or new VM pilot. Manual
transfer and pasteable VM commands-only preference holds; goal active.
Report: [CCTV_DGP_APP_V3_COVERING_RESULTS.md](<C:/xampp/htdocs/YEAR 4/Testing/CCTV_DGP_APP_V3_COVERING_RESULTS.md>).
Evidence: `outputs/dgp_app_covering_review_v3/`,
`outputs/dgp_app_completion_context_v3/`, `scratch/dgp-v3-family-*`.
Do not repeat a failed recipe or retune this exposed development cohort.

## Earlier milestone — 5 October 2026: DGP-led research app browser verified

The main `/face` route now uses our retained trained identity-v2 DGP at 256×256,
with observed-support Auto/On/Off and required operator input/removal-area review.
New engine `dgp_face_workflow_v3.py`; weight SHA256
`646fbb11674e8882ea41e251d6e6e7d468c963979b48c93132def8292834e24b`.
This is functional research integration, not useful-native/family acceptance or
promotion of rejected V18/V19/r2. Historical engines/adapters/checkpoints/splits
and failed gates remain intact. No pretrained restoration fallback is used.

34 regressions pass. Finite CPU audit: 4.56s/eight DGP forwards, exact cached raw
parity on all six native core cases, exact repeat/Off preservation, unchanged
weights. Eighteen other native cases replay operator clearer/out-of-scope decisions;
no automatic classifier claim. All six native rows reviewed: still soft, with no
useful structural improvement. No reserved evaluation or local training.

Bundled inline Playwright verifies real-model flows and 375/768/1280 layout/console
checks. Mask correction, keyboard paint/undo, renewed review, Auto/On/Off, one
output, PNG/ZIP and missing-review/insufficient-input rejection work. Saved
downloads verify native original/raw/result hashes, Off visible bytes and
completed-pixel consistency. The corrected harness retains its earlier mistaken
Auto-always-On assertion failure; no app threshold was changed.

Automatic cloth proposals miss substantial covering and mark visible/background
points. Assisted mild-turn cloth completion is plausible; difficult three-quarter
mask/glasses diagnostic has nose residue. Clear-glasses control with empty mask/Off
is exact. These reused photographs are separate from native CCTV. Full family
usefulness and independent final review are still required and incomplete.

The app was started locally at `http://127.0.0.1:8000/` for review; launch PID/logs
are in `scratch/dgp-app-v3-server.*`. No VM/cloud action or training occurred.
Manual transfer/VM commands-only preference persists. Goal active.
Report: [CCTV_DGP_APP_V3_INTEGRATION.md](<C:/xampp/htdocs/YEAR 4/Testing/CCTV_DGP_APP_V3_INTEGRATION.md>).
Evidence: `outputs/dgp_app_v3_integration_record.json`,
`outputs/dgp_app_v3_inference_audit/`, `outputs/dgp_app_v3_visual_review.json`
and `scratch/dgp-v3-*`. Do not rerun closed V19/r2 or weaken preservation gates.

## Earlier milestone — 5 October 2026: V19 r2 return audited/reviewed; preservation fails

The user returned the successful 1,507,038,572-byte V19 r2 export. Its sidecar,
receipt, SHA256 and results agree; all 189 frozen assets and protocol match.
Archive `fefc78e16ab9a1f4d78d7faf4a8344ad3eede088918ce5d522b266760728ef10`;
results `98c311465351b36c20d26eca2a1d2310194b1ac255ebfad82f6d49b5c039f9c8`.
Safe return: `outputs/cctv_dgp_input_selection_return_v19_r2/`.
The L4 completes 520 development cases, inference 175.04s and full export 356.31s,
with zero optimizer/backward/training calls and all six state hashes unchanged.

Independent local audit 97.61s passes 5,957 artifact bindings, 2,080 PNG metric/
cosine rows, 1,040 raw/PNG compositions, 520 internal DGP bases, 570 decisions,
520 exact aliases, 50 fresh training parity cases, 50 canonical raw previews and
300 original grid cells. All 24 cached CPU decoder replays pass, maximum delta
7.15e-7 within unchanged 5e-5. Full DGP/prior/recognizer/CUDA replay is not claimed.
All 520 fresh canonical DGP PNGs exactly match V15: the normalization correction
works. The original V19 failure and diagnostic remain intact and closed.

**Scientific result remains negative:** automatic degraded synthetic PSNR gains
3.027dB/MSE falls 50.19%, but SSIM falls 0.61930→0.60458 and fixed ArcFace cosine
0.33011→0.22346. Automatic fails 30 group/metric preservation checks; unconditional
spatial output fails 33. Blur/motion PSNR regresses in both source folders; every
degraded source/profile cosine regresses. The selector keeps 101/104 clear cases
on DGP, but three clear cases route to the spatial output and lose SSIM/cosine.
No threshold refit or quality waiver; the candidate is not adopted.

All five original-resolution sheets are reviewed: ten fixed development preview
identities/fifty cases/300 cells. Mottled colour/texture, changed or weak facial
detail and unstable glasses remain visible. This is development review, not
individual review of all 520 outputs or independent final assessment.

A separate saved-output uniform-luminance diagnostic completes in 17.23s, zero
neural/training calls. It reproduces 83.59% of the spatial candidate's mean MSE
reduction in these synthetic cases, while still failing 13 MSE/SSIM checks.
Fresh recognizer scores/full qualification are unavailable. It is exploratory
component analysis, not an accepted restorer. The analysis startup manifest-name
error is preserved; only its new script was corrected, with no frozen changes.

Close V19 r2 launch commands; do not repeat, scale the failed spatial recipe or
retune the selector on this cohort. The next design must isolate exposure
correction from spatial synthesis, preserve clear/blur/motion appearance and
justify any separate finite VM training from training-only evidence. No new VM
pilot or assistant cloud action was performed. Manual execution preference holds.
Report: [CCTV_DGP_INPUT_SELECTION_V19_R2_RESULTS.md](<C:/xampp/htdocs/YEAR 4/Testing/CCTV_DGP_INPUT_SELECTION_V19_R2_RESULTS.md>).
Evidence: returned `local_full_audit.json`/`local_import_and_audit.json`, and
`outputs/cctv_dgp_input_selection_v19_r2_analysis/` (all groups, input choices,
original-sheet review and component analysis).

The 104 repeatedly used photographic development identities/520 synthetic cases
do not establish native CCTV, Zamboanga or independent final performance. Native
24/reserved 32 remain untouched by R2. DGP-led app/override, input usability,
useful native outputs, covering families and independent final/Playwright flow
remain incomplete. Goal active; preserve all original checkpoint/split/failure data.

## Earlier milestone — 5 October 2026: normalization cause confirmed; V19 r2 inference prepared

The returned three-case L4 diagnostic is safely imported and independently
audited. Its 7,380,842-byte archive matches receipt/sidecar/SHA256
`9d910721986df024a3ba346766c6385487603bb5174ebc771456c3d8159d7f1c`.
Eight frozen DGP forwards complete in 12.22s; supervisor/export 13.72s. The
0.49s local saved-array audit verifies eight raws, six normalized tensors,
six PNG compositions and all 148 protected original bindings, with zero local
forwards, optimizer updates or backwards. Repeated GPU forwards are exact and
the frozen DGP state is unchanged; no local whole-network replay is claimed.

NumPy normalization before GPU transfer exactly recovers both fixed V15 PNGs,
including the first stopped development case. CUDA scalar division changes
four colour-channel values by one byte in each of those PNGs, while exactly
matching the fixed V18 training raw/PNG. Replacing V18's convention would
instead change three training PNG channel values. Both paths receive identical
RGB bytes; tiny measured float differences cross the unchanged PNG floor.
The original V19 failure and diagnostic are closed; preserve both.

A separate immutable V19 r2 correction preserves V18's spatial input path and
adds a canonical V15-encoded retained-DGP forward for each development case.
Raw outputs are now saved before baseline checks, and the internal spatial DGP
base is exported separately. All original scientific/parity gates, checkpoints,
splits, selector thresholds and 104-identity/520-case cohort remain unchanged.
This is inference only: no training, fitting, threshold refit or checkpoint search.

Eight regressions pass. Preparation 44.58s produces an 87,831,525-byte archive,
191 members/189 assets, including all 146 original V19 members byte-identically
and the audited diagnostic. Independent transfer audit 2.72s verifies 798 data
files and 35 Python 3.10 sources. Actual frozen-package import/source/data/lineage
preflight passes in 16.34s with zero neural forwards. No R2 VM launch is claimed.
Protocol SHA256:
`5c128d6715785f84858f762035a168f396b46787d6a864b1d8d6437ffc69dc3f`.
Archive SHA256:
`3397a6c3e81e1b04a492f51ec6e5fad939b5a85bc8b318253a04c7696eba581e`.

**Next:** user-run upload/launch/download commands in
[CCTV_DGP_INPUT_SELECTION_V19_R2_VM.md](<C:/xampp/htdocs/YEAR 4/Testing/CCTV_DGP_INPUT_SELECTION_V19_R2_VM.md>).
Fresh root `~/forensic-dgp/cctv_dgp_input_selection_vm_v19_r2`; launcher creates
detached tmux `dgp_input_selection_v19_r2`. Inference 1,200s/audit 300s/overall
1,800s; timing stop, 20GiB VRAM and 4GiB free-disk requirements remain. After
return, independently audit and review all five original-cell development sheets.
Evidence: `outputs/cctv_dgp_input_selection_v19_parity_diagnostic_return_r1/`,
`outputs/cctv_dgp_input_selection_v19_r2_preparation.json`,
`outputs/cctv_dgp_input_selection_v19_r2_transfer_audit.json` and
`outputs/cctv_dgp_input_selection_v19_r2_package_preflight.json`.

V18's unconditional preservation failures stay unchanged. No useful native,
reserved, independent final, Zamboanga, generalization or adoption result exists
from this diagnostic. DGP-led app, insufficient-input behavior and covering-family
readiness remain incomplete. Goal active; manual VM execution preference persists.

## Earlier milestone — 5 October 2026: V19 failure audited; separate parity diagnostic prepared

The user downloaded the original V19 failure. Its167,982,054-byte archive,
sidecar/receipt and all144 original assets match. Safe import preserves the
failure in`outputs/cctv_dgp_input_selection_failure_return_v19/`. SHA256:
f6847aa742acee19c21a16df7f44dce2456b052793cf7fa154a62cd59abcb16b.
After50 successful fresh training parity cases and104 target embeddings, the
first development case`va_ffhq_28544_clear` stops at exact baseline PNG equality.
Supervisor31.59s before export; zero completed development predictions. No
final state/counter receipt or full output audit exists. No quality conclusion.

Separate5.59s partial audit verifies100 training raw/PNG compositions (maximum
raw difference0),50 input decisions,728 identical camera/target/support images,
104 fresh target embeddings (maximum difference2.12e-7≤2e-6), four cache schemas
and six initial state hashes. Zero local forwards/backwards/updates.
Receipt:`outputs/cctv_dgp_input_selection_v19_failure_audit.json`.

V15 normalized in NumPy before GPU transfer; V19 used CUDA scalar division.
PyTorch2.9.1's scalar CUDA kernel multiplies a reciprocal. CPU simulation changes
126/256 byte values by≤5.96e-8; ordinary CPU division does not reproduce this.
The failed raw/PNG was not saved, so causal attribution still needs measurement.

A separate21,214-byte diagnostic/sidecar is ready, with four regressions and
independent0.29s transfer audit. Exactly three fixed cases/eight frozen DGP
forwards; no other networks/fitting/full validation. Worker120s/supervisor240s/
export60s inside overall; separate root/tmux. Save both normalized tensors and
raw/PNG outputs before comparison; preserve all148 protected original bindings.
No gate relaxation, V19 resume or assistant VM launch. Script SHA256:
cad6da856e74ae9dcaa28d463304c93997d7e21ebee49fb65e5fa8a6d030e729.

**Next:** user-run verified gcloud/tmux/download commands in
[CCTV_DGP_INPUT_SELECTION_V19_PARITY_DIAGNOSTIC_VM.md](<C:/xampp/htdocs/YEAR 4/Testing/CCTV_DGP_INPUT_SELECTION_V19_PARITY_DIAGNOSTIC_VM.md>),
then independent local return audit. Original V19 launch commands are closed;
do not repeat or edit that frozen package. Full failure report:
[CCTV_DGP_INPUT_SELECTION_V19_FAILURE.md](<C:/xampp/htdocs/YEAR 4/Testing/CCTV_DGP_INPUT_SELECTION_V19_FAILURE.md>).
V18 failures/scientific gates remain unchanged. No native/reserved/Zamboanga,
generalization/adoption or completion claim. DGP-led app, insufficient-input
handling, covering families and independent final review remain incomplete.
Goal active; user's manual VM execution preference persists.

## Earlier milestone — 5 October 2026: input-only processing verified; V19 inference prepared

After closing V18's audited clear preservation failure, a separate input-detail
study uses only fifty photographic training inputs and canvas support. The frozen
selector retains DGP when Laplacian energy≥5e-5 AND energy/luminance variance≥.002;
exact constant luminance also retains DGP. Runtime takes input pixels/support only,
no labels/identities/targets/output scores. The training-calibrated control chooses
DGP for ten clear cases and fixed V18 update600 for forty degraded cases. Exact
output aliases preserve all unchanged training-group guards; capacity gain84.03%.
Control2.91s; separate50-case raw/PNG/decision/artifact audit2.53s; zero neural/
training calls. Ten focused selector/inference-contract regressions pass. This
does not change failed unconditional V18 results or prove generalization.

**V19 prepared, not launched:** inference-only on unchanged104-reference/520-case
development validation (51/53 source split), preceded by all50 fresh training
parity cases. Five identical-input arms: resizing, retained DGP, audited pretrained
CodeFormer-none PNG baseline, fixed V18 update600, automatic selector alias.
Raw/PNG separate; no display transforms, threshold refitting, checkpoint search,
optimizer or backward. Prior/recognizer remain declared pretrained and frozen.
Native24/reserved32 remain untouched. This repeatedly used photographic validation
is development evidence, not independent final/native/Zamboanga acceptance.

Inference1,200s/audit300s/supervisor1,800s/export180s within overall, VRAM20GiB,
free disk4GiB. Case20 steady timing projection must fit. Original scientific
guards stay unchanged; execution success exports failed quality too. Nine
module counters have frozen expected forward counts; all six state hashes stay
unchanged. Local returned audit covers2,080 physical PNG metrics/cosines,1,040
raw compositions,570 decisions,520 exact aliases,50 parity cases,300 grid cells
and24 cached CPU decoder probes (5e-5). CUDA/frozen whole-network replay limits
are explicit; inspect all five original-cell sheets after return.

Execution archive80,388,709 bytes,146 exact safe members/144 assets. Preparation
35.45s; independent transfer audit2.39s verifies798 data files, own exact train/
development split exclusion, Python3.10 syntax, lineage, limits and counts with
zero neural/backward/training calls. Protocol SHA256:
2d707ebba97524867c4ce7610666e3abeb29638eb9054a8025983b6476e628d2.
Archive SHA256:
da75c16e25c0b83183beda230d90c552c9c24c8f6c070f8a992f77042350c899.
VM root`~/forensic-dgp/cctv_dgp_input_selection_vm_v19`; detached session
`dgp_input_selection_v19`. No assistant cloud call/VM launch/shutdown.

**Next:** user-run exact gcloud/tmux/download commands in
[CCTV_DGP_INPUT_SELECTION_V19_VM.md](<C:/xampp/htdocs/YEAR 4/Testing/CCTV_DGP_INPUT_SELECTION_V19_VM.md>).
Plan:[CCTV_DGP_INPUT_SELECTION_V19_PLAN.md](<C:/xampp/htdocs/YEAR 4/Testing/CCTV_DGP_INPUT_SELECTION_V19_PLAN.md>).
Evidence:`outputs/cctv_dgp_input_selector_v19/`,
`outputs/cctv_dgp_input_selection_v19_preparation.json`,
`outputs/cctv_dgp_input_selection_v19_transfer_audit.json`.
V19 is frozen; preserve partial results/failures and do not regenerate/repeat.
Insufficient-input gating, useful native restoration, covering completion families,
DGP-led app/override and independent final/Playwright flow remain incomplete.
The goal stays active. Manual execution preference persists.

## Completed V18 milestone — 5 October 2026: return audited/reviewed; clear preservation failed

The user downloaded V18 after VM storage maintenance. Successful archive
481,723,919 bytes matches SHA256
d494351f1f1f36d60f79f696633ff6e0dcc5cdf1f03662f84884f4aae71787e5,
sidecar and terminal receipt; results SHA256
f0acefaadbf2890627339038371673aeed30cf9a066b9f74ca71d28766e9c206.
All21 original/returned frozen assets and protocol match. The L4 completed600
updates/6,000 exposures; fitting133.35s, full supervisor/export216.65s. Five
frozen state hashes agree;42 decoder tensors change at each trained snapshot.

Original local audit failed at exact float32 training-error accumulation. Separate
r1 checks against float64 accumulation and retains exact weight derivation, then
fails at binary64 log10/PSNR equality. Largest differences1.49e-9 error mean and
7.11e-15dB. Separate r2 corrects only those arithmetic checks; every non-PSNR
mean, fitting stop, quality decision/failure list and parity threshold stays
unchanged. Both failures, original sources and checkpoints remain. Ten focused
numeric regressions pass. Complete CPU audit60.29s verifies250 output replays/
metrics/cosines,50 exact starting DGP cases,eight fresh VM parity cases and550
original grid cells. Maximum CPU replay difference9.5367431640625e-7≤5e-5.
Zero local training/backwards; no local DGP/prior/recognizer replay.

All ten original-cell sheets are reviewed. Degraded training PSNR15.5527→23.5201,
SSIM0.62517→0.70649 and fixed appearance0.39847→0.95073;84.03% MSE reduction.
This is ten-photograph fitting evidence. Update600 retains four failed clear
preservation checks: aggregate/source SSIM and FFHQ-source clear MSE. Every
trained snapshot remains unqualified for separate generalization under V18.
Soft/patchy compound features and weak glasses remain. No held-out/native/
reserved/teacher/selection/promotion; source folders are not ethnicity or local
CCTV performance. No exact-identity/generalization claim.

Report: [CCTV_DGP_STRUCTURE_V18_RESULTS.md](<C:/xampp/htdocs/YEAR 4/Testing/CCTV_DGP_STRUCTURE_V18_RESULTS.md>).
Evidence: `outputs/cctv_dgp_structure_v18_audit_recovery_r2/`,
`outputs/cctv_dgp_structure_v18_verified_summary.json`,
`outputs/cctv_dgp_structure_review_v18.json` and the unchanged return directory.
V18 launch/import commands are now historical/closed; do not repeat them.
Next: diagnose an input-only clear-preservation/automatic-selection processing
control on training inputs before another separately justified protocol. Do not
waive V18 failures or repeat its recipe. DGP-led app, insufficient-input behavior,
useful native outputs, covering families and independent final/Playwright review
remain incomplete. Goal stays active; training remains user-run VM-only.

## Earlier maintenance — 5 October 2026: VM storage recovered before V18

The user reports V18 upload steps1/2/3 complete and explicitly authorizes direct
gcloud connection/removal of unnecessary VM files. This is a storage-maintenance
authorization; V18 training stays on the manual execution path. No trainer,
GPU task or tmux session was active during the initial inventory.

Read-only SSH confirms `/` initially97% used with3.8GiB available. Cleanup
removes only74 remote archive copies proven byte-identical to preserved Windows
archives, plus104 disposable pip download-cache files. Total178 exact regular
file unlinks; no recursive directory deletion, package change or training.
Archive bytes8,768,743,172; pip cache bytes900,370,624. Verification119.55s;
cleanup and post-integrity verification79.72s. Recovered9,669,484,544 bytes
(9.0GiB). Separate `df -h /` confirms **87% used,13GiB available**.

All10,480 protected hashes and4,739 scientific tensor file records match before/
after. Original source, checkpoints, protocols, splits, input data, result/failure
logs, installed runtime and the three V18 uploads remain. Both the28GiB expanded
feature cache and8.7GiB R2 cache stay intact. The R2 return records cache bindings,
not the full cache tensors; it is not a tensor backup. The unmatched camera-review
archive is retained. Older remote archive paths listed in historical runbooks can
now be absent; their exact copies are in Windows `outputs/` and recorded below.

Local evidence: `outputs/cctv_dgp_vm_storage_cleanup_20261005/` contains the
downloaded cleanup/verification receipts, protected-file fingerprints and exact
removal ledger. Its separate local audit passes all178 removal records, receipt/
plan/script bindings and rehashes all74 preserved Windows archive copies in11.05s;
the independent live disk reading agrees. Receipt: `local_independent_cleanup_audit.json`.
Plan: `outputs/cctv_dgp_vm_cleanup_plan_20261005.json`;
backup bindings: `outputs/cctv_dgp_vm_cleanup_local_archive_backups_20261005.json`.
VM evidence: `~/forensic-dgp/maintenance_storage_20261005/`.
Plan SHA2568a52e4fcd499cf21a522ce31e3bb39df845480768f3de080d54592937fddaea8.
No VM stop or V18 launch. The assistant verified the three existing V18 upload
hashes; re-upload is unnecessary. Resume steps4/5 in
[CCTV_DGP_STRUCTURE_V18_VM.md](<C:/xampp/htdocs/YEAR 4/Testing/CCTV_DGP_STRUCTURE_V18_VM.md>).
The full restoration/completion/app/final-review goal remains incomplete.

## Earlier preparation — 5 October 2026: V18 interface verified; finite manual VM pilot prepared

The new output path retains the audited DGP V2 pixel image and learns a spatial
RGB residual from the camera/DGP images and declared frozen CodeFormer features.
Our audited R2 epoch8 code head remains frozen; only the new 684,395-parameter
decoder is fitted by V18. This changes the failed CE-only output path and loss,
with explicit pretrained provenance. It is not an adopted restoration model.

Two fresh training-only CPU inputs passed exact initial DGP parity, including
the zero-prior control, in13.60s. All frozen states match before/after; zero
backwards/optimizer updates. A separate0.1064s saved-output audit passes eight
artifacts. Thirteen schema/parity/sensitivity/VM-guard/schedule/preservation
regressions passed; actual CUDA gradient preflight is pending VM execution.

**Frozen V18:** ten photographic training references/fifty cases, balanced
600-update/6,000-exposure limit; each case120 exposures. Snapshots0/50/200/600,
50-update≥1% full-cohort fitting stop, strict clear/source/profile preservation
and≥10% degraded MSE gain before a separate generalization experiment.
Cache300s, fit900s, audit300s, supervisor1,800s/30min, export180s within overall,
20GiB VRAM and4GiB free disk. No validation/native24/reserved32 forwards,
automatic selection/resume/promotion or local training.

Archive `outputs/cctv-dgp-structure-v18-execution.tar.gz` is9,063,630 bytes.
Preparation24.33s reverified existing dependencies; separate transfer audit0.3027s
passes23 exact safe members/21 assets, twelve Python3.10 source files, all70
selected data files, schedule/parameter/count arithmetic and unchanged R2
protocol/checkpoint. Protocol SHA256:
e3a7567e9a9c53a1668464ab5c95edbf094c73e9a7c8e550635dcf8d8ef04adb.
Archive SHA256:
ecc8a24aa77e23f80bbd7e482573ed666e443e35c7ee5633ddf4e5a6edeb2ce7.
Evidence: `outputs/cctv_dgp_structure_v18_preparation.json`,
`outputs/cctv_dgp_structure_v18_transfer_audit.json`,
`outputs/cctv_dgp_structure_prototype_v18/`.

**Next:** use the exact user-run gcloud upload/launch/collection commands in
[CCTV_DGP_STRUCTURE_V18_VM.md](<C:/xampp/htdocs/YEAR 4/Testing/CCTV_DGP_STRUCTURE_V18_VM.md>).
Launcher creates detached tmux `dgp_structure_v18`, permits idle outer shells
and preserves competing tasks/partial roots. Pilot preparation used no cloud
operations. Later user-authorized storage maintenance is recorded above; V18
launch/shutdown did not occur and no live V18 execution is independently confirmed.
After return, the separate importer audits transfer/raw outputs, all550 original
grid cells and250 CPU cached-decoder replays within300s, retaining failures.
Review all ten sheets before deciding any next learned path or broader training.
Plan and audit limits:
[CCTV_DGP_STRUCTURE_V18_PLAN.md](<C:/xampp/htdocs/YEAR 4/Testing/CCTV_DGP_STRUCTURE_V18_PLAN.md>).
Useful held-out/native restoration, covering-family scope, DGP-led local app
selection/override, independent final review and full app inline Playwright
remain incomplete. Goal stays active.

## Completed R2/V17 milestone — negative results preserved

The downloaded R2 archive is fully imported, audited and visually reviewed.
All 3,128 updates are verified, but both trained snapshots fail the unchanged
appearance-preservation guards. No model is selected or adopted.

**V17 follow-up complete:** a separate 12-case training-only local inference
control tests unchanged initial/epoch8 codes with original encoder connections,
fidelity w1 and statistics omitted. CPU inference 139.99 seconds; no training.
Independent saved-output audit 2.70 seconds passes 24 raw/PNG compositions,
metrics and 72 original grid cells. All three sheets are reviewed. Clear previews
improve versus w0, but degraded PSNR/SSIM regress and visible fragments persist.
All eight degraded SSIM values worsen versus own w0. This is not a universal
repair, fresh-image verification, native result or final human review.
Report: `CCTV_DGP_FIDELITY_SPOTCHECK_V17_RESULTS.md`.
Evidence: `outputs/cctv_dgp_fidelity_spotcheck_v17/`.
Original R2 failures, sources, protocol, checkpoints and splits are preserved.

The requirement carried forward from this negative review was an explicit
DGP/input-structure-preserving output path, initial parity and a finite VM
capacity pilot before broader training or adoption. The V18 local preparation
above addresses that requirement; actual learned benefit remains unverified.
Do not repeat CE-only training or default w1 processing unchanged. User-run
transfer/VM commands remain the execution preference; no assistant cloud
operation or local training.
Useful native outputs, covering-family scope, DGP-led app integration, independent
final review and the full app Playwright flow remain incomplete. Goal stays active.

## Completed R2 return milestone

**V16 r2 return audit complete:**839,574,170 bytes, SHA256
3555eb65b37bcf4cbc24ff36a31f5a80a6dab02051fe3e2c9e7dd2cad026ed65,
sidecar and pasted/downloaded receipt match. Safe import59.08s preserves the VM
failure,25 manifest-bound assets and3,128 traces/31,280 exposures. Separate full
audit73.05s (bounded recovery73.50s) passes1,710 PNGs/cosines,300 raw previews,
150 training-code probes,100 fresh parity cases,781 teacher arrays,4,425 cache
bindings and600 original grid cells. Zero local neural/backward/optimizer calls;
original failure/source/protocol/checkpoints stay unchanged.

**Scientific result — no adoption:** epoch8 paired degraded PSNR17.402dB versus
retained DGP16.027, but SSIM0.5788 versus0.6193 and fixed ArcFace0.19280 versus
0.33011. Clear PSNR24.023 versus31.132; clear ArcFace0.56713 versus0.95701.
Both trained snapshots fail unchanged preservation guards; all eight degraded
source/profile groups have lower embedding similarity than DGP. All ten original
256-cell sheets were reviewed: changed eyes/mouths/glasses, invented facial-hair/
frame texture and fragments remain, including training previews. Token fitting
improves but does not establish useful restoration. Sources are photographic
provenance groups, not ethnicity/native CCTV/Zamboanga evidence. Native24 and
reserved32 remain unused by R2; no checkpoint/app promotion or Goal completion.

**Follow-up:** the bounded V17 input-feature connection control above is now
closed as a negative degraded-restoration result. Preserve all failed gates and
the learned component. Do not add epochs or repeat this failed recipe unchanged.
R2 report: `CCTV_DGP_BROADER_CODES_V16_R2_RESULTS.md`.
Metrics/pins: `outputs/cctv_dgp_broader_codes_v16_r2_verified_summary.json`.
Review: `outputs/cctv_dgp_broader_codes_review_v16_r2.json`.
Full audit: `outputs/cctv_dgp_broader_codes_v16_r2_audit_recovery_1/local_full_audit.json`.

### Superseded R2 manual reports and audit preparation

**Latest user-reported VM evidence:** V16 r2 reached epoch8, saved50 training
previews plus520 validation outputs and reported3,128 completed updates in
1,055.125922286 seconds (17m35s). The subsequent arithmetic auditor raised
`Timing stop receipt differs` at line143. Independent returned-file verification
has not occurred. The reported CE5.0193 / accuracy0.097 at update3100 describe
observed-token training loss/accuracy, not face-recognition accuracy or usefulness.

**Export readiness reported:** the user pasted `failure_export.json` with
`complete:true`,839,574,170 bytes (800.68MiB), archive SHA256
3555eb65b37bcf4cbc24ff36a31f5a80a6dab02051fe3e2c9e7dd2cad026ed65.
This confirms reported failure-export readiness, not successful model/audit review.
The archive, sidecar and locally aliased `failure_export_v16_r2.json` are not yet
present locally. Preserve the pasted expectation in
`outputs/cctv_dgp_v16_r2_manual_export_receipt_20261005.json`; match it against the
downloaded receipt/hash/size before import and corrected full audit.

**Demonstrated checker defect:** the frozen R2 protocol/helper/trainer use30
source/role cache timing references, but the later auditor tuple still requires20.
Local execution of that actual audit loop reproduces the rejection for valid R2
receipts. A separate checksum-bound audit-only recovery changes just the two
timing count/cap tuples to read the unchanged protocol design. Seven focused
regressions pass in0.124s; zero neural/backward/optimizer calls. The frozen R2
bundle, protocol, archive, trained outputs and original audit failure remain
unchanged. Do not retrain, relax the900s/1,200s/240s caps or regenerate the package.

**Next:** user collects the R2 failure archive, sidecar and export receipt using
`CCTV_DGP_BROADER_CODES_V16_R2_VM.md`; download the receipt as
`failure_export_v16_r2.json` to preserve the original V16 export. First preserve
and audit the completed trace through the existing R2 importer. Then run
`scripts/recover_cctv_dgp_broader_codes_v16_r2_audit.py` in a separate fresh
recovery directory to audit all completed outputs within240s, retaining the
original failed checker/log. Instructions and limits:
`CCTV_DGP_BROADER_CODES_V16_R2_AUDIT_RECOVERY.md`.
No R2 returned archive, full output audit, visual review, useful upgrade or
application adoption is claimed. No assistant cloud execution occurred.

**V16 failure closed — zero-update timing rejection.** The9,069,418-byte failure
archive, checksum and export receipt are verified; core import rechecked all23
frozen execution members and116 parent/6,195 data/2,804 baseline assets. Additional
independent audit verified exact recovery executable provenance, original path
error/corrected preflight,20 train-only teacher arrays and0 backward/optimizer
updates. Supervisor duration39.2795s. The9,695,145-byte partial head is preserved,
SHA256f351693258c11fe3f1c629aa009288cadd1e26231fc49134f0bdfa95a1942d2e.
No learned output, upgrade, resume, selection or application adoption.
Failure report: `CCTV_DGP_BROADER_CODES_V16_FAILURE_AUDIT.md`.
Receipts: `outputs\cctv_dgp_broader_codes_failure_return_v16\local_failure_import.json`
and `local_cache_failure_audit.json` (additional audit0.072s; zero neural/training calls).

**Separate V16 r2 prepared:** model startup and four first-reference warmups are
counted once. The cache samples10 train/5 validation references per source and
uses18 train/8 validation steady-reference timings to estimate remaining work.
All885 references/4,425 cases are still cached once; teacher labels remain train
only. Learning design, head source,3,128-update optimizer schedule, datasets/splits,
previews, quality guards and900s cache/1,200s fit/240s audit/2,400s supervisor caps
are unchanged. A new Path-typed bootstrap records visible preflight errors. New
root `~/forensic-dgp/cctv_dgp_broader_codes_vm_v16_r2/`; tmux `dgp_broader_codes_v16_r2`.
Do not overwrite or resume the original V16 directory or its partial cache/head.

Eight meaningful regressions pass in0.385s with zero model/backward/optimizer calls.
Preparation11.61s reverified116 parent,6,195 data and2,804 baseline assets. The
1,098,502-byte new archive has27 independently rehashed members and an LF sidecar;
bootstrap copy, original archive/failure, split, schedule and head preservation
are independently checked. Protocol4f64cf2204458f8fadcab1824fc4bc293c65803e123fdebb304f7b5bd3a502a0.
Archive6f78d84f0ba044897c41d629dc572f953cb62a9f26dacd9719d95f1ad4528453.
Verified transfer files are under `outputs\`; exact Windows SDK Shell upload,
normal-SSH launcher-created tmux and success/failure download commands are in
`CCTV_DGP_BROADER_CODES_V16_R2_VM.md`. New result receipts are downloaded with
`_v16_r2` filenames to preserve older `failure_export.json` and other receipts.
Local importer: `scripts\import_cctv_dgp_broader_codes_v16_r2.py`.

**Preparation history:** R2 was verified locally before the manual run reported
above. After collection/full corrected audit, inspect all ten fixed original
256-cell grids and unchanged structure/appearance guards. Report paired synthetic
metrics separately from native unpaired evidence. No useful output is claimed yet.
The restoration Goal, DGP-led app integration, covering families, native-development
outputs, final independent review and full-flow Playwright remain incomplete.
The superseded V16 manual-report/preparation history below is preserved.
<!-- V16 r2 current status end -->

## Earlier V16 manual reports

**Latest user-reported VM evidence:** recovery r1 confirmed the original path
error, passed source/data/CUDA-availability preflight and opened detached tmux
`dgp_broader_codes_v16`. Its retained evidence directory is
`~/forensic-dgp/cctv_dgp_broader_codes_v16_preflight_recovery/20261004T162619Z-8198c938/`.
The trainer then failed at line153 with `Cache projection exceeds900s`; the
supervisor reported child exit1. The frozen source places this gate after20
references and before CUDA backward preflight or optimizer updates. The last
printed progress line was1/885 references,5/4425 cases; that is not the failure's
processed-reference count. Actual counters and timing await the returned archive.
No success, useful reconstruction, checkpoint adoption or Goal completion.

The user supplied `cache_timing.json`:20 references,22.75441105899995s elapsed,
1,874.5294464701833s projected versus900s cap. Independent local arithmetic from
the pinned protocol reproduces the estimate exactly: elapsed×81.0625+30 seconds.
This is a31.24-minute projection from a22.75-second sample; the total elapsed
timer includes model startup, and startup/steady-state costs were not recorded
separately. The gate correctly rejected its computed estimate. Returned-file
integrity, counters and actual steady cache rate remain unaudited. Supplement:
`outputs\cctv_dgp_v16_cache_timing_report_20261004T162619Z.json` (zero neural/training calls).

**Next action:** collect `cctv-dgp-broader-codes-v16-failure.tar.gz`, its checksum
and `failure_export.json`; independently audit failure counters, launch recovery
provenance and `outputs/broader_codes_v16/cache_timing.json`. No code/limit changes
or rerun before that audit. Invalid assumption: a startup-inclusive20-reference
measurement reliably estimates the full cache under the900s cap. Source review
shows model loading/state hashing precede the timer sample and are extrapolated
with the per-reference cost; whether this explains the rejection still needs
recorded timings. Preserve the original recipe, partial checkpoint/cache and
failure. Do not raise the cap or repeat the failed recipe unchanged. Human
terminal report: `outputs\cctv_dgp_v16_manual_cache_failure_report_20261004T162619Z.json`.
This is user-reported evidence, not an independent VM/return audit. No assistant
cloud execution occurred. The local preparation history below is retained.

**V16 local preparation complete; restoration Goal incomplete.** A reset
code-only head, streaming float32 cache, finite L4 trainer/supervisor, safe
success/failure importer and independent arithmetic auditor are prepared.
Fourteen boundary checks passed in2.48 seconds with one artificial own-head
inference, zero pretrained forwards, zero backward calls and zero optimizer
updates. Python3.10 syntax is checked. No real-model restoration, CUDA preflight,
VM training, checkpoint selection or application adoption occurred.

Preparation took19.57 seconds and verified116 unchanged parent assets,
6,195 existing target/support/input assets and2,804 audited V15 baseline assets.
The1,088,480-byte execution archive has23 independently verified members; its
bootstrap copy and LF sidecar are verified. Budget:781 training references /
3,905 cases; separate104-reference/520-case development validation; eight epochs,
3,128 updates/31,280 exposures. Teacher labels are restricted to training roles.
Snapshots0/4/8, unchanged appearance guards, early timing/epoch4 fitting stops,
cache900s/fit1,200s/overall2,400s caps and20GiB VRAM limit are frozen. No fitted
V14/V15 initialization, statistics loss, native24/reserved32 access or automatic
best.pth. These photographic proxies do not establish CCTV/Zamboanga performance.
The declared frozen pretrained prior remains an experimental component under
the later architecture decision; a DGP-preserving output path still needs review.

**Latest transport decision:** the user selected “Prepare transfer files and
pasteable VM commands only.” This supersedes earlier automatic gcloud start/
SSH/collection/shutdown instructions below. No cloud query or launch was made;
TERMINATED is the last historical receipt, not a new current-state check.
Transport clarification,5 October2026: the user requests **Windows Google Cloud
SDK Shell (CMD)** with `cd /d` and separate gcloud SCP commands for each file.
The runbook now contains exact upload and success/failure download commands,
with explicit project/zone and VM home paths. The unchanged verified launcher
installs and opens detached tmux itself; an outer tmux would fail its idle guard.
No assistant transfer or VM launch occurred; frozen package hashes remain unchanged.
Manual execution report, 5 October 2026: the user invoked the V16 bootstrap inside
tmux and received `Competing GPU/tmux task; do not launch` at the initial idle
check. This attempt stopped before extraction or training launch. The runbook now
documents closing the failed-launch shell, checking remaining tmux/CUDA processes,
and using the unchanged launcher from normal VM SSH. Do not detach and leave the
outer session alive, terminate other jobs, or alter the frozen recipe. No VM
training receipt or return archive has been received.
Second manual report: normal SSH passed the idle/archive/extraction checks, but
the bootstrap child returned1 at launcher line59 before `supervisor_launch.json`
or tmux creation. Its stderr was hidden by subprocess capture. Local reproduction
confirms the caller passes strings to a Path-only `verify()` function; the actual
VM child stderr has not yet been received. A separate12,320-byte standalone
`outputs\recover_cctv_dgp_broader_codes_v16_preflight_r1.py` and LF checksum are
prepared. SHA256489e85d001b3cca74fdeca3c9fbd50edc2e5bf1fa7aa30c2b45db36bce99bb17.
Ten meaningful regressions pass in0.624s with zero neural/backward/optimizer calls;
the original23 bundle members, original archive and bootstrap hashes are unchanged.
Recovery verifies that specific original error, supplies Path objects, retains
error/provenance outside the frozen bundle, refuses run evidence and launches the
unchanged finite supervisor only on explicit manual `--launch`. No source, recipe,
budget, split, checkpoint or failure was replaced. At preparation, recovery was
not yet verified on the VM; the later manual reports are recorded above.
No assistant cloud execution occurred. Transfer receipt:
`outputs\cctv_dgp_v16_preflight_recovery_r1_transfer_audit.json`. Recovery source/hash,
original error and corrected preflight are embedded in exported launch provenance;
independent return review must bind them to this receipt as well as auditing the
unchanged training outputs. Updated exact upload/launch/evidence-download commands
are in the runbook. Do not rerun the original launcher or delete the extracted root.
Exact commands, three upload files, success/failure returns and stop procedure:
`C:\xampp\htdocs\YEAR 4\Testing\CCTV_DGP_BROADER_CODES_V16_VM.md` ↔ intended
`~/forensic-dgp/CCTV_DGP_BROADER_CODES_V16_VM.md` after separate document transfer.
Local frozen bundle `outputs\cctv_dgp_broader_codes_vm_v16\` ↔ intended VM
`~/forensic-dgp/cctv_dgp_broader_codes_vm_v16/`; extraction and launch are now
user-reported, with independent returned-artifact review pending.

Protocol SHA256 `4331c28c97a659846eab76ee0dee9150811a00da3c6139d8d24b7905173a6681`.
Archive SHA256 `a40f68ccdeb7e7cf2e1420bef8c984a7faad6120461073da4362bcf2fb8ab65e`.
Receipts: `outputs\cctv_dgp_broader_codes_v16_tests.json`,
`outputs\cctv_dgp_broader_codes_v16_preparation.json`,
`outputs\cctv_dgp_broader_codes_v16_transfer_audit.json`.

**Next:** receive the manually executed VM return, independently audit its
serialized outputs or completed failed trace prefix, and inspect all ten fixed
original256-cell grids before proposing an output path. Main-app DGP integration,
useful native-development/covering-family outputs, independent final review and
full-flow bundled Playwright remain pending. No readiness claim or Goal completion.
Historical V15 and earlier evidence below remains preserved.
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

V12 r2 was trained, independently audited and visually reviewed; its conditioner
is not a useful upgrade. V13 above is the latest diagnosis. Historical recipes
and gate failures below remain preserved.

V9 completed **7,820 updates / twenty epochs** on the L4 in **22.61 minutes**;
training, VM audit and export took **24.60 minutes**. The return archive and
independent local audit passed. All five original-cell grids were inspected.
No trained snapshot passed the unchanged selection safeguards: `best.pth`
retains the V2 starting baseline, selected epoch0. Epoch20 degraded PSNR improves
16.0267→21.1986dB and SSIM0.61930→0.65281, but fixed ArcFace similarity falls
0.33011→0.30431 and regresses in all eight degraded source/profile groups.
Clear preservation improves; degraded facial detail and color artifacts remain.
The ten-row previews repeat two faces across five profiles; usefulness and
independent final review remain unestablished. No trained V9 native forwards,
reserved32 use, app promotion or git/document sync occurred.

The **113.84-second local audit** rebuilt2,600 PNG metrics,50 raw previews,
520 input metrics and3,120 embedding cosines; checked7,820 traces/78,200
exposures, loss-weight arithmetic and four changed checkpoint states. It did
not replay CUDA gradients or recognizer forwards. VM source/control fingerprints
and interim/final preview equality are independently bound. After verified idle
GPU/tmux checks, the assistant restored the prior stopped state; Cloud confirms
**TERMINATED**, stop timestamp `2026-10-04T05:06:53.104-07:00`.

The user answered the architecture question: **“if that makes things better
then continue”**. A separately declared pretrained face-generating prior with
our trained DGP conditioning is now within scope, conditional on better reviewed
output. First prepare and run an inference-only feasibility comparison; do not
repeat failed pixel recipes, relabel CodeFormer as our trained DGP or relax
historical guards. All actual training remains VM-only, even for short pilots.
The full Goal is active and incomplete; the main app still uses Palette/CodeFormer.

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

Windows report: `C:\xampp\htdocs\YEAR 4\Testing\CCTV_DGP_MIXED_V9_RESULTS.md`
↔ intended `~/forensic-dgp/CCTV_DGP_MIXED_V9_RESULTS.md` after document sync.
Research: `C:\xampp\htdocs\YEAR 4\Testing\CCTV_DGP_ARCHITECTURE_REVIEW.md`
↔ intended `~/forensic-dgp/CCTV_DGP_ARCHITECTURE_REVIEW.md` after sync.
Return: `C:\xampp\htdocs\YEAR 4\Testing\outputs\cctv-dgp-mixed-v9-results.tar.gz`
↔ `~/forensic-dgp/cctv_dgp_mixed_vm_v9_r2/cctv-dgp-mixed-v9-results.tar.gz`.
Return bytes319,681,838; SHA256
`e12a68b42e36aa1b98d1eedd6aba303a2b625a8acb5a4e53815248eb5dc02a8c`.
Audited extraction: `C:\xampp\htdocs\YEAR 4\Testing\outputs\cctv_dgp_mixed_return_v9\`;
its `outputs/cctv_dgp_mixed_v9/` corresponds to the executed VM output directory.
Receipts include `local_independent_audit.json`, `paired_diagnosis_v9.json`,
`vm_source_bindings.json` and `assistant_preview_review_v9.json` beneath that
local extraction. Local VM stop receipt is in Windows `outputs/`.
The corrected inference loader remains separately versioned, with14 tests and
four artificial-signal CPU forwards proving stable state/matching output.

## Previous execution snapshot — 4 October 2026: V9 running

All actual training uses the existing L4 VM, including short pilots. Direct
Google Cloud start/stop/SSH/SCP access is configured and verified; the obsolete
Goal-tool sentence about absent SSH does not govern current execution. V8's
two-image fitting result is retained as a diagnostic, not a deployable upgrade.

The Asian source-only review is complete: 451 training candidates, 29 exact-target
pages and 49 original-resolution checks; 390 accepted, 28 covering exclusions,
33 quality/pose exclusions. Initial observations and original-image corrections
are preserved. A separate integrity audit checked every decision/hash/role.
Combined training cohort: 391 reviewed HQ FFHQ plus 390 Asian replay references.
Validation remains the same 53 HQ FFHQ and 51 Asian references, 520 cases. All
Asian source images have minimum edges below 256; replay does not create HQ truth.
Direct source hashes are disjoint; historical subject/exposure overlap is unknown.

V9 preparation independently rebuilt all 3,905 camera inputs and checked 885
target pixel hashes, 520 unchanged validation byte copies, 7,820 balanced batches
and 78,200 exposures. Ten meaningful boundary tests passed and a real local
training call was rejected before model/output/backward/update operations. All
6,232 package members independently match pinned hashes. The initial assets-only
failure from legacy HQ thumbnail pixel metadata is preserved; `r2` separately
pins actual canonical pixels without changing historic records or targets.

Frozen recipe: V2 starting state, twenty epochs, batch 10, all five profiles in
both sources every batch, fresh Adam `2e-5/1e-4`, fixed training-only-calibrated
pixel-loss weights, frozen normalization. Evaluate baseline/epochs 2/5/10/20 with
unchanged strict per-source/profile PNG PSNR/SSIM/ArcFace guards. No trained
candidate passing means `best.pth` retains V2. Trainer cap 40 minutes; complete
training/audit/export cap 60 minutes. This is a pixel-foundation pilot with no
identity training loss, native reserved use or automatic production promotion.

Historical execution snapshot: VM start, idle/CUDA checks, upload and all 6,232 exact
asset checks passed. The assistant launched dedicated `dgp_mixed_v9`; unchanged
PyTorch `2.9.1+cu129` / torchvision `0.24.1+cu129` are verified on the L4. CUDA
preflight passed with zero optimizer updates and unchanged normalization buffers.
The update-32 timing projection is 1,492.51 seconds, within the 2,400-second cap.
Later
completion/audit receipts supersede this snapshot. No completed V9 training or
selected useful output is claimed yet. Next: monitor the bounded execution, then independent return
audit and all five original-cell preview reviews. The full Goal remains active.

A separate corrected inference loader is prepared without changing the historical
adapter or app defaults: `C:\xampp\htdocs\YEAR 4\Testing\dgp_frozen_inference_v2.py`
↔ intended `~/forensic-dgp/dgp_frozen_inference_v2.py` after source transfer.
Fourteen tests passed; four retained-Phase3 CPU forwards on artificial signals
match the old loader exactly for batches 1 and 6, with stable model state and
zero backward/update calls. These are contract checks, not quality evidence.
Current inference requirements and receipts are in `DGP_INFERENCE_READINESS.md`
beneath the Windows root; intended VM counterpart is `~/forensic-dgp/` after sync.

Runbook: `C:\xampp\htdocs\YEAR 4\Testing\CCTV_DGP_MIXED_V9.md` ↔ intended
`~/forensic-dgp/CCTV_DGP_MIXED_V9.md` after document sync; executable
`C:\xampp\htdocs\YEAR 4\Testing\outputs\cctv_dgp_mixed_vm_v9_r2\` ↔
`~/forensic-dgp/cctv_dgp_mixed_vm_v9_r2/` after verified upload/extraction.
Protocol SHA256 `6cd9ac8d5f3bf27be773979c2f628e33f5d3cf09748452eeab91a65b05b01c70`.
Package: 305,689,318 bytes; SHA256
`f8fb4196a1515c451ca1f77b1d73e81f50e6335e3280b2287de7b78f48090a99`.
Preparation/audit evidence remains in local `outputs/`; no git/document sync is
claimed. Main-app defaults and all original V1–V8 protocols/failures remain intact.

## Previous milestone — 4 October 2026: V8 fitting audited; broader mixed-source preparation next

All actual training uses the existing VM, including short pilots; the assistant
can start it through configured Google Cloud CLI access. V6/V7 returned audits
are complete and neither experiment qualified a model upgrade. Their original
protocols, failed gates, returns and preview reviews remain preserved below.

V8 completed as a separate frozen training-only diagnostic: the same two V7 blur pairs,
original V2 starting tensors and pixel-MSE optimizer, 1,000 updates, fixed
20/100/1,000 snapshots, a 600-second trainer cap and 900-second complete
execution cap. All 42 inherited byte copies, three Python 3.10 source parses,
archive members and the local-training guard passed offline checks. Its
2,070,573-byte package was uploaded; all 50 assets including three cached
original weight files passed VM hash verification. The assistant started the
previously stopped L4 and launched dedicated tmux `dgp_blur_fit_v8`; unchanged
PyTorch/torchvision CUDA versions and the bounded supervisor are recorded.
Training completed 1,000 updates in 45.95 seconds; training/audit/export took
67.87 seconds. Final two-image blur MSE fell from 0.0095371 to 0.0018928
(ratio 0.19847), passing the fixed training-fit criterion. This shows fitting
capacity on those examples, not generalization. Other eight training cases
regressed: motion MSE increased from 0.0050750 to 0.0134428 and clear MSE from
0.0009495 to 0.0180006. All four grids were reviewed at original cells;
two fitted faces are clearer, while other faces show severe texture/contrast
and color artifacts. No production checkpoint was selected or promoted.

Current status: 85,445,552-byte return downloaded and independently audited;
40 PNGs, 40 raw predictions, 50 embedding arrays, 1,000 traces, three changed
checkpoints and exact ten-image V7 baseline equality checked. Execution and
terminal/source fingerprints are bound by the separate assistant review receipt.
No local model forwards/backward/updates occurred. After checking for competing
jobs, the assistant restored the VM to its prior stopped state; Cloud reports
`TERMINATED`. Do not rerun this root, deploy the two-image overfit or forward
reserved native cases. The full DGP-first Goal remains `active`.

Runbook: `C:\xampp\htdocs\YEAR 4\Testing\CCTV_DGP_BLUR_FIT_V8.md` ↔ intended
`~/forensic-dgp/CCTV_DGP_BLUR_FIT_V8.md` after document sync; runtime
`C:\xampp\htdocs\YEAR 4\Testing\outputs\cctv_dgp_blur_fit_vm_v8\` ↔
`~/forensic-dgp/cctv_dgp_blur_fit_vm_v8/`. Protocol SHA256:
`d58cf74c41d4b87db294286fd04c04292d00730ec8aa97d86af93169c5bc1d2b`.

Return SHA256: `96a80124de9115ed95995c1e6a7c9acc075185380c06841580e244ee95e3b21d`.
Local audit: `C:\xampp\htdocs\YEAR 4\Testing\outputs\cctv_dgp_blur_fit_return_v8\local_independent_audit.json`;
VM audit: `~/forensic-dgp/cctv_dgp_blur_fit_vm_v8/outputs/cctv_dgp_blur_fit_v8/independent_audit_vm.json`.
Local assistant review and VM control receipts remain local; no document/git sync
is claimed. All V1–V8 failures, original protocols and app defaults remain intact.

**Next:** review existing Asian training sources for covering/quality suitability,
then freeze a broader mixed-source finite pilot with clear/degradation replay and
unchanged per-source/profile validation safeguards. A 6.53-second zero-model
coverage audit verified 1,353 original Asian assets, 451 target pixel hashes and
1,684 existing camera inputs. V6 trained only 391 HQ FFHQ references; its 51
Asian cases were validation sentinels. The original split contains 451 Asian
training references eligible for replay consideration, preserving roles and the
104-case-reference validation cohort (53 HQ FFHQ / 51 Asian). No direct
train/validation source-hash collision was found; identity disjointness is not
established. All 451 Asian originals have a minimum edge below 256, and the HQ
FFHQ source review does not cover them. Review them before freezing a generator
training recipe. The next experiment is not executable, started or selected yet.
Evidence: `C:\xampp\htdocs\YEAR 4\Testing\outputs\cctv_dgp_training_coverage_v9_audit.json`
↔ intended VM evidence copy after verified preparation; no copy claimed.

## Previous milestone — 4 October 2026: V6/V7 returned and audited; next isolate blur fitting

**Training location:** all actual training now uses the existing L4 VM, including
short pilots. The user withdrew the brief small-local-training permission after
learning that the assistant can start the VM through the configured Cloud CLI.
Local preparation, inference and audits remain allowed. Direct VM start/stop,
SSH and SCP are verified; the old Goal-tool sentence about absent SSH is historical.

V6 completed 196 updates in 259.18 seconds; all four epochs failed unchanged
selection safeguards. Its return and independent local audit are complete:
2,600 PNGs, 50 raw previews, 3,120 embedding cosines and 196 update records checked.
Both `best.pth` files retain the V2 starting tensors. All five ten-row paired grids
were reviewed: clear HQ detail improves, but degraded-face softness and source/profile
regressions remain. No V6 native forwarding or promotion occurred.

The 782-case training-only arithmetic range audit took 19.77 seconds with zero
model/gradient/update operations. The small unavoidable output-range floors do not
support changing that range alone. A distinct V7 diagnostic then compared pixel-only
MSE against the retained objective on ten fixed training pairs, with matched larger
diagnostic learning rates and 100 updates per arm. The L4 completed 200 steps in
35.15 seconds; training/VM audit/export took 60.93 seconds within a 900-second
supervisor cap. Both arms failed the declared final blur/motion fit criterion.
Local audit checked all 70 PNGs, 80 embedding arrays/70 cosines, 200 traces and six changed
checkpoints. Final grids remain soft; pixel-only outputs show eye/color artifacts.
These are training-fit results, not independent generalization or a model upgrade.

All returns, terminal/source fingerprints, transport failures and audit receipts
are preserved. The 63.6-MB transport's VM pixel drift was repaired with exact original
PNGs; no V6 model/gate changed. Windows command length blocked the first V7 inline
launch before execution; an identical hash-verified file launcher succeeded.
No local actual training, environment installation, app default change or git
commit/push occurred. The 32 reserved native cases remain untouched.

The assistant verified no GPU, other Python or tmux jobs remained after collection
and restored the initially stopped VM. Google Cloud now reports `TERMINATED`.
This ends the finite VM executions, not the Goal. Prepare the next recipe locally,
then start the VM when its verified files are ready.

| Current evidence | Windows local | Linux VM |
| --- | --- | --- |
| V6 results/report | `C:\xampp\htdocs\YEAR 4\Testing\CCTV_DGP_TARGETS_RESULTS_V6.md` and `outputs\cctv_dgp_targets_return_v6\` | Runtime/result under `~/forensic-dgp/cctv_dgp_targets_vm_v6/`; report still local |
| V6 exact-byte execution proof | `C:\xampp\htdocs\YEAR 4\Testing\outputs\cctv_dgp_targets_v6_execution_evidence_v1\` | Original receipts under `~/forensic-dgp/cctv_dgp_targets_vm_v6/` |
| Output-range arithmetic evidence | `C:\xampp\htdocs\YEAR 4\Testing\outputs\cctv_dgp_output_range_v7\training_range_audit.json` | Local diagnostic; no VM copy claimed |
| V7 protocol/runtime/results | `C:\xampp\htdocs\YEAR 4\Testing\CCTV_DGP_CAPACITY_V7.md`, `outputs\cctv_dgp_capacity_vm_v7\` and `outputs\cctv_dgp_capacity_return_v7\` | Runtime/results under `~/forensic-dgp/cctv_dgp_capacity_vm_v7/`; report/review still local |
| VM control receipt | `C:\xampp\htdocs\YEAR 4\Testing\outputs\cctv_dgp_vm_control_20261004.json` | Control-plane start/stop state; no guest document copy |

V6 return SHA256: `3a12b481a4950ed551c98da6955b463f33f63ee48273c7e1e3dfe20245d13c77`.
V7 protocol SHA256: `a1d2fbb2c543c95ec1d0c7afe4670d74e7ad8009a8608daf5a2cce6209540b61`.
V7 return SHA256: `d2eecebb3bc9ffcac7e1abbd71d7455ffe5ed859b18dd9a27b497839fb0f4a9e`.
V7 local audit SHA256: `dab48537db9bc4f96341c0979f8e1c73c6852a561e67d636b90e00ba8fa18471`.

**Next:** freeze a separate isolated-blur fit diagnostic using the same two V7
training pairs/starting weights/objective/optimizer. Proposed budget: 1,000 updates,
snapshots at 20/100/1,000, trainer cap 600 seconds and complete execution cap 900
seconds; evaluate all ten training pairs for collateral effects. The 20-update point
matches each blur pair's V7 exposure. The longer endpoint helps distinguish task coupling
from insufficient exposure. This is a diagnostic proposal; it is not prepared,
running or eligible for native/default promotion yet. Design:
`C:\xampp\htdocs\YEAR 4\Testing\outputs\cctv_dgp_blur_fit_v8_design\proposed_protocol.json`
↔ intended VM copy after executable preparation. Do not repeat V7 mixed training
unchanged or choose pixel-only loss for full training from these failed results.

The full DGP-first Goal remains `active`: useful native restoration, DGP-led main
app integration, covering-family behavior, inline Playwright and independent final
review are still required. Historical milestones below are snapshots, superseded by
this current section where their execution/location/connection status differs.

## Previous milestone — 4 October 2026: V6 trained; output audit/export underway

The latest user instruction restores VM-only actual training, including small
pilots, because the assistant can start the existing VM through the configured
Google Cloud CLI. The earlier conditional local-training permission below is
superseded. Local preparation, inference and audits remain permitted; the full
DGP-first Goal stays active.

V6 standalone CUDA preflight passed in 27.40 seconds with zero updates. The matched
comparison completed 196 optimizer updates in 259.18 seconds on the L4, using
the unchanged frozen target-bandwidth recipe. All four trained epochs failed its
strict selection guards; both `best.pth` files retain the starting V2 tensors.
The VM independent output audit and archive export are running. No local return
audit, V6 native comparison, app promotion or final usefulness claim exists yet.

The smaller transport stopped on the VM before any training because camera-library
versions changed actual input pixels (local NumPy 2.5.2/OpenCV 5.0.0/Pillow 12.3.0;
VM NumPy 1.26.4/OpenCV 4.11.0/Pillow 11.3.0). Its local pass and remote failure are
preserved. Recovery V2 uploaded 1,430 original PNGs in 100,123,557 bytes and restored
all 2,709 original asset hashes. All 391 reduced-target pixels also match under the
original VM verifier. No package versions, data roles, model, loss or gate changed.

The separate portable driver invokes the unchanged verifier/trainer/result auditor
on those exact prepared bytes and binds the independent local derivation receipt.
It avoids the historical shell launcher's cross-platform camera regeneration;
the original launcher remains preserved. Training used dedicated tmux session
`dgp_training_v6`, driver PID 2050 and trainer PID 2069 (historical once terminal).
The pilot cap remains 1,200 seconds; the audit/export supervisor cap is 1,800 seconds.

| Evidence | Windows local | Linux VM |
| --- | --- | --- |
| Frozen runtime/data | `C:\xampp\htdocs\YEAR 4\Testing\outputs\cctv_dgp_targets_vm_v6\` | `~/forensic-dgp/cctv_dgp_targets_vm_v6/` |
| Camera drift/recovery evidence | `C:\xampp\htdocs\YEAR 4\Testing\outputs\cctv_dgp_targets_v6_transport_failure_v1\` and `cctv_dgp_targets_v6_recovery_v2\` | `transport_cache_inventory_failure_v1.json`, `control_pixel_portability_v1.json`, `transport_failures_v1/`, `transport_recovery_v2.json` under V6 root |
| Portable supervisor source | `C:\xampp\htdocs\YEAR 4\Testing\scripts\run_cctv_dgp_targets_v6_portable_v1.py` | `~/forensic-dgp/cctv_dgp_targets_vm_v6/scripts/run_cctv_dgp_targets_v6_portable_v1.py` |
| Execution/preflight evidence | Local copy after return | `~/forensic-dgp/cctv_dgp_targets_vm_v6/portable_execution_v1/` and `outputs/cctv_dgp_targets_v6_preflight/` |
| Results/checkpoints | Intended `C:\xampp\htdocs\YEAR 4\Testing\outputs\cctv_dgp_targets_return_v6\outputs\cctv_dgp_targets_v6\` | `~/forensic-dgp/cctv_dgp_targets_vm_v6/outputs/cctv_dgp_targets_v6/` |

Protocol SHA256: `0fe7d036bf8567bf0860790df553afd627ac50bd5096cfda29cc7d09d71d320c`.
Recovery archive SHA256: `2fd8719652ee3b6dbdf0be1f33c5a9c40100597f648eb1dbbbc89f7433f59c00`.
Portable driver SHA256: `7e8f08473f417c6e5254d9e05a5c884225e4821cd121ad8af1ca0bc34fc6dc4b`.

**Next:** collect the verified export, audit it once locally and inspect all five
paired preview grids/profile regressions before defining another VM experiment.
The 32 reserved native cases remain untouched. Restore the initially stopped VM
after needed collection if no competing work exists. Updated documents are local;
no git commit/push or VM document sync is claimed.

## Previous milestone — 4 October 2026: V6 smaller transport verified locally; VM preflight underway

The previous Goal turn made progress: conditional local-training permission and
hardware limits were recorded, and independent reconstruction of all 1,302 inputs,
391 reduced targets and 444 canonical HQ sources passed. The full Goal remains
active; no model has been promoted and the 32 reserved native crops remain unused.

The L4 instance was initially `TERMINATED` and has now been started for the
authorized bounded V6 comparison. Read-only SSH verified an idle GPU, 21.59 GiB
free disk and unchanged torch `2.9.1+cu129`/torchvision `0.24.1+cu129`. Restarting
assigned external IP `34.171.216.245`. Its offered SSH key matches six previously
trusted instance keys; CLI operations explicitly pin
`SHA256:pGOZAAcM7szdr2S1Ty+8UORXEPPAhxMDw7jDfzZ4I0E`.
The guest host-key attribute is unavailable, and no control-plane host-key match
is claimed. No private keys/credentials were copied into project artifacts.

The initial 376.8-MB upload averaged about 300 KB/s with 19 minutes remaining.
It was explicitly cancelled; its session is terminal, and the remote partial
archive is preserved. Windows pscp's `~/` destination was corrected to the
absolute `/home/janusdominic0/` directory. A separate lossless transport reduces
the upload to 63,567,794 bytes (83% smaller), preserving the frozen V6 protocol
and every asset. It ships 458 assets, reuses 821 exact cached assets and
regenerates 1,039 camera inputs and 391 reduced targets. One exact cached camera
module is bootstrapped before materialization imports it. The local independent
check reproduced all 2,709 assets and a byte-identical recipe in 30.00 seconds
with zero model operations. The smaller upload has completed; VM checksum,
materialization and bounded zero-update CUDA preflight are now underway. No
actual optimizer updates have yet been reported.

Transport SHA256: `16fbe1bcbbe2a54839a858221e1efa5f6b58b3a3f0409ee38f9e043d46c4183a`.
Manifest SHA256: `e7f12655f74fb94f5b43243acc0fd761f5326dfc519cfcabf01ad49bdda5c0d0`.
Materializer SHA256: `b7050bcffa14ebfe94b9e0ca038c5cc32a0a484fc39f6b3bbe864c7f70834b09`.
The original full V6 package, hashes, sources and all historical V1–V5 evidence
remain intact. This is a transport optimization, not a different experiment.

| Artifact | Windows local | Linux VM |
| --- | --- | --- |
| Frozen V6 recipe/runbook | `C:\xampp\htdocs\YEAR 4\Testing\CCTV_DGP_TARGETS_V6.md` | Runtime/data under `~/forensic-dgp/cctv_dgp_targets_vm_v6/`; runbook not yet transferred |
| Smaller transport runbook | `C:\xampp\htdocs\YEAR 4\Testing\CCTV_DGP_V6_TRANSPORT.md` | Intended document `~/forensic-dgp/CCTV_DGP_V6_TRANSPORT.md` after transfer |
| Uploaded transport/checksum | `C:\xampp\htdocs\YEAR 4\Testing\outputs\cctv-dgp-targets-v6-transport-v1.tar.gz` and `.sha256` | `~/cctv-dgp-targets-v6-transport-v1.tar.gz` and `.sha256` |
| Local materialization proof | `C:\xampp\htdocs\YEAR 4\Testing\outputs\cctv_dgp_targets_v6_transport_v1\local_materialization_audit.json` | Local receipt; remote counterpart will be `~/forensic-dgp/cctv_dgp_targets_vm_v6/transport_materialization_v1.json` |
| Standalone CUDA preflight | Local receipt after transfer | `~/forensic-dgp/cctv_dgp_targets_vm_v6/outputs/cctv_dgp_targets_v6_preflight/` when complete |

**Next:** finish VM materialization/CUDA preflight, run the finite matched pilot
only if those checks pass, independently audit the return and inspect paired
output grids before any native/app decision. Training completion alone remains
insufficient for the full Goal.

## Previous milestone — 4 October 2026: conditional local training allowed; V6 package prepared

The latest user instruction permits local training for small pilots estimated at
3–5 minutes on the L4. Larger runs around 30 minutes should use the VM; tell the
user when a workload needs it. Choose by measured compute and memory, not archive
size. This supersedes the earlier blanket VM-only rule for new experiments;
preserve immutable historical protocols and the full active DGP-first Goal.

Verified local hardware: RTX 3050 Laptop GPU, 4,096 MiB VRAM. The current
`C:\xampp\htdocs\YEAR 4\Testing\venv\` has PyTorch `2.13.0+cpu`, CUDA unavailable.
The historical eight-image V5 CUDA preflight allocated 4,398,557,696 bytes (4.10
GiB), exceeding the local GPU's capacity. The frozen eight-image V6 DGP recipe
therefore remains an L4 experiment; no local training, VM start or transfer was
performed. A read-only VM check found the instance `TERMINATED`.

The additive V6 package compares HQ256 reconstruction targets with targets
reduced HQ256→128→256 through PIL LANCZOS. It holds canonical HQ input images,
identity targets, case order, starting V2 tensors, Adam, frozen normalization and
common validation fixed. The reduced arm is a target-bandwidth control, not an
exact reproduction of the old official-thumbnail recipe. Training uses 391
source-reviewed FFHQ references; validation uses 53 HQ FFHQ plus 51 unchanged
Asian sentinel references, each with five fixed camera profiles. Both arms run
two epochs, 98 updates each (196 total), under a 1,200-second pilot cap. The strict
source/profile selection guards remain unchanged; native/visual and independent
final review still govern usefulness, and no production promotion is permitted.

Preparation completed in 76.10 seconds with 2,709 pinned assets and zero local
model forwards/backward calls/updates. The archive is 376,810,544 bytes.
Protocol SHA256: `0fe7d036bf8567bf0860790df553afd627ac50bd5096cfda29cc7d09d71d320c`.
Archive SHA256: `d829012e9a6c2c109335914da207c1fca6c205c7409b756345d41057b064e4c9`.
Five offline checks passed for matched factors, target bandwidth, safe paths,
local-training guard and rejection of aggregate gains hiding profile/identity
regression. Five sources parse as Python 3.10. Independent whole-package data
audit passed in 33.67 seconds, rebuilding all 1,302 input PNGs, 391 reduced
targets and 444 canonical sources with zero model operations. CUDA preflight/full
pilot remain pending. There is no measured V6 L4 runtime or model-quality result yet.

| Artifact | Windows local | Intended VM destination after transfer |
| --- | --- | --- |
| V6 package | `C:\xampp\htdocs\YEAR 4\Testing\outputs\cctv_dgp_targets_vm_v6\` | `~/forensic-dgp/cctv_dgp_targets_vm_v6/` |
| Transfer archive/checksum | `C:\xampp\htdocs\YEAR 4\Testing\outputs\cctv-dgp-targets-v6.tar.gz` and `.sha256` | `~/cctv-dgp-targets-v6.tar.gz` and `.sha256` |
| Preparation receipt | `C:\xampp\htdocs\YEAR 4\Testing\outputs\cctv_dgp_targets_v6_preparation.json` | Local evidence; not transferred |
| Passed independent local data receipt | `C:\xampp\htdocs\YEAR 4\Testing\outputs\cctv_dgp_targets_v6_local_preparation_audit.json` | Local evidence; not transferred |

**Next:** CUDA preflight on the L4; independent preparation audit has passed.
The new permission allows practical small future local experiments; it does not
justify silently changing this already frozen comparison's batch size or device.
All prior V1–V5 failures/returns, source decisions and 32 reserved native crops
remain preserved. DGP app integration and useful output remain unfinished.

## Previous milestone — 4 October 2026: all 510 HQ sources reviewed; 444 candidates frozen

The full DGP-first Goal remains `active`. The assistant acquired 16 genuine
1024×1024 FFHQ counterparts after matching current thumbnail pixels to official
metadata. The acquisition took 67.69 seconds, downloading 290,338,584 bytes
including metadata. Independent audit rebuilt all 16 targets and checked 48
preview cells. Four source-comparison pages and two additional 1024 images were
reviewed: 14 clean-restoration candidates, two covering exclusions. These are
reference images, not restored model outputs. No model forwards/backward calls
or optimizer updates were used; actual training remains VM-only.

All 510 existing FFHQ references now bind to their official thumbnail pixels
(451 original training-role, 59 validation-role), with 510 genuine 256×256 targets.
The four-worker acquisition used a 1 GiB/30-minute cap and stopped after a read
timeout with 507 source files complete. Its failure evidence is preserved.
Independent cache verification identified only three missing files; a bounded
recovery downloaded 4,296,660 bytes (4.1 MiB) and completed the catalog in
42.18 seconds. The full independent audit finished in 61.84 seconds, checking
all 510 thumbnail/source/target/preview bindings and 32 catalog pages. No exact
image or source-photo URL overlap exists between roles; full identity overlap
with historical training remains unestablished. All acquisition/audit process
handles are terminal. All 32 source pages and 38 ambiguous 1024 sources have now
been reviewed under the frozen rubric. A separate ledger accepts 391 training-role
and 53 validation-role references, excludes 60 coverings and excludes six unsuitable
references (severe original blur or an artificial doll). All original images/roles
and the initial 16 sample decisions remain intact. This is assistant development
source review; independent final human/output review remains pending. No new
training is ready or running.

| Artifact | Windows local | Intended VM copy after transfer |
| --- | --- | --- |
| Target-quality status | `C:\xampp\htdocs\YEAR 4\Testing\CCTV_DGP_HQ_TARGETS_STATUS.md` | `~/forensic-dgp/CCTV_DGP_HQ_TARGETS_STATUS.md` |
| Audited 16-source sample | `C:\xampp\htdocs\YEAR 4\Testing\outputs\cctv_dgp_hq_counterparts_v1\` | `~/forensic-dgp/outputs/cctv_dgp_hq_counterparts_v1/` |
| Frozen source decisions/candidate manifest/integrity receipt | `C:\xampp\htdocs\YEAR 4\Testing\outputs\cctv_dgp_hq_source_review_v3\` | `~/forensic-dgp/outputs/cctv_dgp_hq_source_review_v3/` if transferred |
| Full source acquisition plan/review rubric | `C:\xampp\htdocs\YEAR 4\Testing\outputs\cctv_dgp_hq_cohort_plan_v2\` | `~/forensic-dgp/outputs/cctv_dgp_hq_cohort_plan_v2/` |
| Audited source-only catalog | `C:\xampp\htdocs\YEAR 4\Testing\outputs\cctv_dgp_hq_cohort_v2\` | `~/forensic-dgp/outputs/cctv_dgp_hq_cohort_v2/` |

Six integrity checks passed; six new acquisition/audit/preparation sources parse
as Python 3.10. Sample receipt SHA256:
`3d890f16554511b153fadadb65a0fe24918398815bdc18425fd0a1c4e23621b9`.
Full plan SHA256: `ad65e35c70104b72a38dfe3cdac1f1027029445313f9d5530dbb18af7cde9501`.
Source-only rubric SHA256:
`1e0aa441917280168a87f73509e1846b15ada15ee4d8b0e62ec35bf7cc6bad3e`.
Full catalog receipt SHA256:
`d4989390d2c69bcd3c3abb8169aa1c02b4ad6a50aed3c5575b544aa93e96c0c0`.
Frozen review integrity receipt SHA256:
`d3fb788eb775d2c178179e283b7a353653473ed088b2a19141ec02e6467078d4`.
The review freeze checks all 510 actual source/target SHA pairs, 32 page bindings,
38 original-resolution decisions and original sample agreement in 1.14 seconds.
It does not independently assess the assistant's visual judgments. Raw acquisition
review-pending snapshots remain unchanged; the separate frozen ledger is current.
Roles are preserved even where the official FFHQ category differs. Covered
references and all old results remain intact. The Asian-source data is unchanged;
the HQ FFHQ counterparts do not establish Asian representation or local CCTV
performance. Neither image dimensions nor the source-preview sharpness prove
that a newly trained DGP will improve.

**Next:** prepare a controlled finite VM comparison using approved genuine targets
and re-evaluate the starting model on common new references before fitting.
No new training recipe is ready or running. V5 fallback checkpoints
remain the V2 starting tensors; no model was promoted. DGP-led app integration,
useful native restoration and final review remain required. New code/docs are
uncommitted and the image outputs are local; no VM sync has been claimed.

## Previous milestone — 4 October 2026: V5 audited and rejected; target-resolution audit complete

The full DGP-first Goal remains `active`. V5 finished on the existing L4 with
452 updates in 348.84 seconds (5 minutes 49 seconds). Its archive/checksum are
local, and the user's full independent audit, including recognizer verification,
passed. A new experiment needs one local audit; do not rerun extraction/audit on
this unchanged return. The assistant can handle future audits when new files
arrive. The existing receipt and immutable V1–V5 evidence remain intact.

**No V5 trained epoch qualified.** Identity weight 0.4 epoch 2 improves average
similarity but regresses blur/motion MSE and SSIM. PCGrad epoch 2 increases
aggregate degraded PSNR 16.2167→16.9750 dB, yet both source blur/motion groups
regress and Asian low-light similarity falls 0.00432121. Both `best.pth` files
select epoch 0 and retain the V2 identity epoch 2 tensors; no production model
was promoted. All 550 V5 baseline PNGs match V2 identity epoch 2 byte for byte.
All five fixed 10-row previews were reviewed: severe blur/compound outputs remain
soft and indistinct. No V5 native forwards were run; 32 reserved crops remain
untouched, and independent final review remains pending.

The local target audit verifies all 1,012 pinned native/target assets and headers.
896/902 training references are below 256 in both native dimensions; every FFHQ
reference is 128×128 enlarged to a 256×256 target. Every reference has at least
one native dimension below 256. This limits available detail but is not proof
of the failure's cause. Target suitability and exhaustive source identity overlap
remain incomplete.

| Artifact | Windows local | Existing L4 VM |
| --- | --- | --- |
| Returned archive/checksum | `C:\xampp\htdocs\YEAR 4\Testing\outputs\cctv-dgp-conflict-v5-results.tar.gz` and `.sha256` | `~/forensic-dgp/cctv_dgp_vm_bundle/cctv-dgp-conflict-v5-results.tar.gz` and `.sha256` |
| Canonical V5 outputs | `C:\xampp\htdocs\YEAR 4\Testing\outputs\cctv_dgp_conflict_return_v5\outputs\cctv_dgp_conflict_v5\` | `~/forensic-dgp/cctv_dgp_vm_bundle/outputs/cctv_dgp_conflict_v5/` |
| Local independent receipt | `C:\xampp\htdocs\YEAR 4\Testing\outputs\cctv_dgp_conflict_return_v5\local_independent_audit.json` | Local receipt; original VM `outputs/cctv_dgp_conflict_v5/independent_audit.json` remains unchanged |
| Result/next-action report | `C:\xampp\htdocs\YEAR 4\Testing\CCTV_DGP_CONFLICT_RESULTS.md` | Intended `~/forensic-dgp/CCTV_DGP_CONFLICT_RESULTS.md` after document transfer |
| Separate analysis/visual/resolution ledgers | `C:\xampp\htdocs\YEAR 4\Testing\outputs\cctv_dgp_conflict_v5_review\` | Optional `~/forensic-dgp/outputs/cctv_dgp_conflict_v5_review/` after transfer |

Archive SHA256: `eec1a062ea3c3203e5bba4a7a12f0a8fdae6de1a1e37942c705ed318aa6cbcb2`.
Results SHA256: `80d8a9b1d5cec0d6fbe325229664a5c5251ba2739791922ddc728d4466d8cd5b`.
Local receipt SHA256: `93017a6afb593730afc999de6d8e18603a5f0e441a3faa1fc49517405c45f437`.
Post-audit analysis used zero model forwards/backward calls/updates; the full
audit was not repeated or overwritten. Detailed checks and ledger hashes are
in the result report. Direct CLI/SSH capability is verified with TLS enabled,
project/instance `forensic-dgp-thesis`, zone `us-central1-a`; the earlier
no-connection limitation and below live-process snapshot are historical.

**Next:** verify a bounded sample of genuine higher-resolution source counterparts
and clean target suitability while preserving historical split roles, provenance
and results. Freeze any new target protocol and re-evaluate its starting baseline
before a changed finite VM pilot. No new pilot is ready or running. Useful native
output, DGP-led local integration, insufficient-input behavior, covering-family
flow, Playwright checks and independent final review remain incomplete. These
document changes have not been committed/pushed; `git pull` alone does not transfer
them or result archives.

## Previous milestone — 4 October 2026: V4 audited; V5 live-process snapshot; direct SSH verified

Continuation capability check found the installed, configured Google Cloud CLI.
Read-only API/SSH access now works from this harness using a command-scoped
Windows-trusted CA bundle with TLS verification enabled. The earlier statement
that no SSH connection is available is historical. Verified project/instance:
`forensic-dgp-thesis`, zone `us-central1-a`, VM `RUNNING`.

The user's V5 process is already running in attached `dgp_training`; no second
training process was launched. At 07:16 UTC on 4 October, trainer PID `1332` and
launcher PID `1304` were live, both zero-update gradient-policy preflights had
passed, and arm 1 was validating epoch 2. GPU allocation observed earlier:
4,782 MiB. A bounded read-only SSH observer follows this existing run and export.
V5 results/checksum have not yet arrived locally. Direct retrieval and independent
audit can proceed after verified export; manual transfer commands remain usable.

The full DGP-first Goal remains `active`. The V4 archive/checksum and user-created
local receipt are now present. A second independent audit reproduced that receipt
exactly, with no local model forwards, backward calls or optimizer updates.
V4 used 40 training references, 50 CUDA autograd traversals, 21.75 seconds and
zero updates; student and teachers stayed unchanged. This is a completed
diagnostic, not a newly trained or useful model.

Reconstruction and identity parameter gradients oppose one another for blur and
low light in both sources. All four groups also have an opposing total identity
direction under the original weights. The observed-image gradients do not expose
this conflict. Forty cases at one state do not establish causation or predict
Adam behavior. Detailed evidence: local
`C:\xampp\htdocs\YEAR 4\Testing\CCTV_DGP_OBJECTIVE_RESULTS.md`
↔ intended VM `~/forensic-dgp/CCTV_DGP_OBJECTIVE_RESULTS.md` after document transfer.

A changed V5 comparison is prepared: identity weight 0.4 with ordinary gradient
summing versus identity 0.1 with symmetric two-objective PCGrad. Both retain the
audited V2 starting state, postactivation VGG, original 902/110 reference split,
degradation/order/seed, Adam settings and strict source/profile/identity guards.
Budget: 226 updates/arm, 452 total, two epochs/arm, 30-minute training-process cap.
Expected process time is 8–15 minutes, estimated rather than measured for V5.
All actual training/autograd remains guarded to the existing Linux L4 VM. The
32 reserved native crops remain untouched; no default checkpoint was promoted.

| Artifact | Windows local | Existing L4 VM |
| --- | --- | --- |
| New additive upload | `C:\xampp\htdocs\YEAR 4\Testing\outputs\cctv-dgp-conflict-v5.tar.gz` and `.sha256` | Upload separately to `/home/janusdominic0/`, extract into `~/forensic-dgp/` |
| Exact run/return commands | `C:\xampp\htdocs\YEAR 4\Testing\CCTV_DGP_CONFLICT_V5.md` | `~/forensic-dgp/cctv_dgp_vm_bundle/CCTV_DGP_CONFLICT_V5.md` after extraction |
| Preparation evidence | `C:\xampp\htdocs\YEAR 4\Testing\outputs\cctv_dgp_conflict_v5_checks\` | Local evidence; no separate VM copy until transferred |
| Expected V5 return | `C:\xampp\htdocs\YEAR 4\Testing\outputs\cctv-dgp-conflict-v5-results.tar.gz` and `.sha256` | `~/forensic-dgp/cctv_dgp_vm_bundle/cctv-dgp-conflict-v5-results.tar.gz` and `.sha256` after successful run/audit |

The archive is 731,079 bytes with 11 new files; SHA256
`1147291cac26c47d1fd1774a28b3894896e1511a8eaff516999a289aadab6388`.
Protocol SHA256: `ba3a6775ca187ab8199fa2b278685f347004af84bb3a01049ec4e92feb06181f`.
Checks cover exact archive payloads, additive preservation of V1–V4, split
integrity, six Python 3.10 sources, Bash syntax and 40 relevant tests. Both V5
CUDA preflights passed; training is live, final audit/export remains pending.
Reuse `.venv`; no CUDA reinstall. These
files have not been committed/pushed; `git pull` does not install this overlay.

Input-policy preparation also audited the 24 native development inputs: the
current 32-pixel minimum rejects five of six coarse core faces while accepting
one larger insufficient face. The new observed-only quality helper excludes
padding/removal regions and passes six checks; eight existing workflow checks
pass. It is not a structure classifier and was not wired into the app. Evidence:
`C:\xampp\htdocs\YEAR 4\Testing\CCTV_INPUT_POLICY_STATUS.md`
↔ intended VM `~/forensic-dgp/CCTV_INPUT_POLICY_STATUS.md` after transfer.

**Next:** observe the existing V5 process through terminal export, retrieve the
archive/checksum, then independently audit and review paired previews;
only an eligible candidate proceeds to native development review. Useful native
output, DGP-led integration, insufficient-input behavior, covering-family flow,
Playwright verification and independent final review remain incomplete.

## Previous milestone — 4 October 2026: V3 audited and rejected; zero-update objective diagnostic ready

The full DGP-first Goal remains `active`. The preceding transfer-command response
made no authoritative project change; continuation revalidated the actual return
and completed the review and new diagnostic preparation below. DGP-led local
integration and demonstrated useful output remain required; neither archive
arrival nor a training run completes the Goal.

The V3 return is present and independently audited at local
`C:\xampp\htdocs\YEAR 4\Testing\outputs\cctv-dgp-perceptual-v3-results.tar.gz`
and `.sha256`, from VM
`~/forensic-dgp/cctv_dgp_vm_bundle/cctv-dgp-perceptual-v3-results.tar.gz`.
Archive:347,331,047 bytes, SHA256
`f5efd077c7ef8254cbb395dd6bc4ff4b9b66f6472d3f661de9ca33dadc4a1674`.
The audit reconstructed2,750 PNG metrics,50 raw previews,3,300 embedding cosines,
452 update records and52 pinned recognizer preview forwards. The L4 run used452
updates in332.78 seconds; allfour epoch checkpoints changed293 parameter tensors,
with normalization/teachers preserved. Local return auditing used zero restoration
forwards, backward calls and optimizer updates.

**No V3 trained epoch qualified.** Aggregate degraded PSNR rose16.2167→17.3187dB
in the preactivation arm, but fixed ArcFace fell0.32686→0.31471 and source/profile
blur/motion safeguards failed. Both `best.pth` files selectepoch0 and retain the
same starting V2 tensors. All550 V3 baseline PNGs exactly match V2 identityepoch2.
The assistant reviewed allfive fixed10-row grids: severe blur/compound outputs
remain soft and indistinct; brightening does not demonstrate useful recovery.
No V3 native forwards were warranted. The32 reserved native crops remain untouched.

Detailed report: `C:\xampp\htdocs\YEAR 4\Testing\CCTV_DGP_PERCEPTUAL_RESULTS.md`
↔ intended `~/forensic-dgp/CCTV_DGP_PERCEPTUAL_RESULTS.md` after document transfer.
The immutable returned VM receipt is preserved. Separate local receipt:
`outputs\cctv_dgp_perceptual_return_v3\local_independent_audit.json`, SHA256
`2010cb66ea1571972367775bb1dbb1100e8d4c595e253ecb2fe9c1416ce461b1`.
Local analysis/visual ledgers are in `outputs\cctv_dgp_perceptual_v3_review\`
(optional VM copies under `~/forensic-dgp/outputs/cctv_dgp_perceptual_v3_review/`).

The next prepared experiment is a VM-only **zero-update objective diagnostic**,
not another identical training run. It starts at retained V2 identityepoch2;
selects40 training-only cases (four per source/profile); measures weighted pixel,
color, postactivation VGG, Sobel and ArcFace gradients in parameter/image spaces;
and stops after10 groups,50 autograd traversals or600 seconds. No optimizer,
checkpoint export, validation/native/reserved use or production promotion.
Gradient-direction evidence is diagnostic and does not predict Adam outcomes.

| Artifact | Windows local | Existing L4 VM |
| --- | --- | --- |
| Additive upload | `C:\xampp\htdocs\YEAR 4\Testing\outputs\cctv-dgp-objective-v4.tar.gz` and `.sha256` | Upload to `/home/janusdominic0/`, extract into `~/forensic-dgp/` |
| Runbook | `C:\xampp\htdocs\YEAR 4\Testing\CCTV_DGP_OBJECTIVE_DIAGNOSTIC_V4.md` | `~/forensic-dgp/cctv_dgp_vm_bundle/CCTV_DGP_OBJECTIVE_DIAGNOSTIC_V4.md` after extraction |
| Package audit | `C:\xampp\htdocs\YEAR 4\Testing\outputs\cctv_dgp_objective_v4_checks\package_audit.json` | Local preparation evidence; no copy exists until transferred |
| Expected return | `C:\xampp\htdocs\YEAR 4\Testing\outputs\cctv-dgp-objective-v4-results.tar.gz` and `.sha256` | `~/forensic-dgp/cctv_dgp_vm_bundle/cctv-dgp-objective-v4-results.tar.gz` and `.sha256` after successful diagnostic |

The new package is20,122 bytes with10 files; SHA256
`1d86057a8318a703a34dfad99675204c971960f52a3d79aa526fac58c863aa0e`.
Protocol SHA256: `0c7feb50736617fbbbeaa8e79e9987c61c79e05ceaf68613603e989795e75b5c`.
Independent checks verify exact inventory/payloads, additive preservation of all
original/V2/V3 assets,40 training-only references, five Python3.10 sources, Bash
syntax and40 relevant tests. These include resealed cohort/gradient/source
tampering, changed teachers, accidental updates, and local CUDA-probe rejection.
The real `--verify` CLI confirms the diagnostic recipe; actual VM execution is
still pending. Reuse existing `.venv`, data and teachers; do not reinstall CUDA.
The overlay has not been committed/pushed, so `git pull` does not install it.

**Next:** upload the two V4 files, run the new diagnostic in `dgp_training`, and
return its small archive/checksum. Independently audit the report before choosing
a changed finite training recipe. Preserve all prior failures and safeguards.
Useful native restoration, DGP as the primary local restorer, insufficient-input
qualification, covering-family output, Playwright flow verification and independent
final review remain incomplete. No Zamboanga-specific performance is established.

## Previous milestone — 4 October 2026: V2 audited; native review complete; V3 VM comparison ready

A fresh `get_goal` read confirms status `active` with the full DGP-first
objective. The previously paused manual-support state below is historical.
Continue the original restoration, covering-completion, local-app and independent
review requirements; receiving a checkpoint is not completion.

Both return files are now present at
`C:\xampp\htdocs\YEAR 4\Testing\outputs\cctv-dgp-normfix-v2-results.tar.gz`
and its `.sha256`, transferred from
`~/forensic-dgp/cctv_dgp_vm_bundle/cctv-dgp-normfix-v2-results.tar.gz`.
The 348,737,530-byte archive passed the full independent corrected audit: 2,750
PNG metrics, 50 raw previews, 3,300 embedding cosines, 452 update traces, 52 pinned
recognizer forwards and four changed checkpoints with unchanged normalization
buffers/teachers. The L4 run completed 452 updates in 311.77 seconds. The no-identity
arm retains epoch0; identity epoch2 qualifies on paired synthetic guards, improving
degraded PSNR 13.23→16.22 dB and fixed ArcFace similarity 0.277→0.327. These are
paired photograph results, not native or Zamboanga-specific performance.

The existing 24-case native development run and its independent composition audit
are complete: 24 frozen DGP forwards/7.87 seconds and 120 PNG stages reconstructed.
The separate assistant visual ledger reviewed the ten-row preview and all four
gallery pages. The six input-selected core faces remain soft with color shifts;
native useful restoration is not demonstrated. Seven input-selected insufficient
cases require a clearer crop. The 32 reserved native crops remain untouched.
Detailed evidence: local `C:\xampp\htdocs\YEAR 4\Testing\CCTV_DGP_PILOT_RESULTS.md`
(intended VM `~/forensic-dgp/CCTV_DGP_PILOT_RESULTS.md` after explicit transfer).

A changed matched V3 experiment is now prepared and independently package-audited.
Both arms start at the audited identity epoch2 with ArcFace retained; they compare
continued postactivation against training-only calibrated signed preactivation VGG
features. The calibration uses 64 original training references and zero validation
or native cases. It is loss-scale preparation, not evidence of trained improvement.
The experiment retains the original 902/110 reference split, data/order/seed,
optimization and strict source/profile/clear/identity selection safeguards.
Budget: 226 updates/arm, 452 total, two epochs/arm, 90-minute stop limit.

Upload local `C:\xampp\htdocs\YEAR 4\Testing\outputs\cctv-dgp-perceptual-v3.tar.gz`
and its LF `.sha256` to VM `/home/janusdominic0/`. The 724,844-byte additive archive
contains 16 new files only and does not overwrite the parent or V2 correction.
SHA256: `925ef82d27ba1e3bfe010989c79d6e1e28572e8a91c47e83c5da7e09da8d469e`.
Recipe SHA256: `875cb8b5b58ec1f60f1060d3bafd37c13ab9e984f57c3159b1c460ddd13bd576`.
Checks: 27 relevant tests, nine files parsed for Python 3.10, Bash syntax, source
derivation, complete archive/recipe/starting-checkpoint verification and the real
`--verify` CLI. New CUDA backward checks and training remain pending on the VM.
No local training/backward calls, reserved-set use or application promotion.

Exact one-file SCP, guarded extraction, tmux/run, return download and independent
audit commands are in local `C:\xampp\htdocs\YEAR 4\Testing\CCTV_DGP_PERCEPTUAL_V3.md`,
bundled at `~/forensic-dgp/cctv_dgp_vm_bundle/CCTV_DGP_PERCEPTUAL_V3.md`. Reuse the
existing `.venv`, data and teachers; do not reinstall CUDA or rerun the historical
pilot. Local preparation is staged under `outputs/cctv_dgp_perceptual_vm_v3/`;
package receipt: `outputs/cctv_dgp_perceptual_v3_checks/package_audit.json`.

Next: run the new finite VM comparison and return
`~/forensic-dgp/cctv_dgp_vm_bundle/cctv-dgp-perceptual-v3-results.tar.gz` plus `.sha256`
to local `C:\xampp\htdocs\YEAR 4\Testing\outputs\`. Independently audit it and
review eligible epochs on the unchanged native development cohort before default
selection. DGP main-app integration, input qualification, covering-family outputs,
browser verification and independent final review remain required. The Goal stays
active. These changes have not been committed/pushed; the verified overlay carries
the new VM sources.

## Previous VM support — 4 October 2026: confirmed InstanceNorm failure; verified correction ready

The autonomous Goal is paused at the user's request. Subsequent SSH support is
for the user's manually started pilot; it does not mark the Goal complete or
resume autonomous project work. The original transfer package remains unchanged.

The user supplied logs from `~/forensic-dgp/cctv_dgp_vm_bundle/` showing that CUDA
preflight passed and the 550-case baseline validation ran. The pilot then stopped
at `Matched branch did not start at identical baseline`, with
`optimizer_updates_recorded: 0`. Preserve `outputs/cctv_dgp_pilot/`, including
`failure.json`, `preflight.json`, `execution.json`, `partial_state.pth`, and the
baseline outputs, plus `pilot.log`. These are user-reported VM records, not a
locally audited return. No newly trained checkpoint or completed result archive
has been received at `C:\xampp\htdocs\YEAR 4\Testing\outputs\`.

The subsequent user-pasted L4 / PyTorch 2.9.1+cu129 diagnostic confirms the cause:
batch eight changed zero state tensors; batch six changed ten InstanceNorm
running-statistic tensors despite `.eval()` and zero backward/optimizer calls.
The largest drift was 0.00048828125 in `smooth.1.running_var`. Native InstanceNorm
writes averaged repeated statistics back in this runtime. Both fixed validation
and training end with six-image batches. Retain strict matched-start and frozen
buffer checks; do not suppress their errors or repeat the unchanged launcher.

The versioned runtime correction passes cloned running statistics into the same
native operator for the student's five InstanceNorm layers. Checkpoint schema,
original normalization output/gradient path, ArcFace identity loss, matched arms,
data/order/losses/seed/selection and the 452-update / 90-minute budget remain.
Original protocol and model/runner/auditor sources remain unchanged. This is an
additive runtime correction, not a newly trained model or changed experiment.

Verified Windows correction package:
`C:\xampp\htdocs\YEAR 4\Testing\outputs\cctv-dgp-normfix-v2.tar.gz`
plus its ASCII/LF `.sha256`; upload both to `/home/janusdominic0/`.
It is 10,724 bytes with nine safe new-file members, without data/weight transfer
or overwriting frozen assets. Archive SHA256:
`82263d45d70e5c2d0e8277418887f5a23e03b59b288c82af5d5f80b7a46b5221`.
Correction manifest SHA256:
`a3c4cfe30164a3a161bf549af29beac122bbdfc274c74ddd49df7cae29c7bf72`.
Extract its `cctv_dgp_vm_bundle/` prefix under `~/forensic-dgp/`.
Exact upload/extraction/tmux/return/audit commands are in
`C:\xampp\htdocs\YEAR 4\Testing\CCTV_DGP_NORMFIX_V2.md`
↔ `~/forensic-dgp/cctv_dgp_vm_bundle/CCTV_DGP_NORMFIX_V2.md` after extraction.

Local verification: 18 regression/guard checks pass; six Python files parse for
Python3.10; Bash launcher syntax and GNU tar export transforms pass. Four CPU
DGP forwards compare original versus corrected output for batches eight and six:
maximum output error zero, original state hash preserved, no backward/optimizer
calls. Evidence is at
`C:\xampp\htdocs\YEAR 4\Testing\outputs\cctv_dgp_normfix_v2_checks\`
↔ `~/forensic-dgp/outputs/cctv_dgp_normfix_v2_checks/` only after optional evidence
transfer. Corrected L4 evaluation/backward compatibility remains unverified.

Next manual operation: upload/extract the correction, launch
`bash scripts/run_cctv_dgp_normfix_vm.sh --archive-zero-update-failure` inside the
existing VM bundle's activated environment/tmux. The wrapper checks the original
and correction assets, preserves only the exact known zero-update failure at
`~/forensic-dgp/cctv_dgp_vm_bundle/outputs/cctv_dgp_pilot_failed_v1/`, repeats the
original CUDA gradient preflight and verifies one additional six-image forward
has identical student state before baseline validation/training. It refuses
arbitrary/trained failures, overwrite and automatic repeat/resume.

On success return
`~/forensic-dgp/cctv_dgp_vm_bundle/cctv-dgp-normfix-v2-results.tar.gz` and `.sha256`
to `C:\xampp\htdocs\YEAR 4\Testing\outputs\`. The corrected return includes
`normfix_revision.json` and the exact runtime sources. The corrected auditor
retains all original metric/checkpoint/trace checks and verifies that correction
lineage. Preserve the received VM audit receipt; write the local recognizer audit
receipt separately at the extracted archive root. Only an audited return permits
the prepared native development comparison; keep reserved cases untouched.
No app checkpoint promotion, UI change, git commit/push or Goal resumption here.

## Previous milestone — 3 October 2026: DGP adapter verified; post-pilot native comparison frozen

The revised DGP-first Goal was active at this milestone. The prior turn completed the verified
VM pilot package; this continuation completed local inference preparation with
zero training/backward calls. No `cctv-dgp-results.tar.gz` is present locally,
and no configured SSH connection can inspect/start the user's VM. The new archive
is still the unchanged419,370,023-byte package described in the previous milestone.
Use `C:\xampp\htdocs\YEAR 4\Testing\CCTV_DGP_VM.md`
↔ `~/forensic-dgp/cctv_dgp_vm_bundle/CCTV_DGP_VM.md` after extraction for execution.

A strict adapter at
`C:\xampp\htdocs\YEAR 4\Testing\dgp_face_restoration.py`
↔ `~/forensic-dgp/dgp_face_restoration.py` after source transfer supports native
RGB128 padding, bilinear256 RGB and nearest observation/removal masks. It checks
explicit checkpoint fingerprints, strict schema and finite tensors; missing or
incompatible weights stop rather than random-fill. Only frozen float32 256 input
is accepted. Raw DGP float, observation composite and floor-quantized PNG are
separate; no top-k scores, display sharpening, hidden resize or automatic promotion.
The current combined application still selects its existing CodeFormer restorer.

Seven adapter checks pass. Eight real native development forwards (two per size
bin) match the earlier audited Phase3 inputs, raw floats and raw PNGs exactly,
maximum float error0. Observation compositing matches; tensors/buffers remain
unchanged. CPU time after loading is2.33 seconds; no backward or optimizer updates.
This verifies adapter fidelity, not improved quality or complete app integration.
Evidence: `C:\xampp\htdocs\YEAR 4\Testing\outputs\dgp_restoration_adapter_v1\`
↔ `~/forensic-dgp/outputs/dgp_restoration_adapter_v1/` after report transfer.
Results SHA256: `50fb83c65776ee93479312dfe1b7b0a04a87d9a9f0938a479e93950b46d287fb`.

The frozen native candidate plan is at
`C:\xampp\htdocs\YEAR 4\Testing\outputs\cctv_dgp_native_pilot_review_v1\frozen_plan.json`
↔ `~/forensic-dgp/outputs/cctv_dgp_native_pilot_review_v1/frozen_plan.json` after
explicit report transfer. SHA256:
`475c2225a368fe035d8c02d62953e6cb32c836fdb71b7aa5a8704c6d276f3ded`.
Its runner re-audits the actual VM return before native outputs; at most one trained
best checkpoint per arm is eligible. Epoch0 fallback is not admitted as an improved
model. Four selection checks cover fallback, changed epoch/hash and empty outcomes.
The six new Python files parse for the VM's Python3.10; the native runner is local.

All24 development cases remain, with a fixed six-case coarse frontal/mild core.
Seven input-reviewed insufficient cases and profile/uncertain cases remain labeled
diagnostics; pixel size is not a rejection classifier. Reuse audited cached
Phase3/CodeFormer raw outputs with identical observed-context composition. Cap new
candidate inference at48 forwards and480 seconds after loading; no optimizer.
The ten-row preview and four full gallery pages require output review. The auditor
checks source/weight/plan lineage and raw/PNG composition; no native PSNR/SSIM or
identity-accuracy metric is invented. The32 reserved crops remain unaccessed by
this milestone's model/visual work. Candidate execution/visual review is pending.

Application gaps remain explicit: current upload minimum32 excludes some coarse
native crops; current output follows original dimensions with a50% visible blend;
uncovered low-information rejection is uncalibrated. The staged adapter establishes
the 256 contract, but a useful candidate and a versioned composition/input policy
must precede default DGP integration and full Playwright/final-review verification.
Read `C:\xampp\htdocs\YEAR 4\Testing\DGP_INFERENCE_READINESS.md`
↔ `~/forensic-dgp/DGP_INFERENCE_READINESS.md` after source/document transfer for the
post-return commands and unresolved requirements. No UI, app default, checkpoint,
training split, frozen historical protocol or transfer bundle changed here.

Next operational step: run the existing guarded L4 pilot and return
`~/forensic-dgp/cctv_dgp_vm_bundle/cctv-dgp-results.tar.gz` plus `.sha256` to
`C:\xampp\htdocs\YEAR 4\Testing\outputs\` using the documented gcloud commands.
Audit the return, run the prepared native comparison, review its outputs and choose
the next justified model/app action. Do not promote `best.pth` from its name alone.

## Previous milestone — 3 October 2026: DGP-first Goal active; matched VM package independently verified

The app Goal is active with the revised DGP-first objective recorded in
`C:\xampp\htdocs\YEAR 4\Testing\SYSTEM_WORKFLOW_AND_GOAL.md`
↔ `~/forensic-dgp/SYSTEM_WORKFLOW_AND_GOAL.md` after document transfer. The main
outcome remains useful 256×256 degraded-CCTV restoration by our trained DGP;
covering completion supports it. Independent final assessment and the full local
application workflow are still required. No Goal completion or app promotion.

Preparation completed locally with **zero backward calls or optimizer updates**.
Two exact within-training duplicates were removed in a derived manifest. Input
gate V1 is preserved: its 90% full 112-pixel context requirement rejected tight photos
whose facial feature cores were visible. A separate V2 replaces only that
disproved assumption with measured feature-core/landmark support and consistent
unsupported recognizer-context masking. Its independent audit reconstructs all
1,150 saved source/geometry records without new detector/model forwards. V1 used
1,150 input-only detector requests in 44.85 seconds; no restoration or updates.

V2 admits 917 training and 110 validation references. Its 64 input previews identify
two further training exceptions: Asian09274 has a hand over the mouth/closed eyes;
Asian02752 has a facial watermark. The separate reviewed manifest excludes those
and retains the original-quality warnings in validation. Equal-source truncation
by inherited ordering yields **902 training references (451/source)** and all
**110 validation references (59 FFHQ/51 Asian-source), 550 fixed cases**. Targets
are processed photographs, usually below 256 captured pixels. Not all references
were visually reviewed and no full historical identity overlap audit is claimed.

The self-contained Windows bundle directory is
`C:\xampp\htdocs\YEAR 4\Testing\outputs\cctv_dgp_vm_bundle_v1\`.
The archive at
`C:\xampp\htdocs\YEAR 4\Testing\outputs\cctv-dgp-vm-bundle.tar.gz`
with adjacent LF `.sha256` is **419,370,023 bytes**. Upload both to VM `~/`,
then extract under `~/forensic-dgp/`; the archive prefix becomes
`~/forensic-dgp/cctv_dgp_vm_bundle/`. It includes native selected references,
target/observation PNGs, two precomputed camera epochs, fixed validation PNGs,
retained DGP/ArcFace/extracted VGG weights, model code, guarded scripts and auditors.
No full 80,000-image re-download is needed for this bounded pilot.

Independent package verification reconstructs 1,012 reference canvases and all
2,354 prepared cases, checks 5,419 archive entries and validates the LF checksum.
Eight meaningful contract tests and both Bash syntax checks pass. A two-image CPU forward check uses one
DGP batch, frozen converted/ONNX recognizer comparisons and the declared loss;
encoder maximum absolute error is 3.219e-6, states remain unchanged and there are
zero local backward/optimizer calls. CPU forward timing after loading is 2.71 seconds.
This does **not** verify GPU backward compatibility, VRAM or useful trained output.

| Binding | SHA256 |
| --- | --- |
| Transfer archive | `3be6cf4a3a6fbcdb88c5934e8a0c0fd2b426465c0f4a863d9f893b3f32eee97e` |
| Frozen executable protocol | `b652914335e33375f8108ba427e4b5be99fd86e811f455b307ab72ae7cf152f6` |
| Independent archive/pixel audit | `cfff7f45cd38713348b12732293341b6dfe6658bbb61f255fe87e7ac5bb321dc` |
| Local forward-only check | `a91f14a598d78796bcd882e051d7492cdcacbbb05179dd116c9462920551317e` |
| Separate input review | `69b72552932275de8f55393f90c4f94ad778bbd0f42d29999910982dca4fba92` |

Build/audit/forward receipts are in
`C:\xampp\htdocs\YEAR 4\Testing\outputs\cctv_dgp_pilot_protocol_v1\`
↔ `~/forensic-dgp/outputs/cctv_dgp_pilot_protocol_v1/` after explicit report transfer.
Input gate V1/V2 and their review/audit records remain in local
`outputs\cctv_dgp_reference_gate_v1\` and `outputs\cctv_dgp_reference_gate_v2\`;
selected JSON evidence is bundled under VM `cctv_dgp_vm_bundle/evidence/outputs/`.

**Next is the VM pilot**, using
`C:\xampp\htdocs\YEAR 4\Testing\CCTV_DGP_VM.md`
↔ `~/forensic-dgp/cctv_dgp_vm_bundle/CCTV_DGP_VM.md` after extraction. The matched
arms start from retained Phase 3 and differ only in identity coefficient 0/0.1.
Both use the same camera inputs, clear anchors, order, Adam and common
Charbonnier/color/VGG/Sobel loss; no FAN/FFT, AMP or EMA. Normalization buffers
stay frozen. Each arm has 2 epochs/226 updates, batch 8; **452 updates total and
90 minutes including validation/preflight**, with timing projection at update 32.
The runner refuses non-VM/local CPU execution, wrong host/GPU, changed assets,
existing outputs, nonfinite gradients and automatic resume. Its initial batch 8
CUDA backward preflight has **zero optimizer updates** and must pass first.
Actual CUDA execution is pending; no VM training or Git push occurred here.

After return, independently audit every saved PNG/cohort/update/selection and
raw checkpoints, then review all source/profile/clear results and compare qualified
candidates on the 24-case native development gallery. The 32 reserved native crops
remain untouched by output review or models. `best.pth` may retain epoch 0 if any
guard fails; no automatic production promotion. New PNG/observed-identity metrics
are not directly interchangeable with old float/unmasked reports. Preserve the
Phase 3 checkpoint; original Phase 5 full GPU return remains absent. Broader
training, primary DGP app integration and independent final review remain pending.

## Previous milestone — 3 October 2026: updated Goal active; paired regression audited; DGP pilot pool inspected

A fresh app Goal read verifies `active` with the complete revised DGP-first
objective in `C:\xampp\htdocs\YEAR 4\Testing\SYSTEM_WORKFLOW_AND_GOAL.md`
↔ `~/forensic-dgp/SYSTEM_WORKFLOW_AND_GOAL.md` after transfer. The user applied
the objective and resumed the Goal; the preceding paused-control limitation is
historical. Completion still requires a verified DGP-led local workflow and
independent final assessment, including the agreed supporting covering families.

The three input-selected canonical alignment successes now have an audited
common-frame comparison at
`C:\xampp\htdocs\YEAR 4\Testing\outputs\cctv_alignment_restoration_comparison_v1\`
↔ `~/forensic-dgp/outputs/cctv_alignment_restoration_comparison_v1/` after transfer.
There are six CPU restoration forwards, 17.06 seconds after loading, 18 exact
PNG reconstructions and 12 float stages, with zero updates and unchanged model
states. Alignment does not establish a reliable gain across the three cases;
control resampling and uncertain landmarks limit interpretation. No alignment
default, application change or checkpoint is selected.

The **full original Phase 4 split is available locally** at
`C:\xampp\htdocs\YEAR 4\Testing\outputs\downloaded_phase4\outputs\phase4_with_progress\split.json`.
The original VM run path is `~/forensic-dgp/outputs/phase4_with_progress/split.json`.
Its SHA256 is `5c71bc358a351d50e3c0a7d76abe749cb4412aca80d2c7eb4de5fb33dbd29071`:
76,000 training and 4,000 validation paths (3,500 FFHQ thumbnails, 500 Asian-source
validation images). Normalize path separators when checking the inherited split.
It is historical development evidence, not a new untouched identity test.

The frozen paired regression is complete and independently audited at
`C:\xampp\htdocs\YEAR 4\Testing\outputs\cctv_paired_regression_v1\`
↔ `~/forensic-dgp/outputs/cctv_paired_regression_v1/` after transfer. Eight
deterministically selected validation references produce 40 fixed clear/blur,
low-light, motion and compound cases. The input-only review separates six
uncovered frontal/mild references from an already covered sunglasses reference
and a strong profile, retaining all cases in diagnostics. Do not call either
exception a clean uncovered reference. Both frozen models receive identical
aspect-preserving inputs. Reference-only recognizer geometry fixes eligibility
across arms; scores remain development proxies. There are 40 DGP and 40 CodeFormer
CPU forwards, 229.91 seconds after loading, 112 frozen recognition forwards,
zero optimizer updates and unchanged model states. The independent audit
reconstructs 136 PNGs, checks 80 raw float stages and reconstructs 105 saved
embedding cosines. All 40 cases and the ten-row grid have been reviewed.

Over the 24 degraded cases from the six input-selected in-scope references,
mean fixed-reference cosine is 0.3271 for resizing, 0.2268 for DGP and 0.1945 for
CodeFormer. DGP smooths clear controls and darkens low-light inputs. CodeFormer
adds sharp but unverified detail, changes expressions and introduces a hand-like
artifact absent from one reference. Neither is promoted. Full source/profile,
clear-control and exception results are recorded separately in
`C:\xampp\htdocs\YEAR 4\Testing\CCTV_PAIRED_RESTORATION_RESULTS.md`
↔ `~/forensic-dgp/CCTV_PAIRED_RESTORATION_RESULTS.md` after transfer.

A deterministic candidate pool at
`C:\xampp\htdocs\YEAR 4\Testing\outputs\cctv_dgp_pilot_pool_v1\`
↔ `~/forensic-dgp/outputs/cctv_dgp_pilot_pool_v1/` after transfer contains 1,024
inherited training and 128 inherited validation references, balanced by source
before qualification. All 1,152 files fully decode and match recorded bytes;
an independent audit confirms inherited memberships and exact-byte groups.
No selected train/validation byte overlap is found; two exact duplicate groups
exist within Asian-source training. Most targets are below 256 captured pixels.
The 32 predetermined previews include dim/noisy/grayscale originals, coverings,
closed eyes and strong profiles. Do not call every candidate a clean target or
claim all 1,152 were visually reviewed. This is not a full 80,000-file/identity
overlap audit, and no new high-resolution targets have been downloaded.

| Current binding | SHA256 |
| --- | --- |
| Paired results | `afe601078153d133f68870c9ef120029c270099ae9def034efa40357dcd44a30` |
| Independent paired audit | `8a05eb9ec6559e80f9ced6930c4fb8c69d2bfd185d9464c4f698af71d1371964` |
| Paired visual review | `8203c3e53983851f88db35e03b4eb36198b704b514baf28f133cb119d8aa7a7b` |
| Candidate-pool data audit | `422d0646b2685664a42e3eca593acd10ac2bc2a58505381699f3fdb5b10dc79f` |
| Independent pool audit | `0c9bb2573e18fa2687515505fe8ea43b29e30d51a501417cf7f658f2a421433f` |

Next: deduplicate in a derived manifest and freeze input-only quality/pose/landmark
coverage gates, then implement/package the finite VM-only DGP diagnostic in
`C:\xampp\htdocs\YEAR 4\Testing\CCTV_DGP_NEXT_PILOT.md`
↔ `~/forensic-dgp/CCTV_DGP_NEXT_PILOT.md` after transfer. Intended matched arms
compare camera/clear-anchor training without versus with identity supervision,
at most two epochs and 256 updates per arm, 90 minutes total. Final executable
loss/eligibility/optimization settings and transfer files remain pending: **not
training-ready yet**. Keep Phase 3 retained; original Phase 5 full GPU return is
still absent. No local training, Git push or VM run occurred in this milestone.

## Previous milestone — 3 October 2026: native CCTV outputs audited; DGP-first Goal specification updated

The updated copy-paste Goal objective is at the top of
`C:\xampp\htdocs\YEAR 4\Testing\SYSTEM_WORKFLOW_AND_GOAL.md`
↔ `~/forensic-dgp/SYSTEM_WORKFLOW_AND_GOAL.md` after transfer. It prioritizes
our trained DGP for 256×256 degraded CCTV restoration, visible structure over
maximum sharpness, a clearer-crop request for insufficient information, and
public native CCTV evaluation before choosing training. Completion remains
supporting functionality. The actual app Goal reads `paused`; tools cannot edit
its objective or resume it. The file is updated; the app Goal control is unchanged.

The official QMUL-SurvFace source is now sampled into 24 development crops and
32 reserved crops across four native-size ranges. Selected labeled identities
are disjoint and selected bytes are distinct; all 56 copies match the archive.
The reserved set receives only header/hash checks and remains unviewed/unprocessed
by models. This does not establish overlap absence from historical model training.
The first partial selector is retained: V2 corrects a `.jpg`/PNG-content assumption
without changing ranges, quotas or native bytes. Individual country/ethnicity
labels and real Zamboanga CCTV samples remain unavailable.

The first model comparison is complete at
`C:\xampp\htdocs\YEAR 4\Testing\outputs\cctv_native_comparison_v1\`
↔ `~/forensic-dgp/outputs/cctv_native_comparison_v1/` after explicit transfer.
Both models receive the same native-padded bilinear 256 input. There are 24 DGP
and 24 CodeFormer CPU forwards, taking 141.08 seconds after loading, with zero
optimizer updates and unchanged model states. Raw DGP, legacy display-only DGP
and raw CodeFormer outputs are separate. An independent audit reconstructs all
96 PNGs and verifies 48 float stages and 56 native copies. All 24 development
outputs and the ten-row grid were reviewed; independent final review is pending.

DGP often smooths weak features; display processing amplifies patterned texture.
CodeFormer can produce coherent sharper faces but also unverified detail and
artifacts. Neither establishes reliable improvement across the gallery. No
checkpoint, application change or new training recipe is selected. Preserve the
Phase 3 checkpoint; the Phase 5 full GPU return is still absent.

Two processing diagnostics narrow the next work. Existing signal filters change
only one of 24 native inputs. On six input-selected coarse frontal/mild cases,
native-space legacy alignment yields zero canonical results and silently catches
five detector-size errors. Padded-256 alignment avoids those errors and yields
three canonical results; framing/tilt still needs restoration comparison. V2
retains the error findings after the first probe's reporting guard stopped it.
All original partial evidence is preserved. An independent processing audit
reconstructs 96 signal PNGs and 12 alignment PNGs without model forwards.

| Current binding | SHA256 |
| --- | --- |
| Native subset | `c063985b258b808efaa897c88593cbdbcee2848d686d3d056219f8a8e555a98e` |
| Model-comparison results | `3be73af7b0cdae85f54adc8173fd8f48b42511a604fcd60244e3b1d2565a4c34` |
| Independent comparison audit | `c7801b90a817f20403abc3bd99743e8d0f8e782003b1e1d938bfff05be38671a` |
| Independent processing audit | `0bd4b61f6fa2e50a04f20d769f31efca9b050fe5e9400198b548a3c8e7306a32` |

Full findings, Windows/VM paths, source limits and the next sequence are in
`C:\xampp\htdocs\YEAR 4\Testing\CCTV_NATIVE_RESTORATION_RESULTS.md`
↔ `~/forensic-dgp/CCTV_NATIVE_RESTORATION_RESULTS.md` after transfer.
Next at that milestone: compare model outputs with versus without the feasible alignment, freeze
paired low-light/native-size regressions, then choose a bounded VM-only DGP pilot.
Do not infer clean-reference PSNR/SSIM from native unpaired CCTV or promote a
model from sharpness alone. No local training, Git push or VM run occurred here.

## Previous milestone — 3 October 2026: DGP-led CCTV scope confirmed; public native archive audited

The user reconfirmed the main aim: a useful face restoration model for degraded
CCTV imagery in Zamboanga City. Keep the existing Goal and covering-removal
workflow as supporting functionality. Do not let the recent detector/inpainting
experiments replace the primary restoration objective.

The title in `C:\xampp\htdocs\YEAR 4\Testing\docs\THESIS MANUSCRIPT.docx`
is "Forensic Deep Generative Prior Face Reconstruction for Degraded CCTV Video in
Zamboanga City." Its intended Linux counterpart is
`~/forensic-dgp/docs/THESIS MANUSCRIPT.docx` after transfer. The user says the
manuscript is outdated: decisions since the 19 September conversation take
precedence. The manuscript was read, not edited. Its older video/multiple-output
and police-user descriptions are historical requirements pending reconciliation.

The first CCTV question round confirms one already cropped face as input, no real
Zamboanga CCTV samples yet, and improvement of our trained DGP as the main
restoration contribution; pretrained restoration models are comparison baselines.
The user wants all CCTV degradations addressed, with face restoration first,
without ranking individual degradation types or specifying input face sizes.
Public/synthetic images cannot be presented as verified Zamboanga CCTV evidence.

The subsequent rounds confirm visible facial structure over maximum sharpness,
accepting some softness; request a clearer crop when extreme blur/tiny faces leave
insufficient facial information. The initial DGP output/comparison stays 256×256.
Obtain a public real-CCTV benchmark before choosing new training, prioritizing
Asian capture sources where available and reporting broader sources separately.
The assistant reviews development outputs; independent reviewers assess final
thesis results. A separate pretrained completion component is permitted while
DGP remains the main restorer. No individual ethnicity or Zamboanga-performance
claim is implied by a public benchmark's capture location.

The CCTV workflow questions are resolved sufficiently for benchmark acquisition.
Existing covering decisions remain. The announced 36-case context-margin
comparison has not been prepared or executed and is supporting work after the
primary CCTV restoration comparison. The completed V1 comparison and packaged
grayscale detector pilot remain intact; any already-produced return may be
audited independently, but it does not replace the DGP benchmark prerequisite.

Current local inspection verifies the Phase 3 restoration checkpoint at
`C:\xampp\htdocs\YEAR 4\Testing\checkpoints\dgp_zamboanga_final.pth` ↔
`~/forensic-dgp/checkpoints/dgp_zamboanga_final.pth`, SHA256
`b6376f56c4161ef1f26a8f9efb0bb309b3014f4ecdeaca5399a47c1c4607586c`.
Only `outputs/phase5_smoke/` is present among local Phase 5 artifacts; no complete
Phase 5 GPU return is verified. The DGP `/reconstruct` endpoint remains in
`app.py`, while the combined `face_workflow_palette.py` route uses pretrained
CodeFormer completion and visible restoration. These are distinct models:
pretrained-route improvements do not establish an improved trained DGP output.

See `SYSTEM_WORKFLOW_AND_GOAL.md` and `PRACTICAL_OUTPUT_SCOPE.md` in the Windows
root (intended counterparts under `~/forensic-dgp/` after document transfer).
The official QMUL-SurvFace V1 archive is now acquired at
`C:\xampp\htdocs\YEAR 4\Testing\dataset\cctv_survface_raw\QMUL-SurvFace-v1.zip`
↔ intended `~/forensic-dgp/dataset/cctv_survface_raw/QMUL-SurvFace-v1.zip`
after an explicit transfer. It is 408,164,983 bytes; acquisition/inventory takes
48.26 seconds. Observed SHA256
`2fbb0876bc4761217c6de5576905e2524b8ca50ad7905720a5b4b378a0ff8e13`;
the adjacent checksum uses LF. No publisher checksum is claimed.

An independent structure audit verifies exact gallery/probe metadata membership
and 10,051 byte-identical verification-image copies. There are 463,341 canonical
image filenames, 166 fewer than the published total; the discrepancy is retained.
The labeled train/test person IDs are disjoint. There are 28,476 unlabeled
distractor images, so a complete identity census/overlap claim is not justified.
Actual metadata has 3,000 gallery IDs despite an older bundled README claim of
5,319. Original files and counts stay unchanged; no restoration evaluation is
reported from these checks.

Receipt: `C:\xampp\htdocs\YEAR 4\Testing\outputs\cctv_survface_structure_audit_v1\verification.json`
↔ intended `~/forensic-dgp/outputs/cctv_survface_structure_audit_v1/verification.json`,
SHA256 `cf9124983a815776a787eeae577bbbbd173892be2f69dc33c354626641863222`.
Provenance, access terms, Asian-source limitations, discrepancies and next steps
are in `C:\xampp\htdocs\YEAR 4\Testing\CCTV_BENCHMARK_STATUS.md`
↔ intended `~/forensic-dgp/CCTV_BENCHMARK_STATUS.md` after document transfer.
The native CCTV release is not clean paired restoration ground truth; per-image
country/ethnicity attribution and prior model-training identity overlap are unknown.
No source image was decoded/extracted and no model inference ran in acquisition.

Next: inspect a labeled native development contact sheet, freeze a separate
restoration evaluation subset, and compare the retained DGP with declared
baselines before choosing VM training. This is not a claim of Zamboanga validation.
No new training, checkpoint selection or application change occurred here.

## Previous milestone — 3 October 2026: completion context comparison audited; VM detector return pending

The user explicitly confirmed that a coherent rough estimate is acceptable when
the covering is substantially removed and visible appearance remains coherent.
This is acceptance of the product criterion, not individual output approval or a
revision of historical failures. The clarification is recorded in
`C:\xampp\htdocs\YEAR 4\Testing\PRACTICAL_OUTPUT_SCOPE.md` (intended
`~/forensic-dgp/PRACTICAL_OUTPUT_SCOPE.md` after a separate document transfer).

A new local inference comparison is complete at
`C:\xampp\htdocs\YEAR 4\Testing\outputs\completion_context_review_v1\`
↔ intended `~/forensic-dgp/outputs/completion_context_review_v1/` after an explicit
evidence transfer. It tests 2- and 6-pixel input-conditioning margins on four fixed
degraded assisted hand/hair/scarf/flower cases. The reviewed output area stays
unchanged. Nine requests (eight alternatives plus one zero-radius control) take
86.34 CPU seconds after loading: nine completion/nine restoration forwards,
zero detector forwards and zero optimizer updates. Both model states are unchanged;
the control reproduces the cached assisted scarf/glove PNG exactly.

Five support/invalid-input counterexample tests pass. The independent audit checks
37 artifacts and reconstructs all nine output PNGs from captured floats. All four
complete preview rows/eight new alternatives are inspected. Six pixels reduces
scarf wool contamination, but a thin line remains near the upper removal boundary.
Hand/hair/flower estimates are similar to the useful assisted baseline. Small
visible-MAE changes do not establish hidden identity. There is no general-default
or application change, and automatic hair/near-hidden failures remain unresolved.

Report: `C:\xampp\htdocs\YEAR 4\Testing\COMPLETION_CONTEXT_RESULTS.md`
↔ intended `~/forensic-dgp/COMPLETION_CONTEXT_RESULTS.md` after document transfer.
Protocol SHA256 `211945a7187680e12885c6709dea52564593db9a31fb4a69d3a91f152c86c7c3`;
results `e69b41bab828cc99a4744e3fbe61fa217283650a2a81ddd057e9ab98c7683e17`;
independent verification `2cd14db824b1d9302b169c5d3c11540f805227a7ed3173739768092846ed575a`.
The new adapter/runner/auditor are separate from frozen historical inference code
and the already packaged VM experiment. Current frozen source bindings remain intact.

**Next:** obtain `gray-covering-results.tar.gz` and `.sha256` from
`~/forensic-dgp/gray_covering_vm_bundle/` into
`C:\xampp\htdocs\YEAR 4\Testing\outputs\`, use the prepared independent return
auditor, then run the unchanged practical detector/face-output comparisons.
The archive remains absent locally; no VM job/SSH session is confirmed live here.
Use `GRAY_COVERING_VM.md` and `GRAY_COVERING_RETURN_REVIEW.md` under the Windows
root (intended counterparts under `~/forensic-dgp/` after the stated transfers).
The 50.3 MB sent bundle, its checksum, checkpoints, original 425-case failures and
application route remain unchanged. Training remains VM-only. Goal active.

## Previous milestone — 3 October 2026: grayscale VM pilot and independent return auditor prepared

Work continues in `C:\xampp\htdocs\YEAR 4\Testing\`; training remains on
`~/forensic-dgp/`. Both camera and varied-covering returns are audited. The
subsequent 20 automatic face estimates are reconstructed and inspected, with
covering remnants, missed hair and generated artifacts recorded below.

A new zero-forward footprint diagnosis recounts144 saved proposals and72
probability arrays, reusing seven verified prior assisted outputs. All 449 source
bindings remain intact. Hair predictions stay empty: the treatment's peak crop
probability is only 0.0000713 even on the degraded example, so expanding an empty
mask cannot repair it. With the reviewed removal area the unchanged visibility
guard rejects both nearly hidden cases (85.9% facial coverage); automatic areas
cover only 44.5% native/53.9% degraded and miss rejection. All 36 reviewed-mask guard
decisions agree with the declared gallery scope. This diagnoses conditional
detector dependence on these exposed cases, not general guard correctness.

All seven cached assisted comparisons were inspected. Corrected footprints yield
an estimated eye on the hair case and reduce finger/cloth/petal remnants. Exterior
hair and objects outside the selected facial region remain. Scarf/glove texture
and stylized anatomy are still limitations. No hidden facial target or user/expert
acceptance is inferred. Diagnostic:
`outputs/varied_covering_footprint_diagnostic_v1/results.json` SHA256
`b88e696f4cc5183c4e5b44924db9279353dd6126c6dad53fed79cefe61f93469`;
visual review SHA256 `f87d360ced332819f9039ae4f0a2a0634d86d99047d96e3dec740a6279b677bb`.
The receipt's tentative sampler direction is superseded by the recipe below:
inspection confirmed V2 already cycles the five new covering families.

The next independent bundle is
`C:\xampp\htdocs\YEAR 4\Testing\outputs\gray-covering-vm-bundle.tar.gz` plus its
LF checksum, intended for `/home/janusdominic0/` before extraction into
`~/forensic-dgp/gray_covering_vm_bundle/`. It is50,299,398 bytes, SHA256
`212b91935e548ad1423f92f71014681be25abd052bad1d7bf2785e8b2b702d18`.
All 1,022 regular members pass independent inventory/schedule checks; all 1,009
original V2 members remain byte-identical. Twenty-three consumed Python files
compile, seven local contract tests pass and both generated shell scripts pass
Git Bash syntax checking. The original V2 bundle and source models are untouched.

Both `rgb133` and `gray133` start from camera91 with fresh optimizer state and
receive 8×64=512 updates each. They share the existing family sampler, cases,
fixtures, loss and learning rates. Only real-input grayscale exposure differs.
The first128 RGB steps match V2 exactly and must reproduce its previous treatment
model before continuing. Every 133-source native/degraded RGB/gray condition is
exposed in the gray branch. Budget 1,024 total updates /11,440 image forwards,
3,248 measurement masks and one separate zero-update CUDA preflight; hard cap
30 minutes after loading. Grayscale is an unproven transfer hypothesis. No new
labels, held-out training sources, generator/identity training or app selection.

Protocol SHA256 `c29eabcc23b07e2f9353116068c623b35258a56ed5ab259126b13b9c781be47a`;
independent package audit SHA256
`4a341c260c4ed39df3af562136c4da41fd2bdbbfa58bd6dbda05048b86288899`;
shell/contracts receipt SHA256
`ec5056207ea8eeda196846228b73c0d9a7750effa2f0a0d6135b110bb9ff298d`.
All receipt paths are relative to the Windows workspace; Linux counterparts are
`~/forensic-dgp/<same path>` only after an explicit evidence transfer.

**Next:** use `C:\xampp\htdocs\YEAR 4\Testing\GRAY_COVERING_VM.md`
(bundled `~/forensic-dgp/gray_covering_vm_bundle/GRAY_COVERING_VM.md`) to upload,
verify/extract and run `bash scripts/run_gray_covering_vm.sh` inside tmux.
Readiness: `C:\xampp\htdocs\YEAR 4\Testing\GRAY_COVERING_READINESS.md` ↔ intended
`~/forensic-dgp/GRAY_COVERING_READINESS.md` after a future docs transfer.
CUDA preflight and actual training remain pending; zero local training updates.
Return `gray-covering-results.tar.gz` and its `.sha256` file for audit and unchanged
practical-output evaluation. The user-directed SSH command workflow continues;
no agent SSH session or new VM job is open. No git commit or push; original 425
failures and existing app/models remain unchanged. Goal active.

The independent return auditor is now implemented at
`C:\xampp\htdocs\YEAR 4\Testing\scripts\audit_gray_covering_results.py`
(intended `~/forensic-dgp/scripts/audit_gray_covering_results.py` after a separate
source transfer; not inside the frozen VM bundle). Fourteen meaningful integrity
tests pass: partial/unsafe/corrupted archives, grayscale/replay schedule changes,
old moment counters, nonfinite/frozen tensor changes, false promotion claims and
preview corruption are rejected. Both preview reconstructions match the frozen
producer on explicitly labeled local test fixtures, without model construction.
The read-only preparation check verifies the unchanged package/sources, both
schedules and all 812 measurement targets. No new VM return is present locally;
no returned gray-model or output-quality success is claimed.

Return-review commands and limits:
`C:\xampp\htdocs\YEAR 4\Testing\GRAY_COVERING_RETURN_REVIEW.md`
↔ intended `~/forensic-dgp/GRAY_COVERING_RETURN_REVIEW.md` after separate transfer.
Prepared evidence:
`C:\xampp\htdocs\YEAR 4\Testing\outputs\gray_covering_return_audit_preparation_v1\`
↔ intended `~/forensic-dgp/outputs/gray_covering_return_audit_preparation_v1/`.
Preparation receipt `verification.json` SHA256
`5b86055a1bbb74bb627061e7862d81df3dd9a96be7985d373d10a2aab6d0400b`;
it explicitly records `returned_results_audited: false`.
The future audit checks 3,267 members, 3,248 masks, 1,024 logged steps, three
checkpoint states, exact RGB128 reproduction and both ten-row grids. Actual
result verification remains pending; the following step is the unchanged
practical gallery and face-output comparison before any model selection.

## Previous milestone — 3 October 2026: varied-covering return, masks and face outputs audited

The new return is present and verified:
`C:\xampp\htdocs\YEAR 4\Testing\outputs\varied-covering-results.tar.gz`
↔ `~/forensic-dgp/varied_covering_vm_bundle/varied-covering-results.tar.gz`.
It is 108,064,323 bytes, SHA256
`63014d50269bb0b50da0c5b5d1f54cadee1bd180634f0a51ac8b502f0ec9488c`.
All 1,653 members, 1,638 masks and 256 step records pass independent verification;
both checkpoints have 184 finite tensors and 92 unchanged reference-head/BN states.
Each branch logs128 fresh steps/1,122 cumulative updates. Recorded L4 processing
is43.45 seconds excluding loading/export. Optimizer tensors remain on the VM.
Actual training occurred on the VM; zero local training updates.

All five new native/degraded family IoUs improve and all final clear controls are
empty. Old-cohort/reflection retention checks fail; four treatment hair views
remain empty. All ten training-preview rows were inspected. No checkpoint is
selected, no `best.pth` is created and original425 gates are neither evaluated nor
waived. Independent return audit:
`outputs/varied_covering_results_validation_v1/verification.json` SHA256
`ebae5bd787f0fe0f19fa57d483c3ea29766c1ef73825e76560cd0f7f3654c95d`;
visual review SHA256 `7ce8b8b91819eec8603f359e3b92f1a895da3b7bd7e44715ecdb52f85e9f5816`.

The fixed36-case practical comparison is complete, with five counterexample tests
passing before inference. Both returned models each predict53 CPU images:
17 return reproductions plus36 practical inputs. Total106 forwards take10.26 CPU
seconds excluding loading; every34 reproduced mask matches the VM exactly.
Independent probability/metric/state verification covers290 artifacts and144
metric records. All six preview sheets were inspected. New candidates keep all
four clear controls empty, but the retained app-default cache falsely marks the
native uncovered control. Hand/scarf/object transfer improves; gallery hair masks
stay empty and both nearly hidden cases still miss automatic rejection.

Practical protocol `outputs/varied_covering_practical_protocol_v1/protocol.json`
SHA256 `15f6ca20827c4221f7e12630f7d3fab54d7eceae51235eb9401c6f529563382e`;
audit `outputs/varied_covering_practical_validation_v1/verification.json` SHA256
`fb76dce5dbe793d536d6058893edcb736e299069dd3664ffa09baf8b16d09e99`.
All output paths above are relative to `C:\xampp\htdocs\YEAR 4\Testing\`;
intended receiving paths are `~/forensic-dgp/<same path>` only after explicit transfer.
These are exposed development masks; the two three-quarter cases remain outside
first-version pose coverage. Approximate labels are not hidden-face targets.

A ten-case degraded face-output comparison is complete and audited separately:
`C:\xampp\htdocs\YEAR 4\Testing\outputs\varied_covering_completion_review_v1\`
↔ intended `~/forensic-dgp/outputs/varied_covering_completion_review_v1/` after transfer.
Its protocol SHA256 is
`334d7b31ad3ff7b616af7ad06048498e4efb5fdf3f1172253d0ffc570fb56845`.
All 20 automatic-mask generation requests produced PNGs: 18 completion forwards,
20 restoration forwards, zero detector/optimizer work, and 174.15 CPU processing
seconds excluding loading (frozen cap 300 seconds). Two hair masks remain empty;
nearly hidden sources are
excluded. Runtime uses existing pinned libraries; their Windows access requires
the approved unsandboxed inference process, without installing packages or changing ACLs.

Every final PNG reconstructs exactly from captured floats; 59 artifacts and
unchanged model fingerprints are independently verified. All ten rows/20 outputs
were visually inspected. Central masks, hands and objects are often replaced,
but cloth/strap/finger/petal remnants persist. Sunglasses retain or regenerate
frames; hair remains untouched because both predictions are empty. Scarf/glove
and flower estimates have conspicuous generated artifacts. This is assistant
development review, not expert/user acceptance or hidden-identity recovery.
No assisted outputs were rerun and no model was selected.

Downstream results SHA256
`6feceb94e9c21c87f88c1b9fbdbfa6bc6f02ebe9745b35df061f60c02df841cb`;
`independent_verification.json` SHA256
`5cfb067b650c1069f68749a2d58ae0882d5b76e18de18e424e59ae41af117ece`;
`visual_review.json` SHA256
`e08298b5f213c8546d4435c7d0c2451a99672c9a4aa0e8f3ee09a596bd430ae3`.
The audit's earlier visual-pending flag is preserved; this separate receipt
records completed visual inspection.

Report: `C:\xampp\htdocs\YEAR 4\Testing\VARIED_COVERING_RESULTS.md` ↔
`~/forensic-dgp/VARIED_COVERING_RESULTS.md` after transfer.
**Next:** audit missing covering footprints and the hair/near-hidden failures
before choosing a distinct data/processing intervention. No additional VM
training or app swap yet.
The full Goal remains active; no commit or push occurred.

## Previous milestone — 3 October 2026: varied-covering return audit prepared

The camera results and local follow-up are already audited in
`C:\xampp\htdocs\YEAR 4\Testing\REAL_CAMERA_RESULTS.md`. No camera checkpoint
qualified for promotion. The frozen V2 varied-covering pilot below remains the
next experiment. Its return archive/checksum are **not present locally** at this
milestone. No live VM job status was observed; local absence does not establish
whether the remote job has run. Preserve an executed VM workspace rather than
restarting over its evidence.

The new independent local verifier is ready:
`C:\xampp\htdocs\YEAR 4\Testing\scripts\audit_varied_covering_results.py`.
Its intended repository counterpart is
`~/forensic-dgp/scripts/audit_varied_covering_results.py` after a future transfer;
it is outside the frozen VM bundle and is not required for the CUDA pilot.
It expects exactly 1,653 regular return members: two unselected checkpoints,
1,638 saved masks and 256 step records, plus the fixed metadata/preview.
It checks transfer/inventory bindings, supported-pixel counts, frozen schedules,
fresh-step versus cumulative-update lineage, finite checkpoint tensors, unchanged
reference-head/BN state and the full 10-row preview. Passing training-fit checks
cannot select `best.pth` or waive the original 425-case/practical-output gates.
Final optimizer tensors stay on the VM; their logged counters/hashes do not
constitute independent local moment inspection.

Eleven local counterexample tests pass, including unsafe/partial archive refusal,
missing-return refusal before extraction/report creation, changed input conditions,
inherited-moment counters, malformed losses, clear/retention failures and frozen
tensor changes. These tests use arithmetic and tiny tensor fixtures; no model,
optimizer, training or inference is constructed. The initial temporary-directory
permission errors were resolved by using inherited workspace `scratch/` permissions.
All frozen V2 code/data bindings and archive/protocol/audit hashes remain unchanged.

Preparation-only evidence:
`C:\xampp\htdocs\YEAR 4\Testing\outputs\varied_covering_return_audit_preparation_v1\verification.json`
SHA256 `bcde7d19ec93e33341325e9e008e86f3467559ff4c0b39419e76dbe83fcf665c`.
Its intended receiving path is
`~/forensic-dgp/outputs/varied_covering_return_audit_preparation_v1/verification.json`
only after transfer. It explicitly records `actual_return_audit_complete=false`;
no new result extraction, quality report or returned-checkpoint inspection exists.
Review/download instructions are in
`C:\xampp\htdocs\YEAR 4\Testing\VARIED_COVERING_RESULT_REVIEW.md` ↔
`~/forensic-dgp/VARIED_COVERING_RESULT_REVIEW.md` after transfer.

**Next:** execute the packaged V2 pilot on the VM if it has not already run,
download its two return files, run the independent local audit, then inspect the
10-row masks before deciding a separate original-gate and practical face-output
evaluation. Camera files already returned need no repeat transfer. No new CUDA
execution, promotion, app change, commit or push is claimed here; Goal active.

## Previous milestone — 3 October 2026: bounded varied-covering VM pilot packaged

The next finite detector diagnostic is ready for **VM preflight and its frozen
two-branch pilot**. It uses the verified 133-source training registry and 266
native/degraded views. All actual training remains VM-only; no CUDA run, optimizer
update, model selection or application swap has occurred locally.

Use only these verified V2 files:
`C:\xampp\htdocs\YEAR 4\Testing\outputs\varied-covering-vm-bundle-v2.tar.gz`
(50,272,406 bytes) and `.sha256`. Upload through the already configured Windows
Google Cloud SDK Shell to `/home/janusdominic0/` on `forensic-dgp-thesis`.
Archive SHA256 `780c44f2e3f724ff600eb869e99422b1bf81b18678353a59ac68afb137e32d0b`.
Extracted VM workspace is `~/forensic-dgp/varied_covering_vm_bundle/`.
Independent audit verifies all1,010 regular members, the LF sidecar, 21 code
files, all165 raw-source dependencies, unchanged held-out membership/support and
independently reconstructed sampling. Audit:
`C:\xampp\htdocs\YEAR 4\Testing\outputs\varied_covering_package_validation_v2\verification.json`
SHA256 `2c82066ea9a608e5e75907f4b5245e955bacb8aab2a2f85aebe95bff7580c66a`.
This audit is local evidence; the VM counterpart exists only after transfer.

`existing91` and `varied133` independently start from audited camera91 `last.pth`
(994 cumulative model updates) with **fresh empty AdamW state**. Both run two
epochs×64 batches:128 updates each,256 across independent branches. Each final
model has1,122 cumulative updates/fresh moment step128, not256 sequential updates.
The treatment substitutes one new covering and one new clear source per batch;
common slots, conditions and replay match. Batch8 has4 real/4 reflection inputs,
each domain split2 covered/2 clear. Every arm source appears native/degraded and
all280 training fixtures appear; new positive family slots are26 hand,26 hair,
26 cloth,25 object and25 opaque eyewear/mask. No validation/test forward.

Seven local checks pass, including actual Windows refusal before output/model/
optimizer creation, native/degraded source exposure, clear-failure preservation,
explicit hair-source resolution and uncertain-pixel loss exclusion. Static/data
and tensor inspection construct no local model or optimizer. The source's old
moment tensors are not needed or consumed. Training/measurement budget is3,686
image forwards plus a separate one-image/zero-update preflight; hard cap30 minutes
after loading, with new timing pending. Models/fixtures reuse182,552,838 existing
VM bytes rather than re-uploading them.

Frozen protocol: `outputs/varied_covering_protocol_v2/protocol.json` locally;
`~/forensic-dgp/varied_covering_vm_bundle/inputs/varied_covering_protocol.json`
on VM after extraction. SHA256
`a6ff0780c86a9dc207477b5b40b6ccf6da5d88cd10fc5bb3624fb9e177d13928`.
V2 names the reviewed native hair source868/case202 for preflight. The unused
V1 archive/protocol/audit remain preserved: its positional preflight case194
was a valid hand sample incorrectly described as hair. No V1 transfer/run here.

Exact Windows upload, Google Cloud SSH/tmux preflight/pilot and Windows return
commands are in `C:\xampp\htdocs\YEAR 4\Testing\VARIED_COVERING_VM.md` ↔
`~/forensic-dgp/varied_covering_vm_bundle/VARIED_COVERING_VM.md` after extraction.
Reuse `../feature_vm_bundle/.venv/`, read-only `../coverage_vm_bundle/` fixture/
dependency assets and `../real_camera_vm_bundle/` camera91 weights. No new package
installation/model download is prescribed. Original repository dataset directory
presence and `~/forensic-dgp/outputs/phase4_with_progress/split.json` integrity
remain required. The isolated bundle carries uncommitted code and ignored data;
`git pull` does not supply these artifacts. No commit or push has occurred.

**Next:** upload V2, run the one-batch preflight and finite pilot, then return
`~/forensic-dgp/varied_covering_vm_bundle/varied-covering-results.tar.gz` and
`.sha256` to `C:\xampp\htdocs\YEAR 4\Testing\outputs\`. Independently audit
states/budgets/masks/metrics and inspect the10-row preview before original425-gate
and fixed practical-output evaluation. Fit success does not create `best.pth`
or qualify detection/completion. Original failures remain; nearly hidden
automatic rejection is unresolved. The full Goal remains active.

The COFW data-stage report and milestones below preserve their preparation-time
recipe-pending statements. The separately verified V2 protocol above supersedes
those holds for this finite diagnostic only; it does not rewrite old evidence.

## Previous milestone — 3 October 2026: varied COFW covering data verified

Added 42 assistant-reviewed, training-only COFW detector examples after the
camera audit exposed missing hair/scarf/hand coverage. Current registry:
`C:\xampp\htdocs\YEAR 4\Testing\dataset\detector_supported_review_v3\manifest.json`
(intended `~/forensic-dgp/dataset/detector_supported_review_v3/manifest.json` only
after transfer). It has 165 records: 133 training (89 covered/44 clear),
25 unchanged validation and seven unchanged previously inspected test records.
All 123 V2 records, PNG bytes, held-out support and the mannequin remain intact.
The registry is not a ready GPU recipe or selected model.

The complete 503,327,162-byte official COFW archive passes publisher MD5/ZIP
checks. Independent audits verify all 1,345 original training images, 42 selected
native pairs, 210 proposal assets and 493 registry data files. No test RGB was
decoded/admitted. Exact checks find no overlap with 290 declared previous
source/crop files; full-corpus, identity and pretraining separation is unverified.
Current official metadata declares CC BY 4.0; preserve attribution.

New positives include hands, obstructing hair, cloth/scarves, objects and opaque
eyewear/masks. Eleven new clear controls include three clear-glasses examples.
Native-source polygons have explicit unsupervised boundaries. A failed raster
audit is preserved; V3 excludes its 12 ambiguous native pixels before training.
Old gates/held-out support are unchanged. These are approximate pilot detector
labels, not hidden-face targets. The earlier partial archive remains excluded.

Report: `C:\xampp\htdocs\YEAR 4\Testing\COFW_DATA_PREPARATION.md` ↔
`~/forensic-dgp/COFW_DATA_PREPARATION.md` after transfer.
Registry SHA256 `abab07e152941c4ae24d8aea3755fa7aaedb18965b25f6d0d1b6a7e00094aadd`;
dataset audit `outputs/cofw_supported_dataset_validation_v1/verification.json`
SHA256 `a403f14457d7ba850ff2dbea4236b1db88276cd9e5d4f8192a65e0a0159261cc`;
source/geometry audit `outputs/cofw_covering_data_validation_v3/verification.json`
SHA256 `e16625fa2f66d51c33a8ee5481dec821648d531522a93d4f54236e187938faf1`.
Audit paths are relative to both repository roots; new artifacts are local only.

Paired camera cache is now verified locally:
`C:\xampp\htdocs\YEAR 4\Testing\outputs\cofw_camera_pairs_v2\manifest.json`
↔ `~/forensic-dgp/outputs/cofw_camera_pairs_v2/manifest.json` after transfer.
It has 133 native/133 degraded training views, with all previous 91 pairs exact.
Independent reconstruction verifies every degraded pixel and unchanged
target/support/geometry; held-out inputs are absent. Both five-row preview sheets
are inspected. Manifest SHA256
`b6f4a800427e96a52468a0ec10eec478ec8f5875e1302b35ffd6c072dfec041c`;
`outputs/cofw_camera_pairs_v2/independent_verification.json` SHA256
`c7fb19ff5d936c5e43f8a5c3a56b8ca59db40986d04f3be4c156660258652471`.
The incomplete V1 cache is preserved/excluded; zero model/optimizer work.

**Next:** prepare a distinct finite VM comparison of existing data versus
varied coverings, retaining clear/retention checks and the
original 425-case qualification gates. Package previous raw-source dependencies;
V3 alone is not a complete provenance bundle. No new CUDA recipe, upload,
training, promotion, app change, commit or push at this milestone. Actual
training stays on the VM; Goal active.

## Previous milestone — 3 October 2026: camera results audited; covering gaps remain

The user reports the VM result download is complete. Both files are present at
`C:\xampp\htdocs\YEAR 4\Testing\outputs\real-camera-results.tar.gz` (161,238,229
bytes) and its `.sha256` sidecar (93 bytes). VM originals are
`/home/janusdominic0/forensic-dgp/real_camera_vm_bundle/real-camera-results.tar.gz`
and its `.sha256`. Transfer SHA256 matches:
`5dc73ce77291b8622530ae825676728776caa8fdbfd36250f1d0a4e23b1c3c24`.
All 1,866 archive members, 1,848 binary masks, 336 logged steps and three checkpoint
states pass independent verification. Each checkpoint changes encoder/decoder/head
weights while retaining 92 frozen reference-head/BN tensors. Logged L4
training/measurement time is 53.16 seconds, excluding startup and archive export.
Final optimizer tensors remain on the VM and were not independently inspected.
No checkpoint promotion, local training, commit or push has occurred.

Full report is `C:\xampp\htdocs\YEAR 4\Testing\REAL_CAMERA_RESULTS.md` (VM
`~/forensic-dgp/REAL_CAMERA_RESULTS.md` after a future transfer). Return audit:
`C:\xampp\htdocs\YEAR 4\Testing\outputs\real_camera_results_validation_v1\verification.json`,
SHA256 `fbe47739fa7ec05f88b7ee22861d81b64b945c320109e4b7aee5a05c8af74245`.
Local evidence counterparts under `~/forensic-dgp/outputs/` exist only after
explicit transfer; returned VM originals remain within `real_camera_vm_bundle/`.

The predeclared fit checks reproduce two passes and three failures. Camera83
raises old degraded training IoU from the native83 control's 0.7861 to 0.9095.
Camera91 improves new-source IoU to 0.6394 native/0.5185 degraded. However,
retention errors rise or prior fit declines, and camera83 marks 137 supervised
pixels on clear fixture 171. These are training-cohort measurements; original 425
qualification gates are not evaluated, changed or waived. No `best.pth` is created.

The bounded local follow-up verifies 43/44 CPU return masks exactly and one
threshold-ambiguous pixel; it then compares all three branches on the same 36-case
gallery with the existing threshold 0.5/3px margin. All 108 practical probabilities,
216 raw/proposal masks and 144 metric records pass independent verification.
152 detector-image forwards take 13.74 CPU seconds excluding loading; no training.
Degraded hand reference recall rises from source42's 5.3% to camera91's 57.0%.
Masks, eyewear/glare and objects improve; all four clear controls remain empty.
Hair remains missed, one degraded scarf/gloves case is almost empty, and both
nearly hidden cases fail automatic rejection. All six preview sheets are inspected.
This is exposed development evidence, not an unseen holdout or checkpoint selection.

Four actual CodeFormer estimates using cached camera83/camera91 proposals take
38.03 CPU seconds excluding loading, with four completion/four restoration
forwards and zero detector/optimizer work. All four final PNGs reconstruct exactly
from captured stages under the existing Auto restoration/palette policy. Preview:
`C:\xampp\htdocs\YEAR 4\Testing\outputs\real_camera_completion_review_v1\preview.png`
(VM `~/forensic-dgp/outputs/real_camera_completion_review_v1/preview.png` only
after explicit transfer). Plausible mouths/beards appear, but cloth/strap edges
and fingers remain because the automatic masks are incomplete. The main app
retains its baseline detector and manual correction; no candidate is qualified.

**Next:** prepare varied reviewed hand/hair/scarf covering examples and full
removal footprints with clear-control/retention counterexamples before a distinct
VM pilot. Retain camera degradation as a supported ingredient; do not repeat
the same recipe, weaken gates or train the inspected RealOcc validation gallery.
Completion/restoration retraining is not justified by these mask-remnant failures.
The Goal remains active; all actual training stays on the VM.

The Windows Google Cloud CLI is configured for project `forensic-dgp-thesis`.
The PuTTY backend rejected the initial command with two remote sources; separate
one-file SCP commands subsequently completed according to the user. Full install,
prompt answers, exact transfer commands and checksum instructions are saved in
`C:\xampp\htdocs\YEAR 4\Testing\WINDOWS_GCLOUD_TRANSFER.md` (repository VM
counterpart `~/forensic-dgp/WINDOWS_GCLOUD_TRANSFER.md` after a future transfer).
This new guide is not part of the immutable 419-file pilot bundle. No transfer
speed was measured and no VM zone is inferred from the instance name.

The uploaded Mendeley source, eight reviewed detector annotations, supported
dataset V2 and paired camera inputs are verified. The three-arm VM package was
prepared on 2 October and its independent audit is complete. The preparation
details below remain the experiment's historical protocol. The completed return/
follow-up audits above supersede its preparation-time pending CUDA statements.

The research tracks remain separate: restoration retains Phase 3
`C:\xampp\htdocs\YEAR 4\Testing\checkpoints\dgp_zamboanga_final.pth` ↔
`~/forensic-dgp/checkpoints/dgp_zamboanga_final.pth`; Phase 5 ArcFace identity
training awaits its full GPU pilot. Completion retains the gated U-Net research
history and CodeFormer benchmark, documented in
`C:\xampp\htdocs\YEAR 4\Testing\COMPLETION_TRAINING.md` ↔
`~/forensic-dgp/COMPLETION_TRAINING.md`. The working main upload flow below uses
pinned CodeFormer completion and selective visible restoration. This new pilot
changes only the detector; it trains neither restoration nor the generator.

The main application at configured `http://127.0.0.1:8000` now serves upload →
detected removal preview → paint/erase/import correction → reviewed generation of
one estimate. Auto/Off/On visible restoration and PNG/original-mask-result ZIP
downloads work in the existing terminal design. Source:
`C:\xampp\htdocs\YEAR 4\Testing\face_workflow.py`, `face_workflow_web.py`,
`templates\face_workflow.html`, `static\face_workflow.js`, `static\face_workflow.css`;
intended receiving paths are `~/forensic-dgp/face_workflow.py`, `face_workflow_web.py`,
`templates/face_workflow.html`, `static/face_workflow.js`, `static/face_workflow.css`
after transfer. The historical `/reconstruct` endpoint, Phase 3 checkpoint and
original template/script remain unchanged; no failed detector candidate is promoted.

Fifteen unit/adapter/palette checks pass. The active API engine is
`C:\xampp\htdocs\YEAR 4\Testing\face_workflow_palette.py` with input-only color
policy `face_color_policy.py` (VM `~/forensic-dgp/face_workflow_palette.py` and
`~/forensic-dgp/face_color_policy.py` after transfer). It extends the unchanged,
audited `face_workflow.py` base route. Near-grayscale inputs retain grayscale;
Off preserves observed pixels exactly and color inputs receive no palette change.
Bundled inline Playwright checks the actual upload,
mask import, keyboard paint/erase/undo, review gating, generation and bundle download;
375/768/1280 layouts have no accidental overflow or unexpected exceptions/console
errors. Downloaded bundle pixels preserve the reviewed mask exactly and change zero
pixels outside it with restoration off. Additional browser checks verify clear-glasses
Auto bypass, forced On, degraded Auto restoration, final PNG download and rejection
of a reviewed nearly hidden face with no output (expected HTTP 422). The first
256-pixel browser completion request takes 7.8 seconds including lazy loading.
Current inference device is CPU, accurately shown in the UI. These are runtime
checks, not population quality, hidden identity accuracy or full accessibility proof.

Runbook `C:\xampp\htdocs\YEAR 4\Testing\LOCAL_FACE_WORKFLOW.md` (VM
`~/forensic-dgp/LOCAL_FACE_WORKFLOW.md` after transfer) documents required pinned
models, configuration, observed timing and the conditional geometry/quality heuristics.
The automatic detector remains a weak development baseline: mandatory review and
manual correction are explicit, and corrected outputs are not automatic-detection
evidence. Inputs/outputs are returned in the request, not persisted by this API.
Checkpoints under ignored `outputs/` are not transferred by `git pull` alone.

The separately source-reviewed RealOcc gallery is frozen at 16 native/degraded
cases: two hands, one obstructing-hair case, two scarves, two objects and one
nearly hidden rejection source. Author `val` membership and publisher labels remain
intact. Exact files have no match against the previous 115-source review manifest
and ten-source practical cohort; full-corpus/identity/pretraining overlap is unverified.
No training admission. Protocol:
`C:\xampp\htdocs\YEAR 4\Testing\outputs\broad_covering_gallery_v1\frozen_protocol.json`
(VM `~/forensic-dgp/outputs/broad_covering_gallery_v1/frozen_protocol.json` after transfer),
SHA256 `b7a1a447482d650bb0a18b13d649f0c548c3c59acd6db202111169599c805bc9`.
The bounded local inference comparison completed: 28 saved outputs across 32 rows,
16 detector, 21 completion and 15 restoration forwards, 28 generation requests,
186.48 CPU seconds excluding loading and zero optimizer updates. Source/pixel
audits pass; all seven usable assisted degraded cases improve known-visible MAE.
The base assisted mean is 0.017155752 versus input 0.021175412 (18.98% reduction).
Both reviewed nearly hidden cases reject before generation; **automatic detection
misses both** and source review holds those outputs separately. Assistant inspection
finds useful hand/hair/scarf/object estimates with approximate joins; one scarf
retains a textured beard/knit appearance. Native automatic coverings mostly remain.
Outputs are under local `outputs\broad_face_workflow_outputs_v1\`
(VM `~/forensic-dgp/outputs/broad_face_workflow_outputs_v1/` after transfer).

Palette V2 fixes false color in eight of 28 cached outputs without new forwards.
All cached compositions and visible metrics pass an independent audit; the real
browser hand-mouth output matches V2 pixels exactly. The assisted degraded mean
becomes 0.017035153 (19.55% reduction, exposed development cohort only). Four
grayscale preview rows are inspected. V1 threshold/policy evidence is preserved;
V2 threshold 4 is explicitly source-derived, not unseen evaluation. Report:
`C:\xampp\htdocs\YEAR 4\Testing\PRACTICAL_BROAD_OUTPUT_RESULTS.md` (VM
`~/forensic-dgp/PRACTICAL_BROAD_OUTPUT_RESULTS.md` after transfer).

The XSeg1 pretrained visible-face comparison completed on 36 native/degraded
cases. Its official CRC32 matches; observed SHA256 is
`c4d1498b8a03b5fe2a3a5d2ef2a0402ab03bd51edaf5b2d8d5fb764702a97dd3`.
Local model/provenance root is `C:\xampp\htdocs\YEAR 4\Testing\outputs\xseg_pretrained_v1\`
(VM `~/forensic-dgp/outputs/xseg_pretrained_v1/` only after explicit transfer).
XSeg is research-only and not selected by the app. Its frozen ellipse conversion
finds more masks/hands/objects, but marks clear glasses, normal hair and background.
Clear-glasses controls receive 10,918/10,227 unwanted marked pixels; the usable
knit-scarf pair wrongly rejects. Both nearly hidden cases reject correctly.
All 36 probabilities and 108 masks pass an independent audit; six preview sheets
are inspected. V1 interrupted on one probability 1+one float32 ULP; its evidence
is retained. V2 applies the official consumer's clipping, reuses 24 probabilities
and runs 12 new forwards without changing spatial criteria. Report:
`C:\xampp\htdocs\YEAR 4\Testing\PRACTICAL_XSEG_RESULTS.md` (VM
`~/forensic-dgp/PRACTICAL_XSEG_RESULTS.md` after transfer).

The returned direct-occlusion checkpoint comparison is now complete on the same
36 native/degraded cases, with ten reused native masks and 26 new CPU detector
forwards, zero generation and zero optimizer updates. It takes 4.31 seconds
excluding model loading. Source and saved-mask audits pass; all six preview
sheets are inspected. The protocol is frozen at local
`C:\xampp\htdocs\YEAR 4\Testing\outputs\practical_direct_detector_v1\frozen_protocol.json`
(VM `~/forensic-dgp/outputs/practical_direct_detector_v1/frozen_protocol.json` after transfer),
SHA256 `69c6a75e4dc1af099a3cb1a621082a870966170108e9a2d2255d614b708a8db8`.
The direct detector finds native coverings better than the app baseline, but
degraded hand reference recall drops from 31.21% to 5.32%, obstructing-hair masks
remain empty, and both nearly hidden cases miss rejection. All four clear-control
masks are empty. Reference footprints are approximate development proposals.
Reflective epoch42's original synthetic-retention failure remains unchanged;
neither this comparison nor XSeg selects an application checkpoint. Report:
`C:\xampp\htdocs\YEAR 4\Testing\PRACTICAL_DIRECT_DETECTOR_RESULTS.md` ↔
`~/forensic-dgp/PRACTICAL_DIRECT_DETECTOR_RESULTS.md` after transfer.

The user-supplied
`C:\xampp\htdocs\YEAR 4\Testing\Occluded and Low-Light Human Face Detection Datase.zip`
is preserved and Git-ignored. All 11,987 JPEG members pass CRC/decode checks;
6,101 dimension-bound decoded RGB images are distinct and 5,886 instances repeat.
The ZIP supplies no labels or splits. Six source sheets expose 363 file IDs;
this is not a semantic review of the entire archive. No exact byte/native-pixel
match appears against the previous 115 supported raw sources; full-corpus,
identity and pretraining overlap remain unverified. Publisher attribution and
CC BY 4.0 are recorded in
`C:\xampp\htdocs\YEAR 4\Testing\MENDELEY_OCCLUSION_DATA.md` ↔
`~/forensic-dgp/real_camera_vm_bundle/MENDELEY_OCCLUSION_DATA.md` after transfer.
Only eight related captures receive reviewed pilot labels: three hands, one
obstructing-hair case, sunglasses, a mask, combined sunglasses/mask and one clear
control. They share one training-only cohort, not eight independent identities.
V1/V2 proposals remain intact; V3 makes native boundary/crop-edge uncertainty
explicit. All 48 V3 assets pass geometry/source/support checks. These are detector
labels, without uncovered-face targets or independent expert adjudication.

Supported dataset V2 is local
`C:\xampp\htdocs\YEAR 4\Testing\dataset\detector_supported_review_v2\manifest.json`
and is bundled at
`~/forensic-dgp/real_camera_vm_bundle/dataset/detector_supported_review_v2/manifest.json`.
It has 123 records: 91 training (58 covered/33 clear), 25 unchanged validation
and seven unchanged previously inspected test. All 115 old records retain active
metadata and bytes, including full held-out supervision and the mannequin.
The independent audit verifies 291 data files plus the manifest. The new input
cache at local `outputs\real_camera_pairs_v1\` ↔
`~/forensic-dgp/real_camera_vm_bundle/outputs/real_camera_pairs_v1/` contains 91
native/degraded pairs. Camera changes affect observed RGB only; targets, support,
geometry and neutral padding stay fixed. All 182 cases reconstruct exactly.
Five new camera/scheduling contract tests pass without model or optimizer work.
Standalone scarf/general-object training coverage is still missing.

The audited transfer files are
`C:\xampp\htdocs\YEAR 4\Testing\outputs\real-camera-vm-bundle.tar.gz` and
`C:\xampp\htdocs\YEAR 4\Testing\outputs\real-camera-vm-bundle.tar.gz.sha256`.
Upload both to VM `~`, then extract to `~/forensic-dgp/real_camera_vm_bundle/`
using the exact commands in
`C:\xampp\htdocs\YEAR 4\Testing\REAL_CAMERA_VM.md` ↔
`~/forensic-dgp/real_camera_vm_bundle/REAL_CAMERA_VM.md`.
The 18,892,722-byte archive contains 419 verified regular files and an LF checksum
sidecar. SHA256 is
`2b237b2e121951203bde131be5d7c649910392df47bff1bbc953f81a8c8c0e2b`;
protocol SHA256 is
`a8adb21910c5aed249f5f2a75e17eef92db35717bd428326c360a1aecf3b99e1`.
Independent package audit:
`C:\xampp\htdocs\YEAR 4\Testing\outputs\real_camera_package_validation_v1\verification.json`,
SHA256 `b27897d6781907ae0f27e7a76d0b014e63156d33cec10a093532362fc0253f8b`.
It verifies all members, code compilation, dataset/pair bindings, reconstructed
schedules and unchanged original split. The Windows/CPU guard is tested; returned
CUDA execution records are not yet independently verified. No local training or
smoke optimizer ran.

The pilot independently starts three branches from the same reflective epoch42
weights and inherited AdamW moments: `native83` (old native control), `camera83`
(same sources with camera degradation) and `camera91` (adds the eight selected
sources). Each runs two epochs of 56 batches: 112 updates per branch, 336 across
the experiment. All branches replay the same 280 prior reflection fixtures;
only 34/336 real slots in the new-source branch use the related cohort. Existing
VM assets are read-only under `~/forensic-dgp/coverage_vm_bundle/`; reuse
`~/forensic-dgp/feature_vm_bundle/.venv/`. Preflight requires CUDA/VRAM, pinned
dependencies, the original `~/forensic-dgp/outputs/phase4_with_progress/split.json`,
both clean dataset directories, exact file hashes and one finite batch with zero
updates. Plan approximately 5–15 minutes for the first diagnostic; timing is
unmeasured. A 30-minute cap covers training/measurement after loading.

Only training-cohort fit is measured here. The original 425-case qualification
gates are not evaluated or relaxed. No `best.pth` is created: three `last.pth`
files remain unselected. After the VM finishes, download
`~/forensic-dgp/real_camera_vm_bundle/real-camera-results.tar.gz` and its `.sha256`
to `C:\xampp\htdocs\YEAR 4\Testing\outputs\`. Preserve all three final optimizer
files on the VM; they are deliberately omitted from the download. Next local
action is an independent state/mask audit and ten-row preview review, followed
by a decision on original-gate and end-to-end evaluation. The self-contained
bundle requires no `git pull`; new uncommitted code and ignored data are included.
All actual training stays on the L4 VM. Goal remains active.

## Previous development decision — 2 October 2026: selective visible restoration

The fixed degraded comparison completed 80 saved outputs plus 40 DGP intermediates
on 20 inputs, with 24 new nonempty completion forwards, six empty bypasses, ten
reused native completion outputs and 40 restoration forwards. No detector or
optimizer update occurred. The independent pixel audit confirms all fixed
compositions/known-visible metrics and preservation of completed pixels in post
arms. CPU elapsed 124.0 seconds, excluding initial model loading.

Mean known-visible degraded MAE is 0.020387 with restoration off, 0.046924 with
legacy DGP before completion, 0.046970 with visible-only DGP after completion and
0.022522 with a 25% post blend. All forced DGP routes worsen this metric; native
controls also drift. Assistant inspection finds color shifts/smoothing and
conspicuous joins. These routes are not selected for automatic restoration.
Phase 3 weights and historical experiment outcomes remain unchanged.

Report: `C:\xampp\htdocs\YEAR 4\Testing\PRACTICAL_RESTORATION_RESULTS.md`, intended
VM counterpart `~/forensic-dgp/PRACTICAL_RESTORATION_RESULTS.md` after transfer.
Outputs: `C:\xampp\htdocs\YEAR 4\Testing\outputs\practical_restoration_outputs_v1\`,
intended VM counterpart `~/forensic-dgp/outputs/practical_restoration_outputs_v1/`.
Protocol SHA256 `74c652e1d1c604c5229fc953117a45b3c994e17c01fb23158ee9b4bccc17bf00`;
result SHA256 `2f1b2ff00237302bb3fd42c9188f63541292f777ba0d2df8aa10cc1c21f6223d`;
verification SHA256 `569f1c2c5112e17104cb7cfce54f6f262fb8dd56b098bf8520ad47e7a3d982eb`.

Separate official CodeFormer restoration weights were acquired (376,637,898 bytes)
with observed SHA256 `1009e537e0c2a07d4cabce6355f53cb66767cd4b4297ec7a4a64ca4b8a5684b7`.
Strict loading verifies 515 tensor states, codebook 1024, zero trainable parameters;
three adapter tests pass. This is separate from the pinned inpainting checkpoint.
The new inference-only protocol is frozen at
`C:\xampp\htdocs\YEAR 4\Testing\outputs\practical_face_restoration_v1\frozen_protocol.json`
(VM `~/forensic-dgp/outputs/practical_face_restoration_v1/frozen_protocol.json` after
transfer), SHA256 `c384f1b25e9ab80f44ba6ebef3e47c072a4dc57d406d525f9a637f3abf99b229`.
It completed 40 restoration forwards and 100 saved outputs in 230.3 CPU seconds
(excluding model loading). Independent verification checks all saved pixels,
intermediates, masks and known-visible metrics. Fidelity 1.0 with a 50% blend onto
visible pixels lowers degraded mean MAE from 0.020387 to 0.015654 (23.2%), improving
all ten degraded cases and changing zero completed-region pixels. All 20 preview
rows were inspected. This is selected as the development restoration route;
forced native controls drift, so clear inputs must bypass it with an override.
It is not evidence of hidden identity or whole-scope readiness. Report:
`C:\xampp\htdocs\YEAR 4\Testing\PRACTICAL_FACE_RESTORATION_RESULTS.md` (VM
`~/forensic-dgp/PRACTICAL_FACE_RESTORATION_RESULTS.md` after transfer).
Result SHA256 `de6a7bf7e177d7f2f8d3ed2c0536de05c64e6662e464046dda597a50bd9f7cb0`;
verification SHA256 `b34c4442ffbb43ffbb811cd02b631622603e6a2639f86e9741e5334be69c7635`.

The author-linked public RealOcc archive downloaded fully (226,205,702 bytes in
7.7 seconds), SHA256 `de6a26ecc00457c0067991d95906b15638a2f217c5b5ea01a5db42435510577f`.
All 550 images and 550 binary masks load and match the 550 author `val.txt` entries.
The masks label visible face (1), with coverings AND background (0); transparent
glasses follow the publisher's different scope. They must not be blindly inverted
into this project's covering labels. Source-only contact sheets contain hand,
scarf, object and obstructing-hair candidates. Further crop/removal-footprint,
terms and exposure review is required; no training admission occurred. Integrity:
`C:\xampp\htdocs\YEAR 4\Testing\outputs\realocc_source_v1\source_review\integrity.json`
(VM `~/forensic-dgp/outputs/realocc_source_v1/source_review/integrity.json` after transfer),
SHA256 `aaf1645debcbf3d43e794fdf92def776a4ba6f314dac3182a04f679f45b7114e`.
The incomplete COFW archive stays excluded; original author validation membership
is retained. File integrity alone does not establish identity or pretraining overlap.

The existing main app runs locally at its configured `http://127.0.0.1:8000`.
Bundled inline Playwright finds no horizontal overflow, page exceptions or console
errors at 375/768/1280 pixels; evidence is in local git-ignored `scratch/`.
This verifies the original layout, not the still-pending integrated workflow.
Next: freeze the broader covering gallery and integrate selective restoration,
mask correction, visibility rejection and downloads into the existing design.
All actual training remains VM-only. Goal active; no new transfer, commit or push.

## Previous milestone — 2 October 2026: revised covering footprints

Source-reviewed V3 removal footprints include full opaque glasses, wider obscured
lens interiors and the facial mask strap segment while excluding a visible clear
wire. Four coverings plus two empty controls were frozen before generation. These
are operator proposals, separate from unchanged detector labels/splits. Opaque
proposals have up to a 6-pixel margin relative to old labels; no extra dilation
is applied to glare interiors. The turned mask/clear-glasses face is a difficult
diagnostic, beyond the initial frontal/mild-turn cohort.

CodeFormer/AOT completed 12 requests (eight nonempty forwards, four bypasses),
reusing 12 previous outputs, in 36.2 CPU seconds with zero failures or training.
Independent verification checks all saved pixels/proposal deltas and frozen
artifacts; zero changed pixels outside the active masks or in protected eyewear.
CodeFormer now gives useful estimated eyes for the dark sunglasses and mirrored
glasses, and reduces white glare while retaining clear frames. The mask with
clear glasses still has a poor nose transition. CodeFormer remains the development
baseline; no automatic detector checkpoint is promoted.

Report: `C:\xampp\htdocs\YEAR 4\Testing\PRACTICAL_FOOTPRINT_RESULTS.md`, intended
VM counterpart `~/forensic-dgp/PRACTICAL_FOOTPRINT_RESULTS.md` after transfer.
Outputs: `C:\xampp\htdocs\YEAR 4\Testing\outputs\practical_footprint_outputs_v3\`,
intended VM counterpart `~/forensic-dgp/outputs/practical_footprint_outputs_v3/`.
Protocol SHA256 `4a256b9a00c25f68414010877a8cf96c1e3a9a283842360ed64978a676a5c5d3`;
result SHA256 `1c79c0627697a353e57f47adb8943e1e5fda8af1ed51ec00f53aa01dbf3ef17c`;
verification SHA256 `638b13e094082abdf4c8e088c6fb0c33010c843aad62fe723680e44ddc05b40f`.

Next: compare completion alone against pre-completion restoration and visible-only
post-completion restoration on the declared degraded copies. Reference metrics
cover known visible pixels only. Continue broad covering data preparation and
the existing main-UI integration; full-family usefulness and visibility rejection
remain unverified. Goal stays active. No new VM training, upload, commit or push.

## Previous comparison — 2 October 2026: AOT-GAN and covering footprints

Face-specific AOT-GAN CelebA-HQ weights were acquired from the author-linked
public folder. Exact source/license at revision
`2cd1afd8fdfabb101c678f6062d14bc7d302509e` are retained. The 60,829,150-byte model
loads 108 finite tensor states strictly with `weights_only=True`. The observed
SHA256 is an acquisition fingerprint, not a publisher-supplied checksum. 512 is
the tested inference size/repository default; exact original training history
is not verified (see preserved provenance erratum).

Ten unchanged reviewed native inputs/masks were compared with 20 cached baseline
outputs. Eight nonempty forwards and two empty control bypasses completed in
29.4 CPU seconds, with zero failures, detector forwards or optimizer updates.
Independent verification checks all 10 new outputs and 20 reused baselines;
zero pixels change outside the active masks. Assistant triage finds four useful
covered estimates, one partial and three needing fixes, with both controls exact.
AOT supplies facial anatomy where generic LaMa fails, but does not resolve the
dark-eyewear, white-glare or mask/clear-glasses boundary. No backend is promoted.

Report: `C:\xampp\htdocs\YEAR 4\Testing\PRACTICAL_AOT_RESULTS.md`, intended VM
counterpart `~/forensic-dgp/PRACTICAL_AOT_RESULTS.md` after transfer. Outputs:
`C:\xampp\htdocs\YEAR 4\Testing\outputs\practical_aot_outputs_v1\`, intended
VM counterpart `~/forensic-dgp/outputs/practical_aot_outputs_v1/`.
Protocol SHA256 `a53c4b8e3a27f30d9ccf0c49c7e04454f6303d72b3c6c08c5977ee168fb7d09b`;
result SHA256 `bf77388894dd27b68d7f8065a237b6d4a21855770eded8c88e1b79e5ce32a278`;
verification SHA256 `412222ebb436be618a2ada7b92ae1e4ac5bd30a014620b25dc593f33a1bd25af`.

Next: source-review full opaque eyewear rims/bridges, glare boundaries while
preserving clear frames, and the mask edge near the nose. Prepare a new operator
removal-proposal version, separate from unchanged historical labels. Freeze the
new masks before comparing CodeFormer/AOT on targeted failures. Locate remnants
relative to the actual active mask before blaming a detector or generator; a
reviewed approximate label need not include the entire practical removal area.
No additional unchanged pretrained/detector run or VM training is the next action.

The complete agreed workflow remains unmet: degraded restoration, broad covering
families, facial-visibility rejection, main-UI integration and Playwright checks
are pending. Goal stays active. All actual training remains on the L4 VM, without
a fixed user-imposed GPU-hour cap. No new transfer, commit or push occurred.

## Newest comparison — 2 October 2026: generic LaMa is not selected

The Places2 Big-LaMa checkpoint was acquired from the author's recommended
mirror and matched its published SHA256. Minimal generator source at revision
`786f5936b27fb3dacd2b1ad799e4de968ea697e7` retains the Apache-2.0 license.
The generator-only checkpoint loads with restricted `weights_only=True`; no
Lightning training environment or project PyTorch downgrade is required.

Two arms on the same ten native inputs and unchanged reviewed masks compare
native256 and visible-support-normalized 512 inference. Twenty requests comprise
16 nonempty forwards and four empty control bypasses; CPU elapsed 37.2 seconds,
zero failures/detector forwards/optimizer updates. Independent verification
checks all 20 outputs and finds zero changed pixels outside reviewed masks.
Assistant inspection finds blurred/missing/distorted facial anatomy in the eight
covered cases, with white glare unresolved. Neither resolution arm is selected.
This is a failure of this generic pretrained checkpoint on this developmental
cohort, not evidence that every LaMa or face-specific completion model fails.

Report: `C:\xampp\htdocs\YEAR 4\Testing\PRACTICAL_LAMA_RESULTS.md`, intended VM
counterpart `~/forensic-dgp/PRACTICAL_LAMA_RESULTS.md` after transfer. Outputs:
`C:\xampp\htdocs\YEAR 4\Testing\outputs\practical_lama_outputs_v1\`, intended
VM counterpart `~/forensic-dgp/outputs/practical_lama_outputs_v1/`.
Protocol SHA256 `64ff7321ca0580ba9af8df0ab247a22bf3c9171cdfed5f6f01021b2023829d5b`;
result SHA256 `2dde4543215701641323aa0661bd3314a00e87fc2027df2450360770dd055b94`;
verification SHA256 `0d9ad01e7a4b0fd6623494c438fbfa8a6ba7f8d9472fced8ed5a514017514de9`.

Next: verify and benchmark a face-specific pretrained mask-conditioned generator;
the AOT-GAN author supplies a CelebA-HQ checkpoint. It is a research candidate,
not a selected backend. Native-only evidence does not resolve degraded restoration,
missing standalone-hand/hair/scarf/object cases, full-eyewear rims or main-UI
integration. Goal stays active; actual training stays VM-only. No new upload,
training, commit or push occurred.

## Latest evidence — 2 October 2026: practical native completion comparison

Current local evidence lives at
`C:\xampp\htdocs\YEAR 4\Testing\outputs\practical_native_outputs_v1\`;
intended VM counterpart `~/forensic-dgp/outputs/practical_native_outputs_v1/`
only after transfer. No new upload, commit, push or VM training occurred.

Ten previously inspected detector-training sources were frozen before generation.
Parent automatic, experimental reflective-42 automatic and reviewed removal masks
were compared with the same CodeFormer inpainting baseline, restoration off and
existing crops. Thirty wrapper requests comprise 22 nonempty generator forwards
and eight empty-mask bypasses; CPU elapsed 90.6 seconds. No detector forward or
optimizer update occurred. An independent artifact/pixel recount verifies all 30
outputs preserve every pixel outside their active mask and checks immutable
sources, cached masks, checkpoint/code hashes and before/after state records.
This is developmental preservation evidence, not hidden-face ground truth.

Assistant visual triage finds four useful reviewed covered estimates, one partial
and three requiring fixes, with both clear controls preserved. Dark sunglasses
are recreated and white glare persists even with reviewed masks. Another detector
run alone cannot address these completion failures. The mask-plus-clear-glasses
join and remaining mirrored-eyewear rim also need work. No checkpoint is promoted;
reflective-42's historical retention failure remains unchanged.

Report: `C:\xampp\htdocs\YEAR 4\Testing\PRACTICAL_NATIVE_OUTPUT_RESULTS.md`, VM
counterpart `~/forensic-dgp/PRACTICAL_NATIVE_OUTPUT_RESULTS.md` after transfer.
Protocol SHA256 `74b80e0ec9ebabb7c8b3b3989f5bc4f44b9a4cf1e39e691c073543f2db4651a5`;
result SHA256 `38776d1a46967c4338a2082fe94ad50c2ca335d767b57c8c4759f681374f5325`;
verification SHA256 `d69bc0ea2a77bdecfa7b710d9be1c64d965ebe1236808c08207d82702006d9db`.
Native-only comparison is complete; the ten degraded copies have not been run.
Standalone hand, obstructing hair, scarf-over-face and other-object families
remain missing from this practical pilot. The full Goal remains active and unmet.

Additional-source research is in
`C:\xampp\htdocs\YEAR 4\Testing\PRACTICAL_DATA_SOURCES.md` (intended VM
counterpart `~/forensic-dgp/PRACTICAL_DATA_SOURCES.md`). Official COFW metadata and
documentation were verified, but its 503,327,162-byte image archive ended after
2,153,540 bytes. The `.partial` artifact is excluded; no new image/mask was
admitted. This does not justify unchanged training.

Next: benchmark pretrained mask-conditioned completion on the unchanged reviewed
native inputs/masks, then compare restoration on declared degraded cases. LaMa is
a research candidate; no LaMa output or improvement has yet been demonstrated.
Review full-eyewear-rim proposals in a new version. Continue missing-family data
preparation and existing-main-UI integration after processing decisions. Actual
training stays on the L4 VM. Preserve the immutable native-lens pilot archive;
do not automatically run it as the new practical workflow's next experiment.

## Active Goal — 2 October 2026: practical workflow discovery complete

The user explicitly requested a Goal after relevant system/workflow questions.
All three rounds are answered. Before creation, this thread's `get_goal` returned
no Goal; historical goal-status wording below belongs to earlier work. The full
current specification is `C:\xampp\htdocs\YEAR 4\Testing\SYSTEM_WORKFLOW_AND_GOAL.md`,
intended VM counterpart `~/forensic-dgp/SYSTEM_WORKFLOW_AND_GOAL.md` after transfer.
The requested Goal has now been created with status `active` and no token budget.
Its objective is the useful combined local workflow, not merely another detector
checkpoint. The five milestones and completion conditions are in that file.

First-round answers: manual review by school staff/thesis researchers; already
cropped face input; review/correct the detected removal area before generation;
local inference with VM-only training; quality takes priority over speed.
Second-round answers: integrate in the existing main UI; one output alongside
original/removal area; automatic restoration with a user override; final-image
download and optional original/mask/result bundle; useful fixed-gallery results
with manual correction allowed.

Final-round answers: add suitable public research datasets when existing coverage
is insufficient; frontal and mildly turned faces first; the existing L4 /
g2-standard-4 VM (4 vCPUs, 16 GB RAM) has credits and no fixed GPU-hour cap was
imposed. Bound and time individual experiments. Actual training remains VM-only;
there is no configured SSH connection, so prepare exact commands/transfer files
when justified by the output audit. These answers authorize no additional VM
provisioning or unlimited repetitions of failed recipes.

Existing main restoration and completion pages remain separate. Read-only local
inspection confirms an RTX3050 Laptop GPU but CPU-only project Torch2.13.0;
CUDA is currently unavailable in the venv. No package installation, training,
application edit or new model inference was performed in this discovery step.
Known runtime differences and confirmed choices are recorded in
`C:\xampp\htdocs\YEAR 4\Testing\PRACTICAL_OUTPUT_SCOPE.md`, intended repository
counterpart `~/forensic-dgp/PRACTICAL_OUTPUT_SCOPE.md`; no new upload occurred.

First evidence milestone: all 83 supported training records were inspected on
five native-source/accepted-crop sheets. The 51 covered / 32 uncovered records
contain 44 mouth-mask cases, six sunglasses/reflective-lens cases, three strong
glare cases and one hand-over-mask case (overlapping tags). No dedicated
hair-over-features, scarf-over-face, standalone hand or other-object example was
confirmed in this subset; this is not an absence claim for the full datasets.
Source fingerprints and group/exact-byte split checks passed. Source manifest
SHA256 remains `860ea1dfba8fa442cab19bf3ab41f6a678109dcdb067c38ec0bd79ce43b51ace`.
No model forward, source mutation or local training occurred.

Report: `C:\xampp\htdocs\YEAR 4\Testing\PRACTICAL_COVERAGE_AUDIT.md`, intended VM
counterpart `~/forensic-dgp/PRACTICAL_COVERAGE_AUDIT.md` after transfer.
Evidence: `C:\xampp\htdocs\YEAR 4\Testing\outputs\practical_coverage_v1\`, intended
VM counterpart `~/forensic-dgp/outputs/practical_coverage_v1/` if transferred.
At this first milestone the practical gallery was not frozen yet. The latest
evidence section above records the subsequent native-only freeze/comparison.

Next: inspect existing source-pool candidates for missing covering families,
add suitable public research examples if needed, and freeze the practical gallery
before comparing automatic and reviewed masks through pretrained completion.
The native-expert VM archive stays immutable while its role is assessed; it is
not the default next run. Existing main-app behavior is unchanged; no new upload,
commit or push occurred. The new Goal is active and unmet.

## Current user priority — 2 October 2026: practical covering removal and completion

The user clarified that hidden facial features need only be a plausible estimate.
The intended useful result detects face masks, sunglasses or strong lens glare,
replaces the covering with plausible facial content, and restores degraded visible
features. Completion/inpainting generates the hidden region; restoration addresses
observed blur/noise. Exact recovery of the person's actual hidden appearance is
not an acceptance requirement. Visual removal and coherent facial output are the
priority; preservation of visible appearance remains relevant.

The user confirmed four product decisions: remove sunglasses and strong lens
glare while keeping ordinary clear frames/transparent lenses; target any facial
covering, including hands, hair, scarves and objects; permit regeneration of a
small surrounding skin margin to remove the covering completely; and provide
automatic detection with optional manual mask correction. These decisions take
precedence over earlier assumptions about strictly exact covering boundaries.
The user also confirmed: keep hair unless it covers the face, and request a
less-covered image when nearly the entire face is hidden. Apply the hair rule to
strands obstructing facial features while preserving ordinary hairstyle, eyebrows
and normal facial hair. All six scope decisions are resolved. The existing
experimental interface supports manual mask correction, but broad automatic
real-covering performance has not been established.

Do not interpret approximate hidden anatomy as permission to silently relabel
old masks, change fixed experiments or claim a passed historical selection gate.
The confirmed scope and proposed output checks are recorded in
`C:\xampp\htdocs\YEAR 4\Testing\PRACTICAL_OUTPUT_SCOPE.md` (intended VM repository
counterpart `~/forensic-dgp/PRACTICAL_OUTPUT_SCOPE.md`; no new upload). Freeze a
new execution/acceptance protocol before any new evaluation or training. Assess
covering removal, plausible anatomy, seams and visible-feature preservation.
Record any intentional removal margin separately from raw detector accuracy.
Version any changed detection/selection policy and retain original measurements
and outcomes. No training, model promotion or packaged source/protocol modification
is part of this clarification.

Next: audit real-example coverage for the broader scope and compare automatic
versus reviewed removal regions before selecting VM work. A metadata-only check
finds68/83 training records with no covering-family tag; among the15 tagged records
there is one hand-over-mask case, but dedicated hair/scarf/other-object coverage
is not established. Untagged does not mean absent: inspect and classify examples
before making a coverage claim. No model forward was used for this check.
The prior StageA package remains
available and immutable; it targets native lens learning and does not demonstrate
all-covering readiness. Actual training remains VM-only. Windows workspace:
`C:\xampp\htdocs\YEAR 4\Testing\`; VM repository: `~/forensic-dgp/`.
The full goal remains active and unmet.

## Prepared milestone — 2 October 2026: bounded native-expert VM pilot packaged and audited

StageA is ready to transfer for a bounded GPU pilot, not application promotion.
The separate native-expert recipe starts from reflective42 with exact inherited
AdamW moments (step672/model882). Six epochs×56 batches give336 updates, final
global epoch48, model1218 updates and moment step1008. The fixed batch 8 contains
3real (2covered/1clear) and5fixtures (4covered/1clear), with50/50 supported domain
loss. Each existing fixture appears once per epoch; all83 real training cases
appear each epoch. The original parent stays separate and untouched.

This is a changed membership/objective/sampling experiment, not a causal test of
native additions or replay removal. StageA evaluates only83 real/280 fixture
training cases at source42 and the fixed final budget. Five predeclared fit/clear
checks and a fixed ten-row preview determine whether a separate StageB routing
protocol is justified. No development/test model scoring, learned selector or
generator/completion run is part of StageA. Original425-case gates remain exact.

The10,465,197-byte upload (about10MB) reuses297,268,325 bytes of existing model,
moments and fixture-cache assets on the VM. All281 archive files independently
verify:252 supported dataset files, original split snapshot, six lens maps and
17 required code files. The auditor reconstructs all336 deterministic batches,
preserves original full supervision and source171 unknown RGB, and verifies LF
checksum format. Twenty-one frozen/native/VM-boundary/archive contracts pass;
the pasted Bash commands pass syntax checking. Actual CUDA execution is unverified
until the user runs preflight. No actual local training, new model forwards,
optimizer construction or VM upload occurred in this preparation milestone.

Upload these two Windows files through Google Cloud SSH to `/home/janusdominic0/`:
`C:\xampp\htdocs\YEAR 4\Testing\outputs\native-expert-vm-code.tar.gz`
and `C:\xampp\htdocs\YEAR 4\Testing\outputs\native-expert-vm-code.tar.gz.sha256`.
Archive SHA256:
`9e2edb1c42591d01134514184b23041b7e8e6522d68b48beb6f86983b187e7c9`.
Exact commands and fixed checks:
`C:\xampp\htdocs\YEAR 4\Testing\NATIVE_EXPERT_VM.md`, packaged as
`~/forensic-dgp/native_expert_vm_bundle/NATIVE_EXPERT_VM.md`.
New VM working root: `~/forensic-dgp/native_expert_vm_bundle/`.
Read-only existing assets: `~/forensic-dgp/coverage_vm_bundle/`.
Existing environment activation: `~/forensic-dgp/feature_vm_bundle/.venv/bin/activate`.
No reinstall or git pull is needed for this self-contained transfer.

Prepared specification:
`C:\xampp\htdocs\YEAR 4\Testing\outputs\native_expert_protocol_v1\protocol.json`,
SHA256 `711a9def23d53cc32cb645395e8d9154571f5f88797dfbe6a404dadca06ebeab`;
its bundled VM counterpart is `~/forensic-dgp/native_expert_vm_bundle/inputs/native_expert_protocol.json`.
Independent archive audit:
`C:\xampp\htdocs\YEAR 4\Testing\outputs\native_expert_package_validation_v1\verification.json`,
SHA256 `395f11d7c1d409c546841f608997f5f4eb5d82c8c347b3385f1458c7cb216955`.
This local archive audit has no new VM execution counterpart.

Next: run GPU preflight and the bounded StageA pilot. Return
`~/forensic-dgp/native_expert_vm_bundle/native-expert-results.tar.gz` and its
LF checksum to `C:\xampp\htdocs\YEAR 4\Testing\outputs\`. Independently audit,
reproduce final masks and inspect the grid before any StageB or promotion.
Source42 remains ineligible on synthetic retention; the new expert is untrained.
No `best_detector.pth`, application or Phase 3 restoration replacement is selected.
No commit/push occurred. Actual training remains VM-only; the goal remains active
and unmet. Hidden facial features remain plausible estimates.

## Verified milestone — 2 October 2026: frozen detector complementarity audited

The training-only two-head diagnostic is complete and independently verified.
All1,946 saved masks and6,811 case/system recounts pass;353 reused masks equal
the audited VM pixels. Eleven targeted contracts pass. The fixed ten-row preview
was inspected. Both frozen heads preserve their tensor states; there were1,593
new CPU predictions, zero optimizer updates and zero held-out forward passes.

Fixed union/intersection fail the predeclared training safeguards. Both binary
heads miss8,205/21,853 reviewed lens pixels (37.55%); binary selection alone cannot
recover them. The conservative dominance oracle is target-informed and is not
optimal image routing. No learned router, completion generation or model promotion
occurred. The existing archive checksum matches again; no repeat download is needed.

Report: `C:\xampp\htdocs\YEAR 4\Testing\FROZEN_COMPLEMENTARITY_RESULTS.md`.
Evidence: `C:\xampp\htdocs\YEAR 4\Testing\outputs\frozen_complementarity_v1\`.
Independent verification:
`C:\xampp\htdocs\YEAR 4\Testing\outputs\frozen_complementarity_validation_v1\verification.json`,
SHA256 `449ddc0f85908f653099cb229bfdbce16d152df1eb0ec7a7f4c881bd18d8134b`.
These local diagnostics have no new VM execution counterpart. VM repository remains
`~/forensic-dgp/`; executed workspace remains `~/forensic-dgp/coverage_vm_bundle/`.

The separate bounded native-expert VM pilot is now packaged and audited; see
the current status above. No new VM training has started. StageA expert fit alone
cannot qualify the detector or generator. Original development gates,
generator/application and Phase 3 baseline remain unchanged. The goal remains
active and unmet.

## Verified milestone — 2 October 2026: supported real-mask dataset independently verified

The explicit supported reader and separate 115-record registry are complete.
Training contains83 images (51 covered/32 clear), validation25 (15/10), test7
(4/3). All105 previous image/mask bytes, active metadata and split order remain
unchanged. All73 original training tensor pairs exactly equal `ReviewedMasks`.
The ten reviewed native annotations add only training data; fourteen pending
sources remain unlabelled and absent. Labels are approximate assistant annotations.

All252 portable dataset files and raw source hashes pass an independent audit.
Every original record retains full support, including all validation/test pixels.
The mannequin keeps validation index8 and basename `new_covered_40.png` for
separate reporting. Native polygons, affine geometry, masks and support reconstruct;
source171's unknown lower band remains observed RGB context without supervision.
The new schema has `supported_records` and no `records` key: the actual legacy
loader rejects it. Future consumers must explicitly use image/mask/valid triples.
Seven reader/support and independent audit tests pass. Zero model forwards,
held-out forwards, optimizer updates or actual training occurred.

Report: `C:\xampp\htdocs\YEAR 4\Testing\SUPPORTED_REAL_DATA.md`.
Local registry: `C:\xampp\htdocs\YEAR 4\Testing\dataset\detector_supported_review_v1\manifest.json`,
SHA256 `860ea1dfba8fa442cab19bf3ab41f6a678109dcdb067c38ec0bd79ce43b51ace`.
Independent evidence:
`C:\xampp\htdocs\YEAR 4\Testing\outputs\supported_real_dataset_validation_v1\verification.json`,
SHA256 `308e5ca5b00bf8c52b36eba78605e41d188b754044307fd54d4f39edf16cbfe9`.
Tools: `supported_real_data.py`, `scripts/build_supported_real_dataset.py`,
`scripts/audit_supported_real_dataset.py` under the same Windows root.
The intended future repository counterpart is
`~/forensic-dgp/dataset/detector_supported_review_v1/`; no new VM copy/upload exists.
The executed VM bundle remains `~/forensic-dgp/coverage_vm_bundle/`.

Prior teacher/replay, projection, post-update retention and feature-head result
reports were reviewed. They do not justify another unchanged adaptation run.
Next: specify one materially different bounded repair with a training-only
feasibility check before fitting. This dataset is verified; a new GPU recipe is still
pending. No unchanged teacher/projection/retention run or weakened gate is
justified. Original gates, generator/application and Phase 3 restoration remain
unchanged. Training remains VM-only. No commit/push occurred; the full goal remains
active and unmet. Hidden facial features remain plausible estimates.

## Verified milestone — 2 October 2026: ten native annotations reviewed

Four covering annotations and six transparent controls are now reviewed for a
future supported detector dataset. Fourteen sources remain unlabelled, not empty
clear targets. The sources retain original Phase4 training membership and prior
4,210-reference overlap checks. Native source/overlay inspection uses no model
scores. These are approximate assistant pilot labels, not expert ground truth.

Uniform pixel-center geometry preserves source aspect ratio. Source171's unresolved
lower helmet/strap band excludes7,268 native pixels from supervision while retaining
their observed RGB as input context; only padding is neutral. All known positive
targets remain inside valid support. The first source374 boundary included cheek
skin; a native-grid refinement reduced1,174→1,011 native pixels in a separate
preserved version. Nine other proposals and all14 pending records remain unchanged.

Three new native-geometry contracts and two independent support-audit counterexamples
pass. The independent auditor checks all ten RGB inputs, native rasters, binary masks,
geometry, source/loss support and hashes; it confirms pending sources remain
unlabelled and unknown RGB is preserved. The final overlays and all six controls
were inspected. No model forward, optimizer update or actual training occurred.

Report: `C:\xampp\htdocs\YEAR 4\Testing\REAL_REFLECTION_PROPOSALS.md`.
Initial draft: `outputs/real_reflection_proposals_v4/`.
Reviewed version: `outputs/real_reflection_proposals_v4_refined/`, containing
`manifest.json`, `verification.json`, `annotation_decisions.json` and two previews.
These resolve under the Windows root above. Refined manifest SHA256:
`4d75620863314e7ff26e9c3a4abf3e28a99be46744f7f08730e9af697b9ffe91`;
annotation decisions SHA256:
`e6eef67417fa9a152cf7277969a5cae83b8949f5f1038cdadb8bd140c08186ea`.
The proposal/audit preserve their pre-acceptance flags; the later decision sidecar
records ten reviewed annotations. No original accepted dataset or held-out row is
edited. There is no new VM copy/upload; repository remains `~/forensic-dgp/` and
executed bundle `~/forensic-dgp/coverage_vm_bundle/`.

The explicit valid-support reader and separate opt-in registry are now verified;
see the current status and `SUPPORTED_REAL_DATA.md` above.
Legacy `ReviewedMasks` ignores valid support, and its manifest reader does not
enforce format/readiness metadata; existing runners must not consume these partial
annotations. After data checks, retention repair still needs a distinct hypothesis
before a bounded VM pilot. No new GPU recipe is ready.
Original gates, generator/application and Phase 3 restoration remain unchanged.
Training remains VM-only; no commit/push occurred. The full goal remains active
and unmet; hidden features remain plausible estimates.

## Verified milestone — 2 October 2026: real reflection sources qualified

The newly supplied `outputs/reflection-coverage-results.tar.gz.sha256` was rechecked
against the649,143,021-byte archive: SHA256
`a5714641943da67137a51866e4614b2b2841b2447145c816a51a39e7c52b032b` matches
the LF checksum and existing full audit/CPU reproduction. No repeat download or
unchanged training run is required.

The old27-source eyewear queue is now checked against the current105 reviewed
examples. Already-used sources13/46/49 were excluded. The remaining24 retain their
original Phase4 training membership. Fresh byte/decoded/DCT screening against
4,210 reviewed-source/crop, original-validation and benchmark references finds
zero overlap flags or flagged candidate pairs. Three new leakage counterexample
tests pass; native/input/artifact hashes verify after execution. Whole-image
screening does not certify identity separation or exclude every alternate crop.

All five native-aspect pages were inspected: eight covering proposals, ten
transparent-control proposals and six deferred boundaries. Source119 contains a
background face;106 includes foreground food;218 retains eye detail through tint;
310 has tinted lenses and a hand. Several proposed controls contain small glints
or tint that need close inspection. No new masks or empty labels are approved.
There were no model forwards, optimizer updates, split changes or VM uploads.

Report: `C:\xampp\htdocs\YEAR 4\Testing\REAL_REFLECTION_REVIEW.md`.
Local artifacts: `outputs/real_reflection_review_v4/{manifest,reference_signatures,visual_review}.json`
plus24 native review images and five pages. Manifest SHA256:
`76bcfe8061b331f09eaa78ae8f2c2e6a372e9da1042d5b4723454526b81d846c`.
These resolve under the Windows root above; no new VM counterpart exists.
VM repository remains `~/forensic-dgp/`, executed bundle `~/forensic-dgp/coverage_vm_bundle/`.

The subsequent ten native annotations are now reviewed; see the current status
above and `REAL_REFLECTION_PROPOSALS.md`. Fourteen sources remain unlabelled.
The explicit supported reader and separate opt-in registry are now verified. Preserve
original gates, held-out membership and baseline generator/application. Synthetic
retention still needs a distinct repair hypothesis; opaque lenses alone cannot
resolve small/partial glare transfer. No new VM training recipe is ready. Training
remains VM-only; the goal remains active and unmet. No commit/push occurred.

## Verified milestone — 2 October 2026: fixed crop diagnostic rejected

The fixed input-only face-crop diagnostic is complete. It used all73 real training
examples and280 reflection fixtures with two frozen CPU model states; no optimizer
updates or held-out forward passes occurred. There are171 crops,181 no-zoom
fallbacks and one invalid/missing-face fallback. The independent saved-mask audit
recounts all706 masks and353 source/target/affine mappings, checks original mask
hashes, exact fallback and outside-ROI preservation, and reproduces the logged
metrics and predeclared failed decision. Two new counterexample tests pass; the
eight geometry/inference and three protocol tests passed before execution.

Final reflective42 training IoU drops0.92536→0.90770; fixture IoU drops
0.80937→0.78964. Lens-only recovery drops1,369/2,031(67.41%)→1,018/2,031(50.12%).
Visible false positives increase; all26 real/56 fixture clear examples remain
empty. The fixed six-row preview was inspected. The two real lens crops enlarge
only about1.054×/1.032×; interpolation adds no native detail. SCRFD emitted static
output-shape metadata warnings at256 input; box correctness and warning cause
were not independently established. The executed runner checks unchanged model
state; the independent saved-mask auditor performs no model execution.

Report: `C:\xampp\htdocs\YEAR 4\Testing\FACE_CROP_RESULTS.md`.
Immutable specification: `C:\xampp\htdocs\YEAR 4\Testing\FACE_CROP_DIAGNOSTIC.md`.
Local protocol: `outputs/face_crop_protocol_v1/protocol.json`, SHA256
`de64663b607f34f65f105029d4862d29cc78e799c8b993ff10ad36d9280c31a3`.
Local evidence: `outputs/face_crop_diagnostic_v1/{results,cases,verification}.json`
and `preview.png`; verification SHA256:
`c670613b4830daa150f716ef35dad37ac2802f57237abeb285906f6365116261`.
Relative paths resolve under the Windows root above. Frozen model origins remain
`~/forensic-dgp/coverage_vm_bundle/outputs/reflection_coverage_vm/`; these CPU
artifacts have no new VM counterpart. No training, dependency reinstall, commit,
push or new VM upload occurred. Executed recipes/protocols stay unchanged.

Next: stop this crop line without another margin/confidence/threshold/epoch sweep.
Review additional real reflection sources and native covering boundaries in a new
training-only dataset version. The existing27-source eyewear queue has proposals,
not accepted pixel labels; exclude already-used reviewed sources before additions.
Preserve held-out membership and original real/synthetic gates. Retention repair
needs a distinct hypothesis. No new VM training recipe is ready; actual training
remains VM-only. Baseline generator/application and Phase 3 restoration are retained.
The subsequent real-source qualification is complete; see the current status
above. Ten later native annotations are accepted for the separate supported
dataset, not for generator training. The goal remains active and unmet; hidden
features remain plausible estimates.

## Verified milestone — 2 October 2026: reflection return; no eligible checkpoint

The649,143,021-byte reflection archive is now present locally:
`C:\xampp\htdocs\YEAR 4\Testing\outputs\reflection-coverage-results.tar.gz`.
VM origin:
`/home/janusdominic0/forensic-dgp/coverage_vm_bundle/reflection-coverage-results.tar.gz`.
Local SHA256:`a5714641943da67137a51866e4614b2b2841b2447145c816a51a39e7c52b032b`.
The downloaded VM-generated LF checksum now matches. No further checksum download
is required. Safe extraction verifies 4,517 file hashes and all 179 sent input/
provenance members. All 504 logged updates, exposures, losses, counters and original
selection decisions reconstruct. All 4,315 masks independently recount. Source/
final model and moment tensors, frozen statistics and expected steps verify:
each arm adds 252 updates, ending at model update 882 and optimizer step 672.

CPU reproduction compares all 4,315 masks. One pixel differs in reflective 36
synthetic case 359; all other saved masks match. All CPU/VM selection decisions
agree. Its numerical cause was not independently isolated. No local optimizer is
constructed or updated. Remote parent invariance remains an executed-code/log
claim because its VM tensors were not returned.

Both arms have zero eligible checkpoints. Control 42/reflective 42 real IoU is
0.81207/0.81462 and synthetic 0.93134/0.93292 versus required at least 0.97469.
Reflective 36 real IoU 0.83172 is also ineligible. Source 30 already fails retention.
Reflection training-fixture IoU rises to 0.80937, but every candidate misses all
491 real validation lens pixels. Two real training reflections have final
reflective lens-only recall 67.41% versus source 59.48%; this is training fit.

The fixed ten real rows, lens zoom, clean/degraded fixtures and fixed synthetic
sheets were inspected. Medical masks are fuller and clear rows stay empty, while
degraded irregular boundaries lose coverage. Final reflective loses 145,010 parent
true-positive pixels and recovers 20,268 parent misses. No end-to-end completion
improvement is established. Generator/application and Phase 3 baselines are retained.

Report: `C:\xampp\htdocs\YEAR 4\Testing\REFLECTION_COVERAGE_RESULTS.md`.
Extraction: `C:\xampp\htdocs\YEAR 4\Testing\outputs\downloaded_reflection_coverage\outputs\reflection_coverage_vm\`;
VM source: `~/forensic-dgp/coverage_vm_bundle/outputs/reflection_coverage_vm/`.
Local audits: `outputs/reflection_coverage_validation/{results,reproduction,members}.json`;
diagnosis/review: `outputs/reflection_coverage_analysis_v1/{results,visual_review}.json`.
These relative artifacts resolve under the Windows root above. Preliminary
inspection/readiness artifacts retain historical pending flags. Do not rerun
completed tools over their preserved outputs.

The local CPU reproduction and error analysis tools have five and four passing
contract tests respectively. Actual checkpoints were also executed on CPU.
Existing pinned modules were read through the authorized unsandboxed rerun after
the initial sandbox dependency read failed. No dependency reinstall, training,
git commit/push or new VM upload occurred. The sent/executed recipe is unchanged.

The subsequent fixed face-size diagnostic is now complete and rejected; see the
current status above and `FACE_CROP_RESULTS.md`. Additional real reflections need
a separate training-only dataset version and native mask review. Retention repair
needs a distinct predeclared hypothesis; another identical run or renamed teacher/
projection experiment is not justified. No new training recipe is ready. Actual
training stays on the VM. The goal still requires both original safeguards and
reviewed end-to-end improvement; it remains active and unmet. Hidden features
remain plausible estimates.

## Prepared milestone — 1 October 2026: reflection-coverage VM pilot packaged

The independent local return checker is also prepared:
`scripts/audit_reflection_coverage_results.py`. Nine counterexample/fixture tests
pass, and the sent179-file package independently verifies. Report:
`REFLECTION_COVERAGE_RETURN_AUDIT.md`. The result archive and checksum are not
present locally; there is no new GPU/model-quality evidence or live VM job handle.
This checker was added after packaging and requires no new VM upload. It does
not change the sent runner, code/data inventory or selection decisions. After
return, it audits504 updates and4,315 masks, creates the fixed ten-row grid and
records CPU prediction reproduction/visual review as still pending. Do not
infer eligibility or improvement from checker tests alone.

A bounded matched Track2 **detector** pilot is ready for the existing VM bundle.
It forks the audited epoch30 detector and exact step420 AdamW moments into clear
control and reflective arms,252 updates each. Both share the same sanitized core
replay, real training examples and original-parent consistency term. Only the
supplemental reflection slot differs. Original real/synthetic selection gates,
the400-case benchmark, generator and application baselines remain unchanged.
This is a prepared experiment, not a trained improvement or eligible checkpoint.

The additional fixed44 native-aspect source proposals were inspected;24 additions
join the original four. There are28 reviewed low-resolution uncovered bases,
14 per source pool, and280 immutable clean/degraded camera fixtures:224 partial
reflections plus56 clear controls. Uniform affine scaling preserves aspect ratio;
padding contributes no supervised loss. Degraded targets use declared conservative
effect/padding morphology. All280 cases hash/recount and all five lens/camera
sheets were inspected. Frames/textures remain simplified and real transfer is
unproven. Registry counts are28 paired,149 pending and3 unpaired candidates.

Eight uncertain sources `[67,78,107,118,121,122,160,177]` are excluded from both
arms' core exposures using49 registered replacements matching covered/clear and
camera strata. No cached image/target, source split or accepted real mask changed.
The two arms preserve safe/core real positions. The fixed twelve-epoch schedule
uses the old ten epochs followed by its first two as a declared tail. Every224
positive fixture and56 clear fixture appears; supplemental weight is0.25 and new
fixtures receive no parent teacher labels. Training metrics never select models.

The read-only CPU source forward validates finite core/control/reflective/padded
losses, unchanged model/parent tensors and92 exact optimizer states at step420.
No local optimizer was created or updated. Seventeen new tests plus37 existing
completion/geometry contracts pass:54 total. `source_check.json` binds executed
code and data. New CUDA execution is unverified; the VM preflight must pass first.

| Current file/artifact | Windows local | Linux VM |
| --- | --- | --- |
| Fixed recipe | `C:\xampp\htdocs\YEAR 4\Testing\REFLECTION_COVERAGE.md` | `~/forensic-dgp/coverage_vm_bundle/REFLECTION_COVERAGE.md` |
| Exact launch/download commands | `C:\xampp\htdocs\YEAR 4\Testing\REFLECTION_COVERAGE_VM.md` | `~/forensic-dgp/coverage_vm_bundle/REFLECTION_COVERAGE_VM.md` |
| Frozen data | `C:\xampp\htdocs\YEAR 4\Testing\outputs\reflection_coverage_data_v1\` | `~/forensic-dgp/coverage_vm_bundle/outputs/reflection_coverage_data_v1/` |
| Verified code archive | `C:\xampp\htdocs\YEAR 4\Testing\outputs\reflection-coverage-code.tar.gz` | Upload to `~/reflection-coverage-code.tar.gz` |
| Return archive | `C:\xampp\htdocs\YEAR 4\Testing\outputs\reflection-coverage-results.tar.gz` | `/home/janusdominic0/forensic-dgp/coverage_vm_bundle/reflection-coverage-results.tar.gz` |

The archive has179 files,62,664,483 compressed bytes; exact membership/bytes were
verified and the checksum uses LF. SHA256:
`4574929a1c54729ad078f4456dc4bf44cda65cc207ef28c1f83b5e1bbfebdf1a`.
The package preserves all four prior inventories. The initial private packaging
draft is retained in `outputs/reflection_coverage_package_draft1/`; a real-gate
description was corrected before release. The unchanged real gate checks IoU
improvement, visible false positives, empty masks and clear-case errors; real
missed fraction is reported without a separate gate. Synthetic missed fraction
remains constrained. The current archive contains the corrected specification.

Data manifest SHA256:
`6696cee3a3acf48049f2f121a2a6328625c4351f558deabb5475da7fe948baf9`.
Independent data audit SHA256:
`21a77f47d2cdea74b4d04e0aef7433664422a5c8a923d4ac05e5f9468016fbb8`.
Read-only source check SHA256:
`46265f573a72a19d853875e0bad9b2fd14b7eb45dc771614787beacf5e98ac8b`.
Preparation-stage `training_ready:false` stays preserved; later bounded recipe
readiness lives in the separate source check/inventory, not a quality claim.

Next: upload the archive and its `.sha256` to VM home, extract new files with
`tar --keep-old-files` into the existing bundle, run `--preflight`, then the fixed
VM pilot inside `tmux`. Source weights, saved moments, replay and pinned CUDA
packages already exist there; no new install is needed. These new files are not
pushed, and `git pull` alone does not install the isolated package. See the
current runbook above for exact commands. Preserve output if verification fails.

Both arms save global36/42; final counters are252 additional updates,882 lifetime
model updates and optimizer step672. The result archive includes all504 step
records and4,315 raw mask exports. Return it with its `.sha256` for independent
input/log/mask/checkpoint/moment auditing and CPU prediction reproduction.
Inspect the same ten real rows `[0,1,5,6,2,3,4,17,8,23]`, lens-only zoom and
training fixtures. Only a candidate passing unchanged gates advances to reviewed
end-to-end completion. An absent `best_detector.pth` is failed eligibility,
not script cancellation; no automatic promotion occurs.

Track1 retains Phase 3 `checkpoints/dgp_zamboanga_final.pth`; full Phase 5 ArcFace
identity training remains pending. Track2 retains its completion generator and
application baseline. All previously audited detector candidates fail synthetic
retention. Missing facial regions remain plausible estimates. The full project
goal is active and unmet, awaiting this external VM run/result and eventual
reviewed end-to-end improvement. Actual training stays on the VM.

## Prototype milestone — 1 October 2026: four qualified bases before the coverage pilot

`completion_data_v2.py` now provides an opt-in source-qualified path with uniform
scaling, explicit source support and eye-anchored partial reflection fixtures.
It leaves the existing training/benchmark data and application unchanged. Native
training source review admits four low-resolution prototypes (IDs0,4,101,102);
two likely intrinsic occlusions remain unpaired candidates and174 sources remain
pending. There are no new accepted real masks and no training enabled.

Twenty procedural cases cover clear frames, white patches, white streaks, scene
reflections and blue glare. An independent audit verifies123 prepared files and
zero visible pixels changed outside the known reflection targets. The two preview
grids and four native sources were inspected. Textures remain simplified; this
is not evidence of realistic transfer, improved model output or high-resolution
ground truth. Sixteen new tests pass;37 pass including completion/alignment
contracts. No local optimizer updates or model inference in this data prototype.

Report:`QUALIFIED_COMPLETION_DATA.md`. Windows artifacts:
`C:\xampp\htdocs\YEAR 4\Testing\outputs\qualified_reflection_v1\`.
Corresponding VM artifact layout, when transferred:
`~/forensic-dgp/coverage_vm_bundle/outputs/qualified_reflection_v1/`.
Registry SHA256:`9dd48c58bd17a1f95588fb745c4cfda06ed15e9932e0e08d17dcdf524cd19116`;
audit SHA256:`04549c2415f66a8dfea7b376a3866c75f1b3e63883da84ded987a4cb0289ca9f`.
`visual_review.json` binds the assistant review to both. Existing prepared inputs,
renderer/builder and saved evidence are hash-bound; use a new version for changes.

Next: qualify more native training sources and freeze reflection coverage before
implementing a matched VM pilot with valid-support-aware loss reductions. Keep
the400-case benchmark, original gates, teacher and generator/application baseline.
No new GPU recipe/package is ready. Actual training remains VM-only. All audited
detector candidates still fail synthetic retention; there is no eligible
`best_detector.pth`. Track1's Phase 3 baseline and Track2's generator remain;
full Phase 5 identity training and reviewed end-to-end output improvement remain
pending. The full goal is active and unmet.

## Audited result — 1 October 2026: matched loss return; no eligible detector

The matched VM archive is present and both independent audit passes are complete.
Both arms execute210 new updates from the same epoch30 model/AdamW moments.
All420 logged steps, loss terms, counters, code hashes and gates reconstruct;
all2,915 saved masks recount. Strict CPU loading verifies all four states and92
frozen tensors each. Source moments are step420; both final snapshots bind to
their respective epoch40 model at step630/840 lifetime model updates. CPU
compares all2,915 masks with one different pixel (control35, synthetic case394),
and identical selection. Both final states' saved masks are exact. No local
optimizer updates and no generator/application/restoration promotion.

Global40 control/focus real IoU is0.81345/0.82323 and synthetic0.93892/0.92839,
below original synthetic0.97469. Each has1/80 false-positive clear synthetic
cases versus required0. Human-only IoU is0.81562/0.82248; mannequin separately
0.79109/0.83089. Every candidate passes the original real gate and fails retention;
neither arm has`best_detector.pth`. Training reflection recall improves to84.93%
control and90.50% focus from source30's59.48%, but the491-pixel validation
reflection is entirely missed. Full result:`FACE_OCCLUSION_FOCUS_RESULTS.md`.

Both final states were inferred on638 pinned training replay cases, batch 8,
tensors unchanged. Control/focus training IoU0.95688/0.94281 versus source30
0.93217 and original parent0.96904. Source counts reused after provenance,
target/summary and saved-mask checks. Weighting reduces visible false pixels
but misses more synthetic covered pixels; it does not establish a global benefit.
Thirteen new checker/reproduction/review tests pass;30 with previous audit tests.

The declared ten-row visual cohort is reused, including mannequin8 and glare23:
`[0,1,5,6,2,3,4,17,8,23]`. The first recount grid's assumed mannequin index22
was a clear case; it remains preserved. The corrected review grid, reflection
zoom and fixed training failure grid were inspected. Scores were unaffected.

All180 unique source photos used for262 clear training replay cases were
contact-screened in source-ID order; eight also at native resolution. Two likely
intrinsic occlusions (source160 dark lenses and177 scene reflection) have
procedural zero masks; they need source-only mask review. Four other eyewear
cases remain ambiguous. A source33 thumbnail suspicion was native skin
highlights, not lens glare. No annotation, source, cache, split, teacher or gate
changed. The90 Asian-source photos include52 nonsquare images; the90 FFHQ
photos are square. `CompletionDataset` currently resizes them directly to a
square, changing nonsquare geometry. This is a measured training subset,
not a whole-dataset count or proven causal explanation.

Windows archive:
`C:\xampp\htdocs\YEAR 4\Testing\outputs\face-occlusion-focus-results.tar.gz`.
VM source:
`/home/janusdominic0/forensic-dgp/coverage_vm_bundle/face-occlusion-focus-results.tar.gz`.
586,330,604 bytes; SHA256:
`1cc12eec33ec30bc97acb483f9c1806a20fb353b817a04938738b6cf365f3bdd`.
Local extracted model root:
`C:\xampp\htdocs\YEAR 4\Testing\outputs\downloaded_face_occlusion_focus\outputs\face_occlusion_focus_vm\`;
VM models:`~/forensic-dgp/coverage_vm_bundle/outputs/face_occlusion_focus_vm/`.
Audits:`outputs/face_occlusion_focus_validation/{results,reproduction,members}.json`;
review:`outputs/face_occlusion_focus_review/`; fit:`outputs/face_occlusion_focus_fit/`;
source screen:`outputs/clear_replay_scope_v1/`. These diagnostic outputs are local.
The L4 reports103.03 seconds and1,163,518,976 peak allocated CUDA bytes.
The executed package/runner/specification and previous inputs remain immutable;
default output directories cannot be rerun or overwritten.

Next: implement versioned clean-source qualification, geometry preservation and
targeted reflection coverage before another matched VM recipe. Separate paired
uncovered completion bases from unpaired real occlusion images; review proposed
masks and source-only augmentation previews first. Existing benchmarks/gates
remain unchanged. No new GPU package is ready and another identical run is
unjustified. Only an eligible detector advances to reviewed end-to-end completion.
Track1 Phase 3 baseline remains`checkpoints/dgp_zamboanga_final.pth`; full Phase 5
identity training is pending. Track2 generator/application baselines are retained.
External pretraining overlap, label adjudication and final generalization remain
unresolved; the full improvement goal is active and unmet. Older sections are
historical milestones, not instructions to repeat completed runs.

## Preparation record — 1 October 2026: matched loss pilot ready for VM execution

The next bounded detector experiment is packaged and locally verified. New CUDA
execution and improved output are not established. Actual training remains
VM-only. Local Windows root:`C:\xampp\htdocs\YEAR 4\Testing\`; VM repository:
`~/forensic-dgp/`; experiment execution root:`~/forensic-dgp/coverage_vm_bundle/`.

The audited epoch30 detector and its exact saved AdamW moments initialize both
arms independently. Compare the unchanged loss (`control`) with the same loss
plus0.25 region-balanced BCE (`component_focus`). Each arm has10 additional
epochs/210 updates,420 total. Check global epochs35/40:735/840 cumulative model
updates and525/630 optimizer lifetime steps. Architecture, data, batch order,
rates, consistency teacher, original real/synthetic gates and probability
threshold0.5 are fixed. No optimizer reset, adaptive budget or automatic promotion.

Training targets remain73 reviewed real cases and638 frozen replay cases.
The accepted V3-minus-V2 reflection additions in training cases15/46 are separated
from remaining foreground for auxiliary weighting only. One686-pixel glare touches
an11,381-pixel medical mask; this separation stops its weight being absorbed by
the larger region. The other two reflection regions contain822/523 pixels.
No label union, image, split or held-out training membership changes.

Seventeen new fixture tests and six relevant previous tests pass (23 total).
Three runbook Bash blocks pass syntax checking; Python modules compile.
All711 target maps and92 saved optimizer states /14,328,209 parameter elements
are verified. One read-only CPU source batch produces finite supervised/auxiliary
loss0.03546993/0.05711475 with every tensor unchanged and zero local optimizer
updates. Linux/CUDA preflight additionally restores exact moments with zero
steps; full VM execution requires audited source/baseline metrics to reproduce.

Upload these two Windows files to VM home:

- `C:\xampp\htdocs\YEAR 4\Testing\outputs\face-occlusion-focus-code.tar.gz`
- `C:\xampp\htdocs\YEAR 4\Testing\outputs\face-occlusion-focus-code.tar.gz.sha256`

The11-member package is75,230 bytes, all members verified, checksumLF.
Archive SHA256:`3c0834de0c89b5189170a9327a6bfb3112a8c11f03100229122733e73036892c`.
Inventory SHA256:`91b23aaa05c06c377a5c50c3c0bf53a5008e7ec7bd0d5ae657684d81b0b05471`.
New map registration:`outputs/face_occlusion_focus_bundle_v1/target_maps_v2.json`,
SHA256:`7f5144e1d9f80ba9aa92a78d4dac40f20a6a9521581dfd721dcf44c40e7ac1f1`.
Priority registration:`outputs/face_occlusion_focus_bundle_v1/priority_regions.json`,
SHA256:`c3f67d52406f9334a602b915e3852f1059f965dffb6a526746282a1b9e78aea6`.
All registration paths are relative to each workspace/bundle root above.
The earlier local`target_maps.json` is an unpackaged draft; it remains preserved.
Every previous executed input/inventory is preserved. New code/docs are
uncommitted and unpushed; a repository`git pull` does not install this isolated
bundle. No new weight/data upload or dependency installation is needed.

Recipe:`FACE_OCCLUSION_FOCUS.md`; exact SSH/tmux/preflight/run/download commands:
`FACE_OCCLUSION_FOCUS_VM.md`, at Windows root and in the new VM bundle.
Candidate directories are`outputs/face_occlusion_focus_vm/{control,component_focus}/`.
`best_detector.pth` is saved within an arm only when both unchanged gates pass.

Next: execute the matched VM pilot and return
`/home/janusdominic0/forensic-dgp/coverage_vm_bundle/face-occlusion-focus-results.tar.gz`
and its`.sha256` to`C:\xampp\htdocs\YEAR 4\Testing\outputs\`.
Audit420 step logs,2,915 masks, source/moment/final bindings, frozen states and
unchanged selection. Compare matched checkpoints and inspect the ten-row grid
and reflection zoom before an eligible detector advances to end-to-end completion.
Track1 Phase 3`checkpoints/dgp_zamboanga_final.pth` is retained; full Phase 5 identity
training remains pending. Track2 completion generator/application remain at their
current baselines. External pretraining overlap, approximate polygons and tiny
reflection sample size limit generalization claims; the broader improvement goal
remains unmet. Earlier sections are historical milestones.

## Audit record — 1 October 2026: continuation audited; synthetic guard still fails

The continuation archive is received and both audit passes completed. The fixed
420 additional updates reach630 cumulative model updates. Epoch30 synthetic IoU
is0.92039, improving source10's0.84378 but below original parent0.97469. Real IoU
is0.81739 versus source0.82283; human-only0.81997 and mannequin0.79113. Every
candidate passes the original real gate and fails synthetic retention. No
`best_detector.pth` is selected and no application/generator/restoration baseline
is promoted. Full report:`FACE_OCCLUSION_CONTINUATION_RESULTS.md`.

Executed source, package/inventories, all420 batches, counters and selection
match. All2,915 masks independently recount. Strict loading verifies four model
states and92 unchanged frozen tensors each. Final AdamW snapshot has92 correct
parameter states covering14,328,209 elements, step420 and epoch30 hash binding.
CPU reproduces2,915 masks with three pixel differences and identical selection;
epoch30 masks are all exact. Remote parent invariance remains an executed-code
check/log claim; its VM tensors were not returned. No local optimizer updates.

Training replay diagnostic:638 pinned cached cases, batch 8, unchanged model state.
Epoch30 training IoU0.93217 versus parent0.96904; weighted exposure0.93427.
Degraded irregular training/validation IoU0.84717/0.81573. Final real training
IoU0.91719. Original/source10 replay counts reused after integrity/count checks.
The fixed training failure grid and clear-error sources were visually inspected.
Fit still limits the detector; no causal capacity/loss/dose conclusion is proven.

Reflection-only recount excludes the large mouth mask in one training image.
Source10/epoch30 training reflection recall is34.86%/59.48% over2,031 reviewed
V3-minus-V2 pixels. Per-case final recall is60.97% and56.56%, after27/30 cumulative
exposures. The491-pixel validation reflection remains entirely missed at every
checkpoint. Whole-mask glare-image IoU0.84064 must not be presented as lens-only
quality. The three-row reflection zoom was inspected; six new diagnostic tests
pass. Original labels/splits and selection thresholds are unchanged; no test
predictions were scored.

Windows archive:
`C:\xampp\htdocs\YEAR 4\Testing\outputs\face-occlusion-continuation-results.tar.gz`.
VM source:
`/home/janusdominic0/forensic-dgp/coverage_vm_bundle/face-occlusion-continuation-results.tar.gz`.
Archive SHA256:`035777e5aa22ca97c12b07ce96fa2abb076eea5806c408dd7f8d6d6f2bf84028`;
373,872,188 bytes. Model/optimizer root on VM:
`~/forensic-dgp/coverage_vm_bundle/outputs/face_occlusion_continuation_vm/`;
local extracted counterpart:
`C:\xampp\htdocs\YEAR 4\Testing\outputs\downloaded_face_occlusion_continuation\outputs\face_occlusion_continuation_vm\`.
Audits:`outputs/face_occlusion_continuation_validation/{results,reproduction}.json`.
Diagnostics:`outputs/face_occlusion_continuation_fit/` and
`outputs/face_occlusion_continuation_glare/`. All diagnostic outputs are local.
L4 reports99.16 seconds,1,047,887,872 peak allocated CUDA bytes for the run.

Next: prepare a single-change matched VM experiment for small reflection/irregular
fit and clear-face specificity, using equally verified model/optimizer starts.
Pre-register its control, budget, checks and stopping point; a new training
recipe/package is not ready yet. Only an eligible detector advances to reviewed
end-to-end completion. Track1 Phase 3 baseline is retained; full Phase 5 identity
training remains pending. External FFHQ overlap and final generalization remain
unverified. The full goal is active and unmet. Earlier entries are historical.

## Preparation record — 30 September 2026: local audit ready while awaiting return

At this preparation milestone, the expected continuation archive was absent at
`C:\xampp\htdocs\YEAR 4\Testing\outputs\face-occlusion-continuation-results.tar.gz`.
No configured SSH connection or VM execution evidence was available at that
milestone. Actual training remains VM-only. The already sent
seven-member continuation package was reverified, with unchanged SHA256
`93bab8a34d23a48cd8f9232dcbe1cd65c91bd024b2cd6e8480d76e863f82e3f7`.
Use `FACE_OCCLUSION_CONTINUATION_VM.md` for its exact SSH commands; those source
files remain immutable and have not been pushed.

Local return checker:`scripts/audit_face_occlusion_continuation_results.py`.
Seventeen checker tests pass;25 pass with previous return-audit/runner tests.
Small archive fixtures validate safe extraction and member hashes. No new model
inference or optimizer update occurred during this preparation; model-quality
evidence awaits the actual return. Full local instructions and proof limits are
in`FACE_OCCLUSION_CONTINUATION_AUDIT.md`.

The checker verifies420 additional steps/630 cumulative model updates, source
bytes, global checkpoints11/15/20/30, all2,915 saved masks and unchanged gates.
Its second pass verifies frozen tensors, preserved original pilot metadata,
new continuation metadata, final optimizer moments/epoch30 binding and CPU
checkpoint-to-mask reproduction. It records CPU/VM differences explicitly.
The ten-row preview includes glare and the mannequin. The expected optimizer
snapshot exceeds the earlier auditor's64 MiB file bound; the new checker permits
128 MiB for that file and64 MiB for model checkpoints.

Next: return
`/home/janusdominic0/forensic-dgp/coverage_vm_bundle/face-occlusion-continuation-results.tar.gz`
to the Windows path above. Run the local archive audit, then its`--reproduce`
pass, inspect the grid/glare, and advance only an eligible detector to reviewed
end-to-end completion. Current restoration/generator/application baselines are
retained. External FFHQ overlap/generalization remains unresolved; the full goal
is open. No automatic extra training or promotion.

## Preparation record — 30 September 2026: bounded pretrained weight continuation ready

Replay diagnostic completed without optimization. Pretrained epoch10 training
IoU0.84827 versus validation0.84378; original parent0.96904/0.97469. Schedule
weighting still gives0.84747, so the deficit exists on seen examples. Both source
pools/all covered types regress; degraded irregular training/validation IoU
is0.73761/0.71069. Three diagnostic tests pass. Parent confusion counts reused
after cache/source/target checks; both trained detectors inferred on638 cases,
states unchanged. Failure-ranked training preview inspected. Evidence:
`FACE_OCCLUSION_REPLAY_RESULTS.md`, `outputs/face_occlusion_replay_fit/results.json`.

One fixed VM follow-up is packaged:20 additional epochs/420 fresh AdamW updates
from pretrained epoch10 SHA256
`cdd1752bce8e4a087ce2aac5af73ba81b6316ac9118e47f12acd7a24fcb62669`.
Architecture/data/masks/source membership/loss/rates/balance/gates are unchanged.
This is weight continuation with an optimizer reset; original optimizer moments
were unavailable. Two independent copies of the existing ten-epoch schedule;
candidate global epochs11,15,20,30 (231,315,420,630 cumulative model updates).
Four runner tests pass: Linux CUDA guard before work, correct source, independent
schedule copies and distinct new/cumulative counters. Actual new GPU execution
remains unverified. Source metrics must reproduce before optimization.

Windows upload files:
`C:\xampp\htdocs\YEAR 4\Testing\outputs\face-occlusion-continuation-code.tar.gz`
and its`.sha256` file. Seven members verified;11,145 bytes; LF checksum; SHA256:
`93bab8a34d23a48cd8f9232dcbe1cd65c91bd024b2cd6e8480d76e863f82e3f7`.
These new files are not pushed. `FACE_OCCLUSION_CONTINUATION_VM.md` has exact
SSH/tmux/preflight/run/download commands for`~/forensic-dgp/coverage_vm_bundle/`.
Reuse existing dependencies and source/cache; no package reinstall or new
weight upload is required. `FACE_OCCLUSION_CONTINUATION.md` fixes the hypothesis,
budget and stopping/selection/export policy.

Next: run the bounded VM experiment and return
`/home/janusdominic0/forensic-dgp/coverage_vm_bundle/face-occlusion-continuation-results.tar.gz`
to`C:\xampp\htdocs\YEAR 4\Testing\outputs\face-occlusion-continuation-results.tar.gz`.
Audit420 new steps, source/parent/candidate state and masks, all original gates
and final optimizer binding. Only an eligible detector advances to reviewed
end-to-end completion. Glare is still missed and must be addressed explicitly;
external FFHQ pretraining overlap is unknown. Track1 Phase 3/Track2 generator and
application baseline remain unchanged. No automatic extra epoch or promotion.
The full goal remains open; earlier entries below are historical milestones.

## Current result — 30 September 2026: pretrained detector learns real masks

The returned direct-occlusion pilot is audited. Pretrained epoch10 real IoU
is0.82283 versus original parent0.06065 and matched random control0.59269;
real clear false-positive cases are0/10 versus1/10 and10/10 respectively.
Human-only pretrained IoU0.82201; mannequin reported separately at0.83106.
Pretrained real training IoU0.85042. Both pretrained epochs5/10 pass the real
gate, but every candidate fails synthetic retention. Pretrained epoch10 synthetic
IoU0.84378 versus parent0.97469; missed fraction0.10060 versus0.01648. The one
validation glare case remains missed. No `best_detector.pth` is selected and no
restoration/generator/application baseline is promoted.

Independent checks: executed source equals the sent package; both210-update
logs match the exact fixed schedule; all3,413 saved masks recount to logged
scores and unchanged gates. Six checkpoint states/metadata are inspected;
92 frozen head/BatchNorm tensors per state are unchanged and encoder/decoder/
new-head tensors change. Initial heads match. CPU inference compares all3,413
masks: every real validation and pretrained synthetic validation mask is exact;
five pixels differ in other synthetic/training predictions, without changing
selection. No local optimization. The ten-row mask grid is inspected. Actual
end-to-end completion improvement remains unverified.

Returned local archive:
`C:\xampp\htdocs\YEAR 4\Testing\outputs\face-occlusion-results.tar.gz`;
Linux VM source:
`~/forensic-dgp/coverage_vm_bundle/face-occlusion-results.tar.gz`.
SHA256:`6a27d1685717e941f691235e3ee4489f3ecd304712ce39fce046b0663567b3c7`.
Extracted states:`outputs/downloaded_face_occlusion/outputs/face_occlusion_pilot_vm/`.
Audits:`outputs/face_occlusion_validation/{results,reproduction}.json`.
Full report:`FACE_OCCLUSION_RESULTS.md`; checker:`scripts/audit_face_occlusion_results.py`
(four tests pass). L4 reports112.13 seconds, approximately1.11GB peak allocated
CUDA memory for this420-update pilot. Remote parent invariance is an executed
check/log claim; its tensors were not returned. External FFHQ pretraining overlap
remains unresolved, so these reused development results are not final accuracy.

Next: diagnose synthetic error strata and638-case replay training fit before
specifying a bounded VM follow-up using the promising pretrained representation.
Keep unchanged real/synthetic guards and require reviewed end-to-end completion
before promotion. Track1 Phase 3 restoration remains retained; full Phase 5 identity
run is pending. Entries below are historical preparation/results.

## Preparation record — 30 September 2026: direct occlusion pilot packaged

Track 1 restoration still retains `checkpoints/dgp_zamboanga_final.pth` as the
Phase 3 application baseline; Phase 5 identity loss awaits its full GPU run.
Track 2 now has a separate ResNet18 U-Net with a new explicit covered-region
head. The published FaceExtraction encoder/decoder and a matched random control
are fully trainable; their new heads are identical. This is an unproven
initialization experiment, not a promoted completion model. Existing completion
parent/generator, labels, cached replay, validation/test membership and gates
are preserved. External FaceExtraction FFHQ pretraining overlap is unresolved.

Twelve local adapter/VM-guard/runtime/package tests pass; CPU initialization and
reload forwards, five pinned imports and Bash syntax are verified. Zero local
model optimizer updates. CUDA compatibility and quality improvements remain
unverified. The fixed pilot uses73 real training labels and638 cached replay
inputs,210 updates per arm,420 total. Candidates are evaluated only at
epochs1,5,10 against the same original parent. BatchNorm statistics/reference
heads stay fixed; both real and synthetic gates are required for metric eligibility.
No automatic deployment follows a selected `best_detector.pth`.

Ready upload files under the Windows root
`C:\xampp\htdocs\YEAR 4\Testing\outputs\`:
`face-occlusion-vm-code.tar.gz` and `face-occlusion-vm-code.tar.gz.sha256`.
Archive size106,196,726 bytes;16 members independently hash-verified after build;
checksum uses LF. SHA256:
`3af2f1b1fd72dd4506c4d1ca4634ab5d5721d1445c043987989522ef421ffea0`.
The new inventory is `outputs/face_occlusion_bundle_v1/inventory.json`.
These files have not been committed/pushed; upload the package for this run.

Use the existing Linux VM workspace `~/forensic-dgp/coverage_vm_bundle/`.
`FACE_OCCLUSION_VM.md` contains exact upload/extraction/tmux/setup/preflight/run
commands; `FACE_OCCLUSION_PILOT.md` defines the immutable comparison. Setup reuses
CUDA Torch and installs only five pinned packages into a separate target with
`--no-deps`. Original coverage inventory and replay cache must still exist.
The runner refuses existing outputs, verifies old/new input hashes, records
all batch indices and candidate masks, and exports elapsed time/peak CUDA memory.
Three checkpoints per arm total approximately345MB before compression.

Next: run the CUDA preflight and bounded comparison on the VM, then return
`/home/janusdominic0/forensic-dgp/coverage_vm_bundle/face-occlusion-results.tar.gz`
to `C:\xampp\htdocs\YEAR 4\Testing\outputs\face-occlusion-results.tar.gz`.
Independently audit420 updates and six candidates, recount real/synthetic masks,
reproduce checkpoint inference, verify frozen state/parent invariance and review
human/mannequin/glare cases. Only a candidate passing both unchanged gates
advances to the ten-row end-to-end completion review. The overall goal remains
open. This preparation record is superseded by the audited result above.

## Latest compatibility result — 30 September 2026: FaceExtraction loads strictly

Full author checkpoint loads with SMP0.5.0, known module-prefix removal only.
Eight fixed training-only images inferred on CPU, finite256px visible-face outputs;
preview inspected. No model training. `FACE_EXTRACTION_PROBE.md` records dependencies,
limitations and evidence. Upper-face/mask separation is visible, but background
and glasses are excluded too; complement is not a valid direct occlusion mask.
Next: separate pretrained encoder/decoder adapter with a new explicit occlusion
head and matched initialization control, before any bounded VM fitting. FFHQ
pretraining overlap remains unresolved. No application/generator promotion.

## Latest asset audit — 30 September 2026: FaceExtraction weights available

Author source pinned at e75d4a83a696bd7379128319244ef6e5e7885fc8;57.4MB checkpoint
downloaded and tensor-only inspected, all tensors finite. See
`FACE_EXTRACTION_ASSET_AUDIT.md` for hashes and exact labels. Source confirms
visible-face target=parsed support minus occluder alpha; simple inversion is
invalid. FFHQ pretraining overlap remains unresolved. No datasets imported.
Next: isolated strict-load inference adapter and fixed training-only visible-face
diagnostic, before any direct-occlusion adaptation proposal. Required segmentation
library is absent locally; no dependencies installed and no training occurred.

## Latest research — 30 September 2026: face-specific pretraining target audit

Reviewed author sources for FaceOcc/FaceExtraction, NatOcc/RandOcc, S3POT and
SegFormer. `DETECTOR_PRETRAINING_REVIEW.md` records evidence and next steps.
FaceExtraction publishes a pretrained ResNet18 U-Net but predicts visible face;
inversion would incorrectly include background. NatOcc labels include transparent
glasses, requiring mapping to our glare policy. S3POT is distinct from prior
frozen SAM2 heads but its trained adapter availability is not yet verified.
Next: pin and inspect FaceExtraction code/checkpoint/label construction and
source overlap before deciding direct-occlusion adaptation. No training/package,
new model promotion, split changes or automatic external dataset import.

## Latest decision — 30 September 2026: close constrained-pilot branch

Ordinary epoch1 inferred on identical73 real training inputs: IoU0.07567 versus
parent0.07121 and constrained0.06605. Both adapted candidates miss over91% of
covered pixels. Ordinary takes21 full-rate updates; constrained takes18 accepted
updates, six at reduced rates. Not matched accepted dose/compute. Ordinary
epoch1 still fails synthetic missed-fraction guard; no model qualifies.
See `RETENTION_EPOCH1_COMPARISON.md`. No additional constrained training proposed.
Next: consolidate already tested representations/data recipes and research a
materially different segmentation initialization/pretraining strategy using
primary sources, without repeating SAM2 frozen-head or optimizer sweeps. Keep
automatic detection and end-to-end completion requirements intact. No promotion.

## Latest diagnostic — 30 September 2026: constrained pilot lacks training fit

Parent/retained inference on all73 real training images completed, zero updates.
Training IoU0.07121 ->0.06605; empty covered cases14/47 ->16/47 despite supervised
loss2.90742 ->2.77760. The62 images present in accepted batches also regress.
All25 real validation masks reproduced exactly from the returned checkpoint.
See `RETENTION_FIT_RESULTS.md` and `outputs/retention_fit/results.json`.
Next: inference-only comparison with archived ordinary epoch1 on identical73
training inputs to separate short-budget behavior from constraint effects;
explicitly report21 ordinary versus18 retained accepted updates. No VM rerun,
promotion, automatic extension or gate change.

## Latest result — 30 September 2026: retention pilot rejected

VM archive received.21 scheduled batches,18 accepted updates,42 trials; source,
schedule, trial acceptance/counters and final hash verified. All425 masks and
human/mannequin/glare metrics independently recounted. Initial tensors equal
parent; generator unchanged. Real IoU0.06065 ->0.05875, synthetic0.97469 ->0.97464;
both gates fail. Ten-row preview shows omitted/fragmented coverings. No promotion.
See `RETENTION_RESULTS.md` and `outputs/retention_validation/results.json`.
Next: inference-only final-checkpoint reproduction on25 real validation inputs
and parent-versus-final real training fit to distinguish absent adaptation from
poor transfer. No automatic longer run or weakened gates. Local result:
`C:\xampp\htdocs\YEAR 4\Testing\outputs\retention-results.tar.gz`; VM source:
`~/forensic-dgp/coverage_vm_bundle/outputs/retention_training_vm/`.

## Latest preparation — 30 September 2026: return-log audit ready

`scripts/audit_retention_logs.py` reads the returned tar without extraction or
checkpoint deserialization. It checks executed source against the sent bundle,
source/checkpoint hashes, fixed schedule, separate class ceilings, trial order,
counters and stopping rule. Three log-fixture tests pass, including deliberately
invalid acceptance and cross-class compensation. Actual archive validation is
pending: `outputs/retention-results.tar.gz` is not present locally. This checker
does not prove optimizer rollback, reproduce replay inference or recount masks;
those remain return-artifact checks. The sent VM bundle is unchanged.
Next: receive VM results, run the checker, independently verify final masks and
checkpoint state, then inspect completion only if the candidate qualifies.

## Latest VM preparation — 30 September 2026: bounded retention pilot ready

`scripts/train_retention_vm.py`, `coverage_retention.py` and fixed protocol are
packaged in `outputs/retention-code.tar.gz` with an LF checksum. Nine local tests
pass: rollback, scaled retry equivalence, guard, weighted group losses, frozen
replay membership and rejection-streak reset. No model fitting locally; GPU
execution remains unverified. `RETENTION_VM.md` contains exact upload/tmux/SSH
commands for `~/forensic-dgp/coverage_vm_bundle/`. No git push has occurred.
Pilot stops at21 scheduled batches or3 fully rejected batches; outputs are
preserved and archived even on a constraint-driven early stop. Await
`C:\xampp\htdocs\YEAR 4\Testing\outputs\retention-results.tar.gz` for recount,
acceptance/state audit and visual review. No baseline/application promotion.

## Latest implementation — 30 September 2026: transactional retention helper

`coverage_retention.py` adds bounded LR trials with full AdamW rollback on
rejection/exception. Five scalar-fixture tests pass, including half-rate equivalence
and exact restoration of moments, buffers, gradients, modes and Torch RNG.
No local model training. `COVERAGE_RETENTION_PROTOCOL.md` fixes a21-batch VM-only
feasibility pilot, all638 training replay cases per trial, separate covered/clear
parent ceilings plus1e-6 tolerance, four LR factors and stop after3 fully rejected
batches. Validation gates remain unchanged. Runner/package not yet ready.
Next: integrate/test/package the VM runner and provide SSH commands. No promotion.

## Latest result — 30 September 2026: aggregate loss hides replay regression

Completed inference-only loss audit of parent/ordinary/projected checkpoints on
73 real plus638 cached replay cases, weighted by all1,680 scheduled exposures.
Mixed loss1.22272 ->0.47252/0.47222, but replay supervised loss0.03797
->0.07758/0.06859. Both covered and clear replay groups worsen. This is an
objective tradeoff, not solely a binary-threshold discrepancy. No candidate is
promoted. See `REPLAY_LOSS_RESULTS.md` and `outputs/replay_loss_audit/summary.json`.
The model-free summary verifies all711 indices, exposure counts, protocol and
real-manifest hashes; all three inference arms completed. Zero optimizer updates.

Next: specify/test one VM-only pilot with explicit, separate training-replay
retention constraints on actual candidate updates and correct optimizer rollback.
Fix its tolerances, retry limit and compute budget before execution; unchanged
validation gates still apply. This pilot is proposed, not packaged or started.
No further identical ordinary/projection run, local training or gate relaxation.
Local root `C:\xampp\htdocs\YEAR 4\Testing\`; VM root
`~/forensic-dgp/coverage_vm_bundle/`. Earlier entries below record historical steps.

## Latest diagnostic — 30 September 2026: forgetting on seen replay cases

All638 cached training replay inputs inferred under parent/ordinary/projected.
IoU0.96904/0.94926/0.95078; clear false-positive cases2/262,11/262,9/262.
All eight covered type/degradation groups regress relative to parent, so the
failure is not solely unseen synthetic variations. See `REPLAY_FIT_RESULTS.md`.
No fitting. Next: inference-only supervised/teacher loss decomposition on the
same inputs to check surrogate-objective versus binary-metric tradeoffs before
any new training proposal. No model/application changes.

## Latest analysis — 30 September 2026: step magnitudes weaken simple conflict claim

All420 logged steps summarized in `PROJECTION_STEP_MAGNITUDES.md`. Positive
first-order replay-change share3.09% ordinary versus1.59% projected; projection
removed median14.86% real-gradient norm on113 active steps. These are changing-
batch derivative aggregates, not cumulative replay loss. Do not escalate to
actual-step projection based on sign counts alone. Next: inference-only fit on
638 cached replay training cases versus synthetic validation, by covering type
and degradation, before choosing another training hypothesis. Baselines unchanged.

## Latest result — 30 September 2026: projection rejected

Projection archive received. All8,500 masks and subgroup metrics recounted;
420 batch identities, script/inventory hashes, parent initialization and all20
frozen generators verified. No selected epoch. RealIoU ordinary0.32442 versus
projected0.31973; synthetic0.95082 versus0.95268, both below0.97469 baseline.
Projection active113 steps;36 of those still oppose replay after AdamW. Ten-row
preview remains fragmented. See `PROJECTION_RESULTS.md`. Next: quantify logged
step magnitudes/removed real components before choosing a materially distinct
intervention. No new training or model promotion.

## Latest VM preparation — 30 September 2026: matched projection runner

`scripts/train_projection_vm.py` and `coverage_projection.py` packaged as
`outputs/projection-code.tar.gz` plus LF checksum.23 local tests pass; no local
model training. `PROJECTION_VM.md` provides existing-coverage-workspace commands.
Forward-only GPU preflight precedes ordinary/projected210-update arms with shared
initialization/data/cache/schedule, unchanged losses/optimizer/gates and actual
step alignment logs. GPU execution pending. Next return artifact:
`outputs/projection-results.tar.gz`. No baseline/application promotion.

## Latest implementation — 30 September 2026: projection helper tested

`coverage_projection.py` implements one-sided real-gradient projection against
replay plus actual-parameter-step alignment reporting. Seven combined numerical
tests pass on synthetic tensors; no local model/optimizer run. Fixed treatment
defined in `COVERAGE_PROJECTION_PROTOCOL.md`; not symmetric PCGrad, no validation
tuned weight. Next: integrate separate VM-only matched ordinary/projected runner
on extended73 dataset with identical210-update schedules and frozen replay cache.
No new VM package ready yet and no baseline replacement.

## Latest result — 30 September 2026: mixed-gradient interaction verified

`coverage-gradient-results.tar.gz` received. Script/helper/protocol hashes and
six-batch identities verified; zero updates, reported model state unchanged.
Real/replay gradients oppose in5/6 batches at both final checkpoints (median
cosines control-0.1488, extended-0.1132); initial3/6. Decomposition error<=1.209e-6.
Teacher gradient is smaller than covered-real components, not uniformly dominant.
See `COVERAGE_GRADIENT_RESULTS.md`. Next: implement/test one fixed projection
comparison with actual AdamW-step alignment logging, preserving the same recipe
and gates. No new training package yet; no model promotion.

## Latest diagnostic package — 30 September 2026: mixed gradients, no updates

`outputs/coverage-gradient-code.tar.gz` plus LF checksum prepared for existing
`~/forensic-dgp/coverage_vm_bundle/`. Follow `COVERAGE_GRADIENT_VM.md`. First six
fixed extended training batches at three checkpoints; full objective decomposed
with actual weights, gradient sum checked and model state unchanged asserted.
Two small-tensor tests pass; no local model audit/training and no CUDA execution
yet. Next external artifact: `outputs/coverage-gradient-results.tar.gz`. This is
a zero-optimizer diagnostic, not another training run or promotion.

## Latest reconciliation — 30 September 2026: avoid repeating prior negatives

Original overfit/LR/real-only/penalty JSONs inspected; common parent and reused
control hash links verified. Higher LR, no replay and no penalty already failed
retention at tested settings. Historical V2/no-teacher/84-update runs are not
matched controls for current V3/teacher/210-update comparison. See
`COVERAGE_HISTORY_RECONCILIATION.md`. Remaining diagnostic gap: actual mixed
real/replay/teacher parameter-gradient interaction; prior gradient audit was
real-only. Next prepare fixed-batch VM zero-update audit; no new fitting yet.

## Latest diagnostic — 30 September 2026: broad real-training underfit

All73 real training images checked under parent/control/extended (219 inference
masks, zero optimizer updates). Original41 covered-image IoU control0.38258 versus
extended0.36032, extended misses60.7% of target pixels. New eyewear still nearly
unfitted; failure is broader than new categories. Clear false-positive cases8/26
versus9/26. See `COVERAGE_FULL_FIT.md`. Next: reconcile historical real-only/high-LR
fit diagnostics before another VM hypothesis; no unchanged rerun or promotion.

## Latest diagnostic — 30 September 2026: added examples not fitted

Inference-only parent/control/extended check on five added train images complete.
Opaque-lens IoUs <0.001 after7/9 exposures; profile/hand improve modestly but stay
fragmented. Clear control stays empty in all models. Five-row preview inspected.
See `COVERAGE_ADDED_FIT.md` and `outputs/coverage_added_fit/`. No optimizer updates.
Next: all73-real training fit/exposure breakdown before choosing another VM
intervention. Do not infer generalization or promote from these training metrics.

## Latest result — 30 September 2026: coverage experiment rejected

`coverage-results.tar.gz` received and verified. Both210-update arms completed;
no epoch meets unchanged gates. Final realIoU control0.32773 / extended0.32442;
synthetic0.94796 /0.95082 versus baseline0.97469. GlareIoU0 both. All8,500 masks
recounted; initial state equals parent;638 replay cases reproduce exactly;
all20 generators unchanged; final25 real masks/arm reproduced exactly on CPU.
Ten-row preview still shows fragmented/missed coverings. See `COVERAGE_RESULTS.md`.
No promotion or unchanged rerun. Next: local inference-only fit diagnostic on the
five added training images under parent/control/extended; no new VM training yet.

## Result checker prepared — 30 September 2026

`scripts/evaluate_coverage_results.py` prepared for 8,500 saved masks across both
arms/ten epochs. Validates inventory and returned tensor hashes, recounts metrics,
checks selection decisions and builds a ten-row preview. Two count/error tests
pass. No actual coverage result archive is present locally yet; GPU run status
is unknown without the user's SSH output. VM bundle unchanged. Next: user runs
the commands in `COVERAGE_VM.md` and returns `coverage-results.tar.gz`; then run
recount plus independent checkpoint/baseline inference before considering promotion.

## Latest VM package — 30 September 2026: coverage comparison ready for preflight

`outputs/coverage-vm-bundle.tar.gz` (60.8MB,1,425 verified files) and LF checksum
prepared. See `COVERAGE_VM.md` for upload, tmux, CUDA-forward preflight and two-arm
training commands. Fourteen local tests pass; zero local optimizer updates.
Runner uses original segmenter with existing consistency1/background0.25 recipe,
frozen generator, exact shared initialization/replay pixels and pinned schedules.
GPU execution remains unverified until user runs VM preflight. No VM connection
configured. Next external input: `outputs/coverage-results.tar.gz` after complete;
then recount masks, compare arms and inspect completion before promotion.

## Latest protocol — 30 September 2026: matched coverage schedules

Replay-source audit found the added clear control in the prior candidate pool;
excluded it from both arms. Shared200 sources and schedules frozen in
`outputs/coverage_protocol_v1/protocol.json`. Both arms210 updates,840 real/840
synthetic slots; exact synthetic case order and batch positions match; every real
train image sampled. Separate `coverage_protocol.CoverageBatches` avoids legacy
shared-RNG confound; three tests pass. Existing runner unchanged. See
`COVERAGE_COMPARISON_PROTOCOL.md`. Next: VM-only runner consuming fixed sources,
schedules, archived initial state and common replay tensors. No training yet.

## Latest package — 30 September 2026: mixed training extension V2

`dataset/detector_training_extension_v2/` loads 105 reviewed records: 73 train,
25 validation, 7 test. All 100 original V3 records unchanged. Five total train
additions: two opaque eyewear, one clear control, profile respirator, hand/mask.
Patterned-mask proposal remains excluded. See `TRAINING_EXTENSION_V2.md` for
manifest hash, screening and limitations. No actual training or app change.
Next: fixed-budget original/extended-data protocol and replay source exclusion
verification before packaging VM commands. No new VM run currently due.

## Latest package — 30 September 2026: reviewed training extension V1

`dataset/detector_training_extension_v1/manifest.json` packages all 100 original
V3 records unchanged plus three reviewed eyewear training additions. Actual
`detector_training.load_manifest` validation passes: train71 / validation25 /
test7. All copied file hashes verified. See `TRAINING_EXTENSION_V1.md` for hash,
provenance and limitations. Local only; no defaults switched or training run.
Readiness metadata is documentary, not enforced by existing runners. Next:
broader mixed-covering annotations before a bounded VM data-coverage comparison.

## Latest annotation work — 30 September 2026: eyewear sources

Update: corrected `outputs/eyewear_annotation_proposals_v2/` overlays inspected;
two opaque-lens labels plus clear control accepted as approximate assistant pilot
labels, all training-disabled pending integration. Native/resized images screened
against 4,200 frozen reference paths: no exact or DCT<=6 flags. Three nearest
reference images visually distinct; identity separation remains unverified.
`overlap_audit.json` records reference hashes and is hash-bound by review manifest.
Next: versioned training-source extension, followed by remaining mixed-covering
annotations before any VM training recipe. Original V3 remains unchanged.

Four native sources inspected; two opaque-lens polygon proposals and one empty
transparent-glasses control exported to `outputs/eyewear_annotation_proposals_v1/`.
The control is assistant-reviewed; opaque contours need rim/outer-edge correction.
One tinted-lens case remains held because eyes are partly visible. All training
disabled. Source hashes, original training membership and binary masks verified.
See `EYEWEAR_ANNOTATION_PROGRESS.md`. Next: contour and crop review before merging
a training-only extension; no VM run yet. Local artifacts not synced to VM.

## Latest preparation — 30 September 2026: real-source coverage screening

Follow-up: three manual crop/mask proposals exported to local
`outputs/real_expansion_proposals_v2/` with native polygons, source hashes,
nearest-V3-crop screening and inspected three-row overlay. Initial proposals
retained in `real_expansion_proposals_v1/`. One contour correction pass completed;
remaining thin-edge/strap/hand-boundary uncertainty is recorded. None accepted
or assigned to training. Binary shapes, source hashes and unchanged V3 verified.
Next: complete boundary review and non-mask/eyewear annotation coverage; no VM
training yet. Reproducible exporter: `scripts/prepare_real_expansion_proposals.py`.

Screened 1,510 real-occlusion source JPGs against 4,200 unique reference paths:
123 reference-flagged sources, 195 within-pool pairs, 1,100 unflagged candidates.
Reverified the existing 25-source queue. Reviewed 24 new source images; four have
digitally drawn mask appearance and are unsuitable for an unqualified real-mask
claim. Three native-reviewed candidates cover side-profile respirator, patterned
respirator and hand-over-mask. All remain training-disabled, without pixel labels
or split assignment. Whole-image screening does not prove identity/crop separation.
See `REAL_SOURCE_POOL_AUDIT.md` and `outputs/real_source_pool_audit/` under local
`C:\xampp\htdocs\YEAR 4\Testing\`; these artifacts have not been synced to
VM `~/forensic-dgp/`. Next: source-grouped crop/annotation proposals and overlap
review, preserving V3 validation/test membership. No new training command yet;
all actual training remains VM-only. Generator/application baseline unchanged.

## Latest reconciliation — 30 September 2026: legacy consistency already tested

Verified replay/consistency reported parent, 200 sources, recipe and final
checkpoint hashes against V2 evaluation. Consistency synthetic IoU 0.95705 versus
replay 0.95141, both fail retention; real IoU slightly worse. V2 correction does
not reverse ranking. V3 adds glare targets in two train/one validation/one test
images, so no direct matched comparison with modern V3 feature heads. See
`LEGACY_REPLAY_RECONCILIATION.md` and `outputs/detector_decision_review/legacy_reconciliation.json`.
Next: audit available real-source coverage and duplicates for a broader reviewed
training annotation queue, preserving held-out membership and strong-glare scope.
No unchanged recipe rerun, prediction-derived ground truth or new training yet.

## Latest decision — 30 September 2026: stop extending failed head series

Consolidated six final candidates from saved validation counts. Even label-informed
presence correction leaves best synthetic IoU 0.96284 below original 0.97469;
quality failure is not solely gating. Matched anatomical/border/RGB interventions
do not beat their controls. See `DETECTOR_ARCHITECTURE_DECISION.md` and local
`outputs/detector_decision_review/results.json`. No new fitting or promotion.
Teacher preservation was researched, but `detector_replay.py` already implements
synthetic Bernoulli-KL consistency and earlier results exist. Next: reconcile those
original-segmenter results, checkpoint ancestry and label versions before another
architectural proposal; avoid rebranding a failed recipe. No VM run currently due.

## Latest audit — 30 September 2026: initialization discrepancy explained

`refinement-initial-audit.tar.gz` received. Exported recreated tensors independently
reproduce the original reported VM digest. Parent/gate match originals, residual
output layer is zero. Local reconstruction differs only in four refinement tensors
by at most 7.45e-9, explaining the byte-hash mismatch. Underlying runtime/kernel
cause not isolated; historical initial tensors were never saved. See
`REFINEMENT_INITIAL_AUDIT.md` and local `outputs/downloaded_refinement_initial/`.
No training occurred. This removes the reconstruction blocker but does not alter
the quality failure or justify rerunning refinement. Next: consolidate matched
experiments into an architecture decision before another bounded hypothesis;
future runs must save actual initial states before optimization. Original quality
gates and application/generator baselines remain unchanged.

## Latest result — 30 September 2026: refinement rejected; provenance audit pending

Refinement archive evaluated: semantic synthetic IoU 0.95683 / RGB 0.95501;
both fail original retention, RGB increases FP, glare still absent. All 1,700
validation masks recounted and ten-row preview inspected. Parent and gate tensors
match originals exactly. No baseline/generator promotion. See `REFINEMENT_RESULTS.md`.
Local seeded initialization digest differs from reported VM digest across PyTorch
versions; cause unproven. Executed script asserts matched initialization, but initial
tensors were not archived. This remains a limitation, not a verified equality.
Next: upload `scripts/export_refinement_initial_vm.py` and run inference-free,
optimizer-free reconstruction on original VM; commands in result report. Return
`~/forensic-dgp/expanded_feature_bundle/refinement-initial-audit.tar.gz` to local
`C:\xampp\htdocs\YEAR 4\Testing\outputs\refinement-initial-audit.tar.gz`.
No additional training recipe until provenance and failed assumptions are reviewed.

## Latest preparation — 30 September 2026: refinement VM bundle ready

VM-only runner `scripts/train_refinement_vm.py` and input-binding/CPU-guard checks
completed; six combined tests pass. Two-member code archive verified at local
`C:\xampp\htdocs\YEAR 4\Testing\outputs\refinement-vm-code.tar.gz` plus LF checksum.
Follow `REFINEMENT_VM.md`: extract under `~/forensic-dgp/expanded_feature_bundle/`
and use existing venv. No git pull required for this standalone code bundle.
GPU preflight/optimization pending; no local optimizer updates. Both arms receive
800 updates at LR 0.001, same data/order/initial tensors, original loss, frozen
parent/gate. RGB binding checks every cached input, both real and synthetic.
Return VM `~/forensic-dgp/expanded_feature_bundle/refinement-results.tar.gz` to
local `C:\xampp\htdocs\YEAR 4\Testing\outputs\refinement-results.tar.gz`.
Next: unchanged validation/visual comparison. Known frozen-gate failure is still
unresolved; do not promote from training metrics. Earlier runner-not-ready notes
are superseded by this entry.

## Latest implementation — 30 September 2026: residual refinement head

`refinement_head.py` and four passing tests added. Frozen unweighted control parent,
128x128 refinement, RGB-enabled versus zero-input semantic control, zero-initialized
residual output. Both arms preserve actual parent logits exactly at initialization
on a cached training probe; identical initial state, 14,545 trainable parameters.
No local optimization, deployment or quality claim. See `REFINEMENT_EXPERIMENT.md`
and local `outputs/refinement_preparation.json`.
Next: implement VM-only runner and verify regenerated RGB against each cache row,
including real images, before issuing training commands. Runner and GPU preflight
are not ready. Parent VM path is `~/forensic-dgp/expanded_feature_bundle/outputs/expanded_border_training/control_epoch_10.pth`.

## Latest diagnostic — 30 September 2026: synthetic subset localization

Verified 46 exact expanded-input matches in incomplete 77-row local synthetic
cache; regenerated input/target checks pass. Local inference only, 184 saved masks
independently recounted. Six degraded failure examples inspected. Distant false
positives affect background/skin; border weighting increases them (degraded
irregular far FP 986 -> 1,221 across seven cases). Limited subset excludes revised
eye/lower inputs and is not full-training evidence. See
`BORDER_SYNTHETIC_LOCALIZATION.md`, local `outputs/border_synthetic_local/`.
Next: prepare a bounded image-conditioned residual refinement comparison with a
semantic-only control, verifying exact parent preservation at initialization.
No further loss-weight sweep, new local training, threshold change or promotion.

## Latest diagnostic — 30 September 2026: real-training FP localization

Local inference on 68 reviewed training images completed; 272 saved masks
independently recounted and all real raw/gated counts reproduce VM totals exactly.
Gated near-boundary FP (within 8 pixels): control 8,128 / border2 8,893; distant
FP 31 / 68. All 25 clear training images remain empty after gating. Six largest
treatment-FP rows inspected; boundary and strap discrepancies dominate. Coarse
polygon targets require caution, not automatic relabeling. No training occurred.
See `BORDER_TRAINING_LOCALIZATION.md` and local
`C:\xampp\htdocs\YEAR 4\Testing\outputs\border_training_localization\`.
Next: verify available synthetic training-cache provenance and localize its FP
separately; do not extrapolate this real-only result to synthetic retention.

## Latest result — 30 September 2026: border weighting rejected

`outputs/expanded-border-results.tar.gz` received and evaluated. Both VM arms ran
800 matched updates; script, protocol, parent/initial state, schedule and frozen
gate tensors verified. Local inference on 25 real / 400 synthetic cases per arm;
all 1,700 saved masks recounted and ten preview rows inspected.
Synthetic IoU: control 0.95096 / border2 0.94910. Border2 reduces misses but increases
visible FP (0.00283 -> 0.00367). Neither passes original synthetic retention; glare
still missed. Raw scores also fail, so presence alone is not the solution.
See `EXPANDED_BORDER_RESULTS.md`; evidence under local
`C:\xampp\htdocs\YEAR 4\Testing\outputs\expanded_border_validation\`.

Next: training-only false-positive localization on existing local feature caches
before considering a representation/head change. No weight sweep or unchanged
rerun. No local training or model promotion. VM results remain at
`~/forensic-dgp/expanded_feature_bundle/outputs/expanded_border_training/`.

## Latest preparation — 30 September 2026: matched border experiment ready

`scripts/train_expanded_border_vm.py` implements the specified control versus
border-weight-2 comparison, original targets, fixed parent, frozen gate/encoder,
800 updates per arm and final-only checkpoints. Three local loss/guard tests pass;
the schedule independently covers all 3,540 cases. Zero local optimizer updates.
VM GPU preflight and actual training are pending. Follow `EXPANDED_BORDER_VM.md`.

Upload from `C:\xampp\htdocs\YEAR 4\Testing\scripts\train_expanded_border_vm.py`
to VM home; run inside `~/forensic-dgp/expanded_feature_bundle/`. Return
`~/forensic-dgp/expanded_feature_bundle/expanded-border-results.tar.gz` to local
`C:\xampp\htdocs\YEAR 4\Testing\outputs\expanded-border-results.tar.gz`.
This isolates segmentation, not full qualification: known frozen-gate validation
rejections remain and must be addressed separately before promotion. Next: verify
both returned arms and compare original validation/visual safeguards. No baseline
replacement. The older region-report note saying code is not prepared is superseded.

## Latest result — 30 September 2026: region errors verified

Region audit archive received; the missing-mask blocker is resolved. Independently
recounted all 5,632 masks from 1,408 cases; counts and provenance match earlier
audits. Added borders account for 96.58% fixed / 95.21% anatomical missed degraded-
irregular target pixels; core miss rates are only 0.586% / 1.011%. Six predetermined
previews inspected, including background false positives. No blanket dilation fix.
See `EXPANDED_REGIONS_AUDIT_RESULTS.md` and local
`C:\xampp\htdocs\YEAR 4\Testing\outputs\downloaded_expanded_regions_audit\`.

Next: implement a matched VM-only border-weighted BCE experiment versus equal-
budget unchanged-loss control, same fixed parent, frozen presence gate and original
targets/retention safeguards. The report specifies the bounded recipe; execution
code is not prepared yet. No model promoted; generator/application unchanged.
Existing VM caches remain in `~/forensic-dgp/expanded_feature_bundle/`.

## Latest preparation — 30 September 2026: region audit ready for VM

Upload local `C:\xampp\htdocs\YEAR 4\Testing\scripts\audit_expanded_regions_vm.py`
to VM home and follow `EXPANDED_REGIONS_AUDIT_VM.md`. It uses existing caches and
heads in `~/forensic-dgp/expanded_feature_bundle/`, performs zero optimizer updates,
and checks input/target provenance plus previous per-case prediction counts.
Two local tests pass; GPU execution remains pending. It exports raw/gated/core/
target masks for 704 irregular training cases per arm and six input previews.
Return `~/forensic-dgp/expanded_feature_bundle/expanded-regions-audit-results.tar.gz`
to local `C:\xampp\htdocs\YEAR 4\Testing\outputs\expanded-regions-audit-results.tar.gz`.
Next: independently recount and inspect region errors before selecting training.

## Latest diagnostic — 30 September 2026: target geometry inspected

Training-only regeneration of 704 irregular cases completed without model fitting.
Added dilation border comprises 46.83% of degraded target pixels. Target-derived
64x64 roundtrip IoU is 0.99235 degraded / 0.98305 clean; grid shape preservation
alone does not explain learned-head fit. Six deterministic preview rows inspected.
See `EXPANDED_TARGET_GEOMETRY.md` and `outputs/expanded_target_geometry/`.
This does not prove a target defect or localize prediction errors. Next: existing
VM-head inference to separate core/border/outside errors and inspect predictions
on those same examples before choosing training changes. No gate/label changes.

## Latest result — 30 September 2026: grouped VM audit received

`outputs/expanded-fit-audit-results.tar.gz` is now present and checked; the missing
audit-result blocker is resolved. See `EXPANDED_FIT_AUDIT_RESULTS.md` and
`outputs/downloaded_expanded_fit_audit/verification.json`. All 7,080 case records,
group sums, original fit counts, protocol and checkpoint hashes match. The returned
execution script matches the prepared inference-only script. No training occurred
in this audit; feature arrays/predicted training masks still remain on VM.

Degraded irregular training IoU: fixed 0.90155 / anatomical 0.88405; raw scores
0.90177 / 0.88515 show that presence gating alone is not the main problem.
Next: training-only target core/border and spatial-resolution diagnostic before
choosing another VM training recipe. Investigate the 17x17 degraded-target dilation
and 64x64 prediction grid without changing labels, thresholds or retention gates.
Both candidates still fail original synthetic safeguards; no app/generator promotion.

Local evidence root: `C:\xampp\htdocs\YEAR 4\Testing\outputs\downloaded_expanded_fit_audit\`.
VM cache root: `~/forensic-dgp/expanded_feature_bundle/outputs/expanded_feature_training/`.

## Latest preparation — 29 September 2026: grouped VM fit audit ready

`scripts/audit_expanded_fit_vm.py` added for inference-only per-kind/degradation
training-fit analysis of existing checkpoints/caches. No optimizer, encoder rerun,
validation fitting or threshold search. Two local tests and Python syntax pass;
GPU execution pending. Full feature arrays are absent locally and remain on VM.

Upload `C:\xampp\htdocs\YEAR 4\Testing\scripts\audit_expanded_fit_vm.py` to VM home;
run from `~/forensic-dgp/expanded_feature_bundle/` with existing feature-bundle
venv. See `EXPANDED_FIT_AUDIT_VM.md` for exact commands. Script checks provenance,
row hashes, unchanged weights and exact aggregate agreement with original fit.
Return VM `~/forensic-dgp/expanded_feature_bundle/expanded-fit-audit-results.tar.gz`
to local `C:\xampp\htdocs\YEAR 4\Testing\outputs\expanded-fit-audit-results.tar.gz`.
Next: inspect grouped fit to separate degraded-irregular learning from transfer.
No new training or promotion. Tiny-highlight scope question remains pending.

## Latest independent diagnostic — 29 September 2026: segmentation limits retention

While glare-scope clarification is pending, audited the saved synthetic validation
counts. A label-informed perfect presence gate would still fail retention:
IoU 0.95218 fixed / 0.93899 anatomical versus baseline 0.97469. This is diagnostic
only, never a deployable oracle. All actual positive gate misses are irregular
coverings; about 92% of raw missed target pixels occur in degraded cases.
No threshold changes, fitting, new inference or promotion.

See `EXPANDED_RETENTION_AUDIT.md`; local evidence:
`C:\xampp\htdocs\YEAR 4\Testing\outputs\expanded_retention_audit\results.json`.
VM parent remains `~/forensic-dgp/expanded_feature_bundle/outputs/expanded_feature_training/`.
Next independent task: existing training-fit breakdown by covering/degradation,
particularly irregular coverings, using final heads/caches where available.
Pending glare-scope question remains unanswered; ambiguous proposals stay disabled.

## Latest annotation review — 29 September 2026: three proposals remain ambiguous

Native-resolution proposals prepared for glare-search indices 62, 127 and 141
(52 / 65 / 25 pixels). Binary dimensions, source hashes and counts verified;
three-row overlay inspected. These tiny spots overlap pupils/frame regions and
may be ordinary catchlights, not strong lens glare. Earlier thumbnail candidacy
is insufficient to assign occlusion truth. All proposals remain disabled and no
reviewed dataset labels changed. No training or model promotion.

Local preview: `C:\xampp\htdocs\YEAR 4\Testing\outputs\glare_source_search_v1\boundary_proposals\review.png`.
See `GLARE_SOURCE_SEARCH.md`; nothing new was uploaded to `~/forensic-dgp/`.
User clarification requested on tiny-highlight scope; recommendation is exclusion
under the existing strong-glare policy. Next: resolve this narrow policy question
and obtain clearer real-glare examples before adding labels or spending VM time.

## Latest source search — 29 September 2026: three localized-reflection candidates

Bounded source-only search reviewed 200 new original-training images (100 per
dataset), excluding prior pool/held-out content against 4,600 reference paths.
Two perceptual matches excluded. All five thumbnail sheets and 14 enlarged
eyewear cases reviewed: three localized-reflection candidates, five ambiguous
holds, six proposed clear controls, five opaque-eyewear thumbnails. The remaining
181 are not shortlisted, not verified negatives. No masks, training or promotion.

See `GLARE_SOURCE_SEARCH.md`; local artifacts:
`C:\xampp\htdocs\YEAR 4\Testing\outputs\glare_source_search_v1\`.
No VM files changed; prospective mirror `~/forensic-dgp/outputs/glare_source_search_v1/`.
Next: native-resolution reflection boundary proposals for search indices 62, 127,
141, with visible eyes preserved and overlay review before dataset inclusion.

## Latest data review — 29 September 2026: 27 eyewear candidates triaged

Prepared and reviewed a training-only eyewear queue after original split/hash and
held-out content exclusions. Twelve dark/mirrored-eyewear candidates, eleven clear
eyeglass control candidates and four ambiguous/non-glare/other-occlusion holds.
No new confirmed localized clear-lens glare set emerged; sunglasses must not be
counted as equivalent glare coverage. No masks assigned, fitting or promotion.

See `EYEWEAR_REVIEW_QUEUE.md`. Local evidence:
`C:\xampp\htdocs\YEAR 4\Testing\outputs\eyewear_review_v1\`.
No VM package was changed; future reviewed-data mirror would be
`~/forensic-dgp/outputs/eyewear_review_v1/` after packaging.
Next: bounded deterministic search of additional original-training sources for
localized lens reflections, with duplicate/held-out exclusions before review.
Do not move validation glare into training or weaken synthetic retention gates.

## Latest diagnostic — 29 September 2026: glare training fit versus transfer

Inference-only audit completed on all 68 reviewed training cases using verified
cached source features and current V3 targets. All 272 saved masks independently
recounted; six-row preview reviewed. Fixed recovers 80.30% / 73.91% of added glare
pixels in the two training examples; anatomical recovers 82.38% / 43.88%. Both
still recover zero validation glare pixels. All 25 clear training controls remain
empty after gating. Whole-image scores include a large medical mask in one glare
example and must not be presented as glare-only IoU.

See `EXPANDED_TRAINING_GLARE.md`; local evidence:
`C:\xampp\htdocs\YEAR 4\Testing\outputs\expanded_training_glare\`.
VM parent: `~/forensic-dgp/expanded_feature_bundle/outputs/expanded_feature_training/`.
Next: prepare additional training-only reflection candidates and clear-eyeglass
controls from existing reviewed source queues, with overlap exclusions and explicit
annotation review. No automatic labels or held-out-to-training transfer. Synthetic
retention remains failed; no new VM training or baseline promotion is justified yet.

## Latest result — 29 September 2026: expanded VM arms evaluated; neither qualifies

The returned `expanded-feature-results.tar.gz` contains both completed 20-epoch
runs. Protocol/parent/inventory/source/checkpoint checks and reconstructed schedule
pass; 7,080 cache records reviewed. Fixed development validation completed on
425 cases per arm; 1,700 saved masks independently recounted; ten preview rows
visually reviewed. Fixed/anatomical gated real IoU: 0.82150 / 0.81148; synthetic
IoU: 0.94662 / 0.93227. Both pass original real aggregate safeguards but fail
unchanged synthetic retention. Both miss all labeled validation glare pixels even
before gating. No application or generator promotion; no local training.

See `EXPANDED_FEATURE_RESULTS.md`. Local evidence:
`C:\xampp\htdocs\YEAR 4\Testing\outputs\expanded_feature_validation\`.
VM evidence: `~/forensic-dgp/expanded_feature_bundle/outputs/expanded_feature_training/`.
Next: inference-only diagnostic on the two reviewed training glare cases and
clear controls; distinguish failure to fit from limited real glare coverage.
Do not repeat the same VM run or lower validation gates. Independent final
evaluation and measured/visual end-to-end completion improvement remain unmet.

## Latest milestone — 29 September 2026: expanded VM experiment packaged

`scripts/train_expanded_feature_vm.py` now implements the predeclared matched
fixed/anatomical comparison: frozen SAM2, trainable context pixel and spatial
presence heads, identical initialization/schedule and 1,600 updates per arm.
GPU guard precedes workspace access. Preflight requires 6 GiB free VRAM and
35 GiB free disk, verifies hashes/splits and performs a forward without updates.
Disk-backed caches preserve partial state and refuse incomplete/corrupt reads.
Only final checkpoints and final training-fit metrics are produced; no selection
or application promotion happens on the VM. Training has not started.

22 tests pass (no optimizer steps), including CPU refusal, matched initialization,
gradient isolation, cache corruption and matched data. Real parent-state loading
was also checked separately. Bash syntax and
Python 3.10 parsing for 36 bundled Python files pass. Actual initial head digest:
`4c2012f4fd91c13b1d43e36debad50ce5db03480be8f030a615384d5d55d0575`.
CUDA compatibility remains unverified until VM preflight.

Upload local `C:\xampp\htdocs\YEAR 4\Testing\outputs\expanded-feature-vm-bundle.tar.gz`
and its `.sha256` file. Archive: 161,735,438 bytes, 624 verified members,
SHA256 `8eb07308ca85a5f52f7068f32313f8e840ea78362c2a68f220358896af62f2d7`.
Checksum has LF line endings. The failed path-verification archive and pre-runbook
archive remain preserved separately; upload only the filename above.

Runbook: `EXPANDED_FEATURE_VM.md`. Extract into a new
`~/forensic-dgp/expanded_feature_bundle/`, activate the existing
`~/forensic-dgp/feature_vm_bundle/.venv/`, run `scripts/run_expanded_feature_vm.sh`
inside tmux. No new VM connection is configured here; user executes SSH commands.
Return VM `~/forensic-dgp/expanded_feature_bundle/expanded-feature-results.tar.gz`
to local `C:\xampp\htdocs\YEAR 4\Testing\outputs\expanded-feature-results.tar.gz`.
Next: user runs preflight/training; inspect returned fixed validation, mannequin,
glare and completion previews before considering promotion. Goal remains unmet.

## Latest implementation — 29 September 2026: matched arms and disk-backed cache

`feature_disk_cache.py` added: float32/uint8 memory maps, per-row checksums,
provenance checking, atomic progress and refusal of incomplete/changed caches.
18 targeted tests pass, including corruption, partial-write, ordering, matched
membership and shared-texture checks. No optimizer or GPU work ran locally.

Full paired replay passed for 3,472 cases per arm from the same 352 sources;
48 rejected variants are identical. All 2,112 generic/clear variants still match
legacy exactly. Fixed and anatomical eye/lower arms use identical texture RNG;
2,954,499 shared clean covering pixels match. Evidence:
local `C:\xampp\htdocs\YEAR 4\Testing\outputs\expanded_feature_data_v1\matched_check.json`;
intended VM mirror `~/forensic-dgp/feature_vm_bundle/outputs/expanded_feature_data_v1/`.
Historical manifest implementation hashes are retained; matched-check hashes pin
the revised loader/cache. Benchmark, thresholds and application remain unchanged.

`EXPANDED_FEATURE_DATA.md` predeclares identical initialization, trainable context
pixel/spatial presence heads, frozen SAM2 and 1,600 updates per arm. Differences
against older experiments are not attributable solely to added data. Next:
implement GPU-only runner/preflight, package it, provide exact SSH commands.
The VM experiment is not launch-ready; no output-quality improvement claimed.

## Latest implementation — 29 September 2026: expanded loader and fixed-budget sampler

`expanded_feature_data.py` and `scripts/prepare_expanded_feature_data.py` added.
14 targeted tests pass. All 3,472 synthetic cases checked; 2,112 generic/clear
variants exactly reproduce the legacy recipe. 352 source hashes and membership
verified; 48 rejected anatomical variants recorded explicitly. Six balanced groups
use a fixed 1,600-update schedule with cross-epoch coverage, not a larger budget.
Original benchmark and application remain unchanged; no training ran.

Artifacts: local `C:\xampp\htdocs\YEAR 4\Testing\outputs\expanded_feature_data_v1\`;
intended VM mirror `~/forensic-dgp/feature_vm_bundle/outputs/expanded_feature_data_v1/`.
See `EXPANDED_FEATURE_DATA.md` for counts, hashes, tests and limitations.
Next: implement disk-backed GPU-only runner and predeclare the matched experiment,
then package it. The data checks are complete; the VM run is not launch-ready yet.

## Latest implementation — 29 September 2026: explicit crop-aware augmentation

Reviewed 47 rejected sources, 12 accepted pose extremes and all 35 newly clipped
lower-face outlines. Border-only rejection affected 35 Asian-source crops and zero
FFHQ crops. `anatomical_augmentation.py` now supports explicit
`boundary_policy='clip'` (`anatomical-covering-v2-clip`); default V1 remains strict.
Full-polygon rasterization followed by cropping preserves pixel labels; six tests
pass, including crop equivalence and invalid geometry rejection. Replayed all
352 cached sources: default V1 events unchanged; V2 generates both anatomical
kinds for 164/170 Asian and 176/182 FFHQ sources. Twelve sources remain rejected.
No benchmark, model, validation threshold or source file changed. No training ran.

Evidence under local `C:\xampp\htdocs\YEAR 4\Testing\outputs\training_diversity_audit\landmark_trial_v1\`:
`acceptance_review/visual_review.json` and `clipping_v2/results.json`.
This is source-data preparation only; partial procedural coverings do not certify
whole-region coverage, photorealism or improved completed faces. Source cleanliness,
square-resize distortion and sparse real glare remain limitations.
Next: version the expanded loader and source-balanced sampler, then package the
bounded VM experiment for `~/forensic-dgp/feature_vm_bundle/`. Not launch-ready yet.

## Latest verification — 29 September 2026: returned archive and detector dimensions

Newest returned archive found locally: `outputs/pixel-comparison-results.tar.gz`.
All five archive files match the extracted evaluation inputs byte-for-byte;
SHA256 `67f31eebbe8a57d8a1acf2b1ace5543878d2f0c1227cfe862bc78fd28aab5e3f`.
The existing 1,700-mask verification and failed retention decision still apply.
No new qualifying model or training run is claimed.

The anatomical trial's ONNX warning has a verified dimensional explanation:
dynamic input, fixed output metadata for 640, actual outputs consistent with
stride/anchor decoding at both 256 and 640. Installed InsightFace routes this
model to SCRFD, whose decoder builds anchors from actual input dimensions.
This clears the dimensional concern, not landmark accuracy or profile coverage.
Evidence: `outputs/training_diversity_audit/detector_shape_audit.json`.
Next: review rejected/profile cases and source acceptance before packaging the
expanded training data. Training remains VM-only and is not ready to launch.
Local root: `C:\xampp\htdocs\YEAR 4\Testing\`; VM bundle root:
`~/forensic-dgp/feature_vm_bundle/`. See `ANATOMICAL_AUGMENTATION.md`.

## Latest implementation — 29 September 2026: anatomical augmentation prototype

`anatomical_augmentation.py` added, four tests passed. Separate training-only
five-landmark eye/lower polygons with explicit failure reasons; no existing
benchmark or inference integration. 352 source-only landmark inferences completed:
340 eye / 305 lower masks generated; failures recorded. First 12 jointly accepted
previews reviewed, profile reliability still limited. ONNX dynamic output-size
warnings at 256 need configuration verification before using the landmark cache.
See `ANATOMICAL_AUGMENTATION.md` and `outputs/training_diversity_audit/landmark_trial_v1/`.
No training ran. Next: verify detector configuration and review acceptance/profile
failures before versioning the expanded VM data pipeline. No model promotion.

## Latest anatomy review — 29 September 2026: 363 source outlines inspected

All eye/lower outline previews reviewed for 363 candidate clean sources.
Fixed eye bands often miss eyes across both sources; pixel geometry labels are
still correct for pasted objects, but anatomical simulation coverage is weak.
Eleven additional source-cleanliness flags recorded. See
`outputs/training_diversity_audit/anatomy_review/{results,source_status}.json`.
Next: implement versioned landmark-conditioned training augmentation with explicit
failure/review records, preserving generic occlusions and existing benchmarks.
Source landmarks are training synthesis inputs only, never an inference reference
requirement. Expanded-data training remains not ready; no local fitting occurred.

## Latest detailed review — 29 September 2026: source roles separated

All 59 flagged sources inspected at enlarged/native-content scale. Recorded in
`outputs/training_diversity_audit/source_roles_v2.json`: 22 clean candidates pending
geometry checks, 25 real-occlusion cases needing masks, five geometry holds, six
text-overlay exclusions, one unusable-image exclusion. Remaining 341 sources are
still thumbnail-provisional. No originals removed, no inferred hidden-face targets.
`real_occlusion_annotation_queue.json` is training-disabled pending masks/grouping.
Next: inspect source cleanliness and anatomical covering placement across 363
candidate clean sources before versioning the augmentation pipeline. No local
training, new VM launch, or application promotion.

## Latest source review — 29 September 2026: all 400 thumbnails triaged

`outputs/training_diversity_audit/source_triage_v1.json` covers all 400 images:
341 provisional passes, 59 quarantined for full-resolution/geometry review.
Six flagged images were in the prior 40-source pool. Pre-existing eyewear/objects
can conflict with synthetic empty-mask labels and clean reconstruction targets.
Separate provisional V2 manifest has 163 Asian-source and 178 FFHQ entries;
`training_ready` is false. No images deleted or validation labels changed.
Next: inspect flagged occlusions in detail, separate clean-target and real-occlusion
roles, then review generated placement before a balanced VM recipe. See
`TRAINING_DIVERSITY_AUDIT.md`. No local training or application changes.

## Latest screening — 29 September 2026: data geometry needs correction before expansion

400-source candidate screen versus 4200 reference paths found zero decoded-RGB
duplicates / DCT-hash flags at distance <=6. This does not certify identity splits.
Ten-source visual preview revealed a mostly black, tiny rotated face in
`asian_face_07207.jpg`, and fixed eye masks below actual eyes in other Asian-source
portraits. Candidate expansion is explicitly not training-ready. Preserve it;
review source quality and augmentation placement before creating a new version.
See updated `TRAINING_DIVERSITY_AUDIT.md` and
`outputs/training_diversity_audit/{duplicate_screen,geometry_review}.json` plus
`geometry_preview.png`. No model fitting, label changes or application promotion.

## Latest data preparation — 29 September 2026

`TRAINING_DIVERSITY_AUDIT.md` and
`outputs/training_diversity_audit/candidate_sources.json` now record a 400-source
candidate pool (200 Asian/200 FFHQ, retains previous 40), original training
membership, successful decode and exact-byte exclusions against 4196 held-out /
reviewed hashes. One duplicate skipped. Near-duplicate/identity screening pending.
All 76000 original training paths exist locally; only selected images decoded.
Real training still has only two strong-glare cases. Synthetic loader stretches
non-square portraits to squares; effect unmeasured, benchmark unchanged.
Next: duplicate screening and stratified source/covering geometry review before
implementing any expanded-data VM recipe. The old fixed-size sampler cannot be
used unchanged for this larger manifest. No local training or deployment changes.

## Latest pixel result — 29 September 2026: context helps synthetic, fails full retention

Both pixel-head VM runs completed. Fixed validation and independent recount of
1700 saved masks complete; ten-row diagnostic grid inspected. Context gated
synthetic IoU 0.92223 versus pointwise 0.88015, but real IoU falls to 0.81811
versus 0.82858. Both fail original synthetic safeguards; glare and five synthetic
gate misses persist. No application promotion. See `PIXEL_COMPARISON_RESULTS.md`.
Next: audit and expand training diversity with fixed architecture/thresholds,
not another architecture change or repeat on the same 40 source images. Preserve
all validation/test membership; real glare coverage and independent final
evaluation remain unresolved. All fitting must stay on the VM.

## Latest preparation — 29 September 2026: pixel-head VM comparison ready

Upload `scripts/compare_pixel_heads_vm.py`; see `PIXEL_COMPARISON_VM.md` for SSH
commands. Pointwise and 3x3-context arms have matched initial functions, identical
468 training examples, loss, batch order and 1600-update budgets. Frozen encoder
and frozen spatial presence gate; final-only checkpoints. Two local tests pass
(initial equivalence/feature isolation and CPU refusal), Python syntax passes,
no local optimizer updates. GPU run and quality results remain pending.
Result path: `~/forensic-dgp/feature_vm_bundle/pixel-comparison-results.tar.gz`;
return to `C:\xampp\htdocs\YEAR 4\Testing\outputs\`. No application changes.

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
