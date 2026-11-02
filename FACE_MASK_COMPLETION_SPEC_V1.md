# Face-mask detection and completion: design interview V1

25 October 2026. Status: AGREED SCOPE; CodeFormer selected for the manual reviewed-mask prototype. The goal is ACTIVE. This is an active development plan, not a qualified model or ready training packet.

## Confirmed user decisions

1. Develop on FACE_MASK_COMPLETION at C:\xampp\htdocs\YEAR 4\Testing_FACE_MASK_COMPLETION. Restoration development remains separate on RESTORATION.
2. First establish useful completion with a manually reviewed removal area. Keep the removal area fixed for model comparisons and preserve pixels outside it exactly. Automatic detection improvement follows separately.
3. Begin with existing audited covering development examples, retaining all seven covering families, clear-glasses/uncovered controls and existing insufficient-information exclusions.
4. Use up to 120 seconds per completed face as the initial local-app testing ceiling when quality improves noticeably. This is an unmeasured target, not a guarantee for arbitrary devices or an actual training limit.
5. Finished generation must run on the local computer and be installable on other devices. A live cloud VM must not be required for user generation. After model/output review and user satisfaction, port the accepted feature to IMPLEMENTATION with regression and installation checks.

## Retained boundaries

The user selected the existing CodeFormer inpainting component as the preferred completion starting point after reviewing the saved model comparison ("lets use the codeformer its much better"). This is a development selection, not full-family qualification. Its codeformer_inpainting.pth checkpoint is distinct from the CodeFormer restoration checkpoint and restoration adaptations. The tested third-party MAT conversion remains a failed comparison; it was not promoted. Existing images, masks, weights, split membership, provenance, exclusions and failures remain retained. The current restoration model is fixed during isolated completion assessment.

One output is a plausible estimate of the hidden region, not exact hidden identity. Show original, reviewed removal area and result, with PNG and optional evidence-bundle downloads. Preserve transparent glasses, non-obstructing hair and visible appearance. Request less covering when too little face remains. Automatic and assisted evidence stay separate.

Actual new training remains manual on the authorized L4 VM, with finite timing/update/epoch stops and verified transfer/return audits. Local inference and saved-evidence audits are permitted. No new training, model inference, installation or release follows from this interview record.

## Current implementation constraint and measured reference

The inspected local IMPLEMENTATION branch uses Windows install.bat and CPU PyTorch. Port the accepted application feature and its dependencies deliberately; do not merge this research branch wholesale into the small deployment branch.

The first release targets Windows, including machines without an NVIDIA GPU (user decision, 25 October 2026). The current reference laptop has an Intel i5-11260H (6 cores/12 threads) and15.69GiB RAM. CPU-only completion will be timed against the120-second target; this is not a minimum-hardware certification. Candidate selection and final package size depend on measured outputs. Compare real saved outputs and measured runtime before selecting a replacement. All eight retained MAT-versus-CodeFormer comparison sheets were reopened during this interview; copied-pixel preservation passes do not establish plausible anatomy or full-family usefulness.

## Source locations

Research evidence and ignored assets remain in C:\xampp\htdocs\YEAR 4\Testing. They are not automatically duplicated into this worktree. The fixed historical comparison is outputs/completion_mat_mirror_comparison_v1 there; its eight sheets and model assets were rechecked by a read-only inventory. No model was loaded or rerun during the interview. Subsequent candidate inference must carry a separate frozen protocol and independent output audit.

## Quality-first decision and active goal

The user accepted a suitable pretrained completion starting point, with fine-tuning only if fixed comparisons show a justified need. The interview is complete enough to act; prioritize concrete progress. The dedicated FACE_MASK_COMPLETION goal was created and is ACTIVE on25October2026. The final accepted local Windows feature is intended for IMPLEMENTATION after user satisfaction, preservation checks and app/installation regressions.

The official MI-GAN256 FFHQ comparison remains retained as evidence and is not promoted. Its source revision2b793c5ece43f4253e32d4afc257120a5deed6f5 and25,231,129-byte face checkpoint remain in outputs/completion_migan256_assets_v1 with acquisition fingerprints. Speed and exact copied pixels alone do not establish useful completion. All candidate comparisons retain the exact fixed case/mask protocol; do not mix newer footprints with old baseline outputs.

## Current manual CodeFormer prototype

The opt-in local prototype comprises completion_codeformer_web_v1.py, scripts/completion_codeformer_worker_v1.py and templates/completion_codeformer_v1.html. It supports upload/preparation, binary Paint/Erase/import, explicit mask and visible-structure confirmation, one plausible estimate, original/mask/result PNGs and an evidence ZIP. New generations retain separate immutable IDs. Completion preserves source pixels outside the reviewed removal mask on the prepared256 canvas. The original upload remains separately retained; the preparation geometry is not a claim of unchanged native-resolution pixels.

Automatic detection is not used in this prototype. Empty masks bypass generation; insufficient visible support requests a less-covered/clearer crop. Restoration is not called or modified. The local worker is CPU-only with a120-second process cap; usable timing on other Windows computers remains to be measured. Root verification recorded one nonempty CPU completion in12.283seconds, one neural call, an exact empty-mask bypass, insufficient-information refusal, PNG/ZIP downloads and responsive browser checks at375/768/1280pixels. Fifteen focused source/worker tests passed. These are functional checks, not full-family usefulness, minimum-hardware certification or final user acceptance.

No new training has run for this prototype. No model is promoted into IMPLEMENTATION, and no implementation-branch merge has occurred. After manual completion usefulness and user acceptance are established, improve automatic proposals separately and port only accepted sources/assets with installation and regression evidence. See COMPLETION_CODEFORMER_MANUAL_V1.md for the prototype launch and review procedure.

## HQ TRAIN calibration and the next manual learning diagnostic

A fixed six-source paired TRAIN calibration completed30 requests/24 forwards/six exact empty controls; all six sheets were reviewed and raw/PNG outputs independently replayed. It exposes hidden-expression uncertainty and eye/lens boundary weaknesses despite clean aligned context. All5,375,448 outside-mask source bytes remain exact. It does not qualify independent DEV/FINAL or seven real covering families. Read COMPLETION_CODEFORMER_HQ_TRAIN_PROBE_V1_RESULTS.md.

The original probe's launcher-only RAM statistics and actual-worker hard-kill boundary are invalid and retained with that limitation. The local app now uses a separately reviewed Windows Job Object guard, suspended assignment and single-owner lifecycle. Its fresh browser request preserved the same result, with actual job private committed-memory measurement; minimum hardware and portable installation remain unverified.

The finite manual L4 R1 gradient packet is archived and independently verified, with four admitted original TRAIN sources, eight fixed masked cases,24 typed forward slots and16 gradient queries, zero optimizer updates and a600-second cap. It first proves intended CodeFormer inpainting code/fusion paths before any new fine-tuning proposal; the inference-only wrapper is not a training route. It remains UNRUN. Follow COMPLETION_CODEFORMER_GRADIENT_V1_R1_VM.md and independently audit the returned archive. No VM action, learned model promotion, automatic-detector adoption or IMPLEMENTATION integration occurred here. Goal ACTIVE.

## Python3.10 compatibility correction

The manual VM transfer check stopped before neural/gradient work because file_digest is unavailable on Python3.10. A hash-bound two-helper repair is prepared and independently checked; it preserves scientific settings and original packet evidence. Use COMPLETION_CODEFORMER_PYTHON310_REPAIR_V1_VM.md and the new protocol SHA. Windows upload/download files now use the original Testing/outputs transfer folder. Model development remains isolated in FACE_MASK_COMPLETION. Await the manual repair and returned diagnostic; no actual training or model promotion follows from preparation. Goal ACTIVE.

## First actual fine-tuning pilot ready, unrun

The repaired gradient return passed independent numerical and saved-output review. The first actual completion-learning packet is now sealed and independently audited:32FIT/eight adaptive TRAIN monitor photos,64Adam steps maximum, app-aligned input and one terminal complete recovery state. Use COMPLETION_CODEFORMER_FINETUNE_V1_VM.md from original Testing/outputs for manual upload and execution. CodeFormer baseline remains selected; no trained improvement, app promotion, FINAL evaluation or implementation merge is claimed. Goal ACTIVE pending the returned training audit and broader quality/deployment scope.


## Current correction: first fine-tuning return and isolated R1

The original V1 manually executed but failed the fresh app-parity check before any backward or optimizer update. Its complete saved trainable state equals the original; no new trained model or quality output exists. Historical ready/unrun statements above are superseded by this return. All failure evidence remains retained.

R1 is prepared, unrun: exact app active-index input/layout and output composition, all four paired parity arrays and numeric receipts saved before the unchanged 1e-6 gate, same data/contract/LR/losses/gates/resource limits. Independent source review, 10 pure CPU tensor tests, installer static review and seven full fixture checks pass. GPU model parity remains unverified. Follow COMPLETION_CODEFORMER_FINETUNE_V1_R1_VM.md from original Testing/outputs; all actual training stays manual on L4. No promotion, restoration change or IMPLEMENTATION merge. Overall completion, covering-family review and deployment remain incomplete.


## Audited trained R1 and current experimental user test

R1 executed manually on L4 and passed the finite paired numeric screen:32cycles/64Adam updates,6.185334% relative delivered-PNG hole-L1 reduction on adaptive TRAIN monitoring, all macro/per-geometry preservation/seam requirements passing. Saved state, raw/PNG and all16visual sheets were independently checked. Six individual hole-L1 cases worsened and public development visual quality remains mixed; no full-family or hidden-identity qualification.

An isolated trained R1 CPU prototype now provides the same reviewed-mask editor, one estimate, exact outside pixels, insufficient-information refusal, PNG/evidence downloads and120-second owned-process cap. One actual reference request took12.5seconds. Original baseline and restoration remain separate. Follow COMPLETION_CODEFORMER_FINETUNE_R1_APP.md to compare identical inputs/masks. No automatic detector or IMPLEMENTATION integration was added. User acceptance, remaining family defects, independent final human review, minimum-hardware/fresh installation and the full goal remain pending.


## User-selected working model

The user prefers R1 after the local original-versus-trained comparison. Use retained trained R1 as the next completion-development starting point, with original CodeFormer retained for comparison. This preference is narrower than full-family user acceptance or final qualification; known scarf/hair/glasses limitations and independent review/deployment requirements remain. Current admitted app comparison stays fixed during the next diagnosis.

## R1 continuation: prospective coverage data admission

The explicit R1 preference selects the working completion starting state. Preserve the original comparison and the live admitted app bytes. The next data-only design adds128 paired TRAIN augmentations and16 separate adaptive TRAIN challenges after independent source/seed/target/mask and20-sheet visual review. Restore full R1 Adam/RNG state rather than use the inference-only checkpoint as a recovery state. Keep unchanged losses, original teacher, LR and preservation gates. Source and transfer audits must complete before the distinct manual L4 launch; no actual training has occurred for this prospective phase. Saved monochrome output processing remains a separate diagnostic and is not integrated. FINAL-review materials exist, while independent people, ratings and final image evaluation remain incomplete.

## Completion coverage V2: manual-transfer-ready, unrun

The R1-parent continuation is now packaged under protocol c838f4472cf87c138b2d518ef4f543bfcec3e8149ea7a5411b863397bf66859d and archive SHA175ceb02d0dd195dd2b2d0c95d4413714f06e81509251136ba2f12d166f10ac6. Independent data/source and full archive audits passed; actual VM checks and manual training remain pending. Follow COMPLETION_CODEFORMER_COVERAGE_V2_VM.md with7GiB free and finite64cycle/1800second limits, retaining R1 full Adam/RNG parent and original baseline. All returned states and raw/PNG outputs, separate cohorts and every planned visual case must be independently audited before any replacement. Current app, RESTORATION and IMPLEMENTATION remain unchanged. No newly trained V2 model or full-family qualification is implied by this preparation.

## V2 returned: keep R1; no trained replacement

The actual V2 return is audited on26October2026. All64 cycles/evaluations completed in receipts, but terminal serialization timed out and its partial file is invalid; no usable V2 model exists. Fixed clear-cohort error worsened.377504% versus R1; newchallenge improves4.136863% but subgroup requirements fail. Exact outside pixels/empty bypass pass, so generated-region quality failure must not be confused with visible-copy corruption. All20 sheets/224 outputs reviewed; trained R1 remains working model. Keep native/unpaired and synthetic/paired evidence separate. A separately reviewed O(1) safety-writer prototype addresses a measured local save bottleneck without changing limits or failed V2 sources. A distinct fusion-only R1 route-isolation review is prospective, not a new trainer or ready transfer packet. Do not relaunch historical V2 or rename its partial file as a model. Full goal remains active; app and other branches unchanged. See COMPLETION_CODEFORMER_COVERAGE_V2_RESULTS.md and return_milestone_v1.json.

## Fusion-only V3: prepared, not executed

26 October 2026. A distinct R1 continuation is independently reviewed and packaged:
protocol3dc71dac3002e82587cbc2cb758f36df604ed4c7131a3e592db4d8dcb5c18033,
archivef2ef14a22a50756ac31825aa14caeb861a76994e75499fce6498bf28df3a07e0.
Freeze443 tensors and code Adam at32; adapt54 fusion tensors only, at most64 new
updates/256 backwards/480 typed forwards on unchanged data/masks/schedule. Preserve
full R1 state/Adam/RNG, old gates and current apps. Fresh wrapper parity and all48
historical initial raw/PNG compatibility checks precede updates. Use the reviewed
O(1) writer and phase timing within unchanged finite limits. The user launches
manually using COMPLETION_CODEFORMER_FUSION_ONLY_V3_VM.md, then returns all3 files.
No new V3 training or model qualification is implied. Returned states/pixels/all
visual sheets, user review and later independent final review are still required.


## Fusion-only V3 returned: valid model, not a better accepted replacement

26 October2026. Actual manual V3 completed64 fusion updates without timeout and
saved a valid full recovery state. Independent audits verify443 frozen/54 changed
tensors, code Adam32 unchanged, fusion Adam96, all224 outputs and exact visible
pixels. Both incremental PNG hole-error gates versus working R1 fail:0.419741%
clear and0.479152% challenge, with additional raw/PNG hole subgroup regressions.
All20 visual sheets/80 cases reviewed; persistent expression, gray-color,
invented glasses and scarf artifacts remain. Retain R1 and its unchanged22 app
pins. Original CodeFormer remains the permanent comparison. Do not promote V3
from its valid state or6.579113% cumulative gain against original.

Saving and image usefulness are separate. The O(1) writer worked on actual L4;
source/state/failures remain immutable. Saved-array objective diagnosis shows
HQ512 boundary gains can outweigh worsening holes, and delivered256 behavior
also differs. A distinct app-objective design review follows; no new training
packet is ready, no unchanged historical rerun or appended epochs are allowed.
All actual training stays manual and finite. Automatic detector improvement,
independent human final review, all-family acceptance, CPU installation and
IMPLEMENTATION port remain incomplete. See COMPLETION_CODEFORMER_FUSION_ONLY_V3_RESULTS.md.


The distinct next V4 delivered256 hole-only objective design is independently
reviewed: COMPLETION_CODEFORMER_DELIVERED_OBJECTIVE_V4_DESIGN.md. Its exact
existing FIT256 targets are available without new target generation. Runtime
implementation and packet admission are in progress; design review is not a
manual-launch authorization or new trained result. All prior gates remain.


## V4 delivered256 objective: manual-transfer-ready, unrun

26October2026. The reviewed distinct V4 source/runtime/packet and exact five-step
guide are now independently audited and published. Earlier design-only readiness
statements are superseded. Same R1/Adam/RNG,443frozen/54fusion,256FIT schedule and
finite64-step limits; exact app float256 hole loss against existing clean256,
with no additive boundary training term. Every old boundary/hole quality gate
remains. Archive2c5607a06fcc31ded3ebb082ecd88e6914cdd208df4a2c947d3f09896add726f,
protocol8bbb3dd611e95cfbce76e7e6426ffdadb31ce23dadb1dfdcd0246e501213069d.
Use COMPLETION_CODEFORMER_DELIVERED_OBJECTIVE_V4_VM.md. No actual VM training
or qualified replacement exists yet; R1 app is fixed. Return all3files for full
state/pixel/all20sheet audit and user review. Full goal remains ACTIVE.


## V4 actual return: retain R1 and review representation

26October2026. The actual manualV4 run and fullstate are independently audited:
64fusionupdates,443frozen/54changed, all224outputs exact, all20sheets reviewed.
Quality fails:clear+0.310926%,challenge-0.047216% PNG hole-error reduction versus
R1 plus subgroup regressions. Exact outside pixels/empty controls pass. R1 stays
selected with22app/model pins unchanged. NoV4adoption, epochappend or unchanged
rerun. Earlier prepared/unrun instructions are historical.

App256hole-only objective alignment was insufficient with prediction/codeprior
frozen. A separate officialHQ512 teacher asset is notdropin: codebook/generator
differ. Do not use itscodeIDs aslabels or swapprior silently. Next review is a
bounded zerooptimizer SAME-codebook clean-context oracle diagnostic, explicitly
using unavailableclean information only on pairedsynthetic TRAIN. No nextpacket
is ready; all actualtraining remains manual andfinite. See
COMPLETION_CODEFORMER_DELIVERED_OBJECTIVE_V4_RESULTS.md. Fullgoal ACTIVE; all-family
useracceptance, automaticproposal evaluation, independentFINAL andCPUdeployment
remain incomplete. RESTORATION andIMPLEMENTATION unchanged.


## Same-codebook probe V1: independently prepared, unrun, zero updates

26October2026. A finite manualL4 representation diagnostic is now source/packet/
transfer verified. Four conditions compare actualcodes and explicitly labeled
clean-contextORACLEcodes with w1/w0fusion. Exactretained16same-codebooklabels;
48pairedadaptiveTRAINcases,192outputs+4parityarrays,200planned/220hardforwards,
zerooptimizer/gradients,all497R1tensors unchanged. Ordinary4pairparity+48R1raw/PNG
compatibility precede controls/oracle. Guide:
COMPLETION_CODEFORMER_SAME_CODEBOOK_PROBE_V1_VM.md. Archive099e3cac01b107258be8d1af4d5bd2eb72485bdb34b087553c46f0749654e2c7,
protocol51782b57fca9ad9cf7161b0f4451a65e545e3eb2f8f58788fa40b64aec48d1ec.
Noactualdiagnostic/modeltraining yet. R1app staysfixed; noofficialHQ512label/prior
substitution. Returnall3files for independent192output/code/stateinvariance and
all12sheet review. Oracleusesunavailablecleancontext,notappoutput or identity
recovery. Itdoesnotautomaticallyauthorizefurtherlearning orcompletefullgoal.


## Same-codebook probe actual return and predictor-only next design

26 October 2026. The zero-update manual probe is now executed and independently
closed: 200 forwards, all497 R1 state tensors exact, all192 outputs replayed,
all12 sheets reviewed. R1 remains selected. Clean-context SAME-codebook oracle
codes at fixed R1 fusion reduce paired hole error16.488%/16.364%, but use hidden
clean information unavailable in the app. Three individual holes worsen; gray
false color and softness persist. Removing fusion with actual codes is worse.
This is a representation diagnosis, not a new trained model or identity result.

Review a distinct predictor-only CE test:114 predictor tensors learn while the
other383 encoder/fusion/prior tensors remain fixed. Balanced128 existing FIT
observations across32 identities/64labels; keep all16 monitor-label sets out of
optimization. Final32-step endpoint,128backs maximum; earlier checks safety-only.
Runtime source preparation is underway. No execution packet or automatic launch
is ready; manual L4 transfer/source/state/output review remains required. All
original evidence and current R1 app remain unchanged. Automatic proposals,
all-family user acceptance, independent final review and CPU deployment remain
unfinished. See COMPLETION_CODEFORMER_SAME_CODEBOOK_PROBE_V1_RESULTS.md.


## Predictor-only CE V1: manual-transfer ready, unrun

27 October 2026. The distinct predictor-only packet is fully source/transfer/
archive/guide verified. Follow COMPLETION_CODEFORMER_PREDICTOR_CE_V1_VM.md from
original Testing/outputs. ExactR1 weights/Adam/RNG;114predictor tensors learn,
383encoder/fusion/prior tensors frozen;balanced128existingFIT observations,
32newsteps/128backs. Photograph IDs are disjoint, not independently proven human
identities/pretraining separation.16monitor labelsets never train. Ordinaryw1
inference only;nooracle/newteacher/latentMSE orencoder/fusion learning.

Final32-only oldhole/boundary requirements plusgray-style nonworse guard.8/16
are safety checks only. Require8GiBfree+installpayload,2GiBfloor,30minhardworker,
28minwork+120ssave,finite32steps. Packet source tests and all671archive members
verify; noactualVM/modeltraining orqualifiedsuccessor yet. R1app remains exact.
Returnall3files for independentfullstate/logits/pixels/all-case review before
adoption. Noautomatictraining/resume/epochappend orhistorical relaunch.
All-familyusefulness,automatic-versus-assisted evaluation,useracceptance,
independentFINAL andWindowsCPU deployment/IMPLEMENTATION port remain incomplete.


## Predictor-only CE V1 actual return: retain R1; review spatial supervision

27 October 2026. The manual32-update/128-backward pilot executed successfully
but failed quality. All114 intended tensors changed;383 frozen tensors and
inactive Adam slots remained exact. Complete terminal/Adam/RNG/sampler,128saved
CE observations and all256image outputs independently verify. PNGhole gain is
0.307561%clear/0.446027%challenge with lowerface/boundary/GRAY regressions.
All44,001,540outsideRGBbytes exact;all20sheets/80cases/496cells root-reviewed.
No consistent material visible improvement;R1 and its22app/modelpins retained.

See COMPLETION_CODEFORMER_PREDICTOR_CE_V1_RESULTS.md and linkedallcase review.
Both old manualguides nowhistorical/donotrelaunch;exactpreviousbytes retained.
Nextdesign isolates geometric masked-code versus visible-context oracle effects
atfrozenR1 beforeanynewlearning.108forwards isprospectiveonly,no readyexecution
packet/guide,zeroupdates andnoautomaticlaunch.Wholegridoraclebenefit doesnot
provehole-onlyCEbest;65.927%FITlossvisibleallocation isdiagnosis,notcausalproof.
Allquality/preservation gates remain unchanged.Noepochappend orfailedrecipe
repeat,maskexpansion,checkpointrescue orcandidatepromotion.FullgoalACTIVE:
allfamilies/automatic-assisted/useracceptance/humanFINAL/CPUfreshinstall and
selectiveacceptedIMPLEMENTATION integration remain incomplete.


## Spatial-code intervention V1: manual diagnostic ready

27 October 2026. Frozen497R1 H(mask-touchedclean-code) versusV(visible-context
clean-code) diagnostic now source/transfer/archive verified. Reuse exact196A/C
controlfiles andsame48adaptivepairedcases/16labels;108forwards planned,zero
training/optimizer/gradient/teacher. Oraclesareunavailablecleaninformation,not
deployableestimates. Fixedgeometricselector/reviewedmasks;104NPZ/96PNGs,allcase
independentRAW/PNG/code/state/visual review required aftermanualrun.

Follow COMPLETION_CODEFORMER_SPATIAL_CODE_INTERVENTION_V1_VM.md. Require6GiBplus
installpayload,2GiBfloor,15minhardworker andseparate15minexternalexportlimit;
1GiBoutput/1280MiBtar+gzip/384MiBoptionalactualmutationfailurestate. All167
archivemembers/fullCRC/sourceCLI verify. NoactualVMrun ornewmodelcreated.
R1app/model22pins remain exact;RESTORATION/IMPLEMENTATION unchanged. Theprior
32updatepredictortraining failedquality andremainsretained,neverrerununchanged.
FullgoalACTIVE:allfamilies/automatic-assisted/useracceptance/humanFINAL/Windows
CPUminimum/freshinstall/acceptedselectiveport remain incomplete.


## User-directed selective integration of existing R1 with current restoration

27 October 2026. User selected a single guided implementation workflow and
explicitly required that restoration behavior stay unchanged. Preserve current
Conditioned SFT V2 p2_u0224, then offer automatic covering proposal with mandatory
manual review/correction and R1 completion. Keep restoration preparation Protect/
Apply separate from completion removal support. R1 preserves the restored canvas
outside reviewed support exactly; original/restoration/mask/result are retained.

The isolated feature checkout now contains portable R1/detector workers, editor,
exact stage binding, evidence downloads and separate verified model installation.
18 protected restoration files match the current implementation checkout exactly;
launcher changes only ASGI entrypoint. One actual browser flow and cached R1 parity
pass; latest69tests have68pass/1host symlink skip. Four combined covering cases
pass saved-pixel checks. Installer created a fresh isolated CPU environment with
verified existing model files. All nine families/control cases, fresh public
asset download, final browser regressions and publication remain incomplete.

Repeated local response transfers require diagnosis; model results are saved and
completed calls are not repeated. Pending user diagnostic priority follows AGENTS.
GitHubIMPLEMENTATION is not yet updated. New model release is draft/unpublished.
No new training/VM action, overwritten research source or quality qualification.
See outputs/completion_r1_implementation_integration_20261027_v1/INTEGRATION_STATUS.md.
Full thesis goal remains active; independent humanFINAL/minimum devices and
all-family automatic-assisted usefulness remain unproven.


## Published experimental R1 integration with preserved restoration

28 October2026. User-requested selective integration is published on GitHub
IMPLEMENTATION at6059fc32acbe04800241fc5be15f07549d30b0b9; main and RESTORATION
remain unchanged. Current Conditioned SFT V2 p2_u0224 restoration behavior and
all18 protected source/vendor/dependency files are preserved. Add an optional
proposal/edit/confirmation stage and separate frozen R1 completion worker.
Do not conflate restoration Protect/Apply with the reviewed removal mask.
Completion copies the supplied restoration canvas exactly outside that mask.

Portable installer uses public SHA-bound R1/detector assets and no research
runtime path. Fresh empty clone install/start.bat/CPU browser generation and
PNG/ZIP downloads passed.69tests ran68pass/1host fixture skip; independent
nine-case1212artifactchecks preserve1,521,399 outsideRGBbytes. Fresh-clone167
independent checks verify six assets, actual download receipts, oneV2On/oneR1,
zero surviving workers and R1=9.90seconds on the reference computer.
All nine cases/three exact256 sheets and responsive browser captures reviewed.

All nine automatic proposals in the fixed degraded-photo set were empty.
Assistance remains essential; grayscale hair falsecolor and other hidden-face
uncertainty persist. This opt-in experimental integration does not qualify
automatic detection, all-family completion, hidden identity, native CCTV final
performance or minimum devices. Independent blinded human FINAL remains pending.
Full thesis goal ACTIVE. Keep prior failed diagnostics/checkpoints/splits;
prepared spatial-code packet remains unrun and must not launch automatically.
Evidence: outputs/completion_r1_implementation_publication_20261028_v1/.
