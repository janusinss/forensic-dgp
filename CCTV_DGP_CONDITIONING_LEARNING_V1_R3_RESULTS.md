# Conditioning learning R3 loss-alignment stop — 14 October 2026

**R3 produced no learned checkpoint or image-quality gain.** It completed the
100 starting controls and420 gradient queries, then stopped before a parameter
proposal because the differentiable RAW SSIM value did not agree with the saved
scientific score. It did not reach the1% structure/preservation test. The earlier
256×256 pixel-fixture correction passed on the L4.

The reported manual VM duration is **115.99s**:
34 candidate forwards,67 recognizer forwards,420 gradient queries,zero backward
calls,zero parameter proposals and **zero accepted updates/epochs**. The100
initial raw controls match the retained original DGP exactly. Saved raw/PNG
compositions,protected pixels,200 metric rows and reconstructed initial selected
component/frozen-state schemas passed the independent audit. No fresh
post-failure RAM-state hash is claimed; the failure handler reports no rollback
error. Original app/checkpoint files remain unchanged.

All166 return members and5,852 required parent assets pass hash checks. The
121,346,525-byte result archive SHA256 is
`d0748fa47983cb562d5afaa7d3a5a5d95f2850c9a84a406ec3979c5c7872766f`. Four retained single-clear-MSE derivative samples
are finite and have the declared2,580,864-element layout. The gradient round is
incomplete; full aggregates,direction and step decisions do not exist. All420
neural queries are not independently replayed locally.
[Independent returned audit](<C:/xampp/htdocs/YEAR 4/Testing/outputs/cctv_dgp_conditioning_learning_v1_r3_analysis/failure_audit.json>).
[Bound failure and partial-gradient review](<C:/xampp/htdocs/YEAR 4/Testing/outputs/cctv_dgp_conditioning_learning_v1_r3_analysis/loss_alignment_review.json>).

## Demonstrated cause and correction

Location: returned `scripts/cctv_dgp_conditioning_learning_v1_r3_vm.py:284`.
The worker originally returned an analytic float64 pooling expression as its
RAW SSIM forward,while the scientific evaluator uses retained float32
scikit-image operations. The13 completed32-query batches plus four extra
queries identify the next check as RAW query4,blur-profile SSIM for
`v9_tr_asian_00180_blur_lr24`. This location is inferred from the counts and
fixed ordering; the original failed GPU scalar was not exported.

A CPU utility replay checks all100 real saved controls and both raw/PNG MSE,
SSIM and landmark-detail losses:600 values. It reproduces one failing RAW SSIM
row on that same case. The analytic loss is0.20168681532396471,while
the saved scientific loss is0.20168782683428255; difference
**1.0115103178387486e-06** exceeds the unchanged1e-6 alignment limit.
This is a CPU reproduction of the definitions,not an independently measured
GPU error magnitude.
[Retained failed regression](<C:/xampp/htdocs/YEAR 4/Testing/outputs/cctv_dgp_conditioning_learning_v1_r3_analysis/loss_forward_red.json>).

Fresh R4 uses the retained scientific RAW SSIM forward and the original
float64 surrogate's explicitly approximate backward,following the convention
already used for PNG SSIM. Removing this canonical-value correction restores
the exact original RAW derivative source. MSE,feature loss,mask support,
architecture,readouts,initial weights,cohort,quality/alignment bounds and finite
learning limits remain fixed. After correction **all600 values pass**; maximum
RAW SSIM difference is2.2204460492503131e-16. No local neural
model,gradient query,optimizer or parameter displacement was used for the replay.
[Passing all-case regression](<C:/xampp/htdocs/YEAR 4/Testing/outputs/cctv_dgp_conditioning_learning_v1_r3_analysis/loss_forward_green.json>).

R4 also records case,stage,query,actual/expected loss,bound and snapshot if a
future alignment check fails. The used R3 directory,packet,helper,return and
checkers are retained. R4 is a separate immutable,**verified and UNRUN** manual
packet. There have been two processing stops in this new finite-learning
sequence,not two completed improvement experiments. Any third unsuccessful
attempt invokes the existing stop-and-diagnostic-question rule.

## Visual review and pending training

All20 full-resolution sheets and300 cells were directly inspected. Their only
model column is the exact original DGP starting output. Clear eye,lip,hair,
teeth and skin detail is softened in many cases; degraded features remain
diffuse. There is no new trained output to compare or promote. The review covers
20 exposed photographic TRAIN references and100 synthetic paired cases; it is
not native CCTV,independent final review,hidden identity,ethnicity or Zamboanga
evidence. Source labels identify archives only.
[Every sheet and per-sheet observations](<C:/xampp/htdocs/YEAR 4/Testing/outputs/cctv_dgp_conditioning_learning_v1_r3_analysis/visual_review.json>).

The R4 pilot still trains only12 active conditioning tensors,freezing the
original decoder,readouts,bank,reference and recognizer. At most two accepted
full-cohort updates and three actual trials each; refresh all64 current-state
RAW/approximate-PNG derivatives after acceptance. Every accepted step needs
actual preservation across17 RAW and17 PNG metric groups,positive degraded
detail globally/per-source,brightness<=0.2 and cumulative parameter distance
<=0.1%. The endpoint requires1% structure gain globally and per-source on both
raw and delivered PNG. All partial accepted states and rejected trials are
exported. No longer-epoch continuation or app adoption is automatic.
[Current manual upload,tmux and download commands](<C:/xampp/htdocs/YEAR 4/Testing/CCTV_DGP_CONDITIONING_LEARNING_V1_R4_VM.md>).

Preparation verifies all14 archive files,ASCII/LF checksum,5,852 parent assets,
initial saved-state schema,100 scientific baseline cases,the600-value forward
regression,nonfinite rejection,retained gate boundaries and the four previously
rejected proposals. Windows execution is rejected before neural import; Bash
syntax and all command namespaces are checked. These are preparation checks,
not a new CUDA training result. Require9GiB free at launch; the original finite
time,VRAM,RSS,storage and export stops remain.
[Transfer and narrow repair audit](<C:/xampp/htdocs/YEAR 4/Testing/outputs/cctv_dgp_conditioning_learning_v1_r4_transfer_audit.json>).

The permanent baseline checkpoint SHA256 remains
`646fbb11674e8882ea41e251d6e6e7d468c963979b48c93132def8292834e24b`.
Original app design,Auto/override,input-only review,checkpoints/splits/caches/
provenance/full states/failures,all five milestones and all seven automatic/
assisted completion families remain required. Full TRAIN's10% structure gate,
useful native DEV and independent final/app qualification remain pending.
Actual training stays human-manual on the L4; this return audit performs no VM
maintenance or training launch. The full goal remains active and incomplete.
