# Conditioning learning R2 startup failure and R3 repair — 13 October 2026

**R2 stopped before model construction, inference or training. It did not reach
the structure/preservation gate.** The new worker's canonical-pixel fixture was
16×16 while its unchanged delivery helper requires 256×256. This is an
implementation bug in the prepared startup check. Earlier preparation verified
transfer, state schemas, metrics and gates, but missed exercising this fixture
against the real helper.

Location: returned `scripts/cctv_dgp_conditioning_learning_v1_r2_vm.py:152`, calling
`conditioning_losses.py:10`. The fixture's actual shape is `[1,3,16,16]`; the
helper rejects it before the first model constructor. The traceback, failure
record and all original source bytes remain retained.

The VM failure reports **0.114658 seconds**, zero
candidate/recognizer forwards, zero gradient/backward calls, zero proposals and
**zero accepted updates**. Only `failure.json` exists in the run-output stage.
There is no initial model-state receipt, improved raw/PNG image, new checkpoint,
quality measurement or visual sheet. No fresh post-failure memory-state hash is
claimed. The export's `complete: true` certifies archive creation only.

All **16 returned bindings** and **5,852 parent
assets** passed the independent import checks. The 589,270-byte
archive SHA256 is `a90ad5b133bec9dba10d66883300e43d1a2d50ab571e08cf27c85a4de601a12a`. Protected app/checkpoint files
remain unchanged. [Independent failure audit](<C:/xampp/htdocs/YEAR 4/Testing/outputs/cctv_dgp_conditioning_learning_v1_r2_analysis/failure_audit.json>).
[Source-bound startup review](<C:/xampp/htdocs/YEAR 4/Testing/outputs/cctv_dgp_conditioning_learning_v1_r2_analysis/startup_review.json>).

## Demonstrated repair

A CPU utility regression extracts only the sealed worker's actual NumPy fixture
expression and calls the unchanged delivery helper. It reproduces R2's exact
shape assertion. The same regression passes the fresh R3 fixture at
`[1,3,256,256]`, checking all 256 byte levels and exact float32 RGB / float64
feature values. It constructs no neural model and performs no gradients or
parameter changes. This establishes the shape correction locally; it does not
establish a new CUDA run or useful restoration.
[Retained failing regression](<C:/xampp/htdocs/YEAR 4/Testing/outputs/cctv_dgp_conditioning_learning_v1_r2_analysis/pixel_fixture_red.json>).
[Passing regression](<C:/xampp/htdocs/YEAR 4/Testing/outputs/cctv_dgp_conditioning_learning_v1_r3_pixel_fixture_green.json>).

R3 uses a separate immutable directory and filenames. The worker differs only
in that fixture and the namespace. The model's namespace and wrapper paths are
updated; architecture, readout initialization, helper/loss/optimizer code,
initial state, cohort, scientific thresholds and resource limits are unchanged.
The failed R2 packet, return, logs, receipts and checkers remain intact.

R3 is **verified and UNRUN**. It trains only the 12 active conditioning tensors,
with at most two accepted full-cohort updates and three actual trials per update.
Each accepted change needs actual raw/PNG preservation, brightness <=0.2 and
positive detail progress globally and in both source groups. All 64 current-state
loss derivatives are refreshed after acceptance. The endpoint still needs 1%
structure gain globally and per source, on both raw and delivered PNG; cumulative
parameter distance remains 0.1%. All partial accepted components and rejected
trials are exported with true counts. No longer epochs or app adoption follow
automatically. [Verified manual commands](<C:/xampp/htdocs/YEAR 4/Testing/CCTV_DGP_CONDITIONING_LEARNING_V1_R3_VM.md>).

Preparation checked all 12 archive members, exact ASCII/LF checksum, all 5,852
parent hashes, initial saved-state reconstruction, 100 real saved baseline cases,
scientific gate boundaries, rejection of nonfinite metrics, all four previously
rejected proposal groups, Windows pre-neural rejection and read-only Bash syntax.
[Transfer and repair audit](<C:/xampp/htdocs/YEAR 4/Testing/outputs/cctv_dgp_conditioning_learning_v1_r3_transfer_audit.json>).
These checks are preparation evidence; the actual new manual L4 run is pending.

The previously audited activation result remains valid. Neither that result nor
this startup repair demonstrates improved native CCTV faces or more-epoch
qualification. The original app checkpoint/design, final identity separation,
paired/unpaired boundaries, five milestones and seven automatic/assisted
completion families remain. Full TRAIN's 10% structure requirement and useful
native DEV / independent final review remain separate. All actual training is
human-manual on the existing L4. No VM cleanup, direct training launch or app
change was performed during this return audit. The full goal remains active
and incomplete.
