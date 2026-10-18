# Bounded-fusion feasibility V1 prepared — 16 October 2026

**VERIFIED / PREPARED / UNRUN. Zero local neural, gradient, optimizer or VM
calls. This packet measures a prospective route; it does not train or qualify
a new restoration model.**

[Five exact upload/install/tmux/launch/download steps](CCTV_CODEFORMER_BOUNDED_FUSION_FEASIBILITY_V1_VM.md).
[Previous actual FP32 training return](CCTV_CODEFORMER_FP32_COMPARISON_V1_RETURN_REVIEW.md).

The FP32 comparison completed real training and fixed its initial numerical
mismatch, but retains the same three clear-preservation failures and normal-blur
photometric failures as BF16 V2. Both states improve many paired blurred/dark
cases. More unchanged epochs are unsupported. The next route must retain useful
visibility while limiting changes to normal facial appearance.

The [prospective design](CCTV_CODEFORMER_BOUNDED_FUSION_FEASIBILITY_V1_DESIGN.md)
uses original fusion as a base and measures a bounded difference from the
stopped FP32 fusion. A continuous gate uses only observed input pixels.
Its calibration shows substantial normal/dark overlap; it is a crude luminance
probe, not an exposure detector or an established solution. A learned gate also
has an initial zero-delta gradient limitation. These findings justify a finite
zero-update feasibility measurement before spending on two-arm training.

Five paths compare original CodeFormer, stopped FP32, original fusion with
stopped content, bounded constant fusion and bounded input-conditioned fusion.
All use identical observed inputs; stopped-content tokens, logits, skips and
pre-generator AdaIN features must match exactly across the four relevant paths.
Repeats, original/original zero-delta and native-identity canaries check exact
parity and hook restoration. All original/stopped state tensors stay unchanged.
No adapted `.pt/.pth` is exported.

The cohort is **32 observations from 16 matched photographic TRAIN references**,
selected prospectively across the photometry distribution of 200 retained TRAIN
PNGs. Reference IDs do not establish unique people. The measurement decodes
those exact saved RGB8 PNGs to float32; it does not replay the earlier continuous
training tensors. All 200 input hashes and the input-only calibration were
independently verified, including low-luminance normal and brighter-dark stress
canaries. No output/clean-target scores choose the gate or its constants.

All 16 genuine clean sources retain attribution and terms. Direct source1024
to LANCZOS512/256 targets are bound for later paired metrics only; targets never
enter routing or model inputs. No native CCTV, DEV or reserved FINAL pixels are
used. Earlier native source/overlap/terms limits remain intact; no ethnicity,
identity recovery or Zamboanga performance claim follows.

The exact successful budget is **170 full CodeFormer forwards, 960 internal
fusion-block evaluations and zero teacher, LPIPS, gradient, backward or optimizer
calls**. Reservations precede calls. All 170 native512 arrays, delivered256 RAW
arrays, floor-converted PNGs and selected tokens are saved. Selected internal
canary arrays permit CPU replay; other internal statistics are not a full
activation replay. Exact content/parity remains strict. A separately reviewed,
prospectively fixed CPU/CUDA arithmetic allowance covers only feature-canary
rounding; it changes no restoration quality threshold and cannot be refitted
after seeing the return.

Independent source, data, numerical and resource reviews passed. **44 CPU
fixtures passed**, including wrong-gate rejection, schema/call mismatches,
Windows/non-tmux rejection before framework imports, finite stage/memory/write
stops, retained failures, PID/process-group boundaries, archive corruption and
LF checksum handling. These tests do not establish actual L4 feature routing
or image quality; the manual measurement provides that evidence.

The original 282 staged files remain exact. Ten reviewed additions fit the
256-KiB/16-file sealing reserve. Final archive: **293 regular members,
45,093,599 payload bytes, 43,604,878 compressed bytes**. Every member/hash and
exact namespace passed sealing readback. External original/stopped weights
are freshly hash-checked on the VM; no model weights are reuploaded or copied
into this packet. The current FP32 folder is a required read-only dependency.

The complete conservative result/source/failure ledger budgets **800,661,299
payload bytes**, **821,237,532 archive bytes**, **32,972,595 metadata bytes** and
**2,240 files**. Enforced ceilings are 1.25 GiB payload, 1.35 GiB archive,
32 MiB metadata and 2,500 files. Noncheckpoint temporary reserve is 128 MiB.
Require 8 GiB free after install and preserve 4 GiB. Current/peak CUDA reserved
memory is capped at 22 GiB; host peak RSS is reported, not forecast as a bound.

Initialization/measurement/whole-worker caps are 300/600/900 seconds; export
is capped at 300 seconds. The supervisor adds finite termination grace and
caps logs before writes. An idle tmux session alone is not a competing workload.
Active GPU/research processes remain untouched. No automatic launch, retry,
historical pilot, unchanged training continuation, cleanup or app adoption.

Results must be independently audited on return, with all five paths, individual
normal/dark/stress regressions and every planned sheet shown. A good aggregate,
a gate that varies with brightness or a completed export cannot authorize
training automatically. A positive measurement only supports a separately
reviewed finite training design. Existing full 61-clear/53-normal-blur DEV guards
remain unchanged and untested by this small TRAIN diagnostic. User restoration
acceptance and the full thesis goal remain outstanding; detector/completion
development stays deferred.

Protocol SHA-256: `c3b0f5f0f3ff83b508625df0259ed0c9acbc2d071da2fdde066a2bda429a15eb`.
Archive SHA-256: `9e355da89976885ad4d9935b483a1df8f449c916fe9dd0d480973ffa58af77d6`.

[Independent data/resource proof](outputs/codeformer_bounded_fusion_feasibility_v1_preparation/independent_data_resource_v1.json),
[final source approval](outputs/codeformer_bounded_fusion_feasibility_v1_preparation/final_review_approval_v1.json),
[numerical readback](outputs/codeformer_bounded_fusion_feasibility_v1_preparation/numerical_policy_readback_v1.json),
[sealed archive readback](outputs/codeformer_bounded_fusion_feasibility_v1_preparation/execution_archive_v1.json).
