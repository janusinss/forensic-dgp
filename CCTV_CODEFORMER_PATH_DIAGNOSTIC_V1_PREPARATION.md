# This diagnostic has returned — 16 October 2026

**COMPLETED AND AUDITED. Do not relaunch these historical commands.**
The diagnostic performed zero optimizer updates and created no new model.
[Current results](CCTV_CODEFORMER_PATH_DIAGNOSTIC_V1_RETURN_REVIEW.md) and
[next design](CCTV_CODEFORMER_FP32_COMPARISON_V1_DESIGN.md) supersede the old
PREPARED/UNRUN status below. The next design has no runnable packet yet.
All previous preparation/guide bytes are retained unchanged as history.

---

# CodeFormer path diagnostic prepared — 15 October 2026

**PREPARED / UNRUN. No actual neural, gradient, optimizer or VM call occurred
during this preparation.** This is a measurement to guide further restoration
training; it does not yet improve or qualify a checkpoint.

The latest user preference accepts moderate brightening for clearer faces. V2
has genuine paired-image gains, including higher SSIM on 50/53 HQ blur cases,
while three clear faces regress in both fidelity and SSIM. The original V1/V2
stops remain binding evidence. Improving visibility and preserving features are
both necessary; brightness differences alone cannot decide usefulness.

## What is prepared

Eight exact retained TRAIN inputs, four genuine photographic clean512 targets,
and two saved FP32 behavior-anchor pairs are staged. RGB8 and float32 hashes
prove the saved PNGs reproduce the actual V2 input tensors exactly. The targets
match their recorded hashes. No degraded camera input was regenerated, no DEV
case or final identity selected, and no native crop was treated as paired data.

The package checks original/stopped states under four numerical paths, repeat
stability, and conditional illumination-moment/fusion interventions. Component
gradient signatures reuse the selected existing graphs, using exact saved
teacher targets. There is no optimizer construction, backward accumulation,
weight update, hybrid-checkpoint export or application-default change.

| Finite limit | Bound |
|---|---:|
|CodeFormer / HQ / LPIPS forwards|104 / 4 / 10|
|Gradient queries / optimizer updates|64 / 0|
|Worker / export time|900 / 300 seconds|
|Peak CUDA reserved memory|22 GiB|
|Initial free space / retained reserve|6 / 4 GiB|
|Result payload / compressed result archive|256 / 272 MiB|

These are prospective caps, not measured VM timing or evidence the CUDA path
has run. The manual supervisor also bounds transfer preflight to 300 seconds.

## What was verified locally

The independent source review passed **36 CPU fixtures**. They cover exact
TRAIN role/input/target bindings, stale or unreviewed design refusal, scalar
gradient summaries, scoped restoration, call accounting, memory checks, manual
Linux/tmux guards, live-worker export refusal, competing process classification,
standalone supervisor startup, and corrupt/duplicate/escaping archive refusal.
Source/algebra fixtures do not execute model forwards or autograd queries.

The source review fixed peak-memory enforcement and selected exact saved
behavior targets before freezing. Earlier source drafts/readbacks are retained.
The packet contains 57 files / 17,084,686 uncompressed bytes. The execution TAR
contains those exact members and every member hash was reread after creation.
Checksum output uses LF. Original checkpoints, frozen V1/V2 recipes, split roles,
failed guard records, the returned archive and application state stay unchanged.

The 7,445,777-byte upload excludes large parent model assets. The new worker
requires the exact existing V2 sibling directory with its original CodeFormer,
stopped50 state, HQ teacher and LPIPS assets. Their content hashes are checked
again on the VM before execution. The independent local audit binds copies from
the retained packet/return; it cannot establish current remote-file presence.

## Evidence and next action

| Evidence | Location |
|---|---|
|Independent source approval|[source readback](outputs/codeformer_path_diagnostic_v1_preparation/independent_source_review_r1.json)|
|Exact staged input readback|[stage readback](outputs/codeformer_path_diagnostic_v1_preparation/stage_readback_v1.json)|
|Protocol seal|[sealed sources](outputs/codeformer_path_diagnostic_v1_preparation/sealed_sources.json)|
|Execution archive|[archive receipt](outputs/codeformer_path_diagnostic_v1_preparation/execution_archive.json)|
|Manual upload/launch/download|[five-step guide](CCTV_CODEFORMER_PATH_DIAGNOSTIC_V1_VM.md)|

The next action is the user's manual tmux diagnostic. After return, independently
audit output hashes, call counts, actual tokens, repeat/path differences,
intervention invariants, unchanged model states and gradient summaries. Review
every planned RAW/PNG comparison before deciding a new finite training recipe.
No new epochs or automatic reruns are authorized by preparation completion.

Restoration remains the immediate priority. Native CCTV usefulness, user
acceptance, independent final review and full app verification are incomplete.
Completion and covering-detector development remain deferred. The full thesis
goal is active; this preparation does not mark it complete.
