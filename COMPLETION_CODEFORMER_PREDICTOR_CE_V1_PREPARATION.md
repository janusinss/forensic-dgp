# Current status: pilot executed; quality rejected

27 October 2026. The prospective preparation below is historical. All32 new
updates ran; state/output audits pass but quality fails. R1 stays selected.
Read COMPLETION_CODEFORMER_PREDICTOR_CE_V1_RESULTS.md for actual measurements.
Do not relaunch or append epochs. Original preparation bytes remain below.

---

# Predictor-only completion V1: ready for a manual L4 pilot

27 October 2026. **PREPARED, UNRUN; NO NEW TRAINED MODEL.**
R1 remains the user-selected working completion model. The complete thesis goal
is active. [Use the verified five-step guide](COMPLETION_CODEFORMER_PREDICTOR_CE_V1_VM.md).

The previous [same-codebook diagnostic](COMPLETION_CODEFORMER_SAME_CODEBOOK_PROBE_V1_RESULTS.md)
showed that clean-reference codes can help at fixed R1 fusion, while removing
fusion with actual predicted codes worsens results. Those oracle improvements
use unavailable clean information. This pilot tests whether ordinary covered
inputs can learn better predictions; it does not supply oracle codes at inference.

## Exact prepared scope

Start from the retained complete R1 weights, mapped Adam moments at step32 and
parent RNG. Learn114 existing predictor tensors,19,450,880 scalars. Keep383
encoder/fusion/prior tensors frozen and preserve the original512-code prior.
Use CE only with LR1e-5, four one-case microbatches per step,32 new updates and
128 backwards maximum. No optimizer reset, encoder/fusion updates, latent MSE,
pixel-loss surrogate, new teacher generation or hidden clean input at evaluation.

The balanced128 observations are a fixed subset of the retained256 FIT pool,
from32 photograph reference IDs, using64 exact RGB/GRAY FIT label sets. Keep
all16 label sets from the eight adaptive monitor reference IDs out of training.
Photograph-ID separation does not prove distinct people or no pretraining overlap.
All original masks, targets, source attribution, terms, splits and failures remain.

All48 paired cases are read out at0/8/16/32, with32 unpaired public photographs
at0/32 (or the stopped terminal cycle if a catastrophe replaces32). Save256
evaluation outputs,128 FIT logits/labels and four fresh parity bundles, plus80
exact teacher-array copies. Complete execution plans384 network invocations:
128 code-only and256 full generation calls, with168 exact empty bypasses.

Four fresh parity pairs and all48 initial R1 RAW/PNG historical matches precede
updates. Intermediate8/16 readouts are safety-only. Final32 must pass separate
clear/challenge>=1% PNG hole-error reduction, all existing RAW/PNG hole/boundary
subgroup checks, exact outside pixels and a new eight-case gray-style nonworse
guard. Lower CE, code agreement and a valid checkpoint do not qualify a model.
No earlier-checkpoint rescue, automatic epochs, repeat or app promotion follows.

## Limits and recovery

Require8GiB free after installation; the installer additionally reserves its
exact payload before extraction. Retain a2GiB floor. Worker hard cap1800s,
cooperative deadline1680s and120s save reserve; CUDA allocation20GiB maximum.
Whole output2.5GiB, terminal/failed partial1GiB, raw streamed tar/gzip2.75GiB
each, export900s. These are enforced caps, not measured timing guarantees.

One bounded terminal attempt retains actual332 non-prior tensors, merged278
historical code-Adam slots with114 updated,54 frozen fusion slots, RNG, sampler,
progress and partial gradients. Unexpected mutated prior165 tensors are retained
as failure evidence within the reservation. Internal ZIP/pickle/CRC checks run
on the closed bounded partial before commit. Actual scientific failures are
recorded rather than blocking the resource-bounded failure save. Hardkill or IO
failure can still prevent recovery; preserve partials and never auto-resume.

## Verified files and evidence

Original Windows transfer folder: C:\xampp\htdocs\YEAR 4\Testing\outputs.
Archive: cctv-completion-codeformer-predictor-ce-v1-execution.tar.gz,
73,748,641 bytes. SHA256:
`1052d09627cdd69f5e652a6996cee9909e5498ba516cb2b5900981b066a458c3`.
Protocol SHA256:
`15aa3e1478a67c120b2fcb6af50ecae58d0558eb4e65cef473d0339572bf084b`.
The LF checksum and matching guide copies are published there. The archive has
671 regular members, including protocol; payload79,488,873 bytes. Original and
R1 weight artifacts are reused by exact hash on the existing VM, not uploaded again.

Verification includes23 author source tests;4,014 plain CPU tensor fixtures;
2,036 independent source-semantic checks;2,839 independent source/data checks;
the actual unmodified import-blocked transfer CLI (exit0);root seven terminal
and nine installer toy-file fixtures; and3,398 independent final archive/guide
checks. Root separately streamed every archive payload and gzip CRC and checked
all five single-remote SCP commands, tested installer literal, exact tmux launch,
published copies/LF checksum and22 unchanged R1 app/model pins.

These are preparation checks. No neural model, actual checkpoint learning,
optimizer update, actual VM preflight or VM connection was run in preparation.
Plain-logit mathematical/autograd fixtures are distinct from model training.
An unused frozen legacy face_workflow.py snapshot has a newer-Python hash helper;
the actual Python3.10 VM execution path does not import it. All selected runtime
and transport sources were parsed and checked for the VM interface.

Read [the frozen design](COMPLETION_CODEFORMER_PREDICTOR_CE_V1_DESIGN.md), its
root pre-release clarifications, and source/transfer admissions under
research/completion_codeformer_predictor_ce_source_audit_v1. The design's earlier
runtime-unready wording is historical; this preparation record is current.

The next action is the user's manual Windows upload, VM install and tmux run.
Return all three download files for independent weights/Adam/RNG/sampler/logit/
RAW/PNG and all-case visual audit. No new app candidate exists yet. Useful real
covering-family results, automatic-versus-assisted evaluation, user acceptance,
blinded independent FINAL, minimum Windows CPU/fresh-install checks and accepted
selective IMPLEMENTATION integration remain incomplete. RESTORATION and the
current R1 app/checkpoint remain unchanged.
