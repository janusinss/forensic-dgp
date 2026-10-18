# CodeFormer AdaIN gradient isolation V1: preparation evidence

**17 October 2026. VERIFIED / PREPARED / UNRUN. No new model is qualified.**

[Exact five-step manual VM guide](CCTV_CODEFORMER_ADAIN_GRADIENT_ISOLATION_V1_VM.md).
[Prospective design](CCTV_CODEFORMER_ADAIN_GRADIENT_ISOLATION_V1_DESIGN.md).
The prior bounded-fusion result is independently audited and root visually
reviewed all18 sheets plus eight individual canary outputs. Its fixed gate
worsened delivered dark-face MAE on13/16 TRAIN observations relative to constant
bounds. Both bounds sacrificed some stopped-model dark improvement. Retain
those measurements; neither gate nor failed checkpoint is promoted.

## The one new training factor

Start from untouched official CodeFormer. Detach only the encoder's AdaIN style
argument from image-objective gradients during candidate training. Identical
weights must give exact unchanged forward values. Keep CE/HQ/token learning,
fusion learning, all350 trainable tensors, the original165 frozen tensors,
losses, learning rates, seeded FP32, algorithm settings and canonical TRAIN
inputs. This is a mechanism hypothesis, not a demonstrated cause of the prior
brightness tradeoff. Standard CodeFormer inference remains compatible.

Before constructing Adam, recompute391 teacher targets and require exact parity
against metadata derived from the closed prior cache. Prove the graph change
with four CodeFormer plus four LPIPS forwards and eight gradient queries on the
eligible update3 normal/dark A pair. Restore RNG, model state, empty gradient
fields, module modes and the exact wrapped callable even on failure. Norms and
digests are VM source-bound readouts; full gradient vectors are not exported.
No optimizer is constructed if these prerequisites fail.

## Frozen input, evaluation and finite work

The unchanged first50 update positions contain200 TRAIN observations from100
photographic references; distinct-person and pretraining nonoverlap are not
established. All200 saved RGB8/255 float32 hashes match the canonical renderer.
Earlier continuous/unquantized input wording remains a documented historical
error. The50 normal-B behavior anchors are unchanged. All391 HQ bindings are
TRAIN, with53 validation references kept separate.

Each snapshot has406 cases:318 paired HQ,40 older paired and48 native unpaired.
QMUL-SurvFace24 and ChokePoint24 remain separate source groups. The full61-clear
and53-normal-blur RAW/PNG preservation guards retain the original .01 mean/.03
per-case limits. Reserved FINAL pixels are untouched. These thresholds are not
the historical DGP one-percent structure gate. Native CCTV has no aligned clean
reference; synthetic paired metrics do not establish native or Zamboanga
performance, ethnicity, hidden identity or completion usefulness.

| Work and limits | Prospective ceiling |
|---|---:|
|Actual optimizer updates / backwards / gradient queries|50 /200 /9|
|Actual neural forwards / total guarded reservations|1,614 /2,073|
|Full snapshots|update0 and50;812 RAW256+PNG outputs|
|Worker / export / supervised preflight|3,000 /600 /300 seconds|
|Initialization / nested witness / actual training|600 /180 /900 seconds|
|Per snapshot / benchmark|600 /600 seconds|
|GPU reserved / post-install disk free / retained disk reserve|22 /16 /4 GiB|
|Conservative result payload / payload cap|3,442,100,480 /3,758,096,384 bytes|
|Conservative archive / archive cap|3,478,939,758 /3,865,470,566 bytes|
|Planned regular files / enforced cap|6,787 /7,000|

The snapshot artifacts are RAW256 and PNG, not812 native512 arrays. Only the
four witness paths additionally save native512 arrays. Three serialized
checkpoint slots are budgeted; a last-safe state is referenced rather than
serialized a fourth time. A retained incomplete checkpoint occupies one counted
slot. Temporary checkpoint reservation is1 GiB; metadata/logs are capped128 MiB.
The600-position learning-rate/data schedule is retained only for identical
first50 behavior; it does not authorize600 updates. No repeat or resume occurs.

## Independent source, data and storage closure

All1,933 staged/dependency files were freshly hash/size checked before freeze:
24 overlay files and1,909 immutable dependencies. The parent-data refresh
rechecked all1,925 parent static files,444 direct clean512 derivations,53 direct
validation256 targets,200 TRAIN pixels/records,50 anchors and391 HQ cache
bindings. The new upload contains31 files with4,063,063 payload bytes; the
sealed full installed protocol binds1,939 static files plus its own protocol.
No bulky checkpoint/cache upload is needed: the manual installer makes verified
regular sibling copies of1,439,410,618 bytes and preserves the originals.

48 CPU fixtures pass across worker33, transfer/preparation11 and data/resource4.
These cover pure arrays, fake graph metadata, AST/CLI, hashes, paths, finite
calls/storage and logs. They are not local neural parity or successful training.
Separate independent reviewers approved the worker, builders, extra auditors,
prospective design, staged data and resource ledger before root sealed.

All evidence is under
`outputs/codeformer_adain_gradient_isolation_v1_preparation/`:
source_readiness.json; independent_worker_source_readback_v1.json;
independent_transport_source_readback_v1.json;
independent_extra_source_readback_v1.json;
independent_design_data_readback_v2.json;
independent_data_resource_v1.json; final_resource_ledger.json;
hq_cache_parity_v1.json and derivation; parent_data_refresh_v1.json;
stage_inputs.json and execution_archive_v1.json.

The original staging instruction snapshots remain exact. Later preparation
headers in current root documents supersede their historical pending status;
there is no source/checkpoint/input rewrite. Earlier draft corrections and failed
CPU checks are retained: wrong prior cache-manifest location corrected before
metadata writes; Windows provenance-only path separator rejection corrected in
a separate auditor version; pre-seal fixture name collision followed by four
equivalent fixtures under a unique filename, without claiming byte restoration.

Protocol SHA-256: `4e2cc86857628617edbb313b6ef2911602a63ad4e1f9862afe07c57c0ba770c5`.
Archive SHA-256: `b49e476b3570b8dbe5d6160e8ee0225571f767f72a331946f4f428aab412a24e`.
Archive bytes: **1,159,959**. Exact LF checksum sidecar verified.

## What is still unverified

The new VM route witness, fresh target parity, actual training, hardware timing,
weights, all RAW/PNG gates and every output review remain unrun. Planning time
12–20 minutes is an estimate from the prior L4 work, not a new measurement.
A quality stop is a failed trial even when export succeeds. Download all three
return files for independent archive, state, trajectory, pixel and visual review.
No automatic model/app promotion, VM connection, cleanup, Drive offload or local
neural/gradient/optimizer execution occurred in this preparation. Restoration
remains first; completion/detector development waits for restoration acceptance.
User acceptance, independent final evidence and the full app goal are incomplete.
