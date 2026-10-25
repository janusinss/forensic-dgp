# Measure RAW and PNG-surrogate learning signals before another update

23 October 2026. **Source-only design; no worker, transfer file or launch guide
exists for this follow-up.** The [audited gate-aligned pilot](CCTV_CODEFORMER_U64_GATE_ALIGNED_PILOT_V1_RETURN_REVIEW.md)
stopped at update two. Its sparse RGB8 code crossings explain the delivered PNG
regression even though its actual RAW steps and RAW monitor errors improved.
Original u64 is retained; neither failed update two nor earlier update one is
selected as a successor.

## Selected diagnostic question

Can learning signals that explicitly include the existing delivered PNG values
agree with the RAW fidelity/structure signals in the same original-u64 head
space? Answer this before choosing another finite training recipe. A smaller LR,
more epochs or the unchanged eight-RAW-gradient recipe would not measure this
missing information.

The proposed measurement starts again from the exact complete original trained
u64, including frozen R2/U96. It makes **zero updates, zero parameter
applications and zero optimizer constructions**. All 575 tensors must remain
bitwise unchanged. It is a separately reviewed future manual-L4 diagnostic,
not a local neural experiment or an automatic continuation of a stopped run.

## Finite measurement scope

1. Bind the same 16 admitted optimization TRAIN photographs, their four roles
   and targets from the sealed protocol. No monitor, guard, native, wild, DEV
   or FINAL target is introduced for gradient calculation. Photograph separation
   remains distinct from identity/pretraining-overlap proof.
2. Require four historical original-u64 RAW/PNG parity forwards and four full
   same-state repeats. Measure all 64 admitted optimization cases once. Maximum
   **72 full forwards and 256 separate head-gradient queries**: RAW MAE, RAW
   adjacent error, PNG-surrogate MAE and PNG-surrogate adjacent error per case.
3. Preserve the exact exporter: native512 clamp before bilinear256; RGB8 codes
   use the existing FP32 multiply by 255 followed by truncation/floor. The
   surrogate's forward values must match the actual saved PNG decoded to FP32
   divided by 255, bitwise, for every measured case. A nearest-integer fake
   quantizer, observer-calibrated scale or altered output renderer is unsuitable.
4. Use an explicit, labeled custom backward identity **only as a surrogate**
   across the floor conversion. Treat each 256-pixel delivered RAW value as an
   input to that surrogate. Its exact-forward custom operation returns a
   detached clone of the actual saved PNG, decoded to FP32 and divided by 255;
   backward returns the supplied gradient unchanged to RAW and none to that
   detached reference. The worker must bind the reference to its actual PNG
   readback. This avoids claiming that a floating cancellation expression or
   GPU quantizer necessarily reproduces every PNG bit. Leave the model's real
   clamp/resize path and all gates intact.
   Preserve connected graph/scalar/state provenance for every query, plus any
   measured zero or explicitly unused coordinate; do not hide a missing query.
5. Independently audit all gradients, exact forward parity, full-state equality,
   repeated outputs, per-photo and pooled vectors. Four roles × two metrics ×
   two domains produce **16 pooled vectors**, not eight. Report their magnitudes,
   zero/cancellation cases, pairwise conflicts and hypothetical direction dots
   separately. Do not silently feed 16 vectors to the existing eight-vector
   solver, drop a domain/role or apply any direction during measurement.

Local-region measurements stay central128; other roles stay whole256. Use
photo-major fixed ordering and retain each individual query before pooling.
Pool all 16 photographs at one unchanged current state; normalize only after
pooling. RAW and PNG-derived vectors must remain separately named. A zero vector
or absent common direction is informative evidence, not permission to remove
the corresponding endpoint.

The intended finite resource envelope is 900 seconds internally, 960 seconds
externally plus at most 20 seconds cleanup; phases 180 seconds. Export is
separate at 300/330 seconds. Require 8 GiB free and retain 4 GiB; RAM 10 GiB,
reserved CUDA 18 GiB, outputs/archive 2 GiB each. These are **requirements for
an unimplemented worker**, not observed runtime or implemented enforcement.
Retain complete initial/closing575 state, all 256 gradient witnesses, every
RAW/PNG output and eight native/logits/LQ parity/repeat triples, plus operation,
resource, RNG and failure receipts. No automatic retry or historical launch.

## Interpretation and later training decision

Hard floor quantization has zero derivative almost everywhere and discontinuities
at code boundaries. The identity backward rule is a biased estimator, not the
true derivative of the delivered PNG gate. [Bengio et al., 2013](https://arxiv.org/abs/1308.3432)
describe the straight-through heuristic; [Yin et al., ICLR 2019](https://arxiv.org/abs/1903.05662)
show that estimator choice matters in their particular binarized-network setting.
[PyTorch's QAT explanation](https://pytorch.org/blog/quantization-aware-training/)
uses simulated quantization during learning. These sources motivate this
measurement; their results do not establish face restoration or PNG preservation.

A negative surrogate dot product cannot certify a finite PNG change, useful
image detail or preservation on held-out faces. Any later proposed update must
retain both RAW and actual PNG gates, original teachers/limits and complete
quality-bank review. It needs its own finite update count, actual rounded-step
checks, source review and manually launched VM packet. No longer training,
optimizer, LR schedule or new model architecture is selected by this design.

Broader DEV failures remain binding. U96 stays the experimental app incumbent;
original u64 stays the retained challenger. Reviewed Protect/Apply/bypass remains
separate, with automatic matching and completion development deferred. Native
and wild evidence is unpaired; FINAL stays unopened/execution-unfrozen, reviewers
pending. The full goal remains **ACTIVE/incomplete**.
