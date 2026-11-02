# Reviewed covering completion: trained R2

Trained CodeFormer Completion R2 is the accepted Unified R2 v3 continuation of
our R1 adaptation of S-Lab CodeFormer **inpainting**. It is separate from
Conditioned SFT V2 restoration. This update retains the restoration model,
preparation, weights, inference and output quantization.

The guided application uses restoration, a manually selected covering area,
and optional R2 completion in that order. R2 receives the exact reviewed
prepared/restored RGB256 canvas and a binary removal mask. The retained
512-code inpainting architecture uses local CPU FP32 inference at 512 pixels,
fidelity 1 and AdaIN off, followed by 256-pixel delivery and round-to-nearest
PNG. Every pixel outside reviewed support is copied exactly from the supplied
canvas. Empty reviewed masks bypass neural inference. Completion adds no
mask margin or post-completion restoration.

## Admitted inference assets

| Separate inference asset | Bytes | SHA-256 |
|---|---:|---|
| `codeformer_completion_r2.pth` | 370773572 | `ed9f92baf06a73090343ad00cdc23b8493c9dde738a90cde9bdb650e69fee07c` |
| `covering_detector_segmenter_v1.pth` | 2681549 | `ccc22d524073368bc4f5d6b6d2136c90d0e2a117083e4c75fa98d42d7bfff6cf` |

R2 contains 497 frozen inference tensors with logical state SHA-256
`300b13b8f01db17aa95ee95ca41dbbdfbd4e5e3257dd16657fcc7950d3351ea0`.
The training protocol SHA-256 is
`463c1977e9cd74f0a4fd0240df7a784f48b613043ef667714f1ef94adb4cd60a`.
The upstream inpainting checkpoint remains SHA-256
`b0d1b868c3dacf75d637bbcc0d4b3dd4ce2a064a338dfb4d35c740d3b4ae8797`,
from source revision `b33cc7d639d6545bfcccc7e0bc6ae51f24e79c2b`.
Training optimizer/RNG states and research archives are excluded from deployment.

Retain [S-Lab License 1.0](third_party/codeformer_completion/LICENSE), the
[notice](third_party/codeformer_completion/NOTICE.md), the unchanged
[40-source attribution](third_party/codeformer_completion/TRAINING_ATTRIBUTION.json),
and [R2 lineage](third_party/codeformer_completion/R2_PROVENANCE.json).
The R2 protocol uses the same 40 attributed source photographs; the historical
R1 attribution file retains its original protocol binding. R2 provenance records
the continuation separately. These terms do not establish blanket deployment
permission or identity clearance.

## Research evidence and limits

The retained 2 November 2026 research report records R2 promotion after 60
numeric development gates and review of 20 comparison sheets. Against R1,
delivered-PNG masked-region L1 improved 1.052681% on 32 clear monitor cases
and 2.687341% on 16 challenge monitor cases. The retained audit reports
44,001,540 exact outside-mask RGB bytes across 256 evaluated PNGs.
These are adaptive synthetic development observations, not independent FINAL
accuracy, native CCTV hidden-face truth or recovered identity.

The earlier numeric-admission receipt still contains `visual_review_pending:
true` and `model_qualified: false`; the later research report records the visual
review and promotion. Preserve both historical records. Research promotion does
not certify full-family usefulness, identity, or minimum hardware. The application
continues to report `model_qualified: false` and `hidden_identity_verified: false`.
Pretraining identity overlap is unknown. No Zamboanga-specific performance claim
is made. Generated anatomy, expression, color, texture and boundaries can be wrong.

## Covering selection and preservation

The guided editor uses manual covering selection. The unchanged compact epoch-2
detector remains available through the underlying API. It thresholds at 0.5 and
adds one reviewable 3-pixel proposal margin, clipped to captured image support.
It can miss coverings or mark visible regions. Every proposal requires review;
clear glasses, ordinary hair and uncovered faces should retain an empty corrected
mask. Empty automatic predictions are not a face-quality or uncovered-face decision.

Completion preserves the supplied prepared/restored canvas outside the reviewed
area; that canvas can differ from the original upload. Insufficient-information
refusal combines user review with a frontal-crop coverage heuristic. It is not
a calibrated face recognizer or universal blur threshold. Request a clearer or
less-covered crop when meaningful facial structure is unavailable.

## Update and runtime behavior

R2 uses a distinct filename. Existing R1 model files, previous app versions and
saved records are preserved. Missing or changed R2 weights refuse completion;
there is no silent fallback to R1. Exact local assets can be supplied through
`implementation.completion_install --local-r2` and `--local-detector`.

The new immutable model distribution targets `completion-r2-models-v1`.
Publication and acceptance of the next EXE are pending in this working tree.
Old R1 releases remain historical artifacts. No new Windows CPU timing or
arbitrary-hardware guarantee follows from the research L4 run. The existing
2.5 GiB available-memory admission check, 6 GiB worker commitment cap and
120-second stage limit do not certify minimum hardware. Inference is local;
the source installer downloads missing assets during setup, and an offline EXE
must bundle the admitted assets before publication.

Preserve the original, restoration, reviewed mask, model provenance and final
estimate when interpreting or sharing output.
