# Complete-U96 post-fusion actual-learning V1

Manual L4/tmux only. Independent review and new optimization admission required before sealing.
Worker API: `run(root,p,expected_sha)`. Root owns packet/CLI/export/auditor.

```json
{
  "format": "u96-postfusion-image-anchored-B64-v1",
  "max_updates": 64,
  "microbatches_per_update": 4,
  "slot_roles": [
    "local_gaussian_1p2",
    "global_gaussian_1p2",
    "clear",
    "lowlight_linear_0p35"
  ],
  "sampler_offsets": [
    0,
    4,
    8,
    12
  ],
  "snapshot_updates": [
    0,
    1,
    4,
    16,
    32,
    64
  ],
  "warmup_updates": 4,
  "gradient_clip_max_norm": 1.0,
  "optimizer": {
    "type": "Adam",
    "lr": 5e-05,
    "betas": [
      0.9,
      0.99
    ],
    "eps": 1e-08,
    "weight_decay": 0.0,
    "amsgrad": false,
    "foreach": false,
    "fused": false,
    "capturable": false,
    "differentiable": false,
    "maximize": false
  },
  "loss": {
    "image": 1.0,
    "adjacent": 0.25,
    "local_inside": 0.5,
    "local_outside": 0.5,
    "clear_anchor": 2.0,
    "dark_illumination_anchor": 2.0
  },
  "local_half_open_roi": [
    64,
    64,
    192,
    192
  ],
  "luminance_weights": [
    0.2126,
    0.7152,
    0.0722
  ],
  "illumination_block_size": 16,
  "initial_heads_canonical_sha256": "2e7826c07d31298eecf07d7f4d53286aa86247c00157dea9005242f23902726c"
}
```

## Fixed lineage and admission

Complete U96 means exact R2 515 tensors plus retained U96 24 adapters, all539
frozen. Only36 new post-fusion tensors/168960 parameters may train. Immutable
prototype source/design/review and both incumbent files are direct size/hash
bindings. This continues from complete U96; it does not rebase to original
CodeFormer. The initial head canonical state must equal the value above.

The exact96 new TRAIN rolecases comprise16 optimization photographs/64cases
and8 TRAIN-monitor photographs/32cases. Each has clear, local Gaussian1.2,
global Gaussian1.2 and linear-sRGB exposure0.35 roles. Every degraded input needs
new input-only readability admission; clear inputs require source/geometry
review. Photograph groups define splits. The4 earlier preflight photographs and
8 historical guard photographs are excluded from both new cohorts.
The24 legacy guards retain historical initial metrics and teacher bindings;
their dark targets remain null. Deferred cases are not silently admitted.
No DEV/native/FINAL case is optimized.

Each new case binds id,reference_id,role,cohort,split,input,target,RGB8 hashes and
optimization_permitted. Cohorts are optimize/monitor; only optimize permits
updates. Protocol learning equals the constant above plus frozen16-ID
optimization_reference_order and8-ID monitor_reference_order.

## Connected image objective and sampler

64 updates maximum,4 distinct-photo singleton microbatches per update:
local/global/clear/dark slots use photo offsets0/4/8/12 at
(update-1+offset)%16. 256 exposures are exactly4 passes over64 optimization
rolecases. Each scalar objective is divided by4 before backward. Fresh Adam
uses the explicit parameters above,4-update linear LR warmup then constant LR,
global head-gradient clipping norm1, no EMA/GAN/additional network.

Delivered FP32 native RGB is mapped from[-1,1], clipped to[0,1], bilinearly
resized to256 with align_corners=False. D means RGB L1 residual and E adjacent
residual L1, counting only edges wholly within the selected region. M is the
known central half-open[64:192,64:192] footprint, C its complement:

- Local: .5(D_M+D_C)+.125(E_M+E_C), against aligned clean.
- Global: D_all+.25E_all, against aligned clean.
- Clear: 2(D_all+.25E_all), against detached same-input complete-U96 RAW.
- Dark: D_all+.25E_all against clean, plus2 times block16 luminance L1
  difference from detached same-input complete-U96 RAW.

Luminance weights are(.2126,.7152,.0722); block16 means use nonoverlapping
16x16 cells. This anchors coarse brightness without forcing all dark RGB
artifacts to match U96. Coefficients are prospective choices, not proven optimal.

## Finite calls, resource stops and numerical scope

Successful real CodeFormer accounting is712 forwards:120 fixed-U96 reference,
336 candidate snapshots and256 training. There are256 backwards,0 separate
autograd.grad queries,1 Adam constructor,64 optimizer attempts/commits maximum.
One separately counted controlled candidate-wrapper exception probe temporarily
substitutes a raising base stub:1 wrapper attempt,0 original-CodeFormer numerical
forwards. It restores exact callable, hooks, all575 states and RNG. This explicit
probe is additional to712; it is not disguised as no model interaction.

Before Torch imports, exact static pins, Linux manual-tmux stage, NVIDIA L4 and
idle GPU are required. Existing output directories are immutable. Python/NumPy/
CPU/CUDA seeds are20261021. All real forwards use strict deterministic mode and
FP32/no autocast. Only each reserved backward plus CUDA sync uses warn_only=True,
restored in finally. All warnings persist; unexpected nondeterministic operators
stop. Known bilinear CUDA backward nondeterminism is disclosed. Bitwise gradient
repeatability and cross-device resume equivalence are not claimed.

Exact packet.LIMITS:2700s internal,2760s external,180s per phase,8GiB initial
disk,4GiB continuing reserve,10GiB current/peak RSS,18GiB reserved/peak CUDA,
1536MiB output. Export has separate caps. A cap is a stop, not an ETA.

## Gates

Snapshots0/1/4/16/32/64 each produce32 fixed TRAIN-monitor plus24 legacy cases.
Snapshot0 must equal complete-U96 RAW arrays and PNG bytes. All4 first-update
gradient-enabled outputs must also equal their same-input U96 RAW/PNG before
the first commit. Upstream logits/LQ remain exact on all changed-head calls.

Unchanged historical guards, BOTH RAW/PNG:

-8 clear use untouched-CodeFormer teachers;8 dark use R2 teachers.
 Behavior distance percase<=max(.03,old0), mean<=max(.01,old0 mean).
-8 paired filter limit MAE/adjacent increase from original old0 to
 .03 worst/.01 mean. Initial thresholds are never reset to U96.

New monitor gates are additional: each role's paired MAE/adjacent increase from
U96 is<=.03 worst/.01 mean in both views. Local checks whole/central128/outside.
Clear U96 RGB distance and dark U96 block16-luminance distance have the same
.03/.01 limits. At64 only, local-center and global-whole mean adjacent residual
must each improve>=1% in BOTH RAW/PNG; dark mean paired MAE/adjacent must not
exceed U96. No1% gate applies at1/4. This is a new small TRAIN-monitor endpoint,
not the historical DGP feature gate or full DEV qualification.

Any initial inherited failure stops before Adam. Quality stops save artifacts
and exit2. Resource/integrity/numerical failures are distinct. There is no
automatic retry, new recipe, epoch extension, winning-snapshot selection or
promotion.

## Durable output and migration state

Only outputs/u96_postfusion_learning_v1 may be written. All reference/snapshot/
training calls save RAW float32 HWC256 and floor-quantized PNG. Reference/
snapshot and first-update training also save unclipped256. The4 first-slot
reference/training pairs save fullnative512/logits/LQ. Each record binds hashes;
every training objective/term and same-input U96 teacher binding is retained.

Every committed update, including0, saves dense36-head NPZ and resume.pt with
head state, Adam moments/step, explicit parameter order, CPU/CUDA/Python/NumPy
RNG, complete frozen sampler/order/cursor, learning/protocol/lineage and counters.
Every update saves actual postclip36-gradient NPZ, preclip norm/coefficient, LR,
before/after head hashes. Terminal64 has no next_update. These are migration
evidence; no automatic resume is implemented.

Atomic status.json replacement uses status.pending.json; interrupted pending
files remain. Counters reserve risky operations before calls. A confirmed
optimizer commit is counted immediately after step+sync; attempts exceeding
commits are explicitly outcome_unknown. The last fully saved checkpoint is
separate. Supervisors/exports must not substitute false zero updates.

Owned-output bytes use a durable-write ledger, fully checked against the actual
ordinary-file namespace at every checkpoint/snapshot/final. A64KiB reserve
supports failure evidence. SIGTERM raises a timeout; native blocking/SIGKILL
can leave only last durable status/events/pending state/checkpoint. A partially
unsaved optimizer operation is not asserted recoverable. No partial output is
deleted.

## Interpretation

Real R1 preflight proved same-device zero-head parity and connected gradients,
not useful learning. This single pilot jointly tests path/objective/data/anchors;
it does not isolate architecture causally. Frozen tokens cannot recover absent
information. Plausible texture is not exact identity. Surroundings preparation
remains disabled because the paired context study did not support always-on use.

A passed B64 screen is not qualification. Original full DEV preservation/
brightness and R2-dark retention gates, native unpaired review, independent
FINAL identities, all planned visual sheets, user acceptance and full app flow
remain pending. No unpaired PSNR/SSIM or Zamboanga claim. Completion/covering
development remains deferred until restoration acceptance.
