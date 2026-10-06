# V18: preserve DGP pixels while testing a learned spatial correction

**Closed5 October2026:**600 updates completed on the existing L4 and the returned
outputs are independently audited/reviewed. Degraded training PSNR15.5527→23.5201
and SSIM0.62517→0.70649 improve; four clear preservation guards still fail.
All trained snapshots remain unqualified under this unchanged protocol. No
selection/promotion or held-out/native/reserved evidence. Original numeric
audit failures are retained beside separate arithmetic-only corrections;
original source/protocol/checkpoints and scientific gates are unchanged.
Do not repeat V18. Next is a separate input-only processing diagnosis.
[Returned results and audit limits](<C:/xampp/htdocs/YEAR 4/Testing/CCTV_DGP_STRUCTURE_V18_RESULTS.md>).
The preparation rationale and frozen design below remain historical evidence.

Prepared on 5 October 2026 before execution. The separate V18 interface and immutable VM
capacity package are verified locally. **No V18 CUDA gradient check, optimizer
update, trained-quality result, checkpoint selection or app adoption is claimed.**
Manual execution and collection:
[CCTV_DGP_STRUCTURE_V18_VM.md](<C:/xampp/htdocs/YEAR 4/Testing/CCTV_DGP_STRUCTURE_V18_VM.md>).
The user selected pasteable transfer/VM commands only; all actual training stays
on the existing NVIDIA L4 in `~/forensic-dgp`.

## Why this change is justified

V16 r2's audited 3,128-update code-prediction run improved synthetic degraded
pixel PSNR, but both trained snapshots failed the unchanged structure/appearance
guards. Changed eyes, mouths, glasses, facial texture and fragments were visible
even on training previews. V17's audited inference-only feature-connection
control improved clear cases but worsened degraded cases. Preserve those failed
recipes and results; increasing epochs or universally choosing fidelity w1 is
not supported by that evidence.

Reports:
[R2 results](<C:/xampp/htdocs/YEAR 4/Testing/CCTV_DGP_BROADER_CODES_V16_R2_RESULTS.md>)
and [V17 results](<C:/xampp/htdocs/YEAR 4/Testing/CCTV_DGP_FIDELITY_SPOTCHECK_V17_RESULTS.md>).
Official CodeFormer documentation separates code-sequence prediction and
controllable-module training into different stages. That distinction supports
treating successful token fitting and useful appearance preservation as separate
questions. It does not establish that our new decoder works.
[CodeFormer training procedures](https://github.com/sczhou/CodeFormer/blob/master/docs/train.md).

V18 changes the output interface and supervised objective. The audited retained
DGP V2 image remains the pixel base. A new spatial decoder observes the camera
crop, that base and frozen face-prior features, and learns an RGB residual using
pixel, coarse-structure and fixed-affine appearance losses. This is our custom
capacity experiment, not an official CodeFormer training-stage reproduction or
a claim of equivalence to the published DGP architecture. The historical
MobileNet/FPN implementation audit remains in
[CCTV_DGP_ARCHITECTURE_REVIEW.md](<C:/xampp/htdocs/YEAR 4/Testing/CCTV_DGP_ARCHITECTURE_REVIEW.md>).

## Declared learned and pretrained components

| Component | V18 role |
| --- | --- |
| Our retained trained DGP V2 | Frozen 256×256 pixel base and normalization policy |
| Our audited R2 epoch8 code head | Frozen; predicts code indices from the input/DGP path |
| Pretrained CodeFormer | Frozen codebook/generator through block 12; supplies 256-channel 64×64 features |
| Our new 684,395-parameter spatial decoder | The only component fitted by V18 |
| Pretrained ArcFace recognizer | Frozen fixed-affine appearance loss/measurement; no exact-identity claim |

The current user's 4 October clarification permits a declared pretrained
generating prior plus our trained conditioning if reviewed output improves.
Keep CodeFormer pretrained provenance explicit. Its loader metadata records
default AdaIN, while **this effective V18 feature path uses no AdaIN, statistics,
input-feature fusion inside CodeFormer, or prior RGB tail**. These are feature
extraction settings, not a claimed pretrained restoration output.

Code-head checkpoint SHA256:
`3ef704e70f68d633ac7624eb47f79cfada189341dfdc58bad08444595d7d6757`.
New decoder module SHA256:
`e98af67f110b4ebc276ff2394345a81bc6ca7e082b36a493ad5c3fde94593e5d`.
The R2 protocol, original returned checkpoint and all historical splits/failures
remain unchanged. There is no teacher-generated target or new code-head fitting.

The decoder encodes the six camera/DGP RGB channels through spatial skips,
projects the frozen prior feature map and fuses it at 16×16 and 64×64. Its output
is `clamp(DGP + 0.5*tanh(learned_RGB_residual), 0, 1)` at 256×256. A zero final
convolution starts with exact DGP parity. That initial parity prevents a loading
regression; it is not learned improvement or proof of useful prior contribution.
The residual still has enough range to change visible appearance, so strict
preservation metrics and direct visual review remain necessary.

## Verified local interface evidence

Two fresh training-only inputs were run through the unchanged DGP, learned R2
head, frozen prior-feature path and new decoder. CPU inference took **13.60
seconds**; both normal and zero-prior paths retained exact initial DGP raw pixels.
All frozen before/after state hashes match. Execution counts are two DGP, two
prior encoder/classifier, two R2 head, two prior generator and four new decoder
forwards; no unused V11 conditioner, backward or optimizer calls.

An independent saved-output audit checked eight arrays/PNG artifacts and their
composition in **0.1064 seconds**, with no neural replay. Evidence is in
[cctv_dgp_structure_prototype_v18](<C:/xampp/htdocs/YEAR 4/Testing/outputs/cctv_dgp_structure_prototype_v18/results.json>).
Results SHA256:
`b589998457496f5cbecd1e8933b3a9e5a1bc9108c14e9c6381da5d721ab98a7c`.
This proves the fresh inference interface and initial preservation only. The
real CUDA gradient path is deliberately pending VM execution.

Thirteen meaningful regressions passed: five decoder schema/parity/local guard
and inference sensitivity checks, and eight finite schedule/gate/timing/Path/
scope/local-trainer-entry checks. Their inference probes assign test weights;
they do not fit weights. The actual Windows trainer entry rejects before model,
output, differentiable graph or optimizer construction. VM source parses under
Python 3.10. Local torch is `2.13.0+cpu`; the launcher requires the preserved
existing VM runtime receipt, with no reinstall or dependency upgrade.

## Frozen capacity scope

Use the existing ten R2 training-preview photographs, five per provenance source:
`dataset/asian_faces` and `dataset/thumbnails128x128`. Every photograph has clear,
blur_lr24, lowlight_lr32, motion_lr48 and compound_lr24 profiles. Thus there are
50 training cases and 70 bound input/target/support files. Some source photos
are only 128×128; their resized targets are photographic proxies, not newly
acquired high-resolution ground truth.

These source folders are provenance groups, not individual ethnicity labels.
Prior/recognizer pretraining overlap remains unverified. This is paired
synthetic capacity evidence only, not generalization, real CCTV ground truth,
Zamboanga validation or a face-identification accuracy test. No validation,
native-development 24 or reserved 32 inputs are forwarded or used to select
hyperparameters by this pilot. Existing native acquisition/splits stay frozen.
Future CCTV without an aligned clean image remains unpaired evidence; paired
PSNR/SSIM stay separate from that review.

| Quantity | Frozen value |
| --- | --- |
| Seed; fitted capacity | 20261005; 684,395 parameters |
| Updates; batch; exposures | 600; 10; 6,000 maximum |
| Exposure balance | One case per source/profile each update; each case 120 times |
| Cohort epochs; fixed snapshots | 120; updates 0, 50, 200, 600 |
| Optimizer | Fresh AdamW, lr 0.0003, weight decay 0.01, betas 0.9/0.999, clip norm 1 |

No AMP, EMA, resume, best.pth or automatic checkpoint selection. Four fixed
snapshots remain visible regardless of later scientific preservation failures.
The final zero-prior run removes the prior input from the same fitted decoder:
it measures input sensitivity, not a separately trained no-prior ablation or
proof of a causal prior advantage over such a model.

## Objective and stops

Only observed support contributes to RGB MSE and masked coarse MSE at
32×32, 64×64 and 128×128. Coarse terms receive weight 0.5. Fixed source/profile
weights use the initial retained-DGP raw error: clipped inverse-error ratios
0.25–4, normalized to mean one. This prevents the initially noisy group from
dominating the small balanced capacity test; it is not fitted on validation.

The identity-loss coefficient is the ratio of reconstruction/identity gradient
norms on the first balanced training batch, clipped to 0.0001–10 and frozen
thereafter. Both gradient traversals and a joint backward preflight occur only
on the L4. Preflight requires finite nonzero new-decoder gradients and absent
gradients in all frozen components, before constructing the optimizer.

At update 50, the entire 50-case cohort must improve the same fitted stopping
measure by at least **1%**, or training stops and exports its failure. This
measure combines weighted raw reconstruction with exported-PNG identity at both
updates 0 and 50; optimization uses raw RGB identity. Their different numerical
representations are explicit. Gradient finiteness, frozen state, finite timings,
VRAM and update counts also gate execution; there is no automatic retry.

| Limit | Stop policy |
| --- | --- |
| Cache: 300 seconds | Project after 10 cases using nine steady timings; count startup once, safety 1.25, reserve 30 seconds |
| Fit/snapshots/grids: 900 seconds | Project after 20 updates using 19 steady timings; include snapshot allowance, safety 1.25 and reserve 60 seconds |
| Trainer: 1,230 seconds | Supervisor bounds the child, including verification overhead |
| Audit: 300 seconds; supervisor: 1,800 seconds | Separate process deadlines; supervisor budget includes export |
| Export: 180 seconds; resource limits | Export stays inside overall deadline; 20 GiB allocated VRAM cap and 4 GiB free disk before launch |

Preserve failed gates, partial files and logs. Process timeout handling terminates
the child's process group with a bounded grace period; it does not resume the
run. The 30-minute supervisor budget is a stop rule, not a promised duration.

## Preservation, display and returned evidence

Each clear/source/profile/degraded group must retain MSE within 1e-12 of the
baseline and SSIM/fixed appearance cosine within 1e-6. Require at least 0.1 dB
degraded PSNR gain and 10% degraded MSE reduction before qualifying this output
path for a separately frozen generalization experiment. A group regression
remains a failed scientific guard even if training and export finish.
No aggregate pixel gain overrules visible-feature drift or a failed group.

Raw float32 RGB arrays are saved separately from delivered PNGs. PNG composition
restores unobserved padding from the input and rounds/clips to byte range.
No sharpening, contrast changes or preferred display transform hides a result.
Ten fixed original-256-pixel comparison sheets cover all 50 cases: five show
camera/DGP/update50/update200/update600/proxy and five show zero-prior sensitivity.
All **550 original cells** must match the saved images and be reviewed.

Successful execution is expected to contain 600 optimizer steps, 601 backwards,
two gradient-norm traversals and 603 total autograd traversals. Derived expected
neural counts are 58 each DGP/prior encoder/prior classifier/R2 head/prior feature
generator, zero prior RGB-tail and unused V11, 860 new-decoder and 912 recognizer
forwards. These are planned receipts, not verified completed CUDA calls.

The returned-file auditor independently rebuilds raw/PNG composition, 250
metrics/cosines, initial 50-case parity, 50 cache bindings, trace/exposure scope,
coarse-loss arithmetic, calibration/stop arithmetic, checkpoint state and 550
grid cells. Eight saved fresh-image renders at updates 0/600 must match cache
renders within the frozen VM tolerance 2e-6. A separate local CPU cached-decoder
replay checks all 250 outputs within the predeclared 5e-5 tolerance; this is not
a relaxation of any historical gate.

The audit does not independently rerun CUDA optimizer steps, gradient norms or
the recognizer/DGP/prior network. Their serialized receipts, provenance and
frozen hashes are checked; those limits remain explicit. Partial failed runs
receive only trace/source/transfer verification. Training completion, export
`complete:true`, CPU replay or capacity fitting alone does not authorize adoption.

## Frozen preparation and next milestone

The 9,063,630-byte execution archive contains 23 exact regular unique members
and 21 source/assets. Preparation rechecked the existing parent/data/baseline
dependencies in 24.33 seconds without neural calls. A separate transfer auditor
checked archive/member/source hashes, all 70 selected data files, unchanged R2
lineage, 600 balanced batches, 120 exposures per case, parameter arithmetic and
derived forward counts in **0.3027 seconds** with zero neural/training calls.
Receipt:
[cctv_dgp_structure_v18_transfer_audit.json](<C:/xampp/htdocs/YEAR 4/Testing/outputs/cctv_dgp_structure_v18_transfer_audit.json>).

Protocol SHA256:
`e3a7567e9a9c53a1668464ab5c95edbf094c73e9a7c8e550635dcf8d8ef04adb`.
Archive SHA256:
`ecc8a24aa77e23f80bbd7e482573ed666e443e35c7ee5633ddf4e5a6edeb2ce7`.
Pilot preparation used no cloud operations. The user subsequently completed
upload steps1/2/3 and explicitly authorized direct gcloud storage maintenance.
That cleanup is complete:9.0GiB recovered;87% used/13GiB available. Only verified
archive duplicates and disposable pip cache files were removed. Scientific
caches and protected assets remain, and the three V18 uploads match their
expected hashes. Receipts: `outputs/cctv_dgp_vm_storage_cleanup_20261005/`.
No trainer launch or VM shutdown. Actual CUDA preflight/training remains pending
the user's manual commands.

After collection, independently audit and inspect every original-cell sheet
before choosing any next model experiment. Useful held-out/native outputs,
covering-family completion, automatic restoration choice with override, DGP-led
app integration, independent final review and full bundled inline Playwright
verification remain required. The goal stays active and incomplete.
