# Fusion-only V3: valid saved model, quality requirements missed

26 October 2026. Status: EXECUTED_AND_INDEPENDENTLY_AUDITED_QUALITY_REJECTION.
Trained R1 remains the working completion model. V3 is a retained research
checkpoint; it does not replace R1 in the app.

[All20 saved comparison sheets and audits](outputs/completion_codeformer_fusion_only_v3_return/review.html).

## What actually completed

The manual L4 run finished all64 fusion-only cycles in216.371seconds (about3.6min)
without timeout. It made64 new fusion Adam updates/256 backwards and480 whole
model forwards, plus168 empty bypasses. The code Adam stayed at32; fusion Adam
advanced32 to96. All443 frozen model tensors are bit-exact to R1; all54 intended
fusion tensors changed. Both named optimizers, RNG boundary, sampler and counters
passed independent recovery-state checks. All48 initial R1 comparisons match
historical raw arrays and PNGs exactly.

The701,936,549-byte terminal checkpoint is complete and independently valid.
Its SHA256 is3e2ecf54a4810b0ea85e9b52ee48935bb5d0636f9bc47346b40110c1e134ddf3.
Serialization including flush/fsync took3.757seconds, followed by2.081seconds
for hash and atomic commitment. This run repaired the previously demonstrated
saving bottleneck. That operational result is separate from image quality.
Worker exit2 is the prescribed quality failure, not a crash or missing export.

Return archive:1,426,187,988bytes,
SHA25675fe61d5f28eb345a0dc2034d5dc4a43bdc1519282d549e5023a6f0346c124be.
Protocol SHA2563dc71dac3002e82587cbc2cb758f36df604ed4c7131a3e592db4d8dcb5c18033.
All583 regular return members, including582 manifest payloads, verify.

## Generated-region quality versus working R1

These are relative reductions in paired synthetic delivered-PNG hole-L1 error;
they are not restoration structure gain, recovered identity, PSNR/SSIM on CCTV
or independent final accuracy. The32 clear cases and16 challenge cases are
adaptive TRAIN monitoring and remain separate.

| Fixed comparison | PNG hole-error reduction | Required | Outcome |
|---|---:|---:|---|
| Clear monitor versus R1 | +0.419741% | at least1% and all old subgroup gates | Failed |
| Challenge versus R1 | +0.479152% | at least1% and all subgroup gates | Failed |
| Clear cumulative versus original CodeFormer | +6.579113% | Historical context only | Does not replace R1 gates |

Clear single-lens hole error worsens in both raw and PNG. On challenge inputs,
lower-face, cheek and RGB-irregular hole error worsen in both views. All boundary
groups versus R1 are nonworse. The exact observed-pixel preservation requirement
passes:38,414,802 outside-mask RGB bytes and all eight saved empty controls remain
exact. Generated-region error and copied-pixel preservation are different checks.

Historical joint V2 u64 has separate saved comparisons. V3 reduces clear PNG hole
error0.794247% versus that failed route but worsens challenge error3.815555%.
Those separate historical executions do not establish a fully fresh compute-
matched ablation. V2 still has no usable saved checkpoint.

## Complete visual review

All224 actual output raw512-to-raw256/PNG replays pass exactly. All228NPZ files
pass schema/CRC checks separately. All20 primary sheets/80 fixed cases were
visually inspected by root, without
ranking or dropping bad cases. The separate renderer rechecked all576 sheet
cells. [Root visual record](outputs/completion_codeformer_fusion_only_v3_return/root_visual_review.json)
lists every page, case ID, sheet hash and observation. This is an agent development
review, not the required blinded human FINAL review.

V3 mostly resembles R1. Known closed mouths may still become smiles; colored
lips/blue eye patches remain in strict-gray inputs; invented frames persist in
the degraded sunglasses example; scarf/glove texture remains beard-like. Some
mask/glare/object examples look plausible, but their hidden regions are unpaired
and have no clean reference. The public file suffix native means an original
photograph, not native CCTV. No claim of exact hidden appearance follows.

## Demonstrated training-objective limitation and next review

[Saved-array diagnosis](outputs/completion_codeformer_v3_objective_alignment_v1_r1/analysis.json)
compares160 saved paired outputs, without new neural/gradient/training calls.
The trained objective is unclamped512RGB masked L1 plus inner-boundary L1,1+1.
Delivery clamps, resizes to256, copies visible pixels and quantizes to PNG.
The kernel7 boundary also covers different physical widths at512 and256.

Clear HQ512 objective improves about1.259% while delivered PNG hole improves only
0.420%; challenge HQ objective improves about2.180% while PNG hole improves0.479%.
Boundary improvements can outweigh worse holes: challenge lower-face/cheek holes
worsen at both512 and256 even while their1+1 objective improves. Clear lens HQ
hole improves0.1074% while delivered PNG hole worsens0.0695%. This is a geometry-
aggregate reversal caused by different case magnitudes; no individual case flips
from improved HQ hole to worse PNG hole. Recomputed HQ objective values are on
adaptive monitors/challenges, not before/after measurements of the FIT set.
Gradient clipping
was inactive on all64 updates; simply raising its threshold would not fix this run.
These observations support an objective-alignment review, not a claim that one
new loss is proven to solve anatomical uncertainty or the remaining data limits.

Next review: compare a differentiable app-delivered256 masked-hole objective with
the existing objective, keeping boundary requirements and every quality gate.
Do not append epochs, relaunch V3 unchanged, select an intermediate after a failed
terminal gate, or automatically resume its valid state. No successor training
packet is ready or automatically authorized to run. Any distinct pilot remains
finite, independently reviewed and manually launched on the user's L4 VM.

## Preserved app and remaining scope

All22 admitted R1 app/model/manifest hashes were rechecked unchanged. Both current
comparison apps, restoration and IMPLEMENTATION are unchanged. Original launch
guides are retained byte-exact in the return folder before HISTORICAL warnings
were added to the worktree and transfer copies. Sealed packet/source/protocol
remain immutable. All returned states, failures and splits remain retained.

The full goal is ACTIVE: useful all-family completion with reviewed masks,
separately evaluated automatic proposals, user acceptance, blinded independent
FINAL review, fresh Windows CPU installation and selective IMPLEMENTATION port
remain incomplete. A valid trained checkpoint alone does not complete the goal.
