# Practical native output comparison — 2 October 2026

Reviewed removal masks improve several mouth-mask and hand-over-mask outputs.
They do not resolve the darkest sunglasses or white-glare case. Those failures
remain after manual correction, so another detector experiment alone cannot
establish the requested output quality. No checkpoint was promoted.

| Location | Windows local | VM counterpart after transfer |
| --- | --- | --- |
| Repository | `C:\xampp\htdocs\YEAR 4\Testing\` | `~/forensic-dgp/` |
| This report | `C:\xampp\htdocs\YEAR 4\Testing\PRACTICAL_NATIVE_OUTPUT_RESULTS.md` | `~/forensic-dgp/PRACTICAL_NATIVE_OUTPUT_RESULTS.md` |
| Frozen gallery | `C:\xampp\htdocs\YEAR 4\Testing\outputs\practical_gallery_v2\` | `~/forensic-dgp/outputs/practical_gallery_v2/` |
| Outputs | `C:\xampp\htdocs\YEAR 4\Testing\outputs\practical_native_outputs_v1\` | `~/forensic-dgp/outputs/practical_native_outputs_v1/` |

These are local artifacts; no new VM transfer, training, commit or push occurred.

## Frozen experiment and verification

Ten previously inspected detector-training sources were frozen before generation:
three mouth-mask cases, two sunglasses cases, one white-glare case, one mirrored
eyewear case, one hand-over-mask case and two clear controls. This is developmental
evidence, not a pristine holdout or population accuracy estimate. Dedicated
standalone-hand, obstructing-hair, scarf-over-face and other-object cases are still
missing. Ten degraded copies were prepared but were not run in this comparison.

The three arms use the same CodeFormer inpainting weights and native inputs:
the previously audited parent automatic masks, the experimental reflective-42
automatic masks, and source-reviewed removal proposals. Reviewed proposals add
a documented 3-pixel margin at 256 pixels (1 pixel for white glare) and eyewear
bridge polygons. Original labels/splits remain unchanged. Restoration is off;
existing supplied crops are used without a new alignment stage.

The run completed 30 wrapper requests, comprising 22 nonempty generator forwards
and eight empty-mask bypasses, in **90.6 seconds on CPU**. There were zero inference
failures, detector forwards or optimizer updates. Project PyTorch remains
`2.13.0+cpu`; local CUDA is not configured. Recorded model-state digests match
before and after inference.

An independent PIL/NumPy audit rechecked all source/checkpoint/code hashes,
72 frozen gallery PNGs, cached-mask pixel identity, all 30 output hashes and
mask-placement/visible-change calculations. **Zero pixels changed outside each
active removal mask.** This proves preservation/provenance, not hidden facial
correctness. Real covered photos provide no hidden-face ground truth or hole MAE.

| Evidence | SHA256 |
| --- | --- |
| `outputs/practical_gallery_v2/frozen_native_protocol_v1.json` | `74b80e0ec9ebabb7c8b3b3989f5bc4f44b9a4cf1e39e691c073543f2db4651a5` |
| `outputs/practical_native_outputs_v1/results.json` | `38776d1a46967c4338a2082fe94ad50c2ca335d767b57c8c4759f681374f5325` |
| `outputs/practical_native_outputs_v1/independent_verification.json` | `d69bc0ea2a77bdecfa7b710d9be1c64d965ebe1236808c08207d82702006d9db` |
| `outputs/practical_native_outputs_v1/preview.png` | `f6c3af6f5938e89b2e10f4e8e55ca150e1dcbb141bfa23b5b6383ee48a0e521d` |

## Visual review

All ten rows were inspected in `preview/rows_01_05.png` and
`preview/rows_06_10.png`. The assistant's sidecar `visual_review.json` is a
developmental triage, not independent expert scoring. Four reviewed covered cases
are useful estimates, one is partial and three need fixes. Both reviewed controls
are exactly preserved. These counts cannot be interpreted as deployment accuracy.

| Case | Reviewed-output finding |
| --- | --- |
| Cloth mask | Useful estimated lower face; mask removed with coherent join |
| Pink mask | Useful estimate; minor texture at lower boundary |
| Mask with clear glasses | Needs fix: pale patch near nose and inconsistent lower-face join |
| Dark sunglasses | Needs fix: dark glasses recreated, pale halo, eyes remain hidden |
| Sunglasses | Useful estimated eyes; minor seam/temple remnants |
| White glare | Needs fix: strong white reflection remains |
| Mirrored eyewear | Partial: estimated eyes, but circular wire/frame remains |
| Hand over mask | Useful estimated lower face; covering largely removed |
| Uncovered control | Exact preservation with reviewed and candidate empty masks |
| Ordinary clear-glasses control | Exact preservation with reviewed empty mask |

The parent retains coverings in all eight covered cases, including three empty
predicted-mask bypasses, and unnecessarily changes the uncovered control. The
candidate covers more of the labeled area but leaves substantial remnants.
Its historical synthetic-retention failure remains; this output comparison does
not waive the previous gate or select it as `best.pth`.

## Next processing comparison

Compare a pretrained mask-conditioned completion model on these same reviewed
inputs/masks before spending VM time on generator training. The official LaMa
implementation concatenates the removal mask with masked RGB. Its README supplies
a checkpoint mirror, and its source is Apache-2.0. It is a candidate to benchmark,
not evidence that it improves facial anatomy. [Official source](https://github.com/advimman/lama),
[paper](https://arxiv.org/abs/2109.07161)

CodeFormer's current network receives white-filled RGB without a separate mask
channel. Its white-glare result is consistent with an ambiguity in that input,
but this is an inference, not a demonstrated causal explanation. Keep the old
adapter and outputs immutable. Freeze new model/resize/refinement arms before
evaluating them. Compare removal, anatomy, seams and visible appearance; exclude
hidden-identity claims. Any full-eyewear-rim corrections belong to a new proposal
version, because the current masks/results are frozen.

Restoration on the ten degraded copies, the missing covering families and main-UI
integration remain pending. COFW source metadata/documentation were verified,
but its image archive transfer is incomplete and excluded; see
`PRACTICAL_DATA_SOURCES.md`. All actual training remains VM-only.
