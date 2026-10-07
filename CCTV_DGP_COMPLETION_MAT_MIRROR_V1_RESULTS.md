# Converted MAT completion comparison - 7 October 2026

The separately tested MAT FFHQ512 conversion does **not** justify replacing the
current completion component. Some eye and mouth estimates are plausible, but
hand joins, strong glare, obstructing hair and scarf boundaries remain unresolved.
All32 eligible development cases and eight full-resolution comparison pages were
reviewed. This result applies to the pinned converted weights and processing
recipe; it does not establish that every MAT release fails.

The app keeps its original own-trained DGP as the primary restorer. No app source,
checkpoint, restoration selector, reviewed mask or frontend design changed.
No training, gradients, optimizer update, VM action or reserved-final review
occurred in this comparison. The user separately ran V32 r2 on the VM; its
returned50-update failure is audited separately from this completion comparison.

## Source and terms

The earlier official MAT anonymous release links returned404. This comparison
therefore identifies its weights as a **third-party FP16 EMA conversion**,
published by spacepxl at revision`bef6e8b7d535abe2d42fdaeafec903abdfc638a3`.
The publisher describes stripped EMA face-model conversions and does not
guarantee compatibility with other implementations.
[Pinned publisher card](https://huggingface.co/spacepxl/MAT-inpainting-fp16/blob/bef6e8b7d535abe2d42fdaeafec903abdfc638a3/README.md).

The125,280,246-byte MAT_FFHQ_512_fp16.safetensors file matches published SHA256
`eedb8504aef8a07feda7e89ef34e53344eaf3039cb1543615bf1092439ce3d98`.
An independent reader verifies465 tensors and all62,612,683 finite values.
Only the specified tensor format is decoded; no network pickle or arbitrary
checkpoint code is loaded. Original-author FP32/checkpoint equivalence and
original CUDA inference parity are unverified. CPU calculations use FP32 values
expanded from the published half-precision values, preserving conversion loss.

The CPU implementation is pinned ChaiNNer/spandrel at
`e1f2ea14b2eb9dc912bdf335803f8a3d481c45b8`. Its source identifies an adaptation
from lama-cleaner. The original MAT CC-BY-NC4.0 research license and ChaiNNer's
MIT notice are retained. This is an attributed local noncommercial research
comparison, not permission to distribute weights or qualification for deployment.
[Pinned CPU source](https://github.com/chaiNNer-org/spandrel/blob/e1f2ea14b2eb9dc912bdf335803f8a3d481c45b8/libs/spandrel_extra_arches/spandrel_extra_arches/architectures/MAT/__arch/MAT.py),
[retained MAT license](https://github.com/chaiNNer-org/spandrel/blob/e1f2ea14b2eb9dc912bdf335803f8a3d481c45b8/libs/spandrel_extra_arches/spandrel_extra_arches/architectures/MAT/__arch/LICENSE).

The only vendor-source edit routes an unused metadata decorator to an identity
decorator. Generator/helper math is unchanged. Generator is instantiated directly
with512 resolution; the random-latent convenience wrapper is unused. Evaluation
still contains random dropout, so NumPy latent and per-request torch RNG are fixed
at240, truncation1, noise const. One estimate is returned per case; no seed search.

## Frozen scope and processing

The32 eligible cases reuse the same reviewed assisted masks and original/degraded
photographs as the audited app Off comparison. All seven covering families,
uncovered and clear-glasses controls are included. Four earlier input-only
exclusions remain excluded. Their legacy native suffix means original photo,
not native CCTV. They are already exposed development evidence, with unknown
pretraining overlap. No aligned clean hidden-face references exist; hidden
PSNR/SSIM, exact identity, ethnicity and Zamboanga performance are unclaimed.

The current CodeFormer Off outputs are reused on identical inputs/masks.
Restoration is Off to isolate completion. Masks receive **zero extra margin**.
RGB256 is resized to512 through visible-support-normalized bilinear interpolation;
binary masks use nearest interpolation. Covered colors are eliminated before
model input. Raw512 outputs are saved separately. Generated values are clamped
and bilinearly resized to256, then composited only inside reviewed removal support.
PNG quantization rounds255. Every source pixel outside that support is exact.

Finite budget:32 requests,28 nonempty forwards,four bypasses;90 seconds per case,
1200 seconds total with a1230-second external observer;200MiB maximum output.
The run completed in201.10 seconds, external process observation204.89 seconds.
Generator-state hashes before/after match. No timeout or incomplete inference
occurred. Eleven meaningful adapter/tensor-format checks pass.

Independent saved-output checks verify all32 PNG/stage pairs and every28 raw512
composition, preserving **5,225,232 visible source bytes**. All four uncovered/
clear-glasses controls bypass the generator exactly. A fixed hand/eyes CPU replay
matches raw values with maximum error0.0 and exact PNG. Technical completion is
separate from the full qualitative review.

## Full development review

| Family | Findings across the fixed cases |
| --- | --- |
| Masks | Lower-face estimates are present, but degraded cheek/nose shading and pink neck-edge remnants remain; no consistent advantage. |
| Sunglasses | One original-photo eye estimate has a smoother transition; closed-lid estimates are possible hidden states, while a degraded case generates a glasses-like rim. |
| Strong glare | Lens/eye areas remain bright and indistinct within requested removal support. |
| Hands | Lower-face anatomy can be plausible; hand-over-eyes has broken orange arcs/joins and retained boundary fingers. |
| Obstructing hair | Hair-like texture still obscures the requested eye area; visible non-obstructing hair remains exact. |
| Scarves | One smiling lower face is plausible, but irregular nose/scarf joins, coarse nostril texture and orange knitted remnants persist. |
| Other objects | Flower and leaf removal give plausible estimates; hand/finger pixels outside the reviewed leaf footprint remain exact. |
| Clear controls | Both versions of uncovered and ordinary clear-glasses sources are exact bypasses. |

Per-case observations distinguish defects **inside generated support** from
source pixels retained **outside** it. A completion generator cannot remove an
object outside its reviewed mask. Those misses need a new input-only mask review,
not post-hoc diagnostic rectangles, global dilation or a favorable random seed.
Anatomy within the requested region is a separate prior-quality issue.

This assisted-only comparison does not qualify automatic proposals or independent
final performance. The implementing assistant performed development visual review;
independent final review remains outstanding. The converted candidate is retained
as a negative comparison and is not promoted. Original weights, splits, prior
gate failures, cached outputs and the Windows backup receipt remain preserved.

Evidence: `outputs/completion_mat_mirror_comparison_v1/` contains protocol,
pre-generation readback, results, independent saved-output audit, visual review,
all raw stages and eight original-detail pages. Source/weight evidence is in
`outputs/completion_mat_mirror_review_v1_r1/` and
`outputs/completion_mat_mirror_assets_v1/`; initial acquisition/audit failures
are retained. Useful native DGP restoration, automatic/assisted covering quality
and independent final review remain incomplete.
