# Separate completion prior: official MAT source review

7 October 2026. The saved-pixel diagnosis showed both copied covering fragments
and generated scarf/chin or eye defects. A different completion prior can be
compared within the reviewed removal support. It cannot remove a fragment that
the application's exact visible-copy rule keeps outside that support.

Five official MAT source/license files, totaling 48,328 bytes, were acquired at
commit `d273d891ecdad2e1df106516423a75bc45b2d800`. A separate static checker verifies
all five hashes and parses both Python files without importing or executing them.
The restricted-network failure remains in its original folder; the successful
host-network acquisition uses a distinct folder. No checkpoint or model is loaded,
no case is generated, and no application or removal area changes.

The authors provide CelebA-HQ and FFHQ face-completion releases, including a
CelebA-HQ 256 model. Released weights were retrained and need their own fingerprint;
the paper's result is not evidence for downloaded weights or these CCTV cases.
[Official pinned README](https://github.com/fenglinglwb/MAT/blob/d273d891ecdad2e1df106516423a75bc45b2d800/README.md)

The repository carries CC BY-NC 4.0 and describes research-only use. Attribution,
notices and modification records must be retained. Inherited dependency terms and
the actual checkpoint's provenance still require audit before adoption.
[Pinned license](https://github.com/fenglinglwb/MAT/blob/d273d891ecdad2e1df106516423a75bc45b2d800/LICENSE)

The reference entrypoint selects CUDA, uses seed 240 and forces random noise for
non-512 resolution. Its masks use zero for the removal region and one for retained
input, opposite the app's removal mask. It loads a whole-network pickle through
the legacy loader. These are concrete adapter prerequisites; local compatibility,
loading/conversion, dependencies and inference parity have not been verified.
[Generation source](https://github.com/fenglinglwb/MAT/blob/d273d891ecdad2e1df106516423a75bc45b2d800/generate_image.py),
[loader source](https://github.com/fenglinglwb/MAT/blob/d273d891ecdad2e1df106516423a75bc45b2d800/legacy.py).

MAT is a candidate for the separate completion comparison. This source review
does not select a checkpoint or establish usefulness. LaMa is an additional
general-image reference whose code is Apache 2.0; it supplies no face-specific
advantage for these cases without evaluation.
[LaMa author repository](https://github.com/advimman/lama),
[LaMa license](https://github.com/advimman/lama/blob/main/LICENSE).

A subsequent comparison must freeze the exact case list, input-only eligibility,
reviewed removal masks, one deterministic estimate per case, raw/display stages,
finite runtime limits and independent saved-output checks before generation.
It must review masks, sunglasses, strong glare, hands, obstructing hair, scarves
and objects, plus uncovered/clear-glasses controls. Automatic and assisted
results remain separate. Any boundary correction must be selected from the input
and reviewed before generation, with the small visible-appearance margin recorded.
Post-hoc support rectangles are not ground truth or new mask annotations.

Outside the final removal area, prepared source pixels must remain exact. Record
all geometry, resampling and quantization; a 512 inference path cannot be silently
reported as a native 256 model. Do not select a favorable seed or use a same-person
reference gallery. Missing regions have no aligned clean reference, so their
PSNR/SSIM and exact hidden identity remain unclaimed. Pretraining overlap with
exposed photographic cases is unknown; these cannot qualify independent final
performance. No ethnicity or Zamboanga-performance inference follows.

Our trained DGP remains the primary visible-face restorer. The source-only
completion review does not replace it or justify a new DGP training recipe.
Actual new training stays in the manual, finite existing-L4 workflow. Useful
native restoration, automatic/assisted covering usefulness and independent final
review remain incomplete.

Evidence: `outputs/completion_mat_source_review_v1_r1/acquisition.json`,
`independent_source_audit.json`, the five pinned files, and the retained failed
network acquisition in `outputs/completion_mat_source_review_v1/`.
