# Mask-conditioned Big-LaMa comparison — 2 October 2026

The pretrained **Places2 Big-LaMa** checkpoint does not improve useful facial
completion on the eight covered pilot sources. Both 256-pixel and 512-pixel arms
produce blurred, missing or distorted facial features. Strong white glare still
remains. It is not selected as the application's completion backend. This result
does not establish that all LaMa variants or face-specific models fail.

| Location | Windows local | VM counterpart after transfer |
| --- | --- | --- |
| Report | `C:\xampp\htdocs\YEAR 4\Testing\PRACTICAL_LAMA_RESULTS.md` | `~/forensic-dgp/PRACTICAL_LAMA_RESULTS.md` |
| Frozen protocol | `C:\xampp\htdocs\YEAR 4\Testing\outputs\practical_lama_protocol_v1.json` | `~/forensic-dgp/outputs/practical_lama_protocol_v1.json` |
| Outputs | `C:\xampp\htdocs\YEAR 4\Testing\outputs\practical_lama_outputs_v1\` | `~/forensic-dgp/outputs/practical_lama_outputs_v1/` |
| Prepared weights | `C:\xampp\htdocs\YEAR 4\Testing\outputs\lama_pretrained_v1\generator.pth` | `~/forensic-dgp/outputs/lama_pretrained_v1/generator.pth` |

All artifacts are local; no VM transfer/training, commit or push occurred. The
full workflow Goal remains active. Main-app behavior and previous checkpoints
remain unchanged.

## Acquisition and implementation

The [official LaMa repository](https://github.com/advimman/lama) was pinned at
`786f5936b27fb3dacd2b1ad799e4de968ea697e7`. Apache-2.0 source/license are retained
under `third_party/lama/`. Generator/FFT operations are unchanged; imports use
minimal helpers, unused discriminator code is excluded and unused spatial
transform configurations are explicitly rejected. The project PyTorch runtime
was not downgraded and no Lightning training environment was installed.

The author's recommended Hugging Face mirror was pinned at
`05cb2be7f8dbe6ca7c6e78f4fc827a4b2baaa4a9`. Its 381,428,720-byte archive downloaded
in 67.9 seconds and matched published LFS SHA256
`f1b358ca24093b93a106183b98a3dea6e8ed09f3b43ea7251eb2c81e7b4575f6`.
Original `best.ckpt` SHA256 is
`fccb7adffd53ec0974ee5503c3731c2c2f1e7e07856fd9228cdcc0b46fd5d423`.

Preparation uses a restricted `weights_only=True` loader and inert placeholders
for discarded configuration/callback metadata; original metadata classes are
not imported or executed. The 989 generator tensors/buffers are saved separately
with provenance. Prepared generator SHA256:
`54fbd7b0b7eaee1ad6c90ae3fe1f660f8fd50c9c8a88eba37d589be24f68f05a`.
Full source/config/weight hashes are in `outputs/lama_pretrained_v1/provenance.json`.

`lama_completion.py` supplies the official four-channel masked RGB plus removal
mask input, RGB in `[0,1]`, and exact compositing outside the mask. Four meaningful
adapter checks pass: explicit mask/white-pixel separation, isolation from covered
colors, empty-member bypass/padding preservation, invalid outputs and checksum
rejection. The 85% image-area rejection is a crop-area heuristic, not a verified
facial-visibility decision; the agreed near-total-face workflow still needs work.

## Fixed comparison and findings

The same ten previously inspected training sources and unchanged reviewed masks
from `practical_gallery_v2` were used. Ten audited CodeFormer reviewed outputs
were reused as the baseline; they were not regenerated. LaMa native256 performs
no resize. The 512 arm uses visible-support-normalized bilinear RGB, nearest mask,
and resizes generated pixels back before exact original-mask compositing.
Restoration and new alignment are off. No mask changes, detector routing or
hidden-face targets were introduced.

Twenty wrapper requests comprise 16 nonempty generator forwards and four empty
control bypasses. CPU elapsed **37.2 seconds**; zero inference failures, detector
forwards or optimizer updates. Recorded generator-state digests match. Independent
artifact/pixel checks verify all 20 output hashes, frozen gallery/code/weight
hashes and **zero changed pixels outside the reviewed masks**.

Both full-resolution five-row preview pages were inspected; per-case assistant
triage is saved in `visual_review.json`. The eight covered sources have blurred,
missing or distorted anatomy, including duplicate eye-like artifacts and pale
covering remnants. Both uncovered/ordinary-clear-glasses controls remain exact
empty-mask bypasses. Execution `complete=true` is not a visual quality pass.

| Evidence | SHA256 |
| --- | --- |
| Frozen protocol | `64ff7321ca0580ba9af8df0ab247a22bf3c9171cdfed5f6f01021b2023829d5b` |
| `results.json` | `2dde4543215701641323aa0661bd3314a00e87fc2027df2450360770dd055b94` |
| `independent_verification.json` | `0d9ad01e7a4b0fd6623494c438fbfa8a6ba7f8d9472fced8ed5a514017514de9` |
| `preview.png` | `4ae9558515fbfd97cdf0f3a6995f331877bac3cd8a65a78fc259841204fc3c3e` |

## Next decision

Benchmark a **face-specific pretrained, mask-conditioned** completion checkpoint
before choosing generator training. The AOT-GAN author's repository supplies a
CelebA-HQ generator and explicit mask input; this is the next candidate to verify
and compare, not a selected winner. [Official AOT-GAN source](https://github.com/researchmm/AOT-GAN-for-Inpainting)

Keep reviewed proposals/results immutable. Full-eyewear rims need a new proposal
version; degraded-input restoration, missing covering families and integration
in the existing main application remain pending. All actual training is VM-only.
