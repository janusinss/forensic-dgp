# Fixed filter-pair preparation — 18 October 2026

**96 paired synthetic stress inputs are generated, independently verified and
reviewed. No new model has been trained or evaluated on them.** This preparation
addresses the blur, streak and block failures exposed by the five user examples,
while retaining the user's preference for useful R2 brightening.

## Data and research basis

The deterministic cohort contains four previously approved TRAIN photographs
and four existing DEV photographs, selected by sorted approved source IDs before
viewing new corruptions. All eight genuine RGB1024 files and retained RGB256
targets were freshly hash-verified. Clean512 is derived directly from RGB1024
with LANCZOS; target256 retains the existing exact file/pixels, independently
verified as direct1024 LANCZOS. The observed input begins with direct1024
BILINEAR256. No small target was upscaled to manufacture clean detail.

The existing 391 TRAIN/53 DEV membership remains intact. Only photographic source
hashes are proven disjoint, not people or CodeFormer pretraining identities.
These repeatedly exposed DEV photographs are not independent FINAL evidence.
The five unknown-origin user images and all native CCTV/FINAL sources remain
outside this generated cohort. Per-photo attribution and recorded license
conditions accompany the pairs. No new image acquisition or blanket portrait-use
clearance is implied.

The official [CodeFormer stage3 configuration](https://raw.githubusercontent.com/sczhou/CodeFormer/b33cc7d639d6545bfcccc7e0bc6ae51f24e79c2b/options/CodeFormer_stage3.yml)
and [joint dataset implementation](https://raw.githubusercontent.com/sczhou/CodeFormer/b33cc7d639d6545bfcccc7e0bc6ae51f24e79c2b/basicsr/data/ffhq_blind_joint_dataset.py)
use isotropic/anisotropic blur with resize, noise and JPEG variants. [BSRGAN's
degradation research](https://arxiv.org/abs/2103.14006) also motivates diverse blur
and interpolation choices. This supports testing broader corruption coverage;
it does not validate our exact parameters or establish CCTV performance.
Our fixed row repetition and local/global block or local-blur operations are
declared synthetic diagnostic proxies, not recovered versions of the unknown
filters used in wild1–wild5. Parameters use the256 input coordinate system and
are not copied as numerically equivalent512 training parameters.

## Fixed generation and independent checks

Twelve prospectively fixed profiles cover a clear control, broad Gaussian blur,
two anisotropic orientations, repeated rows, global mosaic and local blur/mosaic
variants. [approved_profiles_v1.json](outputs/codeformer_filter_pairs_v1_preparation/approved_profiles_v1.json)
records every parameter, operation, rounding convention and border policy.
The two anisotropic profiles are orientation variants, not severity levels.
No noise, exposure, colour, contrast or model-dependent parameter search is
added by this preparation. Brightening remains a model benefit to evaluate
separately from artifact removal.

The local rectangle `[64,64,192,192]` is an artificial corruption footprint,
not a detected face region, covering, removal mask or completion area. Outside
that rectangle, local operations preserve input pixels exactly. Hard rectangle
edges are intentional recorded stress artifacts. Global mosaic uses block means
followed by constant cell replication; repeating rows copy/blend rows. These
operations have known parameters and aligned references but remain simplified
proxies, not a calibrated native CCTV degradation distribution.

The generated namespace has **238 files / 16,128,800 bytes**. Generation completed
in approximately 8 seconds including process overhead; its internal timed section
was 5.08 seconds. It was bounded to 96 cases, 256 files and 64 MiB, with exclusive writes,
source/target readback and complete file hashes. The data manifest SHA256 is
`7823847ce7a709308fe1f4d35d35e2c14f752b2d5c303f5fa1cb4b7c8de8c3a8`.

Fourteen generator fixtures passed. A separately implemented CPU verifier using
NumPy, Pillow and standard-library math reconstructed all 96 saved inputs without
importing the producer or SciPy.
It independently rebuilt the Gaussian kernels, reflection boundaries, block
means, row repetition and local footprints: **zero final pixel mismatches**.
It verified all 16 clean256/512 target derivations, source/provenance joins, 96
input fingerprints, 96 exact sheet cells and the entire 238-file namespace. Ten
independent verifier fixtures passed. These checks establish data reproducibility,
not restoration success.

## Input-only review and admission boundary

Root personally viewed all 12 input-only sheets at 1024×612, with 96 exact 256
panels. Strong controls came first and clear inputs last, under
[criteria frozen before viewing](outputs/codeformer_filter_pairs_v1_preparation/input_only_review_criteria_v1.json).
No paired clean-target or model-output panel was used. Prior exposure to these
historical photographs and other degraded variants prevents a fully blinded or
independent FINAL claim. Per-sheet decisions were retained before subsequent
pages were opened.

| Root input-only judgement | Cases | Interpretation |
| --- | ---: | --- |
| Readable | 28 | Visible feature boundaries support cautious restoration comparison |
| Borderline | 44 | Coarse structure remains, but local boundaries or appearance are uncertain |
| Insufficient | 24 | Broad cells/blur erase central boundaries; request a clearer input |

Borderline examples stay in the challenging development evidence; they are not
silently removed to improve scores. Strong-profile tags do not automatically
mean insufficient: the repeated-row cases show readable structure in several
inputs despite their prospective stress-only tag. A DEV photograph with bright
novelty-looking lip/eye appearance is flagged for source-suitability review;
preserve its visible appearance and do not infer hidden anatomy or silently
remove an accessory. All original cohort membership and review history remain
preserved.

**No case is currently admitted to optimization.** All 48 DEV cases are explicitly
`EXCLUDED_DEV` regardless of input review. The 20 strong TRAIN controls remain
initially excluded; the 28 other TRAIN cases require a future explicit protocol
and admission decision. Root review alone grants neither training authorization
nor an automatic acceptance threshold. This is a small controlled diagnostic
cohort, not a sufficient complete training dataset or a new trained detector.

## Next model work

Compare untouched CodeFormer, the retained parent and R2 on identical relevant
inputs, preserving RAW, downloaded PNG and source/profile separation. Keep
readable, borderline and insufficient cases separately reported. Use the strong
controls to test abstention rather than claim that synthetic clean references
make their missing details recoverable in real images.

Use this evidence alongside the saved R2 normal/dark loss and feature-path review
to justify the next changed finite manual L4 pilot. Investigate useful illumination
correction separately from spatial appearance drift. Expand approved TRAIN
coverage where warranted; do not treat these four TRAIN photographs as the full
model-improvement dataset, loosen existing gates, add unchanged epochs or
promote R2 because data preparation passed.

No VM connection, neural forward, gradient, optimizer update, model checkpoint,
main-app change or new covering/completion development occurred in this phase.
No upload or training launch is required for these local preparation artifacts.
The full restoration-first goal remains active; useful outputs, preservation,
user acceptance, independent FINAL evaluation and accepted app flow are still
required. Native CCTV remains unpaired and source-separated, with no ethnicity,
Zamboanga or exact hidden-identity claim.

## Retained artifacts

- [Data manifest and all 96 pairs](outputs/codeformer_filter_pairs_v1_data/manifest.json).
- [Independent saved-pixel audit](outputs/codeformer_filter_pairs_v1_independent_audit/saved_pixel_audit_v1.json).
- [Root96-case input-only review](outputs/codeformer_filter_pairs_v1_preparation/root_input_only_review_v1.json).
- [Source cohort, terms and attribution](outputs/codeformer_filter_pairs_v1_preparation/cohort_selection_v1.json).
- [Five user examples and actual R2 failures](CCTV_CODEFORMER_USER_FILTER_STRESS_REVIEW_V1.md).
