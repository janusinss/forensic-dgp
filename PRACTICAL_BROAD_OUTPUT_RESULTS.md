# Broad covering outputs and palette preservation — 2 October 2026

The main app produces useful **assisted** estimates for the inspected hand, hair,
scarf and object examples. Automatic detection still misses most native coverings
in this extension. One scarf output retains a conspicuous textured beard/knit
appearance. The full automatic scope is not demonstrated and the Goal stays active.
No model was trained locally or on the VM during these comparisons.

Windows repository: `C:\xampp\htdocs\YEAR 4\Testing\`. Intended Linux counterpart
after transfer: `~/forensic-dgp/`. All `outputs` artifacts below are local,
Git-ignored evidence; `git pull` alone does not transfer them.

## Frozen source review

Eight RealOcc author-validation sources produce 16 native/degraded cases:
two hands, one hair obstruction, two scarves, two objects and one nearly hidden
face. Native sources and separately drawn removal proposals were inspected before
generation. The author's visible-face labels were not inverted into covering
targets. All 550 contact-sheet sources were inspected; this is development evidence,
not an unseen holdout. Training admission remains false. Publisher membership is
unchanged; exact-file checks find no match against the earlier 115-source review
manifest and ten-source practical cohort, but do not prove identity, full-corpus
or model-pretraining separation. [Author repository](https://github.com/kennyvoo/face-occlusion-generation),
[author paper](https://arxiv.org/html/2205.06218v1)

| Evidence | Windows local | Linux after transfer | SHA256 |
| --- | --- | --- | --- |
| Frozen sources | `C:\xampp\htdocs\YEAR 4\Testing\outputs\broad_covering_gallery_v1\frozen_protocol.json` | `~/forensic-dgp/outputs/broad_covering_gallery_v1/frozen_protocol.json` | `b7a1a447482d650bb0a18b13d649f0c548c3c59acd6db202111169599c805bc9` |
| Broad results | `C:\xampp\htdocs\YEAR 4\Testing\outputs\broad_face_workflow_outputs_v1\results.json` | `~/forensic-dgp/outputs/broad_face_workflow_outputs_v1/results.json` | `afe8cb6b2d768460082890fe8ded09dd125e19f599f54f55a496e2a4073a94d3` |
| Independent broad audit | `C:\xampp\htdocs\YEAR 4\Testing\outputs\broad_face_workflow_outputs_v1\independent_verification.json` | `~/forensic-dgp/outputs/broad_face_workflow_outputs_v1/independent_verification.json` | `f5b4534d227ccd2ec3c4d3bc3564597a5283bcda59d2df8aa7c2977d9602ef33` |

The frozen base route `face_workflow.py` has SHA256
`ed043f9819e4655c297f092037af156cbfa9e32506f6b15cca0c5d5dfe4cacd1`.
It uses the retained parent detector, CodeFormer inpainting and the separately
pinned CodeFormer face restorer at fidelity 1.0 with a 50% visible blend.

## Measured results

The comparison saved 28 outputs across 32 automatic/assisted rows. Four rows are
held for the two nearly hidden native/degraded inputs. CPU execution took 186.48
seconds excluding loading: 16 detector, 21 completion and 15 restoration forwards,
28 generation requests, zero optimizer updates. Model state fingerprints match
before/after. All 28 saved outputs, 32 detection masks, source hashes and previews
pass the independent pixel audit.

| Route | Known-visible degraded MAE, mean | Interpretation |
| --- | ---: | --- |
| Input, restoration Off | 0.0211754122 | Seven usable degraded cases, outside fixed reviewed proposals |
| Automatic proposal, base route | 0.0170107437 | Does not measure whether coverings were removed |
| Reviewed proposal, base route | 0.0171557517 | All seven improve visible error; mean reduction 18.98% |
| Reviewed proposal, palette V2 | 0.0170351531 | Mean reduction 19.55%; development cohort only |

Native assisted outputs change zero pixels outside the reviewed removal area when
restoration is Off. The visibility heuristic rejects both reviewed nearly hidden
cases before generator inference. **Automatic proposals miss both rejections**;
source review held those outputs separately. Do not credit that intervention as an
automatic rejection success. No hidden-face ground truth, hole MAE or identity
accuracy is available for these real photos. Saved broad outputs lack separate
pre-restoration intermediates: hole preservation is covered by the inference unit
checks and earlier restoration benchmark, not reconstructed independently here.

## Assistant inspection

All 16 source/output rows were inspected in four previews under Windows
`C:\xampp\htdocs\YEAR 4\Testing\outputs\broad_face_workflow_outputs_v1\preview\`
(Linux `~/forensic-dgp/outputs/broad_face_workflow_outputs_v1/preview/`).

| Source | Assisted finding | Automatic finding |
| --- | --- | --- |
| `val_18`, hands over eyes | Plausible eyes; approximate nose join and small remnant | Hands remain |
| `val_25`, hand over mouth | Useful lower-face estimate; false grayscale color corrected by V2 | Hand remains |
| `val_362`, hair over eye | Eye estimated; exterior hairstyle retained; V2 removes blue/brown false color | Obstructing hair remains |
| `val_244` / `val_336`, scarves | First useful; second partial with textured beard/knit appearance | Covering removal incomplete |
| `val_6` / `val_7`, flower/leaf | Plausible feature completion; degraded flower grows rough facial hair; hand outside selected face remains | Native objects remain; degraded leaf has color/seam residue |

The nearly hidden `val_26` is a rejection case, not a completion-quality row. These
are assistant judgments from a small exposed gallery, not user acceptance ratings.
The system removes selected facial covering regions, not every hand/scarf/hair
pixel throughout the scene. Anatomy and joins still require human review.

## Palette V2: cached comparison and actual browser verification

Input-only near-grayscale selection uses mean visible channel range at 256 scale,
Gaussian 5/sigma 1, eroded visible support of 9 pixels and at least 512 samples.
The threshold is 4/255. It projects generated pixels to grayscale with restoration
Off, or the full restored output when restoration is applied. Color inputs remain
byte-identical. Clear pixels remain exact when Off. This is a post-output policy,
not an additional model or restoration forward.

V1 threshold 3 missed the degraded near-gray hair input (measured chroma 3.156).
Its frozen evidence and policy snapshot are retained. V2 is an explicitly
source-derived developmental revision, not unseen evaluation. V2 selects eight of
28 outputs from the hand-mouth and hair sources; zero new model forwards occur.
All saved compositions, input signals, hashes and visible metrics are independently
verified without importing the runtime palette helper. The actual browser V2 result
for `val_25_hand_mouth_native` matches cached V2 pixels exactly.

| Evidence | Windows local | Linux after transfer | SHA256 |
| --- | --- | --- | --- |
| Palette V2 protocol | `C:\xampp\htdocs\YEAR 4\Testing\outputs\grayscale_palette_comparison_v2\frozen_protocol.json` | `~/forensic-dgp/outputs/grayscale_palette_comparison_v2/frozen_protocol.json` | `893909bc8d020ad4239cd48ffdc34bf5a5007e55a352f3f17a22b14f69a3b23e` |
| Palette V2 results | `C:\xampp\htdocs\YEAR 4\Testing\outputs\grayscale_palette_comparison_v2\results.json` | `~/forensic-dgp/outputs/grayscale_palette_comparison_v2/results.json` | `3d6e4b90fc3b6f39c230fa06dabc783412542f664d21a9a844d552be95214242` |
| Palette/browser audit | `C:\xampp\htdocs\YEAR 4\Testing\outputs\grayscale_palette_comparison_v2\independent_verification.json` | `~/forensic-dgp/outputs/grayscale_palette_comparison_v2/independent_verification.json` | `c026df0df3b66eb7b1e363b007037bde4a2947a3fda639e7dc57b35d9e48241e` |

Four assisted grayscale rows are inspected in `assisted_grayscale_preview.png`
under that V2 folder. Color artifacts improve; estimated anatomy is still approximate.
The main app uses `face_workflow_palette.py`, policy `reviewed-face-workflow-v2`,
extending the unchanged base route. Fifteen relevant inference/adapter/palette tests
pass. Inline bundled Playwright verifies generation and the active palette metadata
with no page/console errors. Existing 375/768/1280 workflow checks remain applicable;
the palette extension changes backend pixels, not page layout.

Next: benchmark a pinned pretrained visible-face segmenter against fixed removal
proposals and empty controls before selecting a detector remedy or another bounded
VM pilot. A visible-face complement contains non-face background/hair, so it must
not be treated as a covering label without a separately frozen region policy.
