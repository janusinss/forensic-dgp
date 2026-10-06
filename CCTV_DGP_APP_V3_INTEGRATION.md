# DGP-led research application — 5 October 2026

The existing interface now uses our retained trained identity-v2 DGP as its main
visible-face restorer at 256×256, with Auto/On/Off, input/removal-area review, one
estimate and PNG/ZIP downloads. **Functional research integration is verified;
useful native CCTV restoration and full covering-family acceptance are not.**
V18/V19/r2 remain rejected and closed. No new training or cloud action occurred.
Manual VM transfer/execution preference remains in force.

Local app: `app.py`, `http://127.0.0.1:8000/`.
Workspace: `C:\xampp\htdocs\YEAR 4\Testing`.

## Selected research baseline and processing

The new `dgp_face_workflow_v3.py` is wired through `face_workflow_web.py`. Historical
engines, adapters, `/reconstruct`, checkpoints, splits, pilot sources and failed
gates remain intact. This route has no pretrained restoration fallback or failed
V18/V19 spatial selector. A separate pretrained completion component is declared.

| Item | Selection |
| --- | --- |
| DGP weight | `outputs/cctv_dgp_face_code_fit_vm_v12_r2/weights/dgp_v2.pth` |
| SHA256 | `646fbb11674e8882ea41e251d6e6e7d468c963979b48c93132def8292834e24b` |
| Lineage | Audited identity-v2 tensors, reserialized as the retained mixed-pilot baseline |
| Qualification | Paired-development retention; native usefulness remains unestablished |
| Normalization | Five frozen InstanceNorm layers with disposable kernel-statistic copies |

The strict loader pins its historical dependencies and refuses missing, changed
or malformed weights. NumPy float32 division by 255 precedes device transfer,
matching the audited canonical V15 convention. Inference has no optimizer or
backward path. The original Phase 3 checkpoint remains a historical comparator.

Native aspect ratio is preserved by neutral-128 center padding and frozen PIL
bilinear resizing to 256; support/removal masks use nearest resizing. Auto's
developmental blur/noise signals exclude padding, covering and filter borders.
They suggest restoration, not facial usability or pose. On/Off override remains
available. Too little signal support asks for an override/review or less-covered
image rather than silently choosing a model.

DGP processes the prepared observed input. Pinned CodeFormer inpainting estimates
only the reviewed covering from original context. Completed pixels are identical
across restoration overrides. Off keeps prepared visible RGB bytes exact; On uses
DGP without the old 50% pretrained-restoration blend. Padding stays unchanged.
The PNG uses floor(float32×255); input-derived grayscale processing is declared
separately from raw DGP floats. No contrast/detail enhancement is added.

## Input review, mask provenance and downloads

The new route removes the minimum-32-pixel upload rejection. Dimensions alone
neither reject a potentially useful tiny crop nor approve a larger unreadable
one. Operator input-only review must confirm one frontal/mild-turn face with
readable visible features. Insufficient structure requests a clearer crop;
out-of-scope input requests one suitable face crop. Input review and removal-area
confirmation are required server-side, even with an empty mask. This is an
**operator-assisted policy, not a calibrated automatic structure classifier**.
Exactly flat native visible regions and near-total reviewed facial covering are
also rejected before generation; padding is not counted as face evidence.

Automatic removal remains an unqualified development baseline. Proposal margins
are bounded to three pixels at 256 scale; tiny crops receive zero native margin
instead of oversized minimum-one-pixel dilation. No additional expansion follows
review. Automatic-reviewed provenance requires the final native mask hash to
equal this server's input-bound proposal. Changed/imported masks, mismatches or
expired cache entries are assisted-reviewed. This describes the final area, not
every interaction in an editing history.

The UI shows prepared input, aligned mask and one 256 output. The ZIP preserves
native original/native mask, 256 input/mask/result, metadata, and raw DGP float32
array when restoration ran. Hidden anatomy is one plausible estimate, not exact
identity. Raw output, composition and grayscale display processing are separate.

## Verification and development review

**34 tests pass:** ten new route tests plus existing workflow, palette,
observed-quality, strict DGP and frozen-normalization regressions. They cover
required reviews, tiny crop geometry, empty bypass, visible/padding preservation,
mask provenance, overrides, raw bundle export and invalid/missing model rejection.
Unit model stubs are distinguished from actual inference below. Test startup
assumed an unavailable `httpx2` TestClient dependency; the actual async endpoint
is tested without adding a package. The blank-input regression exposed padding
edge variation; the guard now checks native visible pixels before resizing.

The finite `scripts/audit_dgp_app_v3.py` completes in **4.56 seconds** within its
300-second cap: **eight real CPU DGP forwards**, unchanged weights, zero
completion/training/backward calls. All six frozen coarse frontal/mild native
development cases exactly reproduce cached retained raw arrays (maximum delta
zero). Repeat output is exact; Off equals prepared input. Eighteen other native
cases replay conservative operator clearer/out-of-scope decisions before a
forward. This is routing evidence, not automatic classifier accuracy.

All six native rows/four columns were viewed at original sheet resolution. Coarse
facial layout remains, but eye/nose/mouth boundaries are soft or indistinct; no
useful native structural improvement is demonstrated. These QMUL-SurvFace inputs
are unpaired: no native PSNR/SSIM, identity accuracy or Zamboanga performance is
claimed. The 32 reserved crops remain unviewed/unprocessed.

Bundled inline Playwright against real local models passes 375/768/1280 layout
checks with zero overflow/page/console errors. Upload, proposal-before-generation,
insufficient-input request, mask import, keyboard paint/undo, renewed review,
Auto/On/Off, one output, PNG/ZIP and server rejection of missing reviews work.
A separate saved-download audit verifies native original, result/raw hashes,
Off visible bytes and identical completed pixels across overrides. The initial
browser harness wrongly assumed Auto always chooses On for a clear photo; that
failure is retained. Only the assertion was corrected, not the app thresholds.

Automatic and assisted observations are reported separately:

| Case/evidence | Observation | Limit |
| --- | --- | --- |
| Automatic mild-turn cloth-mask proposal | Fragmented face/background marks; much cloth missed | Automatic removal fails; correction needed |
| Assisted mild-turn cloth footprint | Main cloth removed; plausible lower-face estimate; DGP On softens visible detail | Reused development example, not family qualification or hidden truth |
| Mask with clear glasses | Difficult three-quarter diagnostic; automatic misses mask/marks glasses; assisted has pale/blue nose residue | Outside initial pose acceptance; incomplete removal |
| Frontal clear-glasses control, empty mask/Off | Entire prepared input exact, no inpainting/restoration | Preserves glasses and non-obstructing hair in this control |

The covering examples are previously inspected development **photographs, not
native CCTV** or pristine holdouts. They establish no geography or ethnicity
coverage. Hands, obstructing hair, scarves, other objects, sunglasses and strong
lens glare still need separate current-route review. Historical completion reports
do not become v3 acceptance by inheritance. This is assistant development visual
review, not independent final assessment.

## Evidence and remaining milestones

| Evidence | Location |
| --- | --- |
| Native parity/inference receipt and six-row sheet | `outputs/dgp_app_v3_inference_audit/receipt.json`, `native-core-six.png` |
| Actual development visual review | `outputs/dgp_app_v3_visual_review.json` |
| Inline browser layout and flows | `scratch/dgp-v3-layout.json`, `dgp-v3-flow.json`, `dgp-v3-scoped-flow.json` |
| Download check and harness failure | `scratch/dgp-v3-download-audit.json`, `dgp-v3-flow-first-attempt.json` |
| Source/test/evidence binding | `outputs/dgp_app_v3_integration_record.json` |

Milestone 4 has a functioning DGP-led research route and overrides. Milestone 5
has functional browser evidence. Useful native restoration, calibrated usability,
full automatic/assisted covering-family acceptance and independent final review
remain incomplete. Keep the full goal active and failed preservation gates intact.
Any further training requires a separately justified finite VM protocol and exact
manual transfer/launch commands. No new pilot is prepared by this integration.
