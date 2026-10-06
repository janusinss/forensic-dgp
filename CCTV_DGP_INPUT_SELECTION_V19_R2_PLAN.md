# V19 r2 — separate normalization correction, inference only

**Closed 5 October 2026:** returned 520-case execution is independently audited
and all five original-cell sheets reviewed. Normalization parity passes; 30
automatic/33 unconditional preservation checks fail. Do not repeat this frozen
job or adopt the candidate. Sources, gates, splits and checkpoints stay unchanged.
[Returned results and separate brightness diagnosis](<C:/xampp/htdocs/YEAR 4/Testing/CCTV_DGP_INPUT_SELECTION_V19_R2_RESULTS.md>).

The following is the original frozen preparation design.

The returned L4 diagnostic and independent local audit confirm why original
V19 stopped. Preserve original V19, its failed gate, all sources/checkpoints,
and the diagnostic. This new immutable package changes the comparison input
encoding; it performs no training, threshold fitting or checkpoint search.

## Demonstrated cause

The7,380,842-byte diagnostic archive matches its exported receipt and SHA256
`9d910721986df024a3ba346766c6385487603bb5174ebc771456c3d8159d7f1c`.
Results SHA256:
`8a8ac06d6c5948e23c7196c03106a76a1e4902a316952fc47c86fa1a41fc4671`.
L4 worker12.2154s, full supervisor/export13.7215s. Eight frozen DGP forwards,
no other networks or training; exact repeated forwards and unchanged DGP state/
148 original file bindings. Independent saved-array audit0.4919s checks eight
raw arrays, six normalized input tensors, six raw/PNG compositions and all
reference differences with zero local forwards/backwards/optimizer updates.

| Fixed case | NumPy-before-GPU against original reference | CUDA scalar division against original reference |
| --- | --- | --- |
| First failed development case`va_ffhq_28544_clear` | Exact V15 PNG | Four differing colour-channel values, each one byte |
| Fixed clear preview`va_asian_00209_clear` | Exact V15 raw and PNG | Four one-byte channel differences; raw maximum1.25169754e-6 |
| Training parity`v9_tr_ffhq_00084_clear` | Three one-byte channel differences from V18 | Exact V18 raw and PNG |

Both historical paths receive identical RGB bytes. Measured CUDA scalar
division matches reciprocal multiplication and differs from CPU true division
for126/256 byte levels by at most5.96046448e-8. The first development raw
difference is1.07288361e-6. PNG flooring converts a few such differences into
one-byte changes. This is a measured processing incompatibility, not a model
training failure or permission to relax exact PNG equality. Original sources:
[PyTorch2.9.1 CUDA division](https://raw.githubusercontent.com/pytorch/pytorch/v2.9.1/aten/src/ATen/native/cuda/BinaryDivTrueKernel.cu)
and the separately pinned V15/V18/V19 runners.

## Exact correction

Keep both established conventions explicitly:

- The retained DGP comparison arm normalizes with NumPy float32 division before device transfer, exactly matching V15. Every fresh development PNG must still equal its audited historical baseline; all50 fixed raw previews must remain within2e-6.
- The frozen V18 spatial pipeline retains its original CUDA scalar normalization, including its DGP conditioning image, prior features, code head and residual decoder. All50 fresh training raw/PNG parity checks stay unchanged.
- Add one canonical DGP forward per development case. Export the separate internal spatial DGP raw base for every case, so audits cannot confuse the two conventions.
- Save both arm raws/PNGs before the exact baseline check. A stopped case now retains the actual output and a detailed failure receipt.
- CPU cached-head replay uses the measured CUDA input floats; its original5e-5 tolerance stays unchanged. All24 cached DGP bases must exactly bind their saved internal/raw parity arrays.

These conventions differ only in float encoding; source RGB256 inputs, observed
support, crop geometry, floor composition and display processing remain the same.
The spatial candidate's internal base is not the canonical comparison raw; report
them separately. No changed checkpoint, learned layer, loss, selection threshold
or post-processing. DGP remains the image base; CodeFormer features/recognizer are
declared frozen pretrained components. The pretrained restoration comparison
still reuses audited V15 PNGs and embeddings; no fresh raw output claim.

## Frozen cohort, gates and finite limits

Reuse unchanged104 repeatedly used development identities/520 photographic
synthetic cases:51`dataset/asian_faces`,53`dataset/thumbnails128x128`; all five
profiles and ten fixed preview identities. Ten training photographs/fifty cases
are parity-only. Own exact identity/source/target-hash split overlap is excluded;
near duplicates and pretrained corpus overlap remain unexcluded. Folder labels
are not ethnicity, capture provenance, independent final or Zamboanga evidence.
No native24 or reserved32 inputs are used.

Five arms remain resizing, canonical retained DGP, declared pretrained CodeFormer
baseline, fixed V18 update600 spatial output and the unchanged input-only selector
alias. All source/profile and clear/degraded groups are retained. Every original
scientific threshold remains: MSE tolerance1e-12, SSIM/cosine tolerance1e-6,
at least0.1dB degraded PSNR gain and10% MSE reduction. No group regression is
waived. Both unconditional and automatic reports remain; failed quality still
exports. No best checkpoint, target-based selection, threshold refit or promotion.

Inference1,200s, audit300s, supervisor1,800s and export180s within overall.
Case20 projection uses19 steady samples,1.25 safety factor and60s reserve;
stop if the original1,200s cap cannot be met. Original20GiB VRAM/4GiB free disk
requirements remain. Child-process groups receive bounded interrupt/kill stops.
No runtime installation, automatic repeat/resume or VM shutdown.

Expected forwards: DGP1,090=50 parity+520 spatial+520 canonical; prior encoder/
classifier, R2 code head, feature generator and residual decoder570 each;
recognizer624; prior RGB tail/unused V11 zero. All six state hashes stay unchanged;
no trainable parameters, gradients, optimizer construction or backwards.

Returned audit checks2,080 physical PNG metric/cosine rows,1,040 arm raw/PNG
compositions,520 internal DGP raw bases,50 canonical V15 raw previews,50 fresh
training parity cases,570 input choices,520 exact automatic aliases,300 original
grid cells and24 cached CPU decoder replays. Five original-cell sheets require
development review before another model/training decision. Whole CUDA/DGP/prior/
recognizer execution is not independently replayed locally; receipt and source
bindings are evidence with that limit.

## Prepared files and remaining work

Eight normalization/runtime regressions pass in0.122s. Preparation44.58s freezes
an87,831,525-byte archive with191 safe regular members/189 assets. All146 original
V19 members are embedded byte-identically, and the full audited diagnostic is
included. Independent transfer audit2.72s verifies the archive,35 Python3.10
sources,798 data files, exact own split exclusions, lineage, script paths, counts
and unchanged gates, with zero local neural/training calls. Actual frozen-package
import/source/data/lineage preflight passes separately in 16.34s, with zero neural
forwards. No VM launch is claimed.

Protocol SHA256:
`5c128d6715785f84858f762035a168f396b46787d6a864b1d8d6437ffc69dc3f`.
Archive SHA256:
`3397a6c3e81e1b04a492f51ec6e5fad939b5a85bc8b318253a04c7696eba581e`.
Exact manual commands:
[CCTV_DGP_INPUT_SELECTION_V19_R2_VM.md](<C:/xampp/htdocs/YEAR 4/Testing/CCTV_DGP_INPUT_SELECTION_V19_R2_VM.md>).

R2 is a processing correction prepared for controlled development inference.
Execution, complete returned audit, all-five-sheet review, useful native outputs,
insufficient-input handling, DGP-led local application, covering-family readiness
and independent final review remain incomplete. Do not reinterpret original V18
or V19 failures as successes. The full goal stays active.
