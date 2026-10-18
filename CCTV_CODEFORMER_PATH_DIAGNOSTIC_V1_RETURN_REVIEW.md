# CodeFormer path diagnostic V1 returned and independently audited

**16 October 2026 — measurement succeeded; zero optimizer updates; no model promotion.**

The downloaded diagnostic completed within its declared limits. Its eight
paired synthetic TRAIN observations explain part of the training/inference
mismatch and show why removing all learned brightening would sacrifice useful
dark-image improvements. The next experiment should test FP32 optimization
from untouched CodeFormer with the same first-50-update recipe and observation
order. This diagnostic does not prove that precision alone caused V2's failures.

## Download and execution evidence

| Item | Verified result |
|---|---|
|Archive|`cctv-codeformer-path-diagnostic-v1-results.tar.gz`, 79,019,264 bytes|
|Archive SHA-256|`c8ebadef8286ecd56c3080f7ec1311ef209fd3783072a542d46319bdca84b772`|
|Protocol SHA-256|`2d3dcd1dbc3be371de5e10ed9c059e455ce9cc8376bac99f04a1a514acc0d49f`|
|Exact imported namespace|1,243 regular files, 95,606,539 payload bytes; 32 frozen source companions|
|Calls|104 CodeFormer, four HQ teacher, ten LPIPS, 64 gradient queries, **zero optimizer updates**|
|Elapsed|124.916 seconds neural measurement; 138.308 worker; 145.792 supervised run; 13.987 export|
|Peak reserved GPU memory|9,265,217,536 bytes, below the declared 22 GiB bound|

Whole-archive, manifest, source, member hash/size and portable-path checks passed
before exclusive extraction. A final readback rechecked every imported member,
all 56 original packet assets, the destination binding and both scientific audit
receipts. The untouched original, V2 stopped50, HQ teacher and perceptual state
fingerprints match independently loaded CPU tensors. All 13 recorded runtime
state checks agree and report empty parameter `.grad` fields. Runtime tensors
after the run were not exported; their unchanged status is source-bound recorded
evidence, not an independently repeated live measurement.

[Closed evidence chain](outputs/codeformer_path_diagnostic_v1_return_review/closed_chain_v1.json),
[transport](outputs/codeformer_path_diagnostic_v1_return_review/transport_v1.json),
[state audit](outputs/codeformer_path_diagnostic_v1_return_review/state_v1/state_audit.json),
[pixel audit](outputs/codeformer_path_diagnostic_v1_return_review/pixels_v2/pixel_audit.json),
[independent source/science readback](outputs/codeformer_path_diagnostic_v1_return_review/independent_readback_v1.json).

## What was learned

**FP32 training and inference agree on the measured inputs.** At both original
and stopped50 weights, FP32 inference, evaluation with autograd and training
with autograd produce exactly the same saved RAW arrays and selected codes for
all eight observations. All 16 repeated RAW arrays and code arrays are exact.
An ordinary train/evaluation-mode switch is therefore not the observed mismatch
on these cases.

**BF16 changes the content selection before a weight update.** Against FP32,
BF16 changes 145/2,048 selected codes at original weights and 91/2,048 at
stopped50, with 5–41 changes per observation. Mean RAW image differences are
0.005256 and 0.003704 respectively. The two saved original FP32 behavior images
match the new original FP32 outputs exactly, while BF16 introduces image-anchor
L1 of 0.0015124 and 0.0028983. This independently explains the nonzero initial
RGB behavior penalty on those two anchors. Full new logits were not saved, so
the reported logit-gap scalars cannot be independently replayed in this return.

TBF16 records BF16 logits with **FP32 softmax probabilities**. Native probability
ties coexist with changed selected codes. Captured native selections were
checked; no FP32 argmax was substituted. This experiment does not isolate ties
as the only cause of code changes. The observed softmax dtype agrees with
[PyTorch 2.9's CUDA autocast policy](https://docs.pytorch.org/docs/2.9/amp.html#cuda-ops-that-can-autocast-to-float32).

**Learned fusion contributes to both excess brightness and useful exposure.**
Keeping the stopped encoder/content codes fixed, restoring original fusion
reduces paired RGB MAE on all four normal observations (mean 0.04856 to 0.03886)
but increases it on all four dark observations (0.10942 to 0.14957). It removes
some helpful brightening. Original encoder moments alone have smaller sampled
luminance effects. The three disposable interventions keep actual codes exact
and restore the original scoped model states afterward; no hybrid checkpoint
was exported. Their effects do not predict the result of freezing fusion
throughout training.

**The sampled objectives conflict after training.** At stopped50, normal/dark
photometric gradient sums oppose in the encoder (cosines −0.8560/−0.5917) and
fusion (−0.8277/−0.8015). Image losses reaching encoder/fusion but not the hard
token predictor are expected. Token CE/KL reaching the encoder/predictor but
not fusion are also expected. These are not missing-optimizer errors. The
checker verified 64 weighted scalar reductions, 60 cosine formulas and 120
component-norm consistency bounds. Full gradient vectors were intentionally
not archived, so they cannot be independently replayed from this return.

[Saved-pixel interpretation](outputs/codeformer_path_diagnostic_v1_return_review/pixels_v2/scientific_readback.json),
[state and gradient interpretation](outputs/codeformer_path_diagnostic_v1_return_review/state_v1/STATE_AND_GRADIENT_REVIEW.md).

## Visual review

Root inspected all **eight sheets covering 104 conditions**, plus eight actual
native PNGs at 256 pixels. Sheets show unchanged PNGs reduced to 192 pixels for
comparison, without exposure correction or sharpening. The three FP32 columns
match visibly, consistent with the exact arrays. On the severe24 example the
stopped model reduces coarse false texture and brings the mouth/skin appearance
closer to the supplied clean photograph. Some dark outputs become easier to
read but flatter or pale. Returning original fusion darkens them again without
solving every content or contrast discrepancy. Clear glasses remain visible in
these sampled comparisons. These observations are not a user-acceptance verdict
or a substitute for the failed DEV clear controls.

[All visual sheets](outputs/codeformer_path_diagnostic_v1_return_review/visual_sheets/INDEX.md),
[root visual receipt](outputs/codeformer_path_diagnostic_v1_return_review/visual_review_v1.json).

## Audit repairs and retained history

The first local pixel checker expected duplicate input data inside the return.
The actual export intentionally retains inputs in the original frozen packet.
The corrected checker hashes those original inputs against both protocol and
case bindings; the first checker and failure are retained under `pixels_v1`.

The first local state checker incorrectly required BF16 softmax probabilities
when logits were BF16. Source and actual dtype evidence established the FP32
autocast path. The first checker, full traceback and repair basis are retained
under `state_v1_attempt1`. The corrected state audit and pixel audit pass.
The final CPU fixture suite contains **48 passing tests**. No VM worker,
training recipe, model weights or quality gate was changed by either repair.

The initial renderer's per-sheet membership metadata used a cumulative list.
The closed visual receipt reconstructs exact membership from the hashed PNG
bindings; all eight image files and the original render receipt remain intact.

## Decision and limits

Use a **new finite FP32 training comparison**, from untouched CodeFormer, to
test actual optimization under inference-matched precision. Retain V2's first
50 update positions, data, Adam settings, learning rates, losses, trainable
groups, comparisons and quality guards. Do not resume the failed V2 state,
increase its epochs, freeze all fusion or reinterpret its failures as passes.

[Prospective next-training design](CCTV_CODEFORMER_FP32_COMPARISON_V1_DESIGN.md).
This design is not an uploaded or runnable packet. It requires source,
resource-accounting and transfer review before exact manual launch commands
are published. No further optimizer or VM action occurred during this audit.

Evidence here covers **eight already observed TRAIN inputs from four clean
photographs**, not native CCTV, unseen DEV faces or reserved FINAL identities.
The diagnostic creates no trained candidate and establishes no restoration,
identity, Zamboanga, automatic-mask or completion qualification. V1 remains
quality-stopped at 200 and V2 at 50, with their original failures retained.
Moderate helpful brightening remains an accepted preference; washed-out detail
and altered visible appearance remain problems. Restoration acceptance, native
and final review, integration and full app verification remain outstanding.
The full goal stays active; completion/detector development remains deferred.
