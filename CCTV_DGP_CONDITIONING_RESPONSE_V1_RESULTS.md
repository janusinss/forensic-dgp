# Conditioning-response V1 returned result — 14 October 2026

**The diagnostic completed successfully; no improved model qualified.**
The manual L4 run measured the original conditioning path twice and six fixed
signed parameter changes. It constructed no optimizer, solved no new direction,
accepted zero updates and ran zero training epochs. The sampled positive
changes reduce structure error, but none meet the original RAW/PNG preservation
and 1% global/per-source endpoint requirements together.

| Fixed diagnostic point | Signed rate | RAW structure gain | PNG structure gain | Failed preservation checks RAW / PNG |
| --- | ---: | ---: | ---: | ---: |
| minus00 | -0.004105419386 | -0.481426% | -0.469081% | 51 / 51 |
| plus00 | 0.004105419386 | 0.346369% | 0.361388% | 14 / 10 |
| minus01 | -0.0004105419386 | -0.041974% | -0.037387% | 51 / 49 |
| plus01 | 0.0004105419386 | 0.040293% | 0.053512% | 1 / 1 |
| minus02 | -4.105419386e-05 | -0.004125% | -0.006029% | 48 / 42 |
| plus02 | 4.105419386e-05 | 0.004101% | 0.006602% | 0 / 5 |

These are relative reductions of degraded landmark high-frequency error;
they are not percentages of identity recovered. The largest positive point
is the previous RAW R1 pilot's smallest rejected trial: all 204
group scalars match exactly. This repeat measurement establishes no new gain
over that result. Negative points are response probes, not training proposals.
No candidate was selected, even at the smallest positive point where RAW
preservation passes and PNG still fails five checks.

At `plus01`, RAW fails one appearance check for
`dataset/thumbnails128x128/compound_lr24`; PNG fails one for
`dataset/asian_faces/blur_lr24`. At `plus02`, RAW has no preservation failure,
but PNG fails two MSE and three fixed-recognizer appearance checks. Source
labels describe the retained datasets, not ethnicity or local CCTV performance.
[Every signed point and retained decisions](<C:/xampp/htdocs/YEAR 4/Testing/outputs/cctv_dgp_conditioning_response_v1_analysis/response_review.json>).

## What the completed measurement explains

All six source-specific RAW structure central comparisons fall within the
predeclared 5% diagnostic comparison bound. At the smallest magnitude the two
relative slope errors are below 0.202%. MSE passes 29/30 such comparisons and
SSIM passes 29/30. Appearance passes only 12/30: its disagreements are 9/10,
5/10 and 4/10 from largest to smallest magnitude. Every comparison is resolved
under the original declared diagnostic noise rule. A directional comparison
is not an all-dimensional derivative proof, a preservation gate, or training
qualification; the 5% bound does not relax a single image-quality criterion.

The small-step pixel/detail direction is informative, but its first-order
appearance prediction does not reliably describe the actual finite response.
Even response and PNG rounding also matter. This result supports locating that
response boundary before another optimizer or epoch schedule; it does not
uniquely prove a faulty backward operator or a failed architecture.

The saved baseline and target recognizer embeddings are exactly unchanged in
every snapshot. Fresh initial double-precision group scores match both recorded
gradient-forward group values within 1.11e-16. Recomputing scalar dots from the
same saved vectors in float64 leaves the same 9/5/4 central disagreements;
the largest float-versus-double two-sided difference is 2.96e-8. Scalar dot
rounding alone does not explain the remaining sampled disagreements. This
is saved-array reporting only: the network was not re-executed in float64,
and no scientific metric or quality threshold was changed.
[Saved embedding precision review](<C:/xampp/htdocs/YEAR 4/Testing/outputs/cctv_dgp_conditioning_response_v1_analysis/saved_embedding_precision_review.json>).

The smallest positive point changes 22,124 RGB channel values across all 100
PNGs, by at most one byte level each. Its correct delivered conversion still
fails five preservation checks. Correct PNG export is independently verified;
neither a RAW-only pass nor approximate PNG derivatives guarantee delivery
preservation. Pixel-change counts do not establish a visible improvement.
[All six delivered pixel recounts](<C:/xampp/htdocs/YEAR 4/Testing/outputs/cctv_dgp_conditioning_response_v1_analysis/PNG_change_review.json>).

## Retained reproducibility limit

All 32 vector-norm comparisons between the two repeats pass. Their original
elementwise parity requirement passes 31/32 groups; one element in
`RAW/dataset/thumbnails128x128/lowlight_lr32/ArcFace_loss` still misses it.
At index 2,451,098 its difference is 1.1781230568e-8 against the unchanged
1.0507631339e-8 bound. Repeat00 versus prior R1 passes 32/32; repeat01 versus
prior R1 passes 31/32 with the same index missing that comparison's bound.
The previously retained R1 parity failure remains. No numerical tolerance was
relaxed and no blanket claim that every derivative reproduces is made.

## Audit and images

All 1247 return-file bindings and
5852 required parent bindings passed path/size/hash
checks. Archive: 1,715,618,837 bytes, SHA256
`6eb507e1467b5552bf67aa809457822d2ed68faf759db36696d770d2ef24b84e`. The original prospective returned checker passed
its declared saved-evidence checks while explicitly reporting the parity misses;
no failure-only replacement checker was needed for this return.

The audit independently replays 800 RAW outputs, 800 PNGs and 1,600 stage metric
rows, saved embeddings, exact observed-support composition, all 64 aggregate
vectors, contribution arithmetic, six rounded parameter differences, 96 central
responses, unchanged scientific decisions and protected model/app files.
Initial and rollback outputs match the original DGP exactly; selected component
weights, frozen full states, buffers and saved Torch CPU/CUDA RNG states are
restored. This is rollback proof, not a full optimizer-resume checkpoint.
No neural derivatives or embeddings were re-executed locally.
[Prospective independent audit](<C:/xampp/htdocs/YEAR 4/Testing/outputs/cctv_dgp_conditioning_response_v1_analysis/independent_audit.json>).

All 20 planned sheets and 1,000 cells were directly viewed at original
2560x1400 resolution. Eye, mouth, tooth, skin and hair details remain soft on
degraded rows. None shows convincing new whole-face detail over initial.
Some retained source-photo targets also limit available fine detail; this does
not invalidate the separately audited HQ targets from the other source.
[Per-sheet visual observations](<C:/xampp/htdocs/YEAR 4/Testing/outputs/cctv_dgp_conditioning_response_v1_analysis/visual_review.json>).

The worker took 366.59s, with 200 restorer and 360
recognizer forward batches, 640 derivative queries and six temporary component
displacements. Export took 100.64s. `complete: true` describes export, and
`complete_diagnostic: true` now has an independent audit; neither qualifies
training success. USED/DIAGNOSTIC COMPLETE: retain this run and do not rerun it.

These are 100 already exposed paired photographic TRAIN cases, not native
CCTV, reserved final identities, or a completion evaluation. Original app
checkpoint SHA256 remains
`646fbb11674e8882ea41e251d6e6e7d468c963979b48c93132def8292834e24b`.
The original restorer is DeblurGAN-v2-compatible; the comparison branch's
StyleGAN2 bank is externally pretrained and frozen, not our trained face prior.
All checkpoints, splits, caches, provenance, full states, failures and app design
remain. All five milestones and seven automatic/assisted covering families are
still required. More epochs and model promotion remain unqualified.
[Next bounded diagnosis and primary research](<C:/xampp/htdocs/YEAR 4/Testing/CCTV_DGP_APPEARANCE_RESPONSE_REVIEW_V1.md>).
