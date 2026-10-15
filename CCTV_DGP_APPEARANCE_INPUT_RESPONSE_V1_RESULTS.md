# Appearance-input response V1 result — 14 October 2026

**The diagnostic completed and passed independent auditing. No model was trained
or improved by this run.** All 100 repeated RGB-input gradients meet the original
elementwise and norm bounds. The finite appearance predictions still disagree
with several actual scores, even after the reconstruction decoder is removed
from the differentiation path.

| Retained signed magnitude, largest to smallest | RGB-input canonical pass / fail | Saved double-dot reporting pass / fail | Previous parameter prediction pass / fail |
| --- | ---: | ---: | ---: |
| 0 | 1 / 9 | 1 / 9 | 1 / 9 |
| 1 | 4 / 6 | 4 / 6 | 5 / 5 |
| 2 | 7 / 3 | 7 / 3 | 6 / 4 |

Each row contains ten source/profile groups. Both repeats produce these same
classifications. All 60 group comparisons are resolved under the predeclared
diagnostic noise rule. The retained 5% comparison bound is for response diagnosis;
it does not change any restoration, appearance, RAW/PNG or preservation gate.

At the smallest magnitude, input-space failures remain in
`dataset/asian_faces/lowlight_lr32`,
`dataset/thumbnails128x128/motion_lr48`, and
`dataset/thumbnails128x128/compound_lr24`. The latter two groups cancel roughly
91.91% and 94.74% of the signed individual predictions. Asian-source lowlight
cancels roughly 94.25%. Per-case records still contain real prediction misses;
cancellation is a measured complication, not a complete explanation or a reason
to drop those cases. Dataset names describe source provenance, not ethnicity.

The smallest thumbnail clear-group prediction now agrees in RGB space while
its parameter-space prediction disagrees. At the middle magnitude, Asian-source
motion disagrees in RGB space while the parameter prediction agrees. Earlier
reconstruction-to-RGB behavior therefore also contributes in some comparisons.
The evidence does not support attributing every problem exclusively to either
the decoder or the recognizer.

## What is established

The measurement uses the same 100 exposed paired photographic TRAIN cases and
the exact initial and six signed RAW arrays from the prior completed response
diagnostic. It constructs only the frozen measurement recognizer. There are
40 recognizer forward batches, 200 RGB-input derivative queries, zero restoration
forwards, zero neural-weight displacements, zero optimizer or accepted updates,
zero full epochs, and zero native CCTV or reserved-final pixels.

All 193 returned-file bindings, 1,246 stable response bindings and 5,852 bank
bindings pass. Archive: 107,010,813 bytes, SHA256
`40e96ceb18a67bd38813700320d834a97b37e41483f54940a77ce6e23524e361`. Measurement took 77.61s;
export took 6.27s. Original frozen weights, buffers and
saved Torch CPU/CUDA random state are preserved/restored.
[Independent saved-evidence audit](<C:/xampp/htdocs/YEAR 4/Testing/outputs/cctv_dgp_appearance_input_response_v1_analysis/independent_audit.json>).

The local audit independently replays gradient shapes, byte hashes and norms,
embeddings and scalar arithmetic, frozen 2×3 affine grids, exact actual saved
RGB-displacement projections, repeat comparisons and group response receipts.
Off-case zero derivatives and frozen neural state are VM receipts; those neural
computations were not repeated locally. The original local checker and its
pre-run zero-reference reporting repair remain retained. The repair changes
only the undefined relative ratio for a zero reference; this actual run contains
no zero RGB gradients. No original parity or scientific bound was relaxed.

Maximum absolute repeat-gradient difference is
4.656612873e-10; all 100 original
bounds pass. Saved double-dot arithmetic leaves the same 9/6/3 group failures.
This recalculation uses already saved float32 embeddings; it is not a network
forward or backward in float64. Correct full-network high-precision behavior,
the cause of finite disagreements, and all-dimensional derivative correctness
are still unproved.
[All 60 group and 600 case response records](<C:/xampp/htdocs/YEAR 4/Testing/outputs/cctv_dgp_appearance_input_response_v1_analysis/boundary_review.json>).

## Effect on the model goal

No new restored images or visual sheets were planned or generated. The earlier
20 sheets/1,000 cells were directly reviewed and still show soft details and
no convincing whole-face gain. All prior signed-point preservation failures,
including delivered PNG failures, remain binding. This run supplies no new
structure improvement percentage and does not change the previous rejected
gain figures. USED/DIAGNOSTIC COMPLETE: retain the run; do not rerun it.

Next, separate full-network precision and activation behavior from simple
saved-score reporting before preparing another optimizer or epoch schedule.
[Bounded next design](<C:/xampp/htdocs/YEAR 4/Testing/CCTV_DGP_RECOGNIZER_PRECISION_REVIEW_V1.md>). A completed diagnostic is not a qualified
restorer or authorization to bypass a failed stop.

The permanent app checkpoint SHA256 remains
`646fbb11674e8882ea41e251d6e6e7d468c963979b48c93132def8292834e24b`.
The retained restorer is DeblurGAN-v2-compatible; the comparison's frozen
StyleGAN2 bank is externally pretrained. No own-trained replacement prior or
new incumbent is claimed. Checkpoints, splits, caches, provenance, full states,
failures and the app design remain. Native unpaired evidence and reserved final
identities remain separate. All five milestones and seven automatic/assisted
covering families remain required. More epochs and the full goal remain
unqualified/incomplete.
