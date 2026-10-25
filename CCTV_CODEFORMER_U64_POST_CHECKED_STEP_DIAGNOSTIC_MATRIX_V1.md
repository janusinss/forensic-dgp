# U64 after PNG-checked step: diagnostic evidence matrix

23 October 2026. Three preservation routes failed their declared screens.
Model/training changes are stopped while the user chooses the next diagnostic
review. This document compares retained evidence; it selects no next pilot,
scale, optimizer, architecture or training-data change.

Original trained u64 remains the retained challenger. U96 remains the experimental
app incumbent. All rejected states remain retained, with no earlier-checkpoint
or other-trial fallback.

## Three routes, three observed limitations

| Route | Actual mechanism and completed scope | Observed rejection | Supported diagnosis and boundary |
| --- | --- | --- | --- |
| Clean-anchor continuation | Fresh Adam from exact original u64; eight new commits, 32 backward calls, 204 forwards. Photographic targets replaced generated clear/low-light anchors in the existing weighted four-role objective. | At update 8, clear photographic MAE and legacy preservation checks failed. PNG clear MAE rose from 0.03172407 to 0.03567976 (+12.4691%); low-light MAE fell 7.9366%. Blur structure gains against frozen U96 exceeded 1%, but did not override fidelity/behavior failures. | A real fidelity/preservation tradeoff occurred. Later initializer-only measurements support low-light-dominated, conflicting gradients. They do not uniquely explain the eight-update Adam trajectory. Update 4 was not retrospectively accepted; update 16 was never run. |
| Gate-aligned pilot | Two of four permitted plain gradient-descent updates; 300 forwards and 256 gate-gradient queries. Eight pooled RAW MAE/adjacent gradients replaced the weighted objective direction; actual rounded displacements had eight negative RAW dots. | At update 2, local central128 RAW adjacent error fell to 0.0169670592095 while PNG adjacent error rose to 0.0169714465417, versus original PNG 0.0169709439626. PNG increased 5.025791143e-7 (+0.0029614% relative error). | Sparse RGB8 code crossings can reverse the delivered-PNG endpoint despite RAW improvement. Six of eight local PNG cases worsened versus original. The original 1% U96-relative endpoint was not the failed check. No update-3/4 or accepted finite-PNG remedy was established. |
| Actual-PNG checked finite step | Four independent original-state proposals with caps 2e-5 / 1.5e-5 / 1e-5 / 5e-6 and fixed 0.99 headroom; 436 forwards, four trial writes, four original restorations, one selected commit, zero new gradients/optimizer updates. | Trials 0/1/2 passed optimization role means; trial 3 failed. First eligible trial 0 was selected. Its separate final bank failed local PNG adjacent retention: 0.01697094396257641 to 0.01697106960719266, +1.2564461624939294e-7 (+0.00074035137%). Inherited, original-terminal and four strict clear/low-light checks passed. | Actual finite-PNG eligibility on optimization means did not transfer to the separate adaptive monitor means. All eight local RAW cases improved; five PNG cases worsened. This distinguishes the failing endpoint from corruption, frozen-state drift or absence of gradients. Other trial monitor outcomes were not measured or qualified. |

Sources: [clean-anchor return](CCTV_CODEFORMER_U64_CLEANANCHORS_CONTINUATION_V1_RETURN_REVIEW.md),
[gate-aligned return](CCTV_CODEFORMER_U64_GATE_ALIGNED_PILOT_V1_RETURN_REVIEW.md),
[checked-step pixel receipt](outputs/codeformer_u64_png_checked_step_v1_return_pixels/receipt_v1.json),
and [checked-step state receipt](outputs/codeformer_u64_png_checked_step_v1_return_states/receipt_v1.json).
These sources and all three exact protocols are hash-bound in the companion receipt.

## What existing measurements establish and leave open

The latest selected trial improved the optimization local PNG adjacent mean by
4.77448863813e-8 while worsening the different monitor mean. Its optimization bank
still contained 49 increases among 256 case/view/metric rows across 32 of 64 case
IDs. The selected quality bank had 56 increases among 192 available photographic
rows across 30 IDs. These are correlated metric rows, not counts of unusable
photographs. A favorable role mean does not establish individual preservation.
The fixed selection policy did not require every optimization case to improve.

The [integer endpoint diagnosis](outputs/codeformer_u64_png_checked_step_v1_return_audit/endpoint_diagnosis_v1.json)
explains the latest PNG delta from saved pixels: 882 of 393,216 central RGB
components changed; absolute target-edge residual increases and decreases net to
+25 RGB8 units. The integer reconstruction differs from the saved FP32-normalized
delta by 2.94e-13. That is arithmetic readback error, not extra gate tolerance or
evidence of useful visible gain. Root's 21-sheet review covers all 436 outputs
and records no convincing useful visible gain.

The [zero-update geometry evidence](CCTV_CODEFORMER_U64_ROLE_GRADIENTS_V1_RETURN_REVIEW.md)
found low-light objective norm share averaging 62.23%, with local/global/clear
opposition in 14/16, 15/16 and 12/16 matched photographs. Same-role objective/MAE
alignments were positive, but objective/adjacent alignments were not uniformly
positive. These facts support reviewing role weighting and metric alignment;
they do not show a disconnected or universally wrong objective.

The existing 36-head space admits an initializer-local common direction for
eight pooled gates. The later [full-16 saved-vector analysis](outputs/codeformer_u64_png_surrogate_measurement_v1_return_audit/feasibility_v1.json)
found that the old eight-RAW direction already descended all 16 pooled
RAW/identity-PNG-surrogate signals. Its cosine with the full-16 solution was
0.99999338198. Adding those nearly duplicate surrogate vectors is therefore not
a demonstrated corrective mechanism. The surrogate is biased; it is not the
actual derivative of floor-to-RGB8 conversion.

## What a review can establish without another pilot

| Review choice, still pending | Read-only questions and usable retained evidence | Conclusions the review cannot establish |
| --- | --- | --- |
| Training examples and loss design | Audit the fixed 16 optimization and 8 adaptive monitor TRAIN photographs, four correlated synthetic roles, clean-target alignment, degradation/ROI provenance, and per-case concentration of regressions. Compare the actual clean-anchor formulas with MAE/adjacent gates: SmoothL1 clear anchors, block16 luminance low-light anchor, local inside/outside weighting, division by four, pooled gate normalization, and PNG selection. Separate variation across photographs from quantizer code crossings. | These synthetic TRAIN examples do not establish real CCTV degradation coverage, identity preservation, ethnicity or Zamboanga performance. Existing monitor reuse is not an independent holdout. Any eventual revised loss or additional examples still require evidence review, a new finite protocol and the user-manual L4 execution workflow; none is chosen here. |
| Trainable reconstruction path | Read the exact post-fusion formula and parameter census: 36 heads / 168,960 scalars at sites 32/64/128/256; 539 R2/U96 tensors frozen. Heads use existing observed/decoder/fused/coarse features and add tanh(h) times (fused-decoder). At a site, zero stock residual gives zero componentwise correction; upstream frozen code prediction itself is not trainable. Compare these constraints with retained output failures and already favorable local pooled directions. | Source constraints alone do not prove that head capacity caused the failures or that more layers, unfreezing, routing or another architecture would succeed. The heads can still modify downstream decoded features. Saved gradients/outputs are not a representation-rank or counterfactual architecture experiment. No new model call is authorized by this review. |

The [role contract](research/codeformer_u64_role_gradients_v1/contract.py) binds
exactly 64 optimization rolecases, 32 monitor rolecases and 24 inherited guards,
all TRAIN. Some legacy dark guards have no aligned target. The [objective source](research/codeformer_u64_cleananchors_continuation_v1/objective.py)
and [post-fusion head source](research/codeformer_u96_postfusion_v1/model.py) fix
the loss/path descriptions above. Native/wild remain unpaired. Broader DEV
failures remain binding; reserved FINAL remains unopened/execution-unfrozen,
with independent reviewers pending.

## Why another tiny-step or extra-epoch run is not justified now

1. Three different finite recipes already failed preservation; integrity, real
   commits and nonzero gradient paths were verified. More execution is not a
   repair for an absent file or unexecuted update.
2. RAW directional signs and nearly identical surrogate directions have already
   been insufficient for actual PNG/monitor success. Gradient locality, integer
   code boundaries and cohort transfer must remain distinct explanations.
3. The latest fixed four-scale comparison did not show monotonic eligibility as
   scale shrank: the smallest trial failed optimization eligibility, while three
   larger trials passed. Their unmeasured monitor outcomes cannot be inferred.
   More post-hoc scales or a fallback chosen from the failed monitor would change
   the experiment.
4. Larger or repeated movement has no evidenced preservation benefit: Adam
   already harmed clear fidelity, while the audited microsteps showed no
   convincing useful visible improvement. This does not prove that every
   possible step is impossible; it removes the evidence for an unchanged rerun.
5. The next deliverable is the user's chosen evidence review and an explicit
   account of what new mechanism it would test. It is not another worker,
   transfer archive, launch guide, extra epoch or automatic pilot.

All original gates, renderer/quantizer, app behavior, checkpoints, splits and
failed states remain unchanged. This matrix used saved reports, source text and
JSON receipts only: no model import, new gradient, optimizer, parameter
application, VM or browser call. Full goal remains active and incomplete.
