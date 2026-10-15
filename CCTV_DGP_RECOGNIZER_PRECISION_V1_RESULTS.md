# Recognizer precision V1 result — 14 October 2026

**The measurement completed and passed independent auditing. Full FP64
computation does not resolve the sampled appearance-response disagreement.**
Six of ten float32 comparisons miss the declared diagnostic bound; five of ten
float64 comparisons miss it. All 30 original gradient parity checks pass.
No restorer ran, no training update was accepted, and no new image gain was
produced. The existing trained model and scientific gates remain unchanged.

| Selected exposed TRAIN case | FP32 relative prediction residual / status | FP64 relative prediction residual / status |
| --- | ---: | ---: |
| v9_tr_ffhq_59043_clear | 8.2620% / fail | 2.3706% / pass |
| v9_tr_ffhq_59043_blur_lr24 | 5.9861% / fail | 5.8694% / fail |
| v9_tr_ffhq_59043_lowlight_lr32 | 0.6402% / pass | 0.6055% / pass |
| v9_tr_ffhq_59043_motion_lr48 | 2.2422% / pass | 4.6721% / pass |
| v9_tr_ffhq_59043_compound_lr24 | 6.7857% / fail | 6.8584% / fail |
| v9_tr_asian_06782_clear | 6.8005% / fail | 8.8916% / fail |
| v9_tr_asian_06782_blur_lr24 | 3.5842% / pass | 3.9304% / pass |
| v9_tr_asian_06782_lowlight_lr32 | 9.6853% / fail | 9.8010% / fail |
| v9_tr_asian_06782_motion_lr48 | 1.9361% / pass | 2.0580% / pass |
| v9_tr_asian_06782_compound_lr24 | 11.6818% / fail | 11.6883% / fail |

These values are directional prediction errors, not identity recovery or image
restoration gains. The comparison uses the predeclared 5% diagnostic bound and
noise rule. All 20 responses are resolved; no noise tolerance was changed. The
two references were selected from exposed failure evidence, so these ten cases
are not an independent population or final evaluation. Each source contributes
one reference with all five clear/degraded profiles. Dataset names are
provenance labels, not ethnicity or local CCTV claims.

## What changed in the measured computation

The float64 branch promotes the same saved RGB values, fixed float32 grids
and neutral fill, uses a disposable double recognizer copy, and keeps crop,
encoder, normalization and scalar arithmetic in double. It round-trips exactly
to the original frozen float32 weight state. All floating encoder outputs pass
their declared dtype checks. This is a genuine full-computation precision
comparison, unlike recalculating only saved float32 vector dots.

Float64 changes the thumbnail clear comparison from fail to pass. The five
other failed float32 cases remain failures. In the selected Asian-source
lowlight case, residual error is 9.6853% in FP32 and 9.8010% in FP64. In the
thumbnail compound case it is 6.7857% and 6.8584%. Double precision is therefore
not a general repair for these sampled finite responses. Original scientific
FP32/PNG scores and all previous rejected restoration decisions are retained.

Packed activation-map replay finds 1,838 and 1,852 sign changes for plus/minus
FP32 inputs and 1,845 and 1,844 for FP64. Each count spans 37,130,240 units in
50 module/reference maps. Both signed endpoints change signs in every such
map; repeated initial inputs change zero signs. These are PReLU region
transitions, not visible pixels or generated facial regions. They support
finite nonlinear behavior as a contributor but do not prove exclusive
causation, a wrong backward kernel, or correctness of every derivative.
[Paired scores and activation recount](<C:/xampp/htdocs/YEAR 4/Testing/outputs/cctv_dgp_recognizer_precision_v1_analysis/precision_activation_review.json>).

## Independent audit and preserved state

All 462 return-file bindings, 192 stable input-response,
1,246 conditioning-response and 5,852 bank bindings pass. The 41,576,988-byte
archive SHA256 is `6bbb53a45afaf6676675616c75775b35a16e3e4d9d5dbe3cb30f59b28963b54a`. Manual measurement took
48.69s and export 2.98s.
The run completed 16 recognizer forwards and 40 RGB-input derivative queries,
with zero restoration forwards, learned-weight changes, optimizer/accepted
updates or full epochs. Frozen states, buffers and saved Torch CPU/CUDA RNG
are preserved/restored. All ten FP32 repeats, ten FP64 repeats and ten FP32
versus prior-input comparisons meet the original elementwise and norm bounds.

The original prospective auditor passed without a returned-checker repair.
It independently replays complete gradient arrays and actual saved pixel
projections, embedding/scalar arithmetic, packed sign maps, transition counts,
repeat receipts and finite response classifications. Neural state and off-case
zero-gradient assertions remain VM receipts; no neural forward or derivative
was re-executed locally. Original checkpoint and protected app hashes pass.
[Independent audit](<C:/xampp/htdocs/YEAR 4/Testing/outputs/cctv_dgp_recognizer_precision_v1_analysis/independent_audit.json>).

No new face images or visual sheets were planned or generated. The earlier
20 conditioning-response sheets/1,000 cells were directly reviewed and remain
soft/unqualified. Archive integrity and measurement success are not trained
model improvement. USED/DIAGNOSTIC COMPLETE: preserve this run and do not rerun.

## Next model-improvement decision

The assumption that higher recognizer precision alone would restore reliable
finite predictions is unsupported. Repeating precision diagnostics or adding
epochs to the failed single-initial-gradient recipe is not justified. The
next finite training design should consider gradients at the initial state
and nearby measured states, with every actual proposal checked under the
unchanged scientific RAW/PNG gates. This is a change to the learning calculation,
not promotion of the two selected references or any existing failed state.
[Bounded nearby-gradient training design](<C:/xampp/htdocs/YEAR 4/Testing/CCTV_DGP_NEARBY_GRADIENT_LEARNING_REVIEW_V1.md>).

The permanent app checkpoint remains
`646fbb11674e8882ea41e251d6e6e7d468c963979b48c93132def8292834e24b`.
The original restorer is DeblurGAN-v2-compatible; the comparison's StyleGAN2
bank is externally pretrained/frozen, not an own-trained face prior. All
checkpoints, splits, caches, provenance, full states and failures are retained.
Native CCTV remains unpaired and final identities remain outside tuning.
All five milestones and seven automatic/assisted covering-family requirements
remain incomplete. Training, derivatives and parameter changes remain manual
on the existing L4 in tmux; agent VM access remains maintenance only.
