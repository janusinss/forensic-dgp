# V24 — audited early structure failure

The downloaded V24 return is independently audited and all 50 exposed training
cases are visually reviewed on 6 October 2026. The original update50 structure
stop is correct. The degraded landmark-detail improvement is **0.0033227502%**
against the unchanged **1%** early requirement. There is no convincing visible
whole-face improvement over the frozen own-DGP baseline. Close this recipe;
retain its original source, checkpoints, partial head and failed gates.

The exporter completed a failure archive. Its `complete:true` does not imply
training success. No final800 result exists. No candidate is selected for the
application, and the full restoration/completion Goal remains incomplete.

## Transfer and execution evidence

Archive: `outputs/cctv-dgp-degraded-detail-v24-results.tar.gz`, 76,492,526 bytes,
SHA256 `1d758362366f3e1f17ddf8239b0f41940b6b7c6ebf349a86ed76e0abdf63dd15`.
The checksum, export receipt, exact safe manifest and all 371 regular returned
members pass import. All 221 frozen execution assets and the original protocol
`76ab24695a40b411832e2d678c79b0e2a80f6320d4525970b2dd32db2379e114`
are verified. Returned source is compared with pinned local source; it is not
executed by the audit.

The L4 receipts show 50 optimizer updates and 51 backwards including the separate
one-batch gradient check. Nonzero direct-path gradients and changes in all eight
learned tensor groups confirm learning activity. Fixed filters, DGP and ArcFace
states remain unchanged. The stopped state is
`d3dc1892385d3dab80de5e44737b16501e6168179460502e25bd9251e9d2258b`.

The source-bound execution audit verifies all 50 step samples and the original
update20 timing projection. Worker time is 19.4532 seconds; fitting time 6.6165
seconds; supervisor time 25.3731 seconds. Peak torch-allocated CUDA memory is
847,820,288 bytes (808.54 MiB), within the 20 GiB limit. These are verified saved
receipts, not a new live GPU measurement or total device-memory measurement.
The quality gate, rather than a timing or memory limit, stops the run.

## Independent replay and unchanged gates

The 43.42-second local CPU audit replays 100 head predictions and 110 fixed
recognizer predictions. It verifies both complete snapshots (0/50), all 100
raw/PNG compositions, 100 metric rows, all 17 source/profile groups, padding,
cohort scalars, failure arithmetic and retained stopped weights. Maximum raw
error is 5.9605e-8; vector error 3.0547e-7; cosine error 4.1723e-7; metric error
7.1054e-15. Original replay and quality tolerances are unchanged.

The baseline cohort receipt passes a separate 100-fixed-filter CPU check on all
50 rows, with maximum CPU/CUDA scalar difference 4.6566e-10 inside the already
declared 2e-9 receipt tolerance. Forty degraded cases determine the two exact
float32 cohort normalizers; ten clear controls receive no clean-target reward.
This scalar compatibility check does not loosen any restoration-quality gate.

| Degraded landmark high-frequency error | Baseline | Stopped50 | Relative improvement |
| --- | ---: | ---: | ---: |
| Delivered PNG, original gate | 0.0019965976532523044 | 0.001996531311299121 | 0.0033227502% |
| Raw float, separate diagnostic | 0.0019956644956366345 | 0.0019956285043846627 | 0.0018034721% |

The raw improvement is also negligible; PNG quantization alone does not explain
the failure. A supplementary application of the unchanged capacity arithmetic
at stopped50 flags eight fixed-ArcFace group regressions and a brightness-only
gain fraction of 0.289120 above 0.20. That fraction concerns a very small total
gain; it does not establish a large exposure change. These are diagnostics at
the stopped snapshot. The final800 capacity test never executes.

## Every-case visual and spatial review

All ten 1072×1516 sheets are viewed at original detail. The independent saved-
array checker verifies 375 source bindings, ten sheet hashes and all 200 exact
256×256 cells: camera input, own DGP, stopped50 and paired training target.
No display enhancement is applied. Eyes, nose, mouth, face outline and overall
visible appearance are assessed together, following the user's “all are important.”

Clear glasses, visible hair and facial features remain softened by the baseline.
Blur/lowlight/compound rows retain weak boundaries; motion rows retain some
coarse structure. The stopped outputs offer no discernible structure gain in
any of the 50 cases. All PNGs differ numerically, but the maximum difference is
one colour level and only 29,432 pixels change across the 40 degraded canvases.
Nonidentical files do not establish useful reconstruction.

The known-source exact-layer trace runs 50 stopped-head layer replays in 16.27
seconds, matching audited raw outputs. Median observed degraded correction RMS
is 4.3432e-5 in [0,1], or 0.011075 of one byte level. The median ratio of interior
post-projection RMS to pre-projection spatial RMS is 0.0332342. The final Gaussian
high-pass/mean projection strongly attenuates this particular stopped correction.
This describes the saved fields; it is not an isolated training-gradient causal
experiment or proof that every head in the family is incapable of learning.

## Corrected-objective diagnosis and architecture discussion

The original V24 objective is decomposed on both saved states for all 50 cases,
using pinned local function definitions and frozen CPU ArcFace inference. In
74.51 seconds, 210 recognizer calls and 100 saved-output references verify all
seven terms. There are zero head predictions, backwards or optimizer updates.
A separate 0.52-second arithmetic checker verifies 176 bindings, 100 case states
and six group states; maximum term-sum difference is 2.3842e-7.

Equal-case objective changes from 1.2999999526 to 1.2999843621. The degraded cohort
supplies 1.7820e-5 reduction, while clear preservation costs subtract 2.2295e-6.
Clear clean-target reward is exactly zero. Thus V24 corrects the measured V23
reward mismatch, but that correction does not supply adequate useful structure.
This fixed CPU calculation does not prove GPU step-gradient causality.

V22, V23 and V24 all fail the same original early gate. Stop this small filtered
head sequence. [The architecture review](<C:/xampp/htdocs/YEAR 4/Testing/CCTV_DGP_POST_V24_ARCHITECTURE_REVIEW.md>)
compares revising our own DGP spatial/feature path with the previously permitted
declared face-prior extension. No fourth recipe, training packet or model fix is
prepared before that discussion. The user subsequently selects Route A, revising
our DGP spatial/feature path; the original failed V24 recipe remains closed.
The debugging rule is explicit: “DON'T attempt
Fix #4 without architectural discussion.”
[Read the rule](<C:/xampp/htdocs/YEAR 4/Testing/.codex/skills/systematic-debugging/SKILL.md>).

Only the same ten exposed TRAINING photographs and their five synthetic profiles
are used. No native, reserved final or new covering pixels are accessed. Native
CCTV remains unpaired; source labels do not imply ethnicity or Zamboanga
performance. The previously useful native crop remains a usable input needing
better output structure. The original DGP-led app/design and its 22 bindings
remain intact. Historical 34 regressions and bundled inline Playwright results
are preserved; they are not rerun for this audit. Useful native outputs, canonical
app parity, all seven automatic/assisted covering families and independent final
review remain required.

Evidence: `outputs/cctv_dgp_degraded_detail_v24_return_import.json`,
`outputs/cctv_dgp_degraded_detail_v24_independent_audit.json`,
`outputs/cctv_dgp_degraded_detail_v24_diagnostic/`,
`outputs/cctv_dgp_degraded_detail_v24_loss_audit_v1/` and
`outputs/cctv_dgp_post_v24_architecture_review_v1/review.json`.
