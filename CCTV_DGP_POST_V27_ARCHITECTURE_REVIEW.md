# Own-DGP architecture discussion after V27

**User decision, 6 October 2026:** review a separate copy of our original DGP
reconstruction decoder. The direct reply is preserved in
[the decision record](<C:/xampp/htdocs/YEAR 4/Testing/outputs/cctv_dgp_post_v27_architecture_review_v1/user_architecture_decision.json>).
The original checkpoint, frozen encoder and stored normalization statistics stay
protected. Source/initial-parity review precedes a distinct finite VM protocol;
this decision does not establish model usefulness or permit an unchanged rerun.

All three spatial-path attempts are closed after actual return audits and whole-
cohort review. The invalid assumption is that these finite small-head changes on
frozen own-DGP pixels/features would provide enough whole-face structure under
the retained data, objective and constraints. Nonzero gradients and changed
pixels are insufficient evidence of useful facial restoration. This does not
prove every possible head incapable or identify a unique optimization cause.

| Stopped TRAIN pilot | Degraded structure gain | Median correction, byte units |
| --- | ---: | ---: |
| V25 | 0.0282235% | 0.1299423 |
| V26 | 0.0335636% | 0.1348669 |
| V27 | 0.2210886% | 1.0231445 |

Every pilot required at least1% gain at50 before proceeding. V27's shorter feature
connections produce larger corrections, while its reviewed outputs still lack
convincing added eyes, nose, mouth and outline structure. Its stopped50 saved
arithmetic has no original17-group pixel/SSIM/identity violation. The1% early,
10% final and preservation requirements remain intact. No final800 test ran.

The original DGP itself was frozen during V25–V27; only added heads learned. Source
inspection confirms a MobileNet/FPN reconstruction model with four pyramid heads,
two fusion/smoothing blocks and a final residual projection. Those original
reconstruction weights did not adapt in the three attempts. This describes our
retained implementation, not evidence that a published GAN-based DGP was trained
or that pretrained weights are exclusively our work. Preserve truthful provenance.

One concrete review direction is a **separate candidate copy of our original DGP
reconstruction decoder**, keeping the original checkpoint, feature encoder and
stored normalization statistics fixed at first. Review `head1`–`head4`, `smooth`,
`smooth2` and `final` as the explicitly learned contribution. Initial output must
match the retained baseline. A VM-only differentiation proof would be required:
the inference wrapper must not silently suppress the candidate graph's gradients.
This would change the learned path directly, with the existing visible-appearance
controls guarding drift. It is a design direction, not a ready pilot or a quality
claim; no weights, training code or protocol have been modified here.

A second direction is a **larger reconstruction decoder from the same frozen
features and observed crop**, preserving the original DGP as baseline. This needs
a material reconstruction design beyond adding another small readout. It still
risks repeating the current frozen-representation limitation; present failures do
not prove that limitation is the sole cause. Inputs/clean targets/person labels
must remain correctly separated. Pretrained restoration models remain comparison
baselines under the user's restated goal; no pretrained substitute is proposed.

The original-decoder review direction is now selected before another model modification.
Any later pilot needs a distinct justified protocol, exact initial parity and
gradient checks, finite updates/epochs, timing and stop rules, retained failed
gates and independent returned-result audit. All actual training stays on the
existing L4 with verified transfer files and manual pasteable commands. Native
development review, separate paired synthetic evidence, all visible facial
features and all seven automatic/assisted covering families remain required.
Reserved-final identities remain unviewed. The goal remains active/incomplete.

This discussion implements the workspace [AGENTS.md](<C:/xampp/htdocs/YEAR 4/Testing/AGENTS.md>)
circuit breaker: “If an issue remains unresolved after 3 consecutive attempts,
STOP modifying code. Name the invalid assumption and ask one diagnostic question.”
The stopped code is the model/training path; evidence audits and the explicitly
authorized storage cleanup continue within scope.

[Audited V27 results](<C:/xampp/htdocs/YEAR 4/Testing/CCTV_DGP_FEATURE_SKIPS_V27_RESULTS.md>) ·
[Saved comparison arithmetic](<C:/xampp/htdocs/YEAR 4/Testing/outputs/cctv_dgp_feature_skips_v27_saved_output_review/stopped_capacity_arithmetic.json>)
