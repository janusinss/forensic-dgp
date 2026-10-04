# Completion context comparison — 3 October 2026

A narrow conditioning margin reduces wool contamination in one difficult scarf
case, but this four-case comparison does not establish a better general default.
The application keeps its existing completion route. Automatic hair detection
and nearly hidden-face misses still depend on the pending VM detector experiment.

The user confirmed that a coherent rough face estimate is acceptable when the
covering is substantially removed and visible appearance remains coherent.
Perfect detail and recovery of actual hidden features are not required. This is
a product criterion, not acceptance of particular images or a change to old gates.

| Evidence or source | Windows local | Linux counterpart only after separate transfer |
| --- | --- | --- |
| Experiment | `C:\xampp\htdocs\YEAR 4\Testing\outputs\completion_context_review_v1\` | `~/forensic-dgp/outputs/completion_context_review_v1/` |
| Experimental adapter | `C:\xampp\htdocs\YEAR 4\Testing\completion_context_margin.py` | `~/forensic-dgp/completion_context_margin.py` |
| Runner and auditor | `C:\xampp\htdocs\YEAR 4\Testing\scripts\run_completion_context_review.py` / `scripts\audit_completion_context_review.py` | Same paths under `~/forensic-dgp/scripts/` |
| Preview | `C:\xampp\htdocs\YEAR 4\Testing\outputs\completion_context_review_v1\preview.png` | `~/forensic-dgp/outputs/completion_context_review_v1/preview.png` |
| Retained main app | `C:\xampp\htdocs\YEAR 4\Testing\app.py` and `face_workflow_palette.py` | Same paths under `~/forensic-dgp/` |

The fixed inputs are four previously inspected degraded assisted cases: hand over
eyes, obstructing hair, scarf/gloves and flower over mouth. Their existing reviewed
removal masks, crop geometry, pretrained weights and Auto restoration settings
stay fixed. The experimental adapter suppresses an extra 2- or 6-pixel strip
around the input mask at 256 scale, then discards generation outside the original
reviewed removal area. It does not expand the final output mask or change detector
labels. An enlarged conditioning area at or above 85% still rejects before inference.

Nine CPU requests completed in 86.34 seconds after loading: eight new alternatives
and one zero-radius control. The control exactly matches the cached assisted
scarf/glove PNG. There were nine completion and nine restoration forwards, zero
detector forwards and zero optimizer updates. Both model-state hashes stay equal
before/after; no training occurred locally or on the VM for this comparison.

Five counterexample tests cover exact reviewed output support, context scaling,
empty bypass, pre-inference rejection and invalid inputs/outputs. The independent
auditor checks all 37 recorded artifacts, recounts each conditioning mask and
reconstructs all nine PNGs from captured float stages. Auto visible restoration
does not overwrite generated pixels; grayscale projection follows uploaded input.

| Case | Result of the context comparison |
| --- | --- |
| Hand over eyes | Central hands remain replaced; estimated eyes stay stylized. No clear advantage over the existing assisted estimate. |
| Obstructing hair | Estimated eye/cheek remain coherent; exterior hairstyle stays. Automatic empty-mask failure remains unresolved. |
| Scarf/gloves | Six pixels produces a much cleaner lower face, but a thin line remains near the upper removal boundary. Exterior scarf/gloves remain outside the chosen facial area. |
| Flower over mouth | Similar removal and slightly changed estimated mouth/facial hair; no strong advantage over the assisted baseline. |

Visible MAE increases slightly for hand, hair and scarf variants and decreases
slightly for flower. Changes range from -0.0000504 to +0.0008326 in normalized RGB.
These known-visible diagnostics cannot score the person's hidden features. All
eight alternatives and four complete preview rows were inspected; the scarf source
and six-pixel result were also inspected individually. The boundary line's exact
cause remains unproved. No individual user/expert acceptance is claimed.

Frozen protocol SHA256:
`211945a7187680e12885c6709dea52564593db9a31fb4a69d3a91f152c86c7c3`.
Results SHA256:
`e69b41bab828cc99a4744e3fbe61fa217283650a2a81ddd057e9ab98c7683e17`.
Independent verification SHA256:
`2cd14db824b1d9302b169c5d3c11540f805227a7ed3173739768092846ed575a`.
The visual review is a separate `visual_review.json`; the original independent
receipt's pending-visual field is preserved as historical evidence.

The experiment is complete; do not rerun or edit it in place. Native inputs,
ordinary clear-glasses controls and the full practical gallery are not covered by
this four-case assisted comparison. There is no automatic case-specific router,
application switch or checkpoint promotion. Original 425-case failures remain.

Next: obtain the distinct gray-covering VM return using
`C:\xampp\htdocs\YEAR 4\Testing\GRAY_COVERING_RETURN_REVIEW.md` (intended
`~/forensic-dgp/GRAY_COVERING_RETURN_REVIEW.md` after document transfer), independently
audit it, then run the unchanged practical mask/face-output comparisons. A future
context-policy experiment must freeze broader review and boundary criteria before
inference; this case-specific improvement does not justify generator retraining.
