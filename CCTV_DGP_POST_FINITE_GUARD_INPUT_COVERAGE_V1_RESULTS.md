# Input coverage: severity gap and no reliable Auto clear-image bypass

The bounded input-only review completes on the 100 exposed TRAIN cases and
all 24 frozen native ChokePoint development crops. It reads no clean target
pixels, reserved final images or model outputs, performs no model calls or
training, and changes no thresholds or labels. It supplies a concrete next
training-design finding, not a successful restorer or proof of causation.

The four synthetic degraded profiles use 24, 32 or 48-pixel sampling grids.
Their expected geometric eye separation is 5.38–21.40 pixels across these
sampled references. Official native annotations show 23.02–47.04 pixels between
the eyes in the 24 development crops. Thus the sampled synthetic degraded
geometry does not overlap this native eye-spacing range. File dimensions alone
do not establish useful information: blur, lighting, noise, compression and
framing still matter. Synthetic spacing is projected from existing reference
landmarks onto the sampling grid; native spacing uses published capture-pixel
eye coordinates. Neither measurement proves resolved detail or recognition.

| Acquisition source | Blur/compound grid eye spacing | Low-light grid eye spacing | Motion grid eye spacing | Existing Auto suggestions |
|---|---:|---:|---:|---|
| asian_faces, sampled TRAIN | 6.67–10.70 px | 8.63–14.27 px | 12.95–21.40 px | 10/10 clear; 40/40 degraded |
| FFHQ-thumbnail namespace, sampled TRAIN | 5.38–6.30 px | 7.18–8.40 px | 10.77–12.60 px | 0/10 clear; 40/40 degraded |
| ChokePoint/P1E_S1_C1, native development | 23.02–47.04 px, native annotations | Unpaired native capture | Unpaired native capture | 11/24 native |

Source labels describe acquisition and existing namespaces, not ethnicity.
FFHQ counterpart/HQ provenance stays in the original training protocol; the
namespace does not convert all current targets into thumbnails. The two
TRAIN cohorts retain separate source/profile summaries in the result file.
The native source remains separate and unpaired, with no inferred capture
country, aligned clean reference, PSNR/SSIM, identity accuracy or Zamboanga claim.

The existing observed-support Auto rule is blur variance below 24 or noise
sigma at least 8. It suggests restoration for all 80 synthetic degraded cases,
but also all 10 asian_faces clear cases. Their blur variances range 2.50–21.68;
the 10 FFHQ clear cases range 68.49–219.68 and receive no suggestion. Therefore
the current rule cannot be treated as a perfect selector of the synthetic
clear category or used to guarantee unchanged clear outputs in a training
adapter. A synthetic clear label means no added degradation here; it does not
guarantee a sharp native source. No source-specific routing or ethnic inference
is introduced. The rule remains a developmental restoration suggestion.

All 24 native crops retain their previously frozen usable labels. Eleven receive
the restoration suggestion; the other 13 are still usable development inputs.
The suggestion is not a facial-sufficiency decision. Usable inputs cannot be
excluded because the trained model remains soft. Reserved identity roles,
provenance, terms, padding and crop geometry remain unchanged.

The new plan was frozen before the new measurements. The current pure
`quality_signals` function is executed by its exact AST with NumPy/OpenCV only,
so model-loader imports and neural execution are unnecessary. The older native
plan bound its wrapper but omitted the transitive face_workflow.py source.
An attempted historical-hash lookup failed before any output plan or model call;
the exact source/failure is retained. The revised preparation verifies that
wrapper and prospectively binds the current pure function. It does not invent
a historical transitive hash or claim historical filter-source identity.

The saved arithmetic checker verifies all 124 record IDs, source bindings,
sampling-grid/eye geometry, 21 source/cohort/profile summaries and the unchanged
24/8 suggestion arithmetic. It does not independently reproduce the quality
algorithm or prove its validity. Measurement takes 1.110 seconds. There are
zero model forwards, gradients, parameter updates, target/final pixel decodes,
app changes or model-qualification claims.

The selected next design review should broaden resolution/framing/degradation
coverage using audited genuine HQ TRAIN references, while retaining all legacy
profiles, lower-resolution replay anchors, old failures and original gates.
Candidate additional grid sizes such as 64/96/128/192 are a design hypothesis
requiring a frozen paired protocol; they are not acquired CCTV ground truth or
approved training commands. Input-conditioned spatial learning needs a proper
ablation rather than a hard bypass based on the current Auto rule. The full
3,905-case corpus has not been profiled here; these findings describe this
exposed 100-case diagnostic cohort only.

All actual training remains manual on the existing L4 with finite epochs,
updates, timings and stop rules. The current DGP is preserved as the starting
checkpoint. Native usefulness, independent final review, seven automatic/assisted
covering families and the DGP-led app/inline Playwright verification remain
outstanding. The complete goal remains active/incomplete.

Evidence:

- [Frozen input-review plan](<C:/xampp/htdocs/YEAR 4/Testing/outputs/cctv_dgp_post_finite_guard_input_coverage_v1/plan.json>)
- [All 124 input records and separate summaries](<C:/xampp/htdocs/YEAR 4/Testing/outputs/cctv_dgp_post_finite_guard_input_coverage_v1/results.json>)
- [Saved arithmetic verification and its limits](<C:/xampp/htdocs/YEAR 4/Testing/outputs/cctv_dgp_post_finite_guard_input_coverage_v1/saved_arithmetic_audit.json>)
- [Retained preparation failure](<C:/xampp/htdocs/YEAR 4/Testing/outputs/cctv_dgp_post_finite_guard_input_coverage_v1_preparation_failure/failure.json>)
- [Next training design review](<C:/xampp/htdocs/YEAR 4/Testing/CCTV_DGP_POST_FINITE_GUARD_TRAINING_REVIEW.md>)
