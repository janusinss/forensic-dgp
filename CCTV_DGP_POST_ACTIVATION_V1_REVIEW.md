# Post-activation decision: constrain preservation before another trajectory

The invalid assumption was that the corrected initial objective gradient could
be extrapolated to the retained update while its preservation barriers stayed
inactive. The completed measurement disproves that for the tested direction:
positive profile-level RGB slopes exhaust real, tiny regression allowances.
Fresh batch-five evidence confirms the finite loss increase. It does not prove
that every direction in the current decoder is infeasible.

[Audited measurement and limits](<C:/xampp/htdocs/YEAR 4/Testing/CCTV_DGP_DECODER_ACTIVATION_V1_RESULTS.md>).

The best-supported next design is **one finite feasibility experiment in the
current A reconstruction decoder**, using the saved underlying gradients.
It must select a different direction that addresses the measured preservation
conflicts before any new optimizer trajectory. The experiment is not prepared
or authorized for automatic execution by this review. Its solver, numerical
checks, exact transfer assets and timing/storage limits still need a separate
prospective contract and independent review.

1. Reuse the original A0 state and saved 70 gradients from the two existing
   scheduled TRAIN references. Solve at most one nonzero direction in a
   declared saved-gradient span, manually on the L4. Require simultaneous
   per-profile RGB/SSIM/recognizer preservation and progress in both references'
   degraded structure terms, with actual regression slack and nested
   photometric/mean-control clipping represented. A favorable weighted total
   alone is insufficient; no new gradient, optimizer update or epoch is needed
   to establish this bounded feasibility question.
2. Freeze the solver and at most three placements before inspecting new output.
   Keep the original-relative radius, normalization, reconstruction weights and
   scientific gates. Exclude zero, unchanged and vanishing proposals. An
   infeasible solve is a result to retain, not permission to loosen constraints,
   enlarge the radius or add more search attempts.
3. Evaluate every allowed placement on all 20 fixed references/100 cases.
   Independently reconstruct the exact RAW and floor-PNG metrics for all 17
   groups. Require at least 1% global and each-source structure improvement,
   original visible-appearance/preservation checks, brightness fraction at
   most 0.2 and exact exterior pixels. Only two references supply gradients;
   the other 18 provide essential finite checks rather than fitted derivatives.
4. Bind and inspect every planned visual sheet for useful eyes, brows, nose,
   mouth, jaw and visible appearance. Tiny loss changes or PNG-unchanged images
   do not establish useful detail. TRAIN and frozen native/paired DEV evidence,
   with original/pretrained comparators, may justify a finite continuation.
   Finish training selection and freeze the candidate and processing before
   independent final evaluation. Reserved final results must not select extra
   epochs; final review is required before app qualification or promotion.
5. Stop this declared experiment if the solve is infeasible or every allowed
   placement fails. Return to reconstruction/data design review; do not add a
   fourth scale or resume the failed Adam recipe. Infeasibility concerns this
   sampled span, not the whole architecture or all possible restoration models.

This differs from the earlier seven-loss PCGrad, aggregated B-direction and
conditioning-only searches: the current 15-tensor A decoder's underlying
profile losses and exact corrected batch-five path are now exposed. It must
not reproduce those failed methods under a new name. A solver that cannot
represent the required constraints must report that limitation before any
finite neural proposal is made.

The solver's success flag will not admit a candidate. For example, the official
SciPy 1.15.3 HiGHS documentation gives a default primal feasibility tolerance
of 1e-7; that cannot by itself certify this experiment's 1e-12 MSE allowance.
This is a comparison of documented numerical tolerances, not a claim about the
installed VM solver version.
[Official SciPy options](https://docs.scipy.org/doc/scipy-1.15.3/reference/optimize.linprog-highs-ds.html).
Freeze scaling, version, options and finite iteration/time limits before any
solve. Independently certify unscaled constraints and the actual rounded
float32 displacement with exact dyadic or outward-bounded arithmetic; reject
an uncertain certificate. A nonlinear radius must not silently become a
coefficient box: the documented second-order-cone form supports Euclidean
norm constraints, while a conservative box would define a different search.
[Official cone-program example](https://www.cvxpy.org/examples/basic/socp.html).
Failure to obtain a certified candidate is not a proof of whole-model
infeasibility. The sign-dependent clipped mean/brightness terms also need an
explicit derivation; they cannot simply be declared linear.

Route A remains a DeblurGAN-v2-compatible conditional restorer; cached external
GAN features are not injected into this A path. This is not a newly trained GAN
or qualified generative face prior. The goal's justified replacement comparison
remains required. The original baseline `646fbb11674e8882ea41e251d6e6e7d468c963979b48c93132def8292834e24b` stays permanent.
No local gradients, parameter proposals, VM connection or new app change was
performed for this review. Completion remains separate: preserve reviewed
visible pixels and report all seven automatic/assisted covering families.
