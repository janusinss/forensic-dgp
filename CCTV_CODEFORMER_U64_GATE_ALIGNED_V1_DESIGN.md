# Gate-aligned improvement of original trained u64

23 October 2026. **Source-only design. No runnable training worker, transfer
package or launch guide is ready.** This is the next implementation direction
supported by the [audited role-gradient return](CCTV_CODEFORMER_U64_ROLE_GRADIENTS_V1_RETURN_REVIEW.md).
It retains the original trained u64 rather than restarting from U96 or from the
rejected eight-update state. No app default or checkpoint is replaced.

## Concrete change and why

The old mixed objective has a dominant low-light gradient that conflicts with
photographic accuracy on other roles. Its same-role objective also does not
consistently descend the adjacent-error structure gate. Blindly increasing LR,
adding epochs, dropping low light or averaging normalized gradients does not
address both measured issues.

The saved gradients show a common direction exists **in the existing36 heads**
for the eight pooled gate measures. The proposed update therefore uses these
measures directly: MAE and adjacent-residual error for each of four roles. Pool
the16 admitted optimization photographs per role/measure at the same current
state. Localized blur uses central128; other roles use whole256. Normalize the
eight pooled vectors once, then find a minimum-norm convex combination on their
simplex. Its negative unit direction must independently have strictly negative
dot products against all eight original vectors with normalized margin>1e-6.
Zero, missing, nonfinite or uncertifiable directions stop without a fallback.

The inert NumPy solver exhaustively examines at most255 nonempty faces of an
8×8 Gram problem. It does not import a neural framework, generate gradients,
construct an optimizer or apply a parameter. Saved-array proof must remain
separate from any later CUDA computation or actual finite update. The exploratory
SLSQP calculation and exhaustive-face solver are independent formulations; they
are not two trained models.

This follows the multi-objective descent principle studied by
[Sener and Koltun, NeurIPS 2018](https://papers.nips.cc/paper_files/paper/2018/hash/432aca3a1e345e339f35a30c8f65edce-Abstract.html).
Their benchmarks do not establish useful face restoration, identity recovery or
our finite-step preservation. A nonzero common pooled direction does not imply
every individual photograph or unmeasured guard improves.

## Proposed finite test, before any longer training

1. Restore exact complete original-u64 state575; keep R2/U96 tensors539 frozen.
   Verify the full initial56-case bank and four deterministic repeats against
   historical RAW/PNG evidence. Preserve all original metric baselines/teachers.
2. For each of at most **four new gradient-descent updates**, recompute64
   optimization forwards and128 separate MAE/adjacent gradient queries. These
   are four passes over64 correlated rolecases, not four independent datasets.
3. Derive the certified common direction. Use only a positive global scale with
   coordinate step cap2.5e-5 and cumulative original-u64 delta cap1e-4. Validate
   actual FP32-rounded displacement and its eight dot products before committing.
   No Adam/coordinate preconditioner may alter the direction; no line search,
   router, dropped role or adaptive recipe is proposed.
4. Evaluate all56 monitor/legacy-guard cases after every update. Every role-mean
   MAE/adjacent endpoint must not increase versus both originalu64 and the prior
   accepted state in **RAW and PNG separately**. Preserve every original case
   limit and report each individual regression. At terminal4, clear and low-light
   mean MAE must be strictly better, with unchanged original1% blur structure and
   preservation requirements. Any quality stop rejects the pilot; no earlier
   stopped state is selected retrospectively.
5. Independently audit saved states, actual rounded updates, gradients, solver
   certificates, RAW/PNG and every planned sheet. Only useful visible improvement,
   preservation and agreed development/user review can justify progression to
   longer training. Broader inherited DEV failures remain binding throughout.

Maximum proposed operation budget: **540 full forwards,512 gradient queries,
four parameter applications, zero Adam or other optimizer objects**. Plain
gradient-descent parameter application is actual training and therefore belongs
only in a separately reviewed, manually launched L4 worker. All gradients, every
head state and the complete initial/closing575 states must be retained. No local
parameter application is permitted by this prototype.

Worker timing, external termination, disk/memory/output accounting, atomic
receipts, unknown-operation handling, failure export, source sealing and exact
manual transfer commands remain requirements for the future runnable packet.
An inert solver's wall cap does not implement that worker or its supervisor.
No historical pilot may run automatically. No new manual command is supplied
until the complete runner and actual sealed package pass independent review.

## What remains unproved

The saved initializer bank admits eight pooled negative derivatives, while12/128
individual gate derivatives increase for the exploratory direction. It contains
no monitor/guard gradients. Finite steps may fail through curvature, rounding,
clipping, representation limits or generalization. Downloaded PNG changes are
discrete and cannot be guaranteed from RAW derivatives. This is a justified
bounded test design, not a qualified restoration method or a promise of success.

U96 remains the experimental app incumbent; trainedu64 remains the retained
challenger. Reviewed Protect/Apply/bypass stays separate. Automatic matching and
later completion development remain deferred. Native/wild evidence stays
unpaired; FINAL remains unopened/execution-unfrozen with reviewers pending.
Automatic goal advancement did not trigger. Full goal ACTIVE/incomplete.
