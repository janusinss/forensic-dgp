# Role-preserving CodeFormer V1 return — 18 October 2026

**Technical initializer stop before training: zero new updates, zero optimizer
constructions, zero gradients/backwards and no new model checkpoint.**

The 26,395,397-byte archive has SHA-256
`ca152f53a3f733a4cd0ce859ec7d6b4248b47cf5def7c13b7bc52e6dc8697dad`.
Protocol SHA-256 is
`67d71dbb38de308b3f0b9299bc9f34844c51c8445149ce35cdac8e98004f1afb`.
All 460 regular files, 40 frozen static sources, manifest, physical gzip/USTAR
boundaries and four companions passed readback. Root independently hashed the
whole archive and inspected the imported failure. Export completion means the
failure evidence was packaged; it does not mean a trained model exists.

The worker stopped after 54.985 seconds with:
`Actual candidate/full-coarse initialization native/RAW/logits/features/tokens differ`.
Its 100 fresh HQ reference passes produced 200 arrays that independently match
the exact retained arrays and canonical target hashes. They are target-cache
evidence, not model training. One behavior-teacher, one parent and one candidate
initializer call were reserved. There were no training inputs, DEV snapshots,
gradients, Adam construction or optimizer steps. Model quality is unmeasured.

## Demonstrated implementation defect

The initializer check compared a frozen parent in evaluation mode under
`no_grad` with a trainable candidate in training mode with gradients enabled.
Equal weights and a restored random state do not establish equal execution
conditions. That comparison was incorrectly required to be bit-exact. The
captures were written only after the assertion, so the return does not reveal
which tensor or operation first differed. Random dropout is not an explanation:
this CodeFormer architecture sets its dropout probability to zero. A particular
CUDA kernel or attention fast path is not established by this evidence.

This is a software-verification failure before learning, not evidence that the
new restoration objective failed or that more epochs would solve it. The
source-only CPU checks did not reproduce this L4 execution difference.

## Repair and retained requirements

A separate R1 source/package namespace preserves V1 intact. It checks candidate
versus parent using matching evaluation/no-grad conditions, retaining candidate
trainability. A distinct pair of training-mode, gradient-enabled captures checks
the full unscoped objective against the scoped gradient route. Exact array and
loss equality remains mandatory. Both compared arrays, actual contexts, RNG
evidence and per-array differences are written before any parity rejection.
No tolerance or quality gate is relaxed.

The additional two candidate calls change the prospective maximum to 1,399
neural forwards and 1,854 reservations. The restoration recipe remains capped
at 50 new Adam updates and 200 microbatches, with the same model initialization,
trainable partition, objective weights, rates, data and quality requirements.
The repair has not yet passed actual neural execution. Use only a newly sealed
and independently reviewed R1 guide when it becomes available; do not rerun V1.

Original CodeFormer, original DGP, all checkpoints, caches, splits, provenance,
terms and gate failures remain preserved. Native CCTV remains unpaired and
source-separated, and FINAL identities remain reserved. There is no ethnicity,
Zamboanga performance or exact hidden-identity claim. Useful restoration still
requires reviewed outputs, user acceptance, independent FINAL and full existing
app Auto/override/PNG/Playwright checks. Separate seven-family covering detection,
mask review and plausible completion stay deferred until restoration acceptance.
The full goal remains incomplete; storage offload was already completed.

## Evidence

- [Strict import/transport](outputs/codeformer_role_preserving_training_v1_return_review/transport_v1.json)
- [Independent source diagnosis](outputs/codeformer_role_preserving_training_v1_return_review/cause_v1/audit.json)
- [All 100 HQ references](outputs/codeformer_role_preserving_training_v1_return_review/hq_v1/audit.json)
- [Root failure readback](outputs/codeformer_role_preserving_training_v1_return_review/root_failure_readback_v1.json)
