# Role-preserving CodeFormer R1 return — 18 October 2026

**Technical initializer stop: zero new training updates, zero Adam constructions,
zero gradients/backwards and no new checkpoint.** Export completion packaged
the failure evidence; it did not create a trained model or measure improvement.

The actual 34,174,817-byte archive has SHA-256
`075d0fcc9b6dbd35e1ba65050c7837c00f1bd64afb44164a2a45ae991f1ab2e1`.
Protocol: `4edacae4f1e61f06b21fb1308423398d5b9f842def160c02f74f8edc0975fc40`.
All 467 regular files, 32 frozen static files, manifest, gzip/USTAR boundaries
and four companions passed inert readback. The 100 fresh HQ-reference cache
sets contain 200 arrays independently matched to retained parent files and
canonical target payload hashes. Those arrays are cache evidence, not training.

The worker stopped after 59.672 seconds at
`Matched initializer inference parity failed; actual arrays and context retained`.
Its 102 forward reservations cover 100 HQ calls, one frozen parent and one
candidate initializer call. No training-mode witness, gradient, Adam step,
candidate checkpoint or 406-case development quality bank was reached.

## What actually differed

Independent CPU readback of the saved candidate/parent arrays found:

- Native 512 output: exactly equal, including NPY file bytes.
- Delivered 256 RAW output: exactly equal, including NPY file bytes.
- Encoder latent and all 256 selected dictionary indices: exactly equal.
- Intermediate logits: 223,123 of 262,144 values differ; maximum absolute
  difference **0.00000762939453125**, mean absolute difference 0.0000005582493471933958.

All 256 saved indices match both logit argmax arrays. Every parent top-two
logit margin exceeds twice the corresponding maximum observed logit error;
the smallest parent margin is 0.0001087188720703125. These observations show
unchanged hard dictionary selections and reconstructed pixels for this one
saved initializer witness. They do not establish behavior on another input.

All 371 module modes match in evaluation mode, with no-grad FP32 calls and
matching saved before/after RNG hashes. The candidate retains 350 trainable
parameter flags; the frozen parent has zero. Factory fingerprints record the
same 515 tensor values. This return disproves the assumption that those
conditions ensure bit-exact logits across separately allocated objects.

The exact first differing operation and numerical mechanism remain unknown.
No predictor-layer intermediates, same-object repeatability or CUDA kernel
trace was exported. Nondeterministic algorithms were permitted. Neither that
setting nor the different parameter flags proves the cause. The pinned
CodeFormer/PyTorch source review does not support claiming the native
MultiheadAttention fast path caused this difference.

## Prospective numerical test policy

The failed V1 and R1 packages/returns remain immutable. A separate proposed R2
changes the **initializer numerical gate**, not the restoration-quality gates:
permit only float32 intermediate logits within the declared PyTorch 2.9
`assert_close` defaults, absolute tolerance 0.00001 and relative tolerance
0.0000013. In addition, every hard argmax and saved token must remain identical,
and every parent top-two margin must exceed twice its observed maximum logit
error. All native/RAW/latent/token arrays, model state, contexts and RNG evidence
retain exact checks. This is an explicit changed numerical prerequisite;
it is not a measured repair of an identified CUDA operation.

PyTorch documents float32 default comparison tolerances and the limitations
of floating-point reproducibility. Those general documents support declaring
the test policy; they do not identify the cause of this return or qualify
the model. [PyTorch 2.9 assert_close](https://docs.pytorch.org/docs/2.9/testing.html#torch.testing.assert_close),
[PyTorch numerical accuracy](https://docs.pytorch.org/docs/2.9/notes/numerical_accuracy.html).

R2 remains an actual finite restoration-training pilot with unchanged parent,
350/165 trainable/frozen partition, data, objective, rates and quality requirements:
maximum 50 new Adam updates, 200 microbatches, 1,399 neural forwards and 1,854
reservations. It introduces no additional neural diagnostic run or automatic
retry. A newly sealed, independently reviewed manual L4/tmux package is required.
The changed numerical gate and eventual trained outputs remain unverified
until that manual run returns. Do not rerun either failed folder unchanged.

Useful restoration still requires user-visible improvement, appearance
preservation, independent FINAL review and the existing app Auto/override/PNG
flow. Untouched CodeFormer and original DGP/checkpoints/caches/splits/provenance,
source terms and failures remain preserved. Native CCTV stays unpaired and
source-separated; no ethnicity, Zamboanga performance or exact hidden identity
is established. Separate covering/mask review/seven-family completion remains
deferred until restoration acceptance. Storage offload is complete; the full
research goal remains incomplete.

## Evidence

- [Strict transport](outputs/codeformer_role_preserving_training_v1_r1_return_review/transport_v1.json)
- [Independent saved-array cause audit](outputs/codeformer_role_preserving_training_v1_r1_return_review/cause_v1/audit.json)
- [All 100 HQ references](outputs/codeformer_role_preserving_training_v1_r1_return_review/hq_v1/audit.json)
- [Primary source and repair-scope review](outputs/codeformer_role_preserving_training_v1_r1_return_review/repair_scope_source_review_v1.json)
