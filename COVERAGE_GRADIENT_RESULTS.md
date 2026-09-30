# Mixed-gradient audit results — 30 September 2026

Returned VM audit verified. No optimizer updates. All three checkpoint states
reported unchanged after six fixed batches. Executed script/helper hashes match
local sources; protocol hash and every batch index match the predeclared first
six epoch1 extended batches. Maximum full-gradient decomposition relative error
is1.209e-6, below the declared1e-4 numerical tolerance.

| Measurement (six batches) | Initial | Control-final | Extended-final |
| --- | ---: | ---: | ---: |
| Real/replay cosine median | -0.01050 | -0.14883 | -0.11322 |
| Opposing batches | 3/6 | 5/6 | 5/6 |
| Real-existing-covered norm median | 17.017 | 7.619 | 7.634 |
| Replay-supervised norm median | 0.577 | 2.476 | 1.842 |
| Teacher norm median | 3.04e-8 | 0.757 | 0.718 |
| Combined norm median | 18.927 | 15.383 | 12.646 |

The teacher term is nearly zero at initialization, as expected. Real-covered
terms are typically larger than individual replay/teacher terms, so this does
not support simply claiming that the teacher dominates and should be removed.
Medians of component norms cannot be added to reconstruct the combined norm.
Added-covered contributions occur in only a subset of these six batches; this
is not a full-trajectory audit. It includes two opaque examples, the clear added
control and hand-over-mask, but not the added profile respirator.

## Interpretation and next hypothesis

Opposing directions are present at the two failed final states. This is evidence
of local gradient interaction, not proof of the cause of poor validation. Gradient
norms and clipping factors do not equal AdamW update magnitudes.

A bounded gradient-projection comparison is now a distinct hypothesis supported
by this diagnostic. PCGrad projects conflicting task gradients; A-GEM constrains
updates using replay gradients. Primary references:
[PCGrad](https://arxiv.org/abs/2001.06782) and
[A-GEM](https://arxiv.org/abs/1812.00420).
Neither paper establishes that the method will improve this face detector.

Next implementation should compare ordinary mixed gradients with one fixed
replay-preserving projection rule using the exact same extended dataset, initial
state, replay pixels, batches and210-update budget. Keep losses, LR, clipping,
thresholds and real/synthetic quality gates fixed. First unit-test projection
geometry and empty/zero reference cases. Record both raw-gradient and actual
AdamW-step alignment: momentum/preconditioning/weight decay mean raw-gradient
projection alone cannot guarantee retention. No validation-driven projection
weight sweep; no claim of completion improvement before full evaluation.

This is a proposed comparison, not a ready VM training recipe. No new fitting
occurred and no existing checkpoint is eligible for promotion.

## Evidence

Archive `outputs/coverage-gradient-results.tar.gz`, SHA256
`fd1e12c5cc85cea5c6917c08b88b1baff4b691ba9025fc766b689e5bb33f8dbb`.
Extracted report/scripts: `outputs/downloaded_coverage_gradients/`.
`verification.json` records independently checked script and batch identity plus
summary statistics. Raw parameter-gradient vectors were not exported; the
decomposition errors and state invariants were checked on the VM by the verified
script, not recomputed locally.

Local root: `C:\xampp\htdocs\YEAR 4\Testing\`.
VM root: `~/forensic-dgp/coverage_vm_bundle/`.
