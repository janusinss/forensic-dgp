# Completion fine-tuning V1: audited pre-training failure

25 October 2026. EXECUTED, FAILED BEFORE TRAINING; original model retained.

The return archive is intact: 217,636,241 bytes, SHA256
7d51d190d7cba45f6c0f2f99e90e32d06fb5ef43444bf55b07e973ba0ba91bd6.
All six regular members, five manifest payload bindings and the original
0dfe963bd29856ea6201d6397aa25d651ca0291661fa3ec26539c9c770e31b8f
protocol match. Sidecars and original returned bytes remain retained.

The worker stopped at `Fresh frozen app pipeline baseline parity failed`.
It completed 40 teacher encoder calls, 40 teacher quantizer calls and the first
two parity calls: 82 typed slots. It performed zero backward calls and zero
optimizer updates. Supervisor exit 1, actual child exit 1, no timeout or
interruption; elapsed 20.619 seconds, failure receipt 14.760 seconds. There are
no evaluation images, paired output metrics or learned improvement to review.
Export completion describes successful evidence packaging only.

Safe CPU-only deserialization independently verifies every one of 332 trainable
tensors exactly equals the original (58,459,840 coordinates). Both Adam states
are empty; ordered names, zero counters and retained RNG/schedule evidence are
consistent. Reconstructed complete 497-tensor logical state is unchanged:
586a8854ed7b5e3f0e202c258addb7a87de63cd3068d153741b065379ff6668f.
The saved terminal state is a recovery/evidence object, not a new trained model.
No model forward, backward or optimizer operation ran in the local return audit.

## Demonstrated processing mismatch and limits

Pure tensor inspection found identical numeric 512 inputs with different strides.
For the first lower-face case the old pilot used NCHW strides
[786432,262144,512,1]; the admitted app used active-batch indexing and
channels-last strides [786432,1,1536,3]. Maximum input-value difference is zero.
The old pilot also omitted the app's indexed output assignment. This is a
concrete implementation mismatch. Its exact effect on the failed CUDA output
is not proven:V1 asserted before saving its numeric error or intermediate arrays.
Do not describe this failure as harmless scalar rounding or change tolerance.

R1 copies the admitted app's active-index preprocessing and composition exactly.
Ten pure CPU tensor tests cover four geometries with HWC-strided and NCHW inputs,
exact input values/layout, composition and empty bypass. Independent AST review
confirms unchanged training contract, LR, losses, split/case bytes, selection,
resource bounds and terminal-state retention. These are processing/source tests;
actual GPU model parity remains an unchanged `1e-6` requirement before any update.

R1 saves all four paired parity cases before aggregate assertions:actual 512
inputs/layout, raw 512 model outputs, selected codes, delivered 256 outputs and
JSON numeric errors. Captures add no model calls and are removed before training.
If parity still fails, retain the new return rather than weaken the gate.

## Next manual action

[Use the isolated R1 installer and five-step guide](COMPLETION_CODEFORMER_FINETUNE_V1_R1_VM.md).
It verifies all 329 existing packet files, the four retained failure records and
original checkpoint before creating a fresh sibling directory. V1 is untouched;
its failed outputs are not copied into R1. Original restoration, CodeFormer
baseline, RESTORATION and IMPLEMENTATION remain unchanged. R1 is prepared/unrun.

Evidence in `outputs/completion_codeformer_finetune_v1_return/`:

- transport_audit.json:47c6aee449b47b64ad52cae6945e7b9f2ecd93f79297565787eb3725c32a0821
- independent_return_audit.json:2284e514fcc3b1b2bcab68dc5ea14aa9ed0222ce5cb0720d9571eece96e3b0a7
- static_parity_diagnosis.json:0bc4479f9b4f11aa2f793cd9022c5e7421208bf6b43f8bf03562862934edda33
- cpu_preprocess_layout_diagnostic.json:exact-value/different-stride tensor evidence

Independent source admission:
`outputs/completion_codeformer_finetune_v1_r1_source_review/independent_source_admission.json`,
SHA256 a499fbfec894a0be396ed142b43198e92b2ce00c28ae6b3bc1418d7784745af2.

Model usefulness, all covering families, automatic/assisted separation,
independent final review and deployment remain incomplete. No promotion follows
from this return audit or a prepared repair.
