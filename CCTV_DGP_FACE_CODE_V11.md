# V11: our DGP-conditioned face-code prototype

Current status, 4 October 2026: the V12 r2 fitting pilot trained this conditioner
for 300 updates but did not demonstrate a useful upgrade. Its independent audit
and all ten grid reviews are complete; the VM is stopped. See
`CCTV_DGP_FACE_CODE_FIT_V12_RESULTS.md`. Rendering/code-path diagnosis is next.

Historical V11 status: implemented and untrained. Four interface/VM-only boundary
tests pass. The finite local inference check, independent no-forward audit and both
original-cell galleries are complete. Zero-conditioner output matches the
frozen baseline exactly on both artificial cases. State hashes and all forward
counts are unchanged. The check took 161.12 seconds; its audit took 5.39 seconds.
The clean-code oracle renders coherent faces but changes some eyes, lips and
expression detail. It justifies a bounded fitting diagnostic, not adoption.
No production checkpoint or application default changed.

Windows root: `C:\xampp\htdocs\YEAR 4\Testing\`; VM root: `~/forensic-dgp/`.
Files below are local preparation. VM counterparts are intended locations after
explicit transfer, not claims of transfer or VM execution.

## Why this changes the approach

V9 pixel training recovered brightness and reduced error while losing recognizer
similarity and leaving soft faces. V10's direct DGP → CodeFormer chain generated
wrong glasses, facial hair and expressions. A sharper output alone is insufficient.
Their negative results and old selection safeguards remain intact.

Our new module uses the original RGB crop as the prior's input. A frozen, audited
V2 DGP signal supplies three additional conditioning channels. Our convolutional
conditioner maps the six channels to a residual on the prior's 16×16 code-feature
grid. Its final projection starts at zero, so baseline parity can be tested before
training. Only the conditioner's **455,072 parameters** are intended to learn.

CodeFormer encoder, transformer, codebook, renderer and fidelity connections stay
frozen and declared pretrained components. The residual changes code prediction;
original input features supply AdaIN and fidelity connections. The published
prior and original project DGP are not renamed or claimed trained from scratch.
This is our proposed conditioning contribution, whose usefulness is unproved.

The existing restoration wrapper is inference-only and uses hard top-1 code
lookup. The new guarded `training_codes` interface instead exposes logits and
conditioned features for code/feature supervision. Both training enablement and
graph entry reject machines other than the user's existing L4 VM. Local inference
returns no gradient graph. Rendering remains inference-only.

The author training code uses a separate clean-image VQGAN teacher to generate
code labels. That teacher is acquired from the official release rather than
assuming the degraded-input encoder supplies clean labels.
[Official training implementation](https://github.com/sczhou/CodeFormer/blob/master/basicsr/models/codeformer_model.py).

## Teacher and finite inference checks

The official `vqgan_code1024.pth` is 255,078,528 bytes, acquired from release
v0.1.0. Observed SHA256:
`4d1c6741b3cffcbdc2cd1a12b2c3c2442282e042d5de66909cb643d4fa31b20f`.
GitHub supplies no release digest; this is an acquisition fingerprint, not an
author-published checksum. S-Lab License 1.0 and NOTICE remain with the source.

Teacher and restoration codebook/decoder tensors are not byte-identical.
The initial exact-match assumption failed and is preserved in a diagnostic.
All component tensors agree at rtol/atol 1e-5; all 1,024 code vectors match the
same nearest prior index, with minimum cosine 0.9999993. Both weights are retained
unchanged. The oracle check separately measures their rendered difference.

The current check uses ten existing **training** references, five per source,
selected by ascending SHA256 of `face-code-v11:reference_id`. No validation,
native development or reserved inputs are used. Asian targets remain upscaled
source photographs rather than high-quality CCTV truth.

The clean-code oracle renders teacher labels through the frozen prior with w0,
without clean target skip features or target AdaIN. Because it sees clean
references, it measures representation capacity and is **not restoration**.
Compare it against clean references, teacher VQ reconstruction and ordinary
CodeFormer w1 on the same clean input. Inspect layout and expression before detail.

Two artificial RGB cases separately compare the zero-conditioner output with
the original frozen w1 baseline. Required maximum float deviation is 2e-6 and
floor-quantized PNGs must be identical. All frozen states and actual component
forward counts must remain stable. Terminal neural proof is written before
preview rendering, preserving it even if later postprocessing fails.

Budget: CPU four threads, 180 seconds after loading, zero backward calls and
optimizer updates. The original prepared check was superseded before any model
execution after static inspection found an incorrect recognizer-call interface.
Its source and preparation are preserved; r2 uses the actual observed-mask/fixed-grid
embedding interface. Do not restart or overwrite either frozen plan.

## Files and next execution

| Purpose | Windows under the project root | Intended Linux counterpart |
| --- | --- | --- |
| Our module | `dgp_face_code_conditioner_v11.py` | `~/forensic-dgp/dgp_face_code_conditioner_v11.py` |
| Boundary tests | `tests\test_dgp_face_code_conditioner_v11.py` | `~/forensic-dgp/tests/test_dgp_face_code_conditioner_v11.py` |
| Inference runner | `scripts\check_dgp_face_code_prior_v11_r2.py` | `~/forensic-dgp/scripts/check_dgp_face_code_prior_v11_r2.py` |
| No-forward auditor | `scripts\audit_dgp_face_code_prior_v11.py` | `~/forensic-dgp/scripts/audit_dgp_face_code_prior_v11.py` |
| Frozen check/evidence | `outputs\dgp_face_code_prior_checks_v11_r2\` | `~/forensic-dgp/outputs/dgp_face_code_prior_checks_v11_r2/` |

Frozen r2 plan SHA256:
`865cef25d09029615a62d097d5051386bb1981cdd50c252a02bb0a3591761d92`.
These commands document the already prepared execution, not a request to repeat it:

```powershell
Set-Location -LiteralPath 'C:\xampp\htdocs\YEAR 4\Testing'
.\venv\Scripts\python.exe -X utf8 -u scripts/check_dgp_face_code_prior_v11_r2.py --prepare
.\venv\Scripts\python.exe -X utf8 -u scripts/check_dgp_face_code_prior_v11_r2.py --run
.\venv\Scripts\python.exe -X utf8 -u scripts/audit_dgp_face_code_prior_v11.py
```

Next: freeze a bounded **training-only** conditioner fitting diagnostic with a CUDA
gradient/VRAM preflight on the L4. Code-label fitting alone cannot select a useful
checkpoint. A broader pilot requires separate held-out validation, appearance
safeguards and native development review before local application integration.
The active Goal remains incomplete and the VM is stopped until that recipe is ready.

Pretraining and proxy-split limits: `CCTV_FACE_PRIOR_DATA_LIMITS.md` beneath
the Windows root, intended same filename below `~/forensic-dgp/` after document
sync. Our holdout is not proven unseen by the pretrained weights.
