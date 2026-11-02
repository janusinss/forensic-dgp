# Retained completion comparison viewer

25 October 2026. Standalone local review of existing saved outputs. This page does not generate an image or qualify a model.

Run from FACE_MASK_COMPLETION:

```powershell
python -B completion_review_web_v1.py
```

The standard-library-only server binds 127.0.0.1 on an available port and prints its actual URL. It records that URL in scratch/completion-review-runtime-v1.json. No environment installation, model loading, training or cloud service is required for this viewer. To check the retained input/output fingerprints without starting a server:

```powershell
python -B completion_review_web_v1.py --verify-only
```

Retained asset root defaults to C:\xampp\htdocs\YEAR 4\Testing. Use --asset-root only when the exact audited retained evidence has been placed elsewhere. The viewer refuses missing or changed pinned evidence. It does not require copying that evidence into this worktree.

The page provides covering/condition/case selectors, 32 individual comparisons, exact PNG downloads and eight original comparison sheets. Each case shows the prepared input, binary reviewed removal mask, CodeFormer completion baseline and converted MAT comparison. The existing orange overlays remain in the historical full sheets. The page displays saved development observations; it provides no new human acceptance or quality result.

All images are served byte-for-byte from an allowlist after SHA256 verification. There is no arbitrary file browser and no generation endpoint. Both automatic and assisted covering-family qualification remain false. Previously exposed photographs and synthetic degradations are not native CCTV evidence; hidden regions have no aligned clean reference.

Functional verification: seven focused tests pass for catalogue shape, exact downloaded bytes, loopback host restriction, changed fingerprints and path/unknown-resource refusal. Browser verification is performed separately by the root agent. The source-author receipt is outputs/completion_review_web_v1/source_author_verification_v1.json.

## Optional independently audited MI-GAN comparison

The default page remains the original baseline comparison. A fifth MI-GAN FFHQ256 experimental panel can be explicitly enabled only after the independent saved-output audit exists and its exact SHA256 is supplied:

```powershell
python -B completion_review_web_v1.py --migan-audit-sha EXACT_INDEPENDENT_AUDIT_SHA256
```

The only candidate namespace is this worktree's outputs/completion_migan256_comparison_v1. Its independent_output_audit.json must be complete, bind the original fixed32-case/mask protocol, pin results.json and all32 delivered PNGs, confirm outside-mask preservation and the same empty-mask controls, record zero training calls and keep model_qualified=false. A mismatching or incomplete audit refuses startup. The web UI cannot choose a filesystem path or execute generation. The viewer never imports MI-GAN or its runner.

Candidate-enabled /health reports mode=audited_saved_comparison_with_migan with generation_enabled=false. Baseline-only mode remains retained_outputs_only. The candidate adds32 fixed PNGs to the136 baseline/sheet assets. All source images are served unchanged; model usefulness still requires output review and acceptance.

Eleven focused tests pass, including four synthetic candidate-admission checks. The synthetic fixtures test the viewer boundary only; they are not model results. Actual candidate admission and root browser verification remain separate. Updated source-author receipt: outputs/completion_review_web_v1/source_author_verification_r2.json.
