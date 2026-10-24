# Application model card — IMPLEMENTATION

This branch packages frozen models for application development, deployment, and real-world testing. Research and training are maintained separately.

**U96 is the default.** Optional experimental u64 is selectable in the interface. Both use the trained R2 CodeFormer base and U96 residual adapter; u64 additionally loads the terminal-update-64 post-fusion heads. The model names do not denote image dimensions.

## Runtime and assets

Models load in disposable CPU workers only when restoration is required. Each request permits at most one inference call. Processing uses FP32 at 512×512, fidelity 1 and AdaIN, followed by 256×256 delivery. PNG uses floor quantization without extra sharpening or brightness processing. Off returns the resized original without inference.

The installer downloads these assets into `models/implementation` from the [model release](https://github.com/janusinss/forensic-dgp/releases/tag/implementation-models-v1). All three files must pass complete size and SHA-256 verification.

| Asset | Bytes | SHA-256 |
| --- | ---: | --- |
| `codeformer_r2.pth` | 376609465 | `d4abda08831681007fd0c351904ec5dd416a3a5ff6e16d06b25cd8b70b0fc896` |
| `codeformer_u96_adapter.pth` | 212827 | `958e8338d4bc1a0204c314c3af239ab0ab3d783c67e3e532d9fe23c58f8a73d0` |
| `terminal_u0064_heads.npz` | 685818 | `77c0e1f236eb70d37f3f81059b75580e92d606bb2f8b5dd5eeb00da0305dca17` |

## Intended use and limits

Use one already cropped frontal or mildly turned face with readable visible structure. Auto provides an input-quality suggestion, not a usefulness or identity score. U96 and u64 remain experimental and can change facial appearance; application operation does not establish restoration accuracy or recovered identity.

Preparation can be bypassed or reviewed manually with Protect/Apply masks. Exact protected and outside-Apply pixels concern preprocessing only. The generated result has no corresponding facial-pixel guarantee. Neutral padding remains preserved. Completion, covering removal, automatic blur matching, and recovery of hidden identity are outside this app.

Review the original, selected input, and result together. Prefer clearer or less-covered input when information is insufficient. Optional evidence ZIPs contain the original and available masks, preparation records, RAW output, and result. Request records remain locally in `scratch/implementation_requests`; downloaded evidence also contains image data.

## Attribution and license

CodeFormer is an S-Lab architecture used here with trained adaptations. Retain [the S-Lab license](third_party/codeformer/LICENSE) and [notices](third_party/codeformer/NOTICE.md). Its non-commercial terms apply to use and distribution. No training dataset is included.
