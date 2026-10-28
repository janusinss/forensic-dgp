# Application model card — IMPLEMENTATION

This branch packages the frozen **Conditioned SFT V2 (`p2_u0224`)** model for application development, deployment, and real-world testing. Research and training are maintained separately.

**Conditioned SFT V2 (`p2_u0224`) is the sole production restoration model.** It builds on the frozen 575-tensor CodeFormer R2 + U96 adapter + terminal-update-64 spatial head backbone (`codeformer_r2.pth`, `codeformer_u96_adapter.pth`, `terminal_u0064_heads.npz`) and adds the 212-tensor V2 top-down dilated SFT pyramid, dark-noise refiner, and frequency-selective VQGAN spatial head (`v2_p2_u0224_heads.npz`, 787 frozen tensors total) with input-only conditioning and outer-ring-gated high-frequency detail synthesis.

## Runtime and assets

The model loads in disposable CPU workers only when restoration is required. Each request permits at most one inference call. Processing uses FP32 at 512×512, fidelity 1 and AdaIN, followed by 256×256 conditioned delivery. PNG uses floor quantization while preserving exact neutral padding outside observed support. Off returns the resized original without inference.

The installer verifies these four assets in `models/implementation` (`codeformer_r2.pth`, `codeformer_u96_adapter.pth`, and `terminal_u0064_heads.npz` from the [model release](https://github.com/janusinss/forensic-dgp/releases/tag/implementation-models-v1), plus `v2_p2_u0224_heads.npz` staged from `implementation/assets/`). All four files must pass complete size and SHA-256 verification.

| Asset | Bytes | SHA-256 |
| --- | ---: | --- |
| `codeformer_r2.pth` | 376609465 | `d4abda08831681007fd0c351904ec5dd416a3a5ff6e16d06b25cd8b70b0fc896` |
| `codeformer_u96_adapter.pth` | 212827 | `958e8338d4bc1a0204c314c3af239ab0ab3d783c67e3e532d9fe23c58f8a73d0` |
| `terminal_u0064_heads.npz` | 685818 | `77c0e1f236eb70d37f3f81059b75580e92d606bb2f8b5dd5eeb00da0305dca17` |
| `v2_p2_u0224_heads.npz` | 3831362 | `1aee2b840397925049a2c821edd7395c3a9e4b48ce1552e06c5ffd627a33e31f` |

## Intended use and limits

Use one already cropped frontal or mildly turned face with readable visible structure. Auto provides an input-quality suggestion, not a usefulness or identity score. Conditioned SFT V2 remains experimental and can change facial appearance; application operation does not establish restoration accuracy or recovered identity.

Preparation can be bypassed or reviewed manually with Protect/Apply masks. Exact protected and outside-Apply pixels concern preprocessing only. The generated result has no corresponding facial-pixel guarantee. Neutral padding remains preserved. Optional completion is a separate R1 stage described in [COMPLETION_MODEL_CARD.md](COMPLETION_MODEL_CARD.md). Automatic blur matching and recovery of hidden identity are not supported.

Review the original, selected input, and result together. Prefer clearer or less-covered input when information is insufficient. Optional evidence ZIPs contain the original and available masks, preparation records, RAW output, and result. Request records remain locally in `scratch/implementation_requests`; downloaded evidence also contains image data.

## Attribution and license

CodeFormer is an S-Lab architecture used here with trained adaptations. Retain [the S-Lab license](third_party/codeformer/LICENSE) and [notices](third_party/codeformer/NOTICE.md). Its non-commercial terms apply to use and distribution. No training dataset is included.
