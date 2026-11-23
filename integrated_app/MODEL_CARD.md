# Application model card — IMPLEMENTATION

This branch packages the frozen **Pre-Conditioned Frozen CodeFormer V3.4 (`preconditioned_v34_codeformer`)** model for application development, deployment, and real-world testing. Research and training are maintained separately.

**Pre-Conditioned Frozen CodeFormer V3.4 is the only restoration model in the app.** It combines the untouched frozen official S-Lab CodeFormer backbone (`codeformer.pth`, 515 tensors) with a lightweight 8.65 MiB V3.4 pre-conditioning front-end (`v34_precond_modules.npz`, 344 tensors, 2,241,395 parameters: `V34DegradationEncoder`, `V34PhotoCorrectNet`, `V34NAFPreConditionNet`), totaling 859 frozen tensors. It predicts adaptive CodeFormer fidelity (`w_pred`), photometric correction (`g_photo`), NAFNet pre-conditioning (`g_pre`), and clean-input bypass (`alpha_clean`) in a single CPU FP32 pass (~6.46s per 256×256 crop).

## Runtime and assets

The model loads in disposable CPU workers only when restoration is required. Each request permits at most one inference call. Non-square crops use mask-aware nearest-border replicate padding internally and MaskedAdaIN over observed support, while final 256×256 RAW float32 and floor-quantized PNG outputs preserve exact neutral-128 padding outside observed support. Off returns the resized original with zero neural forwards.

The installer verifies these three assets in `models/implementation` (`codeformer.pth`, `v34_precond_modules.npz` staged from `implementation/assets/`, and `fbcnn_color.pth` for Step 2 JPEG/macroblock deblocking). All files must pass complete size and SHA-256 verification.

| Component label | Asset | Bytes | SHA-256 |
| --- | --- | ---: | --- |
| **JPEG Deblocking** | `fbcnn_color.pth` | 287755111 | `8b0e4ef23d59cf7ac934a342cb31a17619e4fa4a0b3374a9d78c5174312387e8` |
| **CodeFormer Base Model** | `codeformer.pth` | 376637898 | `1009e537e0c2a07d4cabce6355f53cb66767cd4b4297ec7a4a64ca4b8a5684b7` |
| **V3.4 Pre-Conditioning Modules** | `v34_precond_modules.npz` | 9065704 | `65541ac31b50becafec3db34cca8936ab92c5958b0afd75244d85ecd0e0e88b1` |

Canonical state digests:
- `codeformer_base_515`: `6cb93cee937f67814edcf4625a68bae4bf8f1c2351f7306864ceefa63d4803bd`
- `v34_precond_modules_344`: `cc679bd6ffd919a21a92a2f07de0c2f1f2602d9449a4c714e3ec82cd5dce61bd`

## Intended use and limits

Use one already cropped frontal or mildly turned face with readable visible structure. Auto provides an input-quality suggestion, not a usefulness or identity score. Pre-Conditioned Frozen CodeFormer V3.4 remains experimental and can change facial appearance; application operation does not establish restoration accuracy or recovered identity.

Preparation can be bypassed or reviewed manually with Protect/Apply masks. Exact protected and outside-Apply pixels concern preprocessing only. The generated result has no corresponding facial-pixel guarantee. Neutral padding remains preserved. Optional completion is a separate R2 stage described in [COMPLETION_MODEL_CARD.md](COMPLETION_MODEL_CARD.md).

Review the original, selected input, and result together. Prefer clearer or less-covered input when information is insufficient. Optional evidence ZIPs contain the original and available masks, preparation records, RAW output, and result. Request records remain locally in `scratch/implementation_requests`; downloaded evidence also contains image data.

## Attribution and license

CodeFormer is an S-Lab architecture used here with trained V3.4 pre-conditioning modules. Retain [the S-Lab license](third_party/codeformer/LICENSE) and [notices](third_party/codeformer/NOTICE.md). Its non-commercial terms apply to use and distribution. No training dataset is included.
