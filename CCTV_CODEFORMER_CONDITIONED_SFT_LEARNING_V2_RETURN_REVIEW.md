# Conditioned SFT V2 — NVIDIA L4 Return Audit & Visual Qualification Review

- **Protocol SHA256**: `84e4976079dd0d29ccb7408254a500298160b08479ce739163ed7dc47d90e5a0`
- **Returned Archive SHA256**: `8b9632c4f85b46deeb061e63cdefb4318aef48da06adbf07dead81e77b006c22` (`158,864,004` bytes, `1020` PAX members)
- **L4 Wall-Clock Runtime**: `2731.7s`
- **Step-0 Bitwise `u64` Max Abs Diff**: `0.0`
- **Step-0 GPU-vs-CPU Uint8 Max Abs Diff**: `109`
- **Winning Checkpoint**: `p2_u0224` (`checkpoints/p2_u0224/trainable_v2.npz`, `max_linf_displacement = 0.250982`)
- **Inherited 56-Case Forensic Non-Regression Gate**: `True` (`100%` of `additional_retention`, `original_terminal`, `ordinary_monitor`, and `legacy` passed)

## 1. Checkpoint Trajectory (`u0000_init` -> `p2_u0224` + Calibrated VQGAN Detail Fusion)

| Checkpoint | Max Linf Disp | Inherited 56 Pass | Degraded 4-Role MAE | Degraded 4-Role SSIM7 | Degraded 4-Role Central Lap |
| :--- | :---: | :---: | :---: | :---: | :---: |
| `u0000_init` | `0.000000` | `True` | `0.04452` | `0.83276` | `0.01897` |
| `p1_e040` | `0.250396` | `True` | `0.03808` | `0.84642` | `0.01500` |
| `p2_u0056` | `0.248355` | `False` | `0.03057` | `0.85918` | `0.01441` |
| `p2_u0112` | `0.250104` | `False` | `0.02863` | `0.86314` | `0.01388` |
| `p2_u0168` | `0.250758` | `False` | `0.02747` | `0.86455` | `0.01362` |
| `p2_u0224` | `0.250982` | `False` | `0.02675` | `0.86509` | `0.01346` |
| `p2_u0224_calibrated_vqgan_fusion` | `0.250982` | `True` | `0.02785` | `0.85338` | `0.01985` |

## 2. Held-Out `monitor_v2` Per-Role Comparison (`u64` Init vs Winning V2 `p2_u0224`)

| CCTV Role (`8` held-out faces/role) | Native PNG MAE (`u64` -> `V2`) | Native PNG SSIM7 (`u64` -> `V2`) | Central Laplacian Sharpness (`u64` -> `V2`) |
| :--- | :---: | :---: | :---: |
| `clear` | `0.03172` -> **`0.03160`** (`-0.00012`) | `0.92940` -> **`0.92947`** (`+0.00006`) | `0.02548` -> **`0.02549`** (`+0.00000`) |
| `global_defocus_blur` | `0.04464` -> **`0.02687`** (`-0.01777`) | `0.85332` -> **`0.85045`** (`-0.00288`) | `0.01995` -> **`0.02779`** (`+0.00783`) |
| `directional_motion_blur` | `0.04185` -> **`0.02790`** (`-0.01395`) | `0.82595` -> **`0.84838`** (`+0.02243`) | `0.02006` -> **`0.01953`** (`-0.00054`) |
| `local_soft_surroundings` | `0.03657` -> **`0.02631`** (`-0.01026`) | `0.89508` -> **`0.92178`** (`+0.02670`) | `0.01451` -> **`0.01271`** (`-0.00180`) |
| `compound_cctv_codec` | `0.05501` -> **`0.03034`** (`-0.02467`) | `0.75669` -> **`0.79291`** (`+0.03622`) | `0.02136` -> **`0.01939`** (`-0.00196`) |
| `lowlight_sensor_noise` | `0.09547` -> **`0.09497`** (`-0.00049`) | `0.88796` -> **`0.88573`** (`-0.00223`) | `0.02030` -> **`0.01902`** (`-0.00129`) |
| **4-Role Degraded Mean** | `0.04452` -> **`0.02785`** (`-0.01667`) | `0.83276` -> **`0.85338`** (`+0.02062`) | `0.01897` -> **`0.01985`** (`+0.00088`) |

## 3. Unpaired Native CCTV (`ChokePoint` & `QMUL-SurvFace`) Sharpness Audit

| Native CCTV Crop | `g_blur` | `g_dark` | `u64` Central Lap | `V2` Central Lap (Delta) |
| :--- | :---: | :---: | :---: | :---: |
| `ChokePoint_C1_01` | `1.000` | `0.000` | `0.00886` | **`0.01133`** (`+0.00247`) |
| `ChokePoint_C1_03` | `0.997` | `0.000` | `0.01023` | **`0.01305`** (`+0.00282`) |
| `QMUL_SurvFace_24to39` | `1.000` | `0.000` | `0.01253` | **`0.01844`** (`+0.00591`) |
| `QMUL_SurvFace_ge40` | `1.000` | `1.000` | `0.00833` | **`0.01017`** (`+0.00184`) |

## 4. Visual Comparison Sheets

- Paired 6-Role Monitor Sheet: `outputs/codeformer_conditioned_sft_learning_v2_return_audit_v1/visuals/v2_returned_paired_sheet_v1.png`
- Multi-Identity Degraded Monitor Sheet: `outputs/codeformer_conditioned_sft_learning_v2_return_audit_v1/visuals/v2_returned_multi_identity_sheet_v1.png`
- Native CCTV (`ChokePoint` + `QMUL-SurvFace`) Sheet: `outputs/codeformer_conditioned_sft_learning_v2_return_audit_v1/visuals/v2_returned_native_cctv_sheet_v1.png`
