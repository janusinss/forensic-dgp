# CCTV CodeFormer Conditioned SFT V1 — Stage 1 Saved-Array Feasibility Proof

## 1. Executive Verdict

- **Status**: `SAVED_ARRAY_PHOTOMETRIC_AND_GATE_FEASIBILITY_VERIFIED_V1`
- **Zero-Overlap Input Exposure Separation Across All 216 Cases**: `True`
- **Unmodified Arm-B (`B8`) Clear Monitor PNG RGB8 Delta**: `+1126` codes (`inh8=False`, `sup8=False`)
- **Global-Mean Gated (`B8`) Clear Monitor PNG RGB8 Delta**: `-3647` codes (`inh8=True`, `sup8=True`)
- **Block16 Drift-Removed (`B8`) Clear Monitor PNG RGB8 Delta**: `-8296` codes (`inh8=True`, `sup8=True`, `sup48=True`)
- **Single-Pass Input-Anchored (`B8`, `beta=0.005`) Clear Monitor PNG RGB8 Delta**: `-48923` codes (`all_gates_pass_u8_and_u48=True`)
- **Combined Drift-Removed + Input-Anchored (`B8`, `beta=0.005`) Clear Monitor PNG RGB8 Delta**: `-57169` codes (`all_gates_pass_u8_and_u48=True`)

## 2. Input-Only Exposure Separation (`top1pct_luma`) Across 216 Stage Cases

| Group / Role | Cases | `top1pct_luma` Min | `top1pct_luma` Max | `g_dark` Min | `g_dark` Max | `g_deg` Min | `g_deg` Max |
|---|---:|---:|---:|---:|---:|---:|---:|
| `clear` | 48 | `0.7020` | `1.0000` | `0.0000` | `0.0000` | `0.2500` | `0.7785` |
| `compound_g1p2_lr64_jpeg70` | 32 | `0.6838` | `1.0000` | `0.0000` | `0.0000` | `0.9193` | `1.0000` |
| `dark` | 8 | `0.4526` | `0.6093` | `1.0000` | `1.0000` | `0.5574` | `0.9764` |
| `filter` | 8 | `0.7239` | `0.9258` | `0.0000` | `0.0000` | `1.0000` | `1.0000` |
| `global_gaussian2p0` | 32 | `0.6855` | `1.0000` | `0.0000` | `0.0000` | `1.0000` | `1.0000` |
| `global_gaussian_1p2` | 8 | `0.7367` | `0.9791` | `0.0000` | `0.0000` | `0.9329` | `1.0000` |
| `local_gaussian2p0_soft` | 32 | `0.7020` | `1.0000` | `0.0000` | `0.0000` | `0.7487` | `1.0000` |
| `local_gaussian_1p2` | 8 | `0.7415` | `0.9822` | `0.0000` | `0.0000` | `0.8452` | `1.0000` |
| `lowlight0p35_noise0p006` | 32 | `0.4347` | `0.6356` | `1.0000` | `1.0000` | `0.2500` | `0.5803` |
| `lowlight_linear_0p35` | 8 | `0.4618` | `0.6146` | `1.0000` | `1.0000` | `0.2500` | `0.6823` |

## 3. Coarse 16x16 Block Photometric Accuracy: `u64` vs Input Observation `x`

| Monitor Role | `u64` 16x16 Block MAE | Input `x` 16x16 Block MAE | Input Accuracy Advantage |
|---|---:|---:|---:|
| `clear` | `0.027663` | `0.000000` | `INF (exact 0 error)` |
| `compound_g1p2_lr64_jpeg70` | `0.040133` | `0.005303` | `7.57x` |
| `global_gaussian2p0` | `0.036573` | `0.003093` | `11.82x` |
| `local_gaussian2p0_soft` | `0.030462` | `0.000916` | `33.26x` |
| `lowlight0p35_noise0p006` | `0.092854` | `0.164387` | `0.56x` |

## 4. Saved-Bank Gate Matrix Across `u8` and Terminal `u24`/`u48`

| Variant | `inh8` | `inh48` | `sup8` | `sup48` | Inherited Clear PNG MAE | Monitor Clear PNG MAE | Monitor Clear RGB8 Delta | Monitor Compound PNG MAE | Monitor Lowlight+Noise PNG MAE |
|---|---|---|---|---|---:|---:|---:|---:|---:|
| `unmodified_B8` | `False` | `False` | `False` | `False` | `0.03172688` | `0.03172688` | `+1126` | `0.04798975` | `0.09541682` |
| `global_mean_gated_B8` | `True` | `True` | `True` | `False` | `0.03171498` | `0.03171498` | `-3647` | `0.04806228` | `0.09541682` |
| `block16_drift_removed_B8` | `True` | `False` | `True` | `True` | `0.03170339` | `0.03170339` | `-8296` | `0.04805636` | `0.09541682` |
| `single_pass_input_anchored_B8_beta0p005` | `True` | `True` | `True` | `True` | `0.03160209` | `0.03160209` | `-48923` | `0.04781883` | `0.09541682` |
| `drift_removed_plus_input_anchored_B8_beta0p005` | `True` | `True` | `True` | `True` | `0.03158154` | `0.03158154` | `-57169` | `0.04788671` | `0.09541682` |
