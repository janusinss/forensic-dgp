# Unified CodeFormer Completion R2 V2 Preparation (`completion_codeformer_unified_r2_v2`)

## 1. Local Preparation Summary

- **Packet Name**: `cctv_completion_codeformer_unified_r2_v2_vm`
- **Archive Stem**: `cctv-completion-codeformer-unified-r2-v2`
- **Protocol Format**: `completion-codeformer-unified-r2-v2`
- **Orthogonal 128-Case Schedule**: `research/completion_codeformer_unified_r2_v2_development/orthogonal_balanced128_subset.json` (`202,247` bytes, SHA-256 `0bb42287249693ab208fa1a609b263950cd3ebeec877480103d28f99b1f6c925`).
- **Architecture Compatibility**: Exact 497-tensor `CodeFormer` (`dim_embd=512, codebook_size=512, n_head=8, n_layers=9, connect_list=['32','64','128']`) drop-in compatible with `IMPLEMENTATION` (`C:\Users\janus\.codex\worktrees\implementation-one-shot\Testing`).
- **Execution Mode**: Manual L4 VM transfer and execution only. Zero VM training calls were launched during local preparation.

## 2. Verification & Admission Gates

1. **Pre-Update Parity & Compatibility**:
   - 4 fresh app parity pairs (`own` vs `CodeFormerCompletion`) must match within `<= 1e-6` RAW error and exact input layout/stride before any optimizer step.
   - All 48 historical R1 baseline cases (32 clear monitor + 16 challenge) must match within `<= 1e-6` RAW error and byte-exact PNG SHA-256 before any optimizer step.
2. **Post-Training Ratchet Gates (Cycle 32)**:
   - Clear monitor macro `PNG_hole_L1` relative improvement `>= 1.0%` over R1 (`0.039815`).
   - Challenge monitor macro `PNG_hole_L1` relative improvement `>= 1.0%` over R1 (`0.051271`).
   - All clear and challenge macro, 4-geometry, and 2-condition RAW/PNG hole and $7\times 7$ inner-boundary L1 subgroups non-worse than R1.
   - All 8 strict-gray challenge macro and 4-geometry `GRAY_raw_mean_channel_spread` and `GRAY_PNG_mean_channel_spread` subgroups non-worse than R1.
   - Exact outside-mask preservation (`0.0` RAW and `0` uint8 PNG diff) and 168 exact empty-mask bypasses (`0` neural calls).
