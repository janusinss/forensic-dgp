# Unified CodeFormer Completion R2 V1 Design (`completion_codeformer_unified_r2_v1`)

## 1. Purpose & Baseline Provenance

This specification defines the unified **Trained CodeFormer Completion R2 V1** (`cctv_completion_codeformer_unified_r2_v1_vm`) continuation from the admitted incumbent **Trained CodeFormer Completion R1**:

- **Parent Full Recovery State**: `outputs/completion_codeformer_finetune_v1_r1_return/cctv_completion_codeformer_finetune_v1_r1_results/outputs/terminal_training_state.pth` (`701952165` bytes, SHA-256 `0a6b4ad907c6ead86d8f26d3446666387493cdce81c6adc1ed78a8677d50fadd`, logical SHA-256 `55feaa296f5164a6bd12995b487c9366c5fe7778f0e9dbe8912fc4ee7c4122d6`).
- **Original Base Checkpoint**: `codeformer_inpainting.pth` (`370786611` bytes, SHA-256 `b0d1b868c3dacf75d637bbcc0d4b3dd4ce2a064a338dfb4d35c740d3b4ae8797`, logical SHA-256 `586a8854ed7b5e3f0e202c258addb7a87de63cd3068d153741b065379ff6668f`).
- **Cached Same-Codebook Teacher Labels & Latents**: 80 `.npz` label sets from `outputs/completion_codeformer_coverage_v2_return/cctv_completion_codeformer_coverage_v2_results/outputs/teacher_labels/index.json` (SHA-256 `0567a9ad54e6397390341a8dfc1ffa8ec50ceb9159da299605fee021ec678a69`): 64 FIT targets used for optimization and 16 monitor sets retained strictly for diagnostic readout (`0` teacher neural calls on VM).
- **Balanced 128-Case Schedule**: `research/completion_codeformer_predictor_ce_design_independent_review_v1/balanced128_subset.json` (SHA-256 `8ec2d535cd6fb231af815d0ec833a1cd08f418458cbc4919c0ebd13f638f2a1d`): 32 paired cycles $\times$ 4 microbatches = 128 balanced observations (64 clear anchors + 64 augmented RGB/GRAY/irregular observations across all 32 FIT references and 4 mask geometries).

---

## 2. Five Unified Training Improvements Over Historical Pilots

1. **Support-Weighted $16\times 16$ Spatial Code Cross-Entropy + `lq_feat` Latent MSE**:
   - Downsamples the $256\times 256$ binary mask to the $16\times 16$ codebook token grid via $m_{16} = \operatorname{max\_pool2d}(m_{256}, 16, 16)$ and extracts the 1-cell context ring $r_{16} = \operatorname{clamp}(\operatorname{max\_pool2d}(m_{16}, 3, 1, 1) - m_{16}, 0, 1)$.
   - Assigns spatial weights $w_{16} = 1.0\,m_{16} + 0.5\,r_{16} + 0.15\,(1 - m_{16} - r_{16})$ normalized over the 256 positions so $\operatorname{mean}(w_{16}) = 1.0$.
   - Supervises both `logits` `(1, 256, 512)` against `nearest_codes` and `lq_feat` `(1, 256, 16, 16)` against `prequant_latent` using $w_{16}$.
2. **Joint 332-Tensor Code + Fusion Co-Adaptation**:
   - Restores R1's 278-tensor `code_optimizer` (164 encoder + 114 predictor tensors) and 54-tensor `fusion_optimizer` (`fuse_convs_dict['32','64','128']`) at Adam step 32.
   - Executes paired code-route and fusion-route updates in each of the 32 cycles (advancing both optimizers $32 \to 64$), while keeping the 165 `quantize`/`generator` prior tensors strictly frozen.
3. **Dual-Resolution (`256` Delivered + `512` Internal) Hole, Seam, and Spatial Gradient L1**:
   - Supervises exact app `delivered_float` at $256\times 256$ (`1.0` hole L1 + `0.5` $7\times 7$ inner-boundary L1 + `0.25` finite-difference gradient L1) jointly with internal $512\times 512$ clamped prediction (`0.5` hole L1 + `0.5` $7\times 7$ inner-boundary L1).
4. **Differentiable Inter-Channel Chroma Consistency Loss**:
   - Penalizes pairwise channel-difference errors $|(\hat{R}-\hat{G})-(R^*-G^*)| + |(\hat{G}-\hat{B})-(G^*-B^*)| + |(\hat{B}-\hat{R})-(B^*-R^*)|$ inside the mask at both $256\times 256$ and $512\times 512$ with weight `0.35` on RGB cases and `0.70` ($2\times$ boost) on strict `GRAY` cases to eliminate grayscale blue/orange false-color bias.
5. **Clear-Anchor Trust-Region Weighting & O(1) `OutputBudget` Serialization**:
   - Weights clear anchors at `1.25` and augmented cases at `0.75` (mean weight `1.0`), applies a cosine learning-rate schedule (`8e-6` $\to$ `2e-6`), anchors trainable parameters to $\theta_{\text{R1}}$ via L2 penalty (`coef=1e-4`), and enforces O(1) disk/extent accounting (`OutputBudget` + `admit_terminal_zip`).
