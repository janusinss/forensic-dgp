# Active Goal V2: Condition-Sensitive SFT Feature Distillation & Photometric-Decoupled Restoration

Created: 25 October 2026
Workspace: `C:\xampp\htdocs\YEAR 4\Testing_RESTORATION` (Branch: `RESTORATION`)
Predecessor Goal: [CCTV_CODEFORMER_U96_ACTIVE_GOAL_V1.md](CCTV_CODEFORMER_U96_ACTIVE_GOAL_V1.md) and [CCTV_CODEFORMER_U96_GOAL_EXTENDED_SPEC_V1.md](CCTV_CODEFORMER_U96_GOAL_EXTENDED_SPEC_V1.md)

---

## 1. Concise `/goal` Command Prompt

Use the `/goal` slash command with the following directive:

```text
/goal Improve the 256x256 CCTV CodeFormer face restoration model in C:\xampp\htdocs\YEAR 4\Testing_RESTORATION (RESTORATION branch) to achieve both strict clear/normal-tone preservation and visible facial detail improvement (eyes, brows, nose, mouth, jaw) over incumbent U96 and retained challenger u64. Follow CCTV_CODEFORMER_RESTORATION_GOAL_V2.md end-to-end: (1) run a local saved-array feasibility proof verifying that channel-mean vs zero-mean spatial decoupling eliminates the Matched-V1 Arm-B clear-PNG RGB8 regression (+1,126 codes), (2) implement and unit-test the condition-sensitive 60-tensor (24 U96 pre-SFT adapters + 36 u64 post-fusion spatial heads, 515 R2 frozen) learning package with input-only degradation/luminance gating, multi-site clean-SFT encoder feature distillation at sites 32/64/128/256, and role-balanced gradient projection, (3) seal and independently audit the manual NVIDIA L4 VM training packet and 5-step gcloud/tmux guide, and (4) upon audited return, verify all 56-case inherited + 40-case monitor RAW/PNG gates, broader DEV/native CCTV visual sheets, and local app integration with inline Playwright checks at 375px, 768px, and 1280px.
```

---

## 2. Technical Diagnosis & Why Prior Pilots Stopped

1. **Untapped SFT Feature Headroom**: The Clean-SFT-Input Oracle ([CCTV_CODEFORMER_CLEAN_SKIP_ORACLE_V1_REVIEW.md](CCTV_CODEFORMER_CLEAN_SKIP_ORACLE_V1_REVIEW.md)) proved that substituting clean R2 encoder features at SFT sites `32`, `64`, `128`, `256` (while keeping codes, logits, and AdaIN fixed) reduces moderate-blur PNG MAE from `0.054767` to `0.035207` (SSIM `0.802 -> 0.916`) and readable strong-blur PNG MAE from `0.063501` to `0.031791` (SSIM `0.686 -> 0.919`), producing substantial visible eye, mouth, and contour improvements.
2. **Photometric Mean-Shift Leakage into Clear Images**: In [CCTV_CODEFORMER_MATCHED_FEATURE_LEARNING_V1_RETURN_REVIEW.md](CCTV_CODEFORMER_MATCHED_FEATURE_LEARNING_V1_RETURN_REVIEW.md) and [CCTV_CODEFORMER_MATCHED_FEATURE_LEARNING_V1_PHOTOMETRIC_DIAGNOSIS.md](CCTV_CODEFORMER_MATCHED_FEATURE_LEARNING_V1_PHOTOMETRIC_DIAGNOSIS.md), Arm B (60 trainable tensors) improved degraded MAE across all 4 degraded profiles and improved clear spatial error (`-0.0000088` RAW MAE contribution), but failed at update 8 because unconditioned channel-mean color drift (`+0.0000084` RAW MAE contribution, `+14,492` RGB8 code units on `tr_ffhq_11039`) increased quantized clear PNG error by `+1,126` code units (`+0.008849%`).
3. **Low-Light Gradient Dominance**: [CCTV_CODEFORMER_U64_ROLE_GRADIENTS_V1_RETURN_REVIEW.md](CCTV_CODEFORMER_U64_ROLE_GRADIENTS_V1_RETURN_REVIEW.md) established that low-light loss gradients represent `62.23%` of total role-gradient norm and oppose clear/blur gradients on 12–15 of 16 matched photographs.
4. **Weak Image-Only Supervision at Tiny LR**: Clamping unconditioned Adam to `LR=1e-5` for 8 updates yielded only `+0.047%` local and `+0.079%` global new PNG structure gain over `u64`, which is visually indistinguishable on 256×256 faces.

---

## 3. Best-Approach Architecture & Training Design (`codeformer_conditioned_sft_learning_v1`)

### A. Initializer & 575-Tensor Ownership
1. **Canonical Starting State**: Initialize from the exact complete 575-tensor `u64` state (`47e0707db446d3e39e9c101e8eedd631e49b45ba4f03a5f00aa2ad02e0281a8a`), comprising:
   - `515` frozen `R2` tensors (`a3b0986c793cb7c937ca82efde6c9e23e70557d76b93406e0f53e51b93406e0f53e51b938f9e7f`)
   - `24` trainable `U96` pre-SFT `ResidualSkipAdapter` tensors (`51,328` parameters, `fbfaa9d01c5ab04d2c4383485ba4e1eebb3d9f43cdab427773447248d14a586a`)
   - `36` trainable `u64` post-fusion `SpatialHead` tensors (`168,960` parameters, `6629d8414321b0d2ba3a9cc476e415d31911fd4bd1fa121a1a8db9097ce235be`)
2. **Exact Zero-Update Parity**: At step 0 (when residual gates/deltas relative to `u64` are zero), the model must reproduce `u64` RAW and `floor(RAW * 255)` PNG outputs bitwise on all control witnesses.

### B. Input-Only Conditioned Photometric & Spatial Decoupling
1. **Strict Input-Only Conditioning**: Compute deterministic scalar/spatial conditioning signals exclusively from the uploaded/prepared 256×256 input tensor `x` (never from clean targets, synthetic role names, or oracle features):
   - **Low-light gate** `g_dark(x) in [0, 1]`: smooth ramp active only when input luminance mean/percentile falls below normal-lit facial exposure thresholds; `g_dark(x) = 0` on normal-lit and clear inputs.
   - **Blur/degradation gate** `g_deg(x) in [0, 1]`: smooth ramp derived from multi-scale high-frequency Laplacian/gradient energy so sharp/clear inputs receive `g_deg(x) -> 0` incremental perturbation relative to the initial state while blurred/compressed inputs receive `g_deg(x) -> 1`.
2. **Orthogonal Mean/Spatial Split in Post-Fusion & Output Updates**:
   - Split incremental feature/output corrections relative to the initial `u64` trajectory into a **spatial zero-mean field** (`delta - mean_hw(delta)`) and a **channel-mean shift** (`mean_hw(delta)`).
   - Gate channel-mean shifts by `g_dark(x)` so normal-lit clear and blurred inputs cannot accumulate global RGB channel-mean drift from low-light training updates.

### C. Multi-Site Clean-SFT Feature Distillation + Role-Balanced Objective
1. **Direct SFT Feature Alignment (`L_SFT`)**: At sites `32`, `64`, `128`, `256`, supervise the adapted encoder features `used_s = Adapter_s(e_deg_s)` against detached frozen-R2 encoder features extracted from the clean 256→512 photographic target `e_clean_s`:
   $$L_{\text{SFT}} = \sum_{s \in \{32, 64, 128, 256\}} w_s \left( \| \text{used}_s - e_{\text{clean}, s} \|_1 + \lambda_{\text{cos}} \left(1 - \cos(\text{used}_s, e_{\text{clean}, s})\right) \right)$$
2. **Decoupled Photographic Loss (`L_photo`)**:
   - **Spatial Structure Term**: Zero-mean RGB residual L1 + multi-scale (1px, 2px, 4px) horizontal/vertical residual gradient L1 (directly targeting eye/brow/nose/mouth sharpness and the 1% adjacent-error gate).
   - **Coarse Color / Tone Term**: 16×16 block-mean RGB residual L1 + channel-mean L1, weighted separately so clear/normal-blur tone preservation is enforced without suppressing local high-frequency edges.
   - **Quantizer-Aware Clear Anchor**: Smooth surrogate penalty on RGB8 floor-boundary crossings for clear/normal-lit roles.
3. **Role-Balanced Gradient Projection**:
   - Normalize per-role gradient contributions across the 5 admitted roles (`clear`, `global_gaussian2p0`, `local_gaussian2p0_soft`, `compound_g1p2_lr64_jpeg70`, `lowlight0p35_noise0p006`) and project out conflicting components against the `clear` preservation gradient before each Adam update.

---

## 4. Bounded Execution Lifecycle

1. **Stage 1 — Saved-Array Photometric & Gate Feasibility Proof (Local CPU)**:
   - Run a deterministic NumPy/CPU verification script on the retained `Matched V1` (`u64`, `A8`, `B8`) saved RAW/PNG outputs to prove how channel-mean decoupling and input-only gating resolve the `+1,126` clear PNG code error and pass all 56-case + 40-case gates.
2. **Stage 2 — Source Implementation & Pure Unit Test Suite**:
   - Implement `research/codeformer_conditioned_sft_learning_v1/` (`contract.py`, `conditioning.py`, `model.py`, `objective.py`, `quality.py`, `worker.py`, `packet.py`) with 100% passing unit tests (`pytest`) under no-VM/no-link guards.
3. **Stage 3 — Sealed Manual L4 VM Packet & 5-Step Launch Guide**:
   - Package the verified transfer archive, protocol SHA256, and `CCTV_CODEFORMER_CONDITIONED_SFT_LEARNING_V1_VM.md` for manual execution on the NVIDIA L4 VM (`~/forensic-dgp`).
4. **Stage 4 — Return Audit, Broader DEV/Native CCTV Review & App Integration**:
   - Audit returned checkpoints, RAW/PNG pixel gates, and 256×256 visual sheets; benchmark on broader paired DEV + unpaired native CCTV (`QMUL-SurvFace`, `ChokePoint`); integrate the qualified model into the web workflow and verify with inline Playwright at `375px`, `768px`, and `1280px`.
