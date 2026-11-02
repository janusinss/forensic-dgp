# Forensic Deep Generative Prior (DGP) & CodeFormer Reconstruction System for Degraded CCTV Video

[![Website: forensic-dgp.pages.dev](https://img.shields.io/badge/Download_Installer-forensic--dgp.pages.dev-0A66C2?style=flat&logo=windows&logoColor=white)](https://forensic-dgp.pages.dev/)
[![Release: v0.2.2 Offline Installer](https://img.shields.io/badge/Windows_Installer-v0.2.2_Offline-2ea44f?style=flat&logo=github)](https://github.com/janusinss/forensic-dgp-downloads/releases/tag/v0.2.2)
[![PyTorch](https://img.shields.io/badge/PyTorch-%23EE4C2C.svg?style=flat&logo=PyTorch&logoColor=white)](https://pytorch.org/)
[![FastAPI](https://img.shields.io/badge/FastAPI-005571?style=flat&logo=fastapi)](https://fastapi.tiangolo.com/)
[![License: Academic Research](https://img.shields.io/badge/License-Academic%20Research-blue.svg)](https://opensource.org/licenses)

> **Thesis Title**: *Forensic Deep Generative Prior Face Reconstruction for Degraded CCTV Video in Zamboanga City*  
> **Institution**: Western Mindanao State University (WMSU), College of Computing Studies, Department of Computer Science  
> **Researchers**:
> - **Janus R. Dominic**
> - **Winston T. Tabotabo**
> - **Roberto "RJ" B. Toribio III**

---

## 📌 Overview

This repository contains the official software implementation, research history, and desktop packaging of our forensic CCTV face reconstruction and reviewed occlusion-completion system developed for severely degraded surveillance footage in Zamboanga City:

1. **Stage 1 — Blind CCTV Face Restoration (`Conditioned SFT V2 / p2_u0224`)**: Exposure-matched conditioned Spatial Feature Transform (SFT) over a `787`-tensor frozen `u64` CodeFormer backbone (`v2_p2_u0224_heads.npz`), ratified across `56/56` inherited forensic retention gates with a **`-37.4%`** reduction in 4-role degraded `MAE` (`0.04452 -> 0.02785`) and **`+22.1%` to `+47.1%`** sharpness gains on native `ChokePoint` and `QMUL-SurvFace` CCTV frames.
2. **Stage 2 — Reviewed Face Mask Detection & Completion (`Trained CodeFormer Completion R2 / v3`)**: Proposal-only covering segmenter (`GatedUNet(3, 1, 24)`, `covering_detector_segmenter_v1.pth`) paired with mandatory human brush review/confirmation and **Trained CodeFormer Completion R2** (`codeformer_completion_r2.pth`, `332` adapted tensors across `497` `params_ema` tensors), ratified across **`60/60` numeric ratchet gates** (`-1.05%` clear hole $L_1$, `-2.69%` challenge hole $L_1$, `-10.32%` strict-gray chroma spread) with **`100%` exact byte-for-byte preservation** of all pixels outside the reviewed mask support.
3. **Foundational Research Pipeline — Morphologically Constrained DGP (`Phase 1–5` & `V6–V42`)**: MobileNetV2 Feature Pyramid Network (FPN) and ResNet-Generator architectures anchored by a **68-point Morphological FAN Loss** and universal blind signal pre-conditioning (`degradation.py`).

For the complete chronological engineering origins from scratch to production ratification, see:
- [**`ORIGINS_RESTORATION.md`**](ORIGINS_RESTORATION.md) — Chronological engineering history of the CCTV Blind Face Restoration model (`Phase 1–5` $\rightarrow$ `V6–V42` $\rightarrow$ `CodeFormer R2 / U96 / u64` $\rightarrow$ `Conditioned SFT V1 & V2 / p2_u0224`).
- [**`ORIGINS_FACE_MASK_COMPLETION.md`**](ORIGINS_FACE_MASK_COMPLETION.md) — Chronological engineering history of the Face Mask Detection & Completion model (`GatedUNet` $\rightarrow$ `AOT / LaMa / MAT` $\rightarrow$ `CodeFormer Completion R1` $\rightarrow$ `Unified CodeFormer Completion R2 v3`).
- [**`PROJECT_HANDOFF.md`**](PROJECT_HANDOFF.md) — Master cross-branch engineering ledger and verification record.

---

## 🌐 Installation & Quick Start

You can install and run the system in three ways depending on your use case:

### Option A (Recommended for End Users): Official Windows Offline Installer Website
**Website**: [**https://forensic-dgp.pages.dev/**](https://forensic-dgp.pages.dev/)  
**Direct Release**: [**`ForensicDGP_Setup-0.2.2.exe` (`v0.2.2`)**](https://github.com/janusinss/forensic-dgp-downloads/releases/tag/v0.2.2)

The standalone Windows desktop installer bundles a private 64-bit Python runtime, all locked CPU wheels, and all verified restoration (**Conditioned SFT V2 `p2_u0224`**) and completion (**CodeFormer Completion R2 `v3`**) model weights (`1,045,576,043` bytes; **zero internet connection or system Python required during setup or inference**):

1. Visit [**https://forensic-dgp.pages.dev/**](https://forensic-dgp.pages.dev/) and click **Download for Windows (64-bit)** to download `ForensicDGP_Setup-0.2.2.exe`.
2. *(Optional)* Verify the installer checksum in PowerShell:
   ```powershell
   Get-FileHash .\ForensicDGP_Setup-0.2.2.exe -Algorithm SHA256
   # Expected SHA-256: bc9952a5e54a3955a06a7e3b636d008f8dee2f227ef470224ee927463d733a25
   ```
3. Run **`ForensicDGP_Setup-0.2.2.exe`**, choose your install folder (supports fresh install, in-place upgrade from `0.2.0`/`0.2.1`, source/package repair, and receipt-bound folder relocation), and launch **Forensic DGP** from the Start Menu or Desktop shortcut.

---

### Option B (Developer Checkout): Integrated Production Workbench (`Conditioned SFT V2` + `Completion R2`)

The `integrated_app/` directory on `main` (and the dedicated [`IMPLEMENTATION`](https://github.com/janusinss/forensic-dgp/tree/IMPLEMENTATION) branch) contains the full production web workbench combining CCTV video frame capture, **Conditioned SFT V2 (`p2_u0224`)** restoration, interactive mask proposal/editing, and **Trained CodeFormer Completion R2 (`v3`)**:

```powershell
git clone https://github.com/janusinss/forensic-dgp.git
cd forensic-dgp

# 1. Install the isolated environment and verify model SHA-256 pins (requires 64-bit Python 3.12 or 3.13)
.\install-integrated.bat

# 2. Launch the integrated restoration + reviewed completion workbench
.\start-integrated.bat
```
If the default port is busy, run `.\start-integrated.bat --port 0`. See [`INTEGRATED_APP_ON_MAIN.md`](INTEGRATED_APP_ON_MAIN.md) and [`integrated_app/README.MD`](integrated_app/README.MD) for details.

---

### Option C (Research Workbench): Original DGP Multi-Loss Research UI

To run the original `Phase 1–5` / `V6–V42` research FastAPI interface at the repository root (`checkpoints/dgp_improved_epoch_20.pth` tracked in-repo):

```powershell
# Windows One-Click Setup & Launch
.\install.bat
.\start.bat

# Or Manual Terminal Launch
python -m venv venv
.\venv\Scripts\activate
pip install -r requirements.txt
uvicorn app:app --reload --host 127.0.0.1 --port 8000
```
Open **`http://127.0.0.1:8000`** in your browser and test with images from [`test_images/`](test_images/).

---

## 🧠 Core Architecture & Two-Stage Forensic Pipeline

### 1. Stage 1 — Blind CCTV Face Restoration (`Conditioned SFT V2 / p2_u0224`)
- **Backbone**: `787` frozen tensors from the `u64` role-preserving CodeFormer restoration checkpoint (`codeformer_dgp_sft_u64_s42_best.pth`, `vqgan_codebook.pth`, `parsing_parsenet.pth`, `detection_Resnet50_Final.pth`).
- **Exposure-Matched Conditioned SFT (`v2_p2_u0224_heads.npz`)**: Modulates decoder SFT affine layers from input luminance/exposure statistics (`g_dark`) without altering clear portraits (`56/56` inherited forensic retention gates passed, `-37.4%` 4-role degraded `MAE`, `+0.02062` `SSIM7`).
- **Strict Separation**: Restoration never hallucinates over occluded masks; its output serves as the exact reference canvas for optional Stage 2 completion.

### 2. Stage 2 — Reviewed Covering Segmentation & CodeFormer Completion R2 (`v3`)
- **Proposal-Only Segmenter (`covering_detector_segmenter_v1.pth`)**: Compact `GatedUNet(3, 1, 24)` (`2,681,549` bytes, SHA-256 `ccc22d524073368bc4f5d6b6d2136c90d0e2a117083e4c75fa98d42d7bfff6cf`) proposes candidate occlusion regions (`0` false-positive pixels on uncovered controls).
- **Mandatory Human Review Gate**: Investigators inspect, paint, erase, or clear the mask and must explicitly confirm the reviewed mask before completion can execute. Empty confirmed masks bypass neural completion (`0` neural calls).
- **Trained CodeFormer Completion R2 (`codeformer_completion_r2.pth`)**: Jointly adapts `332` tensors (`164` encoder + `114` transformer predictor + `54` SFT fusion) while freezing `165` VQGAN codebook/decoder tensors (`370,773,572` bytes, SHA-256 `ed9f92baf06a73090343ad00cdc23b8493c9dde738a90cde9bdb650e69fee07c`).
- **Exact Outside-Support Preservation**: Composited strictly inside the confirmed mask with `0` extra feathering margin outside reviewed support (`44,001,540 / 44,001,540` exact outside-mask RGB bytes verified across all `256` benchmark PNGs).

### 3. Foundational Blind Signal Pre-Conditioning & Morphological DGP (`degradation.py` & `models/`)
- **Blind Signal Pre-Conditioning**: Comb-line de-interlacing (NTSC/PAL 480i/576i), adaptive mosaic dissolution, and sensor noise variance ($\sigma$) estimation.
- **Universal 3-Tier Canonical Face Registration**: Tier 1 5-point FAN alignment (inter-ocular distance $\approx 66\text{ px}$), Tier 2 anti-aliased pre-filter fallback, and Tier 3 aspect-ratio center-crop fallback.
- **Composite Forensic Loss**:
  $$\mathcal{L}_{\text{total}} = \mathcal{L}_{1} + \lambda_{\text{perceptual}}\mathcal{L}_{\text{VGG}} + \lambda_{\text{FAN}}\mathcal{L}_{\text{FAN}}$$

---

## 📊 Ratified Model Checkpoints & Release Inventory

| Stage / Subsystem | Ratified Checkpoint / Artifact | Size (Bytes) | SHA-256 Fingerprint | Verification Benchmark |
| :--- | :--- | :--- | :--- | :--- |
| **Windows Offline Installer (`v0.2.2`)** | [`ForensicDGP_Setup-0.2.2.exe`](https://forensic-dgp.pages.dev/) | `1,045,576,043` | `bc9952a5e54a3955a06a7e3b636d008f8dee2f227ef470224ee927463d733a25` | `391` installer tests, `132` app tests, `34` JS tests, native `0.2.1 -> 0.2.2` upgrade |
| **Stage 1 Restoration (`Conditioned SFT V2`)** | `v2_p2_u0224_heads.npz` + `codeformer_dgp_sft_u64_s42_best.pth` | `12,188` + `376,619,866` | `1aee2b840397925049a2c821edd7395c3a9e4b48ce1552e06c5ffd627a33e31f` | `56/56` forensic retention gates, `-37.4%` degraded `MAE`, `+22.1%–47.1%` CCTV sharpness |
| **Stage 2 Completion (`Unified R2 v3`)** | `codeformer_completion_r2.pth` | `370,773,572` | `ed9f92baf06a73090343ad00cdc23b8493c9dde738a90cde9bdb650e69fee07c` | `60/60` numeric ratchet gates (`-1.05%` clear hole $L_1$, `-2.69%` challenge hole $L_1$, `-10.32%` gray chroma) |
| **Stage 2 Proposal Segmenter (`v1`)** | `covering_detector_segmenter_v1.pth` | `2,681,549` | `ccc22d524073368bc4f5d6b6d2136c90d0e2a117083e4c75fa98d42d7bfff6cf` | `0` false-positive pixels on uncovered control (`08_uncovered_degraded`); proposal-only |
| **Legacy DGP Phase 2 / Phase 3** | `checkpoints/dgp_improved_epoch_20.pth` | Tracked in-repo | In-repo baseline | `25.01 dB` PSNR, `0.7862` SSIM on $128\times 128$ FFHQ + Asian Face curriculum |

---

## 📁 Repository Structure

```text
├── README.MD                        # Primary overview, https://forensic-dgp.pages.dev/ installer guide, and architecture
├── ORIGINS_RESTORATION.md           # Thesis-style chronological origins of the CCTV Face Restoration model
├── ORIGINS_FACE_MASK_COMPLETION.md  # Thesis-style chronological origins of the Face Mask Completion model
├── PROJECT_HANDOFF.md               # Master cross-folder engineering handoff and milestone ledger
├── WORKSPACE_STATUS.md              # Branch, worktree, and model ratification status summary
├── INTEGRATED_APP_ON_MAIN.md        # Guide for running the V2 + R2 integrated application on main
├── DESIGN.md / PRODUCT.md           # UI/UX design system and product specifications
├── integrated_app/                  # Production V2 Restoration + Reviewed R2 Completion application (mirrors IMPLEMENTATION)
├── packaging/                       # Windows Inno Setup (v0.2.2) installer scripts, C# launcher, and verification reports
├── download-site/                   # Cloudflare Pages source for https://forensic-dgp.pages.dev/
├── docs/                            # Thesis manuscript and categorized historical reports (docs/archive/)
│   └── archive/                     # 661 archived milestone reports (restoration/, completion/, dgp_legacy/, vm_ops/)
├── research/                        # Modular experiment packages (Conditioned SFT V1/V2, Completion R1/R2 v1-v3, probes)
├── scripts/                         # Training, VM packet builders, return auditors, and evaluation utilities
├── models/                          # DGP Synthesizer, MobileNetV2 FPN, and FAN composite loss modules
├── checkpoints/                     # In-repo baseline DGP weights (dgp_improved_epoch_20.pth)
├── test_images/                     # Synthetic CCTV test crops (synthetic_data_1-10.png) & real scenes (wild1-5.png)
├── tests/                           # Root research verification suite
├── install-integrated.bat           # One-click setup for integrated_app/ (Conditioned SFT V2 + Completion R2)
├── start-integrated.bat             # One-click launcher for integrated_app/
├── install.bat                      # One-click setup for the original DGP research UI
└── start.bat                        # One-click launcher for the original DGP research UI
```

---

## ⚖️ Legal Admissibility, Forensic Epistemology & Chain-of-Custody

This system is engineered in strict accordance with the **Philippine Rules on Electronic Evidence** (A.M. No. 01-7-01-SC):
1. **Strict Separation of Restoration vs. Completion**: Stage 1 (`Conditioned SFT V2`) restores unoccluded degraded pixels without inventing occluded anatomy; Stage 2 (`CodeFormer Completion R2`) synthesizes plausible facial structure strictly inside human-confirmed occlusion masks and is explicitly labeled as an **investigative lead / plausible estimate, never biometric proof of hidden identity**.
2. **Exact Non-Mask Pixel Integrity**: Every completion output preserves `100%` of RGB bytes outside the confirmed mask exactly as produced by Stage 1 restoration, accompanied by an auditable cryptographic evidence bundle (`.zip` with input, restored canvas, binary mask, result, and model/protocol SHA-256 receipts).
3. **100% Local Air-Gapped Execution**: Both the Windows offline installer ([https://forensic-dgp.pages.dev/](https://forensic-dgp.pages.dev/)) and the local workbench run entirely on the investigator's machine without transmitting surveillance frames to external cloud APIs.
