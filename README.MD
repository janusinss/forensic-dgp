# Forensic Deep Generative Prior (DGP) for CCTV Video Reconstruction

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

This repository contains the official software implementation of a **Morphologically Constrained Deep Generative Prior (DGP)** designed for forensic face reconstruction from severely degraded, real-world CCTV footage in Zamboanga City.

Traditional super-resolution and diffusion models often hallucinate facial geometry, rendering them inadmissible under the **Philippine Rules on Electronic Evidence**. This system enforces strict biometric preservation via a **68-point Morphological FAN Loss**, ensuring that reconstructed facial features remain mathematically anchored to the suspect's true Euclidean landmark coordinates.

---

## ⚡ Quick Start for Team Members (One-Click Setup)

Pre-trained model weights (**`checkpoints/dgp_improved_epoch_20.pth`**) are tracked directly in this repository. You can test the system immediately without retraining.

### 1. Clone the Repository
```bash
git clone https://github.com/janusinss/forensic-dgp.git
cd forensic-dgp
```

### 2. Environment Setup
#### Windows (Automated):
Double-click `install.bat` or run:
```powershell
python -m venv venv
.\venv\Scripts\activate
pip install -r requirements.txt
```

#### Linux / macOS:
```bash
python3 -m venv venv
source venv/bin/activate
pip install -r requirements.txt
```

### 3. Launch Web Interface
#### Windows (One-Click):
Double-click **`start.bat`**. This automatically activates the environment, launches your browser, and starts the FastAPI server.

#### Manual Terminal Command:
```bash
uvicorn app:app --reload --host 127.0.0.1 --port 8000
```
Open **`http://127.0.0.1:8000`** in your browser.

---

## 🔬 How to Test the Reconstruction Pipeline

1. In the Web UI, drag and drop any image from [`test_images/`](file:///test_images/):
   - **`test_images/synthetic_data_1.png` - `synthetic_data_10.png`**: Database synthetic CCTV faces (motion blur, atmospheric haze, downsampling, thermal noise, H.264 compression).
   - **`test_images/wild1.png` - `wild5.png`**: Real unconstrained municipal surveillance frames.
2. Select your **Reconstruction Mode**:
   - **`DIRECT_RESTORATION`**: Native CCTV scale restoration with illumination calibration and edge enhancement.
   - **`SUB-32x32_SUPER_RESOLUTION`**: Sub-32×32 thesis benchmark simulation (simulates distant CCTV cameras).
3. Click **`INITIATE_RECONSTRUCTION`**:
   - **Top-K Forensic Candidates**: View Rank 1 (Best Fit), Rank 2 (Edge Focus), and Rank 3 (Natural Tone) with Morphological FAN loss scores.
   - **Diagnostic Telemetry**: Displays real-time status of landmark registration (`CANONICAL_5POINT`), interlacing detection, mosaic detection, and noise sigma.
   - **In-Context Scene Reconstruction**: Displays the seamless inverse-affine ($M^{-1}$) paste-back of the restored face into the original wide surveillance context.

---

## 🧠 Core Architecture & Pipeline

### 1. Universal Blind Signal Pre-Conditioning (`degradation.py`)
- **Comb-Line De-Interlacing**: Blindly detects even/odd scanline energy from analog CCTV (NTSC/PAL 480i/576i) and dissolves comb artifacts.
- **Adaptive Mosaic Dissolution**: Detects step-edge compression blocks and applies anti-aliasing to melt pixelation before generative synthesis.
- **Sensor Denoising**: Estimates high-frequency noise variance ($\sigma$) and applies edge-preserving bilateral filtering.

### 2. Universal 3-Tier Canonical Face Registration
- **Tier 1 (Canonical 5-Point FAN)**: Aligns suspect pupils, nose tip, and mouth corners to standard FFHQ facial coordinates (inter-ocular distance $\approx 66\text{ px}$).
- **Tier 2 (Anti-Aliased Pre-Filter)**: Landmark extraction fallback on smoothed gradients for pixelated low-resolution faces.
- **Tier 3 (Center-Crop Geometry)**: Aspect-ratio bounding fallback for extreme unresolvable degradation.

### 3. Morphologically Constrained DGP Synthesizer (`models/`)
- **Backbone**: MobileNetV2 Feature Pyramid Network (FPN) with skip connections and reflection padding to eliminate checkerboard artifacts.
- **Composite Forensic Loss**:
  $$\mathcal{L}_{\text{total}} = \mathcal{L}_{1} + \lambda_{\text{perceptual}}\mathcal{L}_{\text{VGG}} + \lambda_{\text{FAN}}\mathcal{L}_{\text{FAN}}$$
  Where $\mathcal{L}_{\text{FAN}}$ calculates Euclidean distance across 68 facial landmarks.
- **Top-K Uncertainty Sampling**: Provides investigators with a ranked spread of probable facial structures rather than a single hallucinated output.

### 4. Seamless In-Context Paste-Back
- Computes the inverse affine matrix ($M^{-1}$) and blends the restored face back into the wide surveillance scene using elliptical boundary feathering.

---

## 📊 Training Curriculum & Checkpoints

| Phase | Epochs | Dataset | Degradation Model | Primary Checkpoint |
| :--- | :--- | :--- | :--- | :--- |
| **Phase 1** | 1 – 10 | FFHQ Thumbnails ($128\times 128$) | First-order Gaussian blur + $4\times$ downsampling | `checkpoints/dgp_improved_epoch_10.pth` |
| **Phase 2** | 11 – 20 | FFHQ Thumbnails ($128\times 128$) | Photometric refinement & Annealed LR | **`checkpoints/dgp_improved_epoch_20.pth`** *(Peak: 25.01 dB PSNR, 0.7862 SSIM)* |
| **Phase 3** | 21 – 26 | FFHQ + Asian Faces (10,000 images) | Second-order compound CCTV (haze, noise, compression, interlacing) | `checkpoints/dgp_improved_epoch_26.pth` |

---

## 📁 Repository Structure

```text
├── checkpoints/             # Pre-trained model weights (dgp_improved_epoch_20.pth included)
├── weights/                 # Initialization weights (mapped_deblurgan.pth)
├── dataset/                 # FFHQ and Asian demographic datasets
├── test_images/             # Synthetic CCTV test crops (synthetic_data_1-10.png) & real scenes (wild1-5.png)
├── outputs/                 # Restorations, comparison collages, and evaluation logs
├── tests/                   # Verification suite (test_model.py, test_pipeline.py)
├── models/                  # DGP Synthesizer, MobileNetV2 FPN, and FAN composite loss
├── scripts/                 # Utility scripts (download_asian_faces.py, run_phase3_gcp.sh, test_wild_batch.py)
├── docs/                    # Thesis manuscript and technical architecture documentation
├── static/                  # Web UI frontend assets (CSS, JS)
├── templates/               # Web UI layout (index.html)
├── app.py                   # FastAPI Forensic Web UI entrypoint
├── train.py                 # Multi-loss curriculum training script
├── dataloader.py            # PyTorch DataLoader pipeline
├── dataset.py               # DegradedFacesDataset with second-order CCTV degradation
├── degradation.py           # Blind signal detection, de-interlacing, de-mosaic, & canonical alignment
├── evaluation.py            # PSNR, SSIM, and ArcFace evaluation module
├── install.bat              # One-click Windows virtualenv setup
└── start.bat                # One-click Windows Web UI launcher
```

---

## ⚖️ Legal Admissibility & Chain-of-Custody

This system was created in strict accordance with the **Philippine Rules on Electronic Evidence** (A.M. No. 01-7-01-SC):
1. **Mathematical Admissibility**: Rather than synthesizing imaginative features, the DGP synthesizer is bound by landmark-preserving loss gradients.
2. **Top-K Transparency**: Multiple ranked variations are output alongside loss scores to indicate reconstruction uncertainty.
3. **Local Air-Gapped Execution**: Does not transmit biometric surveillance evidence to external cloud APIs, preventing data tampering and preserving legal chain of custody.
