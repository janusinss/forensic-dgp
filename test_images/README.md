# Forensic CCTV Benchmark Test Images

This directory contains real and simulated low-resolution, heavily degraded CCTV facial crops used for testing and validating the **Morphologically Constrained Deep Generative Prior (DGP)** model.

## Sample Files

- `test_cctv_blurred_1.png`: Low-resolution CCTV crop showing severe optical motion blur, low photon count, and H.264 compression artifacts.
- `test_cctv_blurred_2.png`: Municipal surveillance frame crop afflicted by tropical atmospheric haze/scattering and sensor thermal noise.

## Usage

### In Web Interface
1. Launch the Web UI (`start.bat` or `uvicorn app:app --reload`).
2. Open `http://localhost:8000`.
3. Upload any image from this folder to perform Top-K forensic face reconstruction.

### In Python Scripts
```python
import cv2

# Load sample image
img = cv2.imread("test_images/test_cctv_blurred_1.png")
```
