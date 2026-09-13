# Forensic CCTV Benchmark Test Images

This directory contains real-world and synthetic degraded facial crops used for testing and validating the **Morphologically Constrained Deep Generative Prior (DGP)** model.

## 1. Synthetic Data (Database-Generated CCTV Benchmarks)

These 10 images simulate authentic physical camera and transmission degradations applied to facial database samples:

- `synthetic_data_1.png`: Severe optical motion blur + low photon count + H.264 compression.
- `synthetic_data_2.png`: Tropical atmospheric humidity scattering + sensor thermal noise.
- `synthetic_data_3.png`: High directional motion blur ($d=8\text{ px}, \theta=35^{\circ}$) + codec quantization.
- `synthetic_data_4.png`: Afternoon atmospheric haze + low-contrast backlight + H.264 ($Q_p=35$).
- `synthetic_data_5.png`: Distant pole-mounted camera ($28\times 28$ sub-32 spatial downsampling).
- `synthetic_data_6.png`: Nighttime low-light surveillance (high ISO sensor thermal noise $\sigma=14$).
- `synthetic_data_7.png`: Severely bandwidth-starved NVR compression (coarse H.264 macroblocking).
- `synthetic_data_8.png`: Analog coaxial CCTV NTSC/PAL comb interlacing scanlines.
- `synthetic_data_9.png`: Defocus lens aberration + high-frequency sensor noise.
- `synthetic_data_10.png`: Compound worst-case CCTV degradation (motion + downsampling + thermal noise + compression).

## 2. Wild Data (Real Surveillance Scenes)

Full contextual CCTV frames captured under unconstrained municipal conditions in Zamboanga City:

- `wild1.png`: Wide-angle outdoor surveillance scene with distant pedestrian.
- `wild2.png`: High-motion transit frame afflicted with analog comb interlacing.
- `wild3.png`: Severe compression macroblocking and edge pixelation.
- `wild4.png`: Oblique perspective surveillance capture.
- `wild5.png`: Low-contrast, diffused surveillance portrait.

## Usage

### In Web Interface
1. Launch the Web UI (`start.bat` or `uvicorn app:app --reload`).
2. Open `http://127.0.0.1:8000`.
3. Drag and drop any file from `test_images/` into `01 // TARGET_EXTRACTION`.

### In Python Scripts
```python
import cv2

# Load synthetic benchmark
img_syn = cv2.imread("test_images/synthetic_data_1.png")

# Load real wild surveillance frame
img_wild = cv2.imread("test_images/wild1.png")
```
