"""Deterministic input-only degradation of reviewed real detector training samples.

This module contains no torch, model or optimizer. It never changes labels,
support, crop geometry or train/validation membership.
"""
import math

import cv2
import numpy as np


def camera_pair(rgb, mask, valid, source_valid, seed):
    if (rgb.dtype != np.uint8 or rgb.ndim != 3 or rgb.shape[2] != 3
            or any(a.dtype != np.bool_ or a.shape != rgb.shape[:2] for a in (mask, valid, source_valid))
            or not valid.any() or (mask & ~valid).any() or (valid & ~source_valid).any()
            or not np.all(rgb[~source_valid] == 96)
            or type(seed) is not int or not 0 <= seed < 2**32):
        raise ValueError("Reviewed RGB/binary support and uint32 seed required")
    yy, xx = np.where(source_valid)
    x0, x1, y0, y1 = int(xx.min()), int(xx.max()) + 1, int(yy.min()), int(yy.max()) + 1
    if source_valid.sum() != (x1 - x0) * (y1 - y0):
        raise ValueError("Real camera pair requires rectangular source support; no unobserved pixels may enter filters")
    rng = np.random.default_rng(seed)
    sigma = float(rng.uniform(.6, 1.4))
    low_side = int(rng.choice([128, 160, 192]))
    noise_sigma = float(rng.uniform(1, 6))
    jpeg_quality = int(rng.integers(65, 96))
    roi = rgb[y0:y1, x0:x1].copy()
    height, width = roi.shape[:2]
    scale = low_side / max(rgb.shape[:2])
    low_width, low_height = max(1, round(width * scale)), max(1, round(height * scale))
    if not all(math.isfinite(n) for n in (sigma, noise_sigma)):
        raise ValueError("Finite camera parameters required")
    # Filter only observed source RGB, using a reflected image boundary. Neutral
    # padding never contaminates the photo or needs a new ignored-loss border.
    blurred = cv2.GaussianBlur(roi, (9, 9), sigma, borderType=cv2.BORDER_REFLECT_101)
    low = cv2.resize(blurred, (low_width, low_height), interpolation=cv2.INTER_AREA)
    resized = cv2.resize(low, (width, height), interpolation=cv2.INTER_LINEAR)
    noisy = np.clip(np.rint(resized.astype(float) + rng.normal(0, noise_sigma, roi.shape)), 0, 255).astype(np.uint8)
    ok, encoded = cv2.imencode(".jpg", cv2.cvtColor(noisy, cv2.COLOR_RGB2BGR),
                               [cv2.IMWRITE_JPEG_QUALITY, jpeg_quality])
    if not ok:
        raise ValueError("Camera JPEG encode failed")
    degraded_roi = cv2.cvtColor(cv2.imdecode(encoded, cv2.IMREAD_COLOR), cv2.COLOR_BGR2RGB)
    result = rgb.copy()
    result[y0:y1, x0:x1] = degraded_roi
    if np.any(result[~source_valid] != 96):
        raise ValueError("Camera transform changed true padding")
    return {"input": result, "mask": mask.copy(), "valid": valid.copy(), "source_valid": source_valid.copy(),
            "camera": {"seed": seed, "source_roi_xyxy": [x0, y0, x1, y1], "blur_sigma": sigma,
                       "blur_kernel": 9, "low_full_side": low_side, "low_roi_size": [low_width, low_height],
                       "noise_sigma_255": noise_sigma, "jpeg_quality": jpeg_quality,
                       "padding_in_filtered_source": False, "target_or_support_changed": False,
                       "spatial_warp": False}}
