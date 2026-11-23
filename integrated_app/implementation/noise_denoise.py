"""Native-scale Y/Cr/Cb Immerkær noise detection and independent NLM denoising."""
from __future__ import annotations

import math
import time

import cv2
import numpy as np

from .quality import (
    NATIVE_CHROMA_NOISE_THRESHOLD,
    NATIVE_Y_NOISE_THRESHOLD,
    native_ycrcb_noise_sigma,
)

POLICY = "ycrcb-immerkaer-nlm-v1"
MIN_SIDE = 16


def _validate_rgb(rgb: np.ndarray) -> np.ndarray | None:
    if (
        not isinstance(rgb, np.ndarray)
        or rgb.ndim != 3
        or rgb.shape[2] != 3
        or rgb.dtype != np.uint8
    ):
        return None
    if min(rgb.shape[:2]) < MIN_SIDE:
        return None
    return rgb


def _resolve_active_mask(
    shape_hw: tuple[int, int],
    mask: np.ndarray | None,
) -> np.ndarray | None:
    """Resolve optional active-ROI mask (True/nonzero = denoise) or full-frame mask."""
    if mask is None:
        return np.ones(shape_hw, dtype=bool)
    arr = np.asarray(mask)
    if arr.shape[:2] != shape_hw:
        return None
    return arr > 0


def _edge_retention_ratio(before_rgb: np.ndarray, after_rgb: np.ndarray, active: np.ndarray) -> float:
    """Measure gradient magnitude retention on the top 15% strongest edges inside `active`."""
    g0 = cv2.cvtColor(before_rgb, cv2.COLOR_RGB2GRAY).astype(np.float32)
    g1 = cv2.cvtColor(after_rgb, cv2.COLOR_RGB2GRAY).astype(np.float32)
    mag0 = np.hypot(cv2.Sobel(g0, cv2.CV_32F, 1, 0), cv2.Sobel(g0, cv2.CV_32F, 0, 1))
    mag1 = np.hypot(cv2.Sobel(g1, cv2.CV_32F, 1, 0), cv2.Sobel(g1, cv2.CV_32F, 0, 1))
    vals0 = mag0[active]
    if vals0.size == 0:
        return 1.0
    thresh = max(float(np.percentile(vals0, 85)), 10.0)
    edges = active & (mag0 >= thresh)
    if not bool(edges.any()):
        return 1.0
    ratio = float(mag1[edges].mean() / max(float(mag0[edges].mean()), 1e-6))
    return round(ratio, 4) if math.isfinite(ratio) else 1.0


def _finite_float(value: float, default: float = 0.0) -> float:
    fval = float(value)
    return fval if math.isfinite(fval) else default


def detect_noise(rgb: np.ndarray, mask: np.ndarray | None = None) -> dict | None:
    """Detect native-scale luma or chroma sensor noise without 256px upsampling.

    Args:
        rgb: Native uint8 RGB image of shape (H, W, 3).
        mask: Optional boolean/uint8 active-ROI mask (nonzero = inspected region).

    Returns:
        JSON-safe detection metrics dict when gated, else ``None``.
    """
    src = _validate_rgb(rgb)
    if src is None:
        return None
    active = _resolve_active_mask(src.shape[:2], mask)
    if active is None or int(active.sum()) < 64:
        return None
    excl = (~active).astype(np.uint8) if mask is not None else None
    sig_y, sig_cr, sig_cb = native_ycrcb_noise_sigma(src, mask=excl)
    triggered = (sig_y >= NATIVE_Y_NOISE_THRESHOLD) or (
        max(sig_cr, sig_cb) >= NATIVE_CHROMA_NOISE_THRESHOLD
    )
    if not triggered:
        return None
    return {
        "policy": POLICY,
        "triggered": True,
        "sigma_before": [_finite_float(sig_y), _finite_float(sig_cr), _finite_float(sig_cb)],
        "thresholds": {"y": NATIVE_Y_NOISE_THRESHOLD, "chroma": NATIVE_CHROMA_NOISE_THRESHOLD},
        "masked": bool(mask is not None),
        "mask_fraction": round(_finite_float(float(active.mean())), 6),
    }


def apply_noise_denoise(
    rgb: np.ndarray,
    mask: np.ndarray | None = None,
    force: bool = False,
) -> tuple[np.ndarray, np.ndarray, dict]:
    """Run native-scale Y/Cr/Cb Immerkær gate and independent NLM denoising.

    Args:
        rgb: Native uint8 RGB image of shape (H, W, 3).
        mask: Optional boolean/uint8 active-ROI mask; pixels outside stay bit-exact.
        force: Bypass noise gate when ``True`` for diagnostic evaluation.

    Returns:
        ``(prepared_rgb, binary_mask_u8, metrics)`` where ``binary_mask_u8`` is uint8
        (0/255) over edited pixels and ``metrics`` is strictly finite and JSON-safe.
    """
    t0 = time.perf_counter()
    src = _validate_rgb(rgb)
    if src is None:
        h, w = (rgb.shape[:2] if isinstance(rgb, np.ndarray) and rgb.ndim >= 2 else (0, 0))
        empty_rgb = rgb.copy() if isinstance(rgb, np.ndarray) else np.zeros((0, 0, 3), dtype=np.uint8)
        return (
            empty_rgb,
            np.zeros((h, w), dtype=np.uint8),
            {
                "policy": POLICY,
                "triggered": False,
                "reason": "invalid_input",
                "changed_native_pixels": 0,
                "ms": round((time.perf_counter() - t0) * 1000.0, 2),
            },
        )

    height, width = src.shape[:2]
    active = _resolve_active_mask((height, width), mask)
    if active is None or int(active.sum()) < 64:
        return (
            src.copy(),
            np.zeros((height, width), dtype=np.uint8),
            {
                "policy": POLICY,
                "triggered": False,
                "reason": "insufficient_mask_area",
                "changed_native_pixels": 0,
                "ms": round((time.perf_counter() - t0) * 1000.0, 2),
            },
        )

    excl = (~active).astype(np.uint8) if mask is not None else None
    sig_y, sig_cr, sig_cb = native_ycrcb_noise_sigma(src, mask=excl)
    triggered = (sig_y >= NATIVE_Y_NOISE_THRESHOLD) or (
        max(sig_cr, sig_cb) >= NATIVE_CHROMA_NOISE_THRESHOLD
    )
    if not triggered and not force:
        return (
            src.copy(),
            np.zeros((height, width), dtype=np.uint8),
            {
                "policy": POLICY,
                "triggered": False,
                "sigma_before": [_finite_float(sig_y), _finite_float(sig_cr), _finite_float(sig_cb)],
                "h_params": [0.0, 0.0, 0.0],
                "changed_native_pixels": 0,
                "masked": bool(mask is not None),
                "mask_fraction": round(_finite_float(float(active.mean())), 6),
                "ms": round((time.perf_counter() - t0) * 1000.0, 2),
            },
        )

    h_y = float(np.clip(0.85 * sig_y, 3.0, 12.0)) if (sig_y >= 2.0 or force) else 0.0
    h_cr = float(np.clip(1.4 * sig_cr, 5.0, 18.0)) if (sig_cr >= 2.0 or force) else 0.0
    h_cb = float(np.clip(1.4 * sig_cb, 5.0, 18.0)) if (sig_cb >= 2.0 or force) else 0.0

    ycrcb = cv2.cvtColor(src, cv2.COLOR_RGB2YCrCb)
    y_ch, cr_ch, cb_ch = cv2.split(ycrcb)
    if h_y > 0.0:
        y_ch = cv2.fastNlMeansDenoising(y_ch, None, h=h_y, templateWindowSize=7, searchWindowSize=21)
    if h_cr > 0.0:
        cr_ch = cv2.fastNlMeansDenoising(cr_ch, None, h=h_cr, templateWindowSize=7, searchWindowSize=21)
    if h_cb > 0.0:
        cb_ch = cv2.fastNlMeansDenoising(cb_ch, None, h=h_cb, templateWindowSize=7, searchWindowSize=21)

    out = cv2.cvtColor(cv2.merge((y_ch, cr_ch, cb_ch)), cv2.COLOR_YCrCb2RGB)
    if mask is not None:
        out[~active] = src[~active]

    edited = active & np.any(out != src, axis=2)
    changed = int(edited.sum())
    after_y, after_cr, after_cb = native_ycrcb_noise_sigma(out, mask=excl)
    edge_ret = _edge_retention_ratio(src, out, active)
    ms = round(_finite_float((time.perf_counter() - t0) * 1000.0), 2)

    metrics = {
        "policy": POLICY,
        "triggered": True,
        "forced": bool(force and not triggered),
        "sigma_before": [_finite_float(sig_y), _finite_float(sig_cr), _finite_float(sig_cb)],
        "sigma_after": [_finite_float(after_y), _finite_float(after_cr), _finite_float(after_cb)],
        "h_params": [round(_finite_float(h_y), 3), round(_finite_float(h_cr), 3), round(_finite_float(h_cb), 3)],
        "edge_retention": _finite_float(edge_ret, 1.0),
        "changed_native_pixels": changed,
        "masked": bool(mask is not None),
        "mask_fraction": round(_finite_float(float(edited.mean())), 6),
        "ms": ms,
    }
    mask_u8 = edited.astype(np.uint8) * 255
    return out, mask_u8, metrics


def prepare_noise_denoise(
    rgb: np.ndarray,
    mask: np.ndarray | None = None,
    force: bool = False,
) -> tuple[np.ndarray | None, np.ndarray | None, dict]:
    """Prepare denoised RGB when the native YCrCb Immerkær gate fires.

    Returns:
        ``(prepared_rgb, binary_mask_u8, metrics)`` when triggered and at least one pixel
        changes; otherwise ``(None, None, metrics)`` with JSON-safe diagnostics.
    """
    out, mask_u8, metrics = apply_noise_denoise(rgb, mask=mask, force=force)
    if not metrics.get("triggered") or int(metrics.get("changed_native_pixels", 0)) == 0:
        metrics["triggered"] = False
        return None, None, metrics
    return out, mask_u8, metrics
