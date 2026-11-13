"""Tightly gated color-cast white balance and face-safe dark-channel dehazing (D1-Tight)."""
from __future__ import annotations

import math
import time

import cv2
import numpy as np

POLICY = "damped-sog-face-safe-dcp-tight-v1"
MIN_SIDE = 16
CAST_D_MIN = 22.0
CAST_K_MIN = 1.50
HAZE_DC_MIN = 0.35
HAZE_SAT_MAX = 45.0


def _finite_float(value: float, default: float = 0.0) -> float:
    fval = float(value)
    return fval if math.isfinite(fval) else default


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
    if mask is None:
        return np.ones(shape_hw, dtype=bool)
    arr = np.asarray(mask)
    if arr.shape[:2] != shape_hw:
        return None
    return arr > 0


def photometric_metrics(rgb: np.ndarray, mask: np.ndarray | None = None) -> dict[str, float]:
    """Compute JSON-safe dark-channel, luma, HSV saturation, and CIELAB cast metrics."""
    src = _validate_rgb(rgb)
    if src is None:
        return {
            "dark_channel_mean": 0.0,
            "luma_mean": 0.0,
            "luma_std": 0.0,
            "sat_mean": 0.0,
            "cast_D": 0.0,
            "cast_factor": 0.0,
            "clip_shadow_pct": 0.0,
            "clip_high_pct": 0.0,
        }
    active = _resolve_active_mask(src.shape[:2], mask)
    if active is None or not bool(active.any()):
        active = np.ones(src.shape[:2], dtype=bool)

    f = src.astype(np.float32)
    y = 0.299 * f[..., 0] + 0.587 * f[..., 1] + 0.114 * f[..., 2]
    k_sz = max(5, min(15, (min(src.shape[:2]) // 10) | 1))
    dark_ch = cv2.erode(src.min(axis=2), np.ones((k_sz, k_sz), np.uint8))
    hsv = cv2.cvtColor(src, cv2.COLOR_RGB2HSV)
    lab = cv2.cvtColor(src, cv2.COLOR_RGB2LAB).astype(np.float32)
    a_vals = lab[..., 1][active] - 128.0
    b_vals = lab[..., 2][active] - 128.0
    cast_d = float(np.hypot(a_vals.mean(), b_vals.mean()))
    cast_m = float(np.hypot(a_vals.std(), b_vals.std()))
    cast_k = cast_d / max(cast_m, 1e-6)
    y_act = y[active]
    return {
        "dark_channel_mean": round(_finite_float(float(dark_ch[active].mean())), 2),
        "luma_mean": round(_finite_float(float(y_act.mean())), 2),
        "luma_std": round(_finite_float(float(y_act.std())), 2),
        "sat_mean": round(_finite_float(float(hsv[..., 1][active].mean())), 2),
        "cast_D": round(_finite_float(cast_d), 2),
        "cast_factor": round(_finite_float(cast_k), 3),
        "clip_shadow_pct": round(_finite_float(float((src.min(axis=2)[active] <= 2).mean() * 100.0)), 2),
        "clip_high_pct": round(_finite_float(float((src.max(axis=2)[active] >= 253).mean() * 100.0)), 2),
    }


def _fast_guided_filter(
    guide: np.ndarray,
    src: np.ndarray,
    r: int,
    eps: float,
    sub: int = 2,
) -> np.ndarray:
    """Pure-OpenCV fast guided filter using ``cv2.blur`` box filtering."""
    h, w = guide.shape[:2]
    if sub > 1 and min(h, w) >= 64:
        sw, sh = max(16, w // sub), max(16, h // sub)
        g_s = cv2.resize(guide, (sw, sh), interpolation=cv2.INTER_AREA)
        p_s = cv2.resize(src, (sw, sh), interpolation=cv2.INTER_AREA)
        rs = max(2, r // sub)
    else:
        g_s, p_s, rs = guide, src, r
    k = (2 * rs + 1, 2 * rs + 1)
    mi = cv2.blur(g_s, k)
    mp = cv2.blur(p_s, k)
    ci = cv2.blur(g_s * g_s, k)
    cip = cv2.blur(g_s * p_s, k)
    vi = ci - mi * mi
    cov = cip - mi * mp
    a = cov / (vi + eps)
    b = mp - a * mi
    ma = cv2.blur(a, k)
    mb = cv2.blur(b, k)
    if sub > 1 and min(h, w) >= 64:
        ma = cv2.resize(ma, (w, h), interpolation=cv2.INTER_LINEAR)
        mb = cv2.resize(mb, (w, h), interpolation=cv2.INTER_LINEAR)
    return ma * guide + mb


def _damped_shades_of_gray(
    img_f: np.ndarray,
    alpha: float = 0.55,
    mask: np.ndarray | None = None,
) -> np.ndarray:
    """Minkowski p=6 Shades-of-Gray white balance damped by ``alpha`` for skin-tone safety."""
    eps = 1e-4
    p = 6.0
    pixels = img_f[mask] if (mask is not None and bool(mask.any())) else img_f.reshape(-1, 3)
    norm_c = np.power(np.mean(np.power(np.maximum(pixels, eps), p), axis=0), 1.0 / p)
    mean_norm = float(norm_c.mean())
    gains = np.clip(mean_norm / np.maximum(norm_c, eps), 0.65, 1.50)
    balanced = img_f * gains[None, None, :]
    out = img_f * (1.0 - alpha) + balanced * alpha
    return np.clip(out, 0.0, 1.0)


def _face_safe_dcp(
    img_f: np.ndarray,
    omega: float = 0.68,
    t0: float = 0.28,
) -> np.ndarray:
    """Face-safe dark-channel dehazing with fast guided filter and highlight rolloff."""
    h, w = img_f.shape[:2]
    k_erode = max(5, min(15, (min(h, w) // 10) | 1))
    min_rgb = img_f.min(axis=2)
    dark = cv2.erode(min_rgb, np.ones((k_erode, k_erode), np.uint8))
    flat_dark = dark.ravel()
    top_k = max(1, int(math.ceil(flat_dark.size * 0.001)))
    idx = np.argpartition(flat_dark, -top_k)[-top_k:]
    airlight = np.clip(img_f.reshape(-1, 3)[idx].mean(axis=0), 0.55, 0.92)
    norm_dark = cv2.erode(
        (img_f / airlight[None, None, :]).min(axis=2),
        np.ones((k_erode, k_erode), np.uint8),
    )
    t_rough = 1.0 - omega * norm_dark
    gray = (0.299 * img_f[..., 0] + 0.587 * img_f[..., 1] + 0.114 * img_f[..., 2]).astype(np.float32)
    r_gf = max(6, min(30, min(h, w) // 5))
    t_ref = _fast_guided_filter(
        gray,
        t_rough.astype(np.float32),
        r=r_gf,
        eps=1e-3,
        sub=2 if min(h, w) >= 128 else 1,
    )
    t_safe = np.clip(t_ref, t0, 1.0)
    # Smooth highlight shoulder keeps bright whites from clipping after atmospheric stretch.
    hi = np.clip((gray - 0.72) / 0.23, 0.0, 1.0)
    hi = hi * hi * (3.0 - 2.0 * hi)
    t_safe = (t_safe + hi * (1.0 - t_safe))[..., None]
    recovered = (img_f - airlight[None, None, :]) / t_safe + airlight[None, None, :]
    return np.clip(recovered, 0.0, 1.0)


def _evaluate_gates(pm: dict[str, float]) -> tuple[bool, bool]:
    cast_fire = (pm["cast_D"] >= CAST_D_MIN) and (pm["cast_factor"] >= CAST_K_MIN)
    haze_fire = (pm["dark_channel_mean"] / 255.0 >= HAZE_DC_MIN) and (pm["sat_mean"] < HAZE_SAT_MAX)
    return bool(cast_fire), bool(haze_fire)


def _route_for_gates(cast_fire: bool, haze_fire: bool) -> str | None:
    if cast_fire and haze_fire:
        return "cast_dehaze"
    if cast_fire:
        return "color_cast_balance"
    if haze_fire:
        return "haze_dehaze"
    return None


def detect_cast_dehaze(rgb: np.ndarray, mask: np.ndarray | None = None) -> dict | None:
    """Evaluate tightened D1-Tight color-cast and haze gates without altering pixels."""
    src = _validate_rgb(rgb)
    if src is None:
        return None
    active = _resolve_active_mask(src.shape[:2], mask)
    if active is None or int(active.sum()) < 64:
        return None
    pm = photometric_metrics(src, mask=active if mask is not None else None)
    cast_fire, haze_fire = _evaluate_gates(pm)
    route = _route_for_gates(cast_fire, haze_fire)
    if route is None:
        return None
    return {
        "policy": POLICY,
        "triggered": True,
        "route": route,
        "gates": {"cast": cast_fire, "haze": haze_fire},
        "before": pm,
    }


def prepare_cast_dehaze(
    rgb: np.ndarray,
    mask: np.ndarray | None = None,
    force_mode: str | None = None,
) -> tuple[np.ndarray | None, np.ndarray | None, str | None, dict]:
    """Run D1-Tight color-cast white balance and/or face-safe dark-channel dehazing.

    Args:
        rgb: Native uint8 RGB image of shape (H, W, 3).
        mask: Optional boolean/uint8 active-ROI mask; pixels outside stay bit-exact.
        force_mode: Optional diagnostic override (``"cast"``, ``"haze"``, or ``"both"``).

    Returns:
        ``(prepared_rgb, binary_mask_u8, route, metrics)`` where ``route`` is one of
        ``"cast_dehaze"``, ``"color_cast_balance"``, ``"haze_dehaze"`` (or ``None`` when
        not triggered) and ``metrics`` is strictly finite and JSON-safe.
    """
    t0 = time.perf_counter()
    src = _validate_rgb(rgb)
    if src is None:
        return (
            None,
            None,
            None,
            {
                "policy": POLICY,
                "triggered": False,
                "route": None,
                "reason": "invalid_input",
                "changed_native_pixels": 0,
                "ms": round((time.perf_counter() - t0) * 1000.0, 2),
            },
        )

    height, width = src.shape[:2]
    active = _resolve_active_mask((height, width), mask)
    if active is None or int(active.sum()) < 64:
        return (
            None,
            None,
            None,
            {
                "policy": POLICY,
                "triggered": False,
                "route": None,
                "reason": "insufficient_mask_area",
                "changed_native_pixels": 0,
                "ms": round((time.perf_counter() - t0) * 1000.0, 2),
            },
        )

    pm_before = photometric_metrics(src, mask=active if mask is not None else None)
    cast_fire, haze_fire = _evaluate_gates(pm_before)
    if force_mode == "cast":
        cast_fire = True
    elif force_mode == "haze":
        haze_fire = True
    elif force_mode == "both":
        cast_fire = True
        haze_fire = True

    route = _route_for_gates(cast_fire, haze_fire)
    if route is None:
        return (
            None,
            None,
            None,
            {
                "policy": POLICY,
                "triggered": False,
                "route": None,
                "gates": {"cast": False, "haze": False},
                "before": pm_before,
                "changed_native_pixels": 0,
                "masked": bool(mask is not None),
                "ms": round((time.perf_counter() - t0) * 1000.0, 2),
            },
        )

    img_f = src.astype(np.float32) / 255.0
    if cast_fire:
        img_f = _damped_shades_of_gray(img_f, alpha=0.55, mask=active if mask is not None else None)
    if haze_fire:
        img_f = _face_safe_dcp(img_f, omega=0.68, t0=0.28)

    out = np.clip(np.rint(img_f * 255.0), 0, 255).astype(np.uint8)
    if mask is not None:
        out[~active] = src[~active]

    edited = active & np.any(out != src, axis=2)
    changed = int(edited.sum())
    if changed == 0:
        return (
            None,
            None,
            None,
            {
                "policy": POLICY,
                "triggered": False,
                "route": None,
                "gates": {"cast": cast_fire, "haze": haze_fire},
                "before": pm_before,
                "changed_native_pixels": 0,
                "masked": bool(mask is not None),
                "ms": round((time.perf_counter() - t0) * 1000.0, 2),
            },
        )

    pm_after = photometric_metrics(out, mask=active if mask is not None else None)
    metrics = {
        "policy": POLICY,
        "triggered": True,
        "route": route,
        "gates": {"cast": cast_fire, "haze": haze_fire},
        "before": pm_before,
        "after": pm_after,
        "changed_native_pixels": changed,
        "masked": bool(mask is not None),
        "mask_fraction": round(_finite_float(float(edited.mean())), 6),
        "ms": round(_finite_float((time.perf_counter() - t0) * 1000.0), 2),
    }
    mask_u8 = edited.astype(np.uint8) * 255
    return out, mask_u8, route, metrics
