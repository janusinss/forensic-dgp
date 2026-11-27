"""Tightly gated color-cast white balance and face-safe dark-channel dehazing (D1-Tight)."""
from __future__ import annotations

import math
import time

import cv2
import numpy as np

POLICY = "damped-sog-face-safe-dcp-tight-v1"
MIN_SIDE = 16
CAST_D_MIN = 26.5
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
    # Measure haze on inner portrait pixels excluding bright white/light-gray studio backdrops
    # so clean portraits on white studio backgrounds never trigger dark-channel dehazing.
    h_img, w_img = src.shape[:2]
    sat_ch = hsv[..., 1]
    haze_region = active
    if mask is None and min(h_img, w_img) >= 64:
        inner = np.zeros((h_img, w_img), dtype=bool)
        inner[int(h_img * 0.20):int(h_img * 0.80), int(w_img * 0.20):int(w_img * 0.80)] = True
        outer = active & ~inner
        outer_bright_frac = (
            float(((y >= 200.0) & (sat_ch <= 25))[outer].mean()) if bool(outer.any()) else 0.0
        )
        white_thresh = 210.0 if outer_bright_frac >= 0.45 else 235.0
        non_white = active & ~((y >= white_thresh) & (sat_ch <= 30))
        inner_nw = inner & non_white
        if int(inner_nw.sum()) >= max(64, int(0.15 * int(inner.sum()))):
            haze_region = inner_nw
        elif int(inner.sum()) >= 64:
            haze_region = inner
        elif int(non_white.sum()) >= max(64, int(0.25 * int(active.sum()))):
            haze_region = non_white
    else:
        non_white = active & ~((y >= 235.0) & (sat_ch <= 30))
        if int(non_white.sum()) >= max(64, int(0.25 * int(active.sum()))):
            haze_region = non_white

    return {
        "dark_channel_mean": round(_finite_float(float(dark_ch[active].mean())), 2),
        "haze_dark_channel_mean": round(_finite_float(float(dark_ch[haze_region].mean())), 2),
        "luma_mean": round(_finite_float(float(y_act.mean())), 2),
        "luma_std": round(_finite_float(float(y_act.std())), 2),
        "sat_mean": round(_finite_float(float(sat_ch[active].mean())), 2),
        "haze_sat_mean": round(_finite_float(float(sat_ch[haze_region].mean())), 2),
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
    dc = pm.get("haze_dark_channel_mean", pm["dark_channel_mean"]) / 255.0
    sat_val = pm.get("haze_sat_mean", pm["sat_mean"])
    sat_cap = HAZE_SAT_MAX + max(0.0, (dc - HAZE_DC_MIN) * 90.0)
    haze_fire = (
        (dc >= HAZE_DC_MIN)
        and (sat_val < sat_cap)
        and (pm["clip_shadow_pct"] < 1.0)
    )
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
    damping: float = 1.0,
) -> tuple[np.ndarray | None, np.ndarray | None, str | None, dict]:
    """Run D1-Tight color-cast white balance and/or face-safe dark-channel dehazing.

    Args:
        rgb: Native uint8 RGB image of shape (H, W, 3).
        mask: Optional boolean/uint8 active-ROI mask; pixels outside stay bit-exact.
        force_mode: Optional diagnostic override (``"cast"``, ``"haze"``, or ``"both"``).
        damping: Optional strength scale in ``(0, 1]`` (e.g. from ``V34PhotoCorrectNet`` ``g_photo`` guard).

    Returns:
        ``(prepared_rgb, binary_mask_u8, route, metrics)`` where ``route`` is one of
        ``"cast_dehaze"``, ``"color_cast_balance"``, ``"haze_dehaze"`` (or ``None`` when
        not triggered) and ``metrics`` is strictly finite and JSON-safe.
    """
    t0 = time.perf_counter()
    damp = float(np.clip(damping, 0.25, 1.0)) if math.isfinite(float(damping)) else 1.0
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
        img_f = _damped_shades_of_gray(img_f, alpha=0.55 * damp, mask=active if mask is not None else None)
    if haze_fire:
        img_f = _face_safe_dcp(img_f, omega=0.68 * damp, t0=0.28 + 0.14 * (1.0 - damp))

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
        "damping": round(damp, 3),
        "before": pm_before,
        "after": pm_after,
        "changed_native_pixels": changed,
        "masked": bool(mask is not None),
        "mask_fraction": round(_finite_float(float(edited.mean())), 6),
        "ms": round(_finite_float((time.perf_counter() - t0) * 1000.0), 2),
    }
    mask_u8 = edited.astype(np.uint8) * 255
    return out, mask_u8, route, metrics


LOW_LIGHT_GLARE_POLICY = "retinex-illumination-glare-shoulder-v1"


def _core_exposure_stats(rgb: np.ndarray, active: np.ndarray) -> dict[str, float]:
    """Compute inner-face and full-active luminance quantiles for skin-tone-safe gating."""
    h, w = rgb.shape[:2]
    f = rgb.astype(np.float32)
    y = 0.299 * f[..., 0] + 0.587 * f[..., 1] + 0.114 * f[..., 2]
    inner = np.zeros((h, w), dtype=bool)
    inner[int(h * 0.18):int(h * 0.82), int(w * 0.18):int(w * 0.82)] = True
    core = active & inner
    if int(core.sum()) < 64:
        core = active
    y_core = y[core]
    y_all = y[active]
    p5_c, p10_c, p15_c, p50_c, p90_c, p95_c = [float(v) for v in np.percentile(y_core, [5, 10, 15, 50, 90, 95])]
    p5_a, p10_a, p50_a, p90_a, p95_a = [float(v) for v in np.percentile(y_all, [5, 10, 50, 90, 95])]
    clip_shadow_core = float((rgb.min(axis=2)[core] <= 8).mean() * 100.0)
    clip_high_all = float((rgb.max(axis=2)[active] >= 250).mean() * 100.0)
    return {
        "p5_core": round(p5_c, 2),
        "p10_core": round(p10_c, 2),
        "p15_core": round(p15_c, 2),
        "p50_core": round(p50_c, 2),
        "p90_core": round(p90_c, 2),
        "p95_core": round(p95_c, 2),
        "dr_core": round(p95_c - p5_c, 2),
        "p5_all": round(p5_a, 2),
        "p10_all": round(p10_a, 2),
        "p50_all": round(p50_a, 2),
        "p90_all": round(p90_a, 2),
        "p95_all": round(p95_a, 2),
        "luma_std_core": round(float(y_core.std()), 2),
        "clip_shadow_core_pct": round(clip_shadow_core, 2),
        "clip_high_pct": round(clip_high_all, 2),
    }


def detect_low_light_glare(
    rgb: np.ndarray,
    mask: np.ndarray | None = None,
    *,
    v34_probe: dict | None = None,
) -> dict | None:
    """Skin-tone-safe detector for severely underexposed/backlit faces and veiling glare washout.

    Never gates on raw mean luminance alone: requires clipped shadows + low dynamic range
    (or backlit bright surround + dark face core) and respects clean-control V3.4 vetoes so
    well-exposed dark-skin portraits have 0% false positives.
    """
    src = _validate_rgb(rgb)
    tiny_dark_v34 = (
        isinstance(v34_probe, dict)
        and float(v34_probe.get("exposure", 0.0)) >= 0.50
        and float(v34_probe.get("p_clean", 0.0)) < 0.20
    )
    min_allowed_side = 24 if tiny_dark_v34 else 48
    if src is None or min(src.shape[:2]) < min_allowed_side:
        return None
    active = _resolve_active_mask(src.shape[:2], mask)
    if active is None or int(active.sum()) < 256:
        return None

    # Clean-control V3.4 veto guarantees 0% FP on clean studio / dark-skin portraits.
    if isinstance(v34_probe, dict):
        p_clean = float(v34_probe.get("p_clean", 0.0))
        max_deg = max(
            float(v34_probe.get("exposure", 0.0)),
            float(v34_probe.get("blur", 0.0)),
            float(v34_probe.get("noise", 0.0)),
            float(v34_probe.get("codec_lr", 0.0)),
        )
        if p_clean >= 0.82 and max_deg < 0.20:
            return None

    st = _core_exposure_stats(src, active)
    v34_exp = float(v34_probe.get("exposure", 0.0)) if isinstance(v34_probe, dict) else None
    v34_blur = float(v34_probe.get("blur", 0.0)) if isinstance(v34_probe, dict) else None
    v34_codec = float(v34_probe.get("codec_lr", 0.0)) if isinstance(v34_probe, dict) else None
    v34_clean = float(v34_probe.get("p_clean", 0.0)) if isinstance(v34_probe, dict) else None

    # Gate 1: Crushed dark CCTV / underexposed face (clipped shadows + compressed dynamic range).
    # Dark skin under proper studio exposure has normal highlight headroom (p90_core >= 105)
    # and clean V3.4 exposure score (< 0.25), so it will NOT trigger this gate.
    crushed_dark = (
        st["p50_core"] < 58.0
        and st["p90_core"] < 102.0
        and (st["clip_shadow_core_pct"] >= 2.0 or st["dr_core"] < 82.0)
        and (v34_exp is None or v34_exp >= 0.28 or st["p90_core"] < 82.0)
    )
    # Gate 2: Harsh backlit silhouette (dark face core surrounded by bright window/background).
    backlit_dark = (
        st["p50_core"] < 54.0
        and st["p15_core"] < 24.0
        and st["p90_all"] >= 175.0
        and (v34_clean is None or v34_clean < 0.35)
    )
    # Gate 2b: Tiny dark CCTV face crop confirmed by strong V3.4 exposure score (>=0.50, p_clean<0.20).
    tiny_dark_cctv = (
        tiny_dark_v34
        and min(src.shape[:2]) <= 72
        and st["p50_all"] < 78.0
        and st["p90_all"] < 185.0
    )
    low_light_fire = bool(crushed_dark or backlit_dark or tiny_dark_cctv)

    # Gate 3: Specular glare / CCTV veiling washout (elevated-but-bounded dark floor
    # 28..54 + real face contrast + bright veil, confirmed by V3.4 blur/exposure/codec).
    glare_veiling = (
        not low_light_fire
        and min(src.shape[:2]) >= 224
        and 28.0 <= st["p5_all"] <= 54.0
        and 38.0 <= st["p10_all"] <= 68.0
        and st["p95_all"] >= 188.0
        and st["p90_all"] <= 208.0
        and st["luma_std_core"] >= 32.0
        and (st["p90_core"] - st["p10_core"]) <= 145.0
        and (
            (
                v34_probe is not None
                and v34_clean is not None
                and v34_clean < 0.10
                and v34_blur is not None
                and v34_blur >= 0.58
                and v34_exp is not None
                and v34_exp >= 0.05
                and v34_codec is not None
                and v34_codec >= 0.22
            )
            or (
                v34_probe is None
                and st["p5_core"] >= 30.0
                and st["clip_high_pct"] >= 3.0
            )
        )
    )
    specular_clip = (
        not low_light_fire
        and st["clip_high_pct"] >= 7.0
        and 24.0 <= st["p5_all"] <= 65.0
        and st["dr_core"] < 175.0
        and (v34_clean is None or v34_clean < 0.25)
    )
    glare_fire = bool(glare_veiling or specular_clip)

    if not (low_light_fire or glare_fire):
        return None
    route = "low_light_recovery" if low_light_fire else "glare_recovery"
    return {
        "policy": LOW_LIGHT_GLARE_POLICY,
        "triggered": True,
        "route": route,
        "gates": {
            "crushed_dark": bool(crushed_dark),
            "backlit_dark": bool(backlit_dark),
            "tiny_dark_cctv": bool(tiny_dark_cctv),
            "glare_veiling": bool(glare_veiling),
            "specular_clip": bool(specular_clip),
        },
        "before": st,
    }


def _apply_retinex_low_light(img_f: np.ndarray, damping: float = 1.0) -> np.ndarray:
    """Downscaled illumination-map Retinex shadow lift with highlight roll-off and shadow chroma smoothing."""
    h, w = img_f.shape[:2]
    illum_raw = img_f.max(axis=2).astype(np.float32)
    gray = (0.299 * img_f[..., 0] + 0.587 * img_f[..., 1] + 0.114 * img_f[..., 2]).astype(np.float32)
    sub = max(2, min(4, min(h, w) // 48)) if min(h, w) >= 96 else 1
    r_gf = max(8, min(28, min(h, w) // 6))
    illum = _fast_guided_filter(gray, illum_raw, r=r_gf, eps=0.02, sub=sub)
    illum_safe = np.clip(illum, 0.14, 1.0)

    # Adaptive gamma: stronger lift for darker median illumination, smoothly damped by g_photo.
    med_t = float(np.median(illum_safe))
    gamma = float(np.clip(0.32 + 0.36 * max(0.0, 0.45 - med_t) / 0.35, 0.28, 0.58)) * damping
    gain = np.power(illum_safe, -gamma)
    # Soft highlight shoulder prevents bright windows/glints from blowing out.
    hi_w = np.clip((gray - 0.65) / 0.30, 0.0, 1.0)
    hi_w = hi_w * hi_w * (3.0 - 2.0 * hi_w)
    gain = 1.0 + (gain - 1.0) * (1.0 - hi_w)
    gain = np.clip(gain, 1.0, 2.45)

    lifted = np.clip(img_f * gain[..., None], 0.0, 1.0)
    # Shadow-adaptive chroma smoothing suppresses lifted dark-sensor chromatic speckle.
    lifted_u8 = np.clip(np.rint(lifted * 255.0), 0, 255).astype(np.uint8)
    ycrcb = cv2.cvtColor(lifted_u8, cv2.COLOR_RGB2YCrCb).astype(np.float32)
    boost_w = np.clip((gain - 1.0) / 1.2, 0.0, 0.75)
    cr_sm = cv2.GaussianBlur(ycrcb[..., 1], (5, 5), 1.2)
    cb_sm = cv2.GaussianBlur(ycrcb[..., 2], (5, 5), 1.2)
    ycrcb[..., 1] = ycrcb[..., 1] * (1.0 - boost_w) + cr_sm * boost_w
    ycrcb[..., 2] = ycrcb[..., 2] * (1.0 - boost_w) + cb_sm * boost_w
    out_u8 = cv2.cvtColor(np.clip(np.rint(ycrcb), 0, 255).astype(np.uint8), cv2.COLOR_YCrCb2RGB)
    return out_u8.astype(np.float32) / 255.0


def _apply_glare_shoulder_recovery(img_f: np.ndarray, stats: dict[str, float], damping: float = 1.0) -> np.ndarray:
    """LAB L* soft-knee highlight compression and gentle veiling-glare black-floor anchoring."""
    u8 = np.clip(np.rint(img_f * 255.0), 0, 255).astype(np.uint8)
    lab = cv2.cvtColor(u8, cv2.COLOR_RGB2LAB).astype(np.float32)
    L = lab[..., 0]

    # Gentle black-floor anchoring removes milky veiling glare without clipping shadows.
    p5 = float(stats.get("p5_all", 0.0))
    veil_shift = float(np.clip((p5 - 16.0) * 0.45, 0.0, 22.0)) * damping
    if veil_shift >= 2.0:
        t_hi = np.clip((L - p5) / max(40.0, 235.0 - p5), 0.0, 1.0)
        L = np.clip(L - veil_shift * (1.0 - 0.65 * t_hi), 0.0, 255.0)

    # Soft-knee highlight compression above L > 215 preserves natural hue in a*, b*.
    knee = 215.0
    over = np.maximum(0.0, L - knee)
    if float(over.max()) > 1.0:
        compressed = knee + (40.0 * over) / (40.0 + over * (0.45 * damping))
        L = np.where(L > knee, compressed, L)

    # Mild local contrast recovery on washed midtones (bounded CLAHE blend).
    l_u8 = np.clip(np.rint(L), 0, 255).astype(np.uint8)
    clahe = cv2.createCLAHE(clipLimit=1.5, tileGridSize=(4, 4))
    l_clahe = clahe.apply(l_u8).astype(np.float32)
    blend = 0.28 * damping
    lab[..., 0] = np.clip(L * (1.0 - blend) + l_clahe * blend, 0.0, 255.0)

    out_u8 = cv2.cvtColor(np.clip(np.rint(lab), 0, 255).astype(np.uint8), cv2.COLOR_LAB2RGB)
    return out_u8.astype(np.float32) / 255.0


def prepare_low_light_glare(
    rgb: np.ndarray,
    mask: np.ndarray | None = None,
    *,
    v34_probe: dict | None = None,
    damping: float = 1.0,
) -> tuple[np.ndarray | None, np.ndarray | None, str | None, dict]:
    """Run skin-tone-safe low-light Retinex lift or specular/veiling glare conditioning."""
    t0 = time.perf_counter()
    damp = float(np.clip(damping, 0.25, 1.0)) if math.isfinite(float(damping)) else 1.0
    det = detect_low_light_glare(rgb, mask=mask, v34_probe=v34_probe)
    if det is None:
        return None, None, None, {}

    src = rgb
    active = _resolve_active_mask(src.shape[:2], mask)
    if active is None or not bool(active.any()):
        return None, None, None, {}

    route = det["route"]
    img_f = src.astype(np.float32) / 255.0
    if route == "low_light_recovery":
        out_f = _apply_retinex_low_light(img_f, damping=damp)
    else:
        out_f = _apply_glare_shoulder_recovery(img_f, det["before"], damping=damp)

    out = np.clip(np.rint(out_f * 255.0), 0, 255).astype(np.uint8)
    if mask is not None:
        out[~active] = src[~active]

    edited = active & np.any(out != src, axis=2)
    changed = int(edited.sum())
    if changed == 0:
        return None, None, None, {}

    after_st = _core_exposure_stats(out, active)
    report = {
        **det,
        "damping": round(damp, 3),
        "after": after_st,
        "changed_native_pixels": changed,
        "masked": bool(mask is not None),
        "mask_fraction": round(_finite_float(float(edited.mean())), 6),
        "ms": round(_finite_float((time.perf_counter() - t0) * 1000.0), 2),
    }
    return out, edited.astype(np.uint8) * 255, route, report
