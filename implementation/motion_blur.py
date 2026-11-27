"""Linear motion-blur classifier (Structure Tensor + Radon + Cepstrum) and conservative Wiener conditioning.

Specifically closes V3.4's linear motion-smear blindspot (e.g. severe horizontal smear where
``V34DegradationEncoder`` reports high ``p_clean``) while staying strictly 0-FP on clean
portraits, isotropic defocus, stripes, and periodic scanlines.
"""
from __future__ import annotations

import math
import time
from typing import Any, Callable

import cv2
import numpy as np
from scipy.ndimage import gaussian_filter, map_coordinates

POLICY = "cepstrum-radon-st-motion-wiener-v1"
MIN_SIDE = 64
MAX_ANALYSIS_SIDE = 256
MIN_LENGTH_256 = 4.0
MAX_LENGTH_256 = 26.0
MAX_WIENER_LENGTH_NATIVE = 21.0


def _finite_float(val: float, default: float = 0.0) -> float:
    fval = float(val)
    return fval if math.isfinite(fval) else default


def _validate_rgb(rgb: Any) -> np.ndarray | None:
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


def _immerkaer_sigma(gray: np.ndarray) -> float:
    """Fast Immerkær 3x3 Laplacian noise estimate on float32 luma."""
    k = np.array([[1.0, -2.0, 1.0], [-2.0, 4.0, -2.0], [1.0, -2.0, 1.0]], dtype=np.float32)
    resp = cv2.filter2D(gray, cv2.CV_32F, k, borderType=cv2.BORDER_REFLECT)
    h, w = resp.shape
    if min(h, w) < 16:
        return 0.0
    inner = resp[2:-2, 2:-2]
    return float(np.mean(np.abs(inner)) * math.sqrt(math.pi / 2.0) / 6.0)


def _resize_luma_256(rgb: np.ndarray) -> tuple[np.ndarray, float]:
    h, w = rgb.shape[:2]
    longest = max(h, w)
    scale = min(1.0, float(MAX_ANALYSIS_SIDE) / float(longest))
    tw, th = max(32, int(round(w * scale))), max(32, int(round(h * scale)))
    if scale < 1.0:
        small = cv2.resize(rgb, (tw, th), interpolation=cv2.INTER_AREA)
    else:
        small = rgb
    f = small.astype(np.float32)
    y = 0.299 * f[..., 0] + 0.587 * f[..., 1] + 0.114 * f[..., 2]
    return y, scale


def _structure_tensor_anisotropy(gray: np.ndarray) -> dict[str, float]:
    """Compute multi-scale structure-tensor coherence and estimated smear angle."""
    h, w = gray.shape
    y0, y1 = max(2, int(h * 0.10)), min(h - 2, int(h * 0.90))
    x0, x1 = max(2, int(w * 0.10)), min(w - 2, int(w * 0.90))
    if y1 - y0 < 16 or x1 - x0 < 16:
        y0, y1, x0, x1 = 0, h, 0, w

    coherences = []
    angles_deg = []
    ratios = []
    for sigma in (0.9, 1.8):
        gx = gaussian_filter(gray, sigma=sigma, order=(0, 1))
        gy = gaussian_filter(gray, sigma=sigma, order=(1, 0))
        gx_r = gx[y0:y1, x0:x1]
        gy_r = gy[y0:y1, x0:x1]
        jxx = float(np.mean(gx_r * gx_r))
        jyy = float(np.mean(gy_r * gy_r))
        jxy = float(np.mean(gx_r * gy_r))
        trace = jxx + jyy
        if trace < 1e-4:
            coherences.append(0.0)
            angles_deg.append(0.0)
            ratios.append(1.0)
            continue
        disc = math.hypot(jxx - jyy, 2.0 * jxy)
        lam1 = 0.5 * (trace + disc)
        lam2 = max(1e-6, 0.5 * (trace - disc))
        coh = (lam1 - lam2) / max(1e-6, lam1 + lam2)
        # Principal gradient direction phi_grad; motion smear direction is orthogonal (+90 deg).
        phi_grad = 0.5 * math.atan2(2.0 * jxy, jxx - jyy)
        theta_motion = (math.degrees(phi_grad) + 90.0) % 180.0
        coherences.append(coh)
        angles_deg.append(theta_motion)
        ratios.append(math.sqrt(lam1 / lam2))

    # Weighted blend favoring the finer scale where linear motion wipes out parallel HF.
    return {
        "coherence_fine": coherences[0],
        "coherence_coarse": coherences[1],
        "coherence": 0.65 * coherences[0] + 0.35 * coherences[1],
        "angle_deg": angles_deg[0],
        "grad_ratio_fine": ratios[0],
        "grad_ratio_coarse": ratios[1],
    }


def _radon_and_cepstrum_analysis(gray: np.ndarray) -> dict[str, float]:
    """Compute directional log-PSD anisotropy (Radon) and 1D blur-axis power cepstrum."""
    # Pad to square power-friendly size for isotropic frequency radial sampling.
    h, w = gray.shape
    side = max(h, w)
    win_y = np.hanning(h).astype(np.float32)[:, None]
    win_x = np.hanning(w).astype(np.float32)[None, :]
    centered = (gray - float(gray.mean())) * (win_y * win_x)
    pad = np.zeros((side, side), dtype=np.float32)
    py = (side - h) // 2
    px = (side - w) // 2
    pad[py : py + h, px : px + w] = centered

    fft2 = np.fft.fftshift(np.fft.fft2(pad))
    power = (fft2.real * fft2.real + fft2.imag * fft2.imag).astype(np.float32)
    log_pow = np.log1p(power)

    # Check narrowband periodic spike (scanlines / stripes / checkerboards) in mid-to-high freq.
    cy, cx = side * 0.5, side * 0.5
    yy, xx = np.indices((side, side), dtype=np.float32)
    rr = np.hypot(xx - cx, yy - cy) / max(1.0, float(side))
    hf_ring = (rr >= 0.08) & (rr <= 0.44)
    if bool(hf_ring.any()):
        ring_vals = power[hf_ring]
        ring_med = max(1e-6, float(np.median(ring_vals)))
        ring_max = float(np.max(ring_vals))
        ring_p99 = float(np.percentile(ring_vals, 99.5))
        narrowband_ratio = ring_max / max(1e-6, ring_p99)
        peak_over_med = ring_max / ring_med
    else:
        narrowband_ratio = 0.0
        peak_over_med = 0.0

    # Sample log-power along radial wedges over theta in [0, 180) deg (step 3 deg = 60 angles).
    angles = np.arange(0.0, 180.0, 3.0, dtype=np.float32)
    n_rad = max(32, side // 2)
    radii = np.linspace(0.0, side * 0.45, n_rad, dtype=np.float32)
    rad_frac = radii / max(1.0, float(side))
    hf_band = (rad_frac >= 0.08) & (rad_frac <= 0.40)

    # Vectorized radial sampling of log_pow(r, theta) averaging +r and -r (symmetric).
    thetas_rad = np.deg2rad(angles)
    cos_t = np.cos(thetas_rad)[:, None]
    sin_t = np.sin(thetas_rad)[:, None]
    xs_pos = cx + cos_t * radii[None, :]
    ys_pos = cy + sin_t * radii[None, :]
    xs_neg = cx - cos_t * radii[None, :]
    ys_neg = cy - sin_t * radii[None, :]
    prof_pos = map_coordinates(log_pow, [ys_pos, xs_pos], order=1, mode="nearest")
    prof_neg = map_coordinates(log_pow, [ys_neg, xs_neg], order=1, mode="nearest")
    radial_log_pow = 0.5 * (prof_pos + prof_neg)

    # Energy along each angle in HF band: linear motion at angle theta suppresses HF along theta
    # (sinc nulls along f_theta) while preserving HF along theta + 90 deg.
    # In log-power space, orth_energy - hf_energy is the true log power ratio (exp = linear ratio).
    hf_energy = radial_log_pow[:, hf_band].mean(axis=1)
    half_n = len(angles) // 2
    orth_energy = np.roll(hf_energy, half_n)
    log_diffs = orth_energy - hf_energy
    best_idx = int(np.argmax(log_diffs))
    best_angle = float(angles[best_idx])
    hf_log_diff = float(log_diffs[best_idx])
    hf_dir_ratio = float(math.exp(min(12.0, max(-12.0, hf_log_diff))))
    hf_blur_axis = float(hf_energy[best_idx])
    hf_orth_axis = float(orth_energy[best_idx])

    # Compute 1D directional power cepstrum along blur axis best_angle (averaged over +-3 deg wedge).
    wedge_idxs = [(best_idx - 1) % len(angles), best_idx, (best_idx + 1) % len(angles)]
    blur_prof = radial_log_pow[wedge_idxs].mean(axis=0)
    # Mirror symmetric 1D log-spectrum across DC and subtract smooth low-freq envelope.
    full_1d = np.concatenate([blur_prof[::-1][:-1], blur_prof])
    envelope = gaussian_filter(full_1d, sigma=max(2.0, n_rad * 0.08))
    detrended = (full_1d - envelope) * np.hanning(full_1d.size).astype(np.float32)
    # Real power cepstrum of detrended log-power spectrum: negative trough or sharp peak at q = L.
    cep = np.fft.rfft(detrended)
    cep_real = np.real(cep)
    cep_mag = np.abs(cep)
    # Map bin k in rfft of length M = full_1d.size sampled at df = 0.45 / (n_rad - 1) cycles/px:
    # A sinc(pi * L * f) has period df_null = 1/L cycles/px -> frequency in f-domain is L samples/cycle.
    # Bin k corresponds to quefrency L_k = k / (M * df).
    df = 0.45 / max(1.0, float(n_rad - 1))
    m_len = float(full_1d.size)
    ks = np.arange(cep_mag.size, dtype=np.float32)
    quefrencies = ks / max(1e-6, m_len * df)

    valid_q = (quefrencies >= MIN_LENGTH_256) & (quefrencies <= MAX_LENGTH_256)
    bg_q = (quefrencies >= 3.0) & (quefrencies <= min(40.0, float(quefrencies[-1])))
    if bool(valid_q.any()) and int(bg_q.sum()) >= 6:
        bg_vals = cep_mag[bg_q]
        med = float(np.median(bg_vals))
        mad = max(1e-4, float(np.median(np.abs(bg_vals - med))))
        valid_indices = np.flatnonzero(valid_q)
        best_k = int(valid_indices[int(np.argmax(cep_mag[valid_indices]))])
        cep_peak = float(cep_mag[best_k])
        cep_snr = (cep_peak - med) / (1.4826 * mad)
        est_len = float(quefrencies[best_k])
    else:
        cep_snr = 0.0
        est_len = 0.0

    # Also measure orthogonal profile cepstrum SNR so isotropic patterns don't masquerade as linear motion.
    orth_idx = (best_idx + half_n) % len(angles)
    orth_wedge = [(orth_idx - 1) % len(angles), orth_idx, (orth_idx + 1) % len(angles)]
    orth_prof = radial_log_pow[orth_wedge].mean(axis=0)
    orth_1d = np.concatenate([orth_prof[::-1][:-1], orth_prof])
    orth_det = (orth_1d - gaussian_filter(orth_1d, sigma=max(2.0, n_rad * 0.08))) * np.hanning(orth_1d.size).astype(np.float32)
    orth_cep = np.abs(np.fft.rfft(orth_det))
    if bool(valid_q.any()) and int(bg_q.sum()) >= 6:
        omed = float(np.median(orth_cep[bg_q]))
        omad = max(1e-4, float(np.median(np.abs(orth_cep[bg_q] - omed))))
        orth_cep_snr = (float(orth_cep[valid_q].max()) - omed) / (1.4826 * omad)
    else:
        orth_cep_snr = 0.0

    # Check if orthogonal axis has a sharp periodic stripe peak (pure synthetic/fabric stripes).
    orth_hf = radial_log_pow[orth_idx, hf_band]
    orth_hf_med = max(0.5, float(np.median(orth_hf)))
    orth_hf_spike = float(np.max(orth_hf)) / orth_hf_med

    return {
        "radon_angle_deg": best_angle,
        "hf_log_diff": hf_log_diff,
        "hf_dir_ratio": hf_dir_ratio,
        "hf_blur_axis": hf_blur_axis,
        "hf_orth_axis": hf_orth_axis,
        "cepstrum_snr": cep_snr,
        "orth_cepstrum_snr": orth_cep_snr,
        "length_256": est_len,
        "narrowband_ratio": narrowband_ratio,
        "peak_over_med": peak_over_med,
        "orth_hf_spike": orth_hf_spike,
    }


def _directional_smear_stats(gray: np.ndarray, angle_deg: float) -> dict[str, float]:
    """Measure directional 2nd-derivative anisotropy, autocorrelation decay, and edge sharpness along angle."""
    h, w = gray.shape
    cx, cy = w * 0.5, h * 0.5
    mat = cv2.getRotationMatrix2D((cx, cy), angle_deg, 1.0)
    rot = cv2.warpAffine(gray, mat, (w, h), flags=cv2.INTER_LINEAR, borderMode=cv2.BORDER_REFLECT)
    y0, y1 = int(h * 0.15), int(h * 0.85)
    x0, x1 = int(w * 0.15), int(w * 0.85)
    inner = rot[y0:y1, x0:x1]
    if min(inner.shape) < 28:
        return {
            "d2_ratio": 1.0,
            "ac_par_10": 0.0,
            "ac_decay_ratio": 1.0,
            "edge_sharp_par": 3.0,
        }
    d2x = float(np.mean(np.abs(np.diff(inner, n=2, axis=1))))
    d2y = float(np.mean(np.abs(np.diff(inner, n=2, axis=0))))
    d2_ratio = d2y / max(1e-4, d2x)

    zm = inner - float(inner.mean())
    var = max(1e-6, float(np.mean(zm * zm)))
    lag = min(12, max(6, min(inner.shape) // 8))
    ac_par = float(np.mean(zm[:, :-lag] * zm[:, lag:]) / var)
    ac_perp = float(np.mean(zm[:-lag, :] * zm[lag:, :]) / var)
    ac_decay_ratio = (1.0 - ac_perp) / max(1e-4, 1.0 - ac_par)

    gx1 = np.abs(gaussian_filter(inner, 0.8, order=(0, 1)))
    gx3 = np.abs(gaussian_filter(inner, 3.0, order=(0, 1)))
    edge_sharp_par = float(np.percentile(gx1, 95) / max(1e-4, np.percentile(gx3, 95)))

    return {
        "d2_ratio": d2_ratio,
        "ac_par_10": ac_par,
        "ac_decay_ratio": ac_decay_ratio,
        "edge_sharp_par": edge_sharp_par,
    }


def _angle_diff_deg(a: float, b: float) -> float:
    diff = abs((a - b) % 180.0)
    return min(diff, 180.0 - diff)


def _is_periodic_stripe_pattern(gray: np.ndarray, angle_deg: float) -> bool:
    """Reject periodic stripes / blinds / scanlines whose profile along stripe axis is flat or strictly periodic."""
    # Rotate so smear/stripe axis aligns with X; stripes have near-zero variance along X
    # and sharp alternating periodicity along Y.
    h, w = gray.shape
    cx, cy = w * 0.5, h * 0.5
    mat = cv2.getRotationMatrix2D((cx, cy), angle_deg, 1.0)
    rot = cv2.warpAffine(gray, mat, (w, h), flags=cv2.INTER_LINEAR, borderMode=cv2.BORDER_REFLECT)
    y0, y1 = int(h * 0.15), int(h * 0.85)
    x0, x1 = int(w * 0.15), int(w * 0.85)
    crop = rot[y0:y1, x0:x1]
    if min(crop.shape) < 20:
        return False
    # If pixels along the smear axis have almost zero variation compared to orthogonal variation,
    # it is a 1D stripe/bar pattern, not a motion-blurred face.
    row_std = float(np.mean(np.std(crop, axis=1)))
    col_std = float(np.mean(np.std(crop, axis=0)))
    if row_std < 3.5 and col_std > 12.0:
        return True
    # Check 1D FFT of mean vertical profile for a single dominant periodic stripe frequency.
    v_prof = crop.mean(axis=1)
    v_prof = (v_prof - v_prof.mean()) * np.hanning(v_prof.size)
    spec = np.abs(np.fft.rfft(v_prof))
    if spec.size > 6:
        hf = spec[3:]
        med = max(1e-4, float(np.median(hf)))
        top = float(hf.max())
        # Pure periodic stripes concentrate most AC energy into 1-2 harmonics.
        if top > 14.0 * med and float(np.sort(hf)[-2:].sum()) > 0.55 * float(hf.sum()):
            return True
    return False


def detect_motion_blur(rgb: np.ndarray) -> dict[str, Any] | None:
    """Classify linear motion blur via Structure Tensor + Radon + Cepstrum.

    Returns a JSON-safe detection report when strong linear motion smear is confirmed,
    or ``None`` for sharp portraits, isotropic defocus, noisy frames, and striped patterns.
    """
    src = _validate_rgb(rgb)
    if src is None:
        return None
    gray_256, scale = _resize_luma_256(src)
    if float(gray_256.std()) < 12.0:
        return None

    noise_sigma = _immerkaer_sigma(gray_256)
    if noise_sigma > 5.0:
        return None

    st = _structure_tensor_anisotropy(gray_256)
    rc = _radon_and_cepstrum_analysis(gray_256)

    angle_agree = _angle_diff_deg(st["angle_deg"], rc["radon_angle_deg"])
    hf_log_diff = rc["hf_log_diff"]
    hf_dir_ratio = rc["hf_dir_ratio"]
    st_coh = st["coherence"]
    st_grad_ratio = st["grad_ratio_fine"]
    cep_snr = rc["cepstrum_snr"]
    orth_cep_snr = rc["orth_cepstrum_snr"]
    length_256 = rc["length_256"]

    # Reject periodic stripes / scanlines / checkerboards.
    if rc["narrowband_ratio"] > 18.0 and rc["peak_over_med"] > 450.0:
        return None
    if rc["orth_hf_spike"] > 2.35:
        return None

    motion_angle = rc["radon_angle_deg"]
    ds = _directional_smear_stats(gray_256, motion_angle)
    d2_ratio = ds["d2_ratio"]
    ac_par_10 = ds["ac_par_10"]
    ac_decay_ratio = ds["ac_decay_ratio"]
    edge_sharp_par = ds["edge_sharp_par"]

    # Dual acceptance path:
    # Path A: Linear motion PSF (synthetic or real) with strong spectral directional collapse
    # (log_diff >= 2.8 => >16x directional power ratio), 2nd-derivative smear anisotropy,
    # and softened blur-axis edges (rejects sharp aliased diagonal contours with edge_sharp_par > 1.85).
    path_a = (
        hf_log_diff >= 2.80
        and d2_ratio >= 1.45
        and edge_sharp_par <= 1.75
        and (
            (cep_snr >= 2.6 and MIN_LENGTH_256 <= length_256 <= MAX_LENGTH_256)
            or hf_log_diff >= 3.60
        )
    )
    # Path B: Photographic long-exposure linear motion smear (like degraded_09 horizontal smear)
    # with strong directional autocorrelation persistence along blur axis, soft blur-axis edges,
    # spectral Radon anisotropy, cepstral/decay support, and agreeing structure-tensor orientation.
    path_b = (
        hf_log_diff >= 1.80
        and st_coh >= 0.30
        and angle_agree <= 30.0
        and ac_par_10 >= 0.91
        and ac_decay_ratio >= 1.75
        and edge_sharp_par <= 1.65
        and (cep_snr >= 2.5 or ac_decay_ratio >= 2.10)
    )
    if not (path_a or path_b):
        return None

    # Final spatial stripe/blinds rejection along the detected angle.
    if _is_periodic_stripe_pattern(gray_256, motion_angle):
        return None

    # Calibrate conservative smear length estimate at 256-scale and native scale.
    if path_a and MIN_LENGTH_256 <= length_256 <= MAX_LENGTH_256:
        est_len_256 = length_256
    elif path_a:
        est_len_256 = float(np.clip(4.0 + 2.4 * (hf_log_diff - 2.2), 5.0, 14.0))
    else:
        est_len_256 = float(np.clip(5.0 + 3.0 * (ac_decay_ratio - 1.5), 6.0, 14.0))

    length_native = float(np.clip(est_len_256 / max(1e-4, scale), 4.0, MAX_WIENER_LENGTH_NATIVE))

    return {
        "policy": POLICY,
        "suspected": True,
        "classification": "linear_motion",
        "angle_deg": round(_finite_float(motion_angle), 2),
        "length_256": round(_finite_float(est_len_256), 2),
        "length_native": round(_finite_float(length_native), 2),
        "st_coherence": round(_finite_float(st_coh), 4),
        "st_grad_ratio": round(_finite_float(st_grad_ratio), 3),
        "hf_log_diff": round(_finite_float(hf_log_diff), 3),
        "hf_dir_ratio": round(_finite_float(hf_dir_ratio), 3),
        "d2_ratio": round(_finite_float(d2_ratio), 3),
        "ac_par_10": round(_finite_float(ac_par_10), 4),
        "ac_decay_ratio": round(_finite_float(ac_decay_ratio), 3),
        "edge_sharp_par": round(_finite_float(edge_sharp_par), 3),
        "cepstrum_snr": round(_finite_float(cep_snr), 3),
        "angle_agreement_deg": round(_finite_float(angle_agree), 2),
        "noise_sigma_256": round(_finite_float(noise_sigma), 3),
    }


def _build_motion_psf(length: float, angle_deg: float) -> np.ndarray:
    """Build normalized subpixel linear motion PSF kernel."""
    ksize = max(5, int(math.ceil(length)) | 1)
    if ksize % 2 == 0:
        ksize += 1
    psf = np.zeros((ksize, ksize), dtype=np.float32)
    center = ksize // 2
    n_samples = max(16, int(math.ceil(length * 4.0)))
    ts = np.linspace(-0.5 * (length - 1.0), 0.5 * (length - 1.0), n_samples, dtype=np.float32)
    rad = math.radians(angle_deg)
    ca, sa = math.cos(rad), math.sin(rad)
    for t in ts:
        x = center + t * ca
        y = center + t * sa
        x0, y0 = int(math.floor(x)), int(math.floor(y))
        fx, fy = x - x0, y - y0
        for dy, wy in ((0, 1.0 - fy), (1, fy)):
            for dx, wx in ((0, 1.0 - fx), (1, fx)):
                yy, xx = y0 + dy, x0 + dx
                if 0 <= yy < ksize and 0 <= xx < ksize:
                    psf[yy, xx] += wx * wy
    s = float(psf.sum())
    if s > 0.0:
        psf /= s
    else:
        psf[center, center] = 1.0
    return psf


def _wiener_deconvolve_luma(
    luma: np.ndarray,
    psf: np.ndarray,
    nsr: float,
    alpha: float = 0.72,
) -> np.ndarray:
    """Conservative edge-tapered regularized Wiener deconvolution with anti-ringing clamp."""
    h, w = luma.shape
    pad = max(16, psf.shape[0] * 2)
    padded = cv2.copyMakeBorder(luma, pad, pad, pad, pad, cv2.BORDER_REFLECT_101).astype(np.float32)
    ph, pw = padded.shape

    # Smooth cosine border taper on the outer reflection band to eliminate FFT wrap ringing.
    mean_val = float(padded.mean())
    taper_y = np.ones(ph, dtype=np.float32)
    taper_x = np.ones(pw, dtype=np.float32)
    t_len = max(6, pad // 2)
    ramp = 0.5 * (1.0 - np.cos(np.linspace(0.0, math.pi, t_len, dtype=np.float32)))
    taper_y[:t_len] = ramp
    taper_y[-t_len:] = ramp[::-1]
    taper_x[:t_len] = ramp
    taper_x[-t_len:] = ramp[::-1]
    taper_2d = taper_y[:, None] * taper_x[None, :]
    tapered = mean_val + (padded - mean_val) * taper_2d

    # Place PSF centered at origin (0, 0) via roll for zero phase shift.
    kh, kw = psf.shape
    psf_pad = np.zeros((ph, pw), dtype=np.float32)
    psf_pad[:kh, :kw] = psf
    psf_pad = np.roll(psf_pad, -(kh // 2), axis=0)
    psf_pad = np.roll(psf_pad, -(kw // 2), axis=1)

    H_f = np.fft.rfft2(psf_pad)
    Y_f = np.fft.rfft2(tapered)
    H_mag2 = (H_f.real * H_f.real + H_f.imag * H_f.imag).astype(np.float32)
    # Regularized Wiener filter with mild high-frequency Laplacian damping to suppress Gibbs ringing
    # while preserving passband deconvolution gain across the first PSF sidelobes.
    fy = np.fft.fftfreq(ph)[:, None].astype(np.float32)
    fx = np.fft.rfftfreq(pw)[None, :].astype(np.float32)
    lap_reg = (2.0 * (1.0 - np.cos(2.0 * math.pi * fx)) + 2.0 * (1.0 - np.cos(2.0 * math.pi * fy))) ** 2
    denom = H_mag2 + float(nsr) + 0.004 * float(nsr) * lap_reg
    deconv_pad = np.fft.irfft2((np.conj(H_f) / np.maximum(1e-6, denom)) * Y_f, s=(ph, pw)).astype(np.float32)
    deconv = deconv_pad[pad : pad + h, pad : pad + w]

    # Local overshoot/undershoot clamp prevents Gibbs halo leaks into V3.4 (w_pred >= 0.60).
    k_loc = max(5, (psf.shape[0] // 2) | 1)
    loc_max = cv2.dilate(luma, np.ones((k_loc, k_loc), np.uint8))
    loc_min = cv2.erode(luma, np.ones((k_loc, k_loc), np.uint8))
    deconv_clamped = np.clip(deconv, loc_min - 12.0, loc_max + 12.0)

    blended = luma * (1.0 - alpha) + deconv_clamped * alpha
    return np.clip(blended, 0.0, 255.0)


def prepare_motion_deblur(
    rgb: np.ndarray,
    *,
    probe_fn: Callable[[np.ndarray], dict[str, float] | None] | None = None,
) -> tuple[np.ndarray | None, np.ndarray | None, dict[str, Any]]:
    """Detect and conservatively condition linear motion blur in native YCrCb luma.

    Returns ``(cleaned_rgb, mask_u8, report)`` or ``(None, None, {})`` when not triggered
    or when self-verification / V3.4 reflection rejects the deblurred candidate.
    """
    t0 = time.perf_counter()
    src = _validate_rgb(rgb)
    if src is None:
        return None, None, {}

    det = detect_motion_blur(src)
    if det is None:
        return None, None, {}

    ycrcb = cv2.cvtColor(src, cv2.COLOR_RGB2YCrCb)
    y_in = ycrcb[..., 0].astype(np.float32)
    noise_before = _immerkaer_sigma(y_in)
    if noise_before > 5.0:
        return None, None, {}

    # Conservative Wiener parameters: cap deconvolution length slightly below full smear
    # so we collapse a heavy smear into a clean mild residual for frozen CodeFormer V3.4.
    eff_len = float(np.clip(det["length_native"] * 0.75, 4.0, 15.0))
    psf = _build_motion_psf(eff_len, det["angle_deg"])
    nsr = float(np.clip(0.012 + 0.004 * noise_before, 0.012, 0.040))
    y_out = _wiener_deconvolve_luma(y_in, psf, nsr=nsr, alpha=0.78)

    noise_after = _immerkaer_sigma(y_out)
    noise_delta = noise_after - noise_before
    if noise_delta > 3.2:
        # Retry once with stronger regularization before giving up.
        y_out = _wiener_deconvolve_luma(y_in, psf, nsr=min(0.08, nsr * 2.0), alpha=0.62)
        noise_after = _immerkaer_sigma(y_out)
        noise_delta = noise_after - noise_before
        if noise_delta > 3.2:
            return None, None, {}

    # Verify directional gradient energy along the smeared axis genuinely improved.
    rad = math.radians(det["angle_deg"])
    ca, sa = math.cos(rad), math.sin(rad)
    gx_b = cv2.Sobel(y_in, cv2.CV_32F, 1, 0, ksize=3)
    gy_b = cv2.Sobel(y_in, cv2.CV_32F, 0, 1, ksize=3)
    gx_a = cv2.Sobel(y_out, cv2.CV_32F, 1, 0, ksize=3)
    gy_a = cv2.Sobel(y_out, cv2.CV_32F, 0, 1, ksize=3)
    par_before = float(np.mean(np.abs(gx_b * ca + gy_b * sa)))
    par_after = float(np.mean(np.abs(gx_a * ca + gy_a * sa)))
    hf_gain = par_after / max(1e-4, par_before)
    if hf_gain < 1.08:
        return None, None, {}

    out_ycrcb = ycrcb.copy()
    out_ycrcb[..., 0] = np.clip(np.rint(y_out), 0, 255).astype(np.uint8)
    out_rgb = cv2.cvtColor(out_ycrcb, cv2.COLOR_YCrCb2RGB)

    changed = np.any(out_rgb != src, axis=2)
    changed_px = int(changed.sum())
    if changed_px == 0:
        return None, None, {}

    # Optional V3.4 non-ringing reflection guard.
    if probe_fn is not None:
        b_probe = probe_fn(src)
        a_probe = probe_fn(out_rgb)
        if isinstance(b_probe, dict) and isinstance(a_probe, dict):
            if float(a_probe.get("noise", 0.0)) > float(b_probe.get("noise", 0.0)) + 0.15:
                return None, None, {}
            if float(a_probe.get("blur", 0.0)) > float(b_probe.get("blur", 0.0)) + 0.15:
                return None, None, {}

    mask_u8 = changed.astype(np.uint8) * 255
    report = {
        **det,
        "triggered": True,
        "route": "motion_deblur",
        "wiener_length_native": round(_finite_float(eff_len), 2),
        "wiener_nsr": round(_finite_float(nsr), 4),
        "blur_axis_hf_gain": round(_finite_float(hf_gain), 3),
        "noise_before": round(_finite_float(noise_before), 3),
        "noise_after": round(_finite_float(noise_after), 3),
        "changed_native_pixels": changed_px,
        "mask_fraction": round(_finite_float(float(changed.mean())), 6),
        "ms": round(_finite_float((time.perf_counter() - t0) * 1000.0), 2),
    }
    return out_rgb, mask_u8, report
