"""Deterministic 12-Regime CCTV Face Degradation Generator for CodeFormer Restoration V3.

Covers all 12 physically grounded CCTV regimes from #333 plus exact live
prepare_crop() non-square/shifted/rotated/padded geometry simulation.

Outputs:
    (inp_256_u8, gt_256_u8, gt_512_u8, observed_256_bool, targets_dict)
"""
from __future__ import annotations

import io
import math
from typing import Any, Dict, Optional, Tuple

import cv2
import numpy as np
from PIL import Image

REGIMES: Tuple[str, ...] = (
    "clear",
    "global_defocus_blur",
    "directional_motion_blur",
    "local_partial_blur",
    "compound_cctv_codec",
    "severe_lr_survface",
    "lowlight_sensor_noise",
    "joint_lowlight_blur_codec",
    "daylight_sensor_noise",
    "h264_macroblock_interlace",
    "haze_colorcast_glare",
    "mild_transition",
)

REGIME_WEIGHTS: Dict[str, float] = {
    "clear": 0.12,
    "global_defocus_blur": 0.10,
    "directional_motion_blur": 0.10,
    "local_partial_blur": 0.06,
    "compound_cctv_codec": 0.12,
    "severe_lr_survface": 0.10,
    "lowlight_sensor_noise": 0.10,
    "joint_lowlight_blur_codec": 0.12,
    "daylight_sensor_noise": 0.06,
    "h264_macroblock_interlace": 0.05,
    "haze_colorcast_glare": 0.04,
    "mild_transition": 0.03,
}

_REGIME_PROBS = np.array([REGIME_WEIGHTS[r] for r in REGIMES], dtype=np.float64)
_REGIME_PROBS /= _REGIME_PROBS.sum()


def _to_f32(img_u8: np.ndarray) -> np.ndarray:
    return img_u8.astype(np.float32) / np.float32(255.0)


def _to_u8(img_f32: np.ndarray) -> np.ndarray:
    return np.clip(np.rint(img_f32 * np.float32(255.0)), 0.0, 255.0).astype(np.uint8)


def _resize_pil_bilinear(img_u8: np.ndarray, size: Tuple[int, int]) -> np.ndarray:
    return np.asarray(Image.fromarray(img_u8).resize(size, Image.Resampling.BILINEAR)).copy()


def _resize_pil_lanczos(img_u8: np.ndarray, size: Tuple[int, int]) -> np.ndarray:
    return np.asarray(Image.fromarray(img_u8).resize(size, Image.Resampling.LANCZOS)).copy()


def _prepare_crop_exact(
    rgb: np.ndarray,
    out_size: int = 256,
    resample: Image.Resampling = Image.Resampling.BILINEAR,
) -> Tuple[np.ndarray, np.ndarray]:
    """Exact neutral-(128,128,128) center padding + resize + nearest observed support.

    Matches integrated_app/implementation/runtime.py:prepare_crop when resample=BILINEAR.
    """
    h, w = rgb.shape[:2]
    side = max(h, w)
    top, left = (side - h) // 2, (side - w) // 2
    canvas = Image.new("RGB", (side, side), (128, 128, 128))
    canvas.paste(Image.fromarray(rgb), (left, top))
    support = Image.new("L", (side, side), 0)
    support.paste(255, (left, top, left + w, top + h))
    target = np.asarray(canvas.resize((out_size, out_size), resample)).copy()
    observed = np.asarray(support.resize((out_size, out_size), Image.Resampling.NEAREST)).copy() > 0
    if resample != Image.Resampling.BILINEAR:
        target[~observed] = 128
    return target, observed


def _jpeg_compress(img_u8: np.ndarray, quality: int, subsampling: int = 2) -> np.ndarray:
    """Deterministic PIL JPEG compression (subsampling=2 is 4:2:0)."""
    q = int(np.clip(quality, 5, 100))
    buf = io.BytesIO()
    Image.fromarray(img_u8).save(buf, format="JPEG", quality=q, subsampling=subsampling)
    buf.seek(0)
    with Image.open(buf) as dec:
        return np.asarray(dec.convert("RGB")).copy()


def _gaussian_or_disk_blur(img_f32: np.ndarray, rng: np.random.Generator, sigma_min: float, sigma_max: float) -> Tuple[np.ndarray, float]:
    lo_in = max(0.1, float(min(sigma_min, sigma_max)))
    hi_in = max(lo_in, float(max(sigma_min, sigma_max)))
    sigma_x = float(rng.uniform(lo_in, hi_in)) if hi_in > lo_in else lo_in
    if rng.random() < 0.30:
        # Disk bokeh kernel
        radius = max(1, int(round(sigma_x * 1.4)))
        yy, xx = np.mgrid[-radius:radius + 1, -radius:radius + 1]
        kernel = ((xx * xx + yy * yy) <= (radius * radius + 0.5)).astype(np.float32)
        kernel /= max(float(kernel.sum()), 1e-6)
        out = cv2.filter2D(img_f32, -1, kernel, borderType=cv2.BORDER_REFLECT_101)
        return np.clip(out, 0.0, 1.0), sigma_x
    # Isotropic or anisotropic Gaussian (guaranteed lo_y <= hi_y for small sigmas / sp_scale)
    if rng.random() < 0.6:
        sigma_y = sigma_x
    else:
        lo_y = max(lo_in, sigma_x * 0.5)
        hi_y = min(hi_in, sigma_x * 1.4)
        sigma_y = float(rng.uniform(lo_y, hi_y)) if hi_y > lo_y else sigma_x
    k_x = max(3, int(round(sigma_x * 4.0)) | 1)
    k_y = max(3, int(round(sigma_y * 4.0)) | 1)
    out = cv2.GaussianBlur(img_f32, (k_x, k_y), sigmaX=sigma_x, sigmaY=sigma_y, borderType=cv2.BORDER_REFLECT_101)
    return np.clip(out, 0.0, 1.0), 0.5 * (sigma_x + sigma_y)


def _motion_blur(img_f32: np.ndarray, rng: np.random.Generator, len_min: int = 5, len_max: int = 19) -> Tuple[np.ndarray, int]:
    length = int(rng.integers(len_min, len_max + 1))
    if length % 2 == 0:
        length += 1
    theta = float(rng.uniform(0.0, 180.0))
    kernel = np.zeros((length, length), dtype=np.float32)
    c = length // 2
    rad = math.radians(theta)
    dx = int(round(c * math.cos(rad)))
    dy = int(round(c * math.sin(rad)))
    if rng.random() < 0.30 and length >= 7:
        # Slight curved quadratic PSF
        t = np.linspace(-1.0, 1.0, length * 3)
        bend = float(rng.uniform(-0.35, 0.35)) * c
        nx, ny = -math.sin(rad), math.cos(rad)
        xs = np.clip(np.rint(c + t * dx + (1.0 - t * t) * bend * nx).astype(int), 0, length - 1)
        ys = np.clip(np.rint(c + t * dy + (1.0 - t * t) * bend * ny).astype(int), 0, length - 1)
        kernel[ys, xs] = 1.0
    else:
        cv2.line(kernel, (c - dx, c - dy), (c + dx, c + dy), 1.0, 1)
    s = float(kernel.sum())
    if s <= 0.0:
        kernel[c, c] = 1.0
    else:
        kernel /= s
    out = cv2.filter2D(img_f32, -1, kernel, borderType=cv2.BORDER_REFLECT_101)
    return np.clip(out, 0.0, 1.0), length


def _downsample_upsample(img_f32: np.ndarray, rng: np.random.Generator, min_side: int, max_side: int) -> Tuple[np.ndarray, int]:
    h, w = img_f32.shape[:2]
    lr_side = int(rng.integers(min_side, max_side + 1))
    scale = lr_side / float(max(h, w))
    lr_w = max(8, int(round(w * scale)))
    lr_h = max(8, int(round(h * scale)))
    down_interp = int(rng.choice([cv2.INTER_AREA, cv2.INTER_LINEAR, cv2.INTER_CUBIC]))
    up_interp = int(rng.choice([cv2.INTER_LINEAR, cv2.INTER_CUBIC]))
    small = cv2.resize(img_f32, (lr_w, lr_h), interpolation=down_interp)
    back = cv2.resize(small, (w, h), interpolation=up_interp)
    return np.clip(back, 0.0, 1.0), lr_side


def _downsample_upsample_smooth(
    img_f32: np.ndarray,
    lr_side: int,
) -> np.ndarray:
    """Clean smooth INTER_AREA -> INTER_LINEAR LR resize for V3.4 x_struct_clean targets."""
    h, w = img_f32.shape[:2]
    scale = float(lr_side) / float(max(h, w))
    lr_w = max(8, int(round(w * scale)))
    lr_h = max(8, int(round(h * scale)))
    small = cv2.resize(img_f32, (lr_w, lr_h), interpolation=cv2.INTER_AREA)
    back = cv2.resize(small, (w, h), interpolation=cv2.INTER_LINEAR)
    return np.clip(back, 0.0, 1.0)


def _nn_pixelate(
    img_f32: np.ndarray,
    rng: np.random.Generator,
    min_side: int = 14,
    max_side: int = 48,
) -> Tuple[np.ndarray, int]:
    """100%-strength INTER_AREA/LINEAR -> cv2.INTER_NEAREST pixelation grid."""
    h, w = img_f32.shape[:2]
    lo = max(8, int(min(min_side, max_side)))
    hi = max(lo, int(max(min_side, max_side)))
    lr_side = int(rng.integers(lo, hi + 1)) if hi > lo else lo
    scale = float(lr_side) / float(max(h, w))
    lr_w = max(8, int(round(w * scale)))
    lr_h = max(8, int(round(h * scale)))
    down_interp = int(rng.choice([cv2.INTER_AREA, cv2.INTER_LINEAR]))
    small = cv2.resize(img_f32, (lr_w, lr_h), interpolation=down_interp)
    back = cv2.resize(small, (w, h), interpolation=cv2.INTER_NEAREST)
    return np.clip(back, 0.0, 1.0), lr_side


def _add_rgb_confetti(
    img_f32: np.ndarray,
    rng: np.random.Generator,
    sigma_min: float = 0.03,
    sigma_max: float = 0.095,
    p_min: float = 0.01,
    p_max: float = 0.05,
    coarse_prob: float = 0.50,
) -> Tuple[np.ndarray, float, bool]:
    """Uncorrelated RGB Gaussian + per-channel impulse confetti (1px or 2-4px coarse upscaled)."""
    h, w = img_f32.shape[:2]
    sigma = float(rng.uniform(sigma_min, sigma_max))
    p_imp = float(rng.uniform(p_min, p_max))
    is_coarse = bool(rng.random() < coarse_prob)

    if is_coarse:
        # Inject on an intermediate 64..160px canvas then upscale to native (2-4px coarse confetti at 256px)
        max_c = max(32, min(160, max(h, w)))
        min_c = min(64, max_c)
        c_side = int(rng.integers(min_c, max_c + 1)) if max_c > min_c else min_c
        scale = float(c_side) / float(max(h, w))
        cw = max(16, int(round(w * scale)))
        ch = max(16, int(round(h * scale)))
        work = cv2.resize(img_f32, (cw, ch), interpolation=cv2.INTER_AREA)
        work = work + rng.normal(0.0, sigma, work.shape).astype(np.float32)
        u_ch = rng.random(work.shape)
        work[u_ch < (p_imp * 0.5)] = 0.0
        work[(u_ch >= (p_imp * 0.5)) & (u_ch < p_imp)] = 1.0
        work = np.clip(work, 0.0, 1.0)
        up_interp = int(rng.choice([cv2.INTER_LINEAR, cv2.INTER_NEAREST]))
        out = cv2.resize(work, (w, h), interpolation=up_interp)
        return np.clip(out, 0.0, 1.0), sigma, True

    out = img_f32.copy()
    out = out + rng.normal(0.0, sigma, out.shape).astype(np.float32)
    u_ch = rng.random(out.shape)
    out[u_ch < (p_imp * 0.5)] = 0.0
    out[(u_ch >= (p_imp * 0.5)) & (u_ch < p_imp)] = 1.0
    return np.clip(out, 0.0, 1.0), sigma, False


def _add_sensor_noise(
    img_f32: np.ndarray,
    rng: np.random.Generator,
    sigma_min: float,
    sigma_max: float,
    *,
    poisson: bool = True,
    chroma_boost: float = 1.35,
    impulse_prob: float = 0.0,
) -> Tuple[np.ndarray, float]:
    out = img_f32.copy()
    sigma = float(rng.uniform(sigma_min, sigma_max))
    if poisson:
        # Shot noise proportional to sqrt(signal)
        shot_scale = float(rng.uniform(0.4, 1.2)) * sigma
        shot = rng.normal(0.0, 1.0, out.shape).astype(np.float32) * np.sqrt(np.clip(out, 0.0, 1.0) + 1e-4) * shot_scale
        out = out + shot
    # Read Gaussian noise (luma + chroma)
    luma_n = rng.normal(0.0, sigma * 0.7, out.shape[:2] + (1,)).astype(np.float32)
    chroma_n = rng.normal(0.0, sigma * chroma_boost * 0.6, out.shape).astype(np.float32)
    out = out + luma_n + chroma_n
    if impulse_prob > 0.0 and rng.random() < impulse_prob:
        p_salt = float(rng.uniform(0.002, 0.012))
        u = rng.random(out.shape[:2])
        out[u < (p_salt * 0.5)] = 0.0
        out[(u >= (p_salt * 0.5)) & (u < p_salt)] = 1.0
    return np.clip(out, 0.0, 1.0), sigma


FFHQ_CANONICAL_5PT_NORM = np.array(
    [
        [0.3725, 0.4685],
        [0.6275, 0.4685],
        [0.5000, 0.6215],
        [0.3980, 0.7420],
        [0.6020, 0.7420],
    ],
    dtype=np.float32,
)


def _apply_geometry_warp_crop_with_meta(
    gt_512: np.ndarray, rng: np.random.Generator
) -> Tuple[np.ndarray, np.ndarray]:
    """Apply random scale (0.75-1.15x), shift (+-12%), rotation (+-10 deg),
    and non-square aspect ratio (0.60-1.50), returning (cropped_rgb, warped_5pt_in_crop)."""
    h, w = gt_512.shape[:2]
    scale = float(rng.uniform(0.75, 1.15))
    angle = float(rng.uniform(-10.0, 10.0))
    tx = float(rng.uniform(-0.12, 0.12)) * w
    ty = float(rng.uniform(-0.12, 0.12)) * h

    mat = cv2.getRotationMatrix2D((w * 0.5, h * 0.5), angle, scale)
    mat[0, 2] += tx
    mat[1, 2] += ty
    warped = cv2.warpAffine(gt_512, mat, (w, h), flags=cv2.INTER_LINEAR, borderMode=cv2.BORDER_REFLECT_101)

    aspect = float(rng.uniform(0.60, 1.50))  # width / height
    if aspect < 1.0:
        crop_h = h
        crop_w = max(64, int(round(h * aspect)))
    else:
        crop_w = w
        crop_h = max(64, int(round(w / aspect)))
    x0 = (w - crop_w) // 2
    y0 = (h - crop_h) // 2

    pts = FFHQ_CANONICAL_5PT_NORM * np.array([float(w), float(h)], dtype=np.float32)
    pts_w = pts @ mat[:, :2].T.astype(np.float32) + mat[:, 2].astype(np.float32)
    pts_c = pts_w - np.array([float(x0), float(y0)], dtype=np.float32)
    return warped[y0:y0 + crop_h, x0:x0 + crop_w].copy(), pts_c


def _apply_geometry_warp_crop(gt_512: np.ndarray, rng: np.random.Generator) -> np.ndarray:
    """Backward-compatible wrapper returning only the warped & cropped RGB array."""
    crop, _ = _apply_geometry_warp_crop_with_meta(gt_512, rng)
    return crop


def _prepare_mask_exact(mask_f32: np.ndarray, out_size: int = 256) -> np.ndarray:
    """Apply exact center-pad (0.0 outside observed) + bilinear resize to a (H, W) float32 mask."""
    h, w = mask_f32.shape[:2]
    side = max(h, w)
    top, left = (side - h) // 2, (side - w) // 2
    canvas = np.zeros((side, side), dtype=np.float32)
    canvas[top:top + h, left:left + w] = mask_f32.astype(np.float32, copy=False)
    resized = cv2.resize(canvas, (out_size, out_size), interpolation=cv2.INTER_LINEAR)
    return np.clip(resized, 0.0, 1.0).astype(np.float32)


def _apply_regime_v34(
    img_u8: np.ndarray,
    regime: str,
    rng: np.random.Generator,
) -> Tuple[np.ndarray, Dict[str, Any]]:
    """V3.4 degradation generator producing exact x_struct_clean and x_photo_tgt targets,
    100%-strength NN pixelation grids, and 1px + 2-4px coarse raw RGB confetti.
    """
    f0 = _to_f32(img_u8)
    f = f0.copy()
    h, w = f.shape[:2]
    sp_scale = max(h, w) / 256.0

    exposure_sev = 0.0
    blur_sev = 0.0
    noise_sev = 0.0
    codec_sev = 0.0
    is_nn_pixelated = False
    is_coarse_confetti = False
    x_struct_clean_u8: Optional[np.ndarray] = None
    x_photo_tgt_u8: Optional[np.ndarray] = None

    if regime == "clear":
        if rng.random() < 0.35:
            q = int(rng.integers(95, 101))
            out_u8 = _jpeg_compress(img_u8, q, subsampling=0)
        else:
            out_u8 = img_u8.copy()
        x_struct_clean_u8 = img_u8.copy()
        x_photo_tgt_u8 = img_u8.copy()

    elif regime == "global_defocus_blur":
        f, sig = _gaussian_or_disk_blur(f, rng, 0.8 * sp_scale, 6.0 * sp_scale)
        blur_sev = float(np.clip((sig / sp_scale - 0.5) / 3.5, 0.0, 1.0))
        out_u8 = _to_u8(f)
        # Pure optical blur: pre_net & photo_net must stay exact identity!
        x_struct_clean_u8 = out_u8.copy()
        x_photo_tgt_u8 = out_u8.copy()

    elif regime == "directional_motion_blur":
        lmin = max(3, int(round(5 * sp_scale)))
        lmax = max(lmin + 2, int(round(19 * sp_scale)))
        f, length = _motion_blur(f, rng, lmin, lmax)
        blur_sev = float(np.clip(((length / sp_scale) - 3.0) / 16.0, 0.0, 1.0))
        out_u8 = _to_u8(f)
        x_struct_clean_u8 = out_u8.copy()
        x_photo_tgt_u8 = out_u8.copy()

    elif regime == "local_partial_blur":
        yy, xx = np.mgrid[0:h, 0:w].astype(np.float32)
        yn = (yy / max(h - 1, 1)) * 2.0 - 1.0
        xn = (xx / max(w - 1, 1)) * 2.0 - 1.0
        if rng.random() < 0.50:
            sig_x = float(rng.uniform(1.8 * sp_scale, 6.0 * sp_scale))
            lo_y = max(1.8 * sp_scale, sig_x * 0.7)
            hi_y = min(6.0 * sp_scale, sig_x * 1.3)
            sig_y = sig_x if (rng.random() < 0.65 or hi_y <= lo_y) else float(rng.uniform(lo_y, hi_y))
            kx = max(3, int(round(sig_x * 4.0)) | 1)
            ky = max(3, int(round(sig_y * 4.0)) | 1)
            blurred = np.clip(
                cv2.GaussianBlur(f, (kx, ky), sigmaX=sig_x, sigmaY=sig_y, borderType=cv2.BORDER_REFLECT_101),
                0.0,
                1.0,
            )
            sig = 0.5 * (sig_x + sig_y)
            rx = float(rng.uniform(0.42, 0.72))
            ry = float(rng.uniform(0.45, 0.76))
            cx = float(rng.uniform(-0.08, 0.08))
            cy = float(rng.uniform(-0.06, 0.10))
            dist = np.sqrt(((xn - cx) / rx) ** 2 + ((yn - cy) / ry) ** 2)
            m = np.clip((1.15 - dist) / 0.45, 0.0, 1.0)
        else:
            blurred, sig = _gaussian_or_disk_blur(f, rng, 1.8 * sp_scale, 3.8 * sp_scale)
            mode = int(rng.integers(0, 3))
            if mode == 0:
                rx = float(rng.uniform(0.35, 0.65))
                ry = float(rng.uniform(0.35, 0.65))
                dist = np.sqrt((xn / rx) ** 2 + (yn / ry) ** 2)
                m = np.clip((dist - 0.75) / 0.55, 0.0, 1.0)
            elif mode == 1:
                cx = float(rng.uniform(-0.5, 0.5))
                cy = float(rng.uniform(-0.5, 0.5))
                r = float(rng.uniform(0.35, 0.75))
                m = np.exp(-((xn - cx) ** 2 + (yn - cy) ** 2) / (2.0 * r * r))
            else:
                ang = float(rng.uniform(0.0, 2.0 * math.pi))
                proj = xn * math.cos(ang) + yn * math.sin(ang)
                m = np.clip((proj + 0.25) / 0.5, 0.0, 1.0)
        m3 = m[:, :, None].astype(np.float32)
        f = f * (1.0 - m3) + blurred * m3
        blur_sev = float(np.clip(float(m.mean()) * (sig / sp_scale) / 3.0, 0.15, 0.9))
        out_u8 = _to_u8(f)
        x_struct_clean_u8 = out_u8.copy()
        x_photo_tgt_u8 = out_u8.copy()

    elif regime == "compound_cctv_codec":
        f_opt, sig = _gaussian_or_disk_blur(f0, rng, 0.6 * sp_scale, 2.0 * sp_scale)
        blur_sev = float(np.clip((sig / sp_scale) / 3.0, 0.0, 1.0))
        if rng.random() < 0.20:
            # 20% 100%-strength NN pixelation grid sub-mode
            is_nn_pixelated = True
            f_in, lr_side = _nn_pixelate(f_opt, rng, 14, 48)
            f_clean = _downsample_upsample_smooth(f_opt, lr_side)
            if rng.random() < 0.60:
                q1 = int(rng.integers(45, 96))
                out_u8 = _jpeg_compress(_to_u8(f_in), q1, subsampling=2)
            else:
                out_u8 = _to_u8(f_in)
            codec_sev = float(np.clip(1.0 - (lr_side - 14.0) / 40.0, 0.45, 1.0))
        else:
            f_in, lr_side = _downsample_upsample(f_opt, rng, 24, 96)
            f_clean = _downsample_upsample_smooth(f_opt, lr_side)
            q1 = int(rng.integers(15, 76))
            out_u8 = _jpeg_compress(_to_u8(f_in), q1, subsampling=2)
            if rng.random() < 0.20:
                lo_q2 = max(15, q1 - 10)
                hi_q2 = max(lo_q2 + 1, min(85, q1 + 15))
                q2 = int(rng.integers(lo_q2, hi_q2))
                out_u8 = _jpeg_compress(out_u8, q2, subsampling=2)
            codec_sev = float(
                np.clip(0.5 * (1.0 - (lr_side - 24.0) / 72.0) + 0.5 * (1.0 - (q1 - 15.0) / 60.0), 0.1, 1.0)
            )
        x_struct_clean_u8 = _to_u8(f_clean)
        x_photo_tgt_u8 = x_struct_clean_u8.copy()

    elif regime == "severe_lr_survface":
        f_opt = f0.copy()
        if rng.random() < 0.70:
            f_opt, sig = _gaussian_or_disk_blur(f_opt, rng, 0.4 * sp_scale, 1.2 * sp_scale)
            blur_sev = float(np.clip((sig / sp_scale) / 3.0, 0.0, 0.6))
        f_in = f_opt.copy()
        if rng.random() < 0.35:
            exp = float(rng.uniform(0.55, 1.25))
            f_in = np.clip(f_in * exp, 0.0, 1.0)
            if exp < 0.75:
                exposure_sev = float(np.clip((0.78 - exp) / 0.40, 0.10, 0.65))
            else:
                exposure_sev = float(np.clip(abs(exp - 1.0) / 0.35, 0.0, 0.40))
        if rng.random() < 0.40:
            # 40% 100%-strength NN pixelation grid sub-mode (14..48px -> 256px)
            is_nn_pixelated = True
            f_in, lr_side = _nn_pixelate(f_in, rng, 14, 48)
            f_clean = _downsample_upsample_smooth(f_opt, lr_side)
            if rng.random() < 0.55:
                q = int(rng.integers(45, 96))
                out_u8 = _jpeg_compress(_to_u8(f_in), q, subsampling=2)
            else:
                out_u8 = _to_u8(f_in)
            codec_sev = float(np.clip(1.0 - (lr_side - 14.0) / 40.0, 0.50, 1.0))
        else:
            f_in, lr_side = _downsample_upsample(f_in, rng, 12, 40)
            f_clean = _downsample_upsample_smooth(f_opt, lr_side)
            if rng.random() < 0.40:
                f_in, nsig = _add_sensor_noise(f_in, rng, 0.004, 0.018, poisson=False)
                noise_sev = float(np.clip(nsig / 0.04, 0.0, 1.0))
            if rng.random() < 0.70:
                q = int(rng.integers(25, 76))
                out_u8 = _jpeg_compress(_to_u8(f_in), q, subsampling=2)
                codec_sev = float(
                    np.clip(0.65 * (1.0 - (lr_side - 12.0) / 28.0) + 0.35 * (1.0 - (q - 25.0) / 50.0), 0.35, 1.0)
                )
            else:
                out_u8 = _to_u8(f_in)
                codec_sev = float(np.clip(1.0 - (lr_side - 12.0) / 28.0, 0.35, 1.0))
        x_struct_clean_u8 = _to_u8(f_clean)
        x_photo_tgt_u8 = x_struct_clean_u8.copy()

    elif regime == "lowlight_sensor_noise":
        exp = float(rng.uniform(0.16, 0.72))
        if rng.random() < 0.35:
            gamma = 1.0
        else:
            max_gamma = max(1.0, math.log(0.08 / max(exp, 1e-4)) / math.log(0.5))
            hi_g = min(2.20, max_gamma)
            gamma = float(rng.uniform(1.0, hi_g)) if hi_g > 1.0 else 1.0
        f = np.clip(np.power(np.clip(f0, 0.0, 1.0), gamma) * exp, 0.0, 1.0)
        u_sub = rng.random()
        if u_sub < 0.25:
            # 25% raw RGB confetti (50% 1px, 50% 2-4px coarse)
            f, nsig, is_coarse_confetti = _add_rgb_confetti(f, rng, 0.03, 0.095, 0.01, 0.05, coarse_prob=0.50)
            noise_sev = float(np.clip(nsig / 0.05, 0.50, 1.0))
        elif u_sub < 0.50:
            if rng.random() < 0.30:
                noise_sev = 0.0
            else:
                f, nsig = _add_sensor_noise(f, rng, 0.002, 0.008, poisson=False, chroma_boost=1.3)
                noise_sev = float(np.clip(nsig / 0.04, 0.0, 0.20))
        else:
            f, nsig = _add_sensor_noise(f, rng, 0.004, 0.035, poisson=True, chroma_boost=1.5)
            noise_sev = float(np.clip(nsig / 0.04, 0.1, 1.0))
        exposure_sev = float(np.clip(1.0 - (exp - 0.16) / 0.62, 0.15, 1.0))
        out_u8 = _to_u8(f)
        x_struct_clean_u8 = img_u8.copy()
        x_photo_tgt_u8 = img_u8.copy()

    elif regime == "joint_lowlight_blur_codec":
        f_opt = f0.copy()
        if rng.random() < 0.40:
            exp = float(rng.uniform(0.16, 0.50))
            if rng.random() < 0.35:
                gamma = 1.0
            else:
                max_gamma = max(1.0, math.log(0.08 / max(exp, 1e-4)) / math.log(0.5))
                hi_g = min(1.85, max_gamma)
                gamma = float(rng.uniform(1.0, hi_g)) if hi_g > 1.0 else 1.0
            if rng.random() < 0.50:
                f_opt, sig = _gaussian_or_disk_blur(f_opt, rng, 0.4 * sp_scale, 1.4 * sp_scale)
                blur_sev = float(np.clip((sig / sp_scale) / 3.0, 0.05, 0.7))
            f_in = np.clip(np.power(np.clip(f_opt, 0.0, 1.0), gamma) * exp, 0.0, 1.0)
            f_in, lr_side = _downsample_upsample(f_in, rng, 12, 36)
            f_clean = _downsample_upsample_smooth(f_opt, lr_side)
            if rng.random() < 0.50:
                noise_sev = 0.0
            else:
                f_in, nsig = _add_sensor_noise(f_in, rng, 0.0005, 0.004, poisson=False, chroma_boost=1.2)
                noise_sev = float(np.clip(nsig / 0.04, 0.0, 0.10))
            if rng.random() < 0.70:
                q = int(rng.integers(35, 86))
                out_u8 = _jpeg_compress(_to_u8(f_in), q, subsampling=2)
                codec_sev = float(
                    np.clip(0.65 * (1.0 - (lr_side - 12.0) / 24.0) + 0.35 * (1.0 - (q - 35.0) / 50.0), 0.35, 1.0)
                )
            else:
                out_u8 = _to_u8(f_in)
                codec_sev = float(np.clip(1.0 - (lr_side - 12.0) / 24.0, 0.35, 1.0))
            exposure_sev = float(np.clip(1.0 - (exp - 0.16) / 0.45, 0.30, 1.0))
        else:
            exp = float(rng.uniform(0.22, 0.55))
            if rng.random() < 0.35:
                gamma = 1.0
            else:
                max_gamma = max(1.0, math.log(0.08 / max(exp, 1e-4)) / math.log(0.5))
                hi_g = min(1.90, max_gamma)
                gamma = float(rng.uniform(1.0, hi_g)) if hi_g > 1.0 else 1.0
            if rng.random() < 0.50:
                f_opt, sig = _gaussian_or_disk_blur(f_opt, rng, 0.7 * sp_scale, 2.4 * sp_scale)
                blur_sev = float(np.clip((sig / sp_scale) / 3.0, 0.1, 1.0))
            else:
                lmin = max(3, int(round(5 * sp_scale)))
                lmax = max(lmin + 2, int(round(13 * sp_scale)))
                f_opt, length = _motion_blur(f_opt, rng, lmin, lmax)
                blur_sev = float(np.clip((length / sp_scale) / 16.0, 0.1, 1.0))
            f_in = np.clip(np.power(np.clip(f_opt, 0.0, 1.0), gamma) * exp, 0.0, 1.0)
            f_in, lr_side = _downsample_upsample(f_in, rng, 20, 64)
            f_clean = _downsample_upsample_smooth(f_opt, lr_side)
            f_in, nsig = _add_sensor_noise(f_in, rng, 0.006, 0.030, poisson=True, chroma_boost=1.4)
            q = int(rng.integers(20, 66))
            out_u8 = _jpeg_compress(_to_u8(f_in), q, subsampling=2)
            exposure_sev = float(np.clip(1.0 - (exp - 0.18) / 0.45, 0.25, 1.0))
            noise_sev = float(np.clip(nsig / 0.04, 0.1, 1.0))
            codec_sev = float(
                np.clip(0.5 * (1.0 - (lr_side - 20.0) / 44.0) + 0.5 * (1.0 - (q - 20.0) / 45.0), 0.2, 1.0)
            )
        x_struct_clean_u8 = _to_u8(f_clean)
        x_photo_tgt_u8 = x_struct_clean_u8.copy()

    elif regime == "daylight_sensor_noise":
        if rng.random() < 0.60:
            exp = 1.0
            exposure_sev = 0.0
        else:
            exp = float(rng.uniform(0.85, 1.20))
            exposure_sev = float(np.clip(abs(exp - 1.0) / 0.35, 0.0, 0.4))
        f = np.clip(f0 * exp, 0.0, 1.0)
        if rng.random() < 0.55:
            # 55% raw RGB confetti (50% 1px, 50% 2-4px coarse)
            f, nsig, is_coarse_confetti = _add_rgb_confetti(f, rng, 0.03, 0.095, 0.01, 0.05, coarse_prob=0.50)
            noise_sev = float(np.clip(nsig / 0.05, 0.50, 1.0))
        else:
            f, nsig = _add_sensor_noise(f, rng, 0.010, 0.050, poisson=True, chroma_boost=1.4, impulse_prob=0.25)
            noise_sev = float(np.clip(nsig / 0.05, 0.2, 1.0))
        out_u8 = _to_u8(f)
        x_struct_clean_u8 = img_u8.copy()
        x_photo_tgt_u8 = img_u8.copy()

    elif regime == "h264_macroblock_interlace":
        if rng.random() < 0.40:
            # 40% 100%-strength NN pixelation grid sub-mode
            is_nn_pixelated = True
            f_in, lr_side = _nn_pixelate(f0, rng, 14, 48)
            f_clean = _downsample_upsample_smooth(f0, lr_side)
            if rng.random() < 0.65:
                q = int(rng.integers(45, 96))
                out_u8 = _jpeg_compress(_to_u8(f_in), q, subsampling=2)
            else:
                out_u8 = _to_u8(f_in)
            codec_sev = float(np.clip(1.0 - (lr_side - 14.0) / 40.0, 0.50, 1.0))
            blur_sev = 0.15
            x_struct_clean_u8 = _to_u8(f_clean)
            x_photo_tgt_u8 = x_struct_clean_u8.copy()
        else:
            f_in = f0.copy()
            if rng.random() < 0.60:
                max_shift = min(max(1, w - 1), max(2, int(round(4 * sp_scale))))
                shift_px = int(rng.integers(1, max_shift + 1))
                interlaced = f_in.copy()
                interlaced[1::2, shift_px:] = f_in[1::2, :-shift_px]
                f_in = interlaced
            if rng.random() < 0.35:
                blk = max(2, int(round(int(rng.choice([4, 8])) * sp_scale)))
                bh, bw = max(8, h // blk), max(8, w // blk)
                mosaic = cv2.resize(
                    cv2.resize(f_in, (bw, bh), interpolation=cv2.INTER_AREA), (w, h), interpolation=cv2.INTER_NEAREST
                )
                alpha_m = float(rng.uniform(0.25, 0.65))
                f_in = f_in * (1.0 - alpha_m) + mosaic * alpha_m
            q = int(rng.integers(8, 32))
            out_u8 = _jpeg_compress(_to_u8(f_in), q, subsampling=2)
            f_clean = f0
            if rng.random() < 0.50:
                f2 = _to_f32(out_u8)
                f2, lr_side = _downsample_upsample(f2, rng, 48, 96)
                out_u8 = _jpeg_compress(_to_u8(f2), max(10, q + 4), subsampling=2)
                f_clean = _downsample_upsample_smooth(f0, lr_side)
            codec_sev = float(np.clip(1.0 - (q - 8.0) / 30.0, 0.4, 1.0))
            blur_sev = 0.25
            x_struct_clean_u8 = _to_u8(f_clean)
            x_photo_tgt_u8 = x_struct_clean_u8.copy()

    elif regime == "haze_colorcast_glare":
        t_val = float(rng.uniform(0.45, 0.90))
        airlight = rng.uniform(0.60, 0.95, size=(1, 1, 3)).astype(np.float32)
        f = f0 * t_val + airlight * (1.0 - t_val)
        wb = rng.choice([
            np.array([1.15, 1.02, 0.78], dtype=np.float32),
            np.array([0.85, 1.08, 0.92], dtype=np.float32),
            np.array([0.82, 0.95, 1.18], dtype=np.float32),
        ])
        wb_strength = float(rng.uniform(0.35, 1.0))
        wb_eff = (1.0 - wb_strength) + wb_strength * wb
        f = np.clip(f * wb_eff.reshape(1, 1, 3), 0.0, 1.0)
        if rng.random() < 0.60:
            yy, xx = np.mgrid[0:h, 0:w].astype(np.float32)
            cx = float(rng.uniform(0.0, w))
            cy = float(rng.uniform(0.0, h * 0.6))
            rad = float(rng.uniform(0.25, 0.60)) * max(h, w)
            glare = np.exp(-((xx - cx) ** 2 + (yy - cy) ** 2) / (2.0 * rad * rad))
            glare_amp = float(rng.uniform(0.15, 0.45))
            f = np.clip(f + glare[:, :, None] * glare_amp, 0.0, 1.0)
        exposure_sev = float(np.clip((1.0 - t_val) / 0.55 + 0.25 * wb_strength, 0.2, 1.0))
        out_u8 = _to_u8(f)
        x_struct_clean_u8 = img_u8.copy()
        x_photo_tgt_u8 = img_u8.copy()

    elif regime == "mild_transition":
        f_opt = f0.copy()
        if rng.random() < 0.60:
            f_opt, sig = _gaussian_or_disk_blur(f_opt, rng, 0.35 * sp_scale, 0.85 * sp_scale)
            blur_sev = float(np.clip((sig / sp_scale) / 3.0, 0.05, 0.30))
        f_in = f_opt.copy()
        f_clean = f_opt.copy()
        if rng.random() < 0.60:
            f_in, lr_side = _downsample_upsample(f_in, rng, 96, 160)
            f_clean = _downsample_upsample_smooth(f_opt, lr_side)
            codec_sev = max(codec_sev, float(np.clip((160.0 - lr_side) / 160.0, 0.05, 0.35)))
        q = int(rng.integers(70, 91))
        out_u8 = _jpeg_compress(_to_u8(f_in), q, subsampling=2)
        codec_sev = max(codec_sev, float(np.clip((95.0 - q) / 80.0, 0.05, 0.30)))
        x_struct_clean_u8 = _to_u8(f_clean)
        x_photo_tgt_u8 = x_struct_clean_u8.copy()

    else:
        raise ValueError(f"Unknown CCTV degradation regime: {regime}")

    sev_max = max(exposure_sev, blur_sev, noise_sev, codec_sev)
    sufficiency = float(
        np.clip(1.0 - 0.55 * codec_sev - 0.25 * blur_sev - 0.15 * exposure_sev - 0.10 * noise_sev, 0.05, 1.0)
    )
    if regime == "clear":
        sufficiency = 1.0

    targets: Dict[str, Any] = {
        "regime": regime,
        "is_clear": regime == "clear",
        "is_lowlight": regime in ("lowlight_sensor_noise", "joint_lowlight_blur_codec"),
        "is_nn_pixelated": bool(is_nn_pixelated),
        "is_coarse_confetti": bool(is_coarse_confetti),
        "exposure": float(exposure_sev),
        "blur": float(blur_sev),
        "noise": float(noise_sev),
        "codec_lr": float(codec_sev),
        "grid": float(codec_sev),
        "severity": float(sev_max),
        "sufficiency": float(sufficiency),
        "_x_struct_clean_native": x_struct_clean_u8,
        "_x_photo_tgt_native": x_photo_tgt_u8,
    }
    return out_u8, targets


def _apply_regime(
    img_u8: np.ndarray,
    regime: str,
    rng: np.random.Generator,
    calibration: str = "v30",
) -> Tuple[np.ndarray, Dict[str, Any]]:
    if calibration not in ("v30", "v31", "v32", "v33", "v34"):
        raise ValueError(f"Unknown degradation calibration '{calibration}'. Expected 'v30', 'v31', 'v32', 'v33', or 'v34'.")
    if calibration == "v34":
        return _apply_regime_v34(img_u8, regime, rng)
    use_v31_regimes = calibration in ("v31", "v32", "v33")
    is_v33 = calibration == "v33"
    f = _to_f32(img_u8)
    h, w = f.shape[:2]
    # Scale spatial blur/LR parameters relative to 256px reference
    sp_scale = max(h, w) / 256.0

    exposure_sev = 0.0
    blur_sev = 0.0
    noise_sev = 0.0
    codec_sev = 0.0
    blur_mask: Optional[np.ndarray] = np.zeros((h, w), dtype=np.float32) if is_v33 else None

    if regime == "clear":
        if rng.random() < 0.35:
            q = int(rng.integers(95, 101))
            out_u8 = _jpeg_compress(img_u8, q, subsampling=0)
        else:
            out_u8 = img_u8.copy()

    elif regime == "global_defocus_blur":
        sig_hi = 6.0 if is_v33 else 3.8
        f, sig = _gaussian_or_disk_blur(f, rng, 0.8 * sp_scale, sig_hi * sp_scale)
        blur_sev = float(np.clip((sig / sp_scale - 0.5) / 3.5, 0.0, 1.0))
        if blur_mask is not None:
            blur_mask.fill(blur_sev)
        out_u8 = _to_u8(f)

    elif regime == "directional_motion_blur":
        lmin = max(3, int(round(5 * sp_scale)))
        lmax = max(lmin + 2, int(round(19 * sp_scale)))
        f, length = _motion_blur(f, rng, lmin, lmax)
        blur_sev = float(np.clip(((length / sp_scale) - 3.0) / 16.0, 0.0, 1.0))
        if blur_mask is not None:
            blur_mask.fill(blur_sev)
        out_u8 = _to_u8(f)

    elif regime == "local_partial_blur":
        yy, xx = np.mgrid[0:h, 0:w].astype(np.float32)
        yn = (yy / max(h - 1, 1)) * 2.0 - 1.0
        xn = (xx / max(w - 1, 1)) * 2.0 - 1.0
        if is_v33 and rng.random() < 0.50:
            # V3.3: 50% center-face-only Gaussian blur (sigma 1.8..6.0), sharp outer hair/ring
            sig_x = float(rng.uniform(1.8 * sp_scale, 6.0 * sp_scale))
            sig_y = sig_x if rng.random() < 0.65 else float(rng.uniform(max(1.8 * sp_scale, sig_x * 0.7), min(6.0 * sp_scale, sig_x * 1.3)))
            kx = max(3, int(round(sig_x * 4.0)) | 1)
            ky = max(3, int(round(sig_y * 4.0)) | 1)
            blurred = np.clip(
                cv2.GaussianBlur(f, (kx, ky), sigmaX=sig_x, sigmaY=sig_y, borderType=cv2.BORDER_REFLECT_101),
                0.0,
                1.0,
            )
            sig = 0.5 * (sig_x + sig_y)
            rx = float(rng.uniform(0.42, 0.72))
            ry = float(rng.uniform(0.45, 0.76))
            cx = float(rng.uniform(-0.08, 0.08))
            cy = float(rng.uniform(-0.06, 0.10))
            dist = np.sqrt(((xn - cx) / rx) ** 2 + ((yn - cy) / ry) ** 2)
            m = np.clip((1.15 - dist) / 0.45, 0.0, 1.0)
        else:
            blurred, sig = _gaussian_or_disk_blur(f, rng, 1.8 * sp_scale, 3.8 * sp_scale)
            mode = int(rng.integers(0, 3))
            if mode == 0:
                # Outer ellipse blur (center sharp)
                rx = float(rng.uniform(0.35, 0.65))
                ry = float(rng.uniform(0.35, 0.65))
                dist = np.sqrt((xn / rx) ** 2 + (yn / ry) ** 2)
                m = np.clip((dist - 0.75) / 0.55, 0.0, 1.0)
            elif mode == 1:
                # Off-center Gaussian blob blur
                cx = float(rng.uniform(-0.5, 0.5))
                cy = float(rng.uniform(-0.5, 0.5))
                r = float(rng.uniform(0.35, 0.75))
                m = np.exp(-((xn - cx) ** 2 + (yn - cy) ** 2) / (2.0 * r * r))
            else:
                # Half-face soft gradient blur
                ang = float(rng.uniform(0.0, 2.0 * math.pi))
                proj = xn * math.cos(ang) + yn * math.sin(ang)
                m = np.clip((proj + 0.25) / 0.5, 0.0, 1.0)
        m3 = m[:, :, None].astype(np.float32)
        f = f * (1.0 - m3) + blurred * m3
        blur_sev = float(np.clip(float(m.mean()) * (sig / sp_scale) / 3.0, 0.15, 0.9))
        if blur_mask is not None:
            blur_mask = m.astype(np.float32)
        out_u8 = _to_u8(f)

    elif regime == "compound_cctv_codec":
        f, sig = _gaussian_or_disk_blur(f, rng, 0.6 * sp_scale, 2.0 * sp_scale)
        lr_min = 24 if use_v31_regimes else 32
        f, lr_side = _downsample_upsample(f, rng, lr_min, 96)
        q1 = int(rng.integers(15, 76))
        out_u8 = _jpeg_compress(_to_u8(f), q1, subsampling=2)
        if rng.random() < 0.20:
            lo_q2 = max(15, q1 - 10)
            hi_q2 = max(lo_q2 + 1, min(85, q1 + 15))
            q2 = int(rng.integers(lo_q2, hi_q2))
            out_u8 = _jpeg_compress(out_u8, q2, subsampling=2)
        blur_sev = float(np.clip((sig / sp_scale) / 3.0, 0.0, 1.0))
        lr_span = float(96 - lr_min)
        codec_sev = float(
            np.clip(0.5 * (1.0 - (lr_side - float(lr_min)) / lr_span) + 0.5 * (1.0 - (q1 - 15.0) / 60.0), 0.1, 1.0)
        )

    elif regime == "severe_lr_survface":
        if use_v31_regimes:
            # Pre-downsample natural brightness jitter (35%)
            if rng.random() < 0.35:
                exp = float(rng.uniform(0.55, 1.25))
                f = np.clip(f * exp, 0.0, 1.0)
                if exp < 0.75:
                    exposure_sev = float(np.clip((0.78 - exp) / 0.40, 0.10, 0.65))
            if rng.random() < 0.7:
                f, sig = _gaussian_or_disk_blur(f, rng, 0.4 * sp_scale, 1.2 * sp_scale)
                blur_sev = float(np.clip((sig / sp_scale) / 3.0, 0.0, 0.6))
            f, lr_side = _downsample_upsample(f, rng, 12, 40)
            if rng.random() < 0.40:
                f, nsig = _add_sensor_noise(f, rng, 0.004, 0.018, poisson=False)
                noise_sev = float(np.clip(nsig / 0.04, 0.0, 1.0))
            if rng.random() < 0.70:
                q = int(rng.integers(25, 76))
                out_u8 = _jpeg_compress(_to_u8(f), q, subsampling=2)
                codec_sev = float(
                    np.clip(0.65 * (1.0 - (lr_side - 12.0) / 28.0) + 0.35 * (1.0 - (q - 25.0) / 50.0), 0.35, 1.0)
                )
            else:
                out_u8 = _to_u8(f)
                codec_sev = float(np.clip(1.0 - (lr_side - 12.0) / 28.0, 0.35, 1.0))
        else:
            if rng.random() < 0.7:
                f, sig = _gaussian_or_disk_blur(f, rng, 0.4 * sp_scale, 1.2 * sp_scale)
                blur_sev = float(np.clip((sig / sp_scale) / 3.0, 0.0, 0.6))
            f, lr_side = _downsample_upsample(f, rng, 16, 40)
            if rng.random() < 0.6:
                f, nsig = _add_sensor_noise(f, rng, 0.004, 0.018, poisson=False)
                noise_sev = float(np.clip(nsig / 0.04, 0.0, 1.0))
            q = int(rng.integers(25, 76))
            out_u8 = _jpeg_compress(_to_u8(f), q, subsampling=2)
            codec_sev = float(np.clip(0.65 * (1.0 - (lr_side - 16.0) / 24.0) + 0.35 * (1.0 - (q - 25.0) / 50.0), 0.35, 1.0))

    elif regime == "lowlight_sensor_noise":
        if use_v31_regimes:
            exp_hi = 0.72 if is_v33 else 0.65
            exp = float(rng.uniform(0.16, exp_hi))
            if rng.random() < 0.35:
                gamma = 1.0
            else:
                max_gamma = max(1.0, math.log(0.08 / max(exp, 1e-4)) / math.log(0.5))
                hi_g = min(2.20, max_gamma)
                gamma = float(rng.uniform(1.0, hi_g)) if hi_g > 1.0 else 1.0
            f = np.clip(np.power(np.clip(f, 0.0, 1.0), gamma) * exp, 0.0, 1.0)
            if rng.random() < 0.35:
                # Clean / low-noise sharp lowlight sub-mode (breaks V3OutputComposer bypass shortcut)
                if is_v33:
                    if rng.random() < 0.25:
                        noise_sev = 0.0
                    else:
                        f, nsig = _add_sensor_noise(f, rng, 0.002, 0.008, poisson=False, chroma_boost=1.3)
                        noise_sev = float(np.clip(nsig / 0.04, 0.0, 0.20))
                else:
                    if rng.random() < 0.40:
                        noise_sev = 0.0
                    else:
                        f, nsig = _add_sensor_noise(f, rng, 0.001, 0.005, poisson=False, chroma_boost=1.3)
                        noise_sev = float(np.clip(nsig / 0.04, 0.0, 0.12))
            else:
                f, nsig = _add_sensor_noise(f, rng, 0.004, 0.035, poisson=True, chroma_boost=1.5)
                noise_sev = float(np.clip(nsig / 0.04, 0.1, 1.0))
            exposure_sev = float(np.clip(1.0 - (exp - 0.16) / (0.62 if is_v33 else 0.55), 0.15 if is_v33 else 0.20, 1.0))
            out_u8 = _to_u8(f)
        else:
            exp = float(rng.uniform(0.18, 0.55))
            if rng.random() < 0.35:
                gamma = 1.0
            else:
                max_gamma = max(1.0, math.log(0.08 / max(exp, 1e-4)) / math.log(0.5))
                gamma = float(rng.uniform(1.0, min(2.20, max_gamma)))
            f = np.clip(np.power(np.clip(f, 0.0, 1.0), gamma) * exp, 0.0, 1.0)
            f, nsig = _add_sensor_noise(f, rng, 0.004, 0.035, poisson=True, chroma_boost=1.5)
            exposure_sev = float(np.clip(1.0 - (exp - 0.15) / 0.50, 0.25, 1.0))
            noise_sev = float(np.clip(nsig / 0.04, 0.1, 1.0))
            out_u8 = _to_u8(f)

    elif regime == "joint_lowlight_blur_codec":
        if use_v31_regimes and rng.random() < 0.40:
            # 40% Dark Smooth SurvFace sub-mode (12..36px, zero/tiny post-upsample noise)
            exp = float(rng.uniform(0.16, 0.50))
            if rng.random() < 0.35:
                gamma = 1.0
            else:
                max_gamma = max(1.0, math.log(0.08 / max(exp, 1e-4)) / math.log(0.5))
                hi_g = min(1.85, max_gamma)
                gamma = float(rng.uniform(1.0, hi_g)) if hi_g > 1.0 else 1.0
            f = np.clip(np.power(np.clip(f, 0.0, 1.0), gamma) * exp, 0.0, 1.0)
            if rng.random() < 0.50:
                f, sig = _gaussian_or_disk_blur(f, rng, 0.4 * sp_scale, 1.4 * sp_scale)
                blur_sev = float(np.clip((sig / sp_scale) / 3.0, 0.05, 0.7))
            f, lr_side = _downsample_upsample(f, rng, 12, 36)
            if rng.random() < 0.50:
                noise_sev = 0.0
            else:
                f, nsig = _add_sensor_noise(f, rng, 0.0005, 0.004, poisson=False, chroma_boost=1.2)
                noise_sev = float(np.clip(nsig / 0.04, 0.0, 0.10))
            if rng.random() < 0.70:
                q = int(rng.integers(35, 86))
                out_u8 = _jpeg_compress(_to_u8(f), q, subsampling=2)
                codec_sev = float(
                    np.clip(0.65 * (1.0 - (lr_side - 12.0) / 24.0) + 0.35 * (1.0 - (q - 35.0) / 50.0), 0.35, 1.0)
                )
            else:
                out_u8 = _to_u8(f)
                codec_sev = float(np.clip(1.0 - (lr_side - 12.0) / 24.0, 0.35, 1.0))
            exposure_sev = float(np.clip(1.0 - (exp - 0.16) / 0.45, 0.30, 1.0))
        else:
            exp = float(rng.uniform(0.22, 0.55))
            if rng.random() < 0.35:
                gamma = 1.0
            else:
                max_gamma = max(1.0, math.log(0.08 / max(exp, 1e-4)) / math.log(0.5))
                hi_g = min(1.90, max_gamma)
                gamma = float(rng.uniform(1.0, hi_g)) if hi_g > 1.0 else 1.0
            f = np.clip(np.power(np.clip(f, 0.0, 1.0), gamma) * exp, 0.0, 1.0)
            if rng.random() < 0.5:
                f, sig = _gaussian_or_disk_blur(f, rng, 0.7 * sp_scale, 2.4 * sp_scale)
                blur_sev = float(np.clip((sig / sp_scale) / 3.0, 0.1, 1.0))
            else:
                lmin = max(3, int(round(5 * sp_scale)))
                lmax = max(lmin + 2, int(round(13 * sp_scale)))
                f, length = _motion_blur(f, rng, lmin, lmax)
                blur_sev = float(np.clip((length / sp_scale) / 16.0, 0.1, 1.0))
            lr_min = 20 if use_v31_regimes else 28
            f, lr_side = _downsample_upsample(f, rng, lr_min, 64)
            f, nsig = _add_sensor_noise(f, rng, 0.006, 0.030, poisson=True, chroma_boost=1.4)
            q = int(rng.integers(20, 66))
            out_u8 = _jpeg_compress(_to_u8(f), q, subsampling=2)
            exposure_sev = float(np.clip(1.0 - (exp - 0.18) / 0.45, 0.25, 1.0))
            noise_sev = float(np.clip(nsig / 0.04, 0.1, 1.0))
            lr_span = float(64 - lr_min)
            codec_sev = float(
                np.clip(0.5 * (1.0 - (lr_side - float(lr_min)) / lr_span) + 0.5 * (1.0 - (q - 20.0) / 45.0), 0.2, 1.0)
            )

    elif regime == "daylight_sensor_noise":
        exp = float(rng.uniform(0.85, 1.25))
        f = np.clip(f * exp, 0.0, 1.0)
        if is_v33 and rng.random() < 0.30:
            # V3.3: Crisp uncorrelated per-channel RGB pixel-confetti noise submode (face_4_raw_sensor Direct)
            sigma = float(rng.uniform(0.035, 0.085))
            rgb_n = rng.normal(0.0, sigma, f.shape).astype(np.float32)
            f = f + rgb_n
            p_conf = float(rng.uniform(0.010, 0.040))
            u_ch = rng.random(f.shape)
            f[u_ch < (p_conf * 0.5)] = 0.0
            f[(u_ch >= (p_conf * 0.5)) & (u_ch < p_conf)] = 1.0
            f = np.clip(f, 0.0, 1.0)
            nsig = sigma
            noise_sev = float(np.clip(nsig / 0.05, 0.50, 1.0))
        else:
            f, nsig = _add_sensor_noise(f, rng, 0.010, 0.050, poisson=True, chroma_boost=1.4, impulse_prob=0.25)
            noise_sev = float(np.clip(nsig / 0.05, 0.2, 1.0))
        exposure_sev = float(np.clip(abs(exp - 1.0) / 0.35, 0.0, 0.4))
        out_u8 = _to_u8(f)

    elif regime == "h264_macroblock_interlace":
        # 8x8 - 16x16 macroblock quantization + odd/even scanline comb interlacing + optional mild mosaic
        if rng.random() < 0.6:
            max_shift = min(max(1, w - 1), max(2, int(round(4 * sp_scale))))
            shift_px = int(rng.integers(1, max_shift + 1))
            interlaced = f.copy()
            interlaced[1::2, shift_px:] = f[1::2, :-shift_px]
            f = interlaced
        if rng.random() < 0.35:
            blk = max(2, int(round(int(rng.choice([4, 8])) * sp_scale)))
            bh, bw = max(8, h // blk), max(8, w // blk)
            mosaic = cv2.resize(cv2.resize(f, (bw, bh), interpolation=cv2.INTER_AREA), (w, h), interpolation=cv2.INTER_NEAREST)
            alpha_m = float(rng.uniform(0.25, 0.65))
            f = f * (1.0 - alpha_m) + mosaic * alpha_m
        # Heavy DCT macroblocking (simulating libx264 CRF 30-46)
        q = int(rng.integers(8, 32))
        out_u8 = _jpeg_compress(_to_u8(f), q, subsampling=2)
        if rng.random() < 0.5:
            # 16x16 block boundary emphasis via slight quantized chroma bleed
            f2 = _to_f32(out_u8)
            f2, _ = _downsample_upsample(f2, rng, 48, 96)
            out_u8 = _jpeg_compress(_to_u8(f2), max(10, q + 4), subsampling=2)
        codec_sev = float(np.clip(1.0 - (q - 8.0) / 30.0, 0.4, 1.0))
        blur_sev = 0.25

    elif regime == "haze_colorcast_glare":
        # Atmospheric scattering I = J*t + A*(1 - t)
        t_val = float(rng.uniform(0.45, 0.90))
        airlight = rng.uniform(0.60, 0.95, size=(1, 1, 3)).astype(np.float32)
        f = f * t_val + airlight * (1.0 - t_val)
        # Fluorescent / sodium white-balance color cast
        wb = rng.choice([
            np.array([1.15, 1.02, 0.78], dtype=np.float32),  # sodium warm
            np.array([0.85, 1.08, 0.92], dtype=np.float32),  # fluorescent green
            np.array([0.82, 0.95, 1.18], dtype=np.float32),  # cool IR/blue
        ])
        wb_strength = float(rng.uniform(0.35, 1.0))
        wb_eff = (1.0 - wb_strength) + wb_strength * wb
        f = np.clip(f * wb_eff.reshape(1, 1, 3), 0.0, 1.0)
        # Optional lens glare bloom
        if rng.random() < 0.6:
            yy, xx = np.mgrid[0:h, 0:w].astype(np.float32)
            cx = float(rng.uniform(0.0, w))
            cy = float(rng.uniform(0.0, h * 0.6))
            rad = float(rng.uniform(0.25, 0.60)) * max(h, w)
            glare = np.exp(-((xx - cx) ** 2 + (yy - cy) ** 2) / (2.0 * rad * rad))
            glare_amp = float(rng.uniform(0.15, 0.45))
            f = np.clip(f + glare[:, :, None] * glare_amp, 0.0, 1.0)
        exposure_sev = float(np.clip((1.0 - t_val) / 0.55 + 0.25 * wb_strength, 0.2, 1.0))
        out_u8 = _to_u8(f)

    elif regime == "mild_transition":
        if rng.random() < 0.6:
            f, sig = _gaussian_or_disk_blur(f, rng, 0.35 * sp_scale, 0.85 * sp_scale)
            blur_sev = float(np.clip((sig / sp_scale) / 3.0, 0.05, 0.30))
        if rng.random() < 0.6:
            f, lr_side = _downsample_upsample(f, rng, 96, 160)
            codec_sev = max(codec_sev, float(np.clip((160.0 - lr_side) / 160.0, 0.05, 0.35)))
        q = int(rng.integers(70, 91))
        out_u8 = _jpeg_compress(_to_u8(f), q, subsampling=2)
        codec_sev = max(codec_sev, float(np.clip((95.0 - q) / 80.0, 0.05, 0.30)))

    else:
        raise ValueError(f"Unknown CCTV degradation regime: {regime}")

    sev_max = max(exposure_sev, blur_sev, noise_sev, codec_sev)
    sufficiency = float(np.clip(1.0 - 0.55 * codec_sev - 0.25 * blur_sev - 0.15 * exposure_sev - 0.10 * noise_sev, 0.05, 1.0))
    if regime == "clear":
        sufficiency = 1.0

    if blur_mask is not None and regime in (
        "compound_cctv_codec",
        "severe_lr_survface",
        "joint_lowlight_blur_codec",
        "h264_macroblock_interlace",
        "mild_transition",
    ):
        blur_mask.fill(float(np.clip(max(blur_sev, 0.85 * codec_sev), 0.0, 1.0)))

    targets: Dict[str, Any] = {
        "regime": regime,
        "is_clear": regime == "clear",
        "is_lowlight": regime in ("lowlight_sensor_noise", "joint_lowlight_blur_codec"),
        "exposure": float(exposure_sev),
        "blur": float(blur_sev),
        "noise": float(noise_sev),
        "codec_lr": float(codec_sev),
        "severity": float(sev_max),
        "sufficiency": float(sufficiency),
    }
    if blur_mask is not None:
        targets["blur_mask"] = blur_mask
    return out_u8, targets


def generate_cctv_pair(
    gt_512_u8: np.ndarray,
    *,
    regime: Optional[str] = None,
    seed: Optional[int] = None,
    geometry_prob: float = 0.5,
    apply_geometry: Optional[bool] = None,
    calibration: str = "v30",
) -> Tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray, Dict[str, Any]]:
    """Generate a single (inp_256_u8, gt_256_u8, gt_512_u8, observed_256_bool, targets_dict) sample.

    Args:
        gt_512_u8: Clean HQ face image (512, 512, 3) uint8 RGB (resized to 512x512 if needed).
        regime: Optional exact regime name from REGIMES (if None, sampled per REGIME_WEIGHTS).
        seed: Optional integer seed for 100% deterministic generation (use for DEV benchmarks).
        geometry_prob: Probability of applying prepare_crop non-square/shifted/rotated geometry (default 0.5).
        apply_geometry: Optional explicit bool override for geometry simulation.
        calibration: Degradation calibration profile ('v30' default bit-exact for DEV-Syn, 'v31', 'v32', or 'v33').

    Returns:
        inp_256_u8: (256, 256, 3) uint8 RGB degraded input canvas.
        gt_256_u8: (256, 256, 3) uint8 RGB aligned ground-truth canvas.
        gt_512_u8: (512, 512, 3) uint8 RGB aligned ground-truth at 512x512.
        observed_256_bool: (256, 256) bool mask (True inside observed face crop, False in (128,128,128) bars).
        targets_dict: Dict of regime metadata and [0, 1] physical severity targets for Stage-1/2 supervision.
    """
    if not (isinstance(gt_512_u8, np.ndarray) and gt_512_u8.dtype == np.uint8 and gt_512_u8.ndim == 3 and gt_512_u8.shape[2] == 3):
        raise ValueError("Expected uint8 (H, W, 3) RGB ground-truth image.")
    if calibration not in ("v30", "v31", "v32", "v33", "v34"):
        raise ValueError(f"Unknown degradation calibration '{calibration}'. Expected 'v30', 'v31', 'v32', 'v33', or 'v34'.")

    use_lanczos_gt = calibration in ("v32", "v33", "v34")
    if gt_512_u8.shape[:2] != (512, 512):
        base_512 = _resize_pil_lanczos(gt_512_u8, (512, 512)) if use_lanczos_gt else _resize_pil_bilinear(gt_512_u8, (512, 512))
    else:
        base_512 = gt_512_u8

    actual_seed = int(seed) if seed is not None else int(np.random.SeedSequence().generate_state(1)[0])
    rng = np.random.default_rng(actual_seed)

    chosen_regime = str(regime) if regime is not None else str(rng.choice(REGIMES, p=_REGIME_PROBS))
    if chosen_regime not in REGIME_WEIGHTS:
        raise ValueError(f"Unknown regime '{chosen_regime}'. Must be one of {REGIMES}.")

    use_geom = bool(apply_geometry) if apply_geometry is not None else bool(rng.random() < geometry_prob)
    is_pure_blur = chosen_regime in ("global_defocus_blur", "directional_motion_blur", "local_partial_blur")

    if use_geom:
        gt_native, pts_crop = _apply_geometry_warp_crop_with_meta(base_512, rng)
        ch, cw = gt_native.shape[:2]
        side = max(ch, cw)
        top, left = (side - ch) // 2, (side - cw) // 2
        landmarks_256 = (pts_crop + np.array([float(left), float(top)], dtype=np.float32)) * (256.0 / float(side))
        if use_lanczos_gt:
            gt_256_u8, observed_256_bool = _prepare_crop_exact(gt_native, out_size=256, resample=Image.Resampling.LANCZOS)
            gt_512_out, _ = _prepare_crop_exact(gt_native, out_size=512, resample=Image.Resampling.LANCZOS)
            if chosen_regime == "clear":
                inp_256_u8, targets = _apply_regime(gt_256_u8, "clear", rng, calibration=calibration)
                inp_256_u8[~observed_256_bool] = gt_256_u8[~observed_256_bool]
                if calibration == "v34":
                    targets["x_struct_clean_256"] = gt_256_u8.copy()
                    targets["x_photo_tgt_256"] = gt_256_u8.copy()
            else:
                deg_native, targets = _apply_regime(gt_native, chosen_regime, rng, calibration=calibration)
                inp_256_u8, _ = _prepare_crop_exact(deg_native, out_size=256, resample=Image.Resampling.BILINEAR)
                if "blur_mask" in targets and targets["blur_mask"] is not None:
                    targets["blur_mask"] = _prepare_mask_exact(targets["blur_mask"], out_size=256)
                if calibration == "v34":
                    if is_pure_blur:
                        targets["x_struct_clean_256"] = inp_256_u8.copy()
                        targets["x_photo_tgt_256"] = inp_256_u8.copy()
                    else:
                        sc_nat = targets.get("_x_struct_clean_native")
                        ph_nat = targets.get("_x_photo_tgt_native")
                        sc_256, _ = _prepare_crop_exact(sc_nat, out_size=256, resample=Image.Resampling.LANCZOS)
                        ph_256, _ = _prepare_crop_exact(ph_nat, out_size=256, resample=Image.Resampling.LANCZOS)
                        targets["x_struct_clean_256"] = sc_256
                        targets["x_photo_tgt_256"] = ph_256
        else:
            gt_256_u8, observed_256_bool = _prepare_crop_exact(gt_native, out_size=256)
            gt_512_out, _ = _prepare_crop_exact(gt_native, out_size=512)
            deg_native, targets = _apply_regime(gt_native, chosen_regime, rng, calibration=calibration)
            inp_256_u8, _ = _prepare_crop_exact(deg_native, out_size=256)
    else:
        gt_512_out = base_512.copy()
        gt_256_u8 = _resize_pil_lanczos(base_512, (256, 256)) if use_lanczos_gt else _resize_pil_bilinear(base_512, (256, 256))
        observed_256_bool = np.ones((256, 256), dtype=bool)
        landmarks_256 = (FFHQ_CANONICAL_5PT_NORM * 256.0).astype(np.float32)
        inp_256_u8, targets = _apply_regime(gt_256_u8, chosen_regime, rng, calibration=calibration)
        if calibration == "v34":
            if is_pure_blur:
                targets["x_struct_clean_256"] = inp_256_u8.copy()
                targets["x_photo_tgt_256"] = inp_256_u8.copy()
            else:
                targets["x_struct_clean_256"] = targets["_x_struct_clean_native"].copy()
                targets["x_photo_tgt_256"] = targets["_x_photo_tgt_native"].copy()

    if calibration == "v34":
        targets.pop("_x_struct_clean_native", None)
        targets.pop("_x_photo_tgt_native", None)
        if not observed_256_bool.all():
            targets["x_struct_clean_256"][~observed_256_bool] = 128
            targets["x_photo_tgt_256"][~observed_256_bool] = 128

    if "blur_mask" in targets and targets["blur_mask"] is not None:
        bm = np.asarray(targets["blur_mask"], dtype=np.float32)
        if bm.shape != (256, 256):
            bm = cv2.resize(bm, (256, 256), interpolation=cv2.INTER_LINEAR).astype(np.float32)
        bm = np.clip(bm, 0.0, 1.0)
        bm[~observed_256_bool] = 0.0
        targets["blur_mask"] = bm

    targets["geometry_applied"] = use_geom
    targets["seed"] = actual_seed
    targets["landmarks_256"] = landmarks_256.astype(np.float32).tolist()
    return inp_256_u8, gt_256_u8, gt_512_out, observed_256_bool, targets
