"""Deterministic unit, integration, UI, and hygiene tests for 0.2.7 artifact solvers."""
from __future__ import annotations

import ast
import asyncio
import base64
import io
import json
import math
from pathlib import Path
import subprocess
import sys
import tempfile
import unittest
from unittest.mock import patch
import zipfile

from fastapi import UploadFile
import numpy as np
from PIL import Image
from scipy.fft import dctn, idctn
from scipy.ndimage import distance_transform_edt, gaussian_filter, label
from starlette.requests import Request

from implementation import app as app_module
from implementation import capture_api
from implementation.antialiasing import POLICY as AA_POLICY, detect_aliasing, prepare_antialiasing
from implementation.automatic_preparation import (
    _strip_dct_macroblocking_components,
    picture_pixelation,
    prepare_composables,
    prepare_macroblocking,
    prepare_noise,
    prepare_photometric,
    prepare_source,
)
from implementation.blur_match import estimate_tiles_normalized
from implementation.cast_dehaze import POLICY as CAST_DEHAZE_POLICY, detect_cast_dehaze, prepare_cast_dehaze
from implementation.color_bleeding import POLICY as BLEED_POLICY, detect_color_bleeding, prepare_color_bleeding
from implementation.conditioning_v2 import compute_v2_conditioning_numpy
from implementation.debanding import detect_banding, prepare_debanding
from implementation.deringing import detect_ringing, prepare_deringing
from implementation.irregular_mosaic import irregular_mosaic_mask
from implementation.jpeg_deblocking import CHECKPOINT, CHECKPOINT_SHA256
from implementation.macroblock_deblocking import (
    _seam_and_detail_metrics,
    detect_macroblocking,
    is_dct_macroblocking_patch,
    prepare_macroblock_deblocking,
)
from implementation.noise_denoise import POLICY as NOISE_POLICY, detect_noise, prepare_noise_denoise
from implementation.preparation import png
from implementation.quality import quality_signals
from implementation.runtime import MODEL_DIR, prepare_crop
from implementation.scanline_cleanup import clean_residual_scanline_comb, detect_and_clean_spatial_comb


APP_ROOT = Path(__file__).resolve().parents[1]


def _sha256(raw: bytes) -> str:
    import hashlib

    return hashlib.sha256(raw).hexdigest()


def _bt601_luma(rgb: np.ndarray) -> np.ndarray:
    arr = rgb.astype(np.float32)
    return arr[..., 0] * 0.299 + arr[..., 1] * 0.587 + arr[..., 2] * 0.114


def _decode_zip_from_data_url(bundle_url: str) -> dict[str, bytes]:
    prefix = 'data:application/zip;base64,'
    if not bundle_url.startswith(prefix):
        raise AssertionError(f'Unexpected bundle URL prefix: {bundle_url[:40]!r}')
    raw = base64.b64decode(bundle_url[len(prefix):])
    with zipfile.ZipFile(io.BytesIO(raw), 'r') as archive:
        return {name: archive.read(name) for name in archive.namelist()}


def _decode_png_bytes(raw: bytes) -> np.ndarray:
    with Image.open(io.BytesIO(raw)) as img:
        return np.asarray(img.convert('RGB')).copy()


def _decode_mask_bytes(raw: bytes) -> np.ndarray:
    with Image.open(io.BytesIO(raw)) as img:
        return np.asarray(img.convert('L')) > 0


def _encode_jpeg_bytes(rgb: np.ndarray, *, quality: int = 12, orientation: int = 1) -> bytes:
    """Encode RGB array as JPEG, optionally rotated for EXIF Orientation=6."""
    pil_img = Image.fromarray(rgb, mode='RGB')
    if orientation == 6:
        # Store pixels rotated 90 deg CCW so Orientation=6 rotates 270 deg CCW back upright.
        pil_img = pil_img.transpose(Image.Transpose.ROTATE_90)
    exif = Image.Exif()
    exif[274] = orientation
    buf = io.BytesIO()
    pil_img.save(buf, format='JPEG', quality=quality, subsampling=2, exif=exif.tobytes())
    return buf.getvalue()


def _fake_one_forward_builder(calls_log: list[bytes]):
    """Build deterministic fake V2 worker matching implementation.app.one_forward contract."""

    def _fake_one_forward(contents: bytes, model: str):
        calls_log.append(contents)
        with Image.open(io.BytesIO(contents)) as img:
            decoded = np.asarray(img.convert('RGB')).copy()
        canvas, observed, _, geometry = prepare_crop(decoded)
        raw = canvas.astype(np.float32) / np.float32(255.0)
        raw = np.clip(raw, 0.0, 1.0).astype(np.float32)
        out_u8 = np.floor(raw * np.float32(255.0)).astype(np.uint8)
        out_u8[~observed] = canvas[~observed]
        out_png = png(out_u8)
        worker_meta = {
            'uploaded_bytes_sha256': _sha256(contents),
            'geometry': geometry,
            'raw_rgb_sha256': _sha256(raw.tobytes()),
            'PNG_RGB8_sha256': _sha256(out_u8.tobytes()),
            'PNG_file_sha256': _sha256(out_png),
        }
        buf = io.BytesIO()
        np.save(buf, raw, allow_pickle=False)
        return buf.getvalue(), out_png, worker_meta

    return _fake_one_forward


def _make_clean_portrait(height: int = 192, width: int = 192, seed: int = 42) -> np.ndarray:
    """Create a smooth, curved, textured portrait-like scene at 2x and downsample via LANCZOS."""
    hi_h, hi_w = height * 2, width * 2
    yy, xx = np.mgrid[0:hi_h, 0:hi_w].astype(np.float32)
    yn = yy / hi_h
    xn = xx / hi_w

    # Rich, varied background with smooth curved gradients and gentle organic texture
    r = 68.0 + 48.0 * np.sin(2.4 * xn + 1.1 * yn) + 18.0 * np.cos(5.3 * xn - 3.7 * yn)
    g = 86.0 + 42.0 * np.cos(2.1 * yn - 0.8 * xn) + 16.0 * np.sin(4.7 * xn + 4.1 * yn)
    b = 112.0 + 50.0 * np.sin(1.9 * xn - 2.3 * yn) + 18.0 * np.cos(5.1 * yn + 3.2 * xn)

    # Smooth oval face region
    cx, cy = 0.50, 0.48
    ell = ((xn - cx) / 0.28) ** 2 + ((yn - cy) / 0.36) ** 2
    face_w = np.clip((1.05 - ell) / 0.14, 0.0, 1.0)
    skin_r = 212.0 - 40.0 * ell + 10.0 * np.sin(9.0 * xn) * np.cos(8.0 * yn)
    skin_g = 166.0 - 34.0 * ell + 8.0 * np.cos(8.5 * xn + 7.5 * yn)
    skin_b = 142.0 - 28.0 * ell + 7.0 * np.sin(7.5 * xn - 8.5 * yn)

    # Soft facial shading features (eyes, nose bridge, mouth) without hard 1-px aliasing
    eye_l = np.exp(-(((xn - 0.40) / 0.045) ** 2 + ((yn - 0.41) / 0.025) ** 2))
    eye_r = np.exp(-(((xn - 0.60) / 0.045) ** 2 + ((yn - 0.41) / 0.025) ** 2))
    mouth = np.exp(-(((xn - 0.50) / 0.085) ** 2 + ((yn - 0.62) / 0.022) ** 2))
    nose = np.exp(-(((xn - 0.50) / 0.025) ** 2 + ((yn - 0.51) / 0.085) ** 2))

    skin_r = skin_r - 85.0 * (eye_l + eye_r) - 25.0 * mouth + 14.0 * nose
    skin_g = skin_g - 85.0 * (eye_l + eye_r) - 55.0 * mouth + 11.0 * nose
    skin_b = skin_b - 75.0 * (eye_l + eye_r) - 50.0 * mouth + 8.0 * nose

    img_hi = np.stack(
        [
            r * (1.0 - face_w) + skin_r * face_w,
            g * (1.0 - face_w) + skin_g * face_w,
            b * (1.0 - face_w) + skin_b * face_w,
        ],
        axis=-1,
    )
    rng = np.random.default_rng(seed)
    # Gentle smooth low-amplitude correlated texture (no grid/staircase)
    tex = gaussian_filter(rng.normal(0.0, 8.0, (hi_h, hi_w, 3)).astype(np.float32), sigma=(2.2, 2.2, 0.0))
    img_hi = np.clip(np.rint(img_hi + tex), 0, 255).astype(np.uint8)
    return np.asarray(Image.fromarray(img_hi).resize((width, height), Image.Resampling.LANCZOS)).copy()


def _make_block_dct_macroblocked(
    base: np.ndarray,
    *,
    period: int = 8,
    quant_step: float = 34.0,
    detail_keep: float = 0.0,
) -> np.ndarray:
    """Apply per-block P x P orthonormal DCT quantization preserving smooth interior gradients."""
    h, w = base.shape[:2]
    assert h % period == 0 and w % period == 0
    arr = base.astype(np.float32)
    out = np.empty_like(arr)
    nh, nw = h // period, w // period

    # Frequency weight matrix: mild quantization on low-AC so interior isn't flat, strong on mid/high AC
    u, v = np.meshgrid(np.arange(period), np.arange(period), indexing='ij')
    freq_scale = np.maximum(1.0, (u + v) * 0.45)
    q_mat = np.full((period, period), quant_step, dtype=np.float32) * freq_scale
    q_mat[0, 0] = 8.0
    if detail_keep > 0.0:
        q_mat[0, 1] = q_mat[1, 0] = quant_step * 0.4

    for c in range(3):
        blocks = arr[..., c].reshape(nh, period, nw, period).transpose(0, 2, 1, 3)
        coeffs = dctn(blocks, axes=(2, 3), norm='ortho')
        quant = np.round(coeffs / q_mat[None, None, :, :]) * q_mat[None, None, :, :]
        recon = idctn(quant, axes=(2, 3), norm='ortho')
        out[..., c] = recon.transpose(0, 2, 1, 3).reshape(h, w)

    if detail_keep > 0.0:
        out = out * (1.0 - detail_keep) + arr * detail_keep
    return np.clip(np.rint(out), 0, 255).astype(np.uint8)


def _make_macroblock_carrier_base(height: int = 160, width: int = 160, seed: int = 7) -> np.ndarray:
    """Create a rich, non-banded diagonal/radial natural texture ideal for DCT macroblock synthesis."""
    yy, xx = np.mgrid[0:height, 0:width].astype(np.float32)
    r = 128.0 + 48.0 * np.sin(0.051 * xx + 0.037 * yy) + 28.0 * np.cos(0.113 * xx - 0.089 * yy)
    g = 122.0 + 44.0 * np.cos(0.043 * xx - 0.061 * yy) + 26.0 * np.sin(0.097 * xx + 0.107 * yy)
    b = 118.0 + 46.0 * np.sin(0.063 * xx - 0.041 * yy) + 24.0 * np.cos(0.121 * xx + 0.079 * yy)
    rng = np.random.default_rng(seed)
    micro = gaussian_filter(rng.normal(0.0, 9.0, (height, width, 3)).astype(np.float32), sigma=(0.9, 0.9, 0.0))
    return np.clip(np.rint(np.stack([r, g, b], axis=-1) + micro), 0, 255).astype(np.uint8)


def _make_flat_mosaic(base: np.ndarray, block: int = 8) -> np.ndarray:
    """Create a genuine flat-cell privacy mosaic by replacing each P x P block with its constant mean."""
    h, w = base.shape[:2]
    out = base.copy()
    for y in range(0, h, block):
        for x in range(0, w, block):
            cell = base[y:min(h, y + block), x:min(w, x + block)]
            mean_rgb = np.rint(cell.reshape(-1, 3).mean(axis=0)).astype(np.uint8)
            out[y:min(h, y + block), x:min(w, x + block)] = mean_rgb
    return out


def _make_ringing_image(size: int = 160) -> tuple[np.ndarray, np.ndarray]:
    """Synthesize a clean circle edge surrounded by explicit Gibbs overshoot/undershoot ripples."""
    yy, xx = np.mgrid[0:size, 0:size].astype(np.float32)
    cx = cy = (size - 1) / 2.0
    rad = size * 0.27
    dist = np.hypot(xx - cx, yy - cy)
    signed_d = dist - rad  # negative inside disc, positive outside disc

    # Smooth low-frequency gradient across disc and background so flanks stay natural & calm
    inside_val = 192.0 - 0.20 * (xx - cx)
    outside_val = 66.0 + 0.16 * (yy - cy)
    core_profile = np.where(signed_d <= 0.0, inside_val, outside_val)
    # 1-px smooth anti-aliased transition right on the core edge
    edge_blend = np.clip(0.5 + 0.5 * (signed_d / 0.55), 0.0, 1.0)
    clean_luma = inside_val * (1.0 - edge_blend) + outside_val * edge_blend

    # Add alternating damped Gibbs ringing lobes across [-4..-1.3] and [1.3..4]
    ad = np.abs(signed_d)
    envelope = np.where((ad >= 1.2) & (ad <= 4.6), np.exp(-(ad - 1.2) * 0.42), 0.0)
    oscillation = 22.0 * envelope * np.sin(signed_d * (math.pi / 1.15))
    ringed_luma = np.clip(np.rint(core_profile * 0.0 + clean_luma + oscillation), 0, 255).astype(np.uint8)
    # Slight warm tint difference so RGB isn't degenerate grayscale
    rgb = np.stack(
        [
            np.clip(ringed_luma.astype(np.int16) + 6, 0, 255).astype(np.uint8),
            ringed_luma,
            np.clip(ringed_luma.astype(np.int16) - 6, 0, 255).astype(np.uint8),
        ],
        axis=-1,
    )
    clean_rgb = np.stack(
        [
            np.clip(np.rint(clean_luma).astype(np.int16) + 6, 0, 255).astype(np.uint8),
            np.clip(np.rint(clean_luma), 0, 255).astype(np.uint8),
            np.clip(np.rint(clean_luma).astype(np.int16) - 6, 0, 255).astype(np.uint8),
        ],
        axis=-1,
    )
    return rgb, clean_rgb


def _make_banded_gradient(size: int = 160, *, step: int = 7) -> tuple[np.ndarray, np.ndarray]:
    """Create a smooth 8-bit curved spherical/sky portrait gradient and its quantized banded version."""
    yy, xx = np.mgrid[0:size, 0:size].astype(np.float32)
    xn = xx / max(1, size - 1)
    yn = yy / max(1, size - 1)
    # Curved smooth illumination without high-frequency aliasing or orthogonal grids
    dist = np.hypot(xn - 0.45, yn - 0.42)
    base_y = 205.0 - 135.0 * np.clip(dist / 0.85, 0.0, 1.0) + 14.0 * np.sin(2.5 * xn + 1.8 * yn)
    # Keep gradient modest everywhere (< 12 luma/px) except one crisp small protected emblem so edge_retention can be tested
    r = np.clip(base_y + 12.0 * (1.0 - xn), 12.0, 245.0)
    g = np.clip(base_y * 0.94 + 4.0 * yn, 10.0, 240.0)
    b = np.clip(base_y * 0.88 + 18.0 * xn, 10.0, 240.0)
    continuous = np.stack([r, g, b], axis=-1)
    # Add one crisp protected circle boundary in the lower-right corner so edge_retention >= 95% is meaningfully exercised
    corner_d = np.hypot(xx - size * 0.80, yy - size * 0.80)
    edge_mask = np.clip(0.5 + 0.5 * (size * 0.08 - corner_d) / 0.7, 0.0, 1.0)[..., None]
    continuous = continuous * (1.0 - edge_mask) + np.array([235.0, 230.0, 225.0], dtype=np.float32) * edge_mask

    continuous_u8 = np.clip(np.rint(continuous), 0, 255).astype(np.uint8)
    quantized = np.round(continuous / float(step)) * float(step)
    # Restore crisp unquantized edge only at the strong emblem border
    banded = quantized * (1.0 - edge_mask) + np.array([235.0, 230.0, 225.0], dtype=np.float32) * edge_mask
    banded_u8 = np.clip(np.rint(banded), 0, 255).astype(np.uint8)
    return banded_u8, continuous_u8


def _make_irregular_voronoi_mosaic(size: int = 160, cell_spacing: int = 9, seed: int = 19) -> np.ndarray:
    """Create a genuine multi-hued irregular Voronoi privacy mosaic."""
    rng = np.random.default_rng(seed)
    xs = np.arange(cell_spacing // 2, size, cell_spacing)
    ys = np.arange(cell_spacing // 2, size, cell_spacing)
    seeds_y, seeds_x = np.meshgrid(ys, xs, indexing='ij')
    pts_y = seeds_y.ravel().astype(np.float32) + rng.uniform(-2.2, 2.2, seeds_y.size).astype(np.float32)
    pts_x = seeds_x.ravel().astype(np.float32) + rng.uniform(-2.2, 2.2, seeds_x.size).astype(np.float32)

    yy, xx = np.mgrid[0:size, 0:size].astype(np.float32)
    dy = yy[..., None] - pts_y[None, None, :]
    dx = xx[..., None] - pts_x[None, None, :]
    nearest = np.argmin(dx * dx + dy * dy, axis=2)

    # Multi-hue distinct RGB palette per Voronoi cell so it is not tonal grayscale banding
    colors = rng.integers(25, 235, size=(pts_y.size, 3), dtype=np.uint8)
    return colors[nearest]


def _make_color_bleeding_image(size: int = 160) -> tuple[np.ndarray, np.ndarray]:
    """Create sharp curved high-resolution luma edges with smeared 4:2:0-style chroma bleed."""
    hi = size * 2
    yy, xx = np.mgrid[0:hi, 0:hi].astype(np.float32)
    clean_hi = np.full((hi, hi, 3), 122.0, dtype=np.float32)
    clean_hi[..., 0] += 18.0 * np.sin(0.015 * xx + 0.012 * yy)
    clean_hi[..., 1] += 14.0 * np.cos(0.013 * xx - 0.016 * yy)
    clean_hi[..., 2] += 16.0 * np.sin(0.014 * (xx - yy))

    c1 = np.hypot(xx - hi * 0.35, yy - hi * 0.42) <= hi * 0.17
    dx2 = (xx - hi * 0.64) * np.cos(0.4) + (yy - hi * 0.58) * np.sin(0.4)
    dy2 = -(xx - hi * 0.64) * np.sin(0.4) + (yy - hi * 0.58) * np.cos(0.4)
    c2 = (dx2 / (hi * 0.20)) ** 2 + (dy2 / (hi * 0.13)) ** 2 <= 1.0

    clean_hi[c1] = clean_hi[c1] * 0.2 + np.array([210.0, 68.0, 68.0]) * 0.8
    clean_hi[c2] = clean_hi[c2] * 0.2 + np.array([66.0, 88.0, 218.0]) * 0.8

    clean_pil = Image.fromarray(np.clip(np.rint(clean_hi), 0, 255).astype(np.uint8)).resize(
        (size, size), Image.Resampling.LANCZOS
    )
    clean_rgb = np.asarray(clean_pil).copy()

    ycbcr = Image.fromarray(clean_rgb).convert('YCbCr')
    y_ch, cb_ch, cr_ch = [np.asarray(ch, dtype=np.float32) for ch in ycbcr.split()]
    cb_bleed = gaussian_filter(cb_ch, sigma=2.4)
    cr_bleed = gaussian_filter(cr_ch, sigma=2.4)
    bled_ycbcr = Image.merge(
        'YCbCr',
        [
            Image.fromarray(np.clip(np.rint(y_ch), 0, 255).astype(np.uint8), mode='L'),
            Image.fromarray(np.clip(np.rint(cb_bleed), 0, 255).astype(np.uint8), mode='L'),
            Image.fromarray(np.clip(np.rint(cr_bleed), 0, 255).astype(np.uint8), mode='L'),
        ],
    )
    return np.asarray(bled_ycbcr.convert('RGB')).copy(), clean_rgb


def _make_unified_compound_scene(size: int = 224) -> np.ndarray:
    """Create one organic curved scene combining ringing, banding, and 4:2:0 chroma bleed."""
    yy, xx = np.mgrid[0:size, 0:size].astype(np.float32)
    xn = xx / (size - 1)
    yn = yy / (size - 1)

    dist_bg = np.hypot(xn - 0.35, yn - 0.35)
    bg_y = 205.0 - 125.0 * np.clip(dist_bg / 0.90, 0.0, 1.0) + 12.0 * np.sin(2.4 * xn + 1.9 * yn)
    bg_y_q = np.round(bg_y / 7.0) * 7.0
    scene = np.stack([np.clip(bg_y_q + 6, 0, 255), np.clip(bg_y_q, 0, 255), np.clip(bg_y_q - 6, 0, 255)], axis=-1)

    signed_r = np.hypot(xx - size * 0.30, yy - size * 0.30) - size * 0.16
    disc_blend = np.clip(0.5 - 0.5 * (signed_r / 0.55), 0.0, 1.0)
    disc_in = 198.0 - 0.12 * (xx - size * 0.30)
    disc_out = scene[..., 1] * 0.65
    base_r = disc_out * (1.0 - disc_blend) + disc_in * disc_blend
    ad = np.abs(signed_r)
    ring_env = np.where((ad >= 1.2) & (ad <= 4.6), np.exp(-(ad - 1.2) * 0.42), 0.0)
    osc = 22.0 * ring_env * np.sin(signed_r * (math.pi / 1.15))
    ring_area = np.hypot(xx - size * 0.30, yy - size * 0.30) <= size * 0.22
    ring_y = np.clip(base_r + osc, 0, 255)
    for c, delta in enumerate((6.0, 0.0, -6.0)):
        scene[..., c] = np.where(ring_area, np.clip(ring_y + delta, 0, 255), scene[..., c])

    hi = size * 2
    yy4, xx4 = np.mgrid[0:hi, 0:hi].astype(np.float32)
    c_red = np.hypot(xx4 - hi * 0.72, yy4 - hi * 0.68) <= hi * 0.13
    dx_b = (xx4 - hi * 0.42) * np.cos(0.45) + (yy4 - hi * 0.76) * np.sin(0.45)
    dy_b = -(xx4 - hi * 0.42) * np.sin(0.45) + (yy4 - hi * 0.76) * np.cos(0.45)
    c_blue = (dx_b / (hi * 0.15)) ** 2 + (dy_b / (hi * 0.09)) ** 2 <= 1.0

    scene_hi = np.asarray(
        Image.fromarray(np.clip(np.rint(scene), 0, 255).astype(np.uint8)).resize((hi, hi), Image.Resampling.BILINEAR),
        dtype=np.float32,
    )
    scene_hi[c_red] = np.array([212.0, 68.0, 68.0])
    scene_hi[c_blue] = np.array([68.0, 90.0, 222.0])
    clean_mixed = np.asarray(
        Image.fromarray(np.clip(np.rint(scene_hi), 0, 255).astype(np.uint8)).resize((size, size), Image.Resampling.LANCZOS)
    ).copy()
    y_ch, cb_ch, cr_ch = [np.asarray(ch, dtype=np.float32) for ch in Image.fromarray(clean_mixed).convert('YCbCr').split()]
    cb_b = np.where((xx > size * 0.25) & (yy > size * 0.48), gaussian_filter(cb_ch, 2.4), cb_ch)
    cr_b = np.where((xx > size * 0.25) & (yy > size * 0.48), gaussian_filter(cr_ch, 2.4), cr_ch)
    bled_region = np.asarray(
        Image.merge(
            'YCbCr',
            [
                Image.fromarray(np.clip(np.rint(y_ch), 0, 255).astype(np.uint8), mode='L'),
                Image.fromarray(np.clip(np.rint(cb_b), 0, 255).astype(np.uint8), mode='L'),
                Image.fromarray(np.clip(np.rint(cr_b), 0, 255).astype(np.uint8), mode='L'),
            ],
        ).convert('RGB')
    )
    return np.where(((xx > size * 0.22) & (yy > size * 0.45))[..., None], bled_region, np.clip(np.rint(scene), 0, 255).astype(np.uint8))


def _make_aliased_diagonal_image(size: int = 160) -> tuple[np.ndarray, np.ndarray]:
    """Create jagged un-antialiased curved contours vs clean LANCZOS contours."""

    def _render_contours(xx_arr: np.ndarray, yy_arr: np.ndarray) -> np.ndarray:
        xn = xx_arr / size
        yn = yy_arr / size
        bg = np.stack(
            [
                78.0 + 18.0 * np.sin(2.5 * xn + 1.1 * yn),
                86.0 + 14.0 * np.cos(2.1 * yn - xn),
                98.0 + 15.0 * np.sin(2.2 * (xn - yn)),
            ],
            axis=-1,
        )
        fg = np.stack(
            [
                202.0 - 16.0 * xn,
                178.0 - 14.0 * yn,
                162.0 + 12.0 * np.cos(2.0 * xn),
            ],
            axis=-1,
        )
        d1 = ((xx_arr - size * 0.50) / (size * 0.36)) ** 2 + ((yy_arr - size * 0.50) / (size * 0.30)) ** 2 <= 1.0
        d2 = ((xx_arr - size * 0.52) / (size * 0.22)) ** 2 + ((yy_arr - size * 0.48) / (size * 0.16)) ** 2 <= 1.0
        dx3 = (xx_arr - size * 0.50) * math.cos(0.55) + (yy_arr - size * 0.50) * math.sin(0.55)
        dy3 = -(xx_arr - size * 0.50) * math.sin(0.55) + (yy_arr - size * 0.50) * math.cos(0.55)
        d3 = (dx3 / (size * 0.38)) ** 2 + (dy3 / (size * 0.08)) ** 2 <= 1.0
        mask = (d1 ^ d2) | d3
        return np.clip(np.rint(np.where(mask[..., None], fg, bg)), 0, 255).astype(np.uint8)

    yy, xx = np.mgrid[0:size, 0:size].astype(np.float32)
    aliased_u8 = _render_contours(xx, yy)

    hi = size * 4
    yy4, xx4 = np.mgrid[0:hi, 0:hi].astype(np.float32) / 4.0
    clean_hi = _render_contours(xx4, yy4)
    clean_u8 = np.asarray(Image.fromarray(clean_hi).resize((size, size), Image.Resampling.LANCZOS)).copy()
    return aliased_u8, clean_u8


def _measure_ringing_wings(rgb: np.ndarray, radius: float) -> tuple[float, float]:
    """Measure edge core contrast and adjacent Gibbs wing ripple energy on circular test image."""
    h, w = rgb.shape[:2]
    yy, xx = np.mgrid[0:h, 0:w].astype(np.float32)
    dist = np.hypot(xx - (w - 1) / 2.0, yy - (h - 1) / 2.0) - radius
    y = _bt601_luma(rgb)
    # Core jump between [-5..-4] vs [4..5] vs [-1..1]: actually local directional diff across dist ~ 0
    gx = np.abs(np.diff(y, axis=1, prepend=y[:, :1]))
    gy = np.abs(np.diff(y, axis=0, prepend=y[:1, :]))
    grad = np.hypot(gx, gy)
    core = grad[np.abs(dist) <= 1.2].mean()
    # Second-derivative ripple energy inside halo band [1.8..4.5]
    lap = np.abs(
        y[1:-1, 1:-1] * 4.0
        - y[:-2, 1:-1]
        - y[2:, 1:-1]
        - y[1:-1, :-2]
        - y[1:-1, 2:]
    )
    halo_interior = (np.abs(dist[1:-1, 1:-1]) >= 1.8) & (np.abs(dist[1:-1, 1:-1]) <= 4.3)
    halo_osc = float(lap[halo_interior].mean())
    return float(core), halo_osc


class DegradationSolverUnitTests(unittest.TestCase):
    """Unit tests covering positive detections, quantitative gates, and negative controls."""

    def test_clean_lanczos_portrait_triggers_zero_artifact_solvers(self):
        clean = _make_clean_portrait(160, 160)
        self.assertIsNone(detect_macroblocking(clean))
        self.assertEqual(prepare_macroblock_deblocking(clean), (None, None, {}))
        self.assertIsNone(detect_ringing(clean))
        self.assertEqual(prepare_deringing(clean), (None, None, {}))
        self.assertIsNone(detect_banding(clean))
        self.assertEqual(prepare_debanding(clean), (None, None, {}))
        self.assertIsNone(detect_color_bleeding(clean))
        self.assertEqual(prepare_color_bleeding(clean), (None, None, {}))
        self.assertIsNone(detect_aliasing(clean))
        self.assertEqual(prepare_antialiasing(clean), (None, None, {}))
        out, stages, reports, evidence = prepare_composables(clean)
        self.assertIsNone(out)
        self.assertEqual(stages, [])
        self.assertEqual(reports, {})
        self.assertEqual(evidence, {})

    def test_macroblocking_8x8_real_fbcnn_deblocks_and_preserves_interior(self):
        checkpoint_path = MODEL_DIR / CHECKPOINT
        self.assertTrue(checkpoint_path.is_file(), f'Missing real FBCNN checkpoint at {checkpoint_path}')
        self.assertEqual(_sha256(checkpoint_path.read_bytes()), CHECKPOINT_SHA256)
        base = _make_macroblock_carrier_base(160, 160)
        blocked = _make_block_dct_macroblocked(base, period=8, quant_step=34.0)

        det = detect_macroblocking(blocked)
        self.assertIsNotNone(det)
        self.assertTrue(det['suspected'])
        self.assertFalse(det['is_flat_privacy_mosaic'])
        self.assertEqual(det['period'], 8)

        cleaned, report, evidence = prepare_macroblock_deblocking(blocked, allow_fbcnn=True)
        self.assertIsNotNone(cleaned)
        self.assertIsNotNone(report)
        self.assertEqual(report['method'], 'fbcnn_rgb')
        self.assertEqual(report['preparation_model'], 'FBCNN RGB')
        self.assertEqual(report['preparation_model_sha256'], CHECKPOINT_SHA256)
        self.assertEqual(report['preparation_forward_returns'], 1)
        self.assertEqual(report['period'], 8)
        self.assertTrue(all(drop >= 0.20 for drop in report['seam_drop_fraction']))
        self.assertGreaterEqual(report['interior_detail_retention'], 0.90)
        self.assertIn('macroblock-cleaned.png', evidence)

        # Independently verify seam drop >= 20% and interior retention >= 90%
        ox, oy = report['grid_origin']
        before = _seam_and_detail_metrics(blocked, 8, ox % 8, oy % 8, None)
        after = _seam_and_detail_metrics(cleaned, 8, ox % 8, oy % 8, None)
        self.assertIsNotNone(before)
        self.assertIsNotNone(after)
        for (b_seam, b_int), (a_seam, a_int) in zip(before, after):
            self.assertGreater(b_seam, 0.25)
            self.assertGreaterEqual((b_seam - abs(a_seam)) / b_seam, 0.20)
            if b_int >= 0.5:
                self.assertGreaterEqual(a_int / b_int, 0.90)

    def test_macroblocking_16x16_hevc_and_out_of_range_sizes_and_masked_localization(self):
        # 16x16 pure macroblocking (160x160) -> macroblock_hevc (hevc_16x16)
        base_160 = _make_macroblock_carrier_base(160, 160, seed=11)
        blocked_16 = _make_block_dct_macroblocked(base_160, period=16, quant_step=44.0)
        cleaned_16, stage_16, rep_16, _ = prepare_macroblocking(blocked_16, allow_fbcnn=True)
        self.assertIsNotNone(cleaned_16)
        self.assertEqual(stage_16, 'macroblock_hevc')
        self.assertEqual(rep_16['method'], 'hevc_16x16')
        self.assertEqual(rep_16['period'], 16)
        self.assertTrue(all(d >= 0.20 for d in rep_16['seam_drop_fraction']))
        self.assertGreaterEqual(rep_16['interior_detail_retention'], 0.90)

        # Out-of-range small (64x64) 8x8 DCT macroblocked -> falls back to HEVC (macroblock_hevc)
        base_64 = _make_macroblock_carrier_base(64, 64, seed=12)
        blocked_64 = _make_block_dct_macroblocked(base_64, period=8, quant_step=36.0)
        cleaned_64, stage_64, rep_64, _ = prepare_macroblocking(blocked_64, allow_fbcnn=True)
        self.assertIsNotNone(cleaned_64)
        self.assertEqual(stage_64, 'macroblock_hevc')
        self.assertEqual(rep_64['method'], 'hevc_8x8')

        # Out-of-range large (480x480) 8x8 DCT macroblocked -> falls back to HEVC (macroblock_hevc)
        base_480 = _make_macroblock_carrier_base(480, 480, seed=13)
        blocked_480 = _make_block_dct_macroblocked(base_480, period=8, quant_step=36.0)
        cleaned_480, stage_480, rep_480, _ = prepare_macroblocking(blocked_480, allow_fbcnn=True)
        self.assertIsNotNone(cleaned_480)
        self.assertEqual(stage_480, 'macroblock_hevc')
        self.assertEqual(rep_480['method'], 'hevc_8x8')

        # Face-localized 8x8 macroblocking inside a clean 224x224 textured portrait -> regional mask & bit-exact exterior
        base_224 = _make_macroblock_carrier_base(224, 224, seed=21)
        mb_full_224 = _make_block_dct_macroblocked(base_224, period=8, quant_step=34.0)
        face_slice = np.s_[48:176, 48:176]
        local_blocked = base_224.copy()
        local_blocked[face_slice] = mb_full_224[face_slice]
        face_mask = np.zeros((224, 224), dtype=bool)
        face_mask[face_slice] = True
        cleaned_loc, stage_loc, rep_loc, ev_loc = prepare_macroblocking(local_blocked, mask=face_mask, allow_fbcnn=True)
        self.assertIsNotNone(cleaned_loc)
        self.assertEqual(stage_loc, 'macroblock_fbcnn')
        self.assertTrue(rep_loc['regional'])
        self.assertIn('macroblock-mask.png', ev_loc)
        mask_out = _decode_mask_bytes(ev_loc['macroblock-mask.png'])
        self.assertTrue(np.array_equal(cleaned_loc[~mask_out], local_blocked[~mask_out]))
        self.assertTrue(np.array_equal(cleaned_loc[~face_mask], local_blocked[~face_mask]))

    def test_flat_8x8_privacy_mosaic_is_rejected_by_macroblock_solver_and_handled_as_mosaic(self):
        portrait = _make_clean_portrait(192, 192, seed=33)
        flat_mosaic = _make_flat_mosaic(portrait, block=8)
        det = detect_macroblocking(flat_mosaic)
        self.assertIsNotNone(det)
        self.assertTrue(det['is_flat_privacy_mosaic'])
        self.assertFalse(det['suspected'])
        self.assertEqual(prepare_macroblock_deblocking(flat_mosaic), (None, None, {}))
        prepared, report, evidence = prepare_source(flat_mosaic)
        self.assertIsNotNone(prepared)
        self.assertIsNotNone(report)
        self.assertIn('full-image-pixelation-mask.png', evidence)

    def test_ringing_gibbs_solver_and_negative_controls(self):
        ringed, clean_edge = _make_ringing_image(160)
        det = detect_ringing(ringed)
        self.assertIsNotNone(det)
        out, report, evidence = prepare_deringing(ringed)
        self.assertIsNotNone(out)
        self.assertIsNotNone(report)
        self.assertEqual(report['policy'], 'gibbs-halo-deringing-v1')
        self.assertTrue(report['edge_core_untouched'])
        self.assertTrue(report['outside_halo_exact'])
        self.assertIn('deringing-halo-mask.png', evidence)

        halo_mask = _decode_mask_bytes(evidence['deringing-halo-mask.png'])
        self.assertTrue(halo_mask.any())
        self.assertTrue(np.array_equal(out[~halo_mask], ringed[~halo_mask]))
        self.assertGreaterEqual(report['halo_oscillation_drop'], 0.25)
        self.assertGreaterEqual(report['halo_highpass_drop'], 0.25)
        self.assertGreaterEqual(report['core_gradient_retention'], 0.97)

        core_before, osc_before = _measure_ringing_wings(ringed, radius=160 * 0.27)
        core_after, osc_after = _measure_ringing_wings(out, radius=160 * 0.27)
        self.assertGreaterEqual(core_after / core_before, 0.97)
        self.assertGreaterEqual((osc_before - osc_after) / osc_before, 0.25)

        # Negative controls: clean crisp edge without ringing & heavy textured noise
        self.assertEqual(prepare_deringing(clean_edge), (None, None, {}))
        rng = np.random.default_rng(55)
        textured = _make_clean_portrait(160, 160, seed=55).astype(np.int16)
        textured = np.clip(textured + rng.integers(-18, 19, textured.shape), 0, 255).astype(np.uint8)
        self.assertEqual(prepare_deringing(textured), (None, None, {}))

    def test_debanding_irregular_mosaic_guard_and_noise_bounds(self):
        banded, continuous = _make_banded_gradient(160, step=7)
        # Crucial 0.2.7 guard: curved posterized tonal gradient must produce 0 irregular_mosaic pixels
        irr_on_banded = irregular_mosaic_mask(banded)
        self.assertEqual(int(irr_on_banded.sum()), 0)

        out, report, evidence = prepare_debanding(banded)
        self.assertIsNotNone(out)
        self.assertIsNotNone(report)
        self.assertEqual(report['method'], 'quantized-plateau-contour-v1')
        self.assertIn('debanding-mask.png', evidence)
        self.assertGreaterEqual(report['step_reduction'], 0.30)
        self.assertGreaterEqual(report['edge_retention'], 0.95)
        self.assertLessEqual(report['laplacian_noise_delta'], 0.60)

        mask = _decode_mask_bytes(evidence['debanding-mask.png'])
        self.assertTrue(mask.any())
        self.assertTrue(np.array_equal(out[~mask], banded[~mask]))

        # Ordered Bayer dither must keep quality_signals noise_sigma_255 & V2 g_sensor_noise strictly bounded
        _, obs, _, _ = prepare_crop(out)
        qs_before = quality_signals(banded, (~obs).astype(np.uint8))
        qs_after = quality_signals(out, (~obs).astype(np.uint8))
        self.assertLess(qs_after['noise_sigma_255'], 2.0)
        self.assertLess(qs_after['noise_sigma_255'], qs_before['noise_threshold'])

        # V2 conditioning g_sensor_noise on 256x256 dark-toned debanded crop must stay low/bounded
        banded_256, _ = _make_banded_gradient(256, step=7)
        out_256, rep_256, _ = prepare_debanding((banded_256 * 0.45).astype(np.uint8))
        self.assertIsNotNone(out_256)
        v2_before = compute_v2_conditioning_numpy((banded_256 * 0.45).astype(np.float32) / 255.0, np=np)
        v2_after = compute_v2_conditioning_numpy(out_256.astype(np.float32) / 255.0, np=np)
        self.assertLessEqual(v2_after['g_sensor_noise'], 0.05)
        self.assertLessEqual(v2_after['g_sensor_noise'] - v2_before['g_sensor_noise'], 0.05)

        # Negative control: continuous 8-bit gradient triggers neither debanding nor irregular mosaic
        self.assertEqual(prepare_debanding(continuous), (None, None, {}))
        self.assertEqual(int(irregular_mosaic_mask(continuous).sum()), 0)

        # Positive irregular mosaic control: Voronoi privacy mosaic triggers irregular_mosaic_mask & skips debanding
        voronoi = _make_irregular_voronoi_mosaic(160)
        irr_vor = irregular_mosaic_mask(voronoi)
        self.assertGreaterEqual(float(irr_vor.mean()), 0.25)
        self.assertEqual(prepare_debanding(voronoi), (None, None, {}))

    def test_color_bleeding_solver_and_negative_controls(self):
        bled, clean = _make_color_bleeding_image(160)
        det = detect_color_bleeding(bled)
        self.assertIsNotNone(det)
        out, report, evidence = prepare_color_bleeding(bled)
        self.assertIsNotNone(out)
        self.assertIsNotNone(report)
        self.assertEqual(report['policy'], BLEED_POLICY)
        self.assertIn('color-bleeding-mask.png', evidence)
        self.assertGreaterEqual(report['off_edge_bleed_reduction'], 0.25)
        self.assertGreaterEqual(report['luma_gradient_retention'], 0.98)
        self.assertLessEqual(report['mean_luma_abs_diff'], 1.0)
        self.assertLessEqual(report['mask_luma_abs_diff'], 1.0)

        mask = _decode_mask_bytes(evidence['color-bleeding-mask.png'])
        self.assertTrue(mask.any())
        self.assertTrue(np.array_equal(out[~mask], bled[~mask]))
        mean_dy = float(np.abs(_bt601_luma(out) - _bt601_luma(bled)).mean())
        self.assertLessEqual(mean_dy, 1.0)

        # Negative controls: clean full-res chroma image & monochrome image
        self.assertEqual(prepare_color_bleeding(clean), (None, None, {}))
        gray_val = np.clip(np.rint(_bt601_luma(bled)), 0, 255).astype(np.uint8)
        mono = np.repeat(gray_val[..., None], 3, axis=-1)
        self.assertEqual(prepare_color_bleeding(mono), (None, None, {}))

    def test_antialiasing_solver_and_negative_controls(self):
        aliased, clean_aa = _make_aliased_diagonal_image(160)
        det = detect_aliasing(aliased)
        self.assertIsNotNone(det)
        out, report, evidence = prepare_antialiasing(aliased)
        self.assertIsNotNone(out)
        self.assertIsNotNone(report)
        self.assertEqual(report['policy'], AA_POLICY)
        self.assertIn('antialiasing-edge-mask.png', evidence)
        self.assertGreaterEqual(report['tangent_roughness_reduction'], 0.25)
        self.assertGreaterEqual(report['cross_edge_gradient_retention'], 0.94)

        mask = _decode_mask_bytes(evidence['antialiasing-edge-mask.png'])
        self.assertTrue(mask.any())
        self.assertTrue(np.array_equal(out[~mask], aliased[~mask]))

        # Negative controls: clean LANCZOS contours, modulo-256 wrap ramp, periodic flat mosaic
        self.assertEqual(prepare_antialiasing(clean_aa), (None, None, {}))

        # Modulo-256 uint8 wrap sawtooth diagonal ramp
        yy, xx = np.mgrid[0:160, 0:160]
        wrap_val = ((xx * 5 + yy * 3) % 256).astype(np.uint8)
        wrap_rgb = np.repeat(wrap_val[..., None], 3, axis=-1)
        self.assertEqual(prepare_antialiasing(wrap_rgb), (None, None, {}))

        # Periodic 6x6 block mosaic must not be misclassified as diagonal aliasing
        portrait_156 = _make_clean_portrait(156, 156, seed=77)
        periodic_mosaic = _make_flat_mosaic(portrait_156, block=6)
        self.assertEqual(prepare_antialiasing(periodic_mosaic), (None, None, {}))

    def test_localized_collar_bleed_below_gate_returns_none(self):
        """A small (~200 px) high-chroma collar highlight on sharp surroundings must stay below the bleed gate."""
        src = _make_blurred_face_collar_highlight_portrait()
        self.assertIsNone(detect_color_bleeding(src))
        self.assertEqual(prepare_color_bleeding(src), (None, None, {}))

    def test_non_power8_mosaic_keeps_component_recovers_holes_and_feathers(self):
        """15px/30px mosaics with weak interior pockets must resist DCT stripping, fill holes, and feather inward."""
        for block, size, radius in ((15, 256, 78), (30, 360, 115)):
            with self.subTest(block=block, size=size):
                src, pocket_yx = _make_non_power8_mosaic_with_pocket(block=block, size=size, radius=radius)
                prepared, report, evidence = picture_pixelation(src)
                self.assertIsNotNone(prepared)
                self.assertIsNotNone(report)
                self.assertEqual(report['policy'], 'multiscale-picture-block-regions-v3')

                mask = _decode_mask_bytes(evidence['full-image-pixelation-mask.png'])
                stripped = _strip_dct_macroblocking_components(src, mask)
                self.assertEqual(int(stripped.sum()), int(mask.sum()))

                self.assertGreater(report['recovered_gap_analysis_pixels'], 0)
                _, num_components = label(mask)
                self.assertEqual(num_components, 1)
                self.assertTrue(bool(mask[pocket_yx]))
                self.assertTrue(np.array_equal(prepared[~mask], src[~mask]))

                dist_in = distance_transform_edt(mask)
                feather_px = min(max(1.0, 3.5 * size / 256.0), max(1.0, float(dist_in.max()) * 0.5))
                boundary_ring = (dist_in >= 1.0) & (dist_in <= 1.2)
                deep_interior = dist_in >= feather_px + 2.0
                self.assertTrue(boundary_ring.any() and deep_interior.any())
                delta = np.abs(prepared.astype(np.float32) - src.astype(np.float32)).mean(axis=2)
                self.assertLess(float(delta[boundary_ring].mean()), float(delta[deep_interior].mean()))

    def test_film_grain_portrait_subthreshold_diagonal_speck_skips_antialiasing(self):
        """High-frequency film grain with a <4% diagonal noise speck must not trigger antialiasing."""
        src = _make_portrait_with_film_grain_and_diag_speck()
        self.assertIsNone(detect_aliasing(src))
        self.assertEqual(prepare_antialiasing(src), (None, None, {}))


def _make_blurred_face_collar_highlight_portrait() -> np.ndarray:
    """Create a 384x384 portrait with a blurred center face, crisp surroundings, and a ~200 px chroma collar bleed."""
    h, w = 384, 384
    yy, xx = np.mgrid[0:h, 0:w].astype(np.float32)
    hy, hx = np.mgrid[0:h * 2, 0:w * 2].astype(np.float32)
    base_2x = _make_clean_portrait(h * 2, w * 2, seed=101).astype(np.float32)
    phase_r = 0.08 * np.hypot(hx - 384.0, hy - 260.0) + 2.4 * np.sin(0.011 * hx - 0.017 * hy)
    strand = np.tanh(5.0 * np.sin(phase_r))
    hair_2x = np.stack(
        [base_2x[..., 0] + 52.0 * strand, base_2x[..., 1] + 28.0 * strand, base_2x[..., 2] - 24.0 * strand],
        axis=-1,
    )
    sharp_bg = np.asarray(
        Image.fromarray(np.clip(np.rint(hair_2x), 0, 255).astype(np.uint8)).resize((w, h), Image.Resampling.LANCZOS)
    ).astype(np.float32)

    base_1x = _make_clean_portrait(h, w, seed=101).astype(np.float32)
    face_pattern = (
        52.0 * np.tanh(3.5 * np.sin((xx - 192.0) * 0.09 + 0.4 * np.cos((yy - 192.0) * 0.07)))
        + 28.0 * np.cos((yy - 180.0) * 0.11)
    )[..., None]
    blurred_face = gaussian_filter(np.clip(base_1x + face_pattern, 0, 255), sigma=(3.8, 3.8, 0.0))

    ell = ((xx - 192.0) / 96.0) ** 2 + ((yy - 192.0) / 108.0) ** 2
    w_face = np.clip((1.05 - ell) / 0.14, 0.0, 1.0)[..., None]
    src = np.clip(np.rint(sharp_bg * (1.0 - w_face) + blurred_face * w_face), 0, 255).astype(np.uint8)

    collar = (yy >= 334) & (yy < 344) & (xx >= 182) & (xx < 202)  # 10x20 = 200 px localized highlight
    y_band = np.where(yy >= 339, 185.0, 85.0)
    cb_b = gaussian_filter(np.where(yy >= 339, 120.0, 0.0), sigma=2.0)
    cr_b = gaussian_filter(np.where(yy >= 339, -90.0, 0.0), sigma=2.0)
    rgb_b = np.stack(
        [
            np.clip(y_band + 1.402 * cb_b, 0, 255),
            np.clip(y_band - 0.344136 * cr_b - 0.714136 * cb_b, 0, 255),
            np.clip(y_band + 1.772 * cr_b, 0, 255),
        ],
        axis=-1,
    )
    src[collar] = np.clip(np.rint(rgb_b[collar]), 0, 255).astype(np.uint8)
    return src


def _make_non_power8_mosaic_with_pocket(*, block: int, size: int, radius: int) -> tuple[np.ndarray, tuple[int, int]]:
    """Create a P x P mosaic face region (non-8/16) containing a uniform low-contrast interior pocket."""
    base = _make_clean_portrait(size, size, seed=300 + block)
    yy, xx = np.mgrid[0:size, 0:size]
    mosaic = _make_flat_mosaic(base, block=block)
    src = base.copy()
    center = size // 2
    face_circle = ((xx - center) ** 2 + (yy - center) ** 2) <= radius ** 2
    src[face_circle] = mosaic[face_circle]
    p0, p1 = int(size * 0.41), int(size * 0.48)
    src[p0:p1, p0:p1] = np.array([185, 145, 125], dtype=np.uint8)
    pocket_center = ((p0 + p1) // 2, (p0 + p1) // 2)
    return src, pocket_center


def _make_portrait_with_film_grain_and_diag_speck() -> np.ndarray:
    """Create a 256x256 portrait with smooth diagonal curves, fine film grain, and a <4% staircased noise speck."""
    h, w = 256, 256
    hy, hx = np.mgrid[0:h * 2, 0:w * 2].astype(np.float32)
    hi_bg = _make_clean_portrait(h * 2, w * 2, seed=505).astype(np.float32)
    dist_diag = ((hx + 0.85 * hy) % 120.0) - 60.0
    step_smooth = 0.5 * (1.0 + np.tanh(dist_diag / 1.8)) * 110.0 - 55.0
    hi_diag = np.clip(hi_bg + step_smooth[..., None], 0, 255).astype(np.uint8)
    base_diag = np.asarray(Image.fromarray(hi_diag).resize((w, h), Image.Resampling.LANCZOS)).astype(np.float32)
    rng = np.random.default_rng(606)
    src = np.clip(np.rint(base_diag + rng.normal(0.0, 4.5, (h, w, 3))), 0, 255).astype(np.uint8)

    src[100:156, 100:156] = np.array([150, 140, 130], dtype=np.uint8)
    yy_p, xx_p = np.mgrid[100:156, 100:156]
    q_step = ((xx_p - 100) // 2) * 2 > ((yy_p - 100) // 2) * 2
    patch_sq = np.where(
        q_step[..., None],
        np.array([220, 210, 200], dtype=np.uint8),
        np.array([50, 45, 40], dtype=np.uint8),
    )
    mask_inner = (yy_p >= 110) & (yy_p < 146) & (xx_p >= 110) & (xx_p < 146)
    src[100:156, 100:156][mask_inner] = patch_sq[mask_inner]
    return src


def _make_request(headers: dict[str, str] | None = None) -> Request:
    raw_headers = [(b'host', b'127.0.0.1:8000')]
    for key, val in (headers or {}).items():
        raw_headers.append((key.lower().encode('ascii'), val.encode('ascii')))
    scope = {
        'type': 'http',
        'asgi': {'version': '3.0'},
        'http_version': '1.1',
        'method': 'POST',
        'path': '/',
        'raw_path': b'/',
        'headers': raw_headers,
        'query_string': b'',
        'client': ('127.0.0.1', 50000),
        'server': ('127.0.0.1', 8000),
    }
    return Request(scope)


def _route_endpoint(path: str):
    return next(route.endpoint for route in app_module.app.routes if getattr(route, 'path', None) == path)


class _E2ERouteTestBase(unittest.TestCase):
    """Shared FastAPI route test harness with optional face-detector stubbing."""

    PATCH_DETECT_FACES: bool = False

    @classmethod
    def setUpClass(cls):
        if not any(getattr(r, 'path', None) == '/capture/status' for r in app_module.app.routes):
            capture_api.register(app_module.app)

    def setUp(self):
        self.forward_inputs: list[bytes] = []
        self._tmpdir = tempfile.TemporaryDirectory()
        self._orig_store = capture_api.STORE
        capture_api.STORE = Path(self._tmpdir.name)
        self._orig_state = dict(app_module.STATE)
        app_module._frozen_frame_pixelation.cache_clear()
        self._patches = [
            patch.object(app_module, 'one_forward', side_effect=_fake_one_forward_builder(self.forward_inputs)),
        ]
        if self.PATCH_DETECT_FACES:
            self._patches.append(patch.object(capture_api, 'detect_faces', return_value=[]))
        for p in self._patches:
            p.start()

    def tearDown(self):
        for p in reversed(self._patches):
            p.stop()
        with capture_api.LOCK:
            capture_api.TOKENS.clear()
            capture_api.FRAMES.clear()
            capture_api.CROPS.clear()
        capture_api.STORE = self._orig_store
        app_module._frozen_frame_pixelation.cache_clear()
        app_module.STATE.clear()
        app_module.STATE.update(self._orig_state)
        self._tmpdir.cleanup()

    def _inspect_picture(self, image_bytes: bytes, *, filename: str = 'input.png', scan_faces: bool = True):
        upload = UploadFile(file=io.BytesIO(image_bytes), filename=filename)
        return asyncio.run(app_module.inspect(file=upload, scan_faces=scan_faces))

    def _restore_picture(self, image_bytes: bytes, *, filename: str = 'input.png',
                         picture_crop: list[int] | None = None):
        upload = UploadFile(file=io.BytesIO(image_bytes), filename=filename)
        body = asyncio.run(
            app_module.restore(
                file=upload,
                input_review='picture_upload',
                restoration='auto',
                preparation_mode='automatic',
                protect_mask=None,
                apply_mask=None,
                radius=1.0,
                preview_id='',
                confirmed=False,
                include_bundle=True,
                model=app_module.DEFAULT_MODEL,
                capture_id='',
                request=_make_request(),
                picture_crop=json.dumps(picture_crop) if picture_crop is not None else '',
                match_blur=False,
                include_smoothed_frame=True,
            )
        )
        json.dumps(body['metadata'], allow_nan=False)
        return body


class DegradationRoutesE2ETests(_E2ERouteTestBase):
    """FastAPI route end-to-end tests with real FBCNN and mocked Stage-1 V2 worker."""

    PATCH_DETECT_FACES = True

    def test_e2e_macroblock_png_exif6_jpeg_and_video_rtsp_capture(self):
        # Verify status invariants before running
        status_data = app_module.status()
        self.assertEqual(status_data['maximum_restore_forwards'], 1)
        self.assertEqual(status_data['maximum_off_forwards'], 0)

        # 1) 8x8 DCT macroblocked PNG (160x160) -> macroblock_fbcnn using real fbcnn_color.pth
        blocked_png_rgb = _make_block_dct_macroblocked(_make_macroblock_carrier_base(160, 160, seed=90), period=8, quant_step=34.0)
        res_png = self._restore_picture(png(blocked_png_rgb), filename='mb8.png')
        meta_png = res_png['metadata']
        self.assertEqual(meta_png['automatic_preparation']['route'], 'macroblock_fbcnn')
        self.assertEqual(meta_png['automatic_preparation']['preparation_model'], 'FBCNN RGB')
        self.assertEqual(meta_png['automatic_preparation']['preparation_model_sha256'], CHECKPOINT_SHA256)
        self.assertEqual(meta_png['neural_forward_attempts'], 1)
        self.assertEqual(meta_png['neural_forward_returns'], 1)
        self.assertTrue(meta_png['Auto_assessed_original_canvas_before_preparation'])
        self.assertTrue(meta_png['preparation_applied'])
        zip_png = _decode_zip_from_data_url(res_png['bundle'])
        self.assertIn('derived-native.png', zip_png)
        self.assertIn('macroblock-mask.png', zip_png)
        self.assertIn('automatic-decision.json', zip_png)
        self.assertIn('processing.json', zip_png)

        # 2) EXIF-rotated Orientation=6 8x8 macroblocked JPEG -> macroblock_fbcnn (not Gaussian)
        exif6_jpeg = _encode_jpeg_bytes(blocked_png_rgb, quality=12, orientation=6)
        with Image.open(io.BytesIO(exif6_jpeg)) as check_jpg:
            self.assertEqual(check_jpg.getexif().get(274), 6)
            self.assertGreaterEqual(int(check_jpg.quantization[0][0]), 45)
        res_exif6 = self._restore_picture(exif6_jpeg, filename='exif6.jpg')
        meta_exif6 = res_exif6['metadata']
        self.assertEqual(meta_exif6['automatic_preparation']['route'], 'macroblock_fbcnn')
        self.assertNotIn('gaussian', meta_exif6['automatic_preparation']['route'])
        self.assertEqual(meta_exif6['automatic_preparation']['preparation_model'], 'FBCNN RGB')
        self.assertEqual(meta_exif6['automatic_preparation']['preparation_forward_returns'], 1)

        # 3) Video/RTSP frozen PNG capture_id -> macroblock_fbcnn (not bypassed or Gaussian-blurred)
        cap_status_endpoint = _route_endpoint('/capture/status')
        cap_status_resp = cap_status_endpoint(_make_request())
        token = json.loads(cap_status_resp.body.decode('utf-8'))['token']
        authed_req = _make_request({'X-Capture-Token': token})

        frame_rgb = _make_block_dct_macroblocked(_make_macroblock_carrier_base(192, 192, seed=91), period=8, quant_step=34.0)
        frame_bytes = png(frame_rgb)
        cap_frame_endpoint = _route_endpoint('/capture/frames')
        frame_data = asyncio.run(
            cap_frame_endpoint(
                request=authed_req,
                file=UploadFile(file=io.BytesIO(frame_bytes), filename='frame.png'),
                source_kind='rtsp',
                source_time=1.25,
            )
        )
        frame_id = frame_data['frame_id']
        cap_crop_endpoint = _route_endpoint('/capture/frames/{ident}/crop')
        crop_data = cap_crop_endpoint(
            request=authed_req,
            ident=frame_id,
            rect=capture_api.Rectangle(x=16, y=16, width=160, height=160),
        )
        crop_png = png(frame_rgb[16:176, 16:176])
        restore_vid = asyncio.run(
            app_module.restore(
                file=UploadFile(file=io.BytesIO(crop_png), filename='crop.png'),
                input_review='video_capture',
                restoration='auto',
                preparation_mode='automatic',
                protect_mask=None,
                apply_mask=None,
                radius=1.0,
                preview_id='',
                confirmed=False,
                include_bundle=True,
                model=app_module.DEFAULT_MODEL,
                capture_id=crop_data['capture_id'],
                request=authed_req,
                picture_crop='',
                match_blur=False,
                include_smoothed_frame=True,
            )
        )
        vid_meta = restore_vid['metadata']
        json.dumps(vid_meta, allow_nan=False)
        self.assertEqual(vid_meta['automatic_preparation']['route'], 'macroblock_fbcnn')
        self.assertIsNone(vid_meta.get('frame_pixelation'))
        self.assertEqual(vid_meta['neural_forward_returns'], 1)

    def test_e2e_16x16_localized_out_of_range_privacy_mosaic_and_clean_controls(self):
        # 16x16 -> macroblock_hevc
        blocked_16 = _make_block_dct_macroblocked(_make_macroblock_carrier_base(160, 160, seed=92), period=16, quant_step=44.0)
        res_16 = self._restore_picture(png(blocked_16))
        self.assertEqual(res_16['metadata']['automatic_preparation']['route'], 'macroblock_hevc')

        # Out-of-range 64x64 -> macroblock_hevc
        blocked_64 = _make_block_dct_macroblocked(_make_macroblock_carrier_base(64, 64, seed=93), period=8, quant_step=36.0)
        res_64 = self._restore_picture(png(blocked_64))
        self.assertEqual(res_64['metadata']['automatic_preparation']['route'], 'macroblock_hevc')

        # Face-localized 8x8 macroblocking inside a clean 224x224 portrait -> macroblock_fbcnn & exterior bit-exact
        base_224 = _make_macroblock_carrier_base(224, 224, seed=94)
        mb_full_224 = _make_block_dct_macroblocked(base_224, period=8, quant_step=34.0)
        local_mb = base_224.copy()
        local_mb[48:176, 48:176] = mb_full_224[48:176, 48:176]
        res_loc = self._restore_picture(png(local_mb))
        self.assertEqual(res_loc['metadata']['automatic_preparation']['route'], 'macroblock_fbcnn')
        zip_loc = _decode_zip_from_data_url(res_loc['bundle'])
        self.assertIn('macroblock-mask.png', zip_loc)
        mb_mask = _decode_mask_bytes(zip_loc['macroblock-mask.png'])
        derived_loc = _decode_png_bytes(zip_loc['derived-native.png'])
        self.assertTrue(np.array_equal(derived_loc[~mb_mask], local_mb[~mb_mask]))

        # Flat 8x8 privacy mosaic -> still takes Gaussian/pixelation privacy route (never macroblock_*)
        flat_mosaic = _make_flat_mosaic(_make_clean_portrait(192, 192, seed=96), block=8)
        res_flat = self._restore_picture(png(flat_mosaic))
        self.assertIn(res_flat['metadata']['automatic_preparation']['route'], ('picture_region_gaussian', 'face_gaussian'))

        # Clean LANCZOS portrait -> direct route
        clean = _make_clean_portrait(192, 192, seed=97)
        res_clean = self._restore_picture(png(clean))
        self.assertEqual(res_clean['metadata']['automatic_preparation']['route'], 'direct')

    def test_e2e_single_composable_routes_and_compound_combinations(self):
        # Single route: edge_deringing
        ringed, _ = _make_ringing_image(160)
        res_ring = self._restore_picture(png(ringed))
        self.assertEqual(res_ring['metadata']['automatic_preparation']['route'], 'edge_deringing')
        zip_ring = _decode_zip_from_data_url(res_ring['bundle'])
        self.assertIn('deringing-halo-mask.png', zip_ring)

        # Single route: gradient_debanding
        banded, _ = _make_banded_gradient(160, step=7)
        res_band = self._restore_picture(png(banded))
        self.assertEqual(res_band['metadata']['automatic_preparation']['route'], 'gradient_debanding')
        zip_band = _decode_zip_from_data_url(res_band['bundle'])
        self.assertIn('debanding-mask.png', zip_band)

        # Single route: chroma_debleeding
        bled, _ = _make_color_bleeding_image(160)
        res_bleed = self._restore_picture(png(bled))
        self.assertEqual(res_bleed['metadata']['automatic_preparation']['route'], 'chroma_debleeding')
        zip_bleed = _decode_zip_from_data_url(res_bleed['bundle'])
        self.assertIn('color-bleeding-mask.png', zip_bleed)

        # Single route: tangent_antialiasing
        aliased, _ = _make_aliased_diagonal_image(160)
        res_aa = self._restore_picture(png(aliased))
        self.assertEqual(res_aa['metadata']['automatic_preparation']['route'], 'tangent_antialiasing')
        zip_aa = _decode_zip_from_data_url(res_aa['bundle'])
        self.assertIn('antialiasing-edge-mask.png', zip_aa)

        # Compound A: 8x8 macroblocking + 4:2:0 chroma bleed stacked on a 192x192 canvas
        mb_base = _make_block_dct_macroblocked(_make_macroblock_carrier_base(192, 192, seed=101), period=8, quant_step=34.0)
        bled_192, _ = _make_color_bleeding_image(192)
        compound_a = mb_base.copy()
        compound_a[80:192, :] = bled_192[80:192, :]
        res_comp_a = self._restore_picture(png(compound_a))
        meta_a = res_comp_a['metadata']
        auto_a = meta_a['automatic_preparation']
        self.assertEqual(auto_a['route'], 'compound_artifact_preparation')
        self.assertIn('macroblock_fbcnn', auto_a['applied_stages'])
        self.assertIn('chroma_debleeding', auto_a['applied_stages'])
        self.assertEqual(auto_a['stages'], auto_a['applied_stages'])
        self.assertIn('macroblock_fbcnn', auto_a['stage_reports'])
        self.assertIn('chroma_debleeding', auto_a['stage_reports'])

        zip_a = _decode_zip_from_data_url(res_comp_a['bundle'])
        self.assertIn('derived-native.png', zip_a)
        self.assertIn('macroblock-mask.png', zip_a)
        self.assertIn('color-bleeding-mask.png', zip_a)
        decision_a = json.loads(zip_a['automatic-decision.json'].decode('utf-8'))
        processing_a = json.loads(zip_a['processing.json'].decode('utf-8'))
        json.dumps(decision_a, allow_nan=False)
        json.dumps(processing_a, allow_nan=False)
        self.assertEqual(decision_a['route'], 'compound_artifact_preparation')
        self.assertEqual(decision_a['applied_stages'], auto_a['applied_stages'])
        self.assertEqual(processing_a['applied_stages'], auto_a['applied_stages'])
        self.assertEqual(processing_a['stages'], auto_a['stages'])
        self.assertEqual(set(processing_a['stage_reports']), set(auto_a['stage_reports']))

        # Compound B: ringing + banding + color bleeding on an organic unified curved scene
        compound_b = _make_unified_compound_scene(224)

        res_comp_b = self._restore_picture(png(compound_b))
        auto_b = res_comp_b['metadata']['automatic_preparation']
        self.assertEqual(auto_b['route'], 'compound_artifact_preparation')
        for expected_stage in ('edge_deringing', 'gradient_debanding', 'chroma_debleeding'):
            self.assertIn(expected_stage, auto_b['applied_stages'])
            self.assertIn(expected_stage, auto_b['stage_reports'])
        zip_b = _decode_zip_from_data_url(res_comp_b['bundle'])
        for expected_file in (
            'derived-native.png',
            'automatic-decision.json',
            'processing.json',
            'deringing-halo-mask.png',
            'debanding-mask.png',
            'color-bleeding-mask.png',
        ):
            self.assertIn(expected_file, zip_b)
        decision_b = json.loads(zip_b['automatic-decision.json'].decode('utf-8'))
        processing_b = json.loads(zip_b['processing.json'].decode('utf-8'))
        json.dumps(decision_b, allow_nan=False)
        json.dumps(processing_b, allow_nan=False)
        self.assertEqual(decision_b['stages'], auto_b['applied_stages'])
        self.assertEqual(processing_b['stages'], auto_b['applied_stages'])
        self.assertEqual(len(self.forward_inputs), 6)
        self.assertEqual(app_module.STATE['neural_forward_attempts'], 0)

    def test_blurred_face_collar_highlight_routes_protect_auto(self):
        """Blurred face + sharp surroundings with ~200 px high-chroma highlight must route protect_auto, not chroma_debleeding."""
        src = _make_blurred_face_collar_highlight_portrait()
        self.assertIsNone(detect_color_bleeding(src))
        with patch.object(capture_api, 'detect_faces', return_value=[[96, 84, 192, 216]]):
            body = self._restore_picture(png(src), filename='blurred_face_collar.png')

        meta = body['metadata']
        auto = meta['automatic_preparation']
        self.assertEqual(auto['route'], 'protect_auto')
        self.assertNotIn('chroma_debleeding', auto.get('applied_stages', []))
        prep = meta['preparation']
        bm = prep['algorithm']['blur_matching']
        self.assertEqual(bm['status'], 'estimated')
        self.assertGreaterEqual(bm['target_radius_at256'], 2.5)
        self.assertTrue(prep['every_protected_native_RGB_exact'])

        zip_entries = _decode_zip_from_data_url(body['bundle'])
        protect_mask = _decode_mask_bytes(zip_entries['effective-Protect-native.png'])
        orig_native = _decode_png_bytes(zip_entries['original-decoded-rgb.png'])
        derived_native = _decode_png_bytes(zip_entries['derived-native.png'])
        self.assertTrue(protect_mask.any())
        self.assertTrue(np.array_equal(derived_native[protect_mask], orig_native[protect_mask]))
        self.assertGreater(prep['changed_native_pixels'], 0)

    def test_e2e_noise_denoise_cast_dehaze_spatial_comb_and_negative_controls(self):
        """Verify Step 5 noise_denoise, cast/dehaze, spatial_comb routes, evidence masks, precedence, and negative controls."""
        clean = _make_clean_portrait(160, 160, seed=314)

        # 1) Negative control: clean portrait must route direct without false triggers
        self.assertIsNone(detect_noise(clean))
        self.assertIsNone(detect_cast_dehaze(clean))
        self.assertEqual(detect_and_clean_spatial_comb(clean), (None, None, {}))
        res_clean = self._restore_picture(png(clean), filename='clean_portrait.png')
        self.assertEqual(res_clean['metadata']['automatic_preparation']['route'], 'direct')
        self.assertFalse(res_clean['metadata']['preparation_applied'])

        # 2) Noise-only -> noise_denoise + noise-denoise-mask.png + native Immerkaer signal
        rng = np.random.default_rng(77)
        noisy = np.clip(clean.astype(np.float32) + rng.normal(0.0, 9.5, clean.shape), 0, 255).astype(np.uint8)
        n_det = detect_noise(noisy)
        self.assertIsNotNone(n_det)
        self.assertEqual(n_det['policy'], NOISE_POLICY)
        res_noise = self._restore_picture(png(noisy), filename='noisy.png')
        meta_noise = res_noise['metadata']
        self.assertEqual(meta_noise['automatic_preparation']['route'], 'noise_denoise')
        self.assertTrue(meta_noise['quality_signals']['noise_gate_triggered'])
        self.assertGreater(meta_noise['quality_signals']['noise_sigma_native'], 4.0)
        zip_noise = _decode_zip_from_data_url(res_noise['bundle'])
        self.assertIn('noise-denoise-mask.png', zip_noise)
        self.assertIn('noise-cleaned.png', zip_noise)

        # 3) Strong color cast -> color_cast_balance + cast-dehaze-mask.png
        warm = clean.astype(np.float32)
        warm[..., 0] = warm[..., 0] * 1.45 + 15.0
        warm[..., 1] *= 0.90
        warm[..., 2] *= 0.45
        warm_u8 = np.clip(np.rint(warm), 0, 255).astype(np.uint8)
        res_cast = self._restore_picture(png(warm_u8), filename='cast.png')
        self.assertEqual(res_cast['metadata']['automatic_preparation']['route'], 'color_cast_balance')
        zip_cast = _decode_zip_from_data_url(res_cast['bundle'])
        self.assertIn('cast-dehaze-mask.png', zip_cast)

        # 4) Atmospheric haze -> haze_dehaze + cast-dehaze-mask.png
        hazy_u8 = np.clip(np.rint(clean.astype(np.float32) * 0.42 + 235.0 * 0.58), 0, 255).astype(np.uint8)
        res_haze = self._restore_picture(png(hazy_u8), filename='haze.png')
        self.assertEqual(res_haze['metadata']['automatic_preparation']['route'], 'haze_dehaze')
        zip_haze = _decode_zip_from_data_url(res_haze['bundle'])
        self.assertIn('cast-dehaze-mask.png', zip_haze)

        # 5) Combined cast + haze -> cast_dehaze (both gates active)
        forced_out, forced_mask, forced_route, forced_report = prepare_cast_dehaze(hazy_u8, force_mode='both')
        self.assertEqual(forced_route, 'cast_dehaze')
        with patch('implementation.automatic_preparation.prepare_cast_dehaze', return_value=(forced_out, forced_mask, forced_route, forced_report)):
            res_both = self._restore_picture(png(hazy_u8), filename='cast_haze.png')
        self.assertEqual(res_both['metadata']['automatic_preparation']['route'], 'cast_dehaze')
        zip_both = _decode_zip_from_data_url(res_both['bundle'])
        self.assertIn('cast-dehaze-mask.png', zip_both)

        # 6a) Full-frame scanline_cleanup suppresses redundant spatial_comb in composables
        full_comb = clean.astype(np.int16)
        full_comb[::2, :, :] += 14
        full_comb[1::2, :, :] -= 14
        res_full_comb = self._restore_picture(png(np.clip(full_comb, 0, 255).astype(np.uint8)), filename='full_comb.png')
        self.assertEqual(res_full_comb['metadata']['automatic_preparation']['route'], 'scanline_cleanup')
        self.assertNotIn('spatial_comb', res_full_comb['metadata']['automatic_preparation']['applied_stages'])

        # 6b) Localized crop spatial comb (missed by full-frame FFT) -> spatial_comb + spatial-comb-cleaned.png
        big_clean = _make_clean_portrait(640, 640, seed=314)
        local_comb = big_clean.astype(np.int16)
        local_comb[240:400:2, 240:400, :] += 14
        local_comb[241:400:2, 240:400, :] -= 14
        res_comb = self._restore_picture(
            png(np.clip(local_comb, 0, 255).astype(np.uint8)),
            filename='comb_crop.png',
            picture_crop=[240, 240, 160, 160],
        )
        auto_comb = res_comb['metadata']['automatic_preparation']
        self.assertIn('spatial_comb', auto_comb['applied_stages'])
        self.assertEqual(res_comb['metadata']['scanline_cleanup']['spacing_native_pixels'], 2.0)
        zip_comb = _decode_zip_from_data_url(res_comb['bundle'])
        self.assertIn('spatial-comb-cleaned.png', zip_comb)

        # 7) Precedence: noise_denoise precedes cast_dehaze & composables when combined
        noisy_cast = np.clip(warm_u8.astype(np.float32) + rng.normal(0.0, 9.5, clean.shape), 0, 255).astype(np.uint8)
        res_compound_nc = self._restore_picture(png(noisy_cast), filename='noisy_cast.png')
        stages_nc = res_compound_nc['metadata']['automatic_preparation']['applied_stages']
        self.assertEqual(stages_nc[0], 'noise_denoise')
        self.assertGreater(len(stages_nc), 1)
        self.assertEqual(app_module.status()['maximum_restore_forwards'], 1)
        self.assertEqual(app_module.status()['maximum_off_forwards'], 0)


_REAL_TEST_IMAGES = ('wild1.png', 'wild2.png', 'wild3.png', 'wild4.png', 'wild5.png', 'synthetic_data_6.png')
_TEST_IMAGES_DIR = APP_ROOT.parent / 'test_images'


@unittest.skipUnless(
    all((_TEST_IMAGES_DIR / name).exists() for name in _REAL_TEST_IMAGES),
    'test_images/ regression fixtures missing',
)
class RealTestImagesE2ETests(_E2ERouteTestBase):
    """End-to-end app.restore regressions against real test_images/ fixtures and real face detection."""

    PATCH_DETECT_FACES = False

    def _read_fixture(self, name: str) -> bytes:
        return (_TEST_IMAGES_DIR / name).read_bytes()

    def test_wild5_full_and_inspect_crop_protect_auto_matches_surroundings(self):
        raw = self._read_fixture('wild5.png')
        inspected = self._inspect_picture(raw, filename='wild5.png', scan_faces=True)
        self.assertTrue(inspected['face_crops'])
        crop_box = inspected['face_crops'][0]

        for label_tag, crop in (('full', None), ('crop', crop_box)):
            with self.subTest(mode=label_tag):
                app_module._frozen_frame_pixelation.cache_clear()
                body = self._restore_picture(raw, filename='wild5.png', picture_crop=crop)
                meta = body['metadata']
                auto = meta['automatic_preparation']
                self.assertEqual(auto['route'], 'protect_auto')
                self.assertNotIn('chroma_debleeding', auto.get('applied_stages', []))

                prep = meta['preparation']
                bm = prep['algorithm']['blur_matching']
                self.assertEqual(bm['status'], 'estimated')
                self.assertGreaterEqual(bm['target_radius_at256'], 3.0)
                self.assertTrue(prep['every_protected_native_RGB_exact'])

                zip_entries = _decode_zip_from_data_url(body['bundle'])
                protect_mask = _decode_mask_bytes(zip_entries['effective-Protect-native.png'])
                orig_native = _decode_png_bytes(zip_entries['original-decoded-rgb.png'])
                derived_native = _decode_png_bytes(zip_entries['derived-native.png'])
                self.assertTrue(protect_mask.any())
                self.assertTrue(np.array_equal(derived_native[protect_mask], orig_native[protect_mask]))
                self.assertGreater(int(np.any(derived_native[~protect_mask] != orig_native[~protect_mask], axis=1).sum()), 0)

                tiles_before, _ = estimate_tiles_normalized(Image.fromarray(orig_native), max_longest=512)
                tiles_after, _ = estimate_tiles_normalized(Image.fromarray(derived_native), max_longest=512)
                self.assertLess(float(np.nanmedian(tiles_before)), 1.5)
                self.assertGreaterEqual(float(np.nanmedian(tiles_after)), 3.0)

    def test_wild3_routes_unified_picture_region_gaussian_radius_3(self):
        body = self._restore_picture(self._read_fixture('wild3.png'), filename='wild3.png')
        meta = body['metadata']
        self.assertEqual(meta['automatic_preparation']['route'], 'picture_region_gaussian')
        pic_pix = meta['picture_pixelation']
        self.assertGreaterEqual(pic_pix['masked_native_pixels'], 88000)
        self.assertEqual(pic_pix['Gaussian_radius_at256'], 3.0)
        self.assertEqual(len(pic_pix['region_bounds']), 1)

        zip_entries = _decode_zip_from_data_url(body['bundle'])
        mask = _decode_mask_bytes(zip_entries['full-image-pixelation-mask.png'])
        orig = _decode_png_bytes(zip_entries['original-native.png'])
        derived = _decode_png_bytes(zip_entries['derived-native.png'])
        self.assertEqual(label(mask)[1], 1)
        self.assertTrue(np.array_equal(derived[~mask], orig[~mask]))

    def test_wild4_routes_picture_region_gaussian_radius_3_5_outside_exact(self):
        body = self._restore_picture(self._read_fixture('wild4.png'), filename='wild4.png')
        meta = body['metadata']
        self.assertEqual(meta['automatic_preparation']['route'], 'picture_region_gaussian')
        pic_pix = meta['picture_pixelation']
        self.assertEqual(pic_pix['Gaussian_radius_at256'], 3.5)

        zip_entries = _decode_zip_from_data_url(body['bundle'])
        mask = _decode_mask_bytes(zip_entries['full-image-pixelation-mask.png'])
        orig = _decode_png_bytes(zip_entries['original-native.png'])
        derived = _decode_png_bytes(zip_entries['derived-native.png'])
        self.assertTrue(mask.any())
        self.assertTrue(np.array_equal(derived[~mask], orig[~mask]))

    def test_wild1_wild2_and_synthetic_data_6_routes(self):
        expected = (
            ('wild1.png', 'direct', False),
            ('wild2.png', 'scanline_cleanup', True),
            ('synthetic_data_6.png', 'compound_artifact_preparation', True),
        )
        for filename, expected_route, expected_prep_applied in expected:
            with self.subTest(filename=filename):
                app_module._frozen_frame_pixelation.cache_clear()
                body = self._restore_picture(self._read_fixture(filename), filename=filename)
                meta = body['metadata']
                auto = meta['automatic_preparation']
                self.assertEqual(auto['route'], expected_route)
                self.assertIs(meta['preparation_applied'], expected_prep_applied)
                if filename == 'wild2.png':
                    self.assertNotIn('Gaussian_radius_at256', meta['scanline_cleanup'])
                    self.assertIn('comb_ela_masked_pixels', meta['scanline_cleanup'])
                    self.assertIn('chroma_cleanup_pixels', meta['scanline_cleanup'])
                if filename == 'synthetic_data_6.png':
                    self.assertEqual(auto['applied_stages'][0], 'noise_denoise')

    def test_synthetic_data_real_routes_step5(self):
        """Verify Step 5 winners route #2, #7, #9, #10 cleanly without #1/#4 cast false triggers."""
        checks = (
            ('synthetic_data_1.png', 'macroblock_fbcnn', None, {'color_cast_balance', 'haze_dehaze', 'cast_dehaze'}),
            ('synthetic_data_2.png', 'compound_artifact_preparation', ('color_cast_balance', 'macroblock_fbcnn'), set()),
            ('synthetic_data_4.png', 'macroblock_hevc', None, {'color_cast_balance', 'haze_dehaze', 'cast_dehaze'}),
            ('synthetic_data_7.png', 'compound_artifact_preparation', ('macroblock_hevc',), {'picture_region_gaussian'}),
            ('synthetic_data_9.png', 'noise_denoise', ('noise_denoise',), set()),
            ('synthetic_data_10.png', 'compound_artifact_preparation', ('haze_dehaze', 'macroblock_fbcnn'), set()),
        )
        for filename, expected_route, required_stages, forbidden_names in checks:
            path = _TEST_IMAGES_DIR / filename
            if not path.exists():
                continue
            with self.subTest(filename=filename):
                app_module._frozen_frame_pixelation.cache_clear()
                body = self._restore_picture(path.read_bytes(), filename=filename)
                meta = body['metadata']
                auto = meta['automatic_preparation']
                self.assertEqual(auto['route'], expected_route)
                self.assertNotIn('gaussian', auto['route'])
                stages = auto.get('applied_stages', [])
                for stage in (required_stages or ()):
                    self.assertIn(stage, stages)
                for forbidden in forbidden_names:
                    self.assertNotIn(forbidden, stages)
                    self.assertNotEqual(auto['route'], forbidden)
                if filename == 'synthetic_data_9.png':
                    zip_entries = _decode_zip_from_data_url(body['bundle'])
                    self.assertIn('noise-denoise-mask.png', zip_entries)
                if filename in ('synthetic_data_2.png', 'synthetic_data_10.png'):
                    zip_entries = _decode_zip_from_data_url(body['bundle'])
                    self.assertIn('cast-dehaze-mask.png', zip_entries)


class UIAndImplementationHygieneTests(unittest.TestCase):
    """Verify UI preparationSummary strings in implementation.js, HTML markup, and AST import hygiene."""

    def test_ui_preparation_summary_routes_and_template_hints(self):
        html_path = APP_ROOT / 'templates' / 'implementation.html'
        html = html_path.read_text(encoding='utf-8')
        self.assertIn('/static/implementation.js?v=37', html)
        self.assertIn('macroblocking, ringing, banding, aliasing, color bleeding', html)

        js_path = (APP_ROOT / 'static' / 'implementation.js').resolve()
        node_script = r"""
const assert = require('node:assert/strict');
const fs = require('node:fs');
const vm = require('node:vm');

const src = fs.readFileSync(__JS_PATH__, 'utf8');
const match = src.match(/const preparationSummary = metadata => \{([\s\S]*?)\n    \};/);
assert.ok(match, 'preparationSummary definition not found in implementation.js');

const fn = vm.runInNewContext('(metadata => {' + match[1] + '\n})');

const expectedRoutes = [
  ['noise_denoise', /additive and chroma noise/i],
  ['cast_dehaze', /balanced color cast and dehazed/i],
  ['color_cast_balance', /residual color cast/i],
  ['haze_dehaze', /atmospheric haze/i],
  ['spatial_comb', /spatial interlacing comb/i],
  ['macroblock_fbcnn', /FBCNN/i],
  ['macroblock_hevc', /adaptive boundary deblocking/i],
  ['edge_deringing', /ringing halos/i],
  ['gradient_debanding', /quantized gradient banding/i],
  ['tangent_antialiasing', /jagged diagonal contours/i],
  ['chroma_debleeding', /chromatic bleeding/i],
];

for (const [route, pattern] of expectedRoutes) {
  const out = fn({ automatic_preparation: { route } });
  assert.equal(typeof out, 'string');
  assert.match(out, pattern, 'Unexpected summary for ' + route + ': ' + out);
}

const compStages = fn({
  automatic_preparation: {
    route: 'compound_artifact_preparation',
    stages: [
      'noise_denoise',
      'cast_dehaze',
      'color_cast_balance',
      'haze_dehaze',
      'spatial_comb',
      'macroblock_fbcnn',
      'edge_deringing',
      'gradient_debanding',
      'tangent_antialiasing',
      'chroma_debleeding',
    ],
  },
});
assert.match(compStages, /noise denoising/);
assert.match(compStages, /color cast and haze correction/);
assert.match(compStages, /color cast balancing/);
assert.match(compStages, /haze dehazing/);
assert.match(compStages, /spatial comb cleanup/);
assert.match(compStages, /FBCNN macroblock deblocking/);
assert.match(compStages, /edge deringing/);
assert.match(compStages, /gradient debanding/);
assert.match(compStages, /tangent anti-aliasing/);
assert.match(compStages, /chroma debleeding/);

const compApplied = fn({
  automatic_preparation: {
    route: 'compound_artifact_preparation',
    applied_stages: ['macroblock_hevc', 'gradient_debanding'],
  },
});
assert.match(compApplied, /adaptive macroblock deblocking/);
assert.match(compApplied, /gradient debanding/);

const compFallback = fn({ automatic_preparation: { route: 'compound_artifact_preparation', stages: [] } });
assert.match(compFallback, /multi-stage artifact preparation/);

assert.match(fn({ automatic_preparation: { route: 'protect_auto' } }), /Protect \+ Auto-match/);
assert.match(fn({ automatic_preparation: { route: 'direct' } }), /direct restoration/);
assert.match(fn({ preparation_applied: true }), /reviewed preparation/);
assert.match(fn({ preparation_applied: false }), /No preparation/);
""".replace('__JS_PATH__', json.dumps(str(js_path)))
        proc = subprocess.run(['node', '-e', node_script], capture_output=True, text=True, check=False)
        self.assertEqual(proc.returncode, 0, f'Node UI check failed:\nSTDOUT:{proc.stdout}\nSTDERR:{proc.stderr}')

    def test_implementation_ast_forbids_skimage_torchvision_and_cv2_ximgproc(self):
        impl_dir = APP_ROOT / 'implementation'
        forbidden_roots = {'skimage', 'torchvision'}
        violations: list[str] = []

        for py_file in sorted(impl_dir.glob('*.py')):
            tree = ast.parse(py_file.read_text(encoding='utf-8'), filename=str(py_file))
            for node in ast.walk(tree):
                if isinstance(node, ast.Import):
                    for alias in node.names:
                        top = alias.name.split('.')[0]
                        if top in forbidden_roots or alias.name == 'cv2.ximgproc' or alias.name.startswith('cv2.ximgproc.'):
                            violations.append(f'{py_file.name}:{node.lineno} import {alias.name}')
                elif isinstance(node, ast.ImportFrom):
                    mod = node.module or ''
                    top = mod.split('.')[0] if mod else ''
                    if top in forbidden_roots or mod == 'cv2.ximgproc' or mod.startswith('cv2.ximgproc.'):
                        violations.append(f'{py_file.name}:{node.lineno} from {mod} import ...')
                    elif mod == 'cv2':
                        for alias in node.names:
                            if alias.name == 'ximgproc':
                                violations.append(f'{py_file.name}:{node.lineno} from cv2 import ximgproc')
                elif isinstance(node, ast.Attribute) and node.attr == 'ximgproc':
                    if isinstance(node.value, ast.Name) and node.value.id == 'cv2':
                        violations.append(f'{py_file.name}:{node.lineno} cv2.ximgproc')

        self.assertEqual(violations, [], f'Forbidden dependencies detected: {violations}')


if __name__ == '__main__':
    unittest.main()
