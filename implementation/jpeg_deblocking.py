"""Guarded CPU preparation for strongly blocked JPEG uploads."""
from __future__ import annotations

import hashlib
import io
import logging
import subprocess
import sys

import numpy as np
from PIL import Image
from scipy.ndimage import distance_transform_edt

from .model_store import model_directory
from .preparation import decode_original, png
from .runtime import ROOT, ordinary_file

CHECKPOINT = 'fbcnn_color.pth'
CHECKPOINT_BYTES = 287755111
CHECKPOINT_SHA256 = '8b0e4ef23d59cf7ac934a342cb31a17619e4fa4a0b3374a9d78c5174312387e8'
MAX_SIDE = 384
MAX_PNG_BYTES = 512 * 1024
LOGGER = logging.getLogger(__name__)


def jpeg_seams(rgb, origin):
    """Measure excess luma jump at the native JPEG phase on both axes."""
    gray = np.asarray(Image.fromarray(rgb).convert('L'), dtype=np.float32)
    values = []
    for axis, offset in enumerate(reversed(origin)):
        profile = np.abs(np.diff(gray, axis=axis)).mean(axis=1-axis)
        phase = (np.arange(len(profile)) + 1 + offset) % 8
        boundary = float(profile[phase == 0].mean())
        interior = float(profile[phase != 0].mean())
        midpoint = float(profile[phase == 4].mean())
        values.append((boundary - interior, midpoint - interior, interior))
    return values


def jpeg_luma_dc(contents):
    if not contents.startswith(b'\xff\xd8'):
        return None
    with Image.open(io.BytesIO(contents)) as source:
        if source.format != 'JPEG' or source.getexif().get(274, 1) != 1:
            return None
        luma = source.quantization.get(0, [])
        return luma[0] if len(luma) == 64 else None


def run_deblock(crop):
    if not checkpoint_path().is_file():
        return None
    try:
        worker = subprocess.run(
            [sys.executable, '-B', '-m', 'implementation.jpeg_deblocking'],
            cwd=ROOT, input=png(crop), stdout=subprocess.PIPE, stderr=subprocess.PIPE,
            timeout=60, check=True, creationflags=getattr(subprocess, 'CREATE_NO_WINDOW', 0),
        )
        if len(worker.stdout) > MAX_PNG_BYTES:
            raise ValueError('JPEG preparation worker output exceeds bounds.')
        cleaned = decode_original(worker.stdout)
        if cleaned.shape != crop.shape:
            raise ValueError('JPEG preparation worker changed crop geometry.')
    except (OSError, subprocess.SubprocessError, ValueError) as error:
        LOGGER.warning('JPEG deblocking skipped after worker failure (%s).', type(error).__name__)
        return None
    return cleaned


def prepare_jpeg(contents, crop, rectangle):
    """Deblock one small native crop when JPEG metadata and pixel seams agree."""
    height, width = crop.shape[:2]
    if min(width, height) < 96 or max(width, height) > MAX_SIDE:
        return None, None
    dc = jpeg_luma_dc(contents)
    if dc is None or dc < 45:
        return None, None
    origin = tuple(rectangle[:2]) if rectangle is not None else (0, 0)
    before = jpeg_seams(crop, origin)
    if any(seam < max(1.0, .65 * interior) or midpoint > .5 * seam
           for seam, midpoint, interior in before):
        return None, None
    cleaned = run_deblock(crop)
    if cleaned is None:
        return None, None
    after = jpeg_seams(cleaned, origin)
    if any(abs(current[0]) >= .8 * original[0] for original, current in zip(before, after)):
        return None, None
    return cleaned, {
        'policy': 'guarded-jpeg-fbcnn-v1',
        'native_size': [width, height],
        'jpeg_luma_dc_quantization': dc,
        'jpeg_grid_origin': list(origin),
        'seam_excess_before': [round(item[0], 4) for item in before],
        'seam_excess_after': [round(item[0], 4) for item in after],
        'preparation_model': 'FBCNN RGB',
        'preparation_model_sha256': CHECKPOINT_SHA256,
        'preparation_forward_returns': 1,
    }


def checkpoint_path():
    return model_directory(ROOT, 'implementation') / CHECKPOINT


def _axis_8px_metrics(gray, axis, offset, mask_bool=None):
    """Measure 8-px seam excess, 4-px midpoint excess, and non-seam interior gradient on one axis."""
    diffs = np.abs(np.diff(gray, axis=axis))
    positions = (np.arange(diffs.shape[axis]) + 1 + offset) % 8
    if mask_bool is None:
        profile = diffs.mean(axis=1 - axis)
        boundary = float(profile[positions == 0].mean())
        interior = float(profile[positions != 0].mean())
        midpoint = float(profile[positions == 4].mean())
        return boundary - interior, midpoint - interior, interior
    allowed = mask_bool[:-1, :] & mask_bool[1:, :] if axis == 0 else mask_bool[:, :-1] & mask_bool[:, 1:]
    phase_view = positions[:, None] if axis == 0 else positions[None, :]
    b_mask = allowed & (phase_view == 0)
    i_mask = allowed & (phase_view != 0)
    m_mask = allowed & (phase_view == 4)
    if int(b_mask.sum()) < 16 or int(i_mask.sum()) < 64 or int(m_mask.sum()) < 16:
        return None
    boundary = float(diffs[b_mask].mean())
    interior = float(diffs[i_mask].mean())
    midpoint = float(diffs[m_mask].mean())
    return boundary - interior, midpoint - interior, interior


def _feather_regional_mask(regional_mask, shape):
    """Return (mask_bool, alpha_float32, is_regional) keeping outside pixels strictly zero."""
    arr = np.asarray(regional_mask)
    if arr.shape != shape:
        return None, None, False
    if arr.dtype == bool:
        mask_bool = arr
        pre_feathered = False
    elif np.issubdtype(arr.dtype, np.integer):
        mask_bool = arr > 0
        pre_feathered = bool(np.any((arr > 1) & (arr < 254))) if arr.max() > 1 else False
    else:
        mask_bool = arr > 0.1
        pre_feathered = bool(np.any((arr > 0.01) & (arr < 0.99)))
    if not np.any(mask_bool):
        return None, None, False
    if bool(np.all(mask_bool)):
        return None, None, False
    if pre_feathered:
        scale = 255.0 if np.issubdtype(arr.dtype, np.integer) else 1.0
        alpha = np.clip(arr.astype(np.float32) / scale, 0.0, 1.0)
        alpha[~mask_bool] = 0.0
        return mask_bool, alpha, True
    dist = distance_transform_edt(mask_bool)
    max_dist = float(dist.max())
    feather = max(1.5, min(6.0, max_dist * 0.5))
    t = np.clip((dist - 0.5) / feather, 0.0, 1.0)
    alpha = (t * t * (3.0 - 2.0 * t)).astype(np.float32)
    alpha[~mask_bool] = 0.0
    return mask_bool, alpha, True


def prepare_fbcnn_deblocking(rgb, origin=(0, 0), regional_mask=None, dc=None, require_seams=True):
    """Deblock any 96..384 uint8 RGB crop (PNG, EXIF-rotated JPEG, Video frame) via FBCNN."""
    if not isinstance(rgb, np.ndarray) or rgb.ndim != 3 or rgb.shape[2] != 3 or rgb.dtype != np.uint8:
        return None, None
    height, width = rgb.shape[:2]
    if min(width, height) < 96 or max(width, height) > MAX_SIDE:
        return None, None
    ox0, oy0 = (int(origin[0]) % 8, int(origin[1]) % 8) if origin is not None else (0, 0)

    mask_bool = alpha = None
    is_regional = False
    if regional_mask is not None:
        mask_bool, alpha, is_regional = _feather_regional_mask(regional_mask, (height, width))
        if mask_bool is None and is_regional is False:
            arr = np.asarray(regional_mask)
            if arr.shape != (height, width) or not np.any(arr > (0.1 if np.issubdtype(arr.dtype, np.floating) else 0)):
                return None, None

    gray_before = np.asarray(Image.fromarray(rgb).convert('L'), dtype=np.float32)
    chosen_offsets = []
    before = []
    for axis, orig_off in ((0, oy0), (1, ox0)):
        orig_metrics = _axis_8px_metrics(gray_before, axis, orig_off, mask_bool)
        if orig_metrics is None:
            return None, None
        seam0, mid0, int0 = orig_metrics
        if seam0 >= max(0.75, 0.28 * int0) and mid0 <= 0.60 * seam0:
            chosen_offsets.append(orig_off)
            before.append(orig_metrics)
            continue
        candidates = []
        for off in range(8):
            metrics = orig_metrics if off == orig_off else _axis_8px_metrics(gray_before, axis, off, mask_bool)
            if metrics is not None:
                candidates.append((off, metrics))
        if not candidates:
            return None, None
        valid = [c for c in candidates if c[1][0] > 0 and c[1][1] <= 0.60 * c[1][0]]
        best_off, best_metrics = max(valid if valid else candidates, key=lambda item: item[1][0])
        chosen_offsets.append(best_off)
        before.append(best_metrics)

    oy, ox = chosen_offsets[0], chosen_offsets[1]
    if require_seams:
        if any(seam < max(0.75, 0.28 * interior) or midpoint > 0.60 * seam
               for seam, midpoint, interior in before):
            return None, None
    elif max(item[0] for item in before) <= 0.0:
        return None, None

    cleaned = run_deblock(rgb)
    if cleaned is None:
        return None, None

    if is_regional:
        w3 = alpha[..., None].astype(np.float32)
        blended = np.clip(
            np.rint(rgb.astype(np.float32) * (1.0 - w3) + cleaned.astype(np.float32) * w3),
            0, 255,
        ).astype(np.uint8)
        blended[~mask_bool] = rgb[~mask_bool]
        cleaned = blended

    if np.array_equal(cleaned, rgb):
        return None, None

    gray_after = np.asarray(Image.fromarray(cleaned).convert('L'), dtype=np.float32)
    after = [
        _axis_8px_metrics(gray_after, 0, oy, mask_bool),
        _axis_8px_metrics(gray_after, 1, ox, mask_bool),
    ]
    if any(item is None for item in after):
        return None, None

    if require_seams:
        if any(abs(curr[0]) > 0.80 * orig[0] for orig, curr in zip(before, after)):
            return None, None
    else:
        active_axes = [(orig, curr) for orig, curr in zip(before, after) if orig[0] > 0.25]
        if not active_axes or any(abs(curr[0]) > 0.80 * orig[0] for orig, curr in active_axes):
            return None, None

    retentions = []
    for orig, curr in zip(before, after):
        if orig[2] >= 0.50:
            ratio = curr[2] / orig[2]
            if ratio < 0.85:
                return None, None
            retentions.append(ratio)
    sum_before = before[0][2] + before[1][2]
    sum_after = after[0][2] + after[1][2]
    if sum_before >= 0.50 and (sum_after / sum_before) < 0.85:
        return None, None
    min_retention = min(retentions) if retentions else (sum_after / sum_before if sum_before >= 0.50 else 1.0)

    drops = [
        (orig[0] - abs(curr[0])) / orig[0] if orig[0] > 0 else 0.0
        for orig, curr in zip(before, after)
    ]
    report = {
        'policy': 'guarded-fbcnn-macroblock-v1',
        'native_size': [int(width), int(height)],
        'jpeg_luma_dc_quantization': int(dc) if dc is not None else None,
        'grid_origin': [int(ox), int(oy)],
        'input_origin': [int(ox0), int(oy0)],
        'regional': bool(is_regional),
        'seam_excess_before': [round(float(item[0]), 4) for item in before],
        'seam_excess_after': [round(float(item[0]), 4) for item in after],
        'seam_drop_fraction': [round(float(d), 4) for d in drops],
        'interior_gradient_before': [round(float(item[2]), 4) for item in before],
        'interior_gradient_after': [round(float(item[2]), 4) for item in after],
        'interior_gradient_retention': round(float(min_retention), 4),
        'preparation_model': 'FBCNN RGB',
        'preparation_model_sha256': CHECKPOINT_SHA256,
        'preparation_forward_returns': 1,
    }
    return cleaned, report


def main():
    import torch
    from third_party.fbcnn.network import FBCNN

    contents = sys.stdin.buffer.read(MAX_PNG_BYTES + 1)
    if len(contents) > MAX_PNG_BYTES or not contents.startswith(b'\x89PNG\r\n\x1a\n'):
        raise ValueError('JPEG preparation requires a bounded native PNG crop.')
    with Image.open(io.BytesIO(contents)) as image:
        if min(image.size) < 96 or max(image.size) > MAX_SIDE:
            raise ValueError('JPEG preparation crop dimensions exceed bounds.')
    crop = decode_original(contents)
    path = ordinary_file(checkpoint_path())
    with path.open('rb') as checkpoint:
        if (path.stat().st_size != CHECKPOINT_BYTES or
                hashlib.file_digest(checkpoint, 'sha256').hexdigest() != CHECKPOINT_SHA256):
            raise ValueError('JPEG preparation requires the pinned RGB checkpoint.')
        checkpoint.seek(0)
        state = torch.load(checkpoint, map_location='cpu', weights_only=True)
    torch.set_num_threads(4)
    model = FBCNN().eval()
    model.load_state_dict(state, strict=True)
    tensor = torch.from_numpy(crop).permute(2, 0, 1)[None].float() / 255
    with torch.no_grad():
        result, _ = model(tensor)
    if not torch.isfinite(result).all():
        raise ValueError('JPEG preparation produced non-finite values.')
    rgb = result[0].permute(1, 2, 0).clamp(0, 1).mul(255).round().byte().numpy()
    sys.stdout.buffer.write(png(rgb))


if __name__ == '__main__':
    main()
