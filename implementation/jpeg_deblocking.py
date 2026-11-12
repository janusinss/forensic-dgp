"""Guarded CPU preparation for strongly blocked JPEG uploads."""
from __future__ import annotations

import hashlib
import io
import logging
import subprocess
import sys

import numpy as np
from PIL import Image

from .managed_paths import data_path
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
    return data_path(ROOT, 'models', 'implementation') / CHECKPOINT


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
